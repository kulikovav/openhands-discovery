"""Command line entry point."""

import asyncio
import os
import sys
import tempfile

import uvicorn

from .api import create_app
from .auth import keys_match
from .backends import unavailable_backends
from .config import DEFAULT_CONFIG_PATH, load_config
from .conversations import decode_cursor, encode_cursor, merge_conversations
from .index import ConversationIndex, IndexedConversation, RunConversationWalk
from .models import BackendEntry, BackendRole
from .runs import AutomationRunsPage, ChatRuns, SingleRun, merge_runs, select_scan

LISTEN_HOST = "127.0.0.1"


def self_check() -> None:
    """Assert the domain rules, including the index round trip. Run with --self-check.

    The deployment verification runs this check on the host. The pytest suite
    covers the same rules with more cases.
    """
    ingress = BackendEntry(
        name="canvas",
        role=BackendRole.INGRESS,
        url="https://oh.example",
        port=8000,
        active=True,
        healthy=True,
    )
    slot = BackendEntry(
        name="backend-1",
        role=BackendRole.SLOT,
        url="https://oh.example:8443",
        port=18100,
        active=True,
        healthy=False,
        error="the health endpoint is unreachable",
    )

    older = {"id": "c1", "updated_at": "2026-10-01T00:00:00Z", "title": "old"}
    newer = {"id": "c2", "updated_at": "2026-10-02T00:00:00Z", "title": "new"}
    merged = merge_conversations(
        [
            (ingress, [older, newer]),
            (slot, [dict(newer), {"id": "c3", "updated_at": "2026-09-30T00:00:00Z"}]),
        ]
    )
    assert [row["id"] for row in merged] == ["c2", "c1", "c3"], merged
    assert merged[0]["backend"].name == "canvas", merged

    cursor = encode_cursor({"backend-1": "page-2"})
    assert decode_cursor(cursor) == {"backend-1": "page-2"}
    try:
        decode_cursor("%%%")
        raise AssertionError("a bad cursor was accepted")
    except ValueError:
        pass

    pages = [
        (
            {"id": "a1", "name": "nightly"},
            AutomationRunsPage(runs=[{"id": "r1", "created_at": "2026-10-01T00:00:00Z"}]),
        ),
        (
            {"id": "a2", "name": "hourly"},
            AutomationRunsPage(
                runs=[{"id": "r2", "created_at": "2026-10-02T00:00:00Z", "status": "FAILED"}],
                status_counts={"FAILED": 1},
            ),
        ),
    ]
    runs, status_counts = merge_runs(pages, limit=1)
    assert [row["id"] for row in runs] == ["r2"], runs
    assert status_counts == {"FAILED": 1}, status_counts
    assert isinstance(select_scan(5, True, "r1"), SingleRun)
    assert isinstance(select_scan(5, True, None), ChatRuns)

    unavailable = unavailable_backends([ingress, slot], BackendRole.SLOT)
    assert [entry.name for entry in unavailable] == ["backend-1"], unavailable
    assert keys_match("secret", "secret")
    assert not keys_match("wrong", "secret")

    async def check_index() -> None:
        """Check the index rules: cold serve, stale serve, and the reached-end drop."""
        with tempfile.TemporaryDirectory() as scratch:
            index = ConversationIndex(os.path.join(scratch, "index.sqlite3"), fresh_seconds=0)
            first_walk = RunConversationWalk(
                rows=[
                    IndexedConversation(conversation_id="c1", run_id="r1", updated_at="2026-10-01T00:00:00Z"),
                ],
                complete=True,
                reached_end=True,
            )
            last_walk = RunConversationWalk(
                rows=[IndexedConversation(conversation_id="c3", run_id="r2", updated_at="2026-10-03T00:00:00Z")],
                complete=True,
                reached_end=True,
            )

            async def walk_first() -> RunConversationWalk:
                return first_walk

            async def walk_last() -> RunConversationWalk:
                return last_walk

            cold = await index.resolve("runs", {"r1"}, walk_first)
            assert cold == {"r1": ["c1"]}, cold
            stale = await index.resolve("runs", {"r1", "r2"}, walk_last)
            assert stale == {"r1": ["c1"]}, stale
            await index.walks["runs"]
            dropped = await index.resolve("runs", {"r1", "r2"}, walk_last)
            assert dropped == {"r2": ["c3"]}, dropped
            await index.close()

    asyncio.run(check_index())
    print("self-check passed")


def main() -> None:
    """Run the service on the loopback address, or the domain self-check."""
    if "--self-check" in sys.argv[1:]:
        self_check()
        return
    # The config path is a fixed argument from the unit definition, never a
    # request value, so it is not attacker-controlled input.
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH
    config = load_config(config_path)
    uvicorn.run(create_app(config), host=LISTEN_HOST, port=config.listen_port)


if __name__ == "__main__":
    main()