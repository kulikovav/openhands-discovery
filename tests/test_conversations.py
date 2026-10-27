import asyncio
import base64

import httpx

from openhands_discovery.conversations import (
    ConversationPage,
    decode_cursor,
    encode_cursor,
    merge_conversations,
    read_walk,
    walk_cursors,
    walk_plan,
)
from openhands_discovery.models import BackendEntry, BackendRole, text_tags


def make_backend(name, role, port, healthy=True, error=""):
    return BackendEntry(
        name=name,
        role=role,
        url=f"https://oh.example:{port}",
        port=port,
        active=True,
        healthy=healthy,
        error=error,
    )


def test_merge_conversations_deduplicates_and_keeps_the_first_backend():
    ingress = make_backend("canvas", BackendRole.INGRESS, 8000)
    slot = make_backend("backend-1", BackendRole.SLOT, 18100)
    older = {"id": "c1", "updated_at": "2026-10-01T00:00:00Z", "title": "old"}
    newer = {"id": "c2", "updated_at": "2026-10-02T00:00:00Z", "title": "new"}

    merged = merge_conversations(
        [
            (ingress, [older, newer]),
            (slot, [dict(newer), {"id": "c3", "updated_at": "2026-09-30T00:00:00Z"}]),
        ]
    )

    assert [row["id"] for row in merged] == ["c2", "c1", "c3"]
    assert merged[0]["backend"].name == "canvas"
    assert merged[0]["title"] == "new"
    assert merged[0]["tags"] == {}


def test_merge_conversations_drops_rows_without_an_id():
    slot = make_backend("backend-1", BackendRole.SLOT, 18100)
    assert merge_conversations([(slot, [{"title": "no id"}])]) == []


def test_merge_conversations_normalizes_title_and_tags():
    ingress = make_backend("canvas", BackendRole.INGRESS, 8000)
    tagged = merge_conversations(
        [
            (
                ingress,
                [
                    {
                        "id": "c4",
                        "updated_at": "2026-10-03T00:00:00Z",
                        "title": " tagged ",
                        "tags": {"archived": "true"},
                    }
                ],
            )
        ]
    )
    untagged = merge_conversations(
        [(ingress, [{"id": "c5", "updated_at": "2026-10-03T00:00:00Z", "tags": "bad"}])]
    )

    assert tagged[0]["title"] == "tagged"
    assert tagged[0]["tags"] == {"archived": "true"}
    assert untagged[0]["tags"] == {}


def test_text_tags_keeps_only_string_values():
    assert text_tags({"archived": "true", "count": 3}) == {"archived": "true", "count": ""}
    assert text_tags(None) == {}


def test_cursor_round_trip():
    cursor = encode_cursor({"backend-1": "page-2"})
    assert decode_cursor(cursor) == {"backend-1": "page-2"}
    assert decode_cursor(None) == {}
    assert encode_cursor({}) is None


def test_cursor_rejects_a_value_this_service_did_not_issue():
    for bad in ("%%%", base64.urlsafe_b64encode(b'"string"').decode("ascii")):
        try:
            decode_cursor(bad)
        except ValueError:
            continue
        raise AssertionError(f"a bad cursor was accepted: {bad}")


def test_walk_plan_skips_an_exhausted_backend():
    ingress = make_backend("canvas", BackendRole.INGRESS, 8000)
    slot = make_backend("backend-1", BackendRole.SLOT, 18100)
    step = walk_cursors(
        {},
        [
            (ingress, ConversationPage(items=[], next_page_id="p2")),
            (slot, ConversationPage(items=[], next_page_id=None)),
        ],
    )

    assert step == {"canvas": "p2"}
    assert walk_plan([ingress, slot], step) == [ingress]
    assert walk_plan([ingress, slot], {}) == [ingress, slot]


def test_walk_cursors_carries_a_missing_backend_forward():
    ingress = make_backend("canvas", BackendRole.INGRESS, 8000)
    slot = make_backend("backend-1", BackendRole.SLOT, 18100)
    resumed = walk_cursors(
        {"canvas": "p2", "backend-1": "s9"},
        [(ingress, ConversationPage(items=[], next_page_id=None))],
    )
    assert resumed == {"backend-1": "s9"}


def test_read_walk_merges_pages_and_reports_a_failed_read():
    ingress = make_backend("canvas", BackendRole.INGRESS, 8000)
    slot = make_backend("backend-1", BackendRole.SLOT, 18100)
    failing = make_backend("backend-2", BackendRole.SLOT, 18101)
    older = {"id": "c1", "updated_at": "2026-10-01T00:00:00Z", "title": "old"}
    newer = {"id": "c2", "updated_at": "2026-10-02T00:00:00Z", "title": "new"}
    requested_ports = []

    def handler(request):
        port = request.url.port
        requested_ports.append(port)
        if port == 18101:
            return httpx.Response(503, request=request)
        if port == 8000:
            if request.url.params.get("page_id") == "p2":
                return httpx.Response(
                    200,
                    json={"items": [{"id": "c4", "updated_at": "2026-09-15T00:00:00Z"}], "next_page_id": None},
                    request=request,
                )
            return httpx.Response(
                200,
                json={"items": [older, newer], "next_page_id": "p2"},
                request=request,
            )
        return httpx.Response(
            200,
            json={
                "items": [dict(newer), {"id": "c3", "updated_at": "2026-09-30T00:00:00Z"}],
                "next_page_id": None,
            },
            request=request,
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            first = await read_walk(client, [ingress, slot, failing], "key", 100, {})
            second = await read_walk(client, [ingress, slot], "key", 100, decode_cursor(first.next_page_id))
            return first, second

    first, second = asyncio.run(run())

    assert [row["id"] for row in first.conversations] == ["c2", "c1", "c3"]
    assert first.conversations[0]["backend"].name == "canvas"
    assert decode_cursor(first.next_page_id) == {"canvas": "p2"}
    assert [entry.name for entry in first.unavailable] == ["backend-2"]
    assert "503" in first.unavailable[0].error
    assert [row["id"] for row in second.conversations] == ["c4"]
    assert second.next_page_id is None
    assert sorted(requested_ports[:3]) == [8000, 18100, 18101]
    assert requested_ports[3:] == [8000]