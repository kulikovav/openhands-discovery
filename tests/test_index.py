import asyncio

from openhands_discovery.index import (
    _LOOKUP_CHUNK,
    ConversationIndex,
    IndexedConversation,
    RunConversationWalk,
)


def index_of(tmp_path, fresh_seconds):
    return ConversationIndex(str(tmp_path / "index.sqlite3"), fresh_seconds=fresh_seconds)


def row(conversation_id, run_id, updated_at):
    return IndexedConversation(
        conversation_id=conversation_id,
        run_id=run_id,
        updated_at=updated_at,
    )


def test_index_serves_a_fresh_lookup_without_a_walk(tmp_path):
    index = index_of(tmp_path, 60.0)
    walks = 0

    async def run():
        nonlocal walks

        async def walk():
            nonlocal walks
            walks += 1
            return RunConversationWalk(
                rows=[row("c1", "r1", "2026-10-01T00:00:00Z"), row("c2", "r1", "2026-10-02T00:00:00Z")],
                complete=True,
                reached_end=True,
            )

        cold = await index.resolve("runs", {"r1"}, walk)
        fresh = await index.resolve("runs", {"r1"}, walk)
        return cold, fresh

    cold, fresh = asyncio.run(run())

    assert cold == {"r1": ["c1", "c2"]}
    assert fresh == {"r1": ["c2", "c1"]}
    assert walks == 1


def test_index_serves_stale_rows_while_a_background_walk_refreshes_it(tmp_path):
    index = index_of(tmp_path, fresh_seconds=0)
    walks = 0
    older_rows = [row("c1", "r1", "2026-10-01T00:00:00Z")]
    newer_rows = [row("c1", "r1", "2026-10-01T00:00:00Z"), row("c2", "r1", "2026-10-02T00:00:00Z")]

    async def walk_with(rows):
        nonlocal walks
        walks += 1
        return RunConversationWalk(rows=rows, complete=True, reached_end=False)

    async def walk_older():
        return await walk_with(older_rows)

    async def walk_newer():
        return await walk_with(newer_rows)

    async def run():
        cold = await index.resolve("runs", {"r1"}, walk_older)
        stale = await index.resolve("runs", {"r1"}, walk_newer)
        await index.walks["runs"]
        refreshed = await index.resolve("runs", {"r1"}, walk_newer)
        await asyncio.gather(*index.walks.values(), return_exceptions=True)
        return cold, stale, refreshed, walks

    cold, stale, refreshed, walks_done = asyncio.run(run())

    assert cold == {"r1": ["c1"]}
    assert stale == {"r1": ["c1"]}
    assert refreshed == {"r1": ["c2", "c1"]}
    assert walks_done == 3


def test_index_drops_the_rows_a_reached_end_walk_did_not_see(tmp_path):
    index = index_of(tmp_path, fresh_seconds=0)

    async def walk_gone():
        return RunConversationWalk(
            rows=[row("c1", "r1", "2026-10-01T00:00:00Z")],
            complete=True,
            reached_end=True,
        )

    async def walk_with_r2():
        return RunConversationWalk(
            rows=[row("c3", "r2", "2026-10-03T00:00:00Z")],
            complete=True,
            reached_end=True,
        )

    async def run():
        cold = await index.resolve("runs", {"r1"}, walk_gone)
        stale = await index.resolve("runs", {"r1", "r2"}, walk_with_r2)
        await index.walks["runs"]
        dropped = await index.resolve("runs", {"r1", "r2"}, walk_with_r2)
        return cold, stale, dropped

    cold, stale, dropped = asyncio.run(run())

    assert cold == {"r1": ["c1"]}
    assert stale == {"r1": ["c1"]}
    assert dropped == {"r2": ["c3"]}


def test_index_keeps_the_previous_answer_when_a_walk_fails(tmp_path):
    index = index_of(tmp_path, fresh_seconds=0)
    walks = 0

    async def run():
        nonlocal walks

        async def walk_ok():
            nonlocal walks
            walks += 1
            return RunConversationWalk(
                rows=[row("c1", "r1", "2026-10-01T00:00:00Z")],
                complete=True,
                reached_end=True,
            )

        async def walk_failing():
            nonlocal walks
            walks += 1
            raise RuntimeError("the deployment is unreachable")

        served = await index.resolve("runs", {"r1"}, walk_ok)
        stale = await index.resolve("runs", {"r1"}, walk_failing)
        await asyncio.gather(index.walks["runs"], return_exceptions=True)
        again = await index.resolve("runs", {"r1"}, walk_ok)
        await asyncio.gather(*index.walks.values(), return_exceptions=True)
        return served, stale, again, walks

    served, stale, again, walks_done = asyncio.run(run())

    assert served == {"r1": ["c1"]}
    assert stale == {"r1": ["c1"]}
    assert again == {"r1": ["c1"]}
    assert walks_done == 3


def test_index_shares_one_walk_between_concurrent_cold_calls(tmp_path):
    index = index_of(tmp_path, 60.0)
    walks = 0
    release = asyncio.Event()

    async def walk():
        nonlocal walks
        walks += 1
        await release.wait()
        return RunConversationWalk(
            rows=[row("c1", "r1", "2026-10-01T00:00:00Z")],
            complete=True,
            reached_end=True,
        )

    async def run():
        tasks = [asyncio.create_task(index.resolve("runs", {"r1"}, walk)) for _ in range(3)]
        await asyncio.sleep(0)
        release.set()
        return await asyncio.gather(*tasks), walks

    results, walks_done = asyncio.run(run())

    assert results == [{"r1": ["c1"]}] * 3
    assert walks_done == 1


def test_index_projects_each_callers_answer_from_one_walk(tmp_path):
    index = index_of(tmp_path, 60.0)
    walks = 0
    release = asyncio.Event()

    async def walk():
        nonlocal walks
        walks += 1
        await release.wait()
        return RunConversationWalk(
            rows=[row("c1", "r1", "2026-10-01T00:00:00Z"), row("c2", "r2", "2026-10-02T00:00:00Z")],
            complete=True,
            reached_end=True,
        )

    async def run():
        first = asyncio.create_task(index.resolve("runs", {"r1"}, walk))
        await asyncio.sleep(0)
        second = asyncio.create_task(index.resolve("runs", {"r2"}, walk))
        await asyncio.sleep(0)
        release.set()
        return await asyncio.gather(first, second), walks

    results, walks_done = asyncio.run(run())

    assert results == [{"r1": ["c1"]}, {"r2": ["c2"]}]
    assert walks_done == 1


def test_index_keeps_a_cold_walk_when_the_caller_is_cancelled(tmp_path):
    index = index_of(tmp_path, 60.0)
    walks = 0
    started = asyncio.Event()
    release = asyncio.Event()

    async def walk():
        nonlocal walks
        walks += 1
        started.set()
        await release.wait()
        return RunConversationWalk(
            rows=[row("c1", "r1", "2026-10-01T00:00:00Z")],
            complete=True,
            reached_end=True,
        )

    async def run():
        caller = asyncio.create_task(index.resolve("runs", {"r1"}, walk))
        await started.wait()
        caller.cancel()
        await asyncio.gather(caller, return_exceptions=True)
        release.set()
        return await index.resolve("runs", {"r1"}, walk), walks

    served, walks_done = asyncio.run(run())

    assert served == {"r1": ["c1"]}
    assert walks_done == 1


def test_index_close_cancels_a_pending_walk(tmp_path):
    index = index_of(tmp_path, fresh_seconds=0)
    cancelled = False
    started = asyncio.Event()

    async def walk_ok():
        return RunConversationWalk(
            rows=[row("c1", "r1", "2026-10-01T00:00:00Z")],
            complete=True,
            reached_end=True,
        )

    async def walk_hanging():
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            nonlocal cancelled
            cancelled = True
            raise
        return RunConversationWalk()

    async def run():
        await index.resolve("runs", {"r1"}, walk_ok)
        await index.resolve("runs", {"r1"}, walk_hanging)
        await started.wait()
        await index.close()
        return cancelled

    assert asyncio.run(run()) is True


def test_index_walks_again_after_an_incomplete_walk(tmp_path):
    index = index_of(tmp_path, 60.0)
    walks = 0

    async def run():
        nonlocal walks

        async def walk_ok():
            nonlocal walks
            walks += 1
            return RunConversationWalk(
                rows=[row("c1", "r1", "2026-10-01T00:00:00Z")],
                complete=True,
                reached_end=False,
            )

        async def walk_incomplete():
            nonlocal walks
            walks += 1
            return RunConversationWalk(
                rows=[row("c2", "r2", "2026-10-02T00:00:00Z")],
                complete=False,
                reached_end=False,
            )

        cold = await index.resolve("runs", {"r1"}, walk_ok)
        index.walked_at["runs"] -= 61.0
        stale = await index.resolve("runs", {"r1"}, walk_incomplete)
        await index.walks["runs"]
        after = await index.resolve("runs", {"r1"}, walk_ok)
        await asyncio.gather(*index.walks.values(), return_exceptions=True)
        return cold, stale, after, walks

    cold, stale, after, walks_done = asyncio.run(run())

    assert cold == {"r1": ["c1"]}
    assert stale == {"r1": ["c1"]}
    assert after == {"r1": ["c1"]}
    assert walks_done == 3


def test_index_falls_back_to_a_walk_when_a_fresh_lookup_fails(tmp_path):
    index = index_of(tmp_path, 60.0)
    walks = 0

    async def run():
        nonlocal walks

        async def walk():
            nonlocal walks
            walks += 1
            return RunConversationWalk(
                rows=[row("c1", "r1", "2026-10-01T00:00:00Z")],
                complete=True,
                reached_end=False,
            )

        cold = await index.resolve("runs", {"r1"}, walk)
        # A store closed behind the index's back is the sqlite read failure.
        index._conn.close()
        fresh = await index.resolve("runs", {"r1"}, walk)
        return cold, fresh, walks

    cold, fresh, walks_done = asyncio.run(run())

    assert cold == {"r1": ["c1"]}
    assert fresh == {"r1": ["c1"]}
    assert walks_done == 2


def test_index_serves_a_lookup_across_the_query_chunks(tmp_path):
    index = index_of(tmp_path, 60.0)

    async def walk():
        return RunConversationWalk(
            rows=[row(f"c{number}", f"r{number}", "2026-10-01T00:00:00Z") for number in range(_LOOKUP_CHUNK + 1)],
            complete=True,
            reached_end=True,
        )

    async def run():
        cold = await index.resolve("runs", {f"r{number}" for number in range(_LOOKUP_CHUNK + 1)}, walk)
        fresh = await index.resolve(
            "runs",
            {f"r{number}" for number in range(_LOOKUP_CHUNK + 1)},
            walk,
        )
        return cold, fresh

    cold, fresh = asyncio.run(run())

    assert set(cold) == {f"r{number}" for number in range(_LOOKUP_CHUNK + 1)}
    assert set(fresh) == set(cold)


def test_a_disabled_index_walks_every_time(tmp_path):
    unopenable = ConversationIndex(str(tmp_path / "missing-dir" / "index.sqlite3"), fresh_seconds=60.0)
    for index in (ConversationIndex(None, fresh_seconds=60.0), unopenable):
        walks = 0

        async def walk():
            nonlocal walks
            walks += 1
            return RunConversationWalk(
                rows=[row("c1", "r1", "2026-10-01T00:00:00Z")],
                complete=True,
                reached_end=True,
            )

        async def run():
            first = await index.resolve("runs", {"r1"}, walk)
            second = await index.resolve("runs", {"r1"}, walk)
            return first, second, walks, index.walked_at

        first, second, walks_done, walked_at = asyncio.run(run())
        assert (first, second) == ({"r1": ["c1"]}, {"r1": ["c1"]})
        assert walks_done == 2
        assert walked_at == {}


def test_a_restart_serves_the_stored_rows_after_one_walk(tmp_path):
    path = tmp_path / "index.sqlite3"
    first = ConversationIndex(str(path), fresh_seconds=60.0)

    async def walk_with_c1():
        return RunConversationWalk(
            rows=[row("c1", "r1", "2026-10-01T00:00:00Z")],
            complete=True,
            reached_end=False,
        )

    async def run():
        await first.resolve("runs", {"r1"}, walk_with_c1)
        await first.close()
        second = ConversationIndex(str(path), fresh_seconds=60.0)
        walks_after_restart = 0

        async def walk_with_c2():
            nonlocal walks_after_restart
            walks_after_restart += 1
            return RunConversationWalk(
                rows=[row("c2", "r1", "2026-10-02T00:00:00Z")],
                complete=True,
                reached_end=False,
            )

        cold = await second.resolve("runs", {"r1"}, walk_with_c2)
        fresh = await second.resolve("runs", {"r1"}, walk_with_c2)
        return cold, fresh, walks_after_restart

    cold, fresh, walks_after_restart = asyncio.run(run())

    assert cold == {"r1": ["c2"]}
    assert fresh == {"r1": ["c2", "c1"]}
    assert walks_after_restart == 1