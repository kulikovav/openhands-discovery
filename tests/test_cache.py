import asyncio

import pytest

from openhands_discovery.cache import MAX_ENTRIES, PageCache


def test_cache_shares_one_read_within_the_window():
    async def run():
        cache = PageCache(fresh_ttl=60, stale_ttl=300)
        reads = 0

        async def read():
            nonlocal reads
            reads += 1
            return {"rows": reads}

        first = await cache.get_or_read(("key",), read)
        second = await cache.get_or_read(("key",), read)
        return first, second, reads

    first, second, reads = asyncio.run(run())
    assert first == second == {"rows": 1}
    assert reads == 1


def test_cache_waits_for_a_page_older_than_the_stale_window():
    async def run():
        cache = PageCache(fresh_ttl=0, stale_ttl=0)
        reads = 0

        async def read():
            nonlocal reads
            reads += 1
            return reads

        first = await cache.get_or_read(("key",), read)
        second = await cache.get_or_read(("key",), read)
        return first, second

    assert asyncio.run(run()) == (1, 2)


def test_cache_serves_stale_page_while_a_background_read_refreshes_it():
    async def run():
        cache = PageCache(fresh_ttl=0, stale_ttl=60)
        reads = 0
        release = asyncio.Event()
        refreshed = asyncio.Event()

        async def read():
            nonlocal reads
            reads += 1
            if reads > 1:
                await release.wait()
                refreshed.set()
            return reads

        first = await cache.get_or_read(("key",), read)
        second = await cache.get_or_read(("key",), read)
        release.set()
        await refreshed.wait()
        refreshed_reads = reads
        third = await cache.get_or_read(("key",), read)
        return first, second, third, refreshed_reads

    first, second, third, refreshed_reads = asyncio.run(run())
    assert (first, second, third) == (1, 1, 2)
    assert refreshed_reads == 2


def test_cache_starts_one_refresh_for_a_stale_page():
    async def run():
        cache = PageCache(fresh_ttl=0, stale_ttl=60)
        reads = 0
        release = asyncio.Event()
        refreshed = asyncio.Event()

        async def read():
            nonlocal reads
            reads += 1
            if reads > 1:
                await release.wait()
                refreshed.set()
            return reads

        await cache.get_or_read(("key",), read)
        results = await asyncio.gather(*[cache.get_or_read(("key",), read) for _ in range(3)])
        release.set()
        await refreshed.wait()
        return results, reads

    results, reads = asyncio.run(run())
    assert results == [1, 1, 1]
    assert reads == 2


def test_cache_keeps_the_last_page_when_a_refresh_fails():
    async def run():
        cache = PageCache(fresh_ttl=0, stale_ttl=60)
        reads = 0
        failed = asyncio.Event()
        refreshed = asyncio.Event()

        async def read():
            nonlocal reads
            reads += 1
            if reads == 2:
                failed.set()
                raise RuntimeError("the deployment is unreachable")
            if reads == 3:
                refreshed.set()
            return reads

        first = await cache.get_or_read(("key",), read)
        second = await cache.get_or_read(("key",), read)
        await failed.wait()
        third = await cache.get_or_read(("key",), read)
        await refreshed.wait()
        fourth = await cache.get_or_read(("key",), read)
        return first, second, third, fourth

    first, second, third, fourth = asyncio.run(run())
    assert (first, second, third) == (1, 1, 1)
    assert fourth == 3


def test_cache_propagates_a_failed_cold_read():
    async def run():
        cache = PageCache(fresh_ttl=60, stale_ttl=300)

        async def read():
            raise RuntimeError("the deployment is unreachable")

        await cache.get_or_read(("key",), read)

    with pytest.raises(RuntimeError):
        asyncio.run(run())


def test_cache_shares_one_read_between_concurrent_callers():
    async def run():
        cache = PageCache(fresh_ttl=60, stale_ttl=300)
        reads = 0
        release = asyncio.Event()

        async def read():
            nonlocal reads
            reads += 1
            await release.wait()
            return "page"

        tasks = [asyncio.create_task(cache.get_or_read(("key",), read)) for _ in range(3)]
        await asyncio.sleep(0)
        release.set()
        return await asyncio.gather(*tasks), reads

    results, reads = asyncio.run(run())
    assert results == ["page", "page", "page"]
    assert reads == 1


def test_cache_evicts_the_oldest_page_at_the_ceiling():
    async def run():
        cache = PageCache(fresh_ttl=60, stale_ttl=300)
        reads = 0

        async def read():
            nonlocal reads
            reads += 1
            return reads

        for index in range(MAX_ENTRIES + 5):
            await cache.get_or_read((index,), read)
        value = await cache.get_or_read((0,), read)
        return value, reads, len(cache.entries)

    value, reads, entries = asyncio.run(run())
    assert value == MAX_ENTRIES + 6
    assert reads == MAX_ENTRIES + 6
    assert entries == MAX_ENTRIES


def test_cache_rejects_a_stale_window_shorter_than_the_fresh_window():
    with pytest.raises(ValueError):
        PageCache(fresh_ttl=60, stale_ttl=30)


def test_cache_keeps_a_cold_read_when_the_caller_is_cancelled():
    async def run():
        cache = PageCache(fresh_ttl=60, stale_ttl=300)
        reads = 0
        started = asyncio.Event()
        release = asyncio.Event()

        async def read():
            nonlocal reads
            reads += 1
            started.set()
            await release.wait()
            return "page"

        caller = asyncio.create_task(cache.get_or_read(("key",), read))
        await started.wait()
        caller.cancel()
        await asyncio.gather(caller, return_exceptions=True)
        release.set()
        second = await cache.get_or_read(("key",), read)
        return second, reads

    second, reads = asyncio.run(run())
    assert second == "page"
    assert reads == 1


def test_cache_close_cancels_a_pending_refresh():
    async def run():
        cache = PageCache(fresh_ttl=0, stale_ttl=60)
        reads = 0
        cancelled = False
        started = asyncio.Event()

        async def read():
            nonlocal reads, cancelled
            reads += 1
            if reads > 1:
                started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled = True
                    raise
            return reads

        await cache.get_or_read(("key",), read)
        await cache.get_or_read(("key",), read)
        await started.wait()
        await cache.close()
        return cancelled

    assert asyncio.run(run()) is True
