"""A merged page cache that serves a stale page while one read refreshes it."""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Generic, TypeVar

ValueT = TypeVar("ValueT")

logger = logging.getLogger(__name__)

# The key ceiling. The bridge walks at most five pages, so a handful of keys is
# normal. A full-record page is several megabytes, so a page above the default
# limit is never cached (see api.merged_page) and the key count stays low: a
# client that invents cursors must not pin many pages in memory.
MAX_ENTRIES = 16


class PageCache(Generic[ValueT]):
    """Caches merged pages and refreshes them off the request path.

    Every page costs one health probe and one fan-out read per backend. The
    bridge polls the list every five seconds and Glass repeats it in bursts,
    so a page is served without a read for a short fresh window, then served
    again while one background read refreshes it. A burst therefore costs one
    read, and a slow fan-out or a short deployment outage never holds a caller.
    A key with no page yet, or a page older than the stale window, waits for
    the read because there is nothing trustworthy to serve. The key carries the
    page size, cursor, and role; a backend set that changed inside the window
    serves the last answer until a read replaces it. A read that raises keeps
    the last page; a read that succeeds replaces it, including an answer that
    reports every backend unavailable, because that answer is the deployment's
    outage report and not a stall.
    """

    def __init__(self, fresh_ttl: float, stale_ttl: float) -> None:
        if stale_ttl < fresh_ttl:
            raise ValueError("the stale window cannot be shorter than the fresh window")
        self.fresh_ttl = fresh_ttl
        self.stale_ttl = stale_ttl
        self.entries: dict[tuple, tuple[float, ValueT]] = {}
        self.refreshes: dict[tuple, asyncio.Task[ValueT]] = {}

    async def get_or_read(
        self,
        key: tuple,
        read: Callable[[], Awaitable[ValueT]],
    ) -> ValueT:
        """Return the page of one key without waiting for a refresh it can survive."""
        entry = self.entries.get(key)
        if entry is not None:
            age = time.monotonic() - entry[0]
            if age < self.fresh_ttl:
                return entry[1]
            if age < self.stale_ttl:
                self._refresh(key, read)
                return entry[1]
        return await asyncio.shield(self._refresh(key, read))

    def _refresh(
        self,
        key: tuple,
        read: Callable[[], Awaitable[ValueT]],
    ) -> asyncio.Task[ValueT]:
        """Start or join the one read of a key.

        The task is detached from the caller that starts it: a Glass call that
        disconnects must not cancel the read that the next call joins.
        """
        task = self.refreshes.get(key)
        if task is not None and not task.done():
            return task
        task = asyncio.create_task(self._read_and_store(key, read))
        self.refreshes[key] = task
        task.add_done_callback(lambda finished: self._refresh_finished(key, finished))
        return task

    async def _read_and_store(self, key: tuple, read: Callable[[], Awaitable[ValueT]]) -> ValueT:
        value = await read()
        self._store(key, value)
        return value

    def _refresh_finished(self, key: tuple, task: asyncio.Task[ValueT]) -> None:
        if self.refreshes.get(key) is task:
            self.refreshes.pop(key, None)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.warning("the merged page read failed; keeping the last page: %s", error)

    async def close(self) -> None:
        """Stop pending refreshes so shutdown does not destroy them mid-store."""
        tasks = list(self.refreshes.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _store(self, key: tuple, value: ValueT) -> None:
        now = time.monotonic()
        self.entries[key] = (now, value)
        stale_before = now - self.stale_ttl
        expired = [cached for cached, (stored_at, _) in self.entries.items() if stored_at < stale_before]
        for cached in expired:
            del self.entries[cached]
        while len(self.entries) > MAX_ENTRIES:
            oldest = min(self.entries, key=lambda cached: self.entries[cached][0])
            del self.entries[oldest]