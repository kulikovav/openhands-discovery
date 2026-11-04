"""A sqlite index of the conversations the automation run lookup has walked.

The deployment answers no conversation lookup by run id, so every chat page
read walked the whole search window of both backends. That walk now feeds this
index: every conversation it visits is stored with the run id its tag or
workspace names, and a read answers from sqlite unless the index is cold or
its fresh window has expired. One walk per window replaces the walk per read,
so a long-lived deployment stops paying the page bound on every chat read.

The index serves three states, one per read:

- cold (never walked): the read awaits one walk and projects its answer from
  the walked rows, which is the behavior of a deployment with no index.
- stale (walked, window expired): the read answers from the stored rows while
  one walk refreshes them in the background, so a poll burst costs no walk. A
  conversation the deployment created since the walk enters the answer after
  that walk, within the window. A dropped conversation leaves the index once a
  walk that reaches the search's last page passes its place; a walk that stops
  at the page bound never sees a row past its place, so a drop beyond the
  bound is answered until the catalog shrinks inside the bound.
- fresh: the read answers from the stored rows alone.

The ids of one run stay in the search page's order on a cold read and in
``updated_at`` order after it. The walked-at times live in memory, so a restart
walks once before serving from the index. An index that cannot open its file
stays disabled and every read walks.
"""

import asyncio
import logging
import os
import sqlite3
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    source TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (source, conversation_id)
);
CREATE INDEX IF NOT EXISTS conversations_run_id ON conversations (run_id);
CREATE TEMP TABLE IF NOT EXISTS walk_seen (conversation_id TEXT PRIMARY KEY);
"""

# The placeholder bound of one lookup query, safely under every
# SQLITE_MAX_VARIABLE_NUMBER default together with the source parameter.
# The wanted ids of one read are chunked into queries of this size per list.
_LOOKUP_CHUNK = 450

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexedConversation:
    """One conversation the walk visited, with the run its tag or workspace names."""

    conversation_id: str
    run_id: str = ""
    updated_at: str = ""


@dataclass(frozen=True)
class RunConversationWalk:
    """The answer of one upstream search walk behind a run lookup.

    ``rows`` carries every visited conversation with the run id its tag or
    workspace names, not only the rows of the runs a read wanted, because each
    caller projects its own answer from them. ``complete`` reports a walk that
    ended without an upstream error or a malformed page, so its walk may mark
    the index freshly walked. ``reached_end`` adds the search's own last page,
    which makes the walked rows the whole catalog of the source within the
    bound and lets the index drop the rows the walk did not see.
    """

    rows: list[IndexedConversation] = field(default_factory=list)
    complete: bool = False
    reached_end: bool = False


def add_resolution(resolved: dict[str, list[str]], run_id: str, conversation_id: str) -> None:
    """Add one run conversation id in first-seen order without repeats."""
    found = resolved.setdefault(run_id, [])
    if conversation_id not in found:
        found.append(conversation_id)


class ConversationIndex:
    """Answers the run conversation lookup from sqlite and refreshes by walk.

    The index owns one sqlite store and one walked-at time per backend source.
    A walk is the caller's closure, so this module reads no upstream: it picks
    the state and the walk serves it. One walk runs per source at a time and is
    detached from the caller that starts it, the way the page cache detaches a
    refresh, so a disconnected call cannot cancel the walk the next call joins.
    """

    def __init__(self, path: str | None, fresh_seconds: float) -> None:
        self.path = path
        self.fresh_seconds = fresh_seconds
        self.walked_at: dict[str, float] = {}
        self.walks: dict[str, asyncio.Task[RunConversationWalk]] = {}
        self._conn: sqlite3.Connection | None = None
        if path is None:
            return
        try:
            connection = sqlite3.connect(path)
            os.chmod(path, 0o600)
            connection.executescript(_SCHEMA)
        except (OSError, sqlite3.Error) as error:
            logger.warning("the conversation index stays disabled: %s", error)
            return
        self._conn = connection

    async def resolve(
        self,
        source: str,
        wanted: set[str],
        walk: Callable[[], Awaitable[RunConversationWalk]],
    ) -> dict[str, list[str]]:
        """Answer the lookup of one source.

        A fresh index answers from its rows without a walk, and a stale one
        answers the same way now and lets the walk refresh them in the
        background. A cold index, or a read whose sqlite read failed, awaits
        the one walk and projects its answer from the walked rows, in page
        order.
        """
        walked_at = self.walked_at.get(source)
        fresh = walked_at is not None and time.monotonic() - walked_at < self.fresh_seconds
        if fresh:
            served = self._lookup(source, wanted)
            if served is not None:
                return served
            # The store read failed while the window is fresh: answer from a
            # walk instead of an empty page for the rest of the window.
            fresh = False
            walked_at = None
        task = self.walks.get(source)
        if task is None or task.done():
            task = asyncio.create_task(self._walk_and_store(source, walk))
            self.walks[source] = task
            task.add_done_callback(lambda finished, key=source: self._walk_finished(key, finished))
        if walked_at is None:
            return self._answered(await asyncio.shield(task), wanted)
        return self._lookup(source, wanted) or {}

    def _answered(self, walked: RunConversationWalk, wanted: set[str]) -> dict[str, list[str]]:
        """Project one caller's answer from the walked rows, per caller.

        The row list is shared by the callers that joined the walk, so the
        answer is built per call instead of returning the walk's own lists.
        """
        answered: dict[str, list[str]] = {}
        for row in walked.rows:
            if row.run_id in wanted:
                add_resolution(answered, row.run_id, row.conversation_id)
        return answered

    async def _walk_and_store(
        self,
        source: str,
        walk: Callable[[], Awaitable[RunConversationWalk]],
    ) -> RunConversationWalk:
        walked = await walk()
        if not walked.complete:
            logger.warning("the conversation index walk of %s stopped early; the index stays cold", source)
        # Only a walk whose rows actually reached the store may silence the
        # next walk: a disabled or failed store answers nothing.
        if self._store(source, walked) and walked.complete:
            self.walked_at[source] = time.monotonic()
        return walked

    def _walk_finished(self, source: str, task: asyncio.Task[RunConversationWalk]) -> None:
        if self.walks.get(source) is task:
            self.walks.pop(source, None)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.warning("the conversation index walk failed: %s", error)

    def _store(self, source: str, walked: RunConversationWalk) -> bool:
        """Record every walked row; report whether the rows reached the store."""
        if self._conn is None:
            return False
        # ponytail: the sqlite calls run on the event loop thread. A walk
        # stores at most a few thousand rows, which stays in the low
        # milliseconds on the local disk. Upgrade path is asyncio.to_thread
        # around the store when a deployment grows past that.
        try:
            with self._conn:
                self._conn.execute("DELETE FROM walk_seen")
                self._conn.executemany(
                    "INSERT OR IGNORE INTO walk_seen (conversation_id) VALUES (?)",
                    [(row.conversation_id,) for row in walked.rows],
                )
                self._conn.executemany(
                    "REPLACE INTO conversations (source, conversation_id, run_id, updated_at)"
                    " VALUES (?, ?, ?, ?)",
                    [(source, row.conversation_id, row.run_id, row.updated_at) for row in walked.rows],
                )
                if walked.reached_end:
                    self._conn.execute(
                        "DELETE FROM conversations"
                        " WHERE source = ?"
                        " AND conversation_id NOT IN (SELECT conversation_id FROM walk_seen)",
                        (source,),
                    )
            # ponytail: a source whose catalog stays past the page bound never
            # reaches its last page, so dropped conversations beyond the bound
            # stay answerable for the file's life. Upgrade path is eviction by
            # updated_at when such a deployment reports stale chats.
        except sqlite3.Error as error:
            logger.warning("the conversation index write failed: %s", error)
            return False
        return True

    def _lookup(self, source: str, wanted: set[str]) -> dict[str, list[str]] | None:
        """The stored conversations of the wanted runs, newest first.

        A sqlite read failure answers ``None`` instead of an empty map, so the
        caller can tell a failed read from a source with no tagged rows.
        """
        if self._conn is None:
            return None
        resolved: dict[str, list[str]] = {}
        wanted_ids = sorted(wanted)
        try:
            for start in range(0, len(wanted_ids), _LOOKUP_CHUNK):
                chunk = wanted_ids[start : start + _LOOKUP_CHUNK]
                placeholders = ", ".join(["?"] * len(chunk))
                rows = self._conn.execute(
                    "SELECT run_id, conversation_id, updated_at FROM conversations"
                    f" WHERE source = ? AND run_id IN ({placeholders})"
                    " ORDER BY updated_at DESC, conversation_id ASC",
                    (source, *chunk),
                ).fetchall()
                for run_id, conversation_id, _ in rows:
                    add_resolution(resolved, run_id, conversation_id)
        except sqlite3.Error as error:
            logger.warning("the conversation index read failed: %s", error)
            return None
        return resolved

    async def close(self) -> None:
        """Stop pending walks and close the sqlite connection."""
        tasks = [task for task in self.walks.values() if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.walks.clear()
        if self._conn is not None:
            self._conn.close()
            self._conn = None