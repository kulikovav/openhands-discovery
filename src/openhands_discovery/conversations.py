"""The conversation fan-out: one merged search walk across the backends."""

import asyncio
import base64
import binascii
import json
from dataclasses import dataclass, field
from typing import Any

import httpx

from .models import BackendEntry, BackendRef, UnavailableBackend, text_tags
from .upstream import fetch_json

CONVERSATION_SEARCH_PATH = "/api/conversations/search"
CONVERSATION_PAGE_SIZE = 100


@dataclass(frozen=True)
class ConversationPage:
    """One search page of one backend."""

    items: list[dict[str, Any]] = field(default_factory=list)
    next_page_id: str | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ConversationPage":
        items = payload.get("items")
        return cls(
            items=items if isinstance(items, list) else [],
            next_page_id=str(payload["next_page_id"]) if payload.get("next_page_id") else None,
        )


@dataclass(frozen=True)
class ConversationWalk:
    """The answer of one walk step across the backends."""

    conversations: list[dict[str, Any]]
    next_page_id: str | None
    unavailable: list[UnavailableBackend]


def merge_conversations(
    pages: list[tuple[BackendEntry, list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    """Merge search pages into one conversation list, newest first.

    The shared state store answers the same row from several servers, so the
    first backend in the fixed order keeps the row and carries the attribution.
    Rows without an id are dropped, because no consumer can address them. The
    title and tags are normalized, so a caller reads the same shape from every
    backend.
    """
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for backend, items in pages:
        for item in items:
            conversation_id = str(item.get("id", "") or "")
            if not conversation_id or conversation_id in seen:
                continue
            seen.add(conversation_id)
            merged.append(
                {
                    **item,
                    "id": conversation_id,
                    "title": str(item.get("title", "") or "").strip(),
                    "tags": text_tags(item.get("tags")),
                    "backend": BackendRef(name=backend.name, role=backend.role, url=backend.url),
                }
            )
    merged.sort(key=lambda row: str(row.get("updated_at", "") or ""), reverse=True)
    return merged


def walk_plan(backends: list[BackendEntry], cursors: dict[str, str]) -> list[BackendEntry]:
    """The backends to fetch on this walk step.

    The first step fetches every backend. A later step fetches only the
    backends whose cursor still carries a page id: a backend that exhausted
    its pages is absent from the cursor, so its first page cannot reappear as
    duplicate rows in a later page of the merged walk.
    """
    if not cursors:
        return backends
    return [backend for backend in backends if cursors.get(backend.name)]


def walk_cursors(
    incoming: dict[str, str],
    pages: list[tuple[BackendEntry, ConversationPage]],
) -> dict[str, str]:
    """The cursor for the next walk step.

    A fetched backend with a next page records it; a fetched backend without
    one finished and leaves the cursor. A backend that was not fetched
    (unhealthy, filtered out, or failed) carries its incoming cursor forward,
    so its remaining pages resume when it answers again.
    """
    cursors = dict(incoming)
    for backend, page in pages:
        if page.next_page_id:
            cursors[backend.name] = page.next_page_id
        else:
            cursors.pop(backend.name, None)
    return cursors


def encode_cursor(page_ids: dict[str, str]) -> str | None:
    """Encode the per-backend page ids into one opaque cursor."""
    if not page_ids:
        return None
    raw = json.dumps(page_ids, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def decode_cursor(cursor: str | None) -> dict[str, str]:
    """Decode an opaque cursor into per-backend page ids.

    A value this service did not issue raises ValueError, so the route answers
    400 instead of silently restarting the walk.
    """
    if not cursor:
        return {}
    try:
        decoded = json.loads(base64.urlsafe_b64decode(cursor.encode("ascii")))
    except (ValueError, binascii.Error, UnicodeEncodeError) as error:
        raise ValueError("the page_id cursor is invalid") from error
    if not isinstance(decoded, dict):
        raise ValueError("the page_id cursor is invalid")
    return {str(name): str(page_id) for name, page_id in decoded.items()}


async def fetch_conversation_page(
    client: httpx.AsyncClient,
    backend: BackendEntry,
    api_key: str,
    limit: int,
    page_id: str,
) -> ConversationPage:
    """Read one search page from a backend's agent server."""
    params: dict[str, Any] = {"limit": limit}
    if page_id:
        params["page_id"] = page_id
    url = f"http://127.0.0.1:{backend.port}{CONVERSATION_SEARCH_PATH}"
    return ConversationPage.from_payload(await fetch_json(client, url, api_key, params))


async def read_walk(
    client: httpx.AsyncClient,
    backends: list[BackendEntry],
    api_key: str,
    limit: int,
    cursors: dict[str, str],
) -> ConversationWalk:
    """Read one walk step: one search page per planned backend, merged.

    A backend that fails the read lands in ``unavailable``; the rows of the
    healthy backends still flow, so a caller sees partial data instead of a
    silent shrinkage.
    """
    fetch_backends = walk_plan(backends, cursors)
    results = await asyncio.gather(
        *[
            fetch_conversation_page(client, backend, api_key, limit, cursors.get(backend.name, ""))
            for backend in fetch_backends
        ],
        return_exceptions=True,
    )

    pages: list[tuple[BackendEntry, ConversationPage]] = []
    unavailable: list[UnavailableBackend] = []
    for backend, result in zip(fetch_backends, results):
        if isinstance(result, Exception):
            unavailable.append(UnavailableBackend(name=backend.name, role=backend.role, error=str(result)))
            continue
        pages.append((backend, result))

    merged = merge_conversations([(backend, page.items) for backend, page in pages])
    return ConversationWalk(
        conversations=merged,
        next_page_id=encode_cursor(walk_cursors(cursors, pages)),
        unavailable=unavailable,
    )