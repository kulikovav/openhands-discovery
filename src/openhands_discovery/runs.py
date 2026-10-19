"""The automation run reads: scan strategies, merge, and chat projection.

The deployment indexes runs per automation and serves no cross-automation run
index, so every route read scans automations. The scan strategy is a domain
type, not a pair of request flags: a plain table read takes the newest page of
each automation, a chat list follows pages until it collected enough runs that
own a conversation, and a run lookup follows pages until it finds one run.
``run_id`` wins when a request sets both selectors.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from .conversations import CONVERSATION_PAGE_SIZE
from .models import UnavailableAutomation, text_tags
from .upstream import API_KEY_HEADER, fetch_json

AUTOMATIONS_PATH = "/api/automation/v1"
CONVERSATION_RECORDS_PATH = "/api/conversations"
CONVERSATION_SEARCH_PATH = "/api/conversations/search"
# AUTOMATION_RUN_TAG is the conversation tag the deployment writes for an
# automation run. It links a run to its conversation when the run record holds
# no conversation id.
AUTOMATION_RUN_TAG = "automationrunid"
MERGED_RUN_LIMIT = 100
RUN_PAGE_SIZE = 100
CONVERSATION_RECORD_BATCH_SIZE = 40
# The conversation lookup follows this many search pages per backend at most.
# It serves a run whose record lost the conversation id, so the bound keeps a
# long-lived deployment from turning one chat list read into a full scan.
RUN_CONVERSATION_PAGE_LIMIT = 20
# A failed or cancelled run can own a conversation the deployment did not link
# back to the run. A completed run is not a candidate: a scheduled check that
# completed without work never created a conversation.
LOST_CONVERSATION_STATUSES = frozenset({"FAILED", "CANCELLED"})
# A chat scan and a run lookup follow this many run pages per automation at
# most, so one request stays bounded. The runs that own a conversation are rare
# among the checks of a scheduled automation, and a run below its newest page
# is invisible to a shallow scan.
#
# ponytail: a chat-poor or lookup-miss automation is followed page by page on
# every request, so a poll costs up to this many pages per automation. Upgrade
# path is a TTL cache or a deployment-side run index.
RUN_SCAN_PAGE_LIMIT = 20
AUTOMATION_SCAN_LIMIT = 25
AUTOMATION_LIST_LIMIT = 100


@dataclass(frozen=True)
class AutomationRunsPage:
    """One upstream run page, normalized."""

    runs: list[dict[str, Any]] = field(default_factory=list)
    status_counts: dict[str, Any] = field(default_factory=dict)
    total: int | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "AutomationRunsPage":
        runs = payload.get("runs")
        status_counts = payload.get("status_counts")
        total = payload.get("total")
        return cls(
            runs=runs if isinstance(runs, list) else [],
            status_counts=status_counts if isinstance(status_counts, dict) else {},
            total=total if isinstance(total, int) else None,
        )


class RunScan(Protocol):
    """How one route read turns the automations into run pages."""

    async def read_page(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        api_key: str,
        automation_id: str,
    ) -> AutomationRunsPage:
        """Read this scan's run page for one automation."""


@dataclass(frozen=True)
class NewestRuns:
    """The newest run page of every automation, for a plain run table."""

    page_size: int

    async def read_page(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        api_key: str,
        automation_id: str,
    ) -> AutomationRunsPage:
        return await fetch_runs(client, base_url, api_key, automation_id, self.page_size)


@dataclass(frozen=True)
class ChatRuns:
    """The newest runs of every automation that own a conversation.

    A scheduled automation checks for work far more often than it starts a run,
    so its newest run page can hold none of its conversations. The scan follows
    the offset pages until it collected ``limit`` matching rows, the automation
    ended, or the page bound is reached. The status counters come from the first
    page, like the unfiltered read.

    ``status`` narrows which confirmed chat stops the scan, so a filtered chat
    page does not under-fill when matching chats sit below the newest page.
    """

    limit: int
    status: str | None = None

    async def read_page(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        api_key: str,
        automation_id: str,
    ) -> AutomationRunsPage:
        collected: list[dict[str, Any]] = []
        status_counts: dict[str, Any] = {}
        offset = 0
        for page_index in range(RUN_SCAN_PAGE_LIMIT):
            page = await fetch_runs(client, base_url, api_key, automation_id, RUN_PAGE_SIZE, offset)
            if page_index == 0:
                status_counts = page.status_counts
            if not page.runs:
                break
            collected.extend(chat_run_candidates(page.runs))
            offset += len(page.runs)
            # Only a run that already carries its conversation id is a known
            # chat. A candidate stays in the page for the id lookup, but it
            # must not stop the scan before the requested number of chats.
            if sum(1 for run in collected if chat_is_confirmed(run, self.status)) >= self.limit:
                break
            if page.total is not None and offset >= page.total:
                break
        return AutomationRunsPage(runs=collected, status_counts=status_counts)


@dataclass(frozen=True)
class SingleRun:
    """One run, found by following the offset pages of every automation.

    The deployment serves no single-run read, so a run below the newest page of
    its automation is invisible to a shallow scan.
    """

    run_id: str

    async def read_page(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        api_key: str,
        automation_id: str,
    ) -> AutomationRunsPage:
        offset = 0
        for _ in range(RUN_SCAN_PAGE_LIMIT):
            page = await fetch_runs(client, base_url, api_key, automation_id, RUN_PAGE_SIZE, offset)
            if not page.runs:
                break
            found = run_by_id(page.runs, self.run_id)
            if found:
                return AutomationRunsPage(runs=found)
            offset += len(page.runs)
            if page.total is not None and offset >= page.total:
                break
        return AutomationRunsPage(runs=[])


def select_scan(
    limit: int, has_conversation: bool, run_id: str | None, status: str | None = None
) -> RunScan:
    """Choose the scan strategy of one request.

    The route accepts both selectors, and ``run_id`` wins: a caller that asks
    for one run wants that row, not the chat window.
    """
    if run_id:
        return SingleRun(run_id)
    if has_conversation:
        return ChatRuns(limit, status)
    return NewestRuns(min(limit, RUN_PAGE_SIZE))


async def fetch_runs(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    automation_id: str,
    limit: int,
    offset: int = 0,
) -> AutomationRunsPage:
    """Read one run page of one automation."""
    payload = await fetch_json(
        client,
        f"{base_url}{AUTOMATIONS_PATH}/{automation_id}/runs",
        api_key,
        {"limit": limit, "offset": offset},
    )
    return AutomationRunsPage.from_payload(payload)


async def fetch_automations(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
) -> list[dict[str, Any]]:
    """Read the automation list page. One page covers the scan bound."""
    payload = await fetch_json(
        client,
        f"{base_url}{AUTOMATIONS_PATH}",
        api_key,
        {"limit": AUTOMATION_LIST_LIMIT, "offset": 0},
    )
    automations = payload.get("automations")
    return automations if isinstance(automations, list) else []


async def fetch_automation(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    automation_id: str,
) -> dict[str, Any]:
    """Read one automation record."""
    return await fetch_json(client, f"{base_url}{AUTOMATIONS_PATH}/{automation_id}", api_key)


@dataclass(frozen=True)
class RunReadResult:
    """The run pages of one fan-out and the automations that failed it."""

    pages: list[tuple[dict[str, Any], AutomationRunsPage]]
    unavailable: list[UnavailableAutomation]


async def read_run_pages(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    automations: list[dict[str, Any]],
    scan: RunScan,
) -> RunReadResult:
    """Read the scan's page for every automation in parallel.

    An automation whose read fails is reported as unavailable instead of
    failing the whole merge, so one broken automation cannot hide the runs of
    the others.
    """
    targets = [
        (automation, str(automation.get("id", "") or ""))
        for automation in automations
        if str(automation.get("id", "") or "")
    ]
    results = await asyncio.gather(
        *[scan.read_page(client, base_url, api_key, automation_id) for _, automation_id in targets],
        return_exceptions=True,
    )
    pages: list[tuple[dict[str, Any], AutomationRunsPage]] = []
    unavailable: list[UnavailableAutomation] = []
    for (automation, automation_id), result in zip(targets, results):
        if isinstance(result, Exception):
            unavailable.append(
                UnavailableAutomation(
                    id=automation_id,
                    name=str(automation.get("name", "") or ""),
                    error=str(result),
                )
            )
            continue
        pages.append((automation, result))
    return RunReadResult(pages=pages, unavailable=unavailable)


def merge_runs(
    pages: list[tuple[dict[str, Any], AutomationRunsPage]],
    limit: int,
    status: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Merge per-automation run pages into one newest-first list.

    ``status`` filters the rows before the ``limit`` cut, so a filtered request
    does not lose matching rows that fell below the newest-``limit`` cut. The
    status counters sum the deployment's own per-automation counters over the
    scanned window. Rows without an id are dropped, and a malformed counter is
    skipped instead of failing the whole merge.
    """
    runs: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    for automation, page in pages:
        for run in page.runs:
            run_id = str(run.get("id", "") or "")
            if not run_id:
                continue
            if status and str(run.get("status", "") or "") != status:
                continue
            runs.append(
                {
                    **run,
                    "id": run_id,
                    "automation_name": str(automation.get("name", "") or ""),
                }
            )
        for key, value in page.status_counts.items():
            try:
                status_counts[str(key)] = status_counts.get(str(key), 0) + int(value)
            except (TypeError, ValueError):
                continue
    runs.sort(key=lambda row: str(row.get("created_at", "") or ""), reverse=True)
    return runs[:limit], status_counts


def newest_automations(automations: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    """Newest automations first, capped.

    The deployment indexes runs per automation and has no cross-automation run
    index, so a bounded scan of the newest automations is the only merge
    source.
    """
    ordered = sorted(
        automations,
        key=lambda automation: str(automation.get("created_at", "") or ""),
        reverse=True,
    )
    return ordered[:cap]


def chat_run_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The runs of one page that can own a chat.

    A run with a conversation id owns one. A failed or cancelled run stays a
    candidate because the deployment can create its conversation and lose the
    id before it records the run result. A completed no-op run stays out: a
    scheduled check that completed without work created no conversation.
    """
    return [
        run
        for run in rows
        if str(run.get("conversation_id", "") or "")
        or str(run.get("status", "") or "").upper() in LOST_CONVERSATION_STATUSES
    ]


def chat_is_confirmed(run: dict[str, Any], status: str | None) -> bool:
    """Report whether a run is a known chat for the requested status filter."""
    if not str(run.get("conversation_id", "") or ""):
        return False
    return not status or str(run.get("status", "") or "") == status


def conversation_run_id(conversation: dict[str, Any]) -> str:
    """The run id a conversation records, from its tag or its run workspace.

    The deployment tags every automation conversation with ``automationrunid``.
    The run workspace is the documented fallback
    ``.../workspaces/automation-runs/<run_id>``, so a conversation that lost
    the tag still resolves.
    """
    tags = conversation.get("tags")
    if isinstance(tags, dict):
        run_id = str(tags.get(AUTOMATION_RUN_TAG, "") or "")
        if run_id:
            return run_id
    workspace = conversation.get("workspace")
    if isinstance(workspace, dict):
        working_dir = str(workspace.get("working_dir", "") or "").rstrip("/")
        marker = "/automation-runs/"
        if marker in working_dir:
            return working_dir.rsplit(marker, 1)[1].split("/", 1)[0]
    return ""


async def resolve_run_conversations(
    client: httpx.AsyncClient,
    runs_base_url: str,
    ingress_base_url: str,
    api_key: str,
    run_ids: list[str],
) -> dict[str, str]:
    """Resolve the conversation of each run whose record lost the id.

    The deployment answers no lookup by run id, so the read walks the
    conversation search newest first and matches the ``automationrunid`` tag or
    the run workspace. The runs server owns the conversations automation runs
    create; the ingress covers an older conversation that server does not
    hold. The walk is bounded per backend, so a run with no conversation costs
    at most that bound and is then dropped by the caller.
    """
    wanted = {run_id for run_id in run_ids if run_id}
    if not wanted:
        return {}
    resolved = await find_run_conversations(client, runs_base_url, api_key, wanted)
    if len(resolved) < len(wanted) and runs_base_url != ingress_base_url:
        missing = wanted - set(resolved)
        resolved.update(await find_run_conversations(client, ingress_base_url, api_key, missing))
    return resolved


async def find_run_conversations(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    run_ids: set[str],
) -> dict[str, str]:
    """Find the conversations of the given runs on one backend."""
    if not run_ids:
        return {}
    resolved: dict[str, str] = {}
    page_id = ""
    for _ in range(RUN_CONVERSATION_PAGE_LIMIT):
        params: dict[str, Any] = {"limit": CONVERSATION_PAGE_SIZE}
        if page_id:
            params["page_id"] = page_id
        try:
            payload = await fetch_json(client, f"{base_url}{CONVERSATION_SEARCH_PATH}", api_key, params)
        except (httpx.HTTPError, ValueError):
            # The id lookup is a repair beside the run page, not a source of
            # it: a search that fails leaves the rows the run read already
            # proved, and the caller drops only the unresolved candidates.
            break
        items = payload.get("items")
        if not isinstance(items, list) or not items:
            break
        for item in items:
            if not isinstance(item, dict):
                continue
            run_id = conversation_run_id(item)
            if run_id not in run_ids or run_id in resolved:
                continue
            conversation_id = str(item.get("id", "") or "")
            if conversation_id:
                resolved[run_id] = conversation_id
        if len(resolved) == len(run_ids):
            break
        page_id = str(payload.get("next_page_id", "") or "")
        if not page_id:
            break
    return resolved


def run_by_id(rows: list[dict[str, Any]], run_id: str) -> list[dict[str, Any]]:
    """The rows of one page that carry the looked-up run id."""
    return [run for run in rows if str(run.get("id", "") or "") == run_id]


def run_conversation_ids(runs: list[dict[str, Any]]) -> list[str]:
    """The unique conversation ids of the merged run rows, in row order.

    A duplicate id is read once, because several runs can point at one
    conversation only when the deployment stored a repeated row.
    """
    ids: list[str] = []
    seen: set[str] = set()
    for run in runs:
        conversation_id = str(run.get("conversation_id", "") or "")
        if not conversation_id or conversation_id in seen:
            continue
        seen.add(conversation_id)
        ids.append(conversation_id)
    return ids


def apply_conversation_records(
    runs: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
) -> None:
    """Attach the conversation title and tags to every run row that has them.

    A record the read did not answer leaves both fields unset, and the route
    answers them empty: an empty title and a null tag map. The bridge reads
    that state itself instead of treating a missing answer as an unarchived
    chat. An answered record with no tags answers an empty tag map, which is a
    chat the deployment reported as active.
    """
    for run in runs:
        conversation_id = str(run.get("conversation_id", "") or "")
        record = records.get(conversation_id)
        if not isinstance(record, dict):
            continue
        run["conversation_tags"] = text_tags(record.get("tags"))
        title = str(record.get("title", "") or "").strip()
        if title:
            run["conversation_title"] = title


async def fetch_conversation_record_batch(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    conversation_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Read one batch of conversation records.

    The deployment answers one entry per requested id, in position order, with
    null for an id its catalog does not hold, and each entry repeats its own
    id; a reply is matched by that id, and by position when the entry carries
    none.
    """
    params = [("ids", conversation_id) for conversation_id in conversation_ids]
    response = await client.get(
        f"{base_url}{CONVERSATION_RECORDS_PATH}",
        params=params,
        headers={API_KEY_HEADER: api_key},
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        return {}
    records: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(payload):
        if not isinstance(record, dict):
            continue
        conversation_id = str(record.get("id", "") or "")
        if not conversation_id and index < len(conversation_ids):
            conversation_id = conversation_ids[index]
        if conversation_id:
            records[conversation_id] = record
    return records


async def fetch_conversation_records(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    conversation_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Read the records of the merged run rows, one batch per request.

    A failed batch is skipped, so its rows keep no conversation fields and the
    bridge falls back to its own record read. The other batches still attach,
    because one refused batch must not hide the state of the rest of the page.
    """
    batches = [
        conversation_ids[start : start + CONVERSATION_RECORD_BATCH_SIZE]
        for start in range(0, len(conversation_ids), CONVERSATION_RECORD_BATCH_SIZE)
    ]
    results = await asyncio.gather(
        *[fetch_conversation_record_batch(client, base_url, api_key, batch) for batch in batches],
        return_exceptions=True,
    )
    records: dict[str, dict[str, Any]] = {}
    for result in results:
        if isinstance(result, Exception):
            continue
        records.update(result)
    return records


async def fetch_run_conversation_records(
    client: httpx.AsyncClient,
    runs_base_url: str,
    ingress_base_url: str,
    api_key: str,
    conversation_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Read the run conversation records from the runs server, then the ingress.

    An older run can own a conversation the runs server does not hold, the same
    way the bridge reads it from the connected deployment after the runs
    deployment. The second read covers only the ids the first read did not
    answer, so a deployment whose runs server holds every record pays one read.
    """
    records = await fetch_conversation_records(client, runs_base_url, api_key, conversation_ids)
    if runs_base_url == ingress_base_url:
        return records
    missing = [conversation_id for conversation_id in conversation_ids if conversation_id not in records]
    if not missing:
        return records
    records.update(await fetch_conversation_records(client, ingress_base_url, api_key, missing))
    return records