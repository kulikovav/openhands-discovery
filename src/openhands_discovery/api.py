"""HTTP entry layer: routes, authentication, and the fan-out orchestration."""

import httpx
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .auth import require_api_key
from .backends import compile_backends, unavailable_backends
from .config import DiscoveryConfig
from .conversations import CONVERSATION_PAGE_SIZE, decode_cursor, read_walk
from .models import (
    BackendRole,
    BackendsResponse,
    ConversationRow,
    ConversationsResponse,
    RunRow,
    RunsResponse,
    UnavailableAutomation,
)
from .runs import (
    AUTOMATION_SCAN_LIMIT,
    MERGED_RUN_LIMIT,
    apply_conversation_records,
    fetch_automation,
    fetch_automations,
    fetch_run_conversation_records,
    merge_runs,
    newest_automations,
    read_run_pages,
    run_conversation_ids,
    select_scan,
)

LIST_TIMEOUT_SECONDS = 10
HEALTH_PATH = "/health"
BACKENDS_PATH = "/api/discovery/backends"
CONVERSATIONS_PATH = "/api/discovery/conversations"
AUTOMATION_RUNS_PATH = "/api/discovery/automation/runs"


async def add_no_store_header(request: Request, call_next):
    """Keep the pre-FastAPI cache behavior: no store on every answer."""
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response


async def http_exception_handler(request: Request, exception: StarletteHTTPException) -> JSONResponse:
    """Keep the error body the pre-FastAPI service answered: {"error": ...}."""
    return JSONResponse(status_code=exception.status_code, content={"error": exception.detail})


async def validation_exception_handler(
    request: Request,
    exception: RequestValidationError,
) -> JSONResponse:
    """Answer a malformed query with the same error body as every other answer."""
    return JSONResponse(status_code=422, content={"error": "the request parameters are invalid"})


async def health() -> dict[str, str]:
    """Liveness. The health path answers without a key."""
    return {"status": "ok"}


async def list_backends(
    request: Request,
    api_key: str = Depends(require_api_key),
) -> BackendsResponse:
    """The compiled backend list."""
    config: DiscoveryConfig = request.app.state.config
    return BackendsResponse(backends=await compile_backends(config, api_key))


async def list_conversations(
    request: Request,
    limit: int = Query(default=CONVERSATION_PAGE_SIZE, ge=1, le=500),
    page_id: str | None = Query(default=None),
    role: BackendRole | None = Query(default=None),
    api_key: str = Depends(require_api_key),
) -> ConversationsResponse:
    """One merged conversation list from every healthy backend.

    Every agent server answers its own search page and the shared state store
    repeats rows across servers, so the merge deduplicates by id and keeps the
    first backend in the fixed order (ingress, slots in slot order, runs
    server) as the attribution. A backend that fails the read lands in
    ``unavailable``; the rows of the healthy backends still flow, so a caller
    sees partial data instead of a silent shrinkage. A backend whose health
    probe failed lands there too, because it contributes no rows: a caller
    that reads an empty page must be able to tell a deployment outage from a
    user who owns nothing. ``role`` narrows the fan-out, for example to the
    slots the bridge schedules on. ``limit`` is the page size per backend, so
    one answer carries up to one page per backend; the opaque ``next_page_id``
    cursor walks the remaining pages. Every row carries the title and tags its
    search page answered, so the bridge splits active and archived chats
    without another read.
    """
    config: DiscoveryConfig = request.app.state.config
    try:
        cursors = decode_cursor(page_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    backends = await compile_backends(config, api_key)
    candidates = [
        backend for backend in backends if backend.healthy and (role is None or backend.role == role)
    ]
    unavailable = unavailable_backends(backends, role)

    # ponytail: every request re-probes health and re-reads one page per
    # backend, so a poll costs one fan-out per backend. Upgrade path is a
    # short TTL cache keyed by the compiled backend set.
    async with httpx.AsyncClient(timeout=LIST_TIMEOUT_SECONDS) as client:
        walk = await read_walk(client, candidates, api_key, limit, cursors)

    return ConversationsResponse(
        conversations=[ConversationRow.model_validate(row) for row in walk.conversations],
        next_page_id=walk.next_page_id,
        unavailable=unavailable + walk.unavailable,
    )


async def list_automation_runs(
    request: Request,
    limit: int = Query(default=MERGED_RUN_LIMIT, ge=1, le=500),
    automation_id: str | None = Query(default=None, pattern=r"^[A-Za-z0-9_-]+$"),
    run_id: str | None = Query(default=None, pattern=r"^[A-Za-z0-9_-]+$"),
    status: str | None = Query(default=None),
    has_conversation: bool = Query(default=False),
    api_key: str = Depends(require_api_key),
) -> RunsResponse:
    """One merged automation run list from the ingress automation API.

    The deployment indexes runs per automation and has no cross-automation run
    index, so the merge reads run pages and combines the results newest first,
    capped at ``limit``. The scan covers the newest automations only; runs of
    older automations stay visible in the per-automation list of the automation
    API. ``automation_id`` narrows the scan to one automation, and ``status``
    filters the merged rows. ``has_conversation`` keeps only the runs that own
    a conversation and follows the offset pages of every automation until it
    collected ``limit`` matching rows or the page bound is reached: a run
    without a conversation has no chat to open, so the bridge's chat list must
    not lose an older chat behind the no-op runs of a schedule. ``run_id``
    resolves one run and follows the offset pages of every automation until it
    is found, because the deployment serves no single-run read and a run below
    the newest page of its automation is invisible to a shallow scan.

    Every merged row that owns a conversation also carries the conversation
    title and tags when the request selects chats (``has_conversation``) or one
    run (``run_id``). The service reads them in batches from the runs server,
    which owns the conversations automation runs create, then from the ingress
    for an id the runs server did not answer. A batch that fails leaves its
    rows with an empty title and a null tag map, so the bridge reads that state
    itself instead of treating a missing answer as an unarchived chat. A plain
    run table request skips the read, because it does not project a chat.
    """
    config: DiscoveryConfig = request.app.state.config
    base_url = f"http://127.0.0.1:{config.ingress_port}"
    scan = select_scan(limit, has_conversation, run_id)
    runs: list[dict] = []
    status_counts: dict[str, int] = {}
    unavailable: list[UnavailableAutomation] = []
    try:
        async with httpx.AsyncClient(timeout=LIST_TIMEOUT_SECONDS) as client:
            if automation_id:
                automations = [await fetch_automation(client, base_url, api_key, automation_id)]
            else:
                automations = newest_automations(
                    await fetch_automations(client, base_url, api_key),
                    AUTOMATION_SCAN_LIMIT,
                )
            read_result = await read_run_pages(client, base_url, api_key, automations, scan)
            runs, status_counts = merge_runs(read_result.pages, limit, status)
            unavailable = read_result.unavailable
            if runs and (has_conversation or run_id):
                records_port = config.runs_port or config.ingress_port
                records = await fetch_run_conversation_records(
                    client,
                    f"http://127.0.0.1:{records_port}",
                    base_url,
                    api_key,
                    run_conversation_ids(runs),
                )
                apply_conversation_records(runs, records)
    except httpx.HTTPStatusError as error:
        upstream_status = error.response.status_code
        raise HTTPException(
            status_code=upstream_status if 400 <= upstream_status < 500 else 502,
            detail=f"the automation API answered HTTP {upstream_status}",
        ) from error
    except httpx.HTTPError as error:
        raise HTTPException(status_code=502, detail=f"the automation API is unreachable: {error}") from error

    return RunsResponse(
        runs=[RunRow.model_validate(row) for row in runs],
        status_counts=status_counts,
        unavailable=unavailable,
    )


def create_app(config: DiscoveryConfig) -> FastAPI:
    """Build the service around one validated deployment configuration."""
    app = FastAPI(title="Agent Canvas discovery")
    app.state.config = config
    app.middleware("http")(add_no_store_header)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_api_route(HEALTH_PATH, health, methods=["GET"])
    app.add_api_route(BACKENDS_PATH, list_backends, methods=["GET"], response_model=BackendsResponse)
    app.add_api_route(
        CONVERSATIONS_PATH,
        list_conversations,
        methods=["GET"],
        response_model=ConversationsResponse,
    )
    app.add_api_route(
        AUTOMATION_RUNS_PATH,
        list_automation_runs,
        methods=["GET"],
        response_model=RunsResponse,
    )
    return app