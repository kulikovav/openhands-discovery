import asyncio

import httpx
import pytest

from openhands_discovery import api
from openhands_discovery.config import DiscoveryConfig
from openhands_discovery.conversations import ConversationWalk
from openhands_discovery.models import BackendEntry, BackendRole, UnavailableBackend
from openhands_discovery.runs import AutomationRunsPage, ChatRuns, RunReadResult, SingleRun


def make_config(tmp_path, key="secret", **overrides):
    env_file = tmp_path / "env"
    env_file.write_text(f"OH_SESSION_API_KEYS_0={key}\n", encoding="utf-8")
    values = {
        "domain": "oh.example",
        "env_file": str(env_file),
        "ingress_port": 8000,
        "listen_port": 18080,
        "quadlet_dir": str(tmp_path / "quadlet"),
        "slot_public_host_prefix": "b",
    }
    values.update(overrides)
    return DiscoveryConfig(**values)


def get(app, path, headers=None, params=None):
    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get(path, headers=headers, params=params)

    return asyncio.run(run())


def key_headers():
    return {"X-Session-API-Key": "secret"}


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


def test_health_needs_no_key(tmp_path):
    response = get(api.create_app(make_config(tmp_path)), "/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["cache-control"] == "no-store"


def test_backends_requires_the_deployment_key(tmp_path):
    response = get(api.create_app(make_config(tmp_path)), "/api/discovery/backends")

    assert response.status_code == 401
    assert response.json() == {"error": "a valid X-Session-API-Key header is required"}
    assert response.headers["cache-control"] == "no-store"


def test_backends_returns_the_compiled_list(tmp_path, monkeypatch):
    entry = make_backend("canvas", BackendRole.INGRESS, 8000)

    async def fake_compile(config, api_key):
        assert api_key == "secret"
        return [entry]

    monkeypatch.setattr(api, "compile_backends", fake_compile)
    response = get(api.create_app(make_config(tmp_path)), "/api/discovery/backends", headers=key_headers())

    assert response.status_code == 200
    assert response.json() == {
        "backends": [
            {
                "name": "canvas",
                "role": "ingress",
                "url": "https://oh.example:8000",
                "port": 8000,
                "active": True,
                "healthy": True,
                "version": "",
                "error": "",
            }
        ]
    }


def test_invalid_role_is_a_validation_error(tmp_path):
    response = get(
        api.create_app(make_config(tmp_path)),
        "/api/discovery/conversations",
        headers=key_headers(),
        params={"role": "unknown"},
    )

    assert response.status_code == 422
    assert response.json() == {"error": "the request parameters are invalid"}


def test_invalid_cursor_is_a_bad_request(tmp_path):
    response = get(
        api.create_app(make_config(tmp_path)),
        "/api/discovery/conversations",
        headers=key_headers(),
        params={"page_id": "%%%"},
    )

    assert response.status_code == 400
    assert response.json() == {"error": "the page_id cursor is invalid"}


def test_conversations_reports_health_failures_then_read_failures(tmp_path, monkeypatch):
    ingress = make_backend("canvas", BackendRole.INGRESS, 8000)
    slot = make_backend(
        "backend-1",
        BackendRole.SLOT,
        18100,
        healthy=False,
        error="the health endpoint is unreachable",
    )

    async def fake_compile(config, api_key):
        return [ingress, slot]

    async def fake_walk(client, backends, api_key, limit, cursors):
        assert [backend.name for backend in backends] == ["canvas"]
        return ConversationWalk(
            conversations=[],
            next_page_id=None,
            unavailable=[
                UnavailableBackend(
                    name="canvas",
                    role=BackendRole.INGRESS,
                    error="the search page answered HTTP 500",
                )
            ],
        )

    monkeypatch.setattr(api, "compile_backends", fake_compile)
    monkeypatch.setattr(api, "read_walk", fake_walk)
    response = get(api.create_app(make_config(tmp_path)), "/api/discovery/conversations", headers=key_headers())

    assert response.status_code == 200
    assert response.json()["unavailable"] == [
        {
            "name": "backend-1",
            "role": "slot",
            "error": "the health endpoint is unreachable",
        },
        {
            "name": "canvas",
            "role": "ingress",
            "error": "the search page answered HTTP 500",
        },
    ]


def test_runs_maps_an_upstream_client_error(tmp_path, monkeypatch):
    async def fake_fetch_automations(client, base_url, api_key):
        request = httpx.Request("GET", base_url)
        raise httpx.HTTPStatusError(
            "too many requests",
            request=request,
            response=httpx.Response(429, request=request),
        )

    monkeypatch.setattr(api, "fetch_automations", fake_fetch_automations)
    response = get(api.create_app(make_config(tmp_path)), "/api/discovery/automation/runs", headers=key_headers())

    assert response.status_code == 429
    assert response.json() == {"error": "the automation API answered HTTP 429"}


def test_runs_maps_a_connection_error(tmp_path, monkeypatch):
    async def fake_fetch_automations(client, base_url, api_key):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(api, "fetch_automations", fake_fetch_automations)
    response = get(api.create_app(make_config(tmp_path)), "/api/discovery/automation/runs", headers=key_headers())

    assert response.status_code == 502
    assert response.json() == {"error": "the automation API is unreachable: refused"}


def test_runs_merges_rows_and_attaches_chat_state(tmp_path, monkeypatch):
    automation = {"id": "a1", "name": "nightly", "created_at": "2026-10-01T00:00:00Z"}
    page = AutomationRunsPage(
        runs=[
            {
                "id": "r1",
                "created_at": "2026-10-02T00:00:00Z",
                "status": "RUNNING",
                "conversation_id": "c1",
            }
        ],
        status_counts={"RUNNING": 1},
    )

    async def fake_fetch_automations(client, base_url, api_key):
        return [automation]

    async def fake_read_run_pages(client, base_url, api_key, automations, scan):
        assert isinstance(scan, ChatRuns)
        return RunReadResult(pages=[(automation, page)], unavailable=[])

    async def fake_fetch_records(client, runs_base_url, ingress_base_url, api_key, conversation_ids):
        assert conversation_ids == ["c1"]
        assert runs_base_url == "http://127.0.0.1:8000"
        return {"c1": {"id": "c1", "title": "chat", "tags": {"archived": "true"}}}

    monkeypatch.setattr(api, "fetch_automations", fake_fetch_automations)
    monkeypatch.setattr(api, "read_run_pages", fake_read_run_pages)
    monkeypatch.setattr(api, "fetch_run_conversation_records", fake_fetch_records)
    response = get(
        api.create_app(make_config(tmp_path)),
        "/api/discovery/automation/runs",
        headers=key_headers(),
        params={"has_conversation": "true"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status_counts"] == {"RUNNING": 1}
    assert body["unavailable"] == []
    assert body["runs"] == [
        {
            "id": "r1",
            "created_at": "2026-10-02T00:00:00Z",
            "status": "RUNNING",
            "conversation_id": "c1",
            "automation_name": "nightly",
            "conversation_title": "chat",
            "conversation_tags": {"archived": "true"},
        }
    ]


def test_runs_resolves_a_failed_run_whose_record_lost_the_conversation_id(tmp_path, monkeypatch):
    automation = {"id": "a1", "name": "nightly"}
    page = AutomationRunsPage(
        runs=[
            {
                "id": "r1",
                "created_at": "2026-10-02T00:00:00Z",
                "status": "FAILED",
                "conversation_id": None,
            }
        ]
    )
    calls = {}

    async def fake_fetch_automations(client, base_url, api_key):
        return [automation]

    async def fake_read_run_pages(client, base_url, api_key, automations, scan):
        return RunReadResult(pages=[(automation, page)], unavailable=[])

    async def fake_resolve(client, runs_base_url, ingress_base_url, api_key, run_ids):
        calls["run_ids"] = run_ids
        calls["runs_base_url"] = runs_base_url
        return {"r1": "c1"}

    async def fake_fetch_records(client, runs_base_url, ingress_base_url, api_key, conversation_ids):
        calls["conversation_ids"] = conversation_ids
        return {"c1": {"id": "c1", "title": "failed chat", "tags": {}}}

    monkeypatch.setattr(api, "fetch_automations", fake_fetch_automations)
    monkeypatch.setattr(api, "read_run_pages", fake_read_run_pages)
    monkeypatch.setattr(api, "resolve_run_conversations", fake_resolve)
    monkeypatch.setattr(api, "fetch_run_conversation_records", fake_fetch_records)
    response = get(
        api.create_app(make_config(tmp_path)),
        "/api/discovery/automation/runs",
        headers=key_headers(),
        params={"has_conversation": "true"},
    )

    assert response.status_code == 200
    assert calls["run_ids"] == ["r1"]
    assert calls["runs_base_url"] == "http://127.0.0.1:8000"
    assert calls["conversation_ids"] == ["c1"]
    assert response.json()["runs"] == [
        {
            "id": "r1",
            "created_at": "2026-10-02T00:00:00Z",
            "status": "FAILED",
            "conversation_id": "c1",
            "automation_name": "nightly",
            "conversation_title": "failed chat",
            "conversation_tags": {},
        }
    ]


def test_runs_run_lookup_repairs_a_lost_conversation_id(tmp_path, monkeypatch):
    automation = {"id": "a1", "name": "nightly"}
    page = AutomationRunsPage(
        runs=[{"id": "r1", "created_at": "2026-10-02T00:00:00Z", "status": "FAILED"}]
    )
    calls = {}

    async def fake_fetch_automations(client, base_url, api_key):
        return [automation]

    async def fake_read_run_pages(client, base_url, api_key, automations, scan):
        assert isinstance(scan, SingleRun)
        return RunReadResult(pages=[(automation, page)], unavailable=[])

    async def fake_resolve(client, runs_base_url, ingress_base_url, api_key, run_ids):
        calls["run_ids"] = run_ids
        return {"r1": "c1"}

    async def fake_fetch_records(client, runs_base_url, ingress_base_url, api_key, conversation_ids):
        calls["conversation_ids"] = conversation_ids
        return {"c1": {"id": "c1", "title": "failed chat", "tags": {}}}

    monkeypatch.setattr(api, "fetch_automations", fake_fetch_automations)
    monkeypatch.setattr(api, "read_run_pages", fake_read_run_pages)
    monkeypatch.setattr(api, "resolve_run_conversations", fake_resolve)
    monkeypatch.setattr(api, "fetch_run_conversation_records", fake_fetch_records)
    response = get(
        api.create_app(make_config(tmp_path)),
        "/api/discovery/automation/runs",
        headers=key_headers(),
        params={"run_id": "r1"},
    )

    assert response.status_code == 200
    assert calls["run_ids"] == ["r1"]
    assert calls["conversation_ids"] == ["c1"]
    assert response.json()["runs"] == [
        {
            "id": "r1",
            "created_at": "2026-10-02T00:00:00Z",
            "status": "FAILED",
            "conversation_id": "c1",
            "automation_name": "nightly",
            "conversation_title": "failed chat",
            "conversation_tags": {},
        }
    ]


def test_runs_chat_page_drops_a_failed_run_with_no_resolvable_conversation(tmp_path, monkeypatch):
    automation = {"id": "a1", "name": "nightly"}
    page = AutomationRunsPage(
        runs=[
            {"id": "r1", "created_at": "2026-10-02T00:00:00Z", "status": "FAILED"},
            {"id": "r2", "created_at": "2026-10-01T00:00:00Z", "conversation_id": "c2"},
        ]
    )

    async def fake_fetch_automations(client, base_url, api_key):
        return [automation]

    async def fake_read_run_pages(client, base_url, api_key, automations, scan):
        return RunReadResult(pages=[(automation, page)], unavailable=[])

    async def fake_resolve(*args, **kwargs):
        return {}

    async def fake_fetch_records(*args, **kwargs):
        return {}

    monkeypatch.setattr(api, "fetch_automations", fake_fetch_automations)
    monkeypatch.setattr(api, "read_run_pages", fake_read_run_pages)
    monkeypatch.setattr(api, "resolve_run_conversations", fake_resolve)
    monkeypatch.setattr(api, "fetch_run_conversation_records", fake_fetch_records)
    response = get(
        api.create_app(make_config(tmp_path)),
        "/api/discovery/automation/runs",
        headers=key_headers(),
        params={"has_conversation": "true"},
    )

    assert response.status_code == 200
    assert [row["id"] for row in response.json()["runs"]] == ["r2"]


def test_runs_plain_table_skips_the_chat_read(tmp_path, monkeypatch):
    automation = {"id": "a1", "name": "nightly"}
    page = AutomationRunsPage(runs=[{"id": "r1", "created_at": "2026-10-02T00:00:00Z"}])

    async def fake_fetch_automations(client, base_url, api_key):
        return [automation]

    async def fake_read_run_pages(client, base_url, api_key, automations, scan):
        assert not isinstance(scan, ChatRuns)
        return RunReadResult(pages=[(automation, page)], unavailable=[])

    async def forbidden_record_read(*args, **kwargs):
        raise AssertionError("a plain run table must not read conversation records")

    monkeypatch.setattr(api, "fetch_automations", fake_fetch_automations)
    monkeypatch.setattr(api, "read_run_pages", fake_read_run_pages)
    monkeypatch.setattr(api, "fetch_run_conversation_records", forbidden_record_read)
    response = get(api.create_app(make_config(tmp_path)), "/api/discovery/automation/runs", headers=key_headers())

    assert response.status_code == 200
    assert response.json()["runs"][0]["conversation_title"] == ""
    assert response.json()["runs"][0]["conversation_tags"] is None


def test_key_rotation_applies_to_the_next_request(tmp_path, monkeypatch):
    async def fake_compile(config, api_key):
        return []

    monkeypatch.setattr(api, "compile_backends", fake_compile)
    app = api.create_app(make_config(tmp_path))

    assert get(app, "/api/discovery/backends", headers=key_headers()).status_code == 200
    (tmp_path / "env").write_text("OH_SESSION_API_KEYS_0=rotated\n", encoding="utf-8")
    assert get(app, "/api/discovery/backends", headers=key_headers()).status_code == 401
    rotated = get(app, "/api/discovery/backends", headers={"X-Session-API-Key": "rotated"})
    assert rotated.status_code == 200


@pytest.mark.parametrize(
    "path",
    ["/api/discovery/conversations", "/api/discovery/automation/runs"],
)
def test_list_routes_require_the_key(tmp_path, path):
    response = get(api.create_app(make_config(tmp_path)), path)

    assert response.status_code == 401
    assert response.json() == {"error": "a valid X-Session-API-Key header is required"}


@pytest.mark.parametrize(
    "path",
    ["/api/discovery/conversations", "/api/discovery/automation/runs"],
)
def test_list_routes_validate_the_limit(tmp_path, path):
    response = get(
        api.create_app(make_config(tmp_path)),
        path,
        headers=key_headers(),
        params={"limit": "0"},
    )

    assert response.status_code == 422
    assert response.json() == {"error": "the request parameters are invalid"}


def test_runs_narrows_to_one_automation_and_resolves_a_run(tmp_path, monkeypatch):
    automation = {"id": "a1", "name": "nightly"}
    page = AutomationRunsPage(
        runs=[{"id": "r1", "created_at": "2026-10-02T00:00:00Z", "conversation_id": "c1"}]
    )
    calls = {}

    async def fake_fetch_automation(client, base_url, api_key, automation_id):
        calls["automation_id"] = automation_id
        return automation

    async def forbidden_fetch_automations(*args, **kwargs):
        raise AssertionError("the narrowed read must not list automations")

    async def fake_read_run_pages(client, base_url, api_key, automations, scan):
        calls["scan"] = scan
        calls["automations"] = automations
        return RunReadResult(pages=[(automation, page)], unavailable=[])

    async def fake_fetch_records(client, runs_base_url, ingress_base_url, api_key, conversation_ids):
        calls["conversation_ids"] = conversation_ids
        calls["runs_base_url"] = runs_base_url
        return {}

    monkeypatch.setattr(api, "fetch_automation", fake_fetch_automation)
    monkeypatch.setattr(api, "fetch_automations", forbidden_fetch_automations)
    monkeypatch.setattr(api, "read_run_pages", fake_read_run_pages)
    monkeypatch.setattr(api, "fetch_run_conversation_records", fake_fetch_records)
    response = get(
        api.create_app(make_config(tmp_path)),
        "/api/discovery/automation/runs",
        headers=key_headers(),
        params={"automation_id": "a1", "run_id": "r1"},
    )

    assert response.status_code == 200
    assert calls["automation_id"] == "a1"
    assert calls["automations"] == [automation]
    assert isinstance(calls["scan"], SingleRun)
    assert calls["scan"].run_id == "r1"
    assert calls["conversation_ids"] == ["c1"]
    assert calls["runs_base_url"] == "http://127.0.0.1:8000"