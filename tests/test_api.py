import asyncio

import httpx

from openhands_discovery import api
from openhands_discovery.config import DiscoveryConfig
from openhands_discovery.conversations import ConversationWalk
from openhands_discovery.models import BackendEntry, BackendRole, UnavailableBackend
from openhands_discovery.runs import AutomationRunsPage, ChatRuns, RunReadResult


def make_config(tmp_path, key="secret", **overrides):
    env_file = tmp_path / "env"
    env_file.write_text(f"OH_SESSION_API_KEYS_0={key}\n", encoding="utf-8")
    values = {
        "domain": "oh.example",
        "env_file": str(env_file),
        "ingress_port": 8000,
        "listen_port": 18080,
        "quadlet_dir": str(tmp_path / "quadlet"),
        "slot_public_base_port": 8443,
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
    entry = make_backend("automations", BackendRole.INGRESS, 8000)

    async def fake_compile(config, api_key):
        assert api_key == "secret"
        return [entry]

    monkeypatch.setattr(api, "compile_backends", fake_compile)
    response = get(api.create_app(make_config(tmp_path)), "/api/discovery/backends", headers=key_headers())

    assert response.status_code == 200
    assert response.json() == {
        "backends": [
            {
                "name": "automations",
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
    ingress = make_backend("automations", BackendRole.INGRESS, 8000)
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
        assert [backend.name for backend in backends] == ["automations"]
        return ConversationWalk(
            conversations=[],
            next_page_id=None,
            unavailable=[
                UnavailableBackend(
                    name="automations",
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
            "name": "automations",
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