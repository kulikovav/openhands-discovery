import asyncio

import httpx

from openhands_discovery import backends
from openhands_discovery.backends import build_targets, compile_backends, slot_ports, unavailable_backends
from openhands_discovery.config import DiscoveryConfig
from openhands_discovery.models import BackendEntry, BackendRole


def make_config(tmp_path, **overrides):
    quadlet_dir = tmp_path / "quadlet"
    quadlet_dir.mkdir(exist_ok=True)
    values = {
        "domain": "oh.example",
        "env_file": str(tmp_path / "env"),
        "ingress_port": 8000,
        "listen_port": 18080,
        "quadlet_dir": str(quadlet_dir),
        "slot_public_host_prefix": "b",
    }
    values.update(overrides)
    return DiscoveryConfig(**values)


def test_slot_ports_reads_units_in_numeric_order(tmp_path):
    config = make_config(tmp_path)
    quadlet_dir = tmp_path / "quadlet"
    (quadlet_dir / "agent-canvas-slot-10.container").write_text(
        "PublishPort=127.0.0.1:18109:8000\n", encoding="utf-8"
    )
    (quadlet_dir / "agent-canvas-slot-2.container").write_text(
        "PublishPort=127.0.0.1:18101:8000\n", encoding="utf-8"
    )
    (quadlet_dir / "agent-canvas-slot-3.container").write_text("[Container]\n", encoding="utf-8")
    (quadlet_dir / "unrelated.container").write_text(
        "PublishPort=127.0.0.1:19999:8000\n", encoding="utf-8"
    )

    assert slot_ports(config) == [(2, 18101), (10, 18109)]


def test_slot_ports_returns_empty_when_the_directory_is_missing(tmp_path):
    config = make_config(tmp_path, quadlet_dir=str(tmp_path / "missing"))
    assert slot_ports(config) == []


def test_build_targets_puts_ingress_first_and_runs_last(tmp_path):
    config = make_config(tmp_path, runs_port=8010, runs_public_host="a1.oh.example")
    quadlet_dir = tmp_path / "quadlet"
    (quadlet_dir / "agent-canvas-slot-1.container").write_text(
        "PublishPort=127.0.0.1:18100:8000\n", encoding="utf-8"
    )

    targets = build_targets(config)

    assert [target.name for target in targets] == ["automations", "backend-1", "automation-runs"]
    assert [target.role for target in targets] == [
        BackendRole.INGRESS,
        BackendRole.SLOT,
        BackendRole.AUTOMATIONS,
    ]
    assert [target.url for target in targets] == [
        "https://oh.example",
        "https://b1.oh.example",
        "https://a1.oh.example",
    ]
    assert [target.port for target in targets] == [8000, 18100, 8010]
    assert [target.reports_version for target in targets] == [True, False, False]


def test_build_targets_names_each_slot_by_its_number(tmp_path):
    config = make_config(tmp_path)
    quadlet_dir = tmp_path / "quadlet"
    (quadlet_dir / "agent-canvas-slot-1.container").write_text(
        "PublishPort=127.0.0.1:18100:8000\n", encoding="utf-8"
    )
    (quadlet_dir / "agent-canvas-slot-10.container").write_text(
        "PublishPort=127.0.0.1:18109:8000\n", encoding="utf-8"
    )

    slot_urls = [target.url for target in build_targets(config) if target.role is BackendRole.SLOT]

    assert slot_urls == ["https://b1.oh.example", "https://b10.oh.example"]


def test_build_targets_skips_the_runs_server_when_a_key_is_missing(tmp_path):
    config = make_config(tmp_path, runs_port=8010)
    assert [target.name for target in build_targets(config)] == ["automations"]


def test_unavailable_backends_filters_by_role():
    ingress = BackendEntry(
        name="automations",
        role=BackendRole.INGRESS,
        url="https://oh.example",
        port=8000,
        active=True,
        healthy=True,
    )
    slot = BackendEntry(
        name="backend-2",
        role=BackendRole.SLOT,
        url="https://oh.example:8445",
        port=18101,
        active=True,
        healthy=False,
        error="the health endpoint is unreachable",
    )

    assert [entry.model_dump() for entry in unavailable_backends([ingress, slot], None)] == [
        {"name": "backend-2", "role": BackendRole.SLOT, "error": "the health endpoint is unreachable"}
    ]
    assert unavailable_backends([ingress, slot], BackendRole.INGRESS) == []


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.request = httpx.Request("GET", "http://127.0.0.1")

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("bad status", request=self.request, response=self)

    def json(self):
        return self.payload


class FakeClient:
    def __init__(self, handler):
        self.handler = handler

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def get(self, url, headers=None):
        return self.handler(url, headers or {})


def test_compile_backends_reports_the_ingress_version_only(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    quadlet_dir = tmp_path / "quadlet"
    (quadlet_dir / "agent-canvas-slot-1.container").write_text(
        "PublishPort=127.0.0.1:18100:8000\n", encoding="utf-8"
    )
    (quadlet_dir / "agent-canvas-slot-2.container").write_text(
        "PublishPort=127.0.0.1:18101:8000\n", encoding="utf-8"
    )

    def handler(url, headers):
        if ":18101" in url:
            return FakeResponse({}, status_code=503)
        if url.endswith("/api/automation/server_info"):
            return FakeResponse({"package_version": "1.2.3", "sdk_version": "0.9"})
        return FakeResponse({})

    async def fake_unit_is_active(unit_name):
        return True

    monkeypatch.setattr(backends, "unit_is_active", fake_unit_is_active)
    monkeypatch.setattr(backends.httpx, "AsyncClient", lambda timeout: FakeClient(handler))

    entries = asyncio.run(compile_backends(config, "secret"))

    assert [entry.name for entry in entries] == ["automations", "backend-1", "backend-2"]
    assert entries[0].version == "1.2.3 (sdk 0.9)"
    assert [entry.version for entry in entries[1:]] == ["", ""]
    assert entries[0].healthy is True
    assert entries[1].healthy is True
    assert entries[2].healthy is False
    assert entries[2].error == "the health endpoint answered HTTP 503"
    assert all(entry.active for entry in entries)


def test_compile_backends_leaves_the_version_empty_on_a_failed_read(tmp_path, monkeypatch):
    config = make_config(tmp_path)

    def handler(url, headers):
        if url.endswith("/api/automation/server_info"):
            return FakeResponse({}, status_code=500)
        return FakeResponse({})

    async def fake_unit_is_active(unit_name):
        return False

    monkeypatch.setattr(backends, "unit_is_active", fake_unit_is_active)
    monkeypatch.setattr(backends.httpx, "AsyncClient", lambda timeout: FakeClient(handler))

    entries = asyncio.run(compile_backends(config, "secret"))

    assert entries[0].version == ""
    assert entries[0].active is False


class CompletedProcess:
    def __init__(self, returncode):
        self.returncode = returncode


def test_unit_is_active_sync_reads_the_return_code(monkeypatch):
    monkeypatch.setattr(backends.subprocess, "run", lambda *args, **kwargs: CompletedProcess(0))
    assert backends.unit_is_active_sync("agent-canvas.service") is True

    monkeypatch.setattr(backends.subprocess, "run", lambda *args, **kwargs: CompletedProcess(3))
    assert backends.unit_is_active_sync("agent-canvas.service") is False


def test_unit_is_active_sync_handles_a_missing_systemctl(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("systemctl is not available")

    monkeypatch.setattr(backends.subprocess, "run", fail)
    assert backends.unit_is_active_sync("agent-canvas.service") is False