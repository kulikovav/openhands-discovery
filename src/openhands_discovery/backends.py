"""The backend inventory: which OpenHands backends exist and how they answer.

The inventory is compiled from the slot Quadlet units on every request, so a
slot that is added, removed, or restarted is reported without a restart of this
service. The probe itself and the suspension of a failing backend live in
``probe``.
"""

import asyncio
import os
import re
import subprocess
from dataclasses import dataclass

import httpx

from .config import DiscoveryConfig
from .models import BackendEntry, BackendRole, UnavailableBackend
from .probe import PROBE_KIND_ERROR, PROBE_TIMEOUT_SECONDS, ProbeCircuits, probe_error, probe_health
from .upstream import API_KEY_HEADER

SERVER_INFO_PATH = "/api/automation/server_info"

PUBLISH_PORT_PATTERN = re.compile(r"^PublishPort=127\.0\.0\.1:(\d+):8000$")
SLOT_UNIT_PATTERN = re.compile(r"^agent-canvas-slot-(\d+)\.container$")


@dataclass(frozen=True)
class BackendTarget:
    """One backend the service probes, with every value the probe needs.

    ``reports_version`` encodes the one difference between the roles: a slot
    runs the agent server alone and serves no automation API, so only the
    ingress reports a version.
    """

    name: str
    role: BackendRole
    url: str
    port: int
    unit: str
    reports_version: bool = False


def read_publish_port(unit_path: str) -> int | None:
    """Return the host loopback port of a slot unit, or None when the unit has
    no loopback publish line."""
    try:
        with open(unit_path, "r", encoding="utf-8") as handle:
            for line in handle:
                match = PUBLISH_PORT_PATTERN.match(line.strip())
                if match:
                    return int(match.group(1))
    except OSError:
        return None
    return None


def slot_ports(config: DiscoveryConfig) -> list[tuple[int, int]]:
    """Return (slot number, loopback port) pairs from the slot units, in slot
    order.

    A unit that cannot be read or publishes no loopback port is skipped, so one
    broken unit cannot hide the other slots.
    """
    try:
        names = os.listdir(config.quadlet_dir)
    except OSError:
        return []
    slots = []
    for name in names:
        match = SLOT_UNIT_PATTERN.match(name)
        if not match:
            continue
        port = read_publish_port(os.path.join(config.quadlet_dir, name))
        if port is None:
            continue
        slots.append((int(match.group(1)), port))
    slots.sort()
    return slots


def build_targets(config: DiscoveryConfig) -> list[BackendTarget]:
    """Compile the probe targets in the fixed order: ingress, slots, runs.

    The names come from the configuration, because they are display
    identifiers: the bridge selects a backend by its role.
    """
    public_root = config.public_root
    targets = [
        BackendTarget(
            name=config.ingress_name,
            role=BackendRole.INGRESS,
            url=public_root,
            port=config.ingress_port,
            unit="agent-canvas.service",
            reports_version=True,
        )
    ]
    for number, port in slot_ports(config):
        targets.append(
            BackendTarget(
                name=f"{config.slot_name_prefix}{number}",
                role=BackendRole.SLOT,
                url=f"{config.scheme}://{config.slot_public_host_prefix}{number}.{config.domain}",
                port=port,
                unit=f"agent-canvas-slot-{number}.service",
            )
        )
    if config.runs_configured:
        targets.append(
            BackendTarget(
                name=config.runs_name,
                role=BackendRole.AUTOMATIONS,
                url=f"{config.scheme}://{config.runs_public_host}",
                port=config.runs_port,
                unit="agent-canvas-runs.service",
            )
        )
    return targets


def unit_is_active_sync(unit_name: str) -> bool:
    """Report whether a user unit is active.

    The discovery service runs in the same user manager as the Quadlet units,
    so ``systemctl --user`` reaches them without an extra environment.
    """
    try:
        result = subprocess.run(
            ["systemctl", "--user", "is-active", unit_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


async def unit_is_active(unit_name: str) -> bool:
    """systemctl is a blocking subprocess, so it runs off the event loop."""
    return await asyncio.to_thread(unit_is_active_sync, unit_name)


async def probe_version(client: httpx.AsyncClient, port: int, api_key: str) -> str:
    """Read the automation server version of the ingress deployment.

    A failure leaves the version empty: the health probe already proved
    reachability.
    """
    url = f"http://127.0.0.1:{port}{SERVER_INFO_PATH}"
    try:
        response = await client.get(url, headers={API_KEY_HEADER: api_key})
        response.raise_for_status()
        info = response.json()
    except (httpx.HTTPError, ValueError):
        return ""
    package = str(info.get("package_version", "") or "").strip()
    sdk = str(info.get("sdk_version", "") or "").strip()
    if package and sdk:
        return f"{package} (sdk {sdk})"
    return package or sdk


async def compile_backends(
    config: DiscoveryConfig,
    api_key: str,
    circuits: ProbeCircuits,
) -> list[BackendEntry]:
    """Compile the backend list with the result of a live probe per backend.

    Probes run concurrently, so a dead backend costs one probe timeout, not one
    per backend. Every probe records its own answer or releases its in-flight
    mark, so a read that is cancelled mid-walk leaves no backend silenced. A
    suspended backend is probed only past its retry time, and a backend whose
    probe is already running answers from its recorded entry, so a concurrent
    read repeats no probe and counts no strike.
    """
    targets = build_targets(config)
    probed_targets = [target for target in targets if circuits.due(target.name)]
    for target in probed_targets:
        circuits.begin(target.name)

    async with httpx.AsyncClient(timeout=PROBE_TIMEOUT_SECONDS) as client:

        async def probe_one(target: BackendTarget) -> None:
            try:
                healthy, kind, error = await probe_health(client, target.port)
            except asyncio.CancelledError:
                circuits.release(target.name)
                raise
            except BaseException as error:
                circuits.release(target.name)
                circuits.record(
                    target.name,
                    False,
                    PROBE_KIND_ERROR,
                    probe_error("the health probe failed", error),
                )
                return
            circuits.record(target.name, healthy, kind, error)

        answers = await asyncio.gather(
            *[unit_is_active(target.unit) for target in targets],
            *[probe_one(target) for target in probed_targets],
            return_exceptions=True,
        )
        # Only the ingress reports a version, and only a healthy answer
        # refreshes it; a suspended ingress answers no version.
        version = ""
        if targets and targets[0].reports_version and circuits.entry(targets[0].name)[0]:
            version = await probe_version(client, targets[0].port, api_key)

    entries = []
    for index, target in enumerate(targets):
        healthy, error = circuits.entry(target.name)
        entries.append(
            BackendEntry(
                name=target.name,
                role=target.role,
                url=target.url,
                port=target.port,
                active=answers[index],
                healthy=healthy,
                version=version if target.reports_version else "",
                error=error,
            )
        )
    return entries


def unavailable_backends(backends: list[BackendEntry], role: BackendRole | None) -> list[UnavailableBackend]:
    """The backends of one walk that contribute no rows: those whose health
    probe failed.

    They are reported like a failed read, so a caller that reads an empty page
    can tell a deployment outage from a user who owns nothing. The role filter
    of the request applies here too, so a narrowed walk reports only the
    backends it would have read.
    """
    return [
        UnavailableBackend(
            name=backend.name,
            role=backend.role,
            error=backend.error or "the health probe failed",
        )
        for backend in backends
        if not backend.healthy and (role is None or backend.role == role)
    ]