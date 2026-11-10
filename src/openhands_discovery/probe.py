"""The backend health probe and the suspension of a failing backend.

The probe classes separate the failure kinds the suspension state machine
reads, and the circuits keep one suspension per backend name: a backend whose
probe keeps timing out stops being probed on every read and rejoins the merge
on the first healthy answer.
"""

import random
import time
from collections.abc import Callable

import httpx

# The probe budget before a backend answers unhealthy. The probes run
# concurrently before the merged read, and the read window is
# LIST_TIMEOUT_SECONDS in api.py; a slow but alive /health answer must still
# probe healthy, or the merged page drops the backend's rows and the bridge's
# agent list flaps.
PROBE_TIMEOUT_SECONDS = 5
HEALTH_PATH = "/health"

# Three consecutive timeout strikes suspend a backend's probe. A suspended
# backend reads no rows, and its probe waits for the retry schedule instead of
# running on every read, so a dead backend costs nothing per merged page. A
# healthy answer rejoins it immediately.
PROBE_TIMEOUTS_TO_SUSPEND = 3

# The suspended probe waits retry_backoff_seconds(the failed retry) before the
# next attempt: the initial wait doubling per failed retry, capped, then scaled
# by the jitter band.
RETRY_BACKOFF_INITIAL_SECONDS = 5.0
RETRY_BACKOFF_MAX_SECONDS = 60.0
RETRY_BACKOFF_JITTER = (0.8, 1.2)

# The failure class of one probe. A timeout and the unnamed transport errors
# of the error class are ambiguous, so both count strikes toward the
# suspension. A refusal proves the server answered nothing at its address, so
# it suspends at once. An HTTP status failure proves something is listening,
# so it marks the round unhealthy without counting toward the suspension.
PROBE_KIND_HEALTHY = "healthy"
PROBE_KIND_TIMEOUT = "timeout"
PROBE_KIND_REFUSAL = "refusal"
PROBE_KIND_STATUS = "status"
PROBE_KIND_ERROR = "error"


async def probe_health(client: httpx.AsyncClient, port: int) -> tuple[bool, str, str]:
    """Return (healthy, kind, error). The health path answers without a key.

    The kind names the failure class the suspension state machine reads: a
    timeout and the unnamed transport errors are ambiguous, a refusal proves
    the backend is down, and an HTTP status failure proves it is listening.
    """
    url = f"http://127.0.0.1:{port}{HEALTH_PATH}"
    try:
        response = await client.get(url)
    except httpx.TimeoutException as error:
        return False, PROBE_KIND_TIMEOUT, probe_error("the health endpoint timed out", error)
    except httpx.ConnectError as error:
        return False, PROBE_KIND_REFUSAL, probe_error("the health endpoint is unreachable", error)
    except httpx.HTTPError as error:
        return False, PROBE_KIND_ERROR, probe_error("the health endpoint is unreachable", error)
    if response.status_code == 200:
        return True, PROBE_KIND_HEALTHY, ""
    return False, PROBE_KIND_STATUS, f"the health endpoint answered HTTP {response.status_code}"


def probe_error(prefix: str, error: Exception) -> str:
    """Render one probe failure, naming its class and its detail.

    The wrapped message can be empty, so the exception class name stands in
    for it and no rendered line ends in a bare colon.
    """
    detail = str(error).strip() or type(error).__name__
    return f"{prefix}: {detail}"


def retry_backoff_seconds(attempt: int) -> float:
    """The wait before the next suspended probe attempt, before the jitter.

    The initial wait doubles with every failed retry and is capped at
    RETRY_BACKOFF_MAX_SECONDS: ``min(retry_backoff_initial * 2^attempt,
    retry_backoff_max)``. The caller scales it inside the jitter band.
    """
    return min(RETRY_BACKOFF_INITIAL_SECONDS * 2**attempt, RETRY_BACKOFF_MAX_SECONDS)


class ProbeCircuit:
    """The suspension state of one backend's health probe."""

    __slots__ = ("attempt", "error", "healthy", "in_flight", "next_probe_at", "strikes", "suspended")

    def __init__(self) -> None:
        self.strikes = 0
        self.suspended = False
        self.attempt = 0
        self.next_probe_at = 0.0
        # A backend with no recorded answer yet is treated healthy: the evidence
        # arrives with its first probe.
        self.healthy = True
        self.error = ""
        self.in_flight = False


class ProbeCircuits:
    """Per-backend health-probe suspension, kept in process.

    One timeout is a jittery poll; three consecutive timeouts suspend the
    probe, and a refusal suspends it at once, because the refusal proves the
    server answered nothing at its address. A suspended backend is probed only
    when its retry is due, so a merged read costs no probe wait for it, and a
    healthy answer rejoins it at once. The suspension survives no restart: a
    fresh process probes every backend with no strikes behind it.
    """

    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        jitter: Callable[[], float] | None = None,
    ) -> None:
        self._clock = clock
        self._jitter = jitter if jitter is not None else default_jitter
        self._circuits: dict[str, ProbeCircuit] = {}

    def due(self, name: str) -> bool:
        """Report whether this read must probe the backend now.

        A backend whose probe is running answers from entry(): the read repeats
        no probe and counts no strike, so overlapping reads add no strikes. A
        suspended backend is probed only past its retry time.
        """
        circuit = self._circuits.get(name)
        if circuit is None:
            return True
        if circuit.in_flight:
            return False
        if not circuit.suspended:
            return True
        return self._clock() >= circuit.next_probe_at

    def begin(self, name: str) -> None:
        """Mark one backend's probe as running, so a concurrent read answers
        its entry instead of repeating the probe."""
        circuit = self._circuit(name)
        circuit.in_flight = True

    def release(self, name: str) -> None:
        """Release an unfinished probe, so the next read can probe again.

        A probe that did not record its answer, because the read was cancelled
        or failed outside the probe, must not leave the backend in flight: the
        in-flight mark would silence its probe until the process restarts.
        """
        circuit = self._circuits.get(name)
        if circuit is not None:
            circuit.in_flight = False

    def entry(self, name: str) -> tuple[bool, str]:
        """The recorded answer of one backend: the row a no-probe read serves."""
        circuit = self._circuit(name)
        if circuit.suspended:
            return False, self._entry_error(circuit)
        return circuit.healthy, circuit.error

    def record(self, name: str, healthy: bool, kind: str, error: str) -> tuple[bool, str]:
        """Store one probe result and answer the row (healthy, error) to serve.

        A healthy answer clears every strike. A timeout strike suspends the
        third one, and the unnamed transport failure counts the same; a refusal
        suspends at once. A failed retry of a suspended backend advances the
        schedule; an HTTP status failure of an active backend breaks the
        timeout chain without suspending it.
        """
        circuit = self._circuit(name)
        circuit.in_flight = False
        circuit.healthy = healthy
        if healthy:
            circuit.strikes = 0
            circuit.suspended = False
            circuit.attempt = 0
            circuit.next_probe_at = 0.0
            circuit.error = ""
            return True, ""
        circuit.error = error
        if kind in (PROBE_KIND_TIMEOUT, PROBE_KIND_ERROR):
            if circuit.suspended:
                circuit.attempt += 1
                circuit.next_probe_at = self._retry_at(circuit.attempt)
            else:
                circuit.strikes += 1
                if circuit.strikes >= PROBE_TIMEOUTS_TO_SUSPEND:
                    circuit.suspended = True
                    circuit.attempt = 0
                    circuit.next_probe_at = self._retry_at(0)
        elif kind == PROBE_KIND_REFUSAL:
            if circuit.suspended:
                circuit.attempt += 1
            else:
                circuit.suspended = True
                circuit.attempt = 0
            circuit.next_probe_at = self._retry_at(circuit.attempt)
        elif circuit.suspended:
            circuit.attempt += 1
            circuit.next_probe_at = self._retry_at(circuit.attempt)
        else:
            circuit.strikes = 0
        return False, self._entry_error(circuit)

    def _circuit(self, name: str) -> ProbeCircuit:
        circuit = self._circuits.get(name)
        if circuit is None:
            circuit = self._circuits[name] = ProbeCircuit()
        return circuit

    def _retry_at(self, attempt: int) -> float:
        return self._clock() + retry_backoff_seconds(attempt) * self._jitter()

    def _entry_error(self, circuit: ProbeCircuit) -> str:
        """The error text one entry serves: the recorded error, with the
        suspension reason and wait when the backend is suspended."""
        if not circuit.suspended:
            return circuit.error
        remaining = max(0, round(circuit.next_probe_at - self._clock()))
        return f"the health probe is suspended, next attempt in {remaining}s: {circuit.error}"


def default_jitter() -> float:
    """One jitter multiplier inside RETRY_BACKOFF_JITTER."""
    return random.uniform(*RETRY_BACKOFF_JITTER)