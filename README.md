# OpenHands Discovery

Backend discovery and read aggregation for an OpenHands deployment.

The service runs on the deployment host. It compiles the list of OpenHands
backends and merges the deployment's list-shaped reads. The bridge reads this
service instead of fanning out to every backend itself.

The backend list has three roles:

- `automations`: the ingress container. It serves the automation API.
- `backend-N`: the cloud-agent slots. The bridge schedules conversations here.
- `automation-runs`: the server that owns the conversations of automation runs.
  The bridge reads transcripts here and never schedules here.

The service listens on `127.0.0.1` only. nginx publishes the
`/api/discovery/` paths on the deployment TLS port. Every path except
`/health` needs the deployment API key in `X-Session-API-Key`.

## Routes

| Route | Answer |
| --- | --- |
| `GET /health` | Liveness. No key. |
| `GET /api/discovery/backends` | The compiled backend list, with a live health result per backend. |
| `GET /api/discovery/conversations` | One merged conversation list, newest first, with a cursor for the remaining pages. |
| `GET /api/discovery/automation/runs` | One merged automation run list, newest first. |

`GET /api/discovery/conversations` accepts `limit`, `page_id`, and `role`.
`limit` is the page size per backend. `role` narrows the fan-out.

`GET /api/discovery/automation/runs` accepts `limit`, `automation_id`,
`run_id`, `status`, and `has_conversation`. `run_id` wins when a request sets
both `run_id` and `has_conversation`.

A backend that contributes no rows, because its health probe failed or its
read failed, appears in `unavailable`. An empty page therefore never hides a
deployment outage.

## Install and run

The service needs Python 3.10 or newer.

```sh
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/openhands-discovery /etc/agent-canvas/discovery.json
```

The config path is optional. Its default is
`/etc/agent-canvas/discovery.json`.

## Configuration

```json
{
  "domain": "oh.example",
  "env_file": "/home/agent-canvas/agent-canvas.env",
  "ingress_port": 8000,
  "listen_port": 18080,
  "quadlet_dir": "/home/agent-canvas/.config/containers/systemd",
  "runs_port": 8010,
  "runs_public_port": 9443,
  "scheme": "https",
  "slot_public_base_port": 8443
}
```

| Key | Meaning |
| --- | --- |
| `domain` | The public host name of the deployment. |
| `env_file` | The Podman environment file. The service reads `OH_SESSION_API_KEYS_0` from it on every request, so a key rotation applies without a restart. |
| `ingress_port` | The loopback port of the ingress automation API. |
| `listen_port` | The loopback port of this service. |
| `quadlet_dir` | The directory with the slot Quadlet units. The service reads the loopback publish port from each unit. |
| `runs_port` | The loopback port of the automation runs server. Optional. |
| `runs_public_port` | The public port of the automation runs server. Optional. |
| `scheme` | The public URL scheme. Default is `https`. |
| `slot_public_base_port` | The public port of slot 1. Slot N uses base + N - 1. |

The runs backend is present only when both `runs_port` and `runs_public_port`
are set.

## Design

The code models the deployment rules as data:

- `backends.BackendTarget` holds every value a probe needs, including whether
  the role reports a version. `build_targets` compiles the fixed order:
  ingress, slots, runs.
- `runs.RunScan` is the scan strategy of one request. `NewestRuns`,
  `ChatRuns`, and `SingleRun` implement it. `select_scan` holds the precedence
  rule in one place.
- `conversations.ConversationWalk` is the answer of one walk step: rows, the
  next cursor, and the backends that failed.
- `config.DiscoveryConfig` validates the document at start, so a bad
  deployment fails in the service log, not on the first request.

## Tests

```sh
.venv/bin/python -m pytest
```

## Layout

```text
src/openhands_discovery/
  api.py            routes, error bodies, fan-out orchestration
  auth.py           deployment key read and constant-time compare
  backends.py       backend targets, probes, compiled backend list
  config.py         typed configuration
  conversations.py  conversation walk, cursor, merge
  models.py         response models and backend roles
  runs.py           run scans, merge, chat projection
  upstream.py       shared upstream JSON read
  __main__.py       command line entry point
tests/              pytest suite
```