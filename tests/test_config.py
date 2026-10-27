import json

import pytest
from pydantic import ValidationError

from openhands_discovery.config import load_config


def write_config(tmp_path, **overrides):
    values = {
        "domain": "oh.example",
        "env_file": "/tmp/env",
        "ingress_port": 8000,
        "listen_port": 18080,
        "quadlet_dir": "/tmp/quadlet",
        "slot_public_host_prefix": "b",
    }
    values.update(overrides)
    path = tmp_path / "discovery.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    return path


def test_load_config_reads_the_optional_runs_keys(tmp_path):
    config = load_config(
        write_config(tmp_path, runs_port=8010, runs_public_host="a1.oh.example")
    )

    assert config.runs_configured is True
    assert config.public_root == "https://oh.example"


def test_load_config_leaves_the_runs_backend_out_without_both_keys(tmp_path):
    config = load_config(write_config(tmp_path, runs_port=8010))

    assert config.runs_configured is False


def test_load_config_rejects_a_missing_key(tmp_path):
    path = write_config(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["domain"]
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_config(path)


def test_load_config_defaults_the_backend_names(tmp_path):
    config = load_config(write_config(tmp_path))

    assert (config.ingress_name, config.slot_name_prefix, config.runs_name) == (
        "canvas",
        "backend-",
        "automation-runs",
    )


def test_load_config_reads_the_backend_names(tmp_path):
    config = load_config(
        write_config(tmp_path, ingress_name="oh", slot_name_prefix="agent-", runs_name="runs")
    )

    assert (config.ingress_name, config.slot_name_prefix, config.runs_name) == (
        "oh",
        "agent-",
        "runs",
    )