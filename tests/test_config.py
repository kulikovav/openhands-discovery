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
        "slot_public_base_port": 8443,
    }
    values.update(overrides)
    path = tmp_path / "discovery.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    return path


def test_load_config_reads_the_optional_runs_keys(tmp_path):
    config = load_config(write_config(tmp_path, runs_port=8010, runs_public_port=9443))

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