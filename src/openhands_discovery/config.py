"""Deployment configuration for the discovery service."""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

DEFAULT_CONFIG_PATH = "/etc/agent-canvas/discovery.json"


class DiscoveryConfig(BaseModel):
    """The deployment settings, read once at start.

    The runs keys are optional: a configuration file from before the runs
    server existed leaves the backend document at the ingress and the slots.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    domain: str
    env_file: str
    ingress_port: int
    listen_port: int
    quadlet_dir: str
    slot_public_host_prefix: str
    scheme: str = "https"
    runs_port: int | None = None
    runs_public_host: str | None = None

    @property
    def public_root(self) -> str:
        """The public root URL of the deployment."""
        return f"{self.scheme}://{self.domain}"

    @property
    def runs_configured(self) -> bool:
        """Whether the automation runs server is part of the deployment."""
        return self.runs_port is not None and self.runs_public_host is not None


def load_config(path: str | Path) -> DiscoveryConfig:
    """Read and validate the configuration document.

    A missing key or an invalid value fails at start, not on the first
    request, so a broken deployment is visible in the service log.
    """
    with open(path, "r", encoding="utf-8") as handle:
        return DiscoveryConfig.model_validate(json.load(handle))