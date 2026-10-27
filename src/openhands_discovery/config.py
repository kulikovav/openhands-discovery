"""Deployment configuration for the discovery service."""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_CONFIG_PATH = "/etc/agent-canvas/discovery.json"


class DiscoveryConfig(BaseModel):
    """The deployment settings, read once at start.

    The runs keys are optional: a configuration file from before the runs
    server existed leaves the backend document at the ingress and the slots.

    The backend names are display identifiers, because the bridge selects a
    backend by its role. The deployment can rename them; a document from before
    they were configurable keeps the defaults.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    domain: str
    env_file: str
    ingress_name: str = Field(default="canvas", min_length=1)
    ingress_port: int
    listen_port: int
    quadlet_dir: str
    runs_name: str = Field(default="automation-runs", min_length=1)
    slot_name_prefix: str = Field(default="backend-", min_length=1)
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