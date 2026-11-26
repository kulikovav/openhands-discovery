"""Deployment configuration for the discovery service."""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_CONFIG_PATH = "/etc/agent-canvas/discovery.json"


class GovernedBackendConfig(BaseModel):
    """A governed backend that runs on another host.

    The service runs on the ingress host, so it cannot read the governed
    server's unit or reach its loopback address. The public host names the
    address the bridge uses, and the probe host names the tailnet address the
    health probe dials.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(default="governed", min_length=1)
    port: int = Field(ge=1, le=65535)
    probe_host: str = Field(min_length=1)
    public_host: str = Field(min_length=1)


class DiscoveryConfig(BaseModel):
    """The deployment settings, read once at start.

    The runs keys are optional: a configuration file from before the runs
    server existed leaves the backend document at the ingress and the slots.

    The backend names are display identifiers, because the bridge selects a
    backend by its role. The deployment can rename them; a document from before
    they were configurable keeps the defaults.

    ``governed`` is optional: a deployment that publishes a governed backend
    adds its entry to the backend list. The entry merges like any other
    backend, so its conversations appear in the merged conversation read.

    ``index_path`` is optional: the sqlite conversation index stays disabled
    while the key is unset, and a chat read then walks the conversation search,
    the way it did before the index existed.
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
    governed: GovernedBackendConfig | None = None
    index_path: str | None = None

    @property
    def public_root(self) -> str:
        """The public root URL of the deployment."""
        return f"{self.scheme}://{self.domain}"

    @property
    def runs_configured(self) -> bool:
        """Whether the automation runs server is part of the deployment."""
        return self.runs_port is not None and self.runs_public_host is not None

    @property
    def governed_configured(self) -> bool:
        """Whether a governed backend is part of the deployment."""
        return self.governed is not None


def load_config(path: str | Path) -> DiscoveryConfig:
    """Read and validate the configuration document.

    A missing key or an invalid value fails at start, not on the first
    request, so a broken deployment is visible in the service log.
    """
    with open(path, "r", encoding="utf-8") as handle:
        return DiscoveryConfig.model_validate(json.load(handle))