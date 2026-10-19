"""Response models and the backend role vocabulary."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BackendRole(str, Enum):
    """What a backend is for, from the bridge's point of view."""

    INGRESS = "ingress"
    SLOT = "slot"
    AUTOMATIONS = "automations"


class BackendEntry(BaseModel):
    """One OpenHands backend as the bridge reads it."""

    name: str
    role: BackendRole
    url: str
    port: int
    active: bool
    healthy: bool
    version: str = ""
    error: str = ""


class BackendRef(BaseModel):
    """The backend a merged row came from."""

    name: str
    role: BackendRole
    url: str


class UnavailableBackend(BaseModel):
    """A backend that contributed no rows to a merged read: a failed fan-out
    read or a failed health probe."""

    name: str
    role: BackendRole
    error: str


class ConversationRow(BaseModel):
    """One conversation row. Upstream fields pass through unchanged."""

    model_config = ConfigDict(extra="allow")

    id: str
    backend: BackendRef
    # The title and tags the search page already answers. They are typed here
    # so every row carries both, even when an upstream value is absent.
    title: str = ""
    tags: dict[str, str] = Field(default_factory=dict)


class ConversationsResponse(BaseModel):
    conversations: list[ConversationRow]
    next_page_id: str | None = None
    unavailable: list[UnavailableBackend] = Field(default_factory=list)


class UnavailableAutomation(BaseModel):
    """An automation whose run page failed."""

    id: str
    name: str
    error: str


class RunRow(BaseModel):
    """One automation run row. Upstream fields pass through unchanged."""

    model_config = ConfigDict(extra="allow")

    id: str
    automation_name: str = ""
    # The conversation state of a run that owns one. The fields stay empty
    # when the record read failed or the run owns no conversation, so the
    # bridge falls back to its own read instead of reading an empty value as
    # an unarchived chat. A null tag map is an unanswered state; an empty map
    # is a chat the deployment answered with no tags.
    conversation_title: str = ""
    conversation_tags: dict[str, str] | None = None


class RunsResponse(BaseModel):
    runs: list[RunRow]
    status_counts: dict[str, int] = Field(default_factory=dict)
    unavailable: list[UnavailableAutomation] = Field(default_factory=list)


class BackendsResponse(BaseModel):
    backends: list[BackendEntry]


def text_tags(tags: Any) -> dict[str, str]:
    """Read a tag map as text.

    A non-string value reads as empty, because the bridge decodes the tags into
    strings and one odd value must not fail the whole page. The deployment owns
    the tag shape, and a tag the bridge cannot read still proves nothing about
    the archive state.
    """
    if not isinstance(tags, dict):
        return {}
    return {str(key): value if isinstance(value, str) else "" for key, value in tags.items()}