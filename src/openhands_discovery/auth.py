"""Deployment API key authentication."""

import hmac

from fastapi import Header, HTTPException, Request

from .config import DiscoveryConfig
from .upstream import API_KEY_HEADER

API_KEY_NAME = "OH_SESSION_API_KEYS_0"


def read_api_key(env_path: str) -> str:
    """Read the session API key from the Podman environment file.

    The file keeps values literal and splits on the first '=', so the value is
    taken as written, without quotes to strip.
    """
    try:
        with open(env_path, "r", encoding="utf-8") as handle:
            for line in handle:
                name, separator, value = line.rstrip("\n").partition("=")
                if separator and name == API_KEY_NAME:
                    return value
    except OSError:
        return ""
    return ""


def keys_match(supplied: str, expected: str) -> bool:
    """Compare a supplied key with the deployment key in constant time.

    The key is a shared secret, and an early exit on the first differing byte
    would leak its prefix. The values are encoded, so a non-ASCII value answers
    401 instead of raising. An unreadable environment file leaves the expected
    key empty, and an empty key matches nothing.
    """
    return bool(expected) and hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))


def require_api_key(
    request: Request,
    x_session_api_key: str = Header(default="", alias=API_KEY_HEADER),
) -> str:
    """Authenticate a request and return the key for the fan-out reads.

    The key is read per request, so a key rotation in the environment file
    applies to the next request instead of the next restart.
    """
    config: DiscoveryConfig = request.app.state.config
    expected = read_api_key(config.env_file)
    if not keys_match(x_session_api_key, expected):
        raise HTTPException(status_code=401, detail=f"a valid {API_KEY_HEADER} header is required")
    return expected