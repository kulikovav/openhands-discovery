"""Shared reads against the OpenHands backends."""

from typing import Any

import httpx

API_KEY_HEADER = "X-Session-API-Key"


async def fetch_json(
    client: httpx.AsyncClient,
    url: str,
    api_key: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Read one upstream page as a JSON object.

    A payload that is not an object reads as an empty object, so a caller
    reads the same shape from every backend.
    """
    response = await client.get(url, params=params, headers={API_KEY_HEADER: api_key})
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {}