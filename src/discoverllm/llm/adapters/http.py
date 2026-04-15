"""Shared HTTP helpers for provider-backed LLM adapters."""

from __future__ import annotations

import json
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ProviderAPIError(RuntimeError):
    """Raised when a provider API request fails or returns malformed output."""


def post_json(
    *,
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    """POST JSON and return the decoded response payload."""

    timeout_seconds = float(timeout_seconds)
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive.")

    request = Request(
        url=str(url),
        data=json.dumps(payload).encode("utf-8"),
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            response_text = response.read().decode("utf-8")
    except HTTPError as exc:
        response_text = exc.read().decode("utf-8", errors="replace")
        raise ProviderAPIError(
            f"Provider request failed with HTTP {exc.code}: {response_text}"
        ) from exc
    except URLError as exc:
        raise ProviderAPIError(f"Provider request failed: {exc.reason}") from exc

    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise ProviderAPIError(
            f"Provider returned invalid JSON: {response_text}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ProviderAPIError("Provider returned a non-object JSON payload.")
    return parsed
