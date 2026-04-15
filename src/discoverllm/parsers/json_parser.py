"""JSON parser with deterministic repair path."""

from __future__ import annotations

import json
from typing import Any

from discoverllm.parsers.common import StructuredParseError, candidate_payload
from discoverllm.parsers.repair import parse_json_or_repair


def parse_json_strict(raw_text: str) -> Any:
    """Parse JSON without repair; raises on malformed payloads."""

    payload = candidate_payload(raw_text, languages=("json",))
    try:
        return json.loads(payload.text)
    except json.JSONDecodeError as exc:
        raise StructuredParseError(
            f"Strict JSON parse failed from {payload.source}: {exc.msg}"
        ) from exc


def parse_json_with_repair(raw_text: str) -> Any:
    """Parse JSON and attempt deterministic repair for common malformed outputs."""

    try:
        return parse_json_strict(raw_text)
    except StructuredParseError:
        try:
            return parse_json_or_repair(raw_text)
        except Exception as exc:  # noqa: BLE001
            raise StructuredParseError("JSON parse failed after deterministic repair.") from exc
