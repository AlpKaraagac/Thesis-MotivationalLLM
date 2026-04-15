"""Deterministic repair helpers for malformed structured outputs."""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from discoverllm.parsers.common import StructuredParseError, candidate_payload

TRAILING_COMMA_PATTERN = re.compile(r",(\s*[}\]])")
SINGLE_QUOTE_KEY_PATTERN = re.compile(r"'([A-Za-z_][A-Za-z0-9_]*)'\s*:")


def normalize_json_text(raw_text: str) -> str:
    """Apply deterministic text normalization for JSON parsing."""

    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be a string.")
    payload = candidate_payload(raw_text, languages=("json",))
    text = payload.text.strip()
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")
    text = TRAILING_COMMA_PATTERN.sub(r"\1", text)
    return text


def parse_python_like_structure(raw_text: str) -> Any:
    """Parse dict/list outputs that resemble Python literals."""

    normalized = normalize_json_text(raw_text)
    patched = _to_python_literal(normalized)
    try:
        value = ast.literal_eval(patched)
    except Exception as exc:  # noqa: BLE001
        raise StructuredParseError("Could not parse python-like structured output.") from exc
    return _coerce_json_compatible(value)


def _to_python_literal(text: str) -> str:
    # Convert JSON booleans/null to Python for literal_eval, while keeping quotes intact.
    text = re.sub(r"\btrue\b", "True", text, flags=re.IGNORECASE)
    text = re.sub(r"\bfalse\b", "False", text, flags=re.IGNORECASE)
    text = re.sub(r"\bnull\b", "None", text, flags=re.IGNORECASE)
    return text


def _coerce_json_compatible(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_coerce_json_compatible(item) for item in value]
    if isinstance(value, tuple):
        return [_coerce_json_compatible(item) for item in value]
    if isinstance(value, dict):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                key = str(key)
            converted[key] = _coerce_json_compatible(item)
        return converted
    raise StructuredParseError(f"Unsupported parsed value type: {type(value)!r}")


def parse_json_or_repair(raw_text: str) -> Any:
    """Parse JSON first, then deterministic repair fallbacks."""

    normalized = normalize_json_text(raw_text)
    try:
        return json.loads(normalized)
    except json.JSONDecodeError:
        return parse_python_like_structure(normalized)
