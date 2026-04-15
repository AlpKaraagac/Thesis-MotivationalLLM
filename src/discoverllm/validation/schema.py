"""Primitive schema-level validation helpers."""

from __future__ import annotations

import re

NODE_ID_PATTERN = re.compile(r"^[1-9]\d*(\.[1-9]\d*)*$")


def validate_node_id(node_id: str) -> str:
    if not isinstance(node_id, str):
        raise TypeError("node_id must be a string.")
    normalized = node_id.strip()
    if not normalized:
        raise ValueError("node_id must be non-empty.")
    if not NODE_ID_PATTERN.match(normalized):
        raise ValueError(
            f"Invalid node_id {normalized!r}. Expected dot-separated positive integers."
        )
    return normalized


def validate_threshold(value: float | int | None) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise TypeError("threshold must be numeric or None.")
    as_float = float(value)
    if not 0.0 <= as_float <= 1.0:
        raise ValueError("threshold must be in [0, 1].")
    return as_float
