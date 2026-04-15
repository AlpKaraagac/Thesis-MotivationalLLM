"""Parser-side validation utilities for structured prompt contracts."""

from __future__ import annotations

from typing import Any, Iterable

from discoverllm.parsers.common import StructuredParseError
from discoverllm.validation.schema import validate_node_id


def require_mapping(payload: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise StructuredParseError(f"{context} must be a mapping object.")
    return payload


def require_keys(
    payload: dict[str, Any], required_keys: Iterable[str], *, context: str
) -> None:
    missing = [key for key in required_keys if key not in payload]
    if missing:
        joined = ", ".join(missing)
        raise StructuredParseError(f"{context} missing required keys: {joined}")


def validate_tree_nodes(root_nodes: list[dict[str, Any]]) -> None:
    """Validate uniqueness and hierarchical parent-child ID relationships."""

    if not isinstance(root_nodes, list):
        raise StructuredParseError("Tree payload must be a list of root node objects.")
    if not root_nodes:
        raise StructuredParseError("Tree payload must contain at least one root node.")

    seen_ids: set[str] = set()

    def visit(node: dict[str, Any], parent_id: str | None) -> None:
        node_obj = require_mapping(node, context="tree node")
        require_keys(node_obj, ("id", "text", "children"), context="tree node")
        node_id = validate_node_id(str(node_obj["id"]))

        text_value = node_obj["text"]
        if not isinstance(text_value, str) or not text_value.strip():
            raise StructuredParseError(f"Node text must be non-empty for node {node_id!r}.")

        if node_id in seen_ids:
            raise StructuredParseError(f"Duplicate node id {node_id!r}.")
        seen_ids.add(node_id)

        if parent_id is not None:
            expected_prefix = f"{parent_id}."
            if not node_id.startswith(expected_prefix):
                raise StructuredParseError(
                    f"Child id {node_id!r} must start with parent prefix {expected_prefix!r}."
                )
            if node_id.count(".") != parent_id.count(".") + 1:
                raise StructuredParseError(
                    f"Child id {node_id!r} must be an immediate child of {parent_id!r}."
                )
        else:
            if "." in node_id:
                raise StructuredParseError(
                    f"Root node id {node_id!r} must be top-level (no dots)."
                )

        children = node_obj["children"]
        if not isinstance(children, list):
            raise StructuredParseError(f"children must be a list for node {node_id!r}.")
        for child in children:
            visit(child, node_id)

    for root in root_nodes:
        visit(root, None)


def validate_evaluator_children_consistency(
    evaluations: list[dict[str, Any]] | list[Any],
) -> None:
    """Reject child evaluations under parents that were not directly engaged."""

    if not isinstance(evaluations, list):
        raise StructuredParseError("evaluations must be a list.")
    engaged_map: dict[str, bool] = {}

    for item in evaluations:
        entry = require_mapping(item, context="evaluation entry")
        require_keys(
            entry,
            ("node_id", "is_satisfied_or_probed", "children_evaluated"),
            context="evaluation entry",
        )
        node_id = validate_node_id(str(entry["node_id"]))
        engaged = entry["is_satisfied_or_probed"]
        if not isinstance(engaged, bool):
            raise StructuredParseError(
                f"is_satisfied_or_probed must be boolean for node {node_id!r}."
            )
        children_evaluated = entry["children_evaluated"]
        if not isinstance(children_evaluated, bool):
            raise StructuredParseError(
                f"children_evaluated must be boolean for node {node_id!r}."
            )

        if not engaged and children_evaluated:
            raise StructuredParseError(
                f"Node {node_id!r} cannot report children_evaluated=true when not engaged."
            )

        parent_id = _parent_id(node_id)
        if parent_id is not None:
            parent_engaged = engaged_map.get(parent_id)
            if parent_engaged is False:
                raise StructuredParseError(
                    f"Child node {node_id!r} cannot be evaluated when parent {parent_id!r} was not engaged."
                )
            if parent_engaged is None:
                raise StructuredParseError(
                    f"Evaluation order invalid: parent {parent_id!r} missing before child {node_id!r}."
                )

        engaged_map[node_id] = engaged


def validate_single_awareness_bucket(
    payload: dict[str, Any],
    *,
    bucket_keys: tuple[str, str, str] = (
        "pursuing_clear",
        "pursuing_fuzzy",
        "latent_goal",
    ),
) -> None:
    """Ensure at most one awareness bucket is populated."""

    obj = require_mapping(payload, context="user response payload")
    populated = [key for key in bucket_keys if _is_populated(obj.get(key))]
    if len(populated) > 1:
        joined = ", ".join(populated)
        raise StructuredParseError(
            f"Only one awareness bucket may be populated; found: {joined}"
        )


def _parent_id(node_id: str) -> str | None:
    if "." not in node_id:
        return None
    return node_id.rsplit(".", 1)[0]


def _is_populated(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) > 0
    if isinstance(value, str):
        return bool(value.strip())
    return True
