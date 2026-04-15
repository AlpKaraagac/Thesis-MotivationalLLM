"""Stage-3 hierarchy organization helpers."""

from __future__ import annotations

from typing import Any
import random

from discoverllm.types import IntentForest, IntentNode, IntentTree


def forest_from_hierarchy_payload(
    payload: dict[str, Any], *, stable_root_sort: bool = True
) -> IntentForest:
    """Build an IntentForest from the parsed hierarchy organizer payload."""

    if not isinstance(payload, dict):
        raise TypeError("payload must be a mapping.")
    trees_raw = payload.get("trees")
    if not isinstance(trees_raw, list) or not trees_raw:
        raise ValueError("payload.trees must be a non-empty list.")

    roots = [_node_from_payload(item) for item in trees_raw]
    if stable_root_sort:
        roots.sort(key=lambda node: _hierarchical_id_key(node.id))
    trees = [IntentTree(root=root) for root in roots]
    return IntentForest(trees=trees)


def assign_thresholds(forest: IntentForest, *, seed: int) -> None:
    """Assign per-node thresholds uniformly from [0, 1] under a fixed seed."""

    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    rng = random.Random(seed)
    for node in forest.iter_depth_first():
        node.threshold = rng.random()


def _node_from_payload(node_payload: dict[str, Any]) -> IntentNode:
    if not isinstance(node_payload, dict):
        raise TypeError("node payload must be a mapping.")
    try:
        node_id = str(node_payload["id"])
        text = str(node_payload["text"])
        raw_children = node_payload["children"]
    except KeyError as exc:
        raise ValueError(f"Missing required hierarchy node field: {exc.args[0]}") from exc

    if not isinstance(raw_children, list):
        raise TypeError(f"children must be a list for node {node_id!r}.")
    children = [_node_from_payload(child) for child in raw_children]
    return IntentNode(id=node_id, text=text, children=children)


def _hierarchical_id_key(node_id: str) -> tuple[int, ...]:
    return tuple(int(part) for part in node_id.split("."))
