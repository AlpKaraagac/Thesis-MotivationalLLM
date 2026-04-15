"""Stage-4 initial request helpers and t=0 discovery application."""

from __future__ import annotations

import random
from typing import Any

from discoverllm.text_grounding import grounding_score
from discoverllm.types import (
    ArtifactSpec,
    ConversationMessage,
    ConversationState,
    IntentForest,
    IntentTree,
    NodeState,
)

_MAX_VISIBLE_ROOT_CRITERIA = 3


def root_criteria_payload(forest: IntentForest) -> list[dict[str, str]]:
    """Serialize root criteria for prompt input."""

    return [{"id": tree.root.id, "text": tree.root.text} for tree in forest.trees]


def partition_root_criteria_for_initial_request(
    forest: IntentForest,
    *,
    seed: int,
    forced_latent_root_ids: list[str] | None = None,
    max_visible_root_criteria: int = _MAX_VISIBLE_ROOT_CRITERIA,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Split root criteria into a small visible set and a latent remainder."""

    if not isinstance(forest, IntentForest):
        raise TypeError("forest must be an IntentForest.")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    if not isinstance(max_visible_root_criteria, int) or max_visible_root_criteria < 1:
        raise ValueError("max_visible_root_criteria must be >= 1.")

    forced_latent = _normalize_root_id_list(forest, forced_latent_root_ids)
    roots = [tree.root for tree in forest.trees]
    selectable_indices = [
        index for index, root in enumerate(roots) if root.id not in forced_latent
    ]
    if not selectable_indices:
        raise ValueError("At least one root criterion must remain visible at t=0.")

    visible_count = min(max_visible_root_criteria, len(selectable_indices))
    visible_index_set: set[int]
    if visible_count == len(selectable_indices):
        visible_index_set = set(selectable_indices)
    else:
        rng = random.Random(seed + 7919)
        visible_index_set = set(rng.sample(selectable_indices, k=visible_count))

    visible: list[dict[str, str]] = []
    latent: list[dict[str, str]] = []
    for index, root in enumerate(roots):
        payload = {"id": root.id, "text": root.text}
        if root.id in forced_latent or index not in visible_index_set:
            latent.append(payload)
        else:
            visible.append(payload)

    if not visible:
        raise ValueError("At least one root criterion must remain visible at t=0.")
    return visible, latent


def constrain_selected_root_ids(
    selected_root_ids: list[str],
    *,
    visible_root_ids: list[str],
    max_selected_root_criteria: int = _MAX_VISIBLE_ROOT_CRITERIA,
) -> list[str]:
    """Restrict stage-4 selections to the visible root subset and cap discovery breadth."""

    if not isinstance(selected_root_ids, list):
        raise TypeError("selected_root_ids must be a list of strings.")
    if not isinstance(visible_root_ids, list) or not visible_root_ids:
        raise ValueError("visible_root_ids must be a non-empty list of strings.")
    if not isinstance(max_selected_root_criteria, int) or max_selected_root_criteria < 1:
        raise ValueError("max_selected_root_criteria must be >= 1.")

    allowed_order: list[str] = []
    allowed_set: set[str] = set()
    for root_id in visible_root_ids:
        if not isinstance(root_id, str) or not root_id.strip():
            raise ValueError("visible_root_ids must contain non-empty strings.")
        normalized = root_id.strip()
        if normalized in allowed_set:
            continue
        allowed_set.add(normalized)
        allowed_order.append(normalized)

    requested: set[str] = set()
    for root_id in selected_root_ids:
        if not isinstance(root_id, str) or not root_id.strip():
            raise ValueError("selected_root_ids must contain non-empty strings.")
        normalized = root_id.strip()
        if normalized in allowed_set:
            requested.add(normalized)

    constrained = [root_id for root_id in allowed_order if root_id in requested]
    if not constrained:
        raise ValueError(
            "Stage-4 selected criteria must include at least one visible root criterion."
    )
    return constrained[:max_selected_root_criteria]


def ground_selected_root_ids_to_initial_request(
    forest: IntentForest,
    *,
    initial_request: str,
    selected_root_ids: list[str],
    max_selected_root_criteria: int = _MAX_VISIBLE_ROOT_CRITERIA,
) -> list[str]:
    """Align t=0 discovered roots with what the generated request actually states."""

    if not isinstance(forest, IntentForest):
        raise TypeError("forest must be an IntentForest.")
    if not isinstance(initial_request, str) or not initial_request.strip():
        raise ValueError("initial_request must be a non-empty string.")
    if not isinstance(max_selected_root_criteria, int) or max_selected_root_criteria < 1:
        raise ValueError("max_selected_root_criteria must be >= 1.")

    fallback_ids = _normalize_root_id_list(forest, selected_root_ids)
    grounded_ids = [
        tree.root.id
        for tree in forest.trees
        if _root_grounding_score(tree, initial_request) >= 2
    ]
    if grounded_ids:
        return grounded_ids[:max_selected_root_criteria]
    return fallback_ids[:max_selected_root_criteria]


def select_latent_root_criteria(
    forest: IntentForest, latent_root_ids: list[str] | None
) -> list[dict[str, str]]:
    """Resolve optional latent root IDs into prompt payload objects."""

    roots = {tree.root.id: tree.root for tree in forest.trees}
    normalized_latent_ids = _normalize_root_id_list(forest, latent_root_ids)
    latent: list[dict[str, str]] = []
    for normalized in normalized_latent_ids:
        root = roots[normalized]
        latent.append({"id": root.id, "text": root.text})
    return latent


def apply_initial_discovery(
    forest: IntentForest, selected_root_ids: list[str]
) -> list[str]:
    """Mark selected root criteria as discovered at t=0."""

    if not isinstance(selected_root_ids, list):
        raise TypeError("selected_root_ids must be a list of strings.")

    roots = {tree.root.id: tree.root for tree in forest.trees}
    applied: list[str] = []
    seen: set[str] = set()
    for node_id in selected_root_ids:
        if not isinstance(node_id, str):
            raise TypeError("selected_root_ids must contain strings.")
        normalized = node_id.strip()
        if not normalized:
            raise ValueError("selected_root_ids must not contain empty strings.")
        if normalized in seen:
            continue
        seen.add(normalized)
        root = roots.get(normalized)
        if root is None:
            raise ValueError(
                f"Selected initial request criterion {normalized!r} is not a root node."
            )
        root.state = NodeState.DISCOVERED
        root.initially_discovered = True
        applied.append(normalized)
    return applied


def build_conversation_state(
    artifact: ArtifactSpec,
    forest: IntentForest,
    *,
    initial_request: str,
    initial_discovered_root_ids: list[str],
    seed: int,
) -> ConversationState:
    """Create simulator initialization state after stage-4 output is applied."""

    if not isinstance(initial_request, str) or not initial_request.strip():
        raise ValueError("initial_request must be a non-empty string.")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")

    return ConversationState(
        artifact=artifact,
        intent_forest=forest,
        messages=[ConversationMessage(role="user", content=initial_request.strip())],
        turn_index=0,
        random_seed=seed,
        initial_discovered_node_ids=list(initial_discovered_root_ids),
    )


def parse_selected_root_ids(stage4_payload: dict[str, Any]) -> list[str]:
    """Extract selected root IDs from parsed stage-4 output."""

    if not isinstance(stage4_payload, dict):
        raise TypeError("stage4_payload must be a mapping.")
    raw_ids = stage4_payload.get("criteria_selected_for_request")
    if not isinstance(raw_ids, list):
        raise ValueError(
            "criteria_selected_for_request must be a list in stage-4 parsed output."
        )
    selected: list[str] = []
    for item in raw_ids:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("criteria_selected_for_request must contain non-empty strings.")
        selected.append(item.strip())
    return selected


def _normalize_root_id_list(
    forest: IntentForest, root_ids: list[str] | None
) -> list[str]:
    if root_ids is None:
        return []
    if not isinstance(root_ids, list):
        raise TypeError("root_ids must be a list of strings or None.")

    roots = {tree.root.id: tree.root for tree in forest.trees}
    normalized_ids: list[str] = []
    seen: set[str] = set()
    for root_id in root_ids:
        if not isinstance(root_id, str):
            raise TypeError("root_ids must contain strings.")
        normalized = root_id.strip()
        if not normalized:
            raise ValueError("root_ids must not contain empty strings.")
        if normalized in seen:
            continue
        seen.add(normalized)
        if normalized not in roots:
            raise ValueError(f"root id {normalized!r} is not in the root set.")
        normalized_ids.append(normalized)
    return normalized_ids


def _root_grounding_score(tree: IntentTree, request_text: str) -> int:
    return max(
        grounding_score(request_text, [node.text])
        for node in tree.iter_depth_first()
    )
