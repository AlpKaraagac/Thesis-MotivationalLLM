"""Deterministic heuristic state updates from evaluator outputs."""

from __future__ import annotations

from dataclasses import dataclass

from discoverllm.config import DEFAULT_CONFIG
from discoverllm.types import ConversationState, EvaluationResult, NodeState

_STATE_RANK = {
    NodeState.UNDISCOVERED: 0,
    NodeState.EMERGING: 1,
    NodeState.DISCOVERED: 2,
}


@dataclass(frozen=True, slots=True)
class StateUpdateResult:
    """Audit-friendly summary of one heuristic simulator-state update."""

    before_discovered_node_ids: frozenset[str]
    after_discovered_node_ids: frozenset[str]
    newly_discovered_node_ids: tuple[str, ...]
    directly_engaged_node_ids: tuple[str, ...]
    newly_satisfied_node_ids: tuple[str, ...]
    tangentially_advanced_node_ids: tuple[str, ...]

    @property
    def discovery_delta(self) -> int:
        return len(self.after_discovered_node_ids) - len(self.before_discovered_node_ids)


def apply_evaluation_result(
    conversation_state: ConversationState,
    evaluation: EvaluationResult,
    *,
    tangential_probability: float = DEFAULT_CONFIG.tangential_probability,
) -> StateUpdateResult:
    """Apply the paper's direct and tangential discovery rules in place."""

    if not isinstance(conversation_state, ConversationState):
        raise TypeError("conversation_state must be a ConversationState.")
    if not isinstance(evaluation, EvaluationResult):
        raise TypeError("evaluation must be an EvaluationResult.")
    if not isinstance(tangential_probability, (int, float)):
        raise TypeError("tangential_probability must be numeric.")
    tangential_probability = float(tangential_probability)
    if not 0.0 <= tangential_probability <= 1.0:
        raise ValueError("tangential_probability must be in [0, 1].")

    before_snapshot = {
        node.id: (node.state, node.satisfied) for node in conversation_state.intent_forest.iter_depth_first()
    }
    before_discovered = frozenset(conversation_state.discovered_node_ids())

    directly_engaged: list[str] = []
    newly_satisfied: list[str] = []
    tangentially_advanced: list[str] = []

    for item in evaluation.evaluations:
        node = conversation_state.intent_forest.get_node(item.node_id)
        if node is None:
            raise ValueError(f"Evaluator referenced unknown node id {item.node_id!r}.")

        if item.is_satisfied_or_probed:
            directly_engaged.append(node.id)
            node.state = NodeState.DISCOVERED
            if evaluation.evaluation_type == "satisfaction" and not node.satisfied:
                node.satisfied = True
                newly_satisfied.append(node.id)
            continue

        near_miss_count = len(item.near_miss)
        if near_miss_count == 0:
            continue

        if node.threshold is None:
            raise ValueError(
                f"Node {node.id!r} is missing threshold required for tangential updates."
            )

        node.tangential_exposure_count += near_miss_count
        score = tangential_probability * near_miss_count
        next_state = _advance_from_tangential(node.state, score=score, threshold=node.threshold)
        if next_state != node.state:
            node.state = next_state
            tangentially_advanced.append(node.id)

    _assert_monotonicity(before_snapshot, conversation_state)
    after_discovered = frozenset(conversation_state.discovered_node_ids())
    newly_discovered = tuple(
        node.id
        for node in conversation_state.intent_forest.iter_depth_first()
        if node.id in after_discovered and node.id not in before_discovered
    )
    return StateUpdateResult(
        before_discovered_node_ids=before_discovered,
        after_discovered_node_ids=after_discovered,
        newly_discovered_node_ids=newly_discovered,
        directly_engaged_node_ids=tuple(directly_engaged),
        newly_satisfied_node_ids=tuple(newly_satisfied),
        tangentially_advanced_node_ids=tuple(tangentially_advanced),
    )


def _advance_from_tangential(
    current_state: NodeState,
    *,
    score: float,
    threshold: float,
) -> NodeState:
    if score <= threshold:
        return current_state
    if current_state == NodeState.UNDISCOVERED:
        return NodeState.EMERGING
    if current_state == NodeState.EMERGING:
        return NodeState.DISCOVERED
    return NodeState.DISCOVERED


def _assert_monotonicity(
    before_snapshot: dict[str, tuple[NodeState, bool]],
    conversation_state: ConversationState,
) -> None:
    for node in conversation_state.intent_forest.iter_depth_first():
        before_state, before_satisfied = before_snapshot[node.id]
        if _STATE_RANK[node.state] < _STATE_RANK[before_state]:
            raise AssertionError(
                f"Node {node.id!r} regressed from {before_state.value!r} to {node.state.value!r}."
            )
        if before_satisfied and not node.satisfied:
            raise AssertionError(f"Node {node.id!r} satisfaction regressed.")
