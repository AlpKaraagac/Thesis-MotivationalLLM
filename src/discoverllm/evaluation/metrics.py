"""Simulator-side evaluation metrics for discovery, satisfaction, and interactivity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from discoverllm.types import ConversationState, IntentForest, IntentNode, NodeState


@dataclass(frozen=True, slots=True)
class LeafSatisfactionEvaluation:
    """Canonical leaf-level artifact satisfaction judgment."""

    requirement_id: str
    requirement_text: str
    reasoning: str
    score: int

    def __post_init__(self) -> None:
        if not isinstance(self.requirement_id, str) or not self.requirement_id.strip():
            raise ValueError("requirement_id must be a non-empty string.")
        if not isinstance(self.requirement_text, str) or not self.requirement_text.strip():
            raise ValueError("requirement_text must be a non-empty string.")
        if not isinstance(self.reasoning, str) or not self.reasoning.strip():
            raise ValueError("reasoning must be a non-empty string.")
        if not isinstance(self.score, int):
            raise TypeError("score must be an int.")
        if not 1 <= self.score <= 5:
            raise ValueError("score must be in [1, 5].")

    @property
    def is_satisfied(self) -> bool:
        return self.score >= 4


@dataclass(frozen=True, slots=True)
class IntentDiscoveryScoreResult:
    """Audit-friendly normalized end-of-conversation discovery score."""

    score: float
    discovered_mass: float
    evaluable_node_count: int
    included_node_ids: tuple[str, ...]
    excluded_node_ids: tuple[str, ...]
    discovered_node_ids: tuple[str, ...]
    emerging_node_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.score, (int, float)):
            raise TypeError("score must be numeric.")
        if not 0.0 <= float(self.score) <= 1.0:
            raise ValueError("score must be in [0, 1].")
        if not isinstance(self.discovered_mass, (int, float)):
            raise TypeError("discovered_mass must be numeric.")
        if self.discovered_mass < 0:
            raise ValueError("discovered_mass must be non-negative.")
        if not isinstance(self.evaluable_node_count, int):
            raise TypeError("evaluable_node_count must be an int.")
        if self.evaluable_node_count < 0:
            raise ValueError("evaluable_node_count must be non-negative.")


@dataclass(frozen=True, slots=True)
class IntentSatisfactionScoreResult:
    """Leaf-level normalized artifact satisfaction score."""

    score: float
    satisfied_count: int
    evaluable_requirement_count: int
    included_requirement_ids: tuple[str, ...]
    excluded_requirement_ids: tuple[str, ...]
    satisfied_requirement_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.score, (int, float)):
            raise TypeError("score must be numeric.")
        if not 0.0 <= float(self.score) <= 1.0:
            raise ValueError("score must be in [0, 1].")
        if not isinstance(self.satisfied_count, int):
            raise TypeError("satisfied_count must be an int.")
        if self.satisfied_count < 0:
            raise ValueError("satisfied_count must be non-negative.")
        if not isinstance(self.evaluable_requirement_count, int):
            raise TypeError("evaluable_requirement_count must be an int.")
        if self.evaluable_requirement_count < 0:
            raise ValueError("evaluable_requirement_count must be non-negative.")


def iter_leaf_nodes(intent_forest: IntentForest) -> Iterable[IntentNode]:
    """Yield leaf nodes in depth-first tree order."""

    if not isinstance(intent_forest, IntentForest):
        raise TypeError("intent_forest must be an IntentForest.")
    for node in intent_forest.iter_depth_first():
        if not node.children:
            yield node


def leaf_requirements_payload(intent_forest: IntentForest) -> list[dict[str, str]]:
    """Serialize leaf intent requirements for the artifact judge prompt."""

    return [{"id": node.id, "text": node.text} for node in iter_leaf_nodes(intent_forest)]


def compute_intent_discovery_score(
    conversation_state: ConversationState,
    *,
    exclude_initially_discovered: bool = True,
    excluded_node_ids: Sequence[str] = (),
    emerging_credit: float = 0.5,
) -> IntentDiscoveryScoreResult:
    """Compute the normalized discovery score from the final simulator state."""

    if not isinstance(conversation_state, ConversationState):
        raise TypeError("conversation_state must be a ConversationState.")
    if not isinstance(exclude_initially_discovered, bool):
        raise TypeError("exclude_initially_discovered must be a bool.")
    if not isinstance(emerging_credit, (int, float)):
        raise TypeError("emerging_credit must be numeric.")
    emerging_credit = float(emerging_credit)
    if not 0.0 <= emerging_credit <= 1.0:
        raise ValueError("emerging_credit must be in [0, 1].")

    excluded = {node_id.strip() for node_id in excluded_node_ids if isinstance(node_id, str) and node_id.strip()}
    if exclude_initially_discovered:
        excluded.update(conversation_state.initial_discovered_node_ids)

    included_node_ids: list[str] = []
    discovered_node_ids: list[str] = []
    emerging_node_ids: list[str] = []
    discovered_mass = 0.0

    for node in conversation_state.intent_forest.iter_depth_first():
        if node.id in excluded:
            continue
        included_node_ids.append(node.id)
        if node.state == NodeState.DISCOVERED:
            discovered_node_ids.append(node.id)
            discovered_mass += 1.0
        elif node.state == NodeState.EMERGING:
            emerging_node_ids.append(node.id)
            discovered_mass += emerging_credit

    denominator = len(included_node_ids)
    score = 0.0 if denominator == 0 else discovered_mass / float(denominator)
    return IntentDiscoveryScoreResult(
        score=score,
        discovered_mass=discovered_mass,
        evaluable_node_count=denominator,
        included_node_ids=tuple(included_node_ids),
        excluded_node_ids=tuple(sorted(excluded)),
        discovered_node_ids=tuple(discovered_node_ids),
        emerging_node_ids=tuple(emerging_node_ids),
    )


def compute_intent_satisfaction_score(
    leaf_evaluations: Sequence[LeafSatisfactionEvaluation],
    *,
    excluded_requirement_ids: Sequence[str] = (),
    satisfied_threshold: int = 4,
) -> IntentSatisfactionScoreResult:
    """Compute normalized leaf-level satisfaction from judged artifact scores."""

    if not isinstance(leaf_evaluations, Sequence):
        raise TypeError("leaf_evaluations must be a sequence.")
    if not isinstance(satisfied_threshold, int):
        raise TypeError("satisfied_threshold must be an int.")
    if not 1 <= satisfied_threshold <= 5:
        raise ValueError("satisfied_threshold must be in [1, 5].")

    excluded = {
        requirement_id.strip()
        for requirement_id in excluded_requirement_ids
        if isinstance(requirement_id, str) and requirement_id.strip()
    }
    included_requirement_ids: list[str] = []
    satisfied_requirement_ids: list[str] = []

    for evaluation in leaf_evaluations:
        if not isinstance(evaluation, LeafSatisfactionEvaluation):
            raise TypeError(
                "leaf_evaluations must contain only LeafSatisfactionEvaluation entries."
            )
        if evaluation.requirement_id in excluded:
            continue
        included_requirement_ids.append(evaluation.requirement_id)
        if evaluation.score >= satisfied_threshold:
            satisfied_requirement_ids.append(evaluation.requirement_id)

    denominator = len(included_requirement_ids)
    satisfied_count = len(satisfied_requirement_ids)
    score = 0.0 if denominator == 0 else satisfied_count / float(denominator)
    return IntentSatisfactionScoreResult(
        score=score,
        satisfied_count=satisfied_count,
        evaluable_requirement_count=denominator,
        included_requirement_ids=tuple(included_requirement_ids),
        excluded_requirement_ids=tuple(sorted(excluded)),
        satisfied_requirement_ids=tuple(satisfied_requirement_ids),
    )


def normalize_interactivity_score(raw_score: float) -> float:
    """Normalize a paper judge score from [1, 3] into [0, 1]."""

    if not isinstance(raw_score, (int, float)):
        raise TypeError("raw_score must be numeric.")
    raw_score = float(raw_score)
    if not 1.0 <= raw_score <= 3.0:
        raise ValueError("raw_score must be in [1, 3].")
    return (raw_score - 1.0) / 2.0


def benchmark_excluded_node_ids(
    states: Sequence[ConversationState],
) -> tuple[str, ...]:
    """Return node IDs that are discovered by all states or by none."""

    if not isinstance(states, Sequence) or not states:
        raise ValueError("states must be a non-empty sequence of ConversationState.")
    normalized_states: list[ConversationState] = []
    universe: list[str] | None = None
    for state in states:
        if not isinstance(state, ConversationState):
            raise TypeError("states must contain only ConversationState entries.")
        node_ids = [node.id for node in state.intent_forest.iter_depth_first()]
        if universe is None:
            universe = node_ids
        elif node_ids != universe:
            raise ValueError("All states must share the same intent forest node order.")
        normalized_states.append(state)
    assert universe is not None

    excluded: list[str] = []
    for node_id in universe:
        final_flags = []
        for state in normalized_states:
            node = state.intent_forest.get_node(node_id)
            assert node is not None
            final_flags.append(node.state == NodeState.DISCOVERED)
        if all(final_flags) or not any(final_flags):
            excluded.append(node_id)
    return tuple(excluded)


def benchmark_excluded_requirement_ids(
    evaluation_sets: Sequence[Sequence[LeafSatisfactionEvaluation]],
    *,
    satisfied_threshold: int = 4,
) -> tuple[str, ...]:
    """Return leaf IDs satisfied by every set or by no set."""

    if not isinstance(evaluation_sets, Sequence) or not evaluation_sets:
        raise ValueError(
            "evaluation_sets must be a non-empty sequence of evaluation sequences."
        )
    requirement_order: list[str] | None = None
    normalized_sets: list[list[LeafSatisfactionEvaluation]] = []
    for evaluations in evaluation_sets:
        if not isinstance(evaluations, Sequence):
            raise TypeError("evaluation_sets must contain sequences.")
        current: list[LeafSatisfactionEvaluation] = []
        current_ids: list[str] = []
        for item in evaluations:
            if not isinstance(item, LeafSatisfactionEvaluation):
                raise TypeError(
                    "evaluation_sets must contain LeafSatisfactionEvaluation entries."
                )
            current.append(item)
            current_ids.append(item.requirement_id)
        if requirement_order is None:
            requirement_order = current_ids
        elif current_ids != requirement_order:
            raise ValueError("All evaluation sets must share the same requirement order.")
        normalized_sets.append(current)
    assert requirement_order is not None

    excluded: list[str] = []
    for index, requirement_id in enumerate(requirement_order):
        satisfied_flags = [
            evaluations[index].score >= satisfied_threshold for evaluations in normalized_sets
        ]
        if all(satisfied_flags) or not any(satisfied_flags):
            excluded.append(requirement_id)
    return tuple(excluded)


def coerce_leaf_satisfaction_evaluations(
    payload: Sequence[Mapping[str, Any]],
    *,
    requirement_text_by_id: Mapping[str, str],
) -> tuple[LeafSatisfactionEvaluation, ...]:
    """Convert parsed judge payload rows into canonical typed evaluations."""

    if not isinstance(payload, Sequence):
        raise TypeError("payload must be a sequence.")
    if not isinstance(requirement_text_by_id, Mapping):
        raise TypeError("requirement_text_by_id must be a mapping.")

    seen: set[str] = set()
    normalized: list[LeafSatisfactionEvaluation] = []
    for item in payload:
        if not isinstance(item, Mapping):
            raise TypeError("payload must contain mapping entries.")
        requirement_id = item.get("requirement_id")
        if not isinstance(requirement_id, str) or not requirement_id.strip():
            raise ValueError("requirement_id must be a non-empty string.")
        normalized_id = requirement_id.strip()
        if normalized_id in seen:
            raise ValueError(f"Duplicate leaf evaluation for requirement {normalized_id!r}.")
        seen.add(normalized_id)
        if normalized_id not in requirement_text_by_id:
            raise ValueError(
                f"Leaf evaluation referenced non-leaf or unknown requirement {normalized_id!r}."
            )
        normalized.append(
            LeafSatisfactionEvaluation(
                requirement_id=normalized_id,
                requirement_text=str(requirement_text_by_id[normalized_id]),
                reasoning=str(item.get("reasoning", "")),
                score=int(item.get("score", 0)),
            )
        )

    missing = [requirement_id for requirement_id in requirement_text_by_id if requirement_id not in seen]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Leaf evaluations missing required ids: {joined}")

    return tuple(
        next(item for item in normalized if item.requirement_id == requirement_id)
        for requirement_id in requirement_text_by_id
    )
