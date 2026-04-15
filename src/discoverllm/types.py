"""Typed schemas for the DiscoverLLM simulator and generation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import re
from typing import Any, Iterable, Literal

NODE_ID_PATTERN = re.compile(r"^[1-9]\d*(\.[1-9]\d*)*$")


def _validate_non_empty(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    return normalized


def _validate_node_id(node_id: str) -> str:
    normalized = _validate_non_empty(node_id, "node_id")
    if not NODE_ID_PATTERN.match(normalized):
        raise ValueError(
            f"Invalid node_id {normalized!r}. Expected dot-separated positive integers."
        )
    return normalized


def _coerce_node_state(state: NodeState | str) -> NodeState:
    if isinstance(state, NodeState):
        return state
    if isinstance(state, str):
        try:
            return NodeState(state)
        except ValueError as exc:
            raise ValueError(f"Invalid node state {state!r}.") from exc
    raise TypeError("state must be a NodeState or string value.")


def _validate_threshold(value: float | None) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise TypeError("threshold must be numeric or None.")
    as_float = float(value)
    if not 0.0 <= as_float <= 1.0:
        raise ValueError("threshold must be in [0, 1].")
    return as_float


class NodeState(str, Enum):
    """Discovery state for one intent node."""

    UNDISCOVERED = "undiscovered"
    EMERGING = "emerging"
    DISCOVERED = "discovered"


@dataclass(slots=True)
class ArtifactSpec:
    """Source artifact used to construct the intent hierarchy."""

    artifact_id: str
    artifact_type: str
    artifact_content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.artifact_id = _validate_non_empty(self.artifact_id, "artifact_id")
        self.artifact_type = _validate_non_empty(self.artifact_type, "artifact_type")
        self.artifact_content = _validate_non_empty(
            self.artifact_content, "artifact_content"
        )
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary.")


@dataclass(slots=True)
class IntentNode:
    """One node in the paper's hierarchical intent representation."""

    id: str
    text: str
    children: list[IntentNode] = field(default_factory=list)
    state: NodeState | str = NodeState.UNDISCOVERED
    satisfied: bool = False
    threshold: float | None = None
    tangential_exposure_count: int = 0
    initially_discovered: bool = False

    def __post_init__(self) -> None:
        self.id = _validate_node_id(self.id)
        self.text = _validate_non_empty(self.text, "text")
        self.state = _coerce_node_state(self.state)
        self.threshold = _validate_threshold(self.threshold)
        if not isinstance(self.satisfied, bool):
            raise TypeError("satisfied must be a bool.")
        if not isinstance(self.initially_discovered, bool):
            raise TypeError("initially_discovered must be a bool.")
        if not isinstance(self.tangential_exposure_count, int):
            raise TypeError("tangential_exposure_count must be an int.")
        if self.tangential_exposure_count < 0:
            raise ValueError("tangential_exposure_count must be non-negative.")
        if self.satisfied and self.state != NodeState.DISCOVERED:
            raise ValueError("A satisfied node must be discovered.")
        if not isinstance(self.children, list):
            raise TypeError("children must be a list of IntentNode.")

        child_ids: set[str] = set()
        parent_depth = self.depth
        expected_prefix = f"{self.id}."
        for child in self.children:
            if not isinstance(child, IntentNode):
                raise TypeError("children must contain only IntentNode values.")
            if child.id in child_ids:
                raise ValueError(f"Duplicate child id {child.id!r} under parent {self.id!r}.")
            child_ids.add(child.id)
            if not child.id.startswith(expected_prefix):
                raise ValueError(
                    f"Child id {child.id!r} must start with parent prefix {expected_prefix!r}."
                )
            if child.depth != parent_depth + 1:
                raise ValueError(
                    f"Child id {child.id!r} is not an immediate child of {self.id!r}."
                )

    @property
    def depth(self) -> int:
        return self.id.count(".") + 1

    def iter_depth_first(self) -> Iterable[IntentNode]:
        yield self
        for child in self.children:
            yield from child.iter_depth_first()


@dataclass(slots=True)
class IntentTree:
    """One independent intent dimension rooted at a top-level criterion."""

    root: IntentNode

    def __post_init__(self) -> None:
        if not isinstance(self.root, IntentNode):
            raise TypeError("root must be an IntentNode.")
        if self.root.depth != 1:
            raise ValueError("Tree root id must be a top-level id such as '1'.")
        node_ids: set[str] = set()
        for node in self.root.iter_depth_first():
            if node.id in node_ids:
                raise ValueError(f"Duplicate node id {node.id!r} in tree.")
            node_ids.add(node.id)

    def iter_depth_first(self) -> Iterable[IntentNode]:
        return self.root.iter_depth_first()

    def all_nodes(self) -> list[IntentNode]:
        return list(self.iter_depth_first())

    def node_ids(self) -> list[str]:
        return [node.id for node in self.iter_depth_first()]


@dataclass(slots=True)
class IntentForest:
    """Ordered list of independent root trees."""

    trees: list[IntentTree]

    def __post_init__(self) -> None:
        if not isinstance(self.trees, list):
            raise TypeError("trees must be a list of IntentTree objects.")
        if not self.trees:
            raise ValueError("IntentForest must contain at least one tree.")
        root_ids: set[str] = set()
        for tree in self.trees:
            if not isinstance(tree, IntentTree):
                raise TypeError("trees must contain only IntentTree values.")
            root_id = tree.root.id
            if root_id in root_ids:
                raise ValueError(f"Duplicate root id {root_id!r} across trees.")
            root_ids.add(root_id)

    def iter_depth_first(self) -> Iterable[IntentNode]:
        for tree in self.trees:
            yield from tree.iter_depth_first()

    def all_nodes(self) -> list[IntentNode]:
        return list(self.iter_depth_first())

    def get_node(self, node_id: str) -> IntentNode | None:
        for node in self.iter_depth_first():
            if node.id == node_id:
                return node
        return None


@dataclass(slots=True)
class NodeEvaluation:
    """Per-node evaluator output inside one assistant-turn evaluation."""

    node_id: str
    node_text: str
    reasoning: str
    is_satisfied_or_probed: bool
    near_miss: list[str] = field(default_factory=list)
    children_evaluated: bool = False

    def __post_init__(self) -> None:
        self.node_id = _validate_node_id(self.node_id)
        self.node_text = _validate_non_empty(self.node_text, "node_text")
        self.reasoning = _validate_non_empty(self.reasoning, "reasoning")
        if not isinstance(self.is_satisfied_or_probed, bool):
            raise TypeError("is_satisfied_or_probed must be a bool.")
        if not isinstance(self.children_evaluated, bool):
            raise TypeError("children_evaluated must be a bool.")
        if not isinstance(self.near_miss, list):
            raise TypeError("near_miss must be a list of strings.")
        cleaned: list[str] = []
        for item in self.near_miss:
            cleaned.append(_validate_non_empty(item, "near_miss item"))
        if self.is_satisfied_or_probed and cleaned:
            raise ValueError("near_miss must be empty when node is directly engaged.")
        self.near_miss = cleaned


ClassificationLabel = Literal["artifact", "dialog_act"]
EvaluationType = Literal["satisfaction", "probing"]


@dataclass(slots=True)
class EvaluationResult:
    """Structured evaluator output from one assistant response."""

    classification_reasoning: str
    classification_label: ClassificationLabel
    evaluation_type: EvaluationType
    evaluations: list[NodeEvaluation]

    def __post_init__(self) -> None:
        self.classification_reasoning = _validate_non_empty(
            self.classification_reasoning, "classification_reasoning"
        )
        if self.classification_label not in ("artifact", "dialog_act"):
            raise ValueError("classification_label must be 'artifact' or 'dialog_act'.")
        if self.evaluation_type not in ("satisfaction", "probing"):
            raise ValueError("evaluation_type must be 'satisfaction' or 'probing'.")
        if self.classification_label == "artifact" and self.evaluation_type != "satisfaction":
            raise ValueError("artifact classification requires satisfaction evaluation_type.")
        if self.classification_label == "dialog_act" and self.evaluation_type != "probing":
            raise ValueError("dialog_act classification requires probing evaluation_type.")
        if not isinstance(self.evaluations, list):
            raise TypeError("evaluations must be a list of NodeEvaluation entries.")

        seen: set[str] = set()
        for item in self.evaluations:
            if not isinstance(item, NodeEvaluation):
                raise TypeError("evaluations must contain only NodeEvaluation entries.")
            if item.node_id in seen:
                raise ValueError(f"Duplicate evaluation for node {item.node_id!r}.")
            seen.add(item.node_id)


@dataclass(slots=True)
class TurnReward:
    """Per-turn reward trace for one assistant candidate."""

    discovery_delta: int
    efficiency_penalty: float
    total_reward: float | None = None
    response_tokens: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.discovery_delta, int):
            raise TypeError("discovery_delta must be an int.")
        if self.discovery_delta < 0:
            raise ValueError("discovery_delta must be non-negative.")
        if not isinstance(self.efficiency_penalty, (int, float)):
            raise TypeError("efficiency_penalty must be numeric.")
        self.efficiency_penalty = float(self.efficiency_penalty)
        if self.efficiency_penalty > 0:
            raise ValueError("efficiency_penalty must be <= 0.")
        if self.efficiency_penalty < -1:
            raise ValueError("efficiency_penalty must be >= -1.")
        if self.response_tokens is not None:
            if not isinstance(self.response_tokens, int):
                raise TypeError("response_tokens must be an int or None.")
            if self.response_tokens < 0:
                raise ValueError("response_tokens must be non-negative.")

        computed = float(self.discovery_delta) + self.efficiency_penalty
        if self.total_reward is None:
            self.total_reward = computed
        else:
            if not isinstance(self.total_reward, (int, float)):
                raise TypeError("total_reward must be numeric.")
            self.total_reward = float(self.total_reward)
            if not math.isclose(self.total_reward, computed, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(
                    "total_reward must equal discovery_delta + efficiency_penalty."
                )


ConversationRole = Literal["system", "user", "assistant"]


@dataclass(slots=True)
class ConversationMessage:
    """One chat message in the simulated interaction history."""

    role: ConversationRole
    content: str

    def __post_init__(self) -> None:
        if self.role not in ("system", "user", "assistant"):
            raise ValueError("role must be one of: system, user, assistant.")
        self.content = _validate_non_empty(self.content, "content")


@dataclass(slots=True)
class ConversationState:
    """Canonical simulator state for one in-progress conversation."""

    artifact: ArtifactSpec
    intent_forest: IntentForest
    messages: list[ConversationMessage] = field(default_factory=list)
    turn_index: int = 0
    random_seed: int | None = None
    initial_discovered_node_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, ArtifactSpec):
            raise TypeError("artifact must be an ArtifactSpec.")
        if not isinstance(self.intent_forest, IntentForest):
            raise TypeError("intent_forest must be an IntentForest.")
        if not isinstance(self.messages, list):
            raise TypeError("messages must be a list of ConversationMessage.")
        for message in self.messages:
            if not isinstance(message, ConversationMessage):
                raise TypeError("messages must contain only ConversationMessage entries.")
        if not isinstance(self.turn_index, int):
            raise TypeError("turn_index must be an int.")
        if self.turn_index < 0:
            raise ValueError("turn_index must be non-negative.")
        if self.random_seed is not None:
            if not isinstance(self.random_seed, int):
                raise TypeError("random_seed must be an int or None.")
            if self.random_seed < 0:
                raise ValueError("random_seed must be non-negative.")
        if not isinstance(self.initial_discovered_node_ids, list):
            raise TypeError("initial_discovered_node_ids must be a list of node ids.")

        valid_ids = {node.id for node in self.intent_forest.iter_depth_first()}
        seen: set[str] = set()
        for node_id in self.initial_discovered_node_ids:
            normalized = _validate_node_id(node_id)
            if normalized not in valid_ids:
                raise ValueError(
                    f"initial_discovered_node_id {normalized!r} is not in intent_forest."
                )
            if normalized in seen:
                raise ValueError(
                    f"Duplicate initial_discovered_node_id {normalized!r} in state."
                )
            seen.add(normalized)

    def discovered_node_ids(self) -> set[str]:
        return {
            node.id
            for node in self.intent_forest.iter_depth_first()
            if node.state == NodeState.DISCOVERED
        }


@dataclass(slots=True)
class TrajectoryTurn:
    """One turn record in a chosen conversation trajectory."""

    turn_index: int
    user_message: str
    chosen_assistant_message: str
    rejected_assistant_message: str | None = None
    chosen_reward: TurnReward | None = None
    rejected_reward: TurnReward | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.turn_index, int):
            raise TypeError("turn_index must be an int.")
        if self.turn_index < 0:
            raise ValueError("turn_index must be non-negative.")
        self.user_message = _validate_non_empty(self.user_message, "user_message")
        self.chosen_assistant_message = _validate_non_empty(
            self.chosen_assistant_message, "chosen_assistant_message"
        )
        if self.rejected_assistant_message is not None:
            self.rejected_assistant_message = _validate_non_empty(
                self.rejected_assistant_message, "rejected_assistant_message"
            )
        if self.chosen_reward is not None and not isinstance(self.chosen_reward, TurnReward):
            raise TypeError("chosen_reward must be a TurnReward or None.")
        if self.rejected_reward is not None and not isinstance(
            self.rejected_reward, TurnReward
        ):
            raise TypeError("rejected_reward must be a TurnReward or None.")


@dataclass(slots=True)
class TrajectoryExample:
    """Structured chosen-branch conversation and metadata."""

    artifact_id: str
    turns: list[TrajectoryTurn]
    messages: list[ConversationMessage] = field(default_factory=list)
    final_state: ConversationState | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        self.artifact_id = _validate_non_empty(self.artifact_id, "artifact_id")
        if not isinstance(self.turns, list):
            raise TypeError("turns must be a list of TrajectoryTurn.")
        if not isinstance(self.messages, list):
            raise TypeError("messages must be a list of ConversationMessage.")
        seen_turns: set[int] = set()
        for turn in self.turns:
            if not isinstance(turn, TrajectoryTurn):
                raise TypeError("turns must contain only TrajectoryTurn entries.")
            if turn.turn_index in seen_turns:
                raise ValueError(f"Duplicate turn index {turn.turn_index} in trajectory.")
            seen_turns.add(turn.turn_index)
        for message in self.messages:
            if not isinstance(message, ConversationMessage):
                raise TypeError("messages must contain only ConversationMessage entries.")
        if self.final_state is not None and not isinstance(self.final_state, ConversationState):
            raise TypeError("final_state must be a ConversationState or None.")
        if self.seed is not None:
            if not isinstance(self.seed, int):
                raise TypeError("seed must be an int or None.")
            if self.seed < 0:
                raise ValueError("seed must be non-negative.")


@dataclass(slots=True)
class PreferenceExample:
    """Per-turn chosen/rejected example for DPO-style preference data."""

    artifact_id: str
    turn_index: int
    prompt_messages: list[ConversationMessage]
    chosen_response: str
    rejected_response: str
    chosen_reward: TurnReward
    rejected_reward: TurnReward
    reward_delta: float | None = None

    def __post_init__(self) -> None:
        self.artifact_id = _validate_non_empty(self.artifact_id, "artifact_id")
        if not isinstance(self.turn_index, int):
            raise TypeError("turn_index must be an int.")
        if self.turn_index < 0:
            raise ValueError("turn_index must be non-negative.")
        if not isinstance(self.prompt_messages, list):
            raise TypeError("prompt_messages must be a list of ConversationMessage.")
        for message in self.prompt_messages:
            if not isinstance(message, ConversationMessage):
                raise TypeError("prompt_messages must contain ConversationMessage entries.")
        self.chosen_response = _validate_non_empty(self.chosen_response, "chosen_response")
        self.rejected_response = _validate_non_empty(
            self.rejected_response, "rejected_response"
        )
        if not isinstance(self.chosen_reward, TurnReward):
            raise TypeError("chosen_reward must be a TurnReward.")
        if not isinstance(self.rejected_reward, TurnReward):
            raise TypeError("rejected_reward must be a TurnReward.")
        computed_delta = self.chosen_reward.total_reward - self.rejected_reward.total_reward
        if self.reward_delta is None:
            self.reward_delta = computed_delta
        else:
            if not isinstance(self.reward_delta, (int, float)):
                raise TypeError("reward_delta must be numeric.")
            self.reward_delta = float(self.reward_delta)
            if not math.isclose(
                self.reward_delta, computed_delta, rel_tol=1e-9, abs_tol=1e-9
            ):
                raise ValueError("reward_delta must equal chosen_reward - rejected_reward.")
