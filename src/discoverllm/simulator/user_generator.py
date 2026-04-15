"""Prompt-backed user response generation from simulator state."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from discoverllm.llm.base import LLMAdapter, LLMRequest, LLMResponse
from discoverllm.parsers.contracts import parse_user_response_generation_output
from discoverllm.prompts.loader import PromptLoader
from discoverllm.simulator.evaluator import select_active_subtree
from discoverllm.types import ConversationMessage, ConversationState, IntentNode, NodeState

AwarenessBucket = str


@dataclass(frozen=True, slots=True)
class GoalStatusItem:
    """Compact node reference exposed to the user generator."""

    node_id: str
    node_text: str

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, str) or not self.node_id.strip():
            raise ValueError("node_id must be a non-empty string.")
        if not isinstance(self.node_text, str) or not self.node_text.strip():
            raise ValueError("node_text must be a non-empty string.")


@dataclass(frozen=True, slots=True)
class GoalStatusView:
    """Paper-faithful compact goal-status representation for one turn."""

    achieved: tuple[GoalStatusItem, ...]
    pursuing_clear: tuple[GoalStatusItem, ...] = ()
    pursuing_fuzzy: tuple[GoalStatusItem, ...] = ()
    latent_goal: tuple[GoalStatusItem, ...] = ()

    def __post_init__(self) -> None:
        for name in ("achieved", "pursuing_clear", "pursuing_fuzzy", "latent_goal"):
            value = getattr(self, name)
            if not isinstance(value, tuple):
                raise TypeError(f"{name} must be a tuple of GoalStatusItem.")
            for item in value:
                if not isinstance(item, GoalStatusItem):
                    raise TypeError(f"{name} must contain only GoalStatusItem values.")

        populated = [
            bucket
            for bucket in ("pursuing_clear", "pursuing_fuzzy", "latent_goal")
            if getattr(self, bucket)
        ]
        if len(populated) > 1:
            joined = ", ".join(populated)
            raise ValueError(f"Only one awareness bucket may be populated; found: {joined}")

    @property
    def active_bucket(self) -> AwarenessBucket | None:
        for bucket in ("pursuing_clear", "pursuing_fuzzy", "latent_goal"):
            if getattr(self, bucket):
                return bucket
        return None

    def to_prompt_payload(self) -> dict[str, list[dict[str, str]]]:
        return {
            "achieved": _items_to_payload(self.achieved),
            "pursuing_clear": _items_to_payload(self.pursuing_clear),
            "pursuing_fuzzy": _items_to_payload(self.pursuing_fuzzy),
            "latent_goal": _items_to_payload(self.latent_goal),
        }


@dataclass(frozen=True, slots=True)
class UserResponseGenerationRun:
    """Audit-friendly record of one user-turn generation call."""

    goal_status: GoalStatusView
    goal_status_payload: dict[str, list[dict[str, str]]]
    chat_history_json: list[dict[str, str]]
    rendered_prompt: str
    llm_text_output: str
    llm_raw_output: str | dict[str, Any]
    parsed_output: dict[str, Any]
    emitted_message: ConversationMessage


class UserResponseGenerator:
    """Generate the next user turn from the simulator's awareness state."""

    def __init__(
        self,
        *,
        llm_adapter: LLMAdapter,
        prompt_loader: PromptLoader | None = None,
    ) -> None:
        if not isinstance(llm_adapter, LLMAdapter):
            raise TypeError("llm_adapter must implement LLMAdapter.")
        if prompt_loader is None:
            prompt_loader = PromptLoader()
        if not isinstance(prompt_loader, PromptLoader):
            raise TypeError("prompt_loader must be a PromptLoader.")

        self._llm = llm_adapter
        self._prompts = prompt_loader

    def generate_next_message(
        self,
        conversation_state: ConversationState,
    ) -> UserResponseGenerationRun:
        """Render prompt inputs from state and emit the next user message."""

        if not isinstance(conversation_state, ConversationState):
            raise TypeError("conversation_state must be a ConversationState.")

        goal_status = build_goal_status_view(conversation_state)
        if goal_status.active_bucket is None:
            raise ValueError("No unsatisfied intents remain for user response generation.")

        goal_status_payload = goal_status.to_prompt_payload()
        chat_history_payload = messages_to_payload(conversation_state.messages)
        rendered_prompt = self._prompts.render(
            "user_response_generation",
            {
                "chat_history_json": _json_block(chat_history_payload),
                "achieved_json": _json_block(goal_status_payload["achieved"]),
                "pursuing_clear_json": _json_block(goal_status_payload["pursuing_clear"]),
                "pursuing_fuzzy_json": _json_block(goal_status_payload["pursuing_fuzzy"]),
                "latent_goal_json": _json_block(goal_status_payload["latent_goal"]),
            },
            strict=True,
        )
        response = self._generate(
            prompt_text=rendered_prompt,
            active_bucket=goal_status.active_bucket,
        )
        parsed = parse_user_response_generation_output(response.text)
        emitted_message = ConversationMessage(role="user", content=str(parsed["user_message"]))
        return UserResponseGenerationRun(
            goal_status=goal_status,
            goal_status_payload=goal_status_payload,
            chat_history_json=chat_history_payload,
            rendered_prompt=rendered_prompt,
            llm_text_output=response.text,
            llm_raw_output=response.raw_output,
            parsed_output=parsed,
            emitted_message=emitted_message,
        )

    def _generate(self, *, prompt_text: str, active_bucket: AwarenessBucket) -> LLMResponse:
        request = LLMRequest(
            messages=[ConversationMessage(role="user", content=prompt_text)],
            temperature=0.0,
            metadata={
                "prompt_name": "user_response_generation",
                "active_bucket": active_bucket,
            },
        )
        return self._llm.generate(request)


def build_goal_status_view(conversation_state: ConversationState) -> GoalStatusView:
    """Construct the strict-priority goal-status view from simulator state."""

    if not isinstance(conversation_state, ConversationState):
        raise TypeError("conversation_state must be a ConversationState.")

    achieved = [
        goal_status_item_from_node(node)
        for node in conversation_state.intent_forest.iter_depth_first()
        if node.satisfied
    ]
    active_tree = select_active_subtree(conversation_state.intent_forest)
    if active_tree is None:
        return GoalStatusView(achieved=tuple(achieved))

    pursuing_clear = _deepest_frontier_items(active_tree.root, NodeState.DISCOVERED)
    pursuing_fuzzy = _topmost_frontier_items(active_tree.root, NodeState.EMERGING)
    latent_goal = _topmost_frontier_items(active_tree.root, NodeState.UNDISCOVERED)

    if pursuing_clear:
        return GoalStatusView(
            achieved=tuple(achieved),
            pursuing_clear=tuple(goal_status_item_from_node(node) for node in pursuing_clear),
        )
    if pursuing_fuzzy:
        return GoalStatusView(
            achieved=tuple(achieved),
            pursuing_fuzzy=tuple(goal_status_item_from_node(node) for node in pursuing_fuzzy),
        )
    if latent_goal:
        return GoalStatusView(
            achieved=tuple(achieved),
            latent_goal=tuple(goal_status_item_from_node(node) for node in latent_goal),
        )
    return GoalStatusView(achieved=tuple(achieved))


def goal_status_item_from_node(node: IntentNode) -> GoalStatusItem:
    """Project one intent node into the compact prompt-facing shape."""

    if not isinstance(node, IntentNode):
        raise TypeError("node must be an IntentNode.")
    return GoalStatusItem(node_id=node.id, node_text=node.text)


def messages_to_payload(messages: list[ConversationMessage]) -> list[dict[str, str]]:
    """Serialize chat history for user-generation prompting."""

    payload: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, ConversationMessage):
            raise TypeError("messages must contain ConversationMessage entries.")
        payload.append({"role": message.role, "content": message.content})
    return payload


def _items_to_payload(items: tuple[GoalStatusItem, ...]) -> list[dict[str, str]]:
    return [{"id": item.node_id, "text": item.node_text} for item in items]


def _json_block(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=True, sort_keys=False)


def _deepest_frontier_items(node: IntentNode, target_state: NodeState) -> list[IntentNode]:
    matching_descendants: list[IntentNode] = []
    for child in node.children:
        matching_descendants.extend(_deepest_frontier_items(child, target_state))
    if matching_descendants:
        return matching_descendants
    if not node.satisfied and node.state == target_state:
        return [node]
    return []


def _topmost_frontier_items(
    node: IntentNode,
    target_state: NodeState,
    *,
    ancestor_matches: bool = False,
) -> list[IntentNode]:
    current_matches = not node.satisfied and node.state == target_state
    if current_matches and not ancestor_matches:
        return [node]

    items: list[IntentNode] = []
    for child in node.children:
        items.extend(
            _topmost_frontier_items(
                child,
                target_state,
                ancestor_matches=ancestor_matches or current_matches,
            )
        )
    return items
