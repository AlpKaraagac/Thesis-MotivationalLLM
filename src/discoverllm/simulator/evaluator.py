"""Assistant-response evaluator wrapper and traversal normalization."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Sequence

from discoverllm.llm.base import LLMAdapter, LLMRequest, LLMResponse
from discoverllm.parsers.common import StructuredParseError
from discoverllm.parsers.contracts import parse_assistant_response_evaluation_output
from discoverllm.prompts.loader import PromptLoader
from discoverllm.text_grounding import grounding_score, overlap_tokens
from discoverllm.types import (
    ConversationMessage,
    EvaluationResult,
    IntentForest,
    IntentNode,
    IntentTree,
    NodeEvaluation,
    NodeState,
)


@dataclass(frozen=True, slots=True)
class EvaluationScopeAudit:
    """Audit summary of which nodes were and were not evaluated in one active subtree."""

    active_root_id: str
    evaluated_node_ids: tuple[str, ...]
    unevaluated_node_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AssistantResponseEvaluationRun:
    """Audit-friendly result for one evaluator invocation."""

    active_root_id: str
    chat_history_json: list[dict[str, str]]
    active_root_subtree_json: dict[str, Any]
    rendered_prompt: str
    llm_text_output: str
    llm_raw_output: str | dict[str, Any]
    parsed_output: dict[str, Any]
    normalized_output: dict[str, Any]
    scope_audit: EvaluationScopeAudit
    evaluation: EvaluationResult


class AssistantResponseEvaluator:
    """Evaluate only the last assistant message against one active subtree."""

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

    def evaluate_messages(
        self,
        messages: Sequence[ConversationMessage],
        intent_forest: IntentForest,
    ) -> AssistantResponseEvaluationRun:
        """Evaluate the final assistant message under paper-faithful subtree scoping."""

        if not isinstance(intent_forest, IntentForest):
            raise TypeError("intent_forest must be an IntentForest.")
        if not messages:
            raise ValueError("messages must not be empty.")
        for message in messages:
            if not isinstance(message, ConversationMessage):
                raise TypeError("messages must contain ConversationMessage entries.")
        if messages[-1].role != "assistant":
            raise ValueError("The last message must be an assistant message for evaluation.")

        active_tree = select_active_subtree(intent_forest)
        if active_tree is None:
            raise ValueError("No active root subtree remains; all intents are already discovered.")

        chat_history_payload = messages_to_payload(messages)
        active_subtree_payload = subtree_to_payload(active_tree.root)
        rendered_prompt = self._prompts.render(
            "assistant_response_evaluation",
            {
                "chat_history_json": _json_block(chat_history_payload),
                "active_root_subtree_json": _json_block(active_subtree_payload),
            },
            strict=True,
        )
        response = self._generate(
            prompt_text=rendered_prompt,
            active_root_id=active_tree.root.id,
        )
        try:
            parsed = parse_assistant_response_evaluation_output(response.text)
        except StructuredParseError as exc:
            raise StructuredParseError(
                "assistant_response_evaluation failed to parse evaluator output: "
                f"{exc}\nRaw output excerpt:\n{_output_excerpt(response.text)}"
            ) from exc
        try:
            normalized = normalize_evaluation_result(parsed, active_tree)
        except ValueError as exc:
            raise ValueError(
                "assistant_response_evaluation produced invalid traversal semantics: "
                f"{exc}\nRaw output excerpt:\n{_output_excerpt(response.text)}"
            ) from exc
        refined = refine_evaluation_result(
            normalized,
            messages=messages,
            active_tree=active_tree,
        )
        scope_audit = build_evaluation_scope_audit(refined, active_tree)
        return AssistantResponseEvaluationRun(
            active_root_id=active_tree.root.id,
            chat_history_json=chat_history_payload,
            active_root_subtree_json=active_subtree_payload,
            rendered_prompt=rendered_prompt,
            llm_text_output=response.text,
            llm_raw_output=response.raw_output,
            parsed_output=evaluation_result_to_payload(parsed),
            normalized_output=evaluation_result_to_payload(refined),
            scope_audit=scope_audit,
            evaluation=refined,
        )

    def _generate(self, *, prompt_text: str, active_root_id: str) -> LLMResponse:
        request = LLMRequest(
            messages=[ConversationMessage(role="user", content=prompt_text)],
            temperature=0.0,
            metadata={
                "prompt_name": "assistant_response_evaluation",
                "active_root_id": active_root_id,
            },
        )
        return self._llm.generate(request)


def select_active_subtree(intent_forest: IntentForest) -> IntentTree | None:
    """Pick the first root subtree with at least one undiscovered node."""

    if not isinstance(intent_forest, IntentForest):
        raise TypeError("intent_forest must be an IntentForest.")
    for tree in intent_forest.trees:
        if any(node.state == NodeState.UNDISCOVERED for node in tree.iter_depth_first()):
            return tree
    return None


def normalize_evaluation_result(
    evaluation: EvaluationResult,
    active_tree: IntentTree,
) -> EvaluationResult:
    """Canonicalize evaluator output against the actual active subtree."""

    if not isinstance(evaluation, EvaluationResult):
        raise TypeError("evaluation must be an EvaluationResult.")
    if not isinstance(active_tree, IntentTree):
        raise TypeError("active_tree must be an IntentTree.")
    if not evaluation.evaluations:
        raise ValueError("Evaluator output must contain at least one node evaluation.")

    index, normalized_evaluations = _consume_branch(
        active_tree.root,
        evaluation.evaluations,
        index=0,
    )
    if index != len(evaluation.evaluations):
        extra = evaluation.evaluations[index].node_id
        raise ValueError(
            f"Evaluator output contains extra node {extra!r} outside the active traversal."
        )
    return EvaluationResult(
        classification_reasoning=evaluation.classification_reasoning.strip(),
        classification_label=evaluation.classification_label,
        evaluation_type=evaluation.evaluation_type,
        evaluations=normalized_evaluations,
    )


def messages_to_payload(messages: Sequence[ConversationMessage]) -> list[dict[str, str]]:
    """Serialize chat history for prompt rendering."""

    payload: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, ConversationMessage):
            raise TypeError("messages must contain ConversationMessage entries.")
        payload.append({"role": message.role, "content": message.content})
    return payload


def refine_evaluation_result(
    evaluation: EvaluationResult,
    *,
    messages: Sequence[ConversationMessage],
    active_tree: IntentTree,
) -> EvaluationResult:
    """Apply deterministic guardrails on top of the model-evaluator output."""

    if not isinstance(evaluation, EvaluationResult):
        raise TypeError("evaluation must be an EvaluationResult.")
    if not isinstance(active_tree, IntentTree):
        raise TypeError("active_tree must be an IntentTree.")

    refined = evaluation
    if refined.evaluation_type == "probing":
        refined = _downgrade_redundant_clarification_probe(
            refined,
            messages=messages,
            active_tree=active_tree,
        )
    if refined.evaluation_type == "satisfaction":
        refined = _preserve_ancestor_credit_and_leaf_strictness(
            refined,
            assistant_text=messages[-1].content,
            active_tree=active_tree,
        )
    return refined


def build_evaluation_scope_audit(
    evaluation: EvaluationResult,
    active_tree: IntentTree,
) -> EvaluationScopeAudit:
    """Summarize the exact portion of the active subtree traversed this turn."""

    if not isinstance(evaluation, EvaluationResult):
        raise TypeError("evaluation must be an EvaluationResult.")
    if not isinstance(active_tree, IntentTree):
        raise TypeError("active_tree must be an IntentTree.")

    evaluated_ids = tuple(item.node_id for item in evaluation.evaluations)
    evaluated_set = set(evaluated_ids)
    unevaluated_ids = tuple(
        node.id for node in active_tree.iter_depth_first() if node.id not in evaluated_set
    )
    return EvaluationScopeAudit(
        active_root_id=active_tree.root.id,
        evaluated_node_ids=evaluated_ids,
        unevaluated_node_ids=unevaluated_ids,
    )


def subtree_to_payload(node: IntentNode) -> dict[str, Any]:
    """Serialize one intent subtree into the prompt contract shape."""

    if not isinstance(node, IntentNode):
        raise TypeError("node must be an IntentNode.")
    return {
        "id": node.id,
        "text": node.text,
        "children": [subtree_to_payload(child) for child in node.children],
    }


def evaluation_result_to_payload(result: EvaluationResult) -> dict[str, Any]:
    """Convert a typed evaluator result back into a persistence-friendly payload."""

    if not isinstance(result, EvaluationResult):
        raise TypeError("result must be an EvaluationResult.")
    return {
        "classification_reasoning": result.classification_reasoning,
        "classification_label": result.classification_label,
        "evaluation_type": result.evaluation_type,
        "evaluations": [
            {
                "node_id": item.node_id,
                "node_text": item.node_text,
                "reasoning": item.reasoning,
                "is_satisfied_or_probed": item.is_satisfied_or_probed,
                "near_miss": list(item.near_miss),
                "children_evaluated": item.children_evaluated,
            }
            for item in result.evaluations
        ],
    }


def _consume_branch(
    node: IntentNode,
    evaluations: list[NodeEvaluation],
    *,
    index: int,
) -> tuple[int, list[NodeEvaluation]]:
    if index >= len(evaluations):
        raise ValueError(f"Missing evaluation for node {node.id!r}.")

    current = evaluations[index]
    if current.node_id != node.id:
        raise ValueError(
            f"Traversal order mismatch: expected node {node.id!r}, got {current.node_id!r}."
        )

    normalized_current = NodeEvaluation(
        node_id=node.id,
        node_text=node.text,
        reasoning=current.reasoning.strip(),
        is_satisfied_or_probed=current.is_satisfied_or_probed,
        near_miss=_normalize_near_miss(current.near_miss),
        children_evaluated=current.children_evaluated,
    )
    index += 1
    normalized = [normalized_current]

    if not node.children:
        if normalized_current.children_evaluated:
            raise ValueError(f"Leaf node {node.id!r} cannot report children_evaluated=true.")
        return index, normalized

    if not normalized_current.is_satisfied_or_probed:
        if normalized_current.children_evaluated:
            raise ValueError(
                f"Node {node.id!r} cannot report children_evaluated=true when not engaged."
            )
        return index, normalized

    if not normalized_current.children_evaluated:
        raise ValueError(
            f"Engaged non-leaf node {node.id!r} must report children_evaluated=true."
        )

    for child in node.children:
        index, child_items = _consume_branch(child, evaluations, index=index)
        normalized.extend(child_items)
    return index, normalized


def _downgrade_redundant_clarification_probe(
    evaluation: EvaluationResult,
    *,
    messages: Sequence[ConversationMessage],
    active_tree: IntentTree,
) -> EvaluationResult:
    latest_user_message = _latest_user_message(messages)
    if latest_user_message is None:
        return evaluation
    if not evaluation.evaluations:
        return evaluation

    root_evaluation = evaluation.evaluations[0]
    if not root_evaluation.is_satisfied_or_probed:
        return evaluation
    if _root_grounding_score(active_tree, latest_user_message.content) < 2:
        return evaluation

    assistant_text = messages[-1].content
    if not _is_clarifying_menu_dialog_act(assistant_text):
        return evaluation

    downgraded_root = NodeEvaluation(
        node_id=root_evaluation.node_id,
        node_text=root_evaluation.node_text,
        reasoning=(
            f"{root_evaluation.reasoning} The user had already specified this dimension "
            "concretely, so this clarification does not count as new probing."
        ),
        is_satisfied_or_probed=False,
        near_miss=_redundant_probe_near_miss(root_evaluation.node_text),
        children_evaluated=False,
    )
    return EvaluationResult(
        classification_reasoning=evaluation.classification_reasoning,
        classification_label=evaluation.classification_label,
        evaluation_type=evaluation.evaluation_type,
        evaluations=[downgraded_root],
    )


def _preserve_ancestor_credit_and_leaf_strictness(
    evaluation: EvaluationResult,
    *,
    assistant_text: str,
    active_tree: IntentTree,
) -> EvaluationResult:
    evaluation_by_id = {item.node_id: item for item in evaluation.evaluations}
    rebuilt = _rebuild_satisfaction_branch(
        active_tree.root,
        evaluation_by_id=evaluation_by_id,
        assistant_text=assistant_text,
    )
    return EvaluationResult(
        classification_reasoning=evaluation.classification_reasoning,
        classification_label=evaluation.classification_label,
        evaluation_type=evaluation.evaluation_type,
        evaluations=rebuilt,
    )


def _normalize_near_miss(values: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for item in values:
        cleaned = item.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        normalized.append(cleaned)
    return normalized


def _rebuild_satisfaction_branch(
    node: IntentNode,
    *,
    evaluation_by_id: dict[str, NodeEvaluation],
    assistant_text: str,
) -> list[NodeEvaluation]:
    current = evaluation_by_id.get(node.id)
    subtree_texts = [item.text for item in node.iter_depth_first()]
    subtree_overlap = overlap_tokens(assistant_text, subtree_texts)
    subtree_score = grounding_score(assistant_text, subtree_texts)

    if not node.children:
        return [
            _rebuild_leaf_evaluation(
                node,
                current=current,
                assistant_text=assistant_text,
            )
        ]

    if current is not None and current.is_satisfied_or_probed:
        root_item = NodeEvaluation(
            node_id=current.node_id,
            node_text=current.node_text,
            reasoning=current.reasoning,
            is_satisfied_or_probed=True,
            near_miss=[],
            children_evaluated=True,
        )
    elif subtree_score >= 1:
        root_item = NodeEvaluation(
            node_id=node.id,
            node_text=node.text,
            reasoning=_ancestor_preservation_reason(node, current=current, overlap=subtree_overlap),
            is_satisfied_or_probed=True,
            near_miss=[],
            children_evaluated=True,
        )
    else:
        return [
            NodeEvaluation(
                node_id=node.id,
                node_text=node.text,
                reasoning=_negative_reasoning(
                    node,
                    current=current,
                    overlap=subtree_overlap,
                ),
                is_satisfied_or_probed=False,
                near_miss=_near_miss_from_overlap(subtree_overlap, exact_leaf=False),
                children_evaluated=False,
            )
        ]

    rebuilt = [root_item]
    for child in node.children:
        rebuilt.extend(
            _rebuild_satisfaction_branch(
                child,
                evaluation_by_id=evaluation_by_id,
                assistant_text=assistant_text,
            )
        )
    return rebuilt


def _rebuild_leaf_evaluation(
    node: IntentNode,
    *,
    current: NodeEvaluation | None,
    assistant_text: str,
) -> NodeEvaluation:
    if current is not None and current.is_satisfied_or_probed:
        return current

    overlap = overlap_tokens(assistant_text, [node.text])
    reasoning = _negative_reasoning(node, current=current, overlap=overlap, exact_leaf=True)
    near_miss = list(current.near_miss) if current is not None else []
    if not near_miss:
        near_miss = _near_miss_from_overlap(overlap, exact_leaf=True)
    return NodeEvaluation(
        node_id=node.id,
        node_text=node.text,
        reasoning=reasoning,
        is_satisfied_or_probed=False,
        near_miss=near_miss,
        children_evaluated=False,
    )


def _ancestor_preservation_reason(
    node: IntentNode,
    *,
    current: NodeEvaluation | None,
    overlap: list[str],
) -> str:
    if current is not None and current.reasoning.strip():
        return (
            f"{current.reasoning} The artifact still clearly matches this broader abstraction "
            "even if the deepest specific detail remains only partial."
        )
    if overlap:
        return (
            "The artifact clearly matches this broader abstraction through concrete cues in "
            f"the subtree ({', '.join(overlap[:4])})."
        )
    return (
        "The artifact clearly matches this broader abstraction even though the most specific "
        "descendant detail is not fully established."
    )


def _negative_reasoning(
    node: IntentNode,
    *,
    current: NodeEvaluation | None,
    overlap: list[str],
    exact_leaf: bool = False,
) -> str:
    if current is not None and current.reasoning.strip():
        if overlap and exact_leaf:
            return (
                f"{current.reasoning} The artifact comes close at a broader level, but the "
                "exact leaf detail is still not explicit enough."
            )
        return current.reasoning
    if overlap and exact_leaf:
        return (
            "The artifact comes close at a broader level, but it does not make this exact "
            f"leaf detail explicit enough ({', '.join(overlap[:4])})."
        )
    if overlap:
        return (
            "The artifact hints at nearby aspects of this branch, but it does not clearly "
            f"establish the requirement ({', '.join(overlap[:4])})."
        )
    return f"The artifact does not clearly establish {node.text!r}."


def _near_miss_from_overlap(overlap: list[str], *, exact_leaf: bool) -> list[str]:
    if not overlap:
        return []
    if exact_leaf:
        return [
            "The artifact matches related leaf-level cues but not the exact leaf detail "
            f"({', '.join(overlap[:4])})."
        ]
    return [
        "The artifact gestures toward related parts of this branch without fully establishing "
        f"the requirement ({', '.join(overlap[:4])})."
    ]


def _latest_user_message(messages: Sequence[ConversationMessage]) -> ConversationMessage | None:
    for message in reversed(messages[:-1]):
        if message.role == "user":
            return message
    return None


def _root_grounding_score(active_tree: IntentTree, text: str) -> int:
    return max(grounding_score(text, [node.text]) for node in active_tree.iter_depth_first())


def _is_clarifying_menu_dialog_act(text: str) -> bool:
    lowered = text.lower()
    if "?" not in lowered and "to clarify" not in lowered:
        return False
    patterns = (
        r"\bwould you like\b",
        r"\bdo you want\b",
        r"\bwhich\b",
        r"\bchoose\b",
        r"\bprefer\b",
        r"\blean into\b",
        r"\bcombine\b",
        r"\bto clarify\b",
        r"\bhere are\b",
        r"\boptions?\b",
        r"\bdirections?\b",
        r"\bexamples?\b",
        r"\b1\.",
    )
    return any(re.search(pattern, lowered) for pattern in patterns)


def _redundant_probe_near_miss(node_text: str) -> list[str]:
    return [
        f"The assistant explores nearby options around {node_text!r} without eliciting "
        "genuinely new information beyond the user's stated request."
    ]


def _json_block(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=True, sort_keys=False)


def _output_excerpt(text: str, *, limit: int = 1200) -> str:
    cleaned = text.strip()
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit].rstrip()}\n...<truncated>"
