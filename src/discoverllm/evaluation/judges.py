"""Prompt-backed evaluation judges for artifact satisfaction and interactivity."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Sequence

from discoverllm.evaluation.metrics import (
    IntentSatisfactionScoreResult,
    LeafSatisfactionEvaluation,
    coerce_leaf_satisfaction_evaluations,
    compute_intent_satisfaction_score,
    leaf_requirements_payload,
    normalize_interactivity_score,
)
from discoverllm.llm.base import LLMAdapter, LLMRequest, LLMResponse
from discoverllm.parsers.contracts import (
    parse_artifact_satisfaction_judge_output,
    parse_interactivity_judge_output,
)
from discoverllm.prompts.loader import PromptLoader
from discoverllm.types import ConversationMessage, IntentForest

DEFAULT_FINAL_ARTIFACT_REQUEST_TEMPLATE = (
    "Using our conversation so far, please generate the complete {{artifact_type}} now."
)


@dataclass(frozen=True, slots=True)
class ArtifactSatisfactionJudgeRun:
    """Audit-friendly record of one artifact satisfaction judge call."""

    leaf_requirements_payload: list[dict[str, str]]
    rendered_prompt: str
    llm_text_output: str
    llm_raw_output: str | dict[str, Any]
    parsed_output: dict[str, Any]
    leaf_evaluations: tuple[LeafSatisfactionEvaluation, ...]
    score: IntentSatisfactionScoreResult


@dataclass(frozen=True, slots=True)
class InteractivityJudgeRun:
    """Audit-friendly record of one conversation-level interactivity judgment."""

    conversation_transcript: str
    rendered_prompt: str
    llm_text_output: str
    llm_raw_output: str | dict[str, Any]
    parsed_output: dict[str, Any]
    raw_score: float
    normalized_score: float


class ArtifactSatisfactionJudge:
    """Judge a final artifact against the leaf requirements of an intent forest."""

    def __init__(
        self,
        *,
        llm_adapter: LLMAdapter,
        prompt_loader: PromptLoader | None = None,
        satisfied_threshold: int = 4,
    ) -> None:
        if not isinstance(llm_adapter, LLMAdapter):
            raise TypeError("llm_adapter must implement LLMAdapter.")
        if prompt_loader is None:
            prompt_loader = PromptLoader()
        if not isinstance(prompt_loader, PromptLoader):
            raise TypeError("prompt_loader must be a PromptLoader.")
        if not isinstance(satisfied_threshold, int):
            raise TypeError("satisfied_threshold must be an int.")
        if not 1 <= satisfied_threshold <= 5:
            raise ValueError("satisfied_threshold must be in [1, 5].")
        self._llm = llm_adapter
        self._prompts = prompt_loader
        self._satisfied_threshold = satisfied_threshold

    def judge_artifact(
        self,
        final_artifact: str,
        intent_forest: IntentForest,
    ) -> ArtifactSatisfactionJudgeRun:
        if not isinstance(final_artifact, str) or not final_artifact.strip():
            raise ValueError("final_artifact must be a non-empty string.")
        if not isinstance(intent_forest, IntentForest):
            raise TypeError("intent_forest must be an IntentForest.")

        leaf_payload = leaf_requirements_payload(intent_forest)
        if not leaf_payload:
            raise ValueError("intent_forest must contain at least one leaf requirement.")

        rendered_prompt = self._prompts.render(
            "artifact_satisfaction_judge",
            {
                "final_artifact": final_artifact.strip(),
                "leaf_requirements_json": _json_block(leaf_payload),
            },
            strict=True,
        )
        response = self._generate(rendered_prompt)
        parsed_output = parse_artifact_satisfaction_judge_output(response.text)
        requirement_text_by_id = {item["id"]: item["text"] for item in leaf_payload}
        leaf_evaluations = coerce_leaf_satisfaction_evaluations(
            parsed_output["leaf_evaluations"],
            requirement_text_by_id=requirement_text_by_id,
        )
        score = compute_intent_satisfaction_score(
            leaf_evaluations,
            satisfied_threshold=self._satisfied_threshold,
        )
        return ArtifactSatisfactionJudgeRun(
            leaf_requirements_payload=leaf_payload,
            rendered_prompt=rendered_prompt,
            llm_text_output=response.text,
            llm_raw_output=response.raw_output,
            parsed_output=parsed_output,
            leaf_evaluations=leaf_evaluations,
            score=score,
        )

    def _generate(self, prompt_text: str) -> LLMResponse:
        request = LLMRequest(
            messages=[ConversationMessage(role="user", content=prompt_text)],
            temperature=0.0,
            metadata={"prompt_name": "artifact_satisfaction_judge"},
        )
        return self._llm.generate(request)


class InteractivityJudge:
    """Judge whole-conversation interactivity on the paper's 1-3 scale."""

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

    def judge_messages(
        self,
        messages: Sequence[ConversationMessage],
    ) -> InteractivityJudgeRun:
        if not messages:
            raise ValueError("messages must not be empty.")
        transcript = messages_to_transcript(messages)
        rendered_prompt = self._prompts.render(
            "interactivity_judge",
            {"conversation_transcript": transcript},
            strict=True,
        )
        response = self._generate(rendered_prompt)
        parsed_output = parse_interactivity_judge_output(response.text)
        raw_score = float(parsed_output["interactivity"])
        return InteractivityJudgeRun(
            conversation_transcript=transcript,
            rendered_prompt=rendered_prompt,
            llm_text_output=response.text,
            llm_raw_output=response.raw_output,
            parsed_output=parsed_output,
            raw_score=raw_score,
            normalized_score=normalize_interactivity_score(raw_score),
        )

    def _generate(self, prompt_text: str) -> LLMResponse:
        request = LLMRequest(
            messages=[ConversationMessage(role="user", content=prompt_text)],
            temperature=0.0,
            metadata={"prompt_name": "interactivity_judge"},
        )
        return self._llm.generate(request)


def build_final_artifact_request(artifact_type: str) -> ConversationMessage:
    """Build the standard final user message requesting the complete artifact."""

    if not isinstance(artifact_type, str) or not artifact_type.strip():
        raise ValueError("artifact_type must be a non-empty string.")
    content = DEFAULT_FINAL_ARTIFACT_REQUEST_TEMPLATE.replace(
        "{{artifact_type}}",
        artifact_type.strip(),
    )
    return ConversationMessage(role="user", content=content)


def append_final_artifact_request(
    messages: Sequence[ConversationMessage],
    *,
    artifact_type: str,
) -> list[ConversationMessage]:
    """Return a copied transcript with the standard final artifact request appended."""

    if not messages:
        raise ValueError("messages must not be empty.")
    copied: list[ConversationMessage] = []
    for message in messages:
        if not isinstance(message, ConversationMessage):
            raise TypeError("messages must contain ConversationMessage entries.")
        copied.append(ConversationMessage(role=message.role, content=message.content))
    copied.append(build_final_artifact_request(artifact_type))
    return copied


def messages_to_transcript(messages: Sequence[ConversationMessage]) -> str:
    """Serialize a conversation into a stable judge-facing transcript string."""

    lines: list[str] = []
    for message in messages:
        if not isinstance(message, ConversationMessage):
            raise TypeError("messages must contain ConversationMessage entries.")
        lines.append(f"{message.role.upper()}: {message.content}")
    return "\n".join(lines)


def _json_block(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=True, sort_keys=False)
