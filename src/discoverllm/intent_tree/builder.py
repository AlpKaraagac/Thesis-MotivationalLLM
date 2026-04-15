"""Four-stage intent-tree construction orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable

from discoverllm.intent_tree.abstraction import (
    criteria_from_checklist,
    normalize_abstractions_for_hierarchy,
)
from discoverllm.intent_tree.initial_request import (
    apply_initial_discovery,
    build_conversation_state,
    constrain_selected_root_ids,
    ground_selected_root_ids_to_initial_request,
    parse_selected_root_ids,
    partition_root_criteria_for_initial_request,
)
from discoverllm.intent_tree.organizer import assign_thresholds, forest_from_hierarchy_payload
from discoverllm.llm.base import LLMAdapter, LLMRequest, LLMResponse
from discoverllm.parsers.common import StructuredParseError
from discoverllm.parsers.contracts import (
    parse_hierarchy_organization_output,
    parse_initial_intent_synthesis_output,
    parse_initial_user_request_output,
    parse_intent_abstraction_output,
)
from discoverllm.prompts.loader import PromptLoader
from discoverllm.types import ArtifactSpec, ConversationMessage, ConversationState, IntentForest


@dataclass(frozen=True, slots=True)
class PromptStageRecord:
    """Prompt input/output persistence record for one tree-building stage."""

    stage_name: str
    prompt_name: str
    rendered_prompt: str
    llm_text_output: str
    llm_raw_output: str | dict[str, Any]
    parsed_output: dict[str, Any]


@dataclass(frozen=True, slots=True)
class IntentTreeBuildAudit:
    """All four stage records for audit-friendly persistence."""

    initial_intent_synthesis: PromptStageRecord
    intent_abstraction: PromptStageRecord
    hierarchy_organization: PromptStageRecord
    initial_user_request: PromptStageRecord


@dataclass(frozen=True, slots=True)
class IntentTreeInitializationPackage:
    """Output package consumed by later simulation milestones."""

    artifact: ArtifactSpec
    intent_forest: IntentForest
    initial_request: str
    initial_discovered_root_ids: list[str]
    conversation_state: ConversationState
    seed: int
    audit: IntentTreeBuildAudit


ParserFn = Callable[[str], dict[str, Any]]


class IntentTreeBuilder:
    """Paper-faithful four-stage builder with explicit prompt+parser boundaries."""

    def __init__(
        self,
        *,
        llm_adapter: LLMAdapter,
        prompt_loader: PromptLoader | None = None,
        num_abstraction_levels: int = 3,
        stable_root_sort: bool = True,
    ) -> None:
        if not isinstance(llm_adapter, LLMAdapter):
            raise TypeError("llm_adapter must implement LLMAdapter.")
        if prompt_loader is None:
            prompt_loader = PromptLoader()
        if not isinstance(prompt_loader, PromptLoader):
            raise TypeError("prompt_loader must be a PromptLoader.")
        if not isinstance(num_abstraction_levels, int) or num_abstraction_levels < 1:
            raise ValueError("num_abstraction_levels must be >= 1.")
        if not isinstance(stable_root_sort, bool):
            raise TypeError("stable_root_sort must be a bool.")

        self._llm = llm_adapter
        self._prompts = prompt_loader
        self._num_levels = num_abstraction_levels
        self._stable_root_sort = stable_root_sort

    def build_from_artifact(
        self,
        artifact: ArtifactSpec,
        *,
        seed: int,
        latent_root_ids: list[str] | None = None,
    ) -> IntentTreeInitializationPackage:
        """Run stages 1-4 and return simulator initialization package."""

        if not isinstance(artifact, ArtifactSpec):
            raise TypeError("artifact must be an ArtifactSpec.")
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer.")

        stage1 = self._run_stage(
            stage_name="stage_1_initial_intent_synthesis",
            prompt_name="initial_intent_synthesis",
            variables={
                "artifact_type": artifact.artifact_type,
                "artifact_content": artifact.artifact_content,
                "examples": "",
            },
            parser=parse_initial_intent_synthesis_output,
        )
        checklist = stage1.parsed_output["checklist"]
        criteria = criteria_from_checklist(checklist)
        criteria_payload = [
            {"criterion_id": item.criterion_id, "criterion": item.criterion} for item in criteria
        ]

        stage2 = self._run_stage(
            stage_name="stage_2_intent_abstraction",
            prompt_name="intent_abstraction",
            variables={
                "artifact_type": artifact.artifact_type,
                "artifact_topic": stage1.parsed_output["artifact_topic"],
                "num_levels": self._num_levels,
                "criteria_json": _json_block(criteria_payload),
                "examples": "",
            },
            parser=parse_intent_abstraction_output,
        )
        hierarchy_input = normalize_abstractions_for_hierarchy(criteria, stage2.parsed_output)

        stage3 = self._run_stage(
            stage_name="stage_3_hierarchy_organization",
            prompt_name="hierarchy_organization",
            variables={
                "abstraction_items_json": _json_block(hierarchy_input),
            },
            parser=parse_hierarchy_organization_output,
        )

        forest = forest_from_hierarchy_payload(
            stage3.parsed_output, stable_root_sort=self._stable_root_sort
        )
        assign_thresholds(forest, seed=seed)

        visible_roots, latent_roots = partition_root_criteria_for_initial_request(
            forest,
            seed=seed,
            forced_latent_root_ids=latent_root_ids,
        )
        stage4 = self._run_stage(
            stage_name="stage_4_initial_user_request",
            prompt_name="initial_user_request",
            variables={
                "artifact_type": artifact.artifact_type,
                "artifact_topic": stage1.parsed_output["artifact_topic"],
                "visible_root_criteria_json": _json_block(visible_roots),
                "latent_criteria_json": _json_block(latent_roots),
            },
            parser=parse_initial_user_request_output,
        )

        selected_root_ids = constrain_selected_root_ids(
            parse_selected_root_ids(stage4.parsed_output),
            visible_root_ids=[item["id"] for item in visible_roots],
        )
        selected_root_ids = ground_selected_root_ids_to_initial_request(
            forest,
            initial_request=stage4.parsed_output["initial_request"],
            selected_root_ids=selected_root_ids,
        )
        stage4.parsed_output["criteria_selected_for_request"] = selected_root_ids
        applied_root_ids = apply_initial_discovery(forest, selected_root_ids)

        initial_request = stage4.parsed_output["initial_request"]
        conversation_state = build_conversation_state(
            artifact,
            forest,
            initial_request=initial_request,
            initial_discovered_root_ids=applied_root_ids,
            seed=seed,
        )

        audit = IntentTreeBuildAudit(
            initial_intent_synthesis=stage1,
            intent_abstraction=stage2,
            hierarchy_organization=stage3,
            initial_user_request=stage4,
        )
        return IntentTreeInitializationPackage(
            artifact=artifact,
            intent_forest=forest,
            initial_request=initial_request,
            initial_discovered_root_ids=applied_root_ids,
            conversation_state=conversation_state,
            seed=seed,
            audit=audit,
        )

    def _run_stage(
        self,
        *,
        stage_name: str,
        prompt_name: str,
        variables: dict[str, Any],
        parser: ParserFn,
    ) -> PromptStageRecord:
        rendered_prompt = self._prompts.render(prompt_name, variables, strict=True)
        response = self._generate(prompt_name=prompt_name, prompt_text=rendered_prompt)
        try:
            parsed_output = parser(response.text)
        except StructuredParseError as exc:
            excerpt = _output_excerpt(response.text)
            raise StructuredParseError(
                f"{stage_name} failed to parse {prompt_name!r} output: {exc}\n"
                f"Raw output excerpt:\n{excerpt}"
            ) from exc
        return PromptStageRecord(
            stage_name=stage_name,
            prompt_name=prompt_name,
            rendered_prompt=rendered_prompt,
            llm_text_output=response.text,
            llm_raw_output=response.raw_output,
            parsed_output=parsed_output,
        )

    def _generate(self, *, prompt_name: str, prompt_text: str) -> LLMResponse:
        request = LLMRequest(
            messages=[ConversationMessage(role="user", content=prompt_text)],
            temperature=0.0,
            metadata={"prompt_name": prompt_name},
        )
        return self._llm.generate(request)


def _json_block(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=True, sort_keys=False)


def _output_excerpt(text: str, *, limit: int = 1200) -> str:
    cleaned = text.strip()
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit].rstrip()}\n...<truncated>"
