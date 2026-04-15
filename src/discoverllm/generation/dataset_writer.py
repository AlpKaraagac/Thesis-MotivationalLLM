"""Serialization and JSONL writers for DiscoverLLM build/simulation outputs."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from discoverllm.generation.turn_loop import ConversationSimulationResult
from discoverllm.intent_tree.builder import (
    IntentTreeBuildAudit,
    IntentTreeInitializationPackage,
    PromptStageRecord,
)
from discoverllm.types import (
    ArtifactSpec,
    ConversationMessage,
    ConversationState,
    IntentForest,
    IntentNode,
    IntentTree,
)


@dataclass(frozen=True, slots=True)
class BuildOutputPaths:
    """Files written for a single build-tree run."""

    intent_tree_json: Path
    simulator_init_json: Path
    builder_audit_json: Path
    raw_llm_transcripts_jsonl: Path


@dataclass(frozen=True, slots=True)
class SimulationOutputPaths:
    """Files written for a single simulate-conversation run."""

    chosen_trajectory_jsonl: Path
    preferences_jsonl: Path
    turn_comparisons_jsonl: Path
    simulation_result_json: Path
    raw_llm_transcripts_jsonl: Path


@dataclass(frozen=True, slots=True)
class DatasetOutputPaths:
    """Top-level aggregate files written for a dataset-generation run."""

    artifact_output_dirs: tuple[Path, ...]
    chosen_trajectories_jsonl: Path
    preferences_jsonl: Path
    turn_comparisons_jsonl: Path
    raw_llm_transcripts_jsonl: Path


def to_jsonable(value: Any) -> Any:
    """Recursively convert dataclasses, enums, and paths into JSON-safe values."""

    if is_dataclass(value):
        return {
            field.name: to_jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [to_jsonable(item) for item in value]
        try:
            return sorted(items)
        except TypeError:
            return items
    return value


def serialize_artifact_spec(artifact: ArtifactSpec) -> dict[str, Any]:
    if not isinstance(artifact, ArtifactSpec):
        raise TypeError("artifact must be an ArtifactSpec.")
    return to_jsonable(artifact)


def serialize_conversation_state(conversation_state: ConversationState) -> dict[str, Any]:
    if not isinstance(conversation_state, ConversationState):
        raise TypeError("conversation_state must be a ConversationState.")
    return to_jsonable(conversation_state)


def serialize_intent_forest(intent_forest: IntentForest) -> dict[str, Any]:
    if not isinstance(intent_forest, IntentForest):
        raise TypeError("intent_forest must be an IntentForest.")
    return to_jsonable(intent_forest)


def serialize_prompt_stage_record(stage_record: PromptStageRecord) -> dict[str, Any]:
    if not isinstance(stage_record, PromptStageRecord):
        raise TypeError("stage_record must be a PromptStageRecord.")
    return to_jsonable(stage_record)


def serialize_intent_tree_build_audit(audit: IntentTreeBuildAudit) -> dict[str, Any]:
    if not isinstance(audit, IntentTreeBuildAudit):
        raise TypeError("audit must be an IntentTreeBuildAudit.")
    return to_jsonable(audit)


def serialize_intent_tree_initialization_package(
    package: IntentTreeInitializationPackage,
) -> dict[str, Any]:
    if not isinstance(package, IntentTreeInitializationPackage):
        raise TypeError("package must be an IntentTreeInitializationPackage.")
    return to_jsonable(package)


def serialize_conversation_simulation_result(
    result: ConversationSimulationResult,
) -> dict[str, Any]:
    if not isinstance(result, ConversationSimulationResult):
        raise TypeError("result must be a ConversationSimulationResult.")
    return to_jsonable(result)


def deserialize_artifact_spec(payload: Mapping[str, Any]) -> ArtifactSpec:
    payload = _require_mapping(payload, context="artifact")
    return ArtifactSpec(
        artifact_id=_require_string(payload.get("artifact_id"), context="artifact_id"),
        artifact_type=_require_string(payload.get("artifact_type"), context="artifact_type"),
        artifact_content=_require_string(
            payload.get("artifact_content"),
            context="artifact_content",
        ),
        metadata=dict(payload.get("metadata", {})),
    )


def deserialize_intent_node(payload: Mapping[str, Any]) -> IntentNode:
    payload = _require_mapping(payload, context="intent node")
    children_payload = payload.get("children", [])
    if not isinstance(children_payload, list):
        raise TypeError("intent node.children must be a list.")
    return IntentNode(
        id=_require_string(payload.get("id"), context="intent node.id"),
        text=_require_string(payload.get("text"), context="intent node.text"),
        children=[deserialize_intent_node(item) for item in children_payload],
        state=payload.get("state", "undiscovered"),
        satisfied=bool(payload.get("satisfied", False)),
        threshold=payload.get("threshold"),
        tangential_exposure_count=int(payload.get("tangential_exposure_count", 0)),
        initially_discovered=bool(payload.get("initially_discovered", False)),
    )


def deserialize_intent_forest(payload: Mapping[str, Any]) -> IntentForest:
    payload = _require_mapping(payload, context="intent_forest")
    trees_payload = payload.get("trees")
    if not isinstance(trees_payload, list) or not trees_payload:
        raise ValueError("intent_forest.trees must be a non-empty list.")
    return IntentForest(
        trees=[
            IntentTree(root=deserialize_intent_node(_require_mapping(item.get("root"), context="tree.root")))
            for item in (_require_mapping(tree, context="intent tree") for tree in trees_payload)
        ]
    )


def deserialize_conversation_message(payload: Mapping[str, Any]) -> ConversationMessage:
    payload = _require_mapping(payload, context="conversation message")
    return ConversationMessage(
        role=_require_string(payload.get("role"), context="conversation message.role"),
        content=_require_string(payload.get("content"), context="conversation message.content"),
    )


def deserialize_conversation_state(
    payload: Mapping[str, Any],
    *,
    artifact: ArtifactSpec | None = None,
    intent_forest: IntentForest | None = None,
) -> ConversationState:
    payload = _require_mapping(payload, context="conversation_state")
    if artifact is None:
        artifact = deserialize_artifact_spec(_require_mapping(payload.get("artifact"), context="conversation_state.artifact"))
    if intent_forest is None:
        intent_forest = deserialize_intent_forest(
            _require_mapping(payload.get("intent_forest"), context="conversation_state.intent_forest")
        )
    messages_payload = payload.get("messages", [])
    if not isinstance(messages_payload, list):
        raise TypeError("conversation_state.messages must be a list.")
    return ConversationState(
        artifact=artifact,
        intent_forest=intent_forest,
        messages=[deserialize_conversation_message(item) for item in messages_payload],
        turn_index=int(payload.get("turn_index", 0)),
        random_seed=payload.get("random_seed"),
        initial_discovered_node_ids=list(payload.get("initial_discovered_node_ids", [])),
    )


def deserialize_prompt_stage_record(payload: Mapping[str, Any]) -> PromptStageRecord:
    payload = _require_mapping(payload, context="prompt stage record")
    return PromptStageRecord(
        stage_name=_require_string(payload.get("stage_name"), context="stage_name"),
        prompt_name=_require_string(payload.get("prompt_name"), context="prompt_name"),
        rendered_prompt=_require_string(payload.get("rendered_prompt"), context="rendered_prompt"),
        llm_text_output=_require_string(payload.get("llm_text_output"), context="llm_text_output"),
        llm_raw_output=payload.get("llm_raw_output", ""),
        parsed_output=dict(_require_mapping(payload.get("parsed_output"), context="parsed_output")),
    )


def deserialize_intent_tree_build_audit(payload: Mapping[str, Any]) -> IntentTreeBuildAudit:
    payload = _require_mapping(payload, context="intent tree build audit")
    return IntentTreeBuildAudit(
        initial_intent_synthesis=deserialize_prompt_stage_record(
            _require_mapping(
                payload.get("initial_intent_synthesis"),
                context="initial_intent_synthesis",
            )
        ),
        intent_abstraction=deserialize_prompt_stage_record(
            _require_mapping(payload.get("intent_abstraction"), context="intent_abstraction")
        ),
        hierarchy_organization=deserialize_prompt_stage_record(
            _require_mapping(
                payload.get("hierarchy_organization"),
                context="hierarchy_organization",
            )
        ),
        initial_user_request=deserialize_prompt_stage_record(
            _require_mapping(payload.get("initial_user_request"), context="initial_user_request")
        ),
    )


def deserialize_intent_tree_initialization_package(
    payload: Mapping[str, Any],
) -> IntentTreeInitializationPackage:
    payload = _require_mapping(payload, context="intent tree initialization package")
    artifact = deserialize_artifact_spec(
        _require_mapping(payload.get("artifact"), context="package.artifact")
    )
    intent_forest = deserialize_intent_forest(
        _require_mapping(payload.get("intent_forest"), context="package.intent_forest")
    )
    conversation_state = deserialize_conversation_state(
        _require_mapping(payload.get("conversation_state"), context="package.conversation_state"),
        artifact=artifact,
        intent_forest=intent_forest,
    )
    return IntentTreeInitializationPackage(
        artifact=artifact,
        intent_forest=intent_forest,
        initial_request=_require_string(payload.get("initial_request"), context="initial_request"),
        initial_discovered_root_ids=list(payload.get("initial_discovered_root_ids", [])),
        conversation_state=conversation_state,
        seed=int(payload.get("seed", 0)),
        audit=deserialize_intent_tree_build_audit(
            _require_mapping(payload.get("audit"), context="package.audit")
        ),
    )


def llm_transcript_rows_from_build(
    package: IntentTreeInitializationPackage,
) -> list[dict[str, Any]]:
    if not isinstance(package, IntentTreeInitializationPackage):
        raise TypeError("package must be an IntentTreeInitializationPackage.")
    artifact_id = package.artifact.artifact_id
    rows: list[dict[str, Any]] = []
    for stage_record in _build_stage_records(package.audit):
        rows.append(
            {
                "artifact_id": artifact_id,
                "phase": "intent_tree_build",
                "stage_name": stage_record.stage_name,
                "prompt_name": stage_record.prompt_name,
                "rendered_prompt": stage_record.rendered_prompt,
                "llm_text_output": stage_record.llm_text_output,
                "llm_raw_output": to_jsonable(stage_record.llm_raw_output),
                "parsed_output": to_jsonable(stage_record.parsed_output),
            }
        )
    return rows


def turn_comparison_rows_from_simulation(
    result: ConversationSimulationResult,
) -> list[dict[str, Any]]:
    if not isinstance(result, ConversationSimulationResult):
        raise TypeError("result must be a ConversationSimulationResult.")
    artifact_id = result.final_state.artifact.artifact_id
    rows: list[dict[str, Any]] = []
    for turn in result.turn_runs:
        chosen = next(
            run for run in turn.candidate_runs if run.candidate_name == turn.chosen_candidate_name
        )
        rejected = next(
            run
            for run in turn.candidate_runs
            if run.candidate_name == turn.rejected_candidate_name
        )
        rows.append(
            {
                "artifact_id": artifact_id,
                "turn_index": turn.turn_index,
                "prompt_messages": to_jsonable(turn.prompt_messages),
                "active_root_id": turn.candidate_runs[0].evaluation_run.active_root_id,
                "chosen_candidate_name": turn.chosen_candidate_name,
                "rejected_candidate_name": turn.rejected_candidate_name,
                "chosen_response": chosen.generation_run.assistant_message.content,
                "rejected_response": rejected.generation_run.assistant_message.content,
                "chosen_reward": to_jsonable(chosen.reward),
                "rejected_reward": to_jsonable(rejected.reward),
                "reward_delta": chosen.reward.total_reward - rejected.reward.total_reward,
                "evaluation_scope_audit": to_jsonable(chosen.evaluation_run.scope_audit),
                "next_user_message": (
                    turn.next_user_generation.emitted_message.content
                    if turn.next_user_generation is not None
                    else None
                ),
            }
        )
    return rows


def llm_transcript_rows_from_simulation(
    result: ConversationSimulationResult,
) -> list[dict[str, Any]]:
    if not isinstance(result, ConversationSimulationResult):
        raise TypeError("result must be a ConversationSimulationResult.")
    artifact_id = result.final_state.artifact.artifact_id
    rows: list[dict[str, Any]] = []
    for turn in result.turn_runs:
        for candidate_run in turn.candidate_runs:
            generation_run = candidate_run.generation_run
            rows.append(
                {
                    "artifact_id": artifact_id,
                    "phase": "simulation",
                    "module": "assistant_candidate_generation",
                    "turn_index": turn.turn_index,
                    "candidate_name": candidate_run.candidate_name,
                    "prompt_messages": generation_run.prompt_messages_json,
                    "system_prompt_name": generation_run.system_prompt_name,
                    "system_prompt_text": generation_run.system_prompt_text,
                    "llm_text_output": generation_run.llm_text_output,
                    "llm_raw_output": to_jsonable(generation_run.llm_raw_output),
                    "parsed_output": to_jsonable(generation_run.parsed_output),
                }
            )
            evaluation_run = candidate_run.evaluation_run
            rows.append(
                {
                    "artifact_id": artifact_id,
                    "phase": "simulation",
                    "module": "assistant_response_evaluation",
                    "turn_index": turn.turn_index,
                    "candidate_name": candidate_run.candidate_name,
                    "active_root_id": evaluation_run.active_root_id,
                    "chat_history_json": evaluation_run.chat_history_json,
                    "active_root_subtree_json": evaluation_run.active_root_subtree_json,
                    "rendered_prompt": evaluation_run.rendered_prompt,
                    "llm_text_output": evaluation_run.llm_text_output,
                    "llm_raw_output": to_jsonable(evaluation_run.llm_raw_output),
                    "parsed_output": to_jsonable(evaluation_run.parsed_output),
                    "normalized_output": to_jsonable(evaluation_run.normalized_output),
                    "scope_audit": to_jsonable(evaluation_run.scope_audit),
                }
            )
        if turn.next_user_generation is not None:
            user_run = turn.next_user_generation
            rows.append(
                {
                    "artifact_id": artifact_id,
                    "phase": "simulation",
                    "module": "user_response_generation",
                    "turn_index": turn.turn_index,
                    "goal_status": to_jsonable(user_run.goal_status),
                    "goal_status_payload": to_jsonable(user_run.goal_status_payload),
                    "chat_history_json": user_run.chat_history_json,
                    "rendered_prompt": user_run.rendered_prompt,
                    "llm_text_output": user_run.llm_text_output,
                    "llm_raw_output": to_jsonable(user_run.llm_raw_output),
                    "parsed_output": to_jsonable(user_run.parsed_output),
                }
            )
    return rows


def write_json(path: Path, payload: Any) -> Path:
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(dict(row)), ensure_ascii=True))
            handle.write("\n")
    return path


def write_build_outputs(
    package: IntentTreeInitializationPackage,
    output_dir: Path,
) -> BuildOutputPaths:
    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a pathlib.Path.")
    output_dir.mkdir(parents=True, exist_ok=True)
    return BuildOutputPaths(
        intent_tree_json=write_json(output_dir / "intent_tree.json", serialize_intent_forest(package.intent_forest)),
        simulator_init_json=write_json(
            output_dir / "simulator_init.json",
            serialize_intent_tree_initialization_package(package),
        ),
        builder_audit_json=write_json(
            output_dir / "builder_audit.json",
            serialize_intent_tree_build_audit(package.audit),
        ),
        raw_llm_transcripts_jsonl=write_jsonl(
            output_dir / "raw_llm_transcripts.jsonl",
            llm_transcript_rows_from_build(package),
        ),
    )


def write_simulation_outputs(
    result: ConversationSimulationResult,
    output_dir: Path,
) -> SimulationOutputPaths:
    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a pathlib.Path.")
    output_dir.mkdir(parents=True, exist_ok=True)
    return SimulationOutputPaths(
        chosen_trajectory_jsonl=write_jsonl(
            output_dir / "chosen_trajectory.jsonl",
            [to_jsonable(result.trajectory)],
        ),
        preferences_jsonl=write_jsonl(
            output_dir / "preferences.jsonl",
            [to_jsonable(item) for item in result.preferences],
        ),
        turn_comparisons_jsonl=write_jsonl(
            output_dir / "turn_comparisons.jsonl",
            turn_comparison_rows_from_simulation(result),
        ),
        simulation_result_json=write_json(
            output_dir / "simulation_result.json",
            serialize_conversation_simulation_result(result),
        ),
        raw_llm_transcripts_jsonl=write_jsonl(
            output_dir / "raw_llm_transcripts.jsonl",
            llm_transcript_rows_from_simulation(result),
        ),
    )


def write_dataset_outputs(
    *,
    output_dir: Path,
    artifact_output_dirs: Sequence[Path],
    trajectories: Sequence[Mapping[str, Any]],
    preferences: Sequence[Mapping[str, Any]],
    turn_comparisons: Sequence[Mapping[str, Any]],
    raw_llm_transcripts: Sequence[Mapping[str, Any]],
) -> DatasetOutputPaths:
    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a pathlib.Path.")
    output_dir.mkdir(parents=True, exist_ok=True)
    return DatasetOutputPaths(
        artifact_output_dirs=tuple(artifact_output_dirs),
        chosen_trajectories_jsonl=write_jsonl(output_dir / "chosen_trajectories.jsonl", trajectories),
        preferences_jsonl=write_jsonl(output_dir / "preferences.jsonl", preferences),
        turn_comparisons_jsonl=write_jsonl(
            output_dir / "turn_comparisons.jsonl",
            turn_comparisons,
        ),
        raw_llm_transcripts_jsonl=write_jsonl(
            output_dir / "raw_llm_transcripts.jsonl",
            raw_llm_transcripts,
        ),
    )


def _build_stage_records(audit: IntentTreeBuildAudit) -> tuple[PromptStageRecord, ...]:
    return (
        audit.initial_intent_synthesis,
        audit.intent_abstraction,
        audit.hierarchy_organization,
        audit.initial_user_request,
    )


def _require_mapping(payload: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{context} must be a mapping.")
    return payload


def _require_string(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string.")
    return value
