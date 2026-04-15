"""Contract-level structured parsers for Milestone 2 prompt outputs."""

from __future__ import annotations

import re
from typing import Any

from discoverllm.parsers.common import StructuredParseError, extract_fenced_payload
from discoverllm.parsers.json_parser import parse_json_with_repair
from discoverllm.parsers.yaml_parser import parse_yaml_with_repair
from discoverllm.parsers.validation import (
    require_keys,
    require_mapping,
    validate_evaluator_children_consistency,
    validate_single_awareness_bucket,
    validate_tree_nodes,
)
from discoverllm.types import EvaluationResult, NodeEvaluation

_MARKDOWN_SECTION_PATTERN = re.compile(r"(?im)^#{1,6}\s*(thought|response)\s*$")


def parse_initial_intent_synthesis_output(raw_text: str) -> dict[str, Any]:
    payload = _parse_mapping(raw_text, context="initial_intent_synthesis")
    require_keys(
        payload,
        ("artifact_topic", "description", "checklist"),
        context="initial_intent_synthesis",
    )
    _require_non_empty_string(payload["artifact_topic"], "artifact_topic")
    _require_non_empty_string(payload["description"], "description")
    _require_non_empty_string_list(payload["checklist"], "checklist")
    return payload


def parse_intent_abstraction_output(raw_text: str) -> dict[str, Any]:
    payload = _parse_mapping(raw_text, context="intent_abstraction")
    criteria = payload.get("criteria", payload.get("results"))
    if not isinstance(criteria, list) or not criteria:
        raise StructuredParseError("intent_abstraction.criteria must be a non-empty list.")

    normalized: list[dict[str, Any]] = []
    for item in criteria:
        obj = require_mapping(item, context="intent_abstraction criterion")
        require_keys(
            obj,
            ("criterion_id", "num_abstractions", "abstractions"),
            context="intent_abstraction criterion",
        )
        _require_non_empty_string(obj["criterion_id"], "criterion_id")
        if not isinstance(obj["num_abstractions"], int) or obj["num_abstractions"] < 1:
            raise StructuredParseError(
                "intent_abstraction criterion.num_abstractions must be >= 1."
            )
        abstractions = obj["abstractions"]
        if not isinstance(abstractions, list) or not abstractions:
            raise StructuredParseError(
                "intent_abstraction criterion.abstractions must be a non-empty list."
            )
        normalized.append(obj)
    return {"criteria": normalized}


def parse_hierarchy_organization_output(raw_text: str) -> dict[str, Any]:
    primary_error: StructuredParseError | None = None
    for candidate_text in _hierarchy_candidate_texts(raw_text):
        try:
            payload = _parse_structured(candidate_text)
            trees = _extract_hierarchy_trees(payload)
            if not isinstance(trees, list) or not trees:
                raise StructuredParseError(
                    "hierarchy_organization.trees must be a non-empty list."
                )
            validate_tree_nodes(trees)
            return {"trees": trees}
        except StructuredParseError as exc:
            if primary_error is None:
                primary_error = exc
            continue

    if primary_error is not None:
        raise primary_error
    raise StructuredParseError("hierarchy_organization.trees must be a non-empty list.")


def parse_initial_user_request_output(raw_text: str) -> dict[str, Any]:
    payload = _parse_mapping(raw_text, context="initial_user_request")
    initial_request = payload.get("initial_request")
    _require_non_empty_string(initial_request, "initial_request")

    selected_raw = payload.get("criteria_selected_for_request")
    if selected_raw is None:
        selected_raw = payload.get("selected_criteria")
    selected_ids = _normalize_selected_criteria(selected_raw)
    return {
        "criteria_selected_for_request": selected_ids,
        "initial_request": initial_request,
    }


def parse_assistant_response_evaluation_output(raw_text: str) -> EvaluationResult:
    payload = _parse_mapping(raw_text, context="assistant_response_evaluation")
    require_keys(
        payload,
        (
            "classification_reasoning",
            "classification_label",
            "evaluations",
        ),
        context="assistant_response_evaluation",
    )
    evaluations_raw = payload["evaluations"]
    if not isinstance(evaluations_raw, list):
        raise StructuredParseError("assistant_response_evaluation.evaluations must be a list.")

    classification_label = _normalize_classification_label(payload["classification_label"])
    evaluation_type = _normalize_evaluation_type(
        payload.get("evaluation_type"),
        classification_label=classification_label,
    )

    normalized_entries: list[dict[str, Any]] = []
    evaluations: list[NodeEvaluation] = []
    for item in evaluations_raw:
        obj = require_mapping(item, context="assistant_response_evaluation entry")
        require_keys(
            obj,
            (
                "node_id",
                "node_text",
                "reasoning",
                "is_satisfied_or_probed",
                "children_evaluated",
            ),
            context="assistant_response_evaluation entry",
        )
        engaged_raw = _normalize_boolean(
            obj["is_satisfied_or_probed"],
            "assistant_response_evaluation entry.is_satisfied_or_probed",
        )
        children_raw = _normalize_boolean(
            obj["children_evaluated"],
            "assistant_response_evaluation entry.children_evaluated",
        )
        near_miss = obj.get("near_miss", obj.get("near_misses", []))
        if near_miss is None:
            near_miss = []
        normalized_entries.append(
            {
                "node_id": str(obj["node_id"]),
                "is_satisfied_or_probed": engaged_raw,
                "children_evaluated": children_raw,
            }
        )
        evaluations.append(
            NodeEvaluation(
                node_id=str(obj["node_id"]),
                node_text=str(obj["node_text"]),
                reasoning=str(obj["reasoning"]),
                is_satisfied_or_probed=engaged_raw,
                near_miss=list(near_miss) if isinstance(near_miss, list) else [str(near_miss)],
                children_evaluated=children_raw,
            )
        )

    validate_evaluator_children_consistency(normalized_entries)

    return EvaluationResult(
        classification_reasoning=str(payload["classification_reasoning"]),
        classification_label=classification_label,
        evaluation_type=evaluation_type,
        evaluations=evaluations,
    )


def parse_user_response_generation_output(raw_text: str) -> dict[str, Any]:
    payload = _parse_mapping(raw_text, context="user_response_generation")
    require_keys(
        payload,
        ("mental_note", "whats_working", "what_to_try_next", "user_message"),
        context="user_response_generation",
    )
    normalized = dict(payload)
    normalized["mental_note"] = _require_non_empty_string(payload["mental_note"], "mental_note")
    normalized["user_message"] = _require_non_empty_string(payload["user_message"], "user_message")
    normalized["whats_working"] = _require_string_list_or_block(
        payload["whats_working"], "whats_working"
    )
    normalized["what_to_try_next"] = _require_string_list_or_block(
        payload["what_to_try_next"], "what_to_try_next"
    )
    validate_single_awareness_bucket(payload)
    return normalized


def parse_artifact_satisfaction_judge_output(raw_text: str) -> dict[str, Any]:
    payload = _parse_mapping(raw_text, context="artifact_satisfaction_judge")
    entries = payload.get("leaf_evaluations", payload.get("evaluations"))
    if not isinstance(entries, list) or not entries:
        raise StructuredParseError(
            "artifact_satisfaction_judge.leaf_evaluations must be non-empty list."
        )
    for item in entries:
        obj = require_mapping(item, context="leaf_evaluation")
        require_keys(obj, ("requirement_id", "reasoning", "score"), context="leaf_evaluation")
        _require_non_empty_string(obj["requirement_id"], "requirement_id")
        _require_non_empty_string(obj["reasoning"], "reasoning")
        score = obj["score"]
        if not isinstance(score, int) or not 1 <= score <= 5:
            raise StructuredParseError("leaf_evaluation.score must be an integer in [1, 5].")
    return {"leaf_evaluations": entries}


def parse_interactivity_judge_output(raw_text: str) -> dict[str, Any]:
    payload = _parse_mapping(raw_text, context="interactivity_judge")
    require_keys(payload, ("thought", "interactivity"), context="interactivity_judge")
    _require_non_empty_string(payload["thought"], "thought")
    score = payload["interactivity"]
    if not isinstance(score, (int, float)) or not 1 <= float(score) <= 3:
        raise StructuredParseError("interactivity must be numeric in [1, 3].")
    return {"thought": payload["thought"], "interactivity": float(score)}


def parse_synthesis_assistant_output(raw_text: str) -> dict[str, str]:
    """Parse the collaborative synthesis-assistant markdown output format."""

    if not isinstance(raw_text, str) or not raw_text.strip():
        raise StructuredParseError("synthesis_assistant output must be a non-empty string.")

    text = raw_text.strip()
    matches = list(_MARKDOWN_SECTION_PATTERN.finditer(text))
    if not matches:
        return {"thought": "", "response": text}

    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        name = match.group(1).lower()
        if name in sections:
            raise StructuredParseError(
                f"synthesis_assistant output contains duplicate {name!r} sections."
            )
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[name] = text[start:end].strip()

    response = sections.get("response")
    if not response:
        raise StructuredParseError(
            "synthesis_assistant output must contain a non-empty Response section."
        )
    thought = sections.get("thought", "")
    return {"thought": thought, "response": response}


def _parse_mapping(raw_text: str, *, context: str) -> dict[str, Any]:
    parsed = _parse_structured(raw_text)
    return require_mapping(parsed, context=context)


def _parse_structured(raw_text: str) -> Any:
    fenced = extract_fenced_payload(raw_text, languages=("json", "yaml", "yml"))
    if fenced is not None:
        text = fenced.text.strip()
        source = fenced.source.lower()
    else:
        if not isinstance(raw_text, str):
            raise TypeError("raw_text must be a string.")
        text = raw_text.strip()
        if not text:
            raise StructuredParseError("Empty model output.")
        source = "raw"

    if source == "fenced:yaml" or source == "fenced:yml":
        try:
            return parse_yaml_with_repair(text)
        except StructuredParseError:
            return parse_json_with_repair(text)

    if source == "fenced:json" or text.startswith("{") or text.startswith("["):
        try:
            return parse_json_with_repair(text)
        except StructuredParseError:
            return parse_yaml_with_repair(text)

    try:
        return parse_yaml_with_repair(text)
    except StructuredParseError:
        return parse_json_with_repair(text)


def _extract_hierarchy_trees(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload

    obj = require_mapping(payload, context="hierarchy_organization")
    direct_trees = obj.get("trees", obj.get("hierarchy"))
    if direct_trees is not None:
        return direct_trees

    if _looks_like_tree_node(obj):
        return [obj]

    for key in ("output", "result", "response"):
        nested = obj.get(key)
        if isinstance(nested, list):
            return nested
        if isinstance(nested, dict):
            nested_trees = nested.get("trees", nested.get("hierarchy"))
            if nested_trees is not None:
                return nested_trees
            if _looks_like_tree_node(nested):
                return [nested]

    raise StructuredParseError("hierarchy_organization must contain a tree list.")


def _looks_like_tree_node(value: Any) -> bool:
    return isinstance(value, dict) and {"id", "text", "children"} <= set(value.keys())


def _hierarchy_candidate_texts(raw_text: str) -> list[str]:
    text = raw_text.strip()
    if not text:
        return [raw_text]

    candidates: list[str] = [raw_text]
    seen: set[str] = {text}
    lines = raw_text.splitlines()

    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith(
            ("step_by_step:", "hierarchy:", "trees:", "- id:", "id:")
        ):
            continue
        candidate = "\n".join(lines[index:]).strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        candidates.append(candidate)

    return candidates


def _require_non_empty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StructuredParseError(f"{name} must be a non-empty string.")
    return value


def _require_non_empty_string_list(
    value: Any, name: str, *, allow_empty: bool = False
) -> list[str]:
    if not isinstance(value, list):
        raise StructuredParseError(f"{name} must be a list of non-empty strings.")
    if not allow_empty and not value:
        raise StructuredParseError(f"{name} must not be empty.")
    result: list[str] = []
    for item in value:
        result.append(_require_non_empty_string(item, name))
    return result


def _require_string_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list):
        raise StructuredParseError(f"{name} must be a list of strings.")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise StructuredParseError(f"{name} must contain only strings.")
        result.append(item)
    return result


def _require_string_list_or_block(value: Any, name: str) -> list[str]:
    if isinstance(value, str):
        lines = [line.strip() for line in value.splitlines() if line.strip()]
        if not lines:
            raise StructuredParseError(f"{name} must be a non-empty string or list of strings.")
        return lines
    return _require_string_list(value, name)


def _normalize_selected_criteria(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        raise StructuredParseError("criteria_selected_for_request must not be empty.")
    selected_ids: list[str] = []
    for item in value:
        if isinstance(item, str):
            selected_ids.append(_require_non_empty_string(item, "criteria_selected_for_request"))
            continue
        obj = require_mapping(item, context="selected_criteria entry")
        criterion_id = obj.get("criterion_id", obj.get("id"))
        selected_ids.append(
            _require_non_empty_string(criterion_id, "criteria_selected_for_request")
        )
    return selected_ids


def _normalize_classification_label(value: Any) -> str:
    if not isinstance(value, str):
        raise StructuredParseError("classification_label must be a string.")
    normalized = re.sub(r"[\s_-]+", " ", value.strip().lower())
    if "artifact" in normalized:
        return "artifact"
    if normalized in {"dialog act", "dialogue act"}:
        return "dialog_act"
    if "dialog" in normalized and "artifact" not in normalized:
        return "dialog_act"
    raise StructuredParseError("classification_label must be 'artifact' or 'dialog_act'.")


def _normalize_evaluation_type(value: Any, *, classification_label: str | None = None) -> str:
    inferred: str | None = None
    if classification_label == "artifact":
        inferred = "satisfaction"
    elif classification_label == "dialog_act":
        inferred = "probing"

    if value is None:
        if inferred is not None:
            return inferred
        raise StructuredParseError("evaluation_type must be a string.")
    if not isinstance(value, str):
        raise StructuredParseError("evaluation_type must be a string.")

    normalized = re.sub(r"[\s_-]+", " ", value.strip().lower())
    if normalized == "satisfaction" or "satisf" in normalized or normalized == "artifact":
        return "satisfaction"
    if normalized == "probing" or "prob" in normalized:
        return "probing"
    if normalized in {"dialog act", "dialogue act"}:
        return "probing"
    if inferred is not None and normalized == classification_label.replace("_", " "):
        return inferred
    raise StructuredParseError("evaluation_type must be 'satisfaction' or 'probing'.")


def _normalize_boolean(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y", "1"}:
            return True
        if normalized in {"false", "no", "n", "0"}:
            return False
    raise StructuredParseError(f"{name} must be boolean.")
