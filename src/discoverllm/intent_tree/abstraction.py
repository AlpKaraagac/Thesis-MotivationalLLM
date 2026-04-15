"""Stage-2 abstraction helpers for intent-tree construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CriterionInput:
    """Concrete criterion from stage 1 prepared for abstraction prompting."""

    criterion_id: str
    criterion: str

    def __post_init__(self) -> None:
        if not isinstance(self.criterion_id, str) or not self.criterion_id.strip():
            raise ValueError("criterion_id must be a non-empty string.")
        if not isinstance(self.criterion, str) or not self.criterion.strip():
            raise ValueError("criterion must be a non-empty string.")


def criteria_from_checklist(checklist: list[str]) -> list[CriterionInput]:
    """Convert stage-1 checklist strings into typed criterion inputs."""

    if not isinstance(checklist, list) or not checklist:
        raise ValueError("checklist must be a non-empty list of strings.")

    criteria: list[CriterionInput] = []
    for index, item in enumerate(checklist, start=1):
        if not isinstance(item, str) or not item.strip():
            raise ValueError("checklist items must be non-empty strings.")
        criteria.append(CriterionInput(criterion_id=f"c{index}", criterion=item.strip()))
    return criteria


def normalize_abstractions_for_hierarchy(
    criteria_inputs: list[CriterionInput],
    abstraction_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    """Reorder stage-2 outputs from abstract->specific and append the original leaf."""

    if not isinstance(criteria_inputs, list) or not criteria_inputs:
        raise ValueError("criteria_inputs must be a non-empty list of CriterionInput.")
    if not isinstance(abstraction_payload, dict):
        raise TypeError("abstraction_payload must be a mapping.")
    criteria_raw = abstraction_payload.get("criteria")
    if not isinstance(criteria_raw, list) or not criteria_raw:
        raise ValueError("abstraction_payload.criteria must be a non-empty list.")

    original_by_id = {item.criterion_id: item.criterion for item in criteria_inputs}
    normalized: list[dict[str, Any]] = []
    for item in criteria_raw:
        if not isinstance(item, dict):
            raise TypeError("Each abstraction entry must be a mapping.")
        criterion_id = str(item.get("criterion_id", "")).strip()
        if criterion_id not in original_by_id:
            raise ValueError(
                f"Abstraction payload referenced unknown criterion_id {criterion_id!r}."
            )
        abstractions = item.get("abstractions")
        if not isinstance(abstractions, list) or not abstractions:
            raise ValueError(
                f"Abstraction payload for {criterion_id!r} must contain a non-empty abstractions list."
            )

        reordered_levels: list[dict[str, Any]] = []
        for level_index, abstraction in enumerate(reversed(abstractions), start=1):
            if not isinstance(abstraction, dict):
                raise TypeError("Each abstraction level must be a mapping.")
            reordered_levels.append(
                {
                    "level": level_index,
                    "reasoning": str(abstraction.get("reasoning", "")).strip(),
                    "checklist": list(abstraction.get("checklist", [])),
                    "criterion": str(abstraction.get("criterion", "")).strip(),
                    "is_final": False,
                }
            )

        original_criterion = original_by_id[criterion_id]
        reordered_levels.append(
            {
                "level": len(reordered_levels) + 1,
                "reasoning": "Original concrete criterion retained as the most specific leaf.",
                "checklist": [original_criterion],
                "criterion": original_criterion,
                "is_final": True,
            }
        )
        normalized.append(
            {
                "criterion_id": criterion_id,
                "num_abstractions": len(reordered_levels),
                "abstractions": reordered_levels,
            }
        )
    return normalized
