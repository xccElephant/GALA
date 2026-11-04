"""Semantic spatial-relation evaluation.

Requires structured ``expected_relations`` from the dataset and delegates
spatial relation evaluation to ``utils.spatial_utils`` (single source of truth).
"""

from __future__ import annotations

from pathlib import Path
import re
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.constants import RELATIVE_RELATION_TEXT
from utils.spatial_utils import check_absolute_relation, check_relative_relation


def _normalize_label(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _build_relation_text(relation: dict[str, Any]) -> str:
    if relation["scope"] == "absolute":
        return relation["relation_text"]

    target_label = relation.get("target_label") or "object"
    target_id = relation.get("target_id")
    target_suffix = f" the {target_label}"
    if target_id is not None:
        target_suffix += f" (id {target_id})"

    relation_type = relation["relation"]
    relation_prefix = RELATIVE_RELATION_TEXT.get(relation_type, relation_type.replace("_", " "))
    return f"{relation_prefix}{target_suffix}"


def _build_relations_from_expected_relations(
    expected_relations: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[str], bool]:
    relations: list[dict[str, Any]] = []
    warnings: list[str] = []
    parse_failed = False

    if expected_relations is None:
        return relations, ["Missing expected_relations in instruction data."], True

    if not isinstance(expected_relations, dict):
        return relations, ["expected_relations is present but not a dict."], True

    absolute_relations = expected_relations.get("absolute_relations", [])
    relative_relations = expected_relations.get("relative_relations", [])

    if not isinstance(absolute_relations, list):
        warnings.append("expected_relations.absolute_relations is not a list.")
        parse_failed = True
    else:
        for item in absolute_relations:
            if not isinstance(item, str):
                warnings.append(f"Invalid absolute relation entry: {item!r}")
                parse_failed = True
                continue
            normalized = _normalize_label(item)
            relations.append(
                {
                    "scope": "absolute",
                    "relation": normalized,
                    "relation_text": normalized,
                }
            )

    if not isinstance(relative_relations, list):
        warnings.append("expected_relations.relative_relations is not a list.")
        parse_failed = True
    else:
        for item in relative_relations:
            if not isinstance(item, dict):
                warnings.append(f"Invalid relative relation entry: {item!r}")
                parse_failed = True
                continue
            relation_type = _normalize_label(str(item.get("type", "")))
            if not relation_type:
                warnings.append(f"Missing relative relation type in expected_relations: {item!r}")
                parse_failed = True
                continue
            target_id = item.get("target_id")
            if not isinstance(target_id, int):
                warnings.append(f"relative_relations.target_id must be int, got: {item!r}")
                parse_failed = True
                continue
            relations.append(
                {
                    "scope": "relative",
                    "relation": relation_type,
                    "target_id": target_id,
                    "target_label": "object",
                    "relation_text": f"{relation_type} target_id={target_id}",
                }
            )

    return relations, warnings, parse_failed


def _resolve_target_object(
    relation: dict[str, Any],
    existing_objects: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[str]]:
    warnings: list[str] = []
    target_id = relation.get("target_id")
    if target_id is not None:
        for obj in existing_objects:
            if obj.get("id") == target_id:
                return obj, warnings
        warnings.append(f"Target object id {target_id} not found for relation '{_build_relation_text(relation)}'.")
        return None, warnings

    target_label = _normalize_label(relation.get("target_label", ""))
    if not target_label or target_label == "object":
        warnings.append(f"Cannot resolve target object without an id for relation '{_build_relation_text(relation)}'.")
        return None, warnings

    matches = []
    for obj in existing_objects:
        candidates = {
            _normalize_label(str(obj.get("category", ""))),
            _normalize_label(str(obj.get("short_desc", ""))),
            _normalize_label(str(obj.get("name", ""))),
        }
        candidates.discard("")
        if target_label in candidates:
            matches.append(obj)

    if len(matches) == 1:
        return matches[0], warnings
    if len(matches) > 1:
        warnings.append(f"Ambiguous target label '{target_label}' for relation '{_build_relation_text(relation)}'.")
    else:
        warnings.append(f"Target label '{target_label}' not found for relation '{_build_relation_text(relation)}'.")
    return None, warnings


def evaluate_semantic(
    pred_pose: list[float],
    new_obj_size: list[float],
    instruction_data: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate spatial relation satisfaction for a predicted pose."""
    existing_objects = list(instruction_data.get("existing_objects", []))
    boundary = instruction_data.get("boundary", [])
    new_object_info = dict(instruction_data.get("new_object", {}))

    new_obj = {
        "id": new_object_info.get("id"),
        "category": new_object_info.get("category", "object"),
        "center": [float(v) for v in pred_pose[:3]],
        "rotation": [float(v) for v in pred_pose[3:6]],
        "size": [float(v) for v in new_obj_size],
    }

    expected_relations, parse_warnings, parse_failed = _build_relations_from_expected_relations(
        instruction_data.get("expected_relations")
    )
    matched_relations: list[str] = []
    missed_relations: list[str] = []
    evaluation_details: list[dict[str, Any]] = []
    extra_warnings: list[str] = []

    for relation in expected_relations:
        relation_text = _build_relation_text(relation)
        matched = False
        if relation["scope"] == "absolute":
            try:
                matched = check_absolute_relation(new_obj, boundary, relation["relation_text"])
            except Exception as exc:
                extra_warnings.append(f"Failed to evaluate absolute relation '{relation_text}': {exc}")
        else:
            target_obj, target_warnings = _resolve_target_object(relation, existing_objects)
            extra_warnings.extend(target_warnings)
            if target_obj is not None:
                try:
                    matched = check_relative_relation(new_obj, target_obj, relation["relation"])
                except Exception as exc:
                    extra_warnings.append(f"Failed to evaluate relative relation '{relation_text}': {exc}")

        if matched:
            matched_relations.append(relation_text)
        else:
            missed_relations.append(relation_text)

        evaluation_details.append(
            {
                "scope": relation["scope"],
                "relation": relation["relation"],
                "text": relation_text,
                "target_id": relation.get("target_id"),
                "matched": matched,
            }
        )

    warnings = parse_warnings + extra_warnings
    expected_count = len(expected_relations)
    matched_count = len(matched_relations)
    missed_count = len(missed_relations)
    has_constraints = expected_count > 0
    all_satisfied = (matched_count == expected_count) and not parse_failed

    return {
        "expected_relations": evaluation_details,
        "matched_relations": matched_relations,
        "missed_relations": missed_relations,
        "parse_warnings": warnings,
        "expected_count": expected_count,
        "matched_count": matched_count,
        "missed_count": missed_count,
        "has_constraints": has_constraints,
        "parse_failed": parse_failed,
        "all_satisfied": all_satisfied,
        "relation_source": "structured",
    }
