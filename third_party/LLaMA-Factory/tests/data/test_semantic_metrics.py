import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "eval_metrics" / "semantic.py"
SPEC = importlib.util.spec_from_file_location("semantic_metrics_module", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
semantic_metrics_module = importlib.util.module_from_spec(SPEC)
sys.modules["semantic_metrics_module"] = semantic_metrics_module
SPEC.loader.exec_module(semantic_metrics_module)

evaluate_semantic = semantic_metrics_module.evaluate_semantic


def test_evaluate_semantic_marks_pose_as_fully_satisfied_from_expected_relations():
    instruction_data = {
        "boundary": [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]],
        "existing_objects": [
            {
                "id": 1,
                "category": "sofa",
                "center": [2.0, 2.0, 0.4],
                "rotation": [0.0, 0.0, 0.0],
                "size": [1.0, 1.0, 0.8],
            }
        ],
        "new_object": {
            "id": 2,
            "category": "television",
            "size": [0.4, 0.4, 0.4],
        },
        "instruction": "place the television in a suitable position.",
        "expected_relations": {
            "absolute_relations": ["in back region"],
            "relative_relations": [
                {"target_id": 1, "type": "next_to"},
                {"target_id": 1, "type": "behind"},
            ],
        },
    }

    metrics = evaluate_semantic(
        pred_pose=[2.0, 0.9, 0.2, 0.0, 0.0, 0.0],
        new_obj_size=[0.4, 0.4, 0.4],
        instruction_data=instruction_data,
    )

    assert metrics["has_constraints"] is True
    assert metrics["all_satisfied"] is True
    assert metrics["matched_count"] == 3
    assert metrics["missed_count"] == 0
    assert metrics["parse_warnings"] == []
    assert metrics["relation_source"] == "structured"


def test_evaluate_semantic_requires_expected_relations():
    instruction_data = {
        "boundary": [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]],
        "existing_objects": [],
        "new_object": {
            "id": 2,
            "category": "television",
            "size": [0.4, 0.4, 0.4],
        },
        "instruction": "place the television in back region.",
    }

    metrics = evaluate_semantic(
        pred_pose=[2.0, 0.9, 0.2, 0.0, 0.0, 0.0],
        new_obj_size=[0.4, 0.4, 0.4],
        instruction_data=instruction_data,
    )

    assert metrics["has_constraints"] is False
    assert metrics["all_satisfied"] is False
    assert metrics["parse_failed"] is True
    assert metrics["parse_warnings"] == ["Missing expected_relations in instruction data."]


def test_evaluate_semantic_does_not_accept_relative_relation_aliases():
    instruction_data = {
        "boundary": [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]],
        "existing_objects": [
            {
                "id": 1,
                "category": "sofa",
                "center": [2.0, 2.0, 0.4],
                "rotation": [0.0, 0.0, 0.0],
                "size": [1.0, 1.0, 0.8],
            }
        ],
        "new_object": {
            "id": 2,
            "category": "television",
            "size": [0.4, 0.4, 0.4],
        },
        "instruction": "place the television in a suitable position.",
        "expected_relations": {
            "absolute_relations": [],
            "relative_relations": [
                {"target_id": 1, "type": "next to"},
            ],
        },
    }

    metrics = evaluate_semantic(
        pred_pose=[2.0, 0.9, 0.2, 0.0, 0.0, 0.0],
        new_obj_size=[0.4, 0.4, 0.4],
        instruction_data=instruction_data,
    )

    assert metrics["has_constraints"] is True
    assert metrics["all_satisfied"] is False
    assert metrics["matched_count"] == 0
    assert metrics["missed_count"] == 1
    assert metrics["parse_warnings"] == []
    assert metrics["missed_relations"] == ["next to the object (id 1)"]
    assert metrics["expected_relations"][0]["relation"] == "next to"
    assert metrics["expected_relations"][0]["text"] == "next to the object (id 1)"
    assert metrics["expected_relations"][0]["matched"] is False
