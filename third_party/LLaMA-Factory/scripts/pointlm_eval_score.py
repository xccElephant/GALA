"""
Evaluate PointLM predictions on an evaluation dataset with physical and semantic metrics.

Usage:
    python scripts/pointlm_eval_score.py \
        --predictions results/predictions.json \
        --data_file /path/to/eval.json \
        -o results/scored.json \
        --summary_output results/summary.json \
        --num_workers 32
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from eval_metrics import evaluate_placement
from eval_metrics.format_check import parse_pose


def _parse_instruction(instruction_text: str) -> dict | None:
    """Extract scene metadata from the instruction field."""
    try:
        return json.loads(instruction_text)
    except (json.JSONDecodeError, TypeError):
        return None


def _map_point_clouds(
    point_clouds: list[str],
    instruction_data: dict,
) -> dict:
    """Map point cloud paths to semantic roles: [floor, existing_0, ..., new_object]."""
    n = len(point_clouds)
    existing_objects = instruction_data.get("existing_objects", [])
    n_existing = len(existing_objects)

    expected = 1 + n_existing + 1
    if n != expected:
        return {
            "floor": point_clouds[0] if n >= 1 else None,
            "existing": point_clouds[1:-1] if n >= 3 else [],
            "new_object": point_clouds[-1] if n >= 2 else None,
        }

    return {
        "floor": point_clouds[0],
        "existing": point_clouds[1 : 1 + n_existing],
        "new_object": point_clouds[-1],
    }


def _score_sample(item: tuple) -> dict:
    """Evaluate a single sample. Designed for Pool.map."""
    sample, voxel_size, overlap_ratio_threshold, height_tolerance = item

    sample_id = sample["sample_id"]
    gt_text = sample.get("gt_output", "")
    gt_pose = parse_pose(gt_text)
    if gt_pose is None:
        return {
            "sample_id": sample_id,
            "error": f"Cannot parse GT pose: {gt_text!r}",
            "metrics": None,
        }

    instruction_data = _parse_instruction(sample["instruction"])
    if instruction_data is None:
        return {
            "sample_id": sample_id,
            "error": "Cannot parse instruction JSON",
            "metrics": None,
        }
    if sample.get("expected_relations") is not None:
        instruction_data["expected_relations"] = sample["expected_relations"]

    pcd_map = _map_point_clouds(
        sample.get("point_clouds", []),
        instruction_data,
    )

    if pcd_map["floor"] is None or pcd_map["new_object"] is None:
        return {
            "sample_id": sample_id,
            "error": "Missing floor or new_object point cloud",
            "metrics": None,
        }

    new_obj_info = instruction_data.get("new_object", {})
    new_obj_size = new_obj_info.get("size")
    predicted_size = sample.get("predicted_size")
    size_parse_error = sample.get("size_parse_error")

    if size_parse_error is not None:
        metrics = evaluate_placement(
            response="",
            instruction_data=instruction_data,
            new_obj_size=new_obj_size,
            new_obj_pcd_path=pcd_map["new_object"],
            existing_obj_pcd_paths=pcd_map["existing"],
            floor_pcd_path=pcd_map["floor"],
            gt_pose=gt_pose,
            boundary=instruction_data.get("boundary"),
            voxel_size=voxel_size,
            overlap_ratio_threshold=overlap_ratio_threshold,
            height_tolerance=height_tolerance,
        )
        metrics["size_parse_error"] = size_parse_error
    else:
        effective_new_obj_size = predicted_size if predicted_size is not None else new_obj_size
        metrics = evaluate_placement(
            response=sample.get("response", ""),
            instruction_data=instruction_data,
            new_obj_size=effective_new_obj_size,
            new_obj_pcd_path=pcd_map["new_object"],
            existing_obj_pcd_paths=pcd_map["existing"],
            floor_pcd_path=pcd_map["floor"],
            gt_pose=gt_pose,
            boundary=instruction_data.get("boundary"),
            voxel_size=voxel_size,
            overlap_ratio_threshold=overlap_ratio_threshold,
            height_tolerance=height_tolerance,
            new_obj_pcd_size=predicted_size,
        )

    return {
        "sample_id": sample_id,
        "instruction": sample["instruction"],
        "system": sample.get("system", ""),
        "gt_output": gt_text,
        "response": sample.get("response", ""),
        "predicted_size": predicted_size,
        "point_clouds": sample.get("point_clouds", []),
        "expected_relations": sample.get("expected_relations"),
        "metrics": metrics,
    }


def _sample_has_point_clouds(sample: dict) -> bool:
    point_clouds = sample.get("point_clouds")
    return isinstance(point_clouds, list) and len(point_clouds) > 0


def _candidate_pointcloud_data_files(data_file: str) -> list[str]:
    dirname = os.path.dirname(data_file)
    basename = os.path.basename(data_file)
    candidates = []
    replacements = [
        ("_without_desc_", "_without_desc_separate_pointcloud_"),
        ("_with_desc_", "_with_desc_separate_pointcloud_"),
        ("_only_pointcloud_", "_separate_pointcloud_"),
    ]
    for old, new in replacements:
        if old in basename and new not in basename:
            candidates.append(os.path.join(dirname, basename.replace(old, new, 1)))
    if "_separate_pointcloud_" not in basename and basename.startswith("3dish_"):
        candidates.append(
            os.path.join(dirname, basename.replace("3dish_", "3dish_separate_pointcloud_", 1))
        )
    return candidates


def _datasets_align_by_sample_id(primary: list[dict], candidate: list[dict]) -> bool:
    primary_ids = [sample.get("sample_id") for sample in primary]
    candidate_ids = [sample.get("sample_id") for sample in candidate]
    return primary_ids == candidate_ids


def _resolve_scoring_data(args: argparse.Namespace, data: list[dict]) -> tuple[str, list[dict]]:
    if args.score_data_file:
        score_data_file = os.path.abspath(args.score_data_file)
        with open(score_data_file, "r", encoding="utf-8") as f:
            score_data = json.load(f)
        if not _datasets_align_by_sample_id(data, score_data):
            raise ValueError(
                "score_data_file does not align with data_file by sample_id: "
                f"{args.data_file} vs {score_data_file}"
            )
        return score_data_file, score_data

    if all(_sample_has_point_clouds(sample) for sample in data):
        return os.path.abspath(args.data_file), data

    for candidate_file in _candidate_pointcloud_data_files(os.path.abspath(args.data_file)):
        if not os.path.exists(candidate_file):
            continue
        with open(candidate_file, "r", encoding="utf-8") as f:
            score_data = json.load(f)
        if _datasets_align_by_sample_id(data, score_data):
            print(f"Auto-resolved scoring dataset with point clouds: {candidate_file}")
            return candidate_file, score_data

    return os.path.abspath(args.data_file), data


def _extract_prompt(sample: dict) -> str | None:
    if "instruction" in sample:
        return sample["instruction"]
    if "conversations" in sample and sample["conversations"]:
        return sample["conversations"][0].get("value")
    return None


def _extract_gt(sample: dict) -> str:
    if "output" in sample:
        return sample.get("output", "")
    if "conversations" in sample and len(sample["conversations"]) > 1:
        return sample["conversations"][1].get("value", "")
    return ""


def _extract_predicted_size(pred: dict) -> tuple[list[float] | None, str | None]:
    if "predict_size" not in pred:
        return None, f"Prediction sample_id={pred.get('sample_id')} is missing predict_size"

    size = pred["predict_size"]
    if not isinstance(size, list) or len(size) != 3:
        return None, f"Prediction sample_id={pred.get('sample_id')} has invalid predict_size: {size!r}"

    try:
        values = [float(v) for v in size]
    except (TypeError, ValueError):
        return None, f"Prediction sample_id={pred.get('sample_id')} has non-numeric predict_size: {size!r}"
    if any(v <= 0 for v in values):
        return None, f"Prediction sample_id={pred.get('sample_id')} has non-positive predict_size: {size!r}"
    return values, None


def _build_scoring_samples(
    data: list[dict],
    predictions: list[dict],
    use_predicted_size: bool = False,
) -> tuple[list[dict], dict]:
    pred_by_id = {}
    duplicate_prediction_ids = 0
    for pred in predictions:
        sid = pred.get("sample_id")
        if sid is None:
            continue
        if sid in pred_by_id:
            duplicate_prediction_ids += 1
        pred_by_id[sid] = pred

    scoring_samples = []
    data_ids = set()
    missing_predictions = 0
    invalid_source_samples = 0

    for sample in data:
        sid = sample.get("sample_id")
        if sid is None:
            invalid_source_samples += 1
            continue

        data_ids.add(sid)
        pred = pred_by_id.get(sid)
        if pred is None:
            missing_predictions += 1
            continue

        prompt = _extract_prompt(sample)
        if prompt is None:
            invalid_source_samples += 1
            continue

        entry = {
            "sample_id": sid,
            "instruction": prompt,
            "system": sample.get("system", "You are a helpful assistant."),
            "gt_output": _extract_gt(sample),
            "response": pred.get("predict", ""),
            "point_clouds": sample.get("point_clouds", pred.get("point_clouds", [])),
            "expected_relations": sample.get("expected_relations"),
        }
        if use_predicted_size:
            predicted_size, size_error = _extract_predicted_size(pred)
            entry["predicted_size"] = predicted_size
            if size_error is not None:
                entry["size_parse_error"] = size_error

        scoring_samples.append(entry)

    extra_predictions = sum(1 for sid in pred_by_id if sid not in data_ids)
    stats = {
        "dataset_samples": len(data),
        "prediction_rows": len(predictions),
        "matched_samples": len(scoring_samples),
        "missing_predictions": missing_predictions,
        "extra_predictions": extra_predictions,
        "duplicate_prediction_ids": duplicate_prediction_ids,
        "invalid_source_samples": invalid_source_samples,
    }
    return scoring_samples, stats


def _count_relative_relation_type(expected_relations: dict | None, relation_type: str) -> int:
    if not isinstance(expected_relations, dict):
        return 0

    relative_relations = expected_relations.get("relative_relations", [])
    if not isinstance(relative_relations, list):
        return 0

    count = 0
    normalized_type = relation_type.strip().lower()
    for item in relative_relations:
        if not isinstance(item, dict):
            continue
        if str(item.get("type", "")).strip().lower() == normalized_type:
            count += 1
    return count


def _count_matched_semantic_relation_type(semantic: dict | None, relation_type: str) -> int:
    if not isinstance(semantic, dict):
        return 0

    evaluation_details = semantic.get("expected_relations", [])
    if not isinstance(evaluation_details, list):
        return 0

    count = 0
    normalized_type = relation_type.strip().lower()
    for item in evaluation_details:
        if not isinstance(item, dict):
            continue
        if str(item.get("relation", "")).strip().lower() == normalized_type and bool(item.get("matched")):
            count += 1
    return count


def _summarize(scored: list[dict], prep_stats: dict, args: argparse.Namespace) -> dict:
    errors = 0
    format_failures = 0
    size_format_failures = 0
    posed_samples = 0
    collisions = 0
    floats = 0
    semantic_all_satisfied = 0
    semantic_parse_warning = 0
    semantic_expected = 0
    semantic_matched = 0
    semantic_constrained = 0
    strict_valid = 0
    within_samples = 0
    within_expected = 0
    within_matched = 0
    within_strict_valid = 0
    support_ratios = []

    for sample in scored:
        if "error" in sample:
            errors += 1
            continue

        metrics = sample["metrics"]
        if metrics.get("size_parse_error"):
            size_format_failures += 1
        within_expected_in_sample = _count_relative_relation_type(sample.get("expected_relations"), "within")
        is_within_sample = within_expected_in_sample > 0
        if is_within_sample:
            within_samples += 1
            within_expected += within_expected_in_sample

        sem = metrics.get("semantic")
        semantic_ok = True
        if sem:
            if sem.get("has_constraints"):
                semantic_constrained += 1
                semantic_ok = bool(sem.get("all_satisfied"))
            if sem.get("all_satisfied"):
                semantic_all_satisfied += 1
            if sem.get("parse_warnings"):
                semantic_parse_warning += 1
            semantic_expected += int(sem.get("expected_count", 0))
            semantic_matched += int(sem.get("matched_count", 0))
            if is_within_sample:
                within_matched += _count_matched_semantic_relation_type(sem, "within")

        if metrics.get("pred_pose") is None:
            format_failures += 1
            continue

        posed_samples += 1
        if metrics.get("has_collision"):
            collisions += 1
        if metrics.get("is_floating"):
            floats += 1
        floating = metrics.get("floating")
        if floating and "support_ratio" in floating:
            support_ratios.append(float(floating["support_ratio"]))

        is_strict_valid = (
            metrics.get("pred_pose") is not None
            and not metrics.get("has_collision", False)
            and not metrics.get("is_floating", False)
            and semantic_ok
        )
        if is_strict_valid:
            strict_valid += 1
            if is_within_sample:
                within_strict_valid += 1

    total_samples = len(scored)
    summary = {
        "predictions_path": os.path.abspath(args.predictions),
        "data_file": os.path.abspath(args.resolved_score_data_file),
        "inference_data_file": os.path.abspath(args.data_file),
        "size_protocol": "predicted" if args.use_predicted_size else "ground_truth",
        "dataset_samples": prep_stats["dataset_samples"],
        "prediction_rows": prep_stats["prediction_rows"],
        "matched_samples": prep_stats["matched_samples"],
        "missing_predictions": prep_stats["missing_predictions"],
        "extra_predictions": prep_stats["extra_predictions"],
        "duplicate_prediction_ids": prep_stats["duplicate_prediction_ids"],
        "invalid_source_samples": prep_stats["invalid_source_samples"],
        "scored_samples": total_samples,
        "errors": errors,
        "format_failures": format_failures,
        "format_failure_rate": format_failures / max(total_samples, 1),
        "size_format_failures": size_format_failures,
        "size_format_failure_rate": size_format_failures / max(total_samples, 1),
        # Physical-plausibility rates are conditioned on a parseable pose;
        # the validity rates further below range over all samples.
        "posed_samples": posed_samples,
        "collisions": collisions,
        "collision_rate": collisions / max(posed_samples, 1),
        "floating": floats,
        "floating_rate": floats / max(posed_samples, 1),
        "support_ratio_mean": float(np.mean(support_ratios)) if support_ratios else None,
        "semantic_constrained": semantic_constrained,
        "semantic_all_satisfied": semantic_all_satisfied,
        "semantic_pass_rate": semantic_all_satisfied / max(semantic_constrained, 1),
        "semantic_parse_warning": semantic_parse_warning,
        "semantic_warning_rate": semantic_parse_warning / max(total_samples, 1),
        "semantic_expected": semantic_expected,
        "semantic_matched": semantic_matched,
        "semantic_match_rate": semantic_matched / max(semantic_expected, 1),
        "within_samples": within_samples,
        "within_expected": within_expected,
        "within_matched": within_matched,
        "within_match_rate": within_matched / max(within_expected, 1),
        "within_strict_valid": within_strict_valid,
        "within_strict_valid_rate": within_strict_valid / max(within_samples, 1),
        "strict_valid": strict_valid,
        "strict_valid_rate": strict_valid / max(total_samples, 1),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate PointLM predictions with physical and semantic metrics")
    parser.add_argument("--predictions", type=str, required=True, help="Predictions JSON from pointlm_infer.py")
    parser.add_argument("--data_file", type=str, required=True, help="Original eval dataset JSON")
    parser.add_argument("--score_data_file", type=str, default=None, help="Optional scoring dataset JSON with point_clouds")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output evaluated JSON")
    parser.add_argument("--summary_output", type=str, default=None, help="Optional output summary JSON")
    parser.add_argument("--voxel_size", type=float, default=0.02)
    parser.add_argument("--overlap_ratio_threshold", type=float, default=0.20)
    parser.add_argument("--height_tolerance", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--use_predicted_size",
        action="store_true",
        help="Use each prediction row's predict_size for semantic, floating, and scaled point-cloud checks.",
    )
    args = parser.parse_args()

    args.data_file = os.path.abspath(args.data_file)

    print(f"Loading eval dataset from {args.data_file} ...")
    with open(args.data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    args.resolved_score_data_file, scoring_data = _resolve_scoring_data(args, data)
    if args.resolved_score_data_file != args.data_file:
        print(f"Scoring with point-cloud dataset {args.resolved_score_data_file} ...")

    print(f"Loading predictions from {args.predictions} ...")
    with open(args.predictions, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    scoring_samples, prep_stats = _build_scoring_samples(
        scoring_data,
        predictions,
        use_predicted_size=args.use_predicted_size,
    )
    print(
        f"Matched {prep_stats['matched_samples']} samples "
        f"(missing={prep_stats['missing_predictions']}, extra={prep_stats['extra_predictions']})"
    )

    work_items = [
        (sample, args.voxel_size, args.overlap_ratio_threshold, args.height_tolerance)
        for sample in scoring_samples
    ]

    if args.num_workers <= 1:
        scored = [_score_sample(item) for item in tqdm(work_items)]
    else:
        with Pool(args.num_workers) as pool:
            scored = list(
                tqdm(
                    pool.imap(_score_sample, work_items, chunksize=16),
                    total=len(work_items),
                )
            )

    summary = _summarize(scored, prep_stats, args)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(scored, f, ensure_ascii=False, indent=2)

    summary_path = args.summary_output or (os.path.splitext(args.output)[0] + "_summary.json")
    summary_dir = os.path.dirname(summary_path)
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print("Eval Summary")
    print(f"{'=' * 60}")
    print(f"  Dataset samples:      {summary['dataset_samples']}")
    print(f"  Prediction rows:      {summary['prediction_rows']}")
    print(f"  Matched samples:      {summary['matched_samples']}")
    print(f"  Size protocol:        {summary['size_protocol']}")
    print(f"  Missing predictions:  {summary['missing_predictions']}")
    print(f"  Extra predictions:    {summary['extra_predictions']}")
    print(f"  Errors:               {summary['errors']}")
    print(
        f"  Format failures:      {summary['format_failures']} "
        f"({summary['format_failure_rate'] * 100:.2f}%)"
    )
    if summary["size_protocol"] == "predicted":
        print(
            f"  Size format failures: {summary['size_format_failures']} "
            f"({summary['size_format_failure_rate'] * 100:.2f}%)"
        )
    print(f"  Posed samples:        {summary['posed_samples']}")
    print(
        f"  Collisions:           {summary['collisions']} / {summary['posed_samples']} "
        f"({summary['collision_rate'] * 100:.2f}%)"
    )
    print(
        f"  Floating:             {summary['floating']} / {summary['posed_samples']} "
        f"({summary['floating_rate'] * 100:.2f}%)"
    )
    if summary["support_ratio_mean"] is not None:
        print(f"  Support ratio mean:   {summary['support_ratio_mean'] * 100:.2f}%")
    if summary["semantic_constrained"]:
        print(
            f"  Semantic pass:        {summary['semantic_all_satisfied']} / "
            f"{summary['semantic_constrained']} "
            f"({summary['semantic_pass_rate'] * 100:.2f}%)"
        )
        print(
            f"  Semantic match:       {summary['semantic_matched']} / "
            f"{summary['semantic_expected']} "
            f"({summary['semantic_match_rate'] * 100:.2f}%)"
        )
    if summary["within_samples"]:
        print(
            f"  Within match:         {summary['within_matched']} / "
            f"{summary['within_expected']} "
            f"({summary['within_match_rate'] * 100:.2f}%)"
        )
        print(
            f"  Within strict valid:  {summary['within_strict_valid']} / "
            f"{summary['within_samples']} "
            f"({summary['within_strict_valid_rate'] * 100:.2f}%)"
        )
    print(
        f"  Strict valid:         {summary['strict_valid']} / {summary['scored_samples']} "
        f"({summary['strict_valid_rate'] * 100:.2f}%)"
    )
    print(f"{'=' * 60}")
    print(f"Evaluated results saved to {args.output}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
