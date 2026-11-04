"""Single-object placement evaluation metrics package.

``evaluate_placement`` aggregates pose parsing, semantic relation satisfaction,
voxel collision detection, bottom-surface support/floating, pose accuracy, and
room boundary metrics.
"""

from __future__ import annotations

from typing import Any

from .accuracy import evaluate_accuracy
from .boundary import evaluate_boundary
from .collision import evaluate_collision
from .floating import evaluate_floating
from .format_check import parse_pose
from .semantic import evaluate_semantic


def evaluate_placement(
    response: str,
    instruction_data: dict[str, Any],
    new_obj_size: list[float],
    new_obj_pcd_path: str,
    existing_obj_pcd_paths: list[str],
    floor_pcd_path: str,
    gt_pose: list[float] | None = None,
    boundary: list[list[float]] | None = None,
    voxel_size: float = 0.02,
    overlap_ratio_threshold: float = 0.20,
    height_tolerance: float = 0.05,
    new_obj_pcd_size: list[float] | None = None,
) -> dict:
    """Evaluate a single predicted placement response."""
    pred_pose = parse_pose(response)
    if pred_pose is None:
        return {
            "pred_pose": None,
            "semantic": None,
            "accuracy": None,
            "collision": None,
            "floating": None,
            "boundary": None,
            "has_collision": False,
            "is_floating": False,
        }

    sem = evaluate_semantic(pred_pose, new_obj_size, instruction_data)
    acc = evaluate_accuracy(pred_pose, gt_pose) if gt_pose is not None else None
    col = evaluate_collision(
        new_obj_pcd_path,
        pred_pose,
        existing_obj_pcd_paths,
        voxel_size=voxel_size,
        overlap_ratio_threshold=overlap_ratio_threshold,
        target_size=new_obj_pcd_size,
    )
    obj_height = new_obj_size[2] if len(new_obj_size) >= 3 else None
    flt = evaluate_floating(
        new_obj_pcd_path,
        pred_pose,
        floor_pcd_path,
        existing_obj_pcd_paths,
        voxel_size=voxel_size,
        obj_height=obj_height,
        height_tolerance=height_tolerance,
        target_size=new_obj_pcd_size,
    )
    scene_boundary = boundary if boundary is not None else instruction_data.get("boundary", [])
    bnd = evaluate_boundary(pred_pose, new_obj_size, scene_boundary)

    return {
        "pred_pose": pred_pose,
        "semantic": sem,
        "accuracy": acc,
        "collision": col,
        "floating": flt,
        "boundary": bnd,
        "has_collision": col["has_collision"],
        "is_floating": flt["is_floating"],
    }
