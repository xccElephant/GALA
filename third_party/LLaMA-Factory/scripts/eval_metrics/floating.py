"""Voxel-based floating (lack-of-support) evaluation for point clouds.

Delegates core point-cloud support height-map computation to
``utils.pointcloud`` (shared with ``PlacementValidator``).
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.pointcloud import compute_pointcloud_floating, place_new_object
from .collision import _load_points_cached


def evaluate_floating(
    new_obj_pcd_path: str,
    predicted_pose: list[float],
    floor_pcd_path: str,
    existing_obj_pcd_paths: list[str],
    voxel_size: float = 0.02,
    gap_threshold: float = 0.05,
    obj_height: float | None = None,
    ground_z: float = 0.0,
    height_tolerance: float = 0.05,
    target_size: list[float] | None = None,
) -> dict:
    """Detect whether the placed new object is floating and compute support ratio."""
    if obj_height is not None:
        expected_z = ground_z + obj_height / 2.0
        if abs(predicted_pose[2] - expected_z) <= height_tolerance:
            return {
                "is_floating": False,
                "is_sinking": False,
                "support_ratio": 1.0,
                "median_gap": 0.0,
                "ground_contact": True,
            }

    raw_pts = _load_points_cached(new_obj_pcd_path)
    new_pts = place_new_object(
        raw_pts, predicted_pose[:3], predicted_pose[3:6], target_size=target_size
    )
    floor_pts = _load_points_cached(floor_pcd_path)
    existing_pts_list = [_load_points_cached(p) for p in existing_obj_pcd_paths]

    return compute_pointcloud_floating(
        new_pts=new_pts,
        floor_pts=floor_pts,
        existing_pts_list=existing_pts_list,
        voxel_size=voxel_size,
        gap_threshold=gap_threshold,
    )
