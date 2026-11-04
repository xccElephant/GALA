"""Voxel-based collision evaluation for point clouds.

Delegates core point-cloud voxelization and collision computation to
``utils.pointcloud`` (shared with ``PlacementValidator``).
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.pointcloud import (
    compute_pointcloud_collision,
    load_ply_points_cached as _load_points_cached,
    place_new_object,
    voxelize_pointcloud as _voxelize,
)


def evaluate_collision(
    new_obj_pcd_path: str,
    predicted_pose: list[float],
    existing_obj_pcd_paths: list[str],
    voxel_size: float = 0.02,
    overlap_threshold: int = 5,
    overlap_ratio_threshold: float = 0.20,
    target_size: list[float] | None = None,
) -> dict:
    """Detect collision between the placed new object and existing objects."""
    raw_pts = _load_points_cached(new_obj_pcd_path)
    new_pts = place_new_object(
        raw_pts, predicted_pose[:3], predicted_pose[3:6], target_size=target_size
    )
    existing_pts_list = [_load_points_cached(p) for p in existing_obj_pcd_paths]

    res = compute_pointcloud_collision(
        new_pts=new_pts,
        existing_pts_list=existing_pts_list,
        voxel_size=voxel_size,
        overlap_threshold=overlap_threshold,
        overlap_ratio_threshold=overlap_ratio_threshold,
    )

    return {
        "has_collision": res["has_collision"],
        "overlap_voxels": res["overlap_voxels"],
        "overlap_volume": res["overlap_volume"],
        "overlap_ratio": res["overlap_ratio"],
    }
