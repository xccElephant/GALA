from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_metrics.floating import evaluate_floating
import eval_metrics.floating as floating_module


def test_evaluate_floating_marks_sunken_ground_contact_as_invalid(monkeypatch):
    object_points = np.array(
        [
            [-0.5, -0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [-0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
        ],
        dtype=float,
    )
    floor_points = np.array(
        [
            [-0.5, -0.5, 0.0],
            [-0.5, 0.5, 0.0],
            [0.5, -0.5, 0.0],
            [0.5, 0.5, 0.0],
        ],
        dtype=float,
    )

    def load_points(path):
        if path == "object.ply":
            return object_points
        if path == "floor.ply":
            return floor_points
        raise AssertionError(f"Unexpected point-cloud path: {path}")

    monkeypatch.setattr(floating_module, "_load_points_cached", load_points)

    metrics = evaluate_floating(
        new_obj_pcd_path="object.ply",
        predicted_pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        floor_pcd_path="floor.ply",
        existing_obj_pcd_paths=[],
        voxel_size=0.02,
        gap_threshold=0.05,
        obj_height=1.0,
        ground_z=0.0,
        height_tolerance=0.05,
    )

    assert metrics["ground_contact"] is False
    assert metrics["is_sinking"] is True
    assert metrics["is_floating"] is True
