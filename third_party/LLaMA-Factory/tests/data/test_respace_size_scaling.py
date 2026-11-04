import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "eval_metrics" / "collision.py"
SPEC = importlib.util.spec_from_file_location("collision_module", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
collision_module = importlib.util.module_from_spec(SPEC)
sys.modules["collision_module"] = collision_module
SPEC.loader.exec_module(collision_module)

place_new_object = collision_module.place_new_object


def test_place_new_object_scales_bbox_to_target_size_before_placement():
    points = np.array(
        [
            [-0.5, -1.0, -1.5],
            [-0.5, -1.0, 1.5],
            [-0.5, 1.0, -1.5],
            [-0.5, 1.0, 1.5],
            [0.5, -1.0, -1.5],
            [0.5, -1.0, 1.5],
            [0.5, 1.0, -1.5],
            [0.5, 1.0, 1.5],
        ],
        dtype=float,
    )

    placed = place_new_object(
        points,
        predicted_center=[10.0, 20.0, 30.0],
        predicted_rotation_deg=[0.0, 0.0, 0.0],
        target_size=[2.0, 1.0, 6.0],
    )

    np.testing.assert_allclose(placed.max(axis=0) - placed.min(axis=0), [2.0, 1.0, 6.0])
    np.testing.assert_allclose((placed.max(axis=0) + placed.min(axis=0)) / 2.0, [10.0, 20.0, 30.0])


def test_place_new_object_scales_non_degenerate_axes_for_flat_point_clouds():
    points = np.array(
        [
            [-1.0, -2.0, 0.0],
            [-1.0, 2.0, 0.0],
            [1.0, -2.0, 0.0],
            [1.0, 2.0, 0.0],
        ],
        dtype=float,
    )

    placed = place_new_object(
        points,
        predicted_center=[0.0, 0.0, 1.0],
        predicted_rotation_deg=[0.0, 0.0, 0.0],
        target_size=[4.0, 2.0, 1.0],
    )

    np.testing.assert_allclose(placed.max(axis=0) - placed.min(axis=0), [4.0, 2.0, 0.0])
