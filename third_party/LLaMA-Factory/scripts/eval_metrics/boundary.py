"""Room boundary evaluation.

Delegates to ``utils.spatial_utils.get_object_boundary_overlap_metrics``
(shared with ``PlacementValidator`` and dataset filtering).
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.spatial_utils import get_object_boundary_overlap_metrics


def evaluate_boundary(
    pred_pose: list[float],
    obj_size: list[float],
    boundary: list[list[float]],
    outside_ratio_threshold: float = 0.05,
) -> dict:
    """Check whether the posed 3D OBB footprint stays within the room boundary polygon."""
    if not boundary or len(boundary) < 3:
        return {
            "within_boundary": True,
            "outside_ratio": 0.0,
            "outside_area": 0.0,
            "footprint_area": 0.0,
        }

    obj = {
        "center": [float(v) for v in pred_pose[:3]],
        "rotation": [float(v) for v in pred_pose[3:6]],
        "size": [float(v) for v in obj_size],
    }
    metrics = get_object_boundary_overlap_metrics(obj, boundary)
    outside_ratio = float(metrics["outside_ratio"])
    return {
        "within_boundary": outside_ratio <= outside_ratio_threshold,
        "outside_ratio": round(outside_ratio, 4),
        "outside_area": round(float(metrics["outside_area"]), 4),
        "footprint_area": round(float(metrics["footprint_area"]), 4),
    }
