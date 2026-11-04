"""Pose accuracy evaluation: position L2 error and per-axis roll/pitch/yaw angular error."""

from __future__ import annotations

import numpy as np


def _angular_diff(a_deg: float, b_deg: float) -> float:
    """Shortest angular distance in degrees (always non-negative)."""
    diff = (a_deg - b_deg) % 360.0
    if diff > 180.0:
        diff = 360.0 - diff
    return float(diff)


def evaluate_accuracy(pred_pose: list[float], gt_pose: list[float]) -> dict:
    """Compute position L2 distance (m) and per-axis angular errors (deg) against GT pose."""
    pred_pos = np.asarray(pred_pose[:3], dtype=float)
    gt_pos = np.asarray(gt_pose[:3], dtype=float)
    pos_error = float(np.linalg.norm(pred_pos - gt_pos))
    height_error = float(abs(pred_pos[2] - gt_pos[2]))

    roll_error = _angular_diff(pred_pose[3], gt_pose[3])
    pitch_error = _angular_diff(pred_pose[4], gt_pose[4])
    yaw_error = _angular_diff(pred_pose[5], gt_pose[5])

    return {
        "pos_error": round(pos_error, 4),
        "height_error": round(height_error, 4),
        "roll_error": round(roll_error, 4),
        "pitch_error": round(pitch_error, 4),
        "yaw_error": round(yaw_error, 4),
    }
