"""物理修复结果的合理性度量(python 环境,需 trimesh/vhacdx 加载真实凸块)。

口径与 XPBD 求解器一致:用 collision_geometry 的 VHACD 凸块 XY 投影 2D-SAT 测穿插、
用 physics_repair_engine._is_hard_overlap_exempt 做硬豁免。可独立 CLI 运行。

度量:
  - overlap_pairs / max_penetration: 真实凸块 XY 互插(跳过 under/on/within/椅子塞桌下硬豁免对)
  - displacement: 相对原始布局的最大/平均/中位位移(catch 发散/全顶上限)
  - below_floor: 底部低于地面的物体数
  - boundary_violations: OBB 角点落在房间多边形外的物体数(给定 boundary 时)
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import collision_geometry as cg  # type: ignore
import physics_repair_engine as pre  # type: ignore
from physics_repair_engine import RepairedTransform  # type: ignore


def load_objects(path: str | Path) -> dict[int, dict[str, Any]]:
    """递归收集 layout 中带 id/center/size 的物体字典。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[int, dict[str, Any]] = {}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if "id" in node and "center" in node and "size" in node and isinstance(
                node["center"], list
            ):
                out[int(node["id"])] = node
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(data)
    return out


def _transform_of(obj: Mapping[str, Any]) -> RepairedTransform:
    rotation = obj.get("rotation", [0.0, 0.0, 0.0])
    return RepairedTransform(center=list(obj["center"]), rotation=list(rotation))


def overlap_metrics(objects: Mapping[int, dict[str, Any]]) -> dict[str, float]:
    obj_list = list(objects.values())
    geoms = cg.build_scene_geoms(obj_list)
    transforms = {i: _transform_of(o) for i, o in objects.items()}
    ids = sorted(geoms)
    pairs = 0
    max_pen = 0.0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            if pre._is_hard_overlap_exempt(
                objects[a], objects[b], transforms[a], transforms[b]
            ):
                continue
            contact = cg.pair_xy_penetration(geoms[a], geoms[b])
            if contact is not None:
                pairs += 1
                max_pen = max(max_pen, contact[1])
    return {"overlap_pairs": float(pairs), "max_penetration": round(max_pen, 5)}


def displacement_metrics(
    original: Mapping[int, dict[str, Any]],
    repaired: Mapping[int, dict[str, Any]],
) -> dict[str, float]:
    deltas = [
        math.dist(original[i]["center"], repaired[i]["center"])
        for i in original
        if i in repaired
    ]
    if not deltas:
        return {"disp_max": 0.0, "disp_mean": 0.0, "disp_median": 0.0}
    return {
        "disp_max": round(max(deltas), 4),
        "disp_mean": round(sum(deltas) / len(deltas), 4),
        "disp_median": round(statistics.median(deltas), 4),
    }


def below_floor_count(objects: Mapping[int, dict[str, Any]], floor_z: float, tol: float = 0.02) -> int:
    count = 0
    for obj in objects.values():
        bottom = float(obj["center"][2]) - float(obj["size"][2]) / 2.0
        if bottom < float(floor_z) - tol:
            count += 1
    return count


def boundary_violations(
    objects: Mapping[int, dict[str, Any]],
    boundary: list[list[float]] | None,
) -> int:
    if not boundary or len(boundary) < 3:
        return 0
    count = 0
    for obj in objects.values():
        rect = pre._object_rectangle(obj, _transform_of(obj))
        cx, cy = rect.center
        (ax, ay), (bx, by) = rect.axes
        hx, hy = rect.half_size
        corners = [
            (cx + sx * ax * hx + sy * bx * hy, cy + sx * ay * hx + sy * by * hy)
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
        ]
        if not all(pre._point_in_polygon_xy(px, py, boundary) for px, py in corners):
            count += 1
    return count


def compute_metrics(
    repaired_path: str | Path,
    original_path: str | Path | None = None,
    boundary: list[list[float]] | None = None,
    floor_z: float | None = None,
) -> dict[str, float]:
    repaired = load_objects(repaired_path)
    metrics: dict[str, float] = {"objects": float(len(repaired))}
    metrics.update(overlap_metrics(repaired))
    if original_path is not None:
        metrics.update(displacement_metrics(load_objects(original_path), repaired))
    if floor_z is not None:
        metrics["below_floor"] = float(below_floor_count(repaired, floor_z))
    metrics["boundary_violations"] = float(boundary_violations(repaired, boundary))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="物理修复结果合理性度量(纯 JSON)")
    parser.add_argument("--layout", "-l", required=True, help="修复后 layout JSON")
    parser.add_argument("--original", "-o", default=None, help="原始 layout JSON(算位移)")
    parser.add_argument("--floor-z", type=float, default=None, help="地面 Z(算沉底)")
    args = parser.parse_args()
    metrics = compute_metrics(args.layout, args.original, floor_z=args.floor_z)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
