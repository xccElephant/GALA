"""
Blender rigid-body post-process for layout visualizations.

The pure helpers in this file are importable by normal pytest runs. Blender
runtime access is kept behind _require_bpy() so the JSON/clamp logic can be
tested without a Blender Python interpreter.
"""

import contextlib
import copy
import importlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
for _p in (str(_THIS_DIR), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.semantic_relations import (  # noqa: E402 - 需先补 sys.path
    SUPPORT_RELATION_TYPES as _SUPPORT_RELATION_TYPES,
    CONTAINMENT_RELATION_TYPES as _CONTAINMENT_RELATION_TYPES,
    UNDER_RELATION_TYPES as _UNDER_RELATION_TYPES,
    all_relation_target_ids as _all_relation_target_ids,
    has_relation_to_target as _has_relation_to_target,
    relation_target_id as _expected_relation_target_id,
)


class RepairedTransform(NamedTuple):
    center: list[float]
    rotation: list[float]


class _Rectangle(NamedTuple):
    center: tuple[float, float]
    axes: tuple[tuple[float, float], tuple[float, float]]
    half_size: tuple[float, float]


_ATTACHMENT_RELATION_TYPES = _SUPPORT_RELATION_TYPES | _CONTAINMENT_RELATION_TYPES
_CHAIR_TABLE_RELATION_TYPES = {"next_to", "left", "right", "front", "behind", "in_front", "face_to"}
_CHAIR_TABLE_FOOTPRINT_XY_MARGIN = 0.15
_PARENT_SUPPORT_CHILD_KEYWORDS = {
    "alarm",
    "bag",
    "basket",
    "blanket",
    "board",
    "book",
    "bottle",
    "bowl",
    "box",
    "can",
    "clock",
    "cup",
    "glove",
    "headphone",
    "keyboard",
    "lamp",
    "laptop",
    "magazine",
    "microwave",
    "monitor",
    "mouse",
    "package",
    "phone",
    "pillow",
    "plant",
    "plate",
    "pot",
    "suitcase",
    "toolbox",
    "tray",
    "tv",
    "vase",
}


def _as_float_triplet(values: Sequence[float], name: str) -> list[float]:
    if len(values) != 3:
        raise ValueError(f"{name} 必须包含 3 个数值: {values}")
    return [float(values[0]), float(values[1]), float(values[2])]


def _round_triplet(values: Sequence[float], digits: int = 4) -> list[float]:
    return [round(float(value), digits) for value in values]


def clamp_center(
    original_center: Sequence[float],
    simulated_center: Sequence[float],
    max_xy_displacement: float,
    max_z_adjustment: float,
) -> list[float]:
    """Clamp a simulated center around its original layout position."""
    if max_xy_displacement < 0:
        raise ValueError("max_xy_displacement 不能为负数")
    if max_z_adjustment < 0:
        raise ValueError("max_z_adjustment 不能为负数")

    original = _as_float_triplet(original_center, "original_center")
    simulated = _as_float_triplet(simulated_center, "simulated_center")
    dx = simulated[0] - original[0]
    dy = simulated[1] - original[1]
    xy_distance = math.hypot(dx, dy)
    if xy_distance > max_xy_displacement and xy_distance > 0.0:
        scale = max_xy_displacement / xy_distance
        dx *= scale
        dy *= scale

    dz = simulated[2] - original[2]
    dz = max(-max_z_adjustment, min(max_z_adjustment, dz))
    return _round_triplet([original[0] + dx, original[1] + dy, original[2] + dz])


def _parse_boundary_xy_bounds(
    boundary: Sequence[Sequence[float]],
) -> tuple[float, float, float, float] | None:
    """Return (xmin, xmax, ymin, ymax) from layout floor polygon vertices."""
    if len(boundary) < 1:
        return None
    x_values: list[float] = []
    y_values: list[float] = []
    for point in boundary:
        if len(point) < 2:
            raise ValueError(f"boundary 顶点至少需要 x,y 两个分量: {point!r}")
        x_values.append(float(point[0]))
        y_values.append(float(point[1]))
    return min(x_values), max(x_values), min(y_values), max(y_values)


def compute_xy_shift_for_axis_aligned_aabb_inside_bounds(
    mins: Sequence[float],
    maxs: Sequence[float],
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> tuple[float, float]:
    """Return (dx, dy) to translate an object so its world AABB fits inside the XY bounds."""
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("boundary XY 区间退化: xmax<=xmin 或 ymax<=ymin")

    ox0 = float(mins[0])
    ox1 = float(maxs[0])
    oy0 = float(mins[1])
    oy1 = float(maxs[1])
    width_x = ox1 - ox0
    width_y = oy1 - oy0
    span_x = xmax - xmin
    span_y = ymax - ymin

    if width_x > span_x:
        center_x = (ox0 + ox1) / 2.0
        dx = (xmin + xmax) / 2.0 - center_x
    else:
        dx = 0.0
        if ox0 < xmin:
            dx += xmin - ox0
        if ox1 > xmax:
            dx += xmax - ox1

    if width_y > span_y:
        center_y = (oy0 + oy1) / 2.0
        dy = (ymin + ymax) / 2.0 - center_y
    else:
        dy = 0.0
        if oy0 < ymin:
            dy += ymin - oy0
        if oy1 > ymax:
            dy += ymax - oy1

    return dx, dy


def _point_on_segment_xy(
    x_coord: float,
    y_coord: float,
    start: Sequence[float],
    end: Sequence[float],
    tolerance: float = 1e-8,
) -> bool:
    x1 = float(start[0])
    y1 = float(start[1])
    x2 = float(end[0])
    y2 = float(end[1])
    cross = (float(x_coord) - x1) * (y2 - y1) - (float(y_coord) - y1) * (x2 - x1)
    if abs(cross) > tolerance:
        return False
    return (
        min(x1, x2) - tolerance <= float(x_coord) <= max(x1, x2) + tolerance
        and min(y1, y2) - tolerance <= float(y_coord) <= max(y1, y2) + tolerance
    )


def _point_in_polygon_xy(
    x_coord: float,
    y_coord: float,
    polygon: Sequence[Sequence[float]],
) -> bool:
    """Return True when an XY point is inside or on the boundary of a polygon."""
    if len(polygon) < 3:
        return False
    x = float(x_coord)
    y = float(y_coord)
    inside = False
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        if _point_on_segment_xy(x, y, start, end):
            return True
        x1 = float(start[0])
        y1 = float(start[1])
        x2 = float(end[0])
        y2 = float(end[1])
        if (y1 > y) == (y2 > y):
            continue
        crossing_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
        if x < crossing_x:
            inside = not inside
    return inside


def _aabb_xy_corners_after_shift(
    mins: Sequence[float],
    maxs: Sequence[float],
    dx: float,
    dy: float,
) -> list[tuple[float, float]]:
    min_x = float(mins[0]) + dx
    max_x = float(maxs[0]) + dx
    min_y = float(mins[1]) + dy
    max_y = float(maxs[1]) + dy
    mid_x = (min_x + max_x) / 2.0
    mid_y = (min_y + max_y) / 2.0
    return [
        (min_x, min_y),
        (min_x, mid_y),
        (min_x, max_y),
        (mid_x, min_y),
        (mid_x, mid_y),
        (mid_x, max_y),
        (max_x, min_y),
        (max_x, mid_y),
        (max_x, max_y),
    ]


def _aabb_inside_polygon_after_shift(
    mins: Sequence[float],
    maxs: Sequence[float],
    boundary: Sequence[Sequence[float]],
    dx: float,
    dy: float,
) -> bool:
    return all(
        _point_in_polygon_xy(x_coord, y_coord, boundary)
        for x_coord, y_coord in _aabb_xy_corners_after_shift(mins, maxs, dx, dy)
    )


def compute_xy_shift_for_axis_aligned_aabb_inside_polygon(
    mins: Sequence[float],
    maxs: Sequence[float],
    boundary: Sequence[Sequence[float]],
) -> tuple[float, float]:
    """Return a small XY shift that places an AABB footprint inside a possibly concave polygon."""
    if len(boundary) < 3:
        raw_bounds = _parse_boundary_xy_bounds(boundary)
        if raw_bounds is None:
            return 0.0, 0.0
        return compute_xy_shift_for_axis_aligned_aabb_inside_bounds(mins, maxs, *raw_bounds)

    if _aabb_inside_polygon_after_shift(mins, maxs, boundary, 0.0, 0.0):
        return 0.0, 0.0

    x_values = sorted({float(point[0]) for point in boundary})
    y_values = sorted({float(point[1]) for point in boundary})
    dx_candidates = {0.0}
    dy_candidates = {0.0}
    for x_value in x_values:
        dx_candidates.add(x_value - float(mins[0]))
        dx_candidates.add(x_value - float(maxs[0]))
    for y_value in y_values:
        dy_candidates.add(y_value - float(mins[1]))
        dy_candidates.add(y_value - float(maxs[1]))

    valid_candidates = [
        (dx, dy)
        for dx in dx_candidates
        for dy in dy_candidates
        if _aabb_inside_polygon_after_shift(mins, maxs, boundary, dx, dy)
    ]
    if not valid_candidates:
        raw_bounds = _parse_boundary_xy_bounds(boundary)
        if raw_bounds is None:
            return 0.0, 0.0
        return compute_xy_shift_for_axis_aligned_aabb_inside_bounds(mins, maxs, *raw_bounds)

    return min(valid_candidates, key=lambda shift: (math.hypot(shift[0], shift[1]), abs(shift[0]) + abs(shift[1])))


def snap_center_z_to_support(
    original_center: Sequence[float],
    current_center: Sequence[float],
    current_bottom_z: float,
    support_z: float,
    max_z_adjustment: float,
) -> list[float]:
    """Move center Z so the current bottom sits on support_z, bounded by original Z."""
    original = _as_float_triplet(original_center, "original_center")
    current = _as_float_triplet(current_center, "current_center")
    desired_center = [current[0], current[1], current[2] + float(support_z) - float(current_bottom_z)]
    return clamp_center(
        original_center=original,
        simulated_center=desired_center,
        max_xy_displacement=math.inf,
        max_z_adjustment=max_z_adjustment,
    )


def update_layout_with_repaired_transforms(
    layout_data: Mapping[str, Any],
    repaired_transforms: Mapping[int, RepairedTransform],
) -> dict[str, Any]:
    """Return a copy of layout_data with final object poses updated by id."""
    result = copy.deepcopy(layout_data)
    if "final_scene" in result:
        scene_data = result["final_scene"]
    elif "objects" in result:
        scene_data = result
    else:
        raise ValueError("无法识别 layout 格式，缺少 final_scene 或 objects")

    if not isinstance(scene_data, dict):
        raise ValueError("场景数据必须是 object")
    if "objects" not in scene_data:
        raise KeyError("场景数据缺少 objects")
    objects = scene_data["objects"]
    if not isinstance(objects, list):
        raise ValueError("场景 objects 必须是 list")

    remaining_ids = set(int(object_id) for object_id in repaired_transforms.keys())
    for obj_data in objects:
        if not isinstance(obj_data, dict):
            raise ValueError(f"场景 object 必须是 object: {obj_data}")
        if "id" not in obj_data:
            raise KeyError(f"场景 object 缺少 id: {obj_data}")
        obj_id = int(obj_data["id"])
        if obj_id not in repaired_transforms:
            continue
        repaired = repaired_transforms[obj_id]
        obj_data["center"] = _round_triplet(repaired.center)
        obj_data["rotation"] = _round_triplet(repaired.rotation)
        remaining_ids.remove(obj_id)

    if remaining_ids:
        raise KeyError(f"repaired_transforms 包含 layout 中不存在的 object_id: {sorted(remaining_ids)}")

    # 修复产物里只有 final_scene 是权威位姿;successful_placements/failed_placements 是
    # 推理阶段产物,其中的 pose 不会随 bake/solve 更新,留着会与 final_scene 不一致、误导
    # 排查(导出 GLB 也只读 final_scene)。原始推理 layout JSON 仍保留这两份,同目录可取。
    result.pop("successful_placements", None)
    result.pop("failed_placements", None)
    return result


def write_repaired_layout(
    layout_data: Mapping[str, Any],
    repaired_transforms: Mapping[int, RepairedTransform],
    output_path: str | Path,
) -> Path:
    repaired_layout = update_layout_with_repaired_transforms(layout_data, repaired_transforms)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(repaired_layout, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def validate_repaired_layout_output_path(layout_path: str | Path, output_path: str | Path) -> Path:
    layout = Path(layout_path).resolve()
    output = Path(output_path).resolve()
    if output == layout:
        raise ValueError(f"physics-output-layout 不能覆盖输入 layout: {layout}")
    return Path(output_path)


def _dot2(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _normalize2(v: tuple[float, float]) -> tuple[float, float]:
    length = math.hypot(v[0], v[1])
    if length <= 1e-9:
        return (1.0, 0.0)
    return (v[0] / length, v[1] / length)


def _object_rectangle(obj_data: Mapping[str, Any], transform: RepairedTransform) -> _Rectangle:
    size = _as_float_triplet(obj_data["size"], "size")
    rotation = _as_float_triplet(transform.rotation, "rotation")
    yaw = math.radians(rotation[2])
    axis_x = (math.cos(yaw), math.sin(yaw))
    axis_y = (-math.sin(yaw), math.cos(yaw))
    return _Rectangle(
        center=(float(transform.center[0]), float(transform.center[1])),
        axes=(axis_x, axis_y),
        half_size=(size[0] / 2.0, size[1] / 2.0),
    )


def _project_rectangle(rect: _Rectangle, axis: tuple[float, float]) -> tuple[float, float]:
    axis = _normalize2(axis)
    center_projection = _dot2(rect.center, axis)
    radius = (
        abs(_dot2(rect.axes[0], axis)) * rect.half_size[0]
        + abs(_dot2(rect.axes[1], axis)) * rect.half_size[1]
    )
    return center_projection - radius, center_projection + radius


def _projection_separation_options(
    projection_a: tuple[float, float],
    projection_b: tuple[float, float],
) -> list[tuple[float, float]]:
    positive_distance = projection_a[1] - projection_b[0]
    negative_distance = projection_b[1] - projection_a[0]
    if positive_distance <= 1e-9 or negative_distance <= 1e-9:
        return []
    return [(positive_distance, 1.0), (negative_distance, -1.0)]


def _xy_push_vector(
    obj_a: Mapping[str, Any],
    transform_a: RepairedTransform,
    obj_b: Mapping[str, Any],
    transform_b: RepairedTransform,
) -> tuple[float, float] | None:
    rect_a = _object_rectangle(obj_a, transform_a)
    rect_b = _object_rectangle(obj_b, transform_b)
    best_axis = (1.0, 0.0)
    best_overlap = math.inf

    for axis in rect_a.axes + rect_b.axes:
        projection_a = _project_rectangle(rect_a, axis)
        projection_b = _project_rectangle(rect_b, axis)
        options = _projection_separation_options(projection_a, projection_b)
        if not options:
            return None
        signed_center_delta = _dot2(
            (
                transform_b.center[0] - transform_a.center[0],
                transform_b.center[1] - transform_a.center[1],
            ),
            axis,
        )
        preferred_sign = 1.0 if signed_center_delta >= 0.0 else -1.0
        same_side_options = [option for option in options if option[1] == preferred_sign]
        overlap, sign = min(same_side_options or options, key=lambda option: option[0])
        if overlap < best_overlap:
            best_overlap = overlap
            normalized_axis = _normalize2(axis)
            best_axis = (normalized_axis[0] * sign, normalized_axis[1] * sign)
    return (best_axis[0] * best_overlap, best_axis[1] * best_overlap)


def _is_floor_covering(obj_data: Mapping[str, Any]) -> bool:
    category = str(obj_data.get("category", "")).lower()
    desc = str(obj_data.get("short_desc", "")).lower()
    text = f"{category} {desc}"
    return "carpet" in text or "rug" in text


def _object_text(obj_data: Mapping[str, Any]) -> str:
    return f"{obj_data.get('category', '')} {obj_data.get('short_desc', '')}".lower()


def _is_chair_like(obj_data: Mapping[str, Any]) -> bool:
    return "chair" in _object_text(obj_data)


def _is_table_or_desk_like(obj_data: Mapping[str, Any]) -> bool:
    text = _object_text(obj_data)
    return "desk" in text or "table" in text


def _is_semantic_chair_table_pair(obj_a: Mapping[str, Any], obj_b: Mapping[str, Any]) -> bool:
    obj_a_id = int(obj_a["id"])
    obj_b_id = int(obj_b["id"])
    if _is_chair_like(obj_a) and _is_table_or_desk_like(obj_b):
        return _has_relation_to_target(obj_a, obj_b_id, _CHAIR_TABLE_RELATION_TYPES)
    if _is_chair_like(obj_b) and _is_table_or_desk_like(obj_a):
        return _has_relation_to_target(obj_b, obj_a_id, _CHAIR_TABLE_RELATION_TYPES)
    return False


def _semantic_chair_table_should_skip_xy_separation(
    obj_a: Mapping[str, Any],
    obj_b: Mapping[str, Any],
    transform_a: RepairedTransform,
    transform_b: RepairedTransform,
) -> bool:
    """Skip XY push-out only when the chair centroid lies near the desk/table XY footprint."""
    if _is_chair_like(obj_a) and _is_table_or_desk_like(obj_b):
        desk_data = obj_b
        chair_t, desk_t = transform_a, transform_b
    elif _is_chair_like(obj_b) and _is_table_or_desk_like(obj_a):
        desk_data = obj_a
        chair_t, desk_t = transform_b, transform_a
    else:
        return False

    desk_rect = _object_rectangle(desk_data, desk_t)
    cx = float(chair_t.center[0])
    cy = float(chair_t.center[1])
    local_x = _dot2(
        (cx - desk_rect.center[0], cy - desk_rect.center[1]),
        desk_rect.axes[0],
    )
    local_y = _dot2(
        (cx - desk_rect.center[0], cy - desk_rect.center[1]),
        desk_rect.axes[1],
    )
    margin = _CHAIR_TABLE_FOOTPRINT_XY_MARGIN
    max_x = desk_rect.half_size[0] + margin
    max_y = desk_rect.half_size[1] + margin
    return abs(local_x) <= max_x and abs(local_y) <= max_y


def _parent_target_id(obj_data: Mapping[str, Any]) -> int | None:
    parent_id = obj_data.get("parent_id")
    if parent_id is None:
        return None
    try:
        return int(parent_id)
    except (TypeError, ValueError):
        return None


def _is_parent_supported_child(obj_data: Mapping[str, Any]) -> bool:
    if _is_floor_covering(obj_data) or _is_chair_like(obj_data):
        return False
    text = " ".join(
        str(obj_data.get(key, "")).lower()
        for key in ("short_desc", "category", "type", "coarse_semantic")
    )
    if any(keyword in text for keyword in _PARENT_SUPPORT_CHILD_KEYWORDS):
        return True
    size = obj_data.get("size")
    if isinstance(size, Sequence) and not isinstance(size, (str, bytes)) and len(size) == 3:
        return max(float(value) for value in size) <= 0.8
    return False


def _parent_support_target_id(obj_data: Mapping[str, Any]) -> int | None:
    if not _is_parent_supported_child(obj_data):
        return None
    return _parent_target_id(obj_data)


def _expected_on_target_id(obj_data: Mapping[str, Any]) -> int | None:
    return _expected_relation_target_id(obj_data, _SUPPORT_RELATION_TYPES) or _parent_support_target_id(obj_data)


def _expected_attachment_target_id(obj_data: Mapping[str, Any]) -> int | None:
    return _expected_relation_target_id(obj_data, _ATTACHMENT_RELATION_TYPES) or _parent_support_target_id(obj_data)


def _expected_containment_target_id(obj_data: Mapping[str, Any]) -> int | None:
    return _expected_relation_target_id(obj_data, _CONTAINMENT_RELATION_TYPES)


def _expected_on_target_ids(obj_data: Mapping[str, Any]) -> list[int]:
    """物体可能同时 `on` 多个目标(如既 on 桌、又 on 桌上某物);全部返回。"""
    ids = _all_relation_target_ids(obj_data, _SUPPORT_RELATION_TYPES)
    if not ids:
        parent_id = _parent_support_target_id(obj_data)
        if parent_id is not None:
            ids = [parent_id]
    return ids


def _expected_containment_target_ids(obj_data: Mapping[str, Any]) -> list[int]:
    return _all_relation_target_ids(obj_data, _CONTAINMENT_RELATION_TYPES)


def _bake_anchor_ids(scene_objects: Sequence[Mapping[str, Any]]) -> set[int]:
    """bake 中应锚定为 PASSIVE 的物体 id。

    Bullet bake 让整场景自由刚体沉降时,2.47m 高的货架会被推倒、塞在同一 XY 的几十个
    货架子件会因初始互穿被炸开——支撑面自己垮掉,上面的件自然落不稳。

    因此把"语义上已被支撑/容纳的件(有 on/within 关系)"及其"支撑/容器目标"锚定为
    PASSIVE:它们的 XY 信任模型放置、不参与重力沉降;Z 方向的贴支撑面交给 python 侧
    `snap_children_z`(下射目标网格,确定性贴面,远比 Bullet 在薄隔板上稳)。其余真正
    自由的件仍是 ACTIVE,在重力下落到地面/被动支撑面。
    """
    by_id: dict[int, Mapping[str, Any]] = {}
    for obj_data in scene_objects:
        if isinstance(obj_data, Mapping) and "id" in obj_data:
            by_id[int(obj_data["id"])] = obj_data

    anchor: set[int] = set()
    for obj_id, obj_data in by_id.items():
        targets = _expected_on_target_ids(obj_data) + _expected_containment_target_ids(obj_data)
        targets = [t for t in targets if t in by_id]
        if targets:
            anchor.add(obj_id)
            anchor.update(targets)
    return anchor


def _xy_separation_movability(obj_data: Mapping[str, Any]) -> float:
    """解纠缠时该物体的"可动性"权重(越大越该让位)。

    语义父子(on/within,贴在别的东西上)的件应少动;大件家具应少动;
    自由小件应多动。返回正数,用于在一对重叠物体间按比例分摊分离位移。
    """
    if (
        _expected_attachment_target_id(obj_data) is not None
        or _expected_containment_target_id(obj_data) is not None
    ):
        relation_weight = 0.2
    else:
        relation_weight = 1.0

    size = obj_data.get("size")
    if isinstance(size, Sequence) and not isinstance(size, (str, bytes)) and len(size) == 3:
        footprint = max(float(size[0]), float(size[1]))
    else:
        footprint = 0.5
    if footprint <= 0.4:
        size_weight = 1.0
    elif footprint >= 1.2:
        size_weight = 0.3
    else:
        size_weight = 1.0 - (footprint - 0.4) / (1.2 - 0.4) * 0.7

    return max(0.05, relation_weight * size_weight)


def _is_hard_overlap_exempt(
    obj_a: Mapping[str, Any],
    obj_b: Mapping[str, Any],
    transform_a: RepairedTransform | None = None,
    transform_b: RepairedTransform | None = None,
) -> bool:
    """**硬豁免**:语义上本就该接触/包含/上下叠放的对,XY 横推会破坏关系,无条件跳过。

    与软豁免(`floor_covering`/`next_to`,受真实 mesh 谓词门控)区分开:硬豁免不看 mesh,
    因为这些对 footprint 本就该重叠(物压在对方下、物在桌上、物在容器内、椅子塞进桌下)。
    打分(_select_xy_push_vector)与解纠缠(_should_skip_overlap_pair)共用本判定,
    保证"选逃逸方向"时不会把这些合法重叠误当成要避开的障碍。
    """
    # `under`(如地毯/托盘 under 家具):语义上就该压在对方下方、footprint 必然重叠。
    if _is_under_pair(obj_a, obj_b):
        return True
    # 地毯/地垫(floor covering)铺地、家具压其上:即便模型没显式给 `under` 关系,
    # 也按几何判定豁免——毯子位于对方下方时,横推会把毯子从家具底下拽出来。
    if _is_floor_covering_under_pair(obj_a, obj_b, transform_a, transform_b):
        return True
    # 椅子-桌/桌面:仅当椅子质心落在桌面 footprint(含 margin)内才豁免(塞进桌下)。
    if _is_semantic_chair_table_pair(obj_a, obj_b):
        if transform_a is None or transform_b is None:
            return True
        return _semantic_chair_table_should_skip_xy_separation(
            obj_a, obj_b, transform_a, transform_b
        )
    # `on` 堆叠:一方 on 另一方 → 上下叠放,footprint 本就重叠。
    if _is_on_support_pair(obj_a, obj_b):
        return True
    # `within` 容纳:只豁免其首要(最外层)容器这一对。
    if _is_primary_container_pair(obj_a, obj_b):
        return True
    return False


def _is_floor_covering_under_pair(
    obj_a: Mapping[str, Any],
    obj_b: Mapping[str, Any],
    transform_a: RepairedTransform | None = None,
    transform_b: RepairedTransform | None = None,
) -> bool:
    """一方是地毯/地垫且铺在另一方下方(几何判定,无需显式 `under` 关系)。

    仅当**恰好一方**是 floor covering 时生效;两件家具压同一张毯子是常态、合法重叠,
    不该被横向推开。有位姿时按"毯子中心更低"门控,避免毯子放到桌面等反常情形被误豁免。
    """
    a_cov = _is_floor_covering(obj_a)
    b_cov = _is_floor_covering(obj_b)
    if a_cov == b_cov:
        return False
    cover_t, other_t = (transform_a, transform_b) if a_cov else (transform_b, transform_a)
    if cover_t is None or other_t is None:
        return True
    return cover_t.center[2] <= other_t.center[2]


def _is_under_pair(
    obj_a: Mapping[str, Any], obj_b: Mapping[str, Any]
) -> bool:
    """一方对另一方声明了 `under` 关系(如地毯 under 桌子)。"""
    obj_a_id = int(obj_a["id"])
    obj_b_id = int(obj_b["id"])
    return _has_relation_to_target(
        obj_a, obj_b_id, _UNDER_RELATION_TYPES
    ) or _has_relation_to_target(obj_b, obj_a_id, _UNDER_RELATION_TYPES)


def _is_on_support_pair(
    obj_a: Mapping[str, Any], obj_b: Mapping[str, Any]
) -> bool:
    obj_a_id = int(obj_a["id"])
    obj_b_id = int(obj_b["id"])
    if _has_relation_to_target(obj_a, obj_b_id, _SUPPORT_RELATION_TYPES):
        return True
    if _has_relation_to_target(obj_b, obj_a_id, _SUPPORT_RELATION_TYPES):
        return True
    if _parent_support_target_id(obj_a) == obj_b_id:
        return True
    if _parent_support_target_id(obj_b) == obj_a_id:
        return True
    return False


def _is_primary_container_pair(
    obj_a: Mapping[str, Any], obj_b: Mapping[str, Any]
) -> bool:
    obj_a_id = int(obj_a["id"])
    obj_b_id = int(obj_b["id"])
    return (
        _expected_containment_target_id(obj_a) == obj_b_id
        or _expected_containment_target_id(obj_b) == obj_a_id
    )


def _clamp_transform_within_target(
    transform: RepairedTransform,
    obj_data: Mapping[str, Any],
    target_transform: RepairedTransform,
    target_data: Mapping[str, Any],
) -> RepairedTransform:
    obj_size = _as_float_triplet(obj_data["size"], "size")
    target_rect = _object_rectangle(target_data, target_transform)
    obj_half_x = obj_size[0] / 2.0
    obj_half_y = obj_size[1] / 2.0
    local_x = _dot2(
        (
            transform.center[0] - target_rect.center[0],
            transform.center[1] - target_rect.center[1],
        ),
        target_rect.axes[0],
    )
    local_y = _dot2(
        (
            transform.center[0] - target_rect.center[0],
            transform.center[1] - target_rect.center[1],
        ),
        target_rect.axes[1],
    )
    max_x = max(0.0, target_rect.half_size[0] - obj_half_x)
    max_y = max(0.0, target_rect.half_size[1] - obj_half_y)
    clamped_x = max(-max_x, min(max_x, local_x))
    clamped_y = max(-max_y, min(max_y, local_y))
    world_x = (
        target_rect.center[0]
        + target_rect.axes[0][0] * clamped_x
        + target_rect.axes[1][0] * clamped_y
    )
    world_y = (
        target_rect.center[1]
        + target_rect.axes[0][1] * clamped_x
        + target_rect.axes[1][1] * clamped_y
    )
    return RepairedTransform(
        center=_round_triplet([world_x, world_y, transform.center[2]]),
        rotation=transform.rotation,
    )


def _apply_container_constraint(
    transform: RepairedTransform,
    obj_data: Mapping[str, Any],
    objects_by_id: Mapping[int, Mapping[str, Any]],
    transforms: Mapping[int, RepairedTransform],
    original: RepairedTransform,
    max_xy_displacement: float,
    max_z_adjustment: float,
) -> RepairedTransform:
    target_id = _expected_containment_target_id(obj_data)
    if target_id is None or target_id not in transforms or target_id not in objects_by_id:
        return transform
    constrained = _clamp_transform_within_target(
        transform=transform,
        obj_data=obj_data,
        target_transform=transforms[target_id],
        target_data=objects_by_id[target_id],
    )
    return RepairedTransform(
        center=clamp_center(
            original_center=original.center,
            simulated_center=constrained.center,
            max_xy_displacement=max_xy_displacement,
            max_z_adjustment=max_z_adjustment,
        ),
        rotation=constrained.rotation,
    )


def _apply_support_constraint(
    transform: RepairedTransform,
    obj_data: Mapping[str, Any],
    objects_by_id: Mapping[int, Mapping[str, Any]],
    transforms: Mapping[int, RepairedTransform],
    original: RepairedTransform,
    max_xy_displacement: float,
    max_z_adjustment: float,
) -> RepairedTransform:
    target_id = _expected_on_target_id(obj_data)
    if target_id is None or target_id not in transforms or target_id not in objects_by_id:
        return transform
    constrained = _clamp_transform_within_target(
        transform=transform,
        obj_data=obj_data,
        target_transform=transforms[target_id],
        target_data=objects_by_id[target_id],
    )
    return RepairedTransform(
        center=clamp_center(
            original_center=original.center,
            simulated_center=constrained.center,
            max_xy_displacement=max_xy_displacement,
            max_z_adjustment=max_z_adjustment,
        ),
        rotation=constrained.rotation,
    )


def _clamp_transforms_to_originals(
    transforms: Mapping[int, RepairedTransform],
    original_transforms: Mapping[int, RepairedTransform],
    max_xy_displacement: float,
    max_z_adjustment: float,
) -> dict[int, RepairedTransform]:
    clamped_transforms = dict(transforms)
    for object_id, transform in transforms.items():
        original = original_transforms.get(object_id)
        if original is None:
            continue
        clamped_transforms[object_id] = RepairedTransform(
            center=clamp_center(
                original_center=original.center,
                simulated_center=transform.center,
                max_xy_displacement=max_xy_displacement,
                max_z_adjustment=max_z_adjustment,
            ),
            rotation=transform.rotation,
        )
    return clamped_transforms


def _require_bpy():
    try:
        return importlib.import_module("bpy")
    except ModuleNotFoundError as exc:
        raise RuntimeError("Blender 物理修正必须在 Blender Python 中运行") from exc


def _object_id(obj: Any) -> int:
    if "object_id" not in obj.keys():
        raise KeyError(f"Blender object 缺少 object_id 自定义属性: {obj.name}")
    return int(obj["object_id"])


def _rotation_degrees(obj: Any) -> list[float]:
    euler = obj.rotation_euler
    return _round_triplet(
        [
            math.degrees(float(euler.x)),
            math.degrees(float(euler.y)),
            math.degrees(float(euler.z)),
        ]
    )


def capture_original_transforms(objects: Sequence[Any]) -> dict[int, RepairedTransform]:
    transforms: dict[int, RepairedTransform] = {}
    for obj in objects:
        obj_id = _object_id(obj)
        transforms[obj_id] = RepairedTransform(
            center=_round_triplet([obj.location.x, obj.location.y, obj.location.z]),
            rotation=_rotation_degrees(obj),
        )
    return transforms


def _select_active(obj: Any) -> None:
    bpy = _require_bpy()
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


_COLLISION_DECIMATE_MODIFIER = "_repair_collision_decimate"


def _apply_collision_decimate(obj: Any, target_faces: int) -> bool:
    """给碰撞网格挂一个 Decimate 代理（保留凹形，只降面数）。

    通过 rigid_body.mesh_source='FINAL' 让 Bullet 用减面后的网格构建 MESH 碰撞，
    可见网格本身不被破坏（modifier 在仿真结束后移除），因此后续 GLB 导出不受影响。
    返回是否实际添加了 modifier。
    """
    if target_faces < 4:
        raise ValueError(f"collision_decimate_target_faces 太小: {target_faces}")
    mesh = getattr(obj, "data", None)
    if mesh is None or not hasattr(mesh, "polygons"):
        return False
    current_faces = len(mesh.polygons)
    if current_faces <= target_faces:
        return False
    modifier = obj.modifiers.new(name=_COLLISION_DECIMATE_MODIFIER, type="DECIMATE")
    modifier.decimate_type = "COLLAPSE"
    modifier.ratio = min(1.0, target_faces / current_faces)
    obj.rigid_body.mesh_source = "FINAL"
    return True


def _remove_collision_decimate(obj: Any) -> None:
    modifier = obj.modifiers.get(_COLLISION_DECIMATE_MODIFIER)
    if modifier is not None:
        obj.modifiers.remove(modifier)


_VHACD_HULL_PREFIX = "_repair_vhacd_hull"


@contextlib.contextmanager
def _suppressed_c_stdout(enabled: bool = True):
    """临时把 fd=1 重定向到 /dev/null,屏蔽 Blender(C 层)在 COMPOUND+父子刚体下
    每帧打印的 "Dependency cycle detected" 噪声(无害但会刷出十几万行)。
    只屏蔽 stdout;异常仍走 Python 抛出,不受影响。"""
    if not enabled:
        yield
        return
    sys.stdout.flush()
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fd = os.dup(1)
    try:
        os.dup2(devnull_fd, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved_fd, 1)
        os.close(saved_fd)
        os.close(devnull_fd)


def _evaluated_tri_mesh_arrays(obj: Any):
    """取 obj 求值后网格的 (vertices, triangles) numpy 数组(本地坐标),供 V-HACD 用。"""
    import numpy as np  # 延迟导入:仅 Blender/numpy 环境需要

    bpy = _require_bpy()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        mesh.calc_loop_triangles()
        verts = np.array([(v.co.x, v.co.y, v.co.z) for v in mesh.vertices], dtype=np.float64)
        faces = np.array([tuple(t.vertices) for t in mesh.loop_triangles], dtype=np.uint32)
        return verts, faces
    finally:
        evaluated.to_mesh_clear()


def _require_vhacdx():
    """导入 vhacdx;未安装时抛出带安装指引的清晰错误。

    注意:Blender 用自带的 Python 解释器,需把 vhacdx 装到该解释器,而非 conda 环境:
    `<blender>/<ver>/python/bin/python3.x -m pip install vhacdx`
    """
    try:
        import vhacdx  # 延迟导入:仅 vhacd 碰撞模式需要
    except ImportError as exc:
        raise RuntimeError(
            "collision_mode='vhacd' 需要 vhacdx 库,但当前 Python 环境未安装。"
            "请装到 Blender 自带解释器:"
            "`<blender>/<ver>/python/bin/python3.x -m pip install vhacdx`"
            "(或改用 --collision mesh)。"
        ) from exc
    return vhacdx


def _apply_vhacd_compound(
    obj: Any,
    max_hulls: int,
    resolution: int,
) -> list[Any]:
    """对 obj 做 V-HACD 凸分解,生成凸包子体并以 Bullet COMPOUND 复合到 obj。

    碰撞改走凸-凸(远快于凹面 GImpact)。返回创建的子体列表供仿真后清理。
    obj 自身网格不变(COMPOUND 只用子体形状),因此求解器仍可对 obj 减面网格判穿插。
    """
    vhacdx = _require_vhacdx()
    bpy = _require_bpy()
    verts, faces = _evaluated_tri_mesh_arrays(obj)
    if len(faces) == 0:
        return []
    hulls = vhacdx.compute_vhacd(
        verts, faces, maxConvexHulls=int(max_hulls), resolution=int(resolution)
    )
    children: list[Any] = []
    for index, (hull_verts, hull_faces) in enumerate(hulls):
        mesh = bpy.data.meshes.new(f"{_VHACD_HULL_PREFIX}_{obj.name}_{index}")
        mesh.from_pydata(
            [tuple(float(c) for c in v) for v in hull_verts],
            [],
            [tuple(int(i) for i in f) for f in hull_faces],
        )
        mesh.update()
        child = bpy.data.objects.new(mesh.name, mesh)
        bpy.context.scene.collection.objects.link(child)
        child.parent = obj
        # hull 顶点是 obj 局部坐标,需让 child.matrix_world == obj.matrix_world 才能让凸包
        # 与物体几何对齐(进而正确组成 COMPOUND 碰撞体)。child.matrix_basis 为单位阵时,
        # 必须把 matrix_parent_inverse 设为单位阵;若沿用 parent.matrix_world.inverted()
        # 会把凸包放到世界原点,导致 COMPOUND 碰撞体与物体错位、碰撞形同虚设、穿透地板。
        child.matrix_parent_inverse = _require_mathutils().Matrix.Identity(4)
        _add_rigid_body(child, body_type="ACTIVE", collision_shape="CONVEX_HULL")
        children.append(child)
    obj.rigid_body.collision_shape = "COMPOUND"
    return children


def _remove_vhacd_compound(children: Sequence[Any]) -> None:
    """直接删除凸包子体数据块(连带其 rigid_body),避免逐个走 bpy.ops 操作符——
    子体可能上百个,操作符 + 选中态切换会非常慢。"""
    bpy = _require_bpy()
    for child in children:
        mesh = getattr(child, "data", None)
        bpy.data.objects.remove(child, do_unlink=True)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def _add_rigid_body(obj: Any, body_type: str, collision_shape: str) -> None:
    bpy = _require_bpy()
    _select_active(obj)
    if obj.rigid_body is None:
        bpy.ops.rigidbody.object_add(type=body_type)
    else:
        obj.rigid_body.type = body_type
    obj.rigid_body.collision_shape = collision_shape
    obj.rigid_body.friction = 0.8
    obj.rigid_body.restitution = 0.0
    obj.rigid_body.linear_damping = 0.6
    obj.rigid_body.angular_damping = 1.0
    # 开启休眠:物体速度持续低于阈值后由 Bullet 置为睡眠,不再参与后续帧仿真,
    # bake 尾段(多数物体已静止)几乎零成本。不改碰撞形状/帧数,对修复结果影响最小。
    obj.rigid_body.use_deactivation = True


def _ensure_rigid_body_world(
    frame_start: int,
    frame_end: int,
    substeps_per_frame: int = 5,
    solver_iterations: int = 10,
) -> None:
    bpy = _require_bpy()
    scene = bpy.context.scene
    if scene.rigidbody_world is None:
        bpy.ops.rigidbody.world_add()
    scene.frame_start = int(frame_start)
    scene.frame_end = int(frame_end)
    scene.frame_current = int(frame_start)
    scene.rigidbody_world.point_cache.frame_start = int(frame_start)
    scene.rigidbody_world.point_cache.frame_end = int(frame_end)
    scene.rigidbody_world.substeps_per_frame = int(substeps_per_frame)
    scene.rigidbody_world.solver_iterations = int(solver_iterations)
    scene.gravity = (0.0, 0.0, -9.81)


def _clear_rigid_body_cache() -> None:
    bpy = _require_bpy()
    try:
        bpy.ops.ptcache.free_bake_all()
    except RuntimeError:
        pass


def _restore_original_rotation(obj: Any, original: RepairedTransform) -> None:
    obj.rotation_euler = (
        math.radians(original.rotation[0]),
        math.radians(original.rotation[1]),
        math.radians(original.rotation[2]),
    )


def _clamp_object_location(
    obj: Any,
    original: RepairedTransform,
    max_xy_displacement: float,
    max_z_adjustment: float,
) -> list[float]:
    # 刚体仿真结果写在 matrix_world(depsgraph 求值变换)上,obj.location 仍停留在
    # 仿真前的原值;必须读世界平移才能拿到真正沉降后的位置。资产均为无父级顶层物体,
    # 故世界平移即可直接写回 obj.location。
    world_t = obj.matrix_world.translation
    clamped_center = clamp_center(
        original_center=original.center,
        simulated_center=[world_t.x, world_t.y, world_t.z],
        max_xy_displacement=max_xy_displacement,
        max_z_adjustment=max_z_adjustment,
    )
    obj.location = clamped_center
    return clamped_center


def _remove_rigid_body(obj: Any) -> None:
    bpy = _require_bpy()
    if obj.rigid_body is None:
        return
    _select_active(obj)
    bpy.ops.rigidbody.object_remove()


def _require_mathutils():
    try:
        return importlib.import_module("mathutils")
    except ModuleNotFoundError as exc:
        raise RuntimeError("容器内支撑面检测必须在 Blender Python 中运行") from exc


def run_physics_repair(
    floor_obj: Any,
    asset_objects: Sequence[Any],
    scene_objects: Sequence[Mapping[str, Any]] | None = None,
    frame_count: int = 40,
    max_xy_displacement: float = 0.8,
    max_z_adjustment: float = 0.5,
    preserve_rotation: bool = True,
    boundary: Sequence[Sequence[float]] | None = None,
    boundary_max_xy_displacement: float | None = None,
    collision_decimate_target_faces: int | None = None,
    substeps_per_frame: int = 5,
    solver_iterations: int = 10,
    collision_mode: str = "vhacd",
    vhacd_max_hulls: int = 16,
    vhacd_resolution: int = 100000,
) -> dict[int, RepairedTransform]:
    """跑一段有界刚体重力沉降(bake),返回限幅后的位姿。

    几何后处理(去穿插/落地/容器/边界/货架换层)不在 Blender 内做,而是交给 python 侧
    `solve_layout.py`(VHACD 凸块 2D-SAT + XPBD)。本函数只负责产出可靠的 bake 初始位姿。

    collision_decimate_target_faces: 若给定，则对每个动态资产挂临时 Decimate 代理，
        把碰撞网格降到约该面数（仍用 MESH，保留凹形），以加速 Bullet GImpact 求交；
        None 表示不减面，沿用原始高模碰撞。
    """
    if frame_count < 1:
        raise ValueError("frame_count 必须大于 0")
    if floor_obj is None:
        raise ValueError("physics repair 需要地板物体")
    if not asset_objects:
        return {}

    bpy = _require_bpy()
    start_time = time.perf_counter()
    original_transforms = capture_original_transforms(asset_objects)
    if collision_mode not in ("mesh", "vhacd"):
        raise ValueError(f"collision_mode 只支持 mesh/vhacd: {collision_mode}")
    if collision_mode == "vhacd":
        _require_vhacdx()  # 尽早失败:未装 vhacdx 时不要白跑 bake 前置流程
    rigid_body_objects = []
    decimated_objects: list[Any] = []
    vhacd_children: list[Any] = []
    # vhacd 的 COMPOUND 父子刚体会让 Blender 每帧打印依赖循环警告(无害但刷屏),
    # 整个 bake 阶段(建代理→仿真→清理)屏蔽 C 层 stdout。
    _bake_stdout_guard = _suppressed_c_stdout(collision_mode == "vhacd")
    _bake_stdout_guard.__enter__()
    try:
        _ensure_rigid_body_world(
            frame_start=1,
            frame_end=frame_count,
            substeps_per_frame=substeps_per_frame,
            solver_iterations=solver_iterations,
        )
        _add_rigid_body(floor_obj, body_type="PASSIVE", collision_shape="MESH")
        rigid_body_objects.append(floor_obj)
        anchor_ids = (
            _bake_anchor_ids(scene_objects) if scene_objects is not None else set()
        )
        for obj in asset_objects:
            # 语义上已被支撑/容纳的件及其支撑/容器目标锚定为 PASSIVE:用真网格(MESH)做
            # 静态碰撞体(btBvhTriangleMeshShape,精确且高效,能挡住自由件),自身不沉降;
            # Z 贴支撑面交给 python 侧 snap。其余自由件 ACTIVE,在重力下沉降。
            if _object_id(obj) in anchor_ids:
                _add_rigid_body(obj, body_type="PASSIVE", collision_shape="MESH")
                rigid_body_objects.append(obj)
                obj.rigid_body.enabled = True
                if collision_decimate_target_faces is not None and _apply_collision_decimate(
                    obj, collision_decimate_target_faces
                ):
                    decimated_objects.append(obj)
                continue
            _add_rigid_body(obj, body_type="ACTIVE", collision_shape="MESH")
            rigid_body_objects.append(obj)
            obj.rigid_body.mass = 1.0
            obj.rigid_body.enabled = True
            # 碰撞代理:vhacd 用凸分解 COMPOUND(凸-凸,远快于 GImpact),
            # mesh 用减面后的真网格。两种模式都仍对 obj 挂减面修饰器,
            # 供仿真后求解器的 BVH/raycast 复用(COMPOUND 只用子体形状,不影响 obj 网格)。
            if collision_mode == "vhacd":
                vhacd_children.extend(
                    _apply_vhacd_compound(obj, vhacd_max_hulls, vhacd_resolution)
                )
            if collision_decimate_target_faces is not None and _apply_collision_decimate(
                obj, collision_decimate_target_faces
            ):
                decimated_objects.append(obj)

        _clear_rigid_body_cache()
        for frame in range(1, frame_count + 1):
            bpy.context.scene.frame_set(frame)
            bpy.context.view_layer.update()

        repaired_transforms: dict[int, RepairedTransform] = {}
        for obj in asset_objects:
            obj_id = _object_id(obj)
            original = original_transforms[obj_id]
            clamped_center = _clamp_object_location(
                obj=obj,
                original=original,
                max_xy_displacement=max_xy_displacement,
                max_z_adjustment=max_z_adjustment,
            )
            if preserve_rotation:
                _restore_original_rotation(obj, original)
            bpy.context.view_layer.update()
            repaired_transforms[obj_id] = RepairedTransform(
                center=clamped_center,
                rotation=_rotation_degrees(obj),
            )
    finally:
        _remove_vhacd_compound(vhacd_children)
        for obj in rigid_body_objects:
            _remove_rigid_body(obj)
        bpy.context.view_layer.update()
        _bake_stdout_guard.__exit__(None, None, None)
    boundary_xy_cap = max_xy_displacement
    if boundary is not None and len(boundary) > 0:
        boundary_xy_cap = (
            float("inf")
            if boundary_max_xy_displacement is None
            else float(boundary_max_xy_displacement)
        )

    # 仅产出 bake 沉降后(限幅)的位姿,几何后处理交给 python 侧 XPBD 求解器。
    repaired_transforms = _clamp_transforms_to_originals(
        transforms=repaired_transforms,
        original_transforms=original_transforms,
        max_xy_displacement=boundary_xy_cap,
        max_z_adjustment=max_z_adjustment,
    )
    for obj in decimated_objects:
        _remove_collision_decimate(obj)
    bpy.context.view_layer.update()
    elapsed = time.perf_counter() - start_time
    print(
        "[PhysicsRepair] 完成刚体微调(bake-only): "
        f"{len(repaired_transforms)} 个物体, frames={frame_count}, "
        f"collision={collision_mode}, 耗时={elapsed:.2f}s"
    )
    return repaired_transforms
