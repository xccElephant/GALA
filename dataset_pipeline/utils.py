"""
训练数据生成的公共工具函数
"""
import csv
import json
from pathlib import Path
import shutil
from typing import Any, Dict, Iterable, List, Optional, Tuple

from config import ProjectPaths
from utils.scene_types import AutoregressiveMetadata
from utils.spatial_utils import get_aabb_from_obj, get_object_z_bounds


# ──────────────────────────────────────────────────────────────────
# JSON / 路径解析工具（多脚本共用）
# ──────────────────────────────────────────────────────────────────

def iter_samples_from_json_file(json_file: Path) -> List[Dict[str, Any]]:
    """读取单个 autoregressive JSON 文件，返回样本列表。"""
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [data]
    return data


def count_samples_in_file(json_file: Path) -> int:
    """统计单个 JSON 文件中的样本数量。"""
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return 1
        return len(data)
    except Exception:
        return 0


def cleanup_output_files(paths: Iterable[Path], description: str = "输出文件") -> int:
    """删除指定的旧输出文件，避免旧产物被误认为本轮生成结果。"""
    stale_files = []
    seen_paths = set()
    for path in paths:
        file_path = Path(path)
        if file_path in seen_paths:
            continue
        seen_paths.add(file_path)
        if file_path.exists():
            if not file_path.is_file():
                raise ValueError(f"待清理路径不是文件: {file_path}")
            stale_files.append(file_path)

    if stale_files:
        print(f"清理旧{description}: {len(stale_files)} 个")
        for file_path in stale_files:
            file_path.unlink()

    return len(stale_files)


def recreate_output_directory(path: Path, description: str = "输出目录") -> None:
    """删除并重建输出目录，避免残留 sample_* 或点云文件污染新结果。"""
    output_dir = Path(path)
    resolved = output_dir.resolve()
    if resolved == Path("/"):
        raise ValueError("拒绝清理根目录")

    if output_dir.exists():
        print(f"清理旧{description}: {output_dir}")
        if output_dir.is_dir():
            shutil.rmtree(output_dir)
        else:
            output_dir.unlink()

    output_dir.mkdir(parents=True, exist_ok=True)


def resolve_autoregressive_dir(
    dataset: str, dataset_dir: Optional[str], input_dir: Optional[str]
) -> Path:
    """根据 dataset / input_dir / dataset_dir 解析 autoregressive 数据目录。"""
    if input_dir:
        result = Path(input_dir)
        if not result.exists():
            raise ValueError(f"输入目录不存在: {result}")
        return result

    if dataset == "custom":
        raise ValueError("custom 模式请使用 --input-dir 或 --input-file")

    if dataset_dir is None:
        return ProjectPaths.get_autoregressive_dir(dataset)

    default_dir = ProjectPaths.get_autoregressive_dir(dataset)
    return Path(dataset_dir) / default_dir.name


def parse_asset_pc_dir(dataset: str, asset_pc_dir_arg: Optional[str]) -> "Path | List[Path]":
    """解析 --asset-pc-dir 参数，返回单个或多个资产点云目录。"""
    if asset_pc_dir_arg:
        if "," in asset_pc_dir_arg:
            dirs = [Path(p.strip()) for p in asset_pc_dir_arg.split(",") if p.strip()]
            if not dirs:
                raise ValueError("无效的 --asset-pc-dir 参数")
            return dirs
        return Path(asset_pc_dir_arg)

    if dataset == "custom":
        raise ValueError("custom 模式请提供 --asset-pc-dir")
    return ProjectPaths.get_asset_pointcloud_dir(dataset)


def resolve_num_workers(
    requested_num_workers: Optional[int],
    total_tasks: int,
    available_cpus: Optional[int] = None,
) -> int:
    """统一解析 worker 数量：默认 CPU 一半，并裁剪到任务数范围内。"""
    if total_tasks <= 0:
        return 0

    if available_cpus is None:
        import os

        available_cpus = os.cpu_count()

    available_cpus = max(1, available_cpus or 1)
    default_num_workers = max(1, available_cpus // 2)
    num_workers = default_num_workers if requested_num_workers is None else requested_num_workers
    return max(1, min(num_workers, total_tasks))


def short_desc_contains_keyword(obj: Dict[str, Any], keyword: str) -> bool:
    """判断 obj["short_desc"] 是否包含指定关键字（大小写不敏感）。"""
    if not keyword:
        return False
    short_desc = obj.get("short_desc")
    if short_desc is None:
        return False
    return keyword.lower() in str(short_desc).lower()


def filter_floating_objects(
    objects: List[Dict[str, Any]],
    floor_z: float = 0,
    z_tolerance: float = 0.05,
) -> List[Dict[str, Any]]:
    """
    过滤悬浮物体，返回有支撑的物体列表

    过滤逻辑：
    1. 优先检查地板支撑（物体底部接近 floor_z）
    2. 然后检查物体间支撑（物体底部接近另一物体顶部，且 xy 平面有重叠）
    """
    if not objects:
        return objects

    bounds_list = []
    for i, obj in enumerate(objects):
        v_min, v_max = get_aabb_from_obj(obj)
        bounds_list.append(
            {
                "idx": i,
                "obj": obj,
                "z_bottom": float(v_min[2]),
                "z_top": float(v_max[2]),
                "x_min": float(v_min[0]),
                "x_max": float(v_max[0]),
                "y_min": float(v_min[1]),
                "y_max": float(v_max[1]),
            }
        )

    sorted_bounds = sorted(bounds_list, key=lambda b: b["z_bottom"])

    supported_objects = []
    supported_bounds = []

    for bounds in sorted_bounds:
        z_bottom = bounds["z_bottom"]
        is_supported = False

        if abs(z_bottom - floor_z) <= z_tolerance:
            is_supported = True
        else:
            for sup in supported_bounds:
                if abs(z_bottom - sup["z_top"]) <= z_tolerance:
                    x_overlap = not (
                        bounds["x_max"] <= sup["x_min"] or sup["x_max"] <= bounds["x_min"]
                    )
                    y_overlap = not (
                        bounds["y_max"] <= sup["y_min"] or sup["y_max"] <= bounds["y_min"]
                    )
                    if x_overlap and y_overlap:
                        is_supported = True
                        break

        if is_supported:
            supported_objects.append(bounds["obj"])
            supported_bounds.append(bounds)

    return supported_objects


def remove_objects_by_indices(
    scene: Dict[str, Any],
    indices_to_remove: set[int] | List[int],
) -> int:
    """就地删除场景中指定索引的物体，返回实际删除数量。"""
    if not indices_to_remove:
        return 0

    removal_set = set(indices_to_remove)
    objects = scene.get("objects", [])
    original_count = len(objects)
    scene["objects"] = [
        obj for idx, obj in enumerate(objects) if idx not in removal_set
    ]
    return original_count - len(scene["objects"])


def _contains_any_token(text: str, tokens: List[str]) -> bool:
    t = (text or "").lower()
    return any(tok in t for tok in tokens)


# 地面覆盖物关键词：地毯本质上是地面装饰，不应被视为"支撑物"
_FLOOR_COVERING_TOKENS = ["carpet", "rug"]


def is_floor_covering(obj: Dict[str, Any]) -> bool:
    """判断物体是否为地面覆盖物（地毯/rug）。"""
    return _contains_any_token(obj.get("short_desc", ""), _FLOOR_COVERING_TOKENS)


def prefer_under_or_within(reference_short_desc: str) -> str:
    """
    当同一个物体对同时满足 under 和 within 时，用 reference 的 short_desc 做一个轻量偏好选择。

    返回:
        "under" 或 "within"
    """
    container_tokens = [
        "wardrobe",
        "closet",
        "cabinet",
        "cupboard",
        "bookshelf",
        "bookcase",
        "shelf",
        "drawer",
        "storage",
        "chest",
        "trunk",
        "box",
        "basket",
    ]
    support_tokens = [
        "table",
        "desk",
        "bed",
        "nightstand",
        "sideboard",
        "tv stand",
        "coffee table",
        "dining table",
        "console",
        "counter",
        "workbench",
    ]

    if _contains_any_token(reference_short_desc, container_tokens):
        return "within"
    if _contains_any_token(reference_short_desc, support_tokens):
        return "under"
    return "within"


def normalize_boundary_order(
    boundary: List[List[float]],
    *,
    make_ccw: bool = True,
    start_from_min_xy: bool = True,
) -> List[List[float]]:
    """规范化 boundary 的顶点顺序（不改变几何形状，只做环方向与起点的规范化）。"""
    if not boundary:
        return boundary

    pts = [list(map(float, p)) for p in boundary]
    if len(pts) >= 2 and pts[0][0] == pts[-1][0] and pts[0][1] == pts[-1][1]:
        pts = pts[:-1]

    if len(pts) < 3:
        return pts

    area2 = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area2 += x1 * y2 - x2 * y1

    is_ccw = area2 > 0.0
    if make_ccw and not is_ccw:
        pts = list(reversed(pts))
    if (not make_ccw) and is_ccw:
        pts = list(reversed(pts))

    if start_from_min_xy:
        min_idx = min(range(len(pts)), key=lambda i: (pts[i][0], pts[i][1]))
        pts = pts[min_idx:] + pts[:min_idx]

    return pts


def translate_scene_to_nonnegative_xy(
    scene: Dict[str, Any],
    boundary_key: str = "boundary",
    objects_key: str = "objects",
    round_boundary: int = 2,
    round_pos: int = 2,
) -> Tuple[float, float]:
    """将场景在 XY 平面平移到非负空间，并保证 boundary 外接矩形的最小点为 (0, 0)。"""
    boundary = scene.get(boundary_key)
    if not boundary:
        return 0.0, 0.0

    min_x = float(min(p[0] for p in boundary))
    min_y = float(min(p[1] for p in boundary))
    dx, dy = -min_x, -min_y

    shifted_boundary: List[List[float]] = []
    for p in boundary:
        x = round(float(p[0]) + dx, round_boundary)
        y = round(float(p[1]) + dy, round_boundary)
        if abs(x) < 1e-9:
            x = 0.0
        if abs(y) < 1e-9:
            y = 0.0
        shifted_boundary.append([x, y])
    scene[boundary_key] = shifted_boundary

    objects = scene.get(objects_key) or []
    for obj in objects:
        center = obj.get("center")
        if not center or len(center) < 2:
            raise ValueError(
                f"转换格式错误：缺少 center。scene_id={scene.get('scene_id')} jid={obj.get('jid')}"
            )
        center[0] = round(float(center[0]) + dx, round_pos)
        center[1] = round(float(center[1]) + dy, round_pos)
        if abs(center[0]) < 1e-9:
            center[0] = 0.0
        if abs(center[1]) < 1e-9:
            center[1] = 0.0

    return dx, dy


def build_metadata(
    sample: Dict[str, Any],
    sample_id: int,
    scene_pointcloud_path: str = None,
    new_object_pointcloud_path: str = None,
) -> Dict[str, Any]:
    """从样本构建元数据。"""
    return AutoregressiveMetadata.from_sample(
        sample,
        sample_id=sample_id,
        scene_pointcloud_path=scene_pointcloud_path,
        new_object_pointcloud_path=new_object_pointcloud_path,
    ).to_dict()


def export_metadata_to_csv(
    metadata_list: List[Dict[str, Any]],
    csv_file: str,
    include_pointcloud_paths: bool = False,
):
    """将元数据导出到 CSV 文件。"""
    if not metadata_list:
        return

    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)

        headers = [
            "sample_id",
            "scene_id",
            "scene_description",
            "boundary",
            "scene_height",
            "step",
            "existing_objects",
            "new_object",
            "expected_relations",
        ]

        if include_pointcloud_paths:
            headers.extend(["scene_pointcloud_path", "new_object_pointcloud_path"])

        writer.writerow(headers)

        for metadata in metadata_list:
            row = [
                metadata["sample_id"],
                metadata["scene_id"],
                metadata["scene_description"],
                json.dumps(metadata["boundary"], ensure_ascii=False),
                metadata["scene_height"],
                metadata["step"],
                json.dumps(metadata["existing_objects"], ensure_ascii=False),
                json.dumps(metadata["new_object"], ensure_ascii=False),
                json.dumps(metadata["expected_relations"], ensure_ascii=False),
            ]

            if include_pointcloud_paths:
                row.extend(
                    [
                        metadata.get("scene_pointcloud_path", ""),
                        metadata.get("new_object_pointcloud_path", ""),
                    ]
                )

            writer.writerow(row)
