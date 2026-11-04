"""
点云处理工具函数

提供点云采样、变换等公共函数，供训练数据生成和推理使用。
"""
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R


# ──────────────────────────────────────────────────────────────────
# 资产点云加载（带进程内缓存）
# ──────────────────────────────────────────────────────────────────

_pointcloud_cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]] = {}


def _normalize_search_dirs(
    asset_pointcloud_dir: Union[str, Path, Sequence[Union[str, Path]]],
) -> List[Path]:
    base_dirs = (
        [Path(asset_pointcloud_dir)]
        if isinstance(asset_pointcloud_dir, (str, Path))
        else [Path(d) for d in asset_pointcloud_dir]
    )
    search_dirs: List[Path] = []
    seen = set()
    for base_dir in base_dirs:
        for candidate in (
            base_dir,
            base_dir / "imaginarium_assets",
            base_dir / "3D-FUTURE-model",
        ):
            key = str(candidate)
            if key not in seen and candidate.exists():
                search_dirs.append(candidate)
                seen.add(key)
    return search_dirs or base_dirs


def resolve_asset_ply_path(
    jid: str,
    asset_pointcloud_dir: Union[str, Path, Sequence[Union[str, Path]]],
) -> Tuple[Path, Path]:
    """根据 jid 在资产点云目录中查找 .ply 文件，返回 (ply_path, found_dir)。"""
    search_dirs = _normalize_search_dirs(asset_pointcloud_dir)
    for d in search_dirs:
        for candidate in (d / f"{jid}.ply", d / jid / f"{jid}.ply"):
            if candidate.exists():
                return candidate, d
    raise ValueError(f"点云文件未找到: {jid}（搜索目录: {search_dirs}）")


def resolve_pointcloud_path(path: Union[str, Path]) -> str:
    """解析样本点云路径，支持相对路径 datasets/pointclouds/... 或跨机器重定位。"""
    p = str(path)
    if os.path.exists(p):
        return p
    if "pointclouds/" in p:
        rel = p.split("pointclouds/", 1)[1]
        datasets_root = os.environ.get("DATASETS_ROOT") or str(
            Path(__file__).resolve().parent.parent / "datasets"
        )
        candidate = os.path.join(datasets_root, "pointclouds", rel)
        if os.path.exists(candidate):
            return candidate
        return candidate
    return p


@lru_cache(maxsize=2048)
def load_ply_points_cached(path: str) -> np.ndarray:
    """加载 .ply 文件的 XYZ 坐标，返回 (N, 3) float64 数组（带 LRU 缓存）。"""
    import open3d as o3d

    resolved = resolve_pointcloud_path(path)
    pcd = o3d.io.read_point_cloud(resolved)
    return np.asarray(pcd.points, dtype=np.float64)


def load_asset_pointcloud(
    jid: str,
    asset_pointcloud_dir: Union[str, Path, Sequence[Union[str, Path]]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """加载资产点云（PLY + metadata JSON），强制要求坐标、颜色、法向量均存在。

    Args:
        jid: 资产 ID
        asset_pointcloud_dir: 单个目录或目录列表（融合数据集）

    Returns:
        (points, colors, normals, metadata)
        - points:  (N, 3) float32
        - colors:  (N, 3) float32，值域 [0, 1]
        - normals: (N, 3) float32，单位向量
        - metadata: PLY 对应的 JSON 元数据

    Raises:
        ValueError: PLY 文件不存在、metadata 不存在、缺少坐标/颜色/法向量
    """
    if jid in _pointcloud_cache:
        return _pointcloud_cache[jid]

    ply_file, found_dir = resolve_asset_ply_path(jid, asset_pointcloud_dir)

    metadata_file: Optional[Path] = None
    for candidate in [
        found_dir / "metadata" / f"{jid}.json",
        found_dir / f"{jid}.json",
        found_dir / jid / f"{jid}_metadata.json",
    ]:
        if candidate.exists():
            metadata_file = candidate
            break

    if metadata_file is None:
        raise ValueError(f"metadata 文件未找到: {jid}（目录: {found_dir}）")

    plydata = PlyData.read(str(ply_file))
    vertex = plydata["vertex"]
    names = vertex.data.dtype.names

    if "x" not in names:
        raise ValueError(f"PLY 缺少顶点坐标 (x/y/z): {ply_file}")
    if "red" not in names:
        raise ValueError(f"PLY 缺少颜色信息 (red/green/blue): {ply_file}")
    if "nx" not in names:
        raise ValueError(f"PLY 缺少法向量 (nx/ny/nz): {ply_file}")

    points = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T.astype(np.float32)

    r = np.asarray(vertex["red"], dtype=np.float32)
    g = np.asarray(vertex["green"], dtype=np.float32)
    b = np.asarray(vertex["blue"], dtype=np.float32)
    if r.max() > 1.1 or g.max() > 1.1 or b.max() > 1.1:
        colors = np.vstack([r, g, b]).T / 255.0
    else:
        colors = np.vstack([r, g, b]).T
    colors = colors.astype(np.float32)

    nx = np.asarray(vertex["nx"], dtype=np.float32)
    ny = np.asarray(vertex["ny"], dtype=np.float32)
    nz = np.asarray(vertex["nz"], dtype=np.float32)
    normals = np.vstack([nx, ny, nz]).T.astype(np.float32)

    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata: Dict[str, Any] = json.load(f)

    result = (points, colors, normals, metadata)
    _pointcloud_cache[jid] = result
    return result


def transform_pointcloud(
    pointcloud: np.ndarray,
    center: List[float],
    rotation: List[float],
    scale: Optional[Union[float, Sequence[float], np.ndarray]] = 1.0,
) -> np.ndarray:
    """
    对点云进行变换（缩放 + 旋转 + 平移）

    Args:
        pointcloud: 点云数据 (N, 3), 中心在原点
        center: 目标中心 [x, y, z]（世界坐标系）
        rotation: 旋转角度 [roll, pitch, yaw]（度）
        scale: 缩放比例。支持标量或 [sx, sy, sz]。为 None 时不缩放。

    Returns:
        变换后的点云 (N, 3)（世界坐标系）

    注意：
        - 变换顺序：缩放 -> 旋转 -> 平移
        - 旋转使用 XYZ 欧拉角顺序
    """
    pc = np.asarray(pointcloud, dtype=np.float32)

    # 1. 缩放：以原点为中心进行缩放
    if scale is not None:
        s = np.asarray(scale, dtype=np.float32)
        if s.ndim == 0:
            pc = pc * float(s)
        else:
            if s.shape[0] != 3:
                raise ValueError(
                    f"scale must be a scalar or a 3-element sequence, got shape {s.shape}"
                )
            pc = pc * s.reshape(1, 3)

    # 2. 旋转：从欧拉角创建旋转矩阵（XYZ 顺序）
    rot = R.from_euler("xyz", rotation, degrees=True)
    rotated_pc = rot.apply(pc)

    # 3. 平移：移动到目标位置
    transformed_pc = rotated_pc + np.array(center)

    return transformed_pc.astype(np.float32)


def translate_pointcloud_to_nonnegative(
    pointcloud: np.ndarray,
) -> np.ndarray:
    """
    将点云平移到非负空间，使各轴最小坐标对齐到 0。

    Args:
        pointcloud: 点云数据 (N, 3)

    Returns:
        平移后的点云 (N, 3)
    """
    pc = np.asarray(pointcloud, dtype=np.float32)
    if pc.ndim != 2 or pc.shape[1] != 3:
        raise ValueError(f"Expected pointcloud shape (N, 3), got {pc.shape}")
    if len(pc) == 0:
        return pc.copy()

    return (pc - pc.min(axis=0)).astype(np.float32)


def transform_normals(
    normals: np.ndarray,
    rotation: List[float],
) -> np.ndarray:
    """
    对法向量进行旋转变换

    Args:
        normals: 法向量数据 (N, 3), 单位向量
        rotation: 旋转角度 [roll, pitch, yaw]（度）

    Returns:
        变换后的法向量 (N, 3)（世界坐标系，单位向量）

    注意：
        - 法向量只需要旋转，不需要缩放和平移
        - 旋转使用 XYZ 欧拉角顺序
    """
    n = np.asarray(normals, dtype=np.float32)

    # 旋转：从欧拉角创建旋转矩阵（XYZ 顺序）
    rot = R.from_euler("xyz", rotation, degrees=True)
    rotated_normals = rot.apply(n)

    # 重新归一化（防止数值误差）
    norms = np.linalg.norm(rotated_normals, axis=1, keepdims=True).clip(1e-8, None)
    rotated_normals = rotated_normals / norms

    return rotated_normals.astype(np.float32)


def sample_floor_pointcloud(
    boundary: List[List[float]],
    num_points: int = 2048,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    从地板边界采样点云（使用 shapely 加速）

    Args:
        boundary: 地板边界的 2D 点列表 [[x1, y1], [x2, y2], ...]
        num_points: 采样点数
        seed: 可选随机种子；提供时，相同输入会得到一致的 floor 点云

    Returns:
        地板点云 (N, 3), Z 坐标为 0（地面高度）

    注意：
        - 对于凹多边形会正确处理，只在多边形内部采样点
        - 使用 shapely 进行高效的点在多边形内判断
        - 边界点占比约 10%，确保轮廓清晰
    """
    if not boundary or len(boundary) < 3:
        return np.zeros((0, 3), dtype=np.float32)

    try:
        # shapely 仅用于地面采样的点在多边形内判断；为避免无关模块（如评估）引入硬依赖，这里做懒加载
        from shapely import contains_xy  # type: ignore
        from shapely.geometry import Polygon  # type: ignore

        # 将 2D 边界转换为 shapely Polygon
        vertices_2d = np.array(boundary, dtype=np.float32)
        polygon = Polygon(vertices_2d)
        rng = np.random.default_rng(seed)

        # 1. 获取边界框
        x_min, y_min = vertices_2d.min(axis=0)
        x_max, y_max = vertices_2d.max(axis=0)

        # 2. 估算多边形面积占边界框的比例
        bbox_area = (x_max - x_min) * (y_max - y_min)
        polygon_area = polygon.area

        if polygon_area <= 0 or bbox_area <= 0:
            # 退化的多边形，返回边界点
            floor_points = np.column_stack([
                vertices_2d[:, 0],
                vertices_2d[:, 1],
                np.zeros(len(vertices_2d)),
            ])
            return floor_points.astype(np.float32)

        # 过采样系数：确保有足够的点在多边形内
        area_ratio = polygon_area / bbox_area
        oversample_factor = max(2.0 / area_ratio, 1.5)
        n_samples = int(num_points * oversample_factor)

        # 3. 批量生成随机点
        random_points = rng.uniform(
            low=[x_min, y_min],
            high=[x_max, y_max],
            size=(n_samples, 2),
        ).astype(np.float32)

        # 4. 使用 shapely 判断点是否在多边形内
        inside_mask = contains_xy(polygon, random_points[:, 0], random_points[:, 1])
        inside_points = random_points[inside_mask]

        # 5. 如果点数不够，继续采样
        max_attempts = 5
        attempt = 0
        while len(inside_points) < num_points and attempt < max_attempts:
            additional_samples = int((num_points - len(inside_points)) * oversample_factor)
            random_points = rng.uniform(
                low=[x_min, y_min],
                high=[x_max, y_max],
                size=(additional_samples, 2),
            ).astype(np.float32)

            inside_mask = contains_xy(polygon, random_points[:, 0], random_points[:, 1])
            additional_inside = random_points[inside_mask]

            if len(additional_inside) > 0:
                inside_points = np.vstack([inside_points, additional_inside])
            attempt += 1

        # 6. 添加边界点（约占 10%）以确保轮廓清晰
        n_boundary_target = max(int(num_points * 0.1), len(vertices_2d))
        perimeter = polygon.boundary.length

        if perimeter > 0:
            distances = np.linspace(0, perimeter, n_boundary_target, endpoint=False)
            boundary_points = np.array(
                [polygon.boundary.interpolate(d).coords[0] for d in distances],
                dtype=np.float32,
            )
        else:
            boundary_points = vertices_2d

        # 7. 组合点云：优先保留边界点，剩余名额给内部点
        if len(boundary_points) >= num_points:
            indices = np.linspace(0, len(boundary_points) - 1, num_points, dtype=int)
            sampled_points = boundary_points[indices]
        else:
            n_needed = num_points - len(boundary_points)
            if len(inside_points) >= n_needed:
                indices = rng.choice(len(inside_points), n_needed, replace=False)
                sampled_interior = inside_points[indices]
                sampled_points = np.vstack([boundary_points, sampled_interior])
            else:
                if len(inside_points) > 0:
                    sampled_points = np.vstack([boundary_points, inside_points])
                else:
                    sampled_points = boundary_points

        # 8. 转换为 3D 点云（Z=0）
        floor_points = np.column_stack([
            sampled_points[:, 0],
            sampled_points[:, 1],
            np.zeros(len(sampled_points)),
        ])

        return floor_points.astype(np.float32)

    except Exception as e:
        raise RuntimeError(f"Floor pointcloud sampling failed for boundary={boundary}") from e


def build_pointcloud(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    normals: Optional[np.ndarray],
) -> np.ndarray:
    """
    构建 (N, 9) 点云: [x, y, z, r, g, b, nx, ny, nz]
    """
    points = np.asarray(points, dtype=np.float32)
    n_points = len(points)

    if colors is None:
        colors = np.full((n_points, 3), 0.5, dtype=np.float32)
    else:
        colors = np.asarray(colors, dtype=np.float32)

    if normals is None:
        normals = np.zeros((n_points, 3), dtype=np.float32)
        normals[:, 2] = 1.0
    else:
        normals = np.asarray(normals, dtype=np.float32)

    return np.concatenate([points, colors, normals], axis=1)


def save_pointcloud_as_ply(pointcloud: np.ndarray, file_path: Path):
    """
    保存点云到 PLY

    (N, 9): x, y, z, r, g, b, nx, ny, nz
    """
    if pointcloud.ndim != 2:
        raise ValueError(f"Expected 2D pointcloud, got shape {pointcloud.shape}")
    if pointcloud.shape[1] != 9:
        raise ValueError(f"Expected pointcloud shape (N, 9), got {pointcloud.shape}")

    n_points = pointcloud.shape[0]

    vertices = np.zeros(
        n_points,
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("nx", "f4"),
            ("ny", "f4"),
            ("nz", "f4"),
        ],
    )
    vertices["x"] = pointcloud[:, 0]
    vertices["y"] = pointcloud[:, 1]
    vertices["z"] = pointcloud[:, 2]
    vertices["red"] = (np.clip(pointcloud[:, 3], 0, 1) * 255).astype(np.uint8)
    vertices["green"] = (np.clip(pointcloud[:, 4], 0, 1) * 255).astype(np.uint8)
    vertices["blue"] = (np.clip(pointcloud[:, 5], 0, 1) * 255).astype(np.uint8)
    vertices["nx"] = pointcloud[:, 6]
    vertices["ny"] = pointcloud[:, 7]
    vertices["nz"] = pointcloud[:, 8]

    ply_element = PlyElement.describe(vertices, "vertex")
    PlyData([ply_element], text=False).write(str(file_path))


# ──────────────────────────────────────────────────────────────────
# 点云体素碰撞与支撑/悬空检测（评测与 Agent 验证共用单一来源）
# ──────────────────────────────────────────────────────────────────


def place_new_object(
    points: np.ndarray,
    predicted_center: Sequence[float],
    predicted_rotation_deg: Sequence[float],
    target_size: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """将资产局部点云按包围盒居中、可选缩放至 target_size、旋转并平移到世界坐标系。"""
    pts = np.asarray(points[:, :3], dtype=np.float64).copy()

    bbox_min = pts.min(axis=0)
    bbox_max = pts.max(axis=0)
    bbox_center = (bbox_max + bbox_min) / 2.0
    pts = pts - bbox_center

    if target_size is not None:
        bbox_extent = bbox_max - bbox_min
        target = np.asarray(target_size, dtype=np.float64)
        if target.shape != (3,):
            raise ValueError(f"target_size must contain three values, got {target_size!r}")
        if np.any(target <= 0):
            raise ValueError(f"target_size values must be positive, got {target_size!r}")
        scale = np.ones(3, dtype=np.float64)
        non_degenerate = bbox_extent > 0
        scale[non_degenerate] = target[non_degenerate] / bbox_extent[non_degenerate]
        pts = pts * scale

    rot = R.from_euler("xyz", predicted_rotation_deg, degrees=True).as_matrix()
    pts = (rot @ pts.T).T
    pts += np.asarray(predicted_center, dtype=np.float64)
    return pts


def voxelize_pointcloud(
    points: np.ndarray, voxel_size: float = 0.02
) -> set[tuple[int, int, int]]:
    """将点云离散化为占据体素集合。"""
    coords = np.floor(points[:, :3] / voxel_size).astype(np.int64)
    return set(map(tuple, coords))


def compute_pointcloud_collision(
    new_pts: np.ndarray,
    existing_pts_list: Sequence[np.ndarray],
    voxel_size: float = 0.02,
    overlap_threshold: int = 5,
    overlap_ratio_threshold: float = 0.20,
) -> Dict[str, Any]:
    """基于点云体素重叠率检测新物体与已有物体的碰撞（天然支持柜内/架内等镂空几何）。"""
    new_voxels = voxelize_pointcloud(new_pts, voxel_size)
    new_voxel_count = len(new_voxels)

    per_object_overlaps: List[int] = []
    for obj_pts in existing_pts_list:
        obj_voxels = voxelize_pointcloud(obj_pts, voxel_size)
        per_object_overlaps.append(len(new_voxels & obj_voxels))

    total_overlap = sum(per_object_overlaps)
    overlap_volume = total_overlap * (voxel_size ** 3)

    if new_voxel_count > 0:
        overlap_ratio = total_overlap / new_voxel_count
        has_collision = overlap_ratio > overlap_ratio_threshold
    else:
        overlap_ratio = 0.0
        has_collision = total_overlap >= overlap_threshold

    return {
        "has_collision": has_collision,
        "overlap_voxels": total_overlap,
        "overlap_volume": round(overlap_volume, 6),
        "overlap_ratio": round(overlap_ratio, 4),
        "per_object_overlaps": per_object_overlaps,
        "new_voxel_count": new_voxel_count,
    }


def compute_pointcloud_floating(
    new_pts: np.ndarray,
    floor_pts: np.ndarray,
    existing_pts_list: Sequence[np.ndarray],
    voxel_size: float = 0.02,
    gap_threshold: float = 0.05,
) -> Dict[str, Any]:
    """基于底面 10% 点云与 2D 支撑高度图（5x5 体素邻域）检测悬空与支撑率。"""
    z_vals = new_pts[:, 2]
    z_10th = float(np.percentile(z_vals, 10))
    bottom_pts = new_pts[z_vals <= z_10th]

    if len(bottom_pts) == 0:
        return {
            "is_floating": False,
            "is_sinking": False,
            "support_ratio": 1.0,
            "median_gap": 0.0,
            "ground_contact": False,
        }

    support_map: dict[tuple[int, int], float] = {}

    def _add_to_support(points: np.ndarray) -> None:
        for pt in points:
            vx = int(np.floor(pt[0] / voxel_size))
            vy = int(np.floor(pt[1] / voxel_size))
            key = (vx, vy)
            cur = support_map.get(key, -1e9)
            if pt[2] > cur:
                support_map[key] = float(pt[2])

    if len(floor_pts) > 0:
        _add_to_support(floor_pts)

    for obj_pts in existing_pts_list:
        below_mask = obj_pts[:, 2] <= z_10th + gap_threshold
        if below_mask.any():
            _add_to_support(obj_pts[below_mask])

    gaps: list[float] = []
    supported_count = 0
    for pt in bottom_pts:
        vx = int(np.floor(pt[0] / voxel_size))
        vy = int(np.floor(pt[1] / voxel_size))
        best_support_z = -1e9
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                sz = support_map.get((vx + dx, vy + dy), -1e9)
                if sz > best_support_z:
                    best_support_z = sz
        gap = float(pt[2] - best_support_z)
        gaps.append(gap)
        if gap <= gap_threshold:
            supported_count += 1

    support_ratio = supported_count / len(bottom_pts)
    median_gap = float(np.median(gaps))

    is_sinking = median_gap < -gap_threshold
    is_floating = support_ratio < 0.3 or is_sinking

    return {
        "is_floating": is_floating,
        "is_sinking": is_sinking,
        "support_ratio": round(support_ratio, 4),
        "median_gap": round(median_gap, 4),
        "ground_contact": False,
    }
