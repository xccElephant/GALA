import os
from typing import Final

import numpy as np

from .pcd_loader import get_points_and_colors, load_o3d_pcd

NORMALIZATION_PRESET: Final[dict[str, tuple[float, float]]] = {
    "world": (0.0, 32.0),
    "height": (0.0, 25.6),
    "angle": (-6.2832, 6.2832),
}
DEFAULT_FLOOR_GRID_SCALE: Final[float] = 2.0


def _is_floor_point_cloud_path(pcd_path: str) -> bool:
    return os.path.basename(str(pcd_path)).lower().startswith("floor")


def resolve_point_cloud_path(pcd_path: str) -> str:
    """Resolve relative or relocated point cloud paths against DATASETS_ROOT or <repo>/datasets."""
    p = str(pcd_path).strip()
    if os.path.isfile(p):
        return p
    if "pointclouds/" in p:
        rel = p.split("pointclouds/", 1)[1]
        datasets_root = os.environ.get("DATASETS_ROOT") or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "datasets")
        )
        return os.path.join(datasets_root, "pointclouds", rel)
    return p


def _fnv_hash_vec(arr: np.ndarray) -> np.ndarray:
    """FNV64-1A hash for grid coordinate deduplication."""
    assert arr.ndim == 2
    arr = arr.copy().astype(np.uint64, copy=False)
    hashed = np.uint64(14695981039346656037) * np.ones(arr.shape[0], dtype=np.uint64)
    for j in range(arr.shape[1]):
        hashed *= np.uint64(1099511628211)
        hashed = np.bitwise_xor(hashed, arr[:, j])
    return hashed


def preprocess_object_point_cloud_utonia(
    pcd_path: str,
    scale: float = 1.0,
    utonia_grid_size: float = 0.02,
    normalize_coord: bool = False,
    center_shift: bool = False,
    floor_grid_scale: float = DEFAULT_FLOOR_GRID_SCALE,
) -> np.ndarray:
    """Preprocess a point cloud for the Utonia encoder.

    Transform pipeline (each step is optional):
      NormalizeCoord -> Scale -> CenterShift -> GridSample(fnv) -> NormalizeColor(/255)

    When ``normalize_coord=False`` and ``center_shift=False``, world coordinates
    are preserved so the encoder can see absolute position — important for
    scene-layout tasks where text references world-frame poses.

    Returns (N', 12) = [grid_x, grid_y, grid_z, x, y, z, r, g, b, nx, ny, nz].
    """
    pcd = load_o3d_pcd(pcd_path)
    points, colors = get_points_and_colors(pcd, pcd_path)

    normals = np.asarray(pcd.normals)
    if normals.size == 0 or normals.shape[0] != points.shape[0]:
        print(f"Warning: Point cloud {pcd_path} has no normals")
        normals = np.zeros_like(points)

    # NormalizeCoord (for objects: centroid-shift + unit-sphere)
    if normalize_coord:
        centroid = points.mean(axis=0)
        points = points - centroid
        m = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
        if m > 0:
            points = points / m

    # Scale
    points = points * scale

    # CenterShift
    if center_shift:
        x_min, y_min, z_min = points.min(axis=0)
        x_max, y_max, _ = points.max(axis=0)
        shift = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, z_min])
        points = points - shift

    # GridSample(fnv hash) — enlarge grid for floor point clouds
    effective_grid_size = (
        utonia_grid_size * floor_grid_scale
        if _is_floor_point_cloud_path(pcd_path)
        else utonia_grid_size
    )
    scaled_coord = points / effective_grid_size
    grid_coord = np.floor(scaled_coord).astype(np.int64)
    min_gc = grid_coord.min(axis=0)
    grid_coord -= min_gc

    key = _fnv_hash_vec(grid_coord)
    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]
    _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)

    idx_select = (
        np.cumsum(np.insert(count, 0, 0)[:-1])
        + np.random.randint(0, count.max(), count.size) % count
    )
    idx_unique = idx_sort[idx_select]

    grid_coord = grid_coord[idx_unique]
    points = points[idx_unique]
    colors = colors[idx_unique]
    normals = normals[idx_unique]

    # 5) NormalizeColor (/255)
    colors_normalized = colors.astype(np.float32) / 255.0

    # output: [grid_coord(3), coord(3), color(3), normal(3)] = 12 dims
    result = np.concatenate(
        [grid_coord.astype(np.float32), points.astype(np.float32),
         colors_normalized, normals.astype(np.float32)],
        axis=1,
    )
    return result


def _npy_cache_path(pcd_path: str, backbone_type: str) -> str:
    base, _ = os.path.splitext(pcd_path)
    return f"{base}.{backbone_type}.npy"


def load_cached_or_preprocess(
    pcd_path: str,
    backbone_type: str,
    floor_grid_scale: float = DEFAULT_FLOOR_GRID_SCALE,
    utonia_scale: float = 1.0,
    utonia_grid_size: float = 0.02,
    utonia_normalize_coord: bool = False,
    utonia_center_shift: bool = False,
) -> np.ndarray:
    """Load pre-cached .npy if available, otherwise fall back to full preprocessing."""
    resolved_path = resolve_point_cloud_path(pcd_path)
    cache = _npy_cache_path(resolved_path, backbone_type)
    if os.path.isfile(cache):
        return np.load(cache)
    return preprocess_object_point_cloud(
        resolved_path,
        backbone_type=backbone_type,
        floor_grid_scale=floor_grid_scale,
        utonia_scale=utonia_scale,
        utonia_grid_size=utonia_grid_size,
        utonia_normalize_coord=utonia_normalize_coord,
        utonia_center_shift=utonia_center_shift,
    )


def preprocess_object_point_cloud(
    pcd_path: str,
    backbone_type: str,
    floor_grid_scale: float = DEFAULT_FLOOR_GRID_SCALE,
    utonia_scale: float = 1.0,
    utonia_grid_size: float = 0.02,
    utonia_normalize_coord: bool = False,
    utonia_center_shift: bool = False,
) -> np.ndarray:
    if backbone_type != "utonia":
        raise ValueError(f"Unsupported object backbone type: {backbone_type}")
    resolved_path = resolve_point_cloud_path(pcd_path)
    return preprocess_object_point_cloud_utonia(
        resolved_path,
        scale=utonia_scale,
        utonia_grid_size=utonia_grid_size,
        normalize_coord=utonia_normalize_coord,
        center_shift=utonia_center_shift,
        floor_grid_scale=floor_grid_scale,
    )


def preprocess_point_cloud_batch(
    pcd_paths: list[str],
    backbone_type: str,
    floor_grid_scale: float = DEFAULT_FLOOR_GRID_SCALE,
    utonia_scale: float = 1.0,
    utonia_grid_size: float = 0.02,
    utonia_normalize_coord: bool = False,
    utonia_center_shift: bool = False,
) -> np.ndarray:
    """加载并预处理一批点云文件，返回 NaN-padded ``(N, max_pts, 12)`` ndarray。

    结果可直接用 ``torch.as_tensor()`` 转为 tensor 送入模型。
    """
    points_list = []
    max_len = 0
    for pcd_path in pcd_paths:
        point_feats = load_cached_or_preprocess(
            str(pcd_path),
            backbone_type=backbone_type,
            floor_grid_scale=floor_grid_scale,
            utonia_scale=utonia_scale,
            utonia_grid_size=utonia_grid_size,
            utonia_normalize_coord=utonia_normalize_coord,
            utonia_center_shift=utonia_center_shift,
        )
        max_len = max(max_len, len(point_feats))
        points_list.append(point_feats)

    for i in range(len(points_list)):
        points_list[i] = np.pad(
            points_list[i],
            ((0, max_len - len(points_list[i])), (0, 0)),
            mode="constant",
            constant_values=np.nan,
        )

    return np.stack(points_list, axis=0)


def make_random_noise_pcd_batch(
    n_clouds: int,
    num_bins: int = 1280,
    num_points: int = 1024,
) -> np.ndarray:
    """生成一批随机噪声点云，用于消融实验。返回 grid + coord + color + normal。"""
    extent = NORMALIZATION_PRESET["world"][1] - NORMALIZATION_PRESET["world"][0]
    clouds = []
    for _ in range(n_clouds):
        grid_coords = np.random.randint(0, num_bins, (num_points, 3), dtype=np.int32).astype(np.float32)
        xyz = np.random.uniform(0.0, extent, (num_points, 3)).astype(np.float32)
        rgb = np.random.uniform(-1.0, 1.0, (num_points, 3)).astype(np.float32)
        normals = np.random.uniform(-1.0, 1.0, (num_points, 3)).astype(np.float32)
        clouds.append(np.concatenate([grid_coords, xyz, rgb, normals], axis=1))
    return np.stack(clouds, axis=0)
