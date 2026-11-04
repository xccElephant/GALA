"""共享的 Canonical 网格加载、变换矩阵构建与 VHACD 凸分解缓存（单一来源）。

供后处理求解器（``scripts/physics_repair/``）与场景网格导出（``scripts/export_layout_scene.py``）
共同复用。

VHACD 分解在 **canonical（未缩放）网格** 上做一次、按 ``jid`` 落盘到 ``output/vhacd_cache``；
缩放 / 旋转由调用方在运行时烘进顶点（凸分解对线性变换封闭，先分解后缩放等价于先缩放后分解）。
"""

from __future__ import annotations

from functools import lru_cache
import pickle
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R
import trimesh

from config.paths import ProjectPaths

VHACD_MAX_HULLS = 16
VHACD_RESOLUTION = 100000
CACHE_VERSION = 2
CACHE_DIR = ProjectPaths.OUTPUT_DIR / "vhacd_cache"


def _transform_coordinates_y_to_z(points: np.ndarray) -> np.ndarray:
    """将资产坐标从 Y-up 转到项目统一使用的 Z-up：(x, y, z) -> (x, -z, y)。"""
    transform_matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    return points @ transform_matrix.T


def resolve_mesh_source(
    jid: str, *, include_non_obj: bool = False
) -> Tuple[Optional[Path], Optional[str]]:
    """解析资产网格来源（支持 3D-FUTURE 与 Imaginarium）。"""
    candidates = [
        (ProjectPaths.THREED_FUTURE_MODEL_DIR / jid / "raw_model.obj", "3dfuture_obj"),
        (ProjectPaths.THREED_FUTURE_MODEL_DIR / jid / f"{jid}.obj", "3dfuture_obj"),
        (ProjectPaths.IMAGINARIUM_ASSETS_DIR / jid / f"{jid}.obj", "imaginarium_obj"),
    ]
    if include_non_obj:
        candidates.extend(
            [
                (ProjectPaths.IMAGINARIUM_ASSETS_DIR / jid / f"{jid}.glb", "imaginarium_glb"),
                (ProjectPaths.IMAGINARIUM_ASSETS_DIR / jid / f"{jid}.gltf", "imaginarium_gltf"),
                (ProjectPaths.IMAGINARIUM_ASSETS_DIR / jid / f"{jid}.fbx", "imaginarium_fbx"),
            ]
        )

    for candidate_path, source in candidates:
        if candidate_path.exists():
            return candidate_path, source
    return None, None


@lru_cache(maxsize=256)
def load_canonical_mesh(
    jid: str,
    *,
    skip_materials: bool = False,
    include_non_obj: bool = True,
) -> trimesh.Trimesh:
    """加载资产 canonical 网格（Y-up -> Z-up 并按包围盒中心居中），带缓存。"""
    mesh_path, source = resolve_mesh_source(jid, include_non_obj=include_non_obj)
    if mesh_path is None or source is None:
        raise FileNotFoundError(f"未找到资产网格: jid={jid}")

    try:
        mesh = trimesh.load(
            str(mesh_path),
            process=False,
            force="mesh",
            skip_materials=skip_materials,
        )
    except Exception as exc:
        raise RuntimeError(
            f"加载资产失败: jid={jid}, path={mesh_path}, error={exc}"
        ) from exc

    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"资产不是 Trimesh: jid={jid}, source={source}")
    if len(mesh.vertices) == 0:
        raise ValueError(f"资产没有顶点: jid={jid}, source={source}")

    mesh = mesh.copy()
    mesh.vertices = _transform_coordinates_y_to_z(
        np.asarray(mesh.vertices, dtype=np.float32)
    )
    bounds = mesh.bounds
    bbox_center = (bounds[0] + bounds[1]) / 2.0
    mesh.apply_translation(-bbox_center)
    return mesh


def build_transform_matrix(
    center: Sequence[float],
    rotation: Sequence[float],
    scale: np.ndarray,
) -> np.ndarray:
    """构建 4x4 变换矩阵（缩放 -> 旋转 -> 平移）。"""
    rot = R.from_euler("xyz", rotation, degrees=True)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot.as_matrix() @ np.diag(scale)
    T[:3, 3] = np.asarray(center, dtype=np.float64)
    return T


def cache_path(jid: str) -> Path:
    return CACHE_DIR / f"{jid}_v{CACHE_VERSION}_h{VHACD_MAX_HULLS}_r{VHACD_RESOLUTION}.pkl"


def decompose_canonical(
    jid: str, base_mesh: Optional[trimesh.Trimesh] = None
) -> list[tuple[np.ndarray, np.ndarray]]:
    """对 canonical 网格做 VHACD 凸分解，返回 ``[(verts(N,3) float64, faces(F,3) int)]``。

    按 ``jid`` 落盘缓存（命中即直接读）。若未传入 ``base_mesh``，自动调用 ``load_canonical_mesh(jid)``。
    """
    cache_file = cache_path(jid)
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    if base_mesh is None:
        base_mesh = load_canonical_mesh(jid)

    import vhacdx  # noqa: F401 - trimesh.convex_decomposition 的 VHACD 后端;缺失时显式报错

    hulls = base_mesh.convex_decomposition(
        maxConvexHulls=VHACD_MAX_HULLS,
        resolution=VHACD_RESOLUTION,
    )
    parts: list[tuple[np.ndarray, np.ndarray]] = [
        (
            np.asarray(hull.vertices, dtype=np.float64),
            np.asarray(hull.faces, dtype=np.int64),
        )
        for hull in hulls
        if len(hull.vertices) >= 4 and len(hull.faces) >= 4
    ]
    if not parts:
        raise RuntimeError(
            f"VHACD 凸分解 jid={jid} 未产出有效凸块(网格空/退化);拒绝回退凸包"
        )

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(parts, f)
    return parts
