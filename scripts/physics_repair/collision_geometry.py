"""场景碰撞几何:VHACD 凸分解 + 凸块 XY 投影,作为求解器唯一的碰撞原语。

与 Blender 无关,运行在 python 环境。每件物体被 VHACD 分解为若干凸块(canonical 网格上
分解一次、按 jid 落盘缓存),缩放/旋转烘进凸块顶点,逐轮只更新平移(中心)。

碰撞量统一为「**仅水平移动**去除真实凸块重叠所需的最小 XY 位移」:对每对凸块,
若其 **Z 区间相交**(避免把上下错层的件误判为碰撞),则在 XY 投影上做 2D SAT 取
MTV,整对取最深者。这取代了旧实现的 OBB(过估)与 FCL/EPA(最小分离方向常为竖直、
投影到 XY 后失真)。`pair_xy_penetration` 直接给出去穿插推力,既是检测也是方向。

资产网格加载复用 `export_layout_scene._load_base_mesh`,几何与导出 GLB 一致。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import trimesh
from scipy.spatial import ConvexHull, QhullError

_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR.parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_THIS_DIR), str(_SCRIPTS_DIR), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.convex_decomposition import (  # noqa: E402
    build_transform_matrix as _build_transform_matrix,
    decompose_canonical as _shared_decompose,
    load_canonical_mesh as _load_base_mesh,
)

# 接触深度小于该阈值视为「贴合」而非穿插,不产生推力(米)。
CONTACT_EPS = 1e-4


def _decompose_canonical(jid: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """对 canonical 网格做 VHACD 凸分解(委托共享模块,与推理期碰撞共用缓存)。"""
    return _shared_decompose(jid, _load_base_mesh(jid))


def _convex_hull_2d(points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """返回 (有序凸包顶点(K,2), 单位外法线(K,2));点退化(共线)返回 None。"""
    if len(points_xy) < 3:
        return None
    try:
        hull = ConvexHull(points_xy)
    except QhullError:
        return None
    verts = points_xy[hull.vertices]  # 逆时针
    edges = np.roll(verts, -1, axis=0) - verts
    normals = np.stack([edges[:, 1], -edges[:, 0]], axis=1)  # 右手外法线
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    keep = lengths[:, 0] > 1e-9
    if not np.any(keep):
        return None
    normals = normals[keep] / lengths[keep]
    return verts, normals


@dataclass
class _Piece:
    hull_xy: np.ndarray  # (K,2) oriented(缩放+旋转后)凸包顶点,绕原点
    normals: np.ndarray  # (K,2) 单位外法线
    z_min: float
    z_max: float


@dataclass
class ObjectGeom:
    """单件物体的碰撞代理:多个凸块的 XY 投影 + Z 区间 + 可变平移(中心)。"""

    obj_id: int
    center: np.ndarray  # (3,) 当前世界中心
    pieces: list[_Piece] = field(default_factory=list)
    _xy_min: np.ndarray = field(default_factory=lambda: np.zeros(2))  # oriented 包络(绕原点)
    _xy_max: np.ndarray = field(default_factory=lambda: np.zeros(2))
    _z_min: float = 0.0  # oriented Z 区间(绕原点)
    _z_max: float = 0.0

    def set_center(self, center: Sequence[float]) -> None:
        self.center = np.asarray(center, dtype=np.float64)

    @property
    def aabb_xy(self) -> tuple[np.ndarray, np.ndarray]:
        c = self.center[:2]
        return self._xy_min + c, self._xy_max + c

    @property
    def z_range(self) -> tuple[float, float]:
        cz = float(self.center[2])
        return cz + self._z_min, cz + self._z_max

    def aabb_at(self, center: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        """不改状态,返回该件假设位于 center 时的世界 XY 包络(用于 boundary footprint)。"""
        c = np.asarray(center, dtype=np.float64)[:2]
        return self._xy_min + c, self._xy_max + c


def build_object_geom(obj_data: Mapping[str, Any]) -> ObjectGeom:
    """从场景 object(含 jid/size/center/rotation)构建碰撞几何。

    各向异性缩放与旋转烘进凸块顶点(对凸块仍是线性变换,保持凸性),平移随中心逐轮更新。
    每个凸块取其 XY 投影凸包与 Z 区间(均在 oriented 坐标、绕原点)。
    """
    for key in ("id", "jid", "size", "center"):
        if key not in obj_data:
            raise KeyError(f"object 缺少字段 {key}: {obj_data}")

    obj_id = int(obj_data["id"])
    jid = str(obj_data["jid"])
    size = np.asarray(obj_data["size"], dtype=np.float64)
    center = np.asarray(obj_data["center"], dtype=np.float64)
    rotation = obj_data["rotation"] if "rotation" in obj_data else [0.0, 0.0, 0.0]

    base = _load_base_mesh(jid)
    mesh_size = np.asarray(base.bounds[1] - base.bounds[0], dtype=np.float64)
    scale = size / np.maximum(mesh_size, 1e-6)
    rot_only = _build_transform_matrix(center=[0.0, 0.0, 0.0], rotation=rotation, scale=np.ones(3))
    rotation_matrix = np.asarray(rot_only[:3, :3], dtype=np.float64)

    pieces: list[_Piece] = []
    all_xy: list[np.ndarray] = []
    for verts, _faces in _decompose_canonical(jid):
        oriented = (verts * scale) @ rotation_matrix.T  # 绕原点的世界朝向顶点
        hull = _convex_hull_2d(oriented[:, :2])
        if hull is None:
            continue
        hull_xy, normals = hull
        pieces.append(
            _Piece(
                hull_xy=hull_xy,
                normals=normals,
                z_min=float(oriented[:, 2].min()),
                z_max=float(oriented[:, 2].max()),
            )
        )
        all_xy.append(hull_xy)

    if not pieces:
        raise ValueError(f"object {obj_id} (jid={jid}) 无有效碰撞凸块")

    stacked = np.concatenate(all_xy, axis=0)
    geom = ObjectGeom(
        obj_id=obj_id,
        center=center,
        pieces=pieces,
        _xy_min=stacked.min(axis=0),
        _xy_max=stacked.max(axis=0),
        _z_min=min(p.z_min for p in pieces),
        _z_max=max(p.z_max for p in pieces),
    )
    geom.set_center(center)
    return geom


def build_scene_geoms(objects: Sequence[Mapping[str, Any]]) -> dict[int, ObjectGeom]:
    return {int(o["id"]): build_object_geom(o) for o in objects}


def build_world_trimesh(obj_data: Mapping[str, Any], transform: Any) -> trimesh.Trimesh:
    """按给定位姿构建世界坐标 trimesh,供 Z 贴面下射使用。"""
    jid = str(obj_data["jid"])
    size = np.asarray(obj_data["size"], dtype=np.float64)
    center = list(transform.center)
    rotation = list(transform.rotation)

    base = _load_base_mesh(jid)
    mesh = base.copy()
    mesh_size = np.asarray(mesh.bounds[1] - mesh.bounds[0], dtype=np.float64)
    scale = size / np.maximum(mesh_size, 1e-6)
    mesh.apply_transform(_build_transform_matrix(center=center, rotation=rotation, scale=scale))
    return mesh


# --------------------------------------------------------------------------- #
# 2D SAT(凸多边形最小平移向量)
# --------------------------------------------------------------------------- #
def _project(points: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
    proj = points @ axis
    return float(proj.min()), float(proj.max())


def _poly_xy_mtv(
    verts_a: np.ndarray,
    normals_a: np.ndarray,
    verts_b: np.ndarray,
    normals_b: np.ndarray,
) -> tuple[np.ndarray, float] | None:
    """两个凸多边形(逆时针顶点)的 2D SAT;相交返回 (单位轴, 穿透深度),否则 None。
    轴方向未定向(调用方按质心方向定向)。"""
    best_depth = float("inf")
    best_axis: np.ndarray | None = None
    for axis in (*normals_a, *normals_b):
        min_a, max_a = _project(verts_a, axis)
        min_b, max_b = _project(verts_b, axis)
        overlap = min(max_a, max_b) - max(min_a, min_b)
        if overlap <= 0.0:
            return None
        if overlap < best_depth:
            best_depth = overlap
            best_axis = axis
    if best_axis is None:
        return None
    return best_axis, best_depth


def _aabb_xy_overlap(
    a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray], margin: float
) -> bool:
    return bool(np.all(a[1] + margin >= b[0]) and np.all(b[1] + margin >= a[0]))


# 「整体出口」救正参数:小件 footprint 面积须 ≤ 大件的此比例,才认定为「嵌入大体块」。
_ESCAPE_AREA_RATIO = 0.5
# 竖直嵌入余量:小件底必须比大件顶低至少这么多、且小件顶比大件底高至少这么多,才算
# 「贯穿/嵌入」(区别于「叠放在上/垫在下」——后者是合理支撑关系,不可横向驱逐)。
_ESCAPE_Z_MARGIN = 0.1


def pair_xy_escape(
    geom_a: ObjectGeom, geom_b: ObjectGeom
) -> tuple[np.ndarray, float] | None:
    """「整体出口」穿透:把**嵌入**大体块的小件整体移出大件 world-AABB 的最小 XY 平移。

    逐块 SAT 的 MTV 只给「最浅逃逸」——细长 / 小件贯穿大体块时(如落地灯杆插进床垫),
    最近逃逸只是沿细方向探出一点点(≈件半径),无法把件移出大件。这里改用「把小件整体
    推出大件 world-AABB」的距离,方向与 `pair_xy_penetration` 一致(由 a 指向 b)。

    仅在以下全部成立时触发(否则返回 None,沿用逐块 SAT):
      1. 小件 footprint ≤ 大件的 ``_ESCAPE_AREA_RATIO``(确属「小 vs 大」);
      2. 小件 XY 中心落在大件 world-AABB 内(确属「嵌在里面」而非边缘擦碰);
      3. 竖直方向真正嵌入:小件底比大件顶低 ≥ ``_ESCAPE_Z_MARGIN`` 且小件顶比大件底高
         ≥ ``_ESCAPE_Z_MARGIN``。这把「叠放在上(vase on table)/垫在下(furniture on
         rug)」排除在外——那是合理支撑,绝不能横向驱逐。
    """
    a_lo, a_hi = geom_a.aabb_xy
    b_lo, b_hi = geom_b.aabb_xy
    area_a = float((a_hi[0] - a_lo[0]) * (a_hi[1] - a_lo[1]))
    area_b = float((b_hi[0] - b_lo[0]) * (b_hi[1] - b_lo[1]))

    small_is_a = area_a <= area_b
    small = geom_a if small_is_a else geom_b
    large = geom_b if small_is_a else geom_a
    s_lo, s_hi = (a_lo, a_hi) if small_is_a else (b_lo, b_hi)
    l_lo, l_hi = (b_lo, b_hi) if small_is_a else (a_lo, a_hi)
    small_area = area_a if small_is_a else area_b
    large_area = area_b if small_is_a else area_a

    if large_area <= 1e-9 or small_area > _ESCAPE_AREA_RATIO * large_area:
        return None
    sc = small.center[:2]
    if not (l_lo[0] <= sc[0] <= l_hi[0] and l_lo[1] <= sc[1] <= l_hi[1]):
        return None
    s_z0, s_z1 = small.z_range
    l_z0, l_z1 = large.z_range
    if not (s_z0 < l_z1 - _ESCAPE_Z_MARGIN and s_z1 > l_z0 + _ESCAPE_Z_MARGIN):
        return None

    candidates = [
        (np.array([1.0, 0.0]), float(l_hi[0] - s_lo[0])),
        (np.array([-1.0, 0.0]), float(s_hi[0] - l_lo[0])),
        (np.array([0.0, 1.0]), float(l_hi[1] - s_lo[1])),
        (np.array([0.0, -1.0]), float(s_hi[1] - l_lo[1])),
    ]
    axis, depth = min(candidates, key=lambda c: c[1])
    if depth <= CONTACT_EPS:
        return None
    # 方向规范为「由 a 指向 b」:小件是 b 时即把 b 推向 axis;小件是 a 时,a 要被推向
    # axis,而 deltas 用 a 朝 -direction、b 朝 +direction,故返回 -axis。
    direction = axis if not small_is_a else -axis
    return direction, depth


def pair_xy_penetration(
    geom_a: ObjectGeom,
    geom_b: ObjectGeom,
    aabb_margin: float = 1e-3,
) -> tuple[np.ndarray, float] | None:
    """返回把 b 推离 a 的 XY 最小平移 ``((dx,dy) 单位方向, depth)``;不接触返回 None。

    仅统计 Z 区间相交的凸块对(避免上下错层误判)。方向规范为「由 a 指向 b」。
    """
    if not _aabb_xy_overlap(geom_a.aabb_xy, geom_b.aabb_xy, aabb_margin):
        return None

    ca_xy, cb_xy = geom_a.center[:2], geom_b.center[:2]
    az, bz = float(geom_a.center[2]), float(geom_b.center[2])
    best_depth = 0.0
    best_axis: np.ndarray | None = None
    for pa in geom_a.pieces:
        a_zmin, a_zmax = pa.z_min + az, pa.z_max + az
        va = pa.hull_xy + ca_xy
        for pb in geom_b.pieces:
            if a_zmax <= pb.z_min + bz or pb.z_max + bz <= a_zmin:
                continue  # Z 区间不相交
            vb = pb.hull_xy + cb_xy
            mtv = _poly_xy_mtv(va, pa.normals, vb, pb.normals)
            if mtv is None:
                continue
            axis, depth = mtv
            if depth > best_depth:
                best_depth = depth
                best_axis = axis

    if best_axis is None or best_depth <= CONTACT_EPS:
        return None

    direction = best_axis / (np.linalg.norm(best_axis) + 1e-12)
    if float(np.dot(direction, cb_xy - ca_xy)) < 0.0:
        direction = -direction
    return direction, best_depth
