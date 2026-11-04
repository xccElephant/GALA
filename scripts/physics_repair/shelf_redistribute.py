r"""多隔间容器换层重分配(python 侧,trimesh.ray 版,作为 XPBD 求解的**末步**后处理)。

对每个多隔间容器(shelf/cabinet 等,见 ``_SHELF_LIKE_CONTAINER_CATEGORIES``),把「与
兄弟件真实重叠 / 悬空 / 骑层板边 / 捅穿层板 / 被竖直构件扎进」的子件,搜索同容器里「净高
够 + XY 无重叠 + 下方确有支撑 + 无内壁穿插」的**最近干净落点**并迁移;找不到干净落点则
保留原位(绝不挪到另一处扎板/悬空位)。

它不是连续约束,而是离散的换层规划,故独立于 XPBD;放在求解**末步**——用去穿插/贴面/
夹紧已定的最终容器位姿摆放,精确摆好的子件之后不再被任何步骤搅动。几何探测用
`trimesh.ray`:支撑面枚举(下射取上向面)+ 容器地面、净高(上射取最近天花板)、骑边/捅穿
(下射)、内壁/竖直分隔(横射取竖向面);落点搜索**逐 XY 列**(容器不同列隔板高度可不同,
不能全局聚类);兄弟重叠用 `collision_geometry.pair_xy_penetration`(真实凸块 XY+Z)。

sofa/pot 等单腔或异形容器不在此列(其"层"实为座垫面假层),由 xpbd_solver.clamp_within_interior
顶内壁处理。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import collision_geometry as cg
import physics_repair_engine as pre
from physics_repair_engine import RepairedTransform

# 适用换层重分配的「多隔间容器」类别:内部有**贯穿 footprint 的水平隔板 + 竖直分隔**,
# 故"逐列支撑面/净高/内壁"射线探测成立。sofa 的"层"实为座垫面(0.03~0.05 假层)、pot 是
# 单腔异形且子件通常 <2,套用换层会乱动,故不在此列——它们仍由 clamp_within_interior 顶内壁。
_SHELF_LIKE_CONTAINER_CATEGORIES = {"shelf", "cabinet"}

_TIER_CLUSTER_EPS = 0.03
_TIER_GRID_NX = 5
_TIER_GRID_NY = 4
_TIER_MIN_CLEARANCE = 0.02
_RELOCATE_FLOAT_TOL = 0.01
_SLOT_GRID_NX = 9
_SLOT_GRID_NY = 6
_SLOT_WALL_MARGIN = 0.005


# --------------------------------------------------------------------------- #
# trimesh.ray 探测(等价于旧实现的 Blender raycast 语义)
# --------------------------------------------------------------------------- #
def _ray_support_z(mesh: Any, x: float, y: float, z_start: float) -> list[float]:
    """从 (x,y,z_start) 下射,返回所有「上向面」(法线 z>0.25)命中点的世界 z。"""
    locs, _idx_ray, idx_tri = mesh.ray.intersects_location(
        np.array([[x, y, z_start]], dtype=np.float64),
        np.array([[0.0, 0.0, -1.0]], dtype=np.float64),
        multiple_hits=True,
    )
    out: list[float] = []
    for loc, tri in zip(locs, idx_tri):
        if float(mesh.face_normals[tri][2]) > 0.25:
            out.append(float(loc[2]))
    return out


def _ray_vertical_dists(
    mesh: Any, origin: Sequence[float], direction: Sequence[float]
) -> list[float]:
    """沿 direction 从 origin 射出,返回所有「竖向面」(与方向对齐且 |法线.z|<0.5)命中距离。"""
    d = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(d))
    if norm < 1e-9:
        return []
    d = d / norm
    o = np.asarray(origin, dtype=np.float64)
    locs, _idx_ray, idx_tri = mesh.ray.intersects_location(
        o[None, :], d[None, :], multiple_hits=True
    )
    out: list[float] = []
    for loc, tri in zip(locs, idx_tri):
        normal = mesh.face_normals[tri]
        if abs(float(normal @ d)) > 0.25 and abs(float(normal[2])) < 0.5:
            out.append(float(np.linalg.norm(loc - o)))
    return out


def _footprint_samples(
    cx: float, cy: float, half_x: float, half_y: float, inset: float = 0.6
) -> list[tuple[float, float]]:
    return [
        (cx, cy),
        (cx - half_x * inset, cy - half_y * inset),
        (cx - half_x * inset, cy + half_y * inset),
        (cx + half_x * inset, cy - half_y * inset),
        (cx + half_x * inset, cy + half_y * inset),
    ]


def _enumerate_tiers(
    mesh: Any, mins: np.ndarray, maxs: np.ndarray
) -> list[tuple[float, float]]:
    z_hits: list[float] = []
    for ix in range(_TIER_GRID_NX):
        for iy in range(_TIER_GRID_NY):
            x = mins[0] + (maxs[0] - mins[0]) * (ix + 0.5) / _TIER_GRID_NX
            y = mins[1] + (maxs[1] - mins[1]) * (iy + 0.5) / _TIER_GRID_NY
            z_hits.extend(_ray_support_z(mesh, x, y, maxs[2] + 0.05))
    z_hits.sort()
    clusters: list[float] = []
    for z in z_hits:
        if not clusters or z - clusters[-1] > _TIER_CLUSTER_EPS:
            clusters.append(z)
    tiers: list[tuple[float, float]] = []
    for i, z in enumerate(clusters):
        ceiling = clusters[i + 1] if i + 1 < len(clusters) else float(maxs[2])
        if ceiling - z < _TIER_MIN_CLEARANCE:
            continue
        tiers.append((z, ceiling))
    return tiers


def _nearest_tier_index(tiers: list[tuple[float, float]], bottom_z: float) -> int:
    return min(range(len(tiers)), key=lambda i: abs(tiers[i][0] - bottom_z))


def _tier_board_supports_footprint(
    mesh: Any, cx: float, cy: float, half_x: float, half_y: float, tier_z: float
) -> bool:
    for sx, sy in _footprint_samples(cx, cy, half_x, half_y):
        hits = _ray_support_z(mesh, sx, sy, tier_z + 0.1)
        if not any(abs(h - tier_z) <= _TIER_CLUSTER_EPS for h in hits):
            return False
    return True


def _board_crosses_body(
    mesh: Any, cx: float, cy: float, half_x: float, half_y: float, bottom_z: float, top_z: float
) -> bool:
    lo = bottom_z + _TIER_CLUSTER_EPS
    hi = top_z - _TIER_CLUSTER_EPS
    if hi <= lo:
        return False
    for sx, sy in _footprint_samples(cx, cy, half_x, half_y, inset=0.9):
        hits = _ray_support_z(mesh, sx, sy, top_z + 0.05)
        if any(lo < h < hi for h in hits):
            return True
    return False


def _ceiling_z(
    mesh: Any, cx: float, cy: float, half_x: float, half_y: float, z_from: float
) -> float:
    """从 footprint 各采样点 z_from 向上射线,返回最近的「天花板」世界 z(任意朝向面)。

    用于算某支撑面上方的真实净高;无命中(开放到架顶以上)返回 +inf。取 footprint 上**最低**
    的天花板(最保守),保证件不会扎到任一处上方隔板。"""
    ceiling = math.inf
    for sx, sy in _footprint_samples(cx, cy, half_x, half_y, inset=0.9):
        locs, _idx_ray, _idx_tri = mesh.ray.intersects_location(
            np.array([[sx, sy, z_from]], dtype=np.float64),
            np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
            multiple_hits=True,
        )
        if len(locs) == 0:
            continue
        above = [float(z) for z in locs[:, 2] if z > z_from + 1e-4]
        if above:
            ceiling = min(ceiling, min(above))
    return ceiling


def _footprint_internal_surface(
    mesh: Any,
    cx: float,
    cy: float,
    z_samples: Sequence[float],
    half_x: float,
    half_y: float,
    edge_margin: float = 0.005,
) -> bool:
    x0, x1 = cx - half_x, cx + half_x
    y0, y1 = cy - half_y, cy + half_y
    fracs = (0.15, 0.35, 0.5, 0.65, 0.85)
    for fy in fracs:
        sy = y0 + (y1 - y0) * fy
        for z in z_samples:
            for d in _ray_vertical_dists(mesh, [x0 - 0.002, sy, z], [1.0, 0.0, 0.0]):
                hx = x0 - 0.002 + d
                if x0 + edge_margin < hx < x1 - edge_margin:
                    return True
    for fx in fracs:
        sx = x0 + (x1 - x0) * fx
        for z in z_samples:
            for d in _ray_vertical_dists(mesh, [sx, y0 - 0.002, z], [0.0, 1.0, 0.0]):
                hy = y0 - 0.002 + d
                if y0 + edge_margin < hy < y1 - edge_margin:
                    return True
    return False


def _find_container_support_z(
    mesh: Any, c_mins: np.ndarray, c_maxs: np.ndarray, geom: cg.ObjectGeom
) -> float | None:
    mins_xy, maxs_xy = geom.aabb_xy
    bottom, top = geom.z_range
    cx = (float(mins_xy[0]) + float(maxs_xy[0])) / 2.0
    cy = (float(mins_xy[1]) + float(maxs_xy[1])) / 2.0
    hx = (float(maxs_xy[0]) - float(mins_xy[0])) / 2.0
    hy = (float(maxs_xy[1]) - float(mins_xy[1])) / 2.0
    z_start = float(c_maxs[2]) + 0.05
    max_support_z = float(geom.center[2]) + max(0.05, (top - bottom) * 0.5)
    inset = 0.2
    samples = [
        (cx, cy),
        (cx - hx * (1 - inset), cy - hy * (1 - inset)),
        (cx - hx * (1 - inset), cy + hy * (1 - inset)),
        (cx + hx * (1 - inset), cy - hy * (1 - inset)),
        (cx + hx * (1 - inset), cy + hy * (1 - inset)),
    ]
    candidates: list[float] = []
    for sx, sy in samples:
        for sz in _ray_support_z(mesh, sx, sy, z_start):
            if float(c_mins[2]) - 0.05 <= sz <= max_support_z:
                candidates.append(sz)
    if not candidates:
        return None
    return min(candidates, key=lambda z: abs(z - bottom))


def redistribute_container_children(
    objects: Sequence[Mapping[str, Any]],
    geoms: Mapping[int, cg.ObjectGeom],
    transforms: Mapping[int, RepairedTransform],
    originals: Mapping[int, RepairedTransform],
) -> tuple[dict[int, RepairedTransform], dict[int, RepairedTransform]]:
    """把多隔间容器(shelf/cabinet 等,见 ``_SHELF_LIKE_CONTAINER_CATEGORIES``)里
    「悬空/扎板/骑层板边/撞兄弟」的 within 子件搬进同容器真实存在的干净隔间。

    返回 (新 transforms, 新 originals);被迁移件以新位为锚。"""
    out = dict(transforms)
    out_orig = dict(originals)
    obj_by_id = {int(o["id"]): o for o in objects}
    for oid, g in geoms.items():
        g.set_center(out[oid].center)

    groups: dict[int, list[int]] = {}
    for so in objects:
        for target in pre._expected_containment_target_ids(so):
            groups.setdefault(int(target), []).append(int(so["id"]))

    support_ids: set[int] = set()
    for obj in objects:
        for t in pre._expected_on_target_ids(obj):
            support_ids.add(int(t))
        for t in pre._expected_containment_target_ids(obj):
            support_ids.add(int(t))

    for cid in sorted(groups):
        if (
            cid not in obj_by_id
            or str(obj_by_id[cid].get("category", ""))
            not in _SHELF_LIKE_CONTAINER_CATEGORIES
        ):
            continue
        children = [c for c in groups[cid] if c in geoms and c in out]
        if len(children) < 2:
            continue

        mesh = cg.build_world_trimesh(obj_by_id[cid], out[cid])
        c_mins = np.asarray(mesh.bounds[0], dtype=np.float64)
        c_maxs = np.asarray(mesh.bounds[1], dtype=np.float64)
        tiers = _enumerate_tiers(mesh, c_mins, c_maxs)
        if not tiers:
            continue
        interior = (
            float(c_mins[0]) + _SLOT_WALL_MARGIN,
            float(c_maxs[0]) - _SLOT_WALL_MARGIN,
            float(c_mins[1]) + _SLOT_WALL_MARGIN,
            float(c_maxs[1]) - _SLOT_WALL_MARGIN,
        )

        geo: dict[int, dict[str, float]] = {}
        for c in children:
            mins_xy, maxs_xy = geoms[c].aabb_xy
            bottom, top = geoms[c].z_range
            geo[c] = {
                "half_x": (float(maxs_xy[0]) - float(mins_xy[0])) / 2.0,
                "half_y": (float(maxs_xy[1]) - float(mins_xy[1])) / 2.0,
                "height": top - bottom,
                "bottom": bottom,
                "top": top,
                "z_offset": float(out[c].center[2]) - bottom,
            }

        bad: list[int] = []
        for c in children:
            if c in support_ids:
                continue
            cx, cy = out[c].center[0], out[c].center[1]
            g = geo[c]
            sibling_clash = any(
                cg.pair_xy_penetration(geoms[c], geoms[other]) is not None
                for other in children
                if other != c
            )
            sup_z = _find_container_support_z(mesh, c_mins, c_maxs, geoms[c])
            floating = sup_z is not None and (g["bottom"] - sup_z) > _RELOCATE_FLOAT_TOL
            cur_tier_z = tiers[_nearest_tier_index(tiers, g["bottom"])][0]
            perched = not _tier_board_supports_footprint(
                mesh, cx, cy, g["half_x"], g["half_y"], cur_tier_z
            )
            poking = _board_crosses_body(
                mesh, cx, cy, g["half_x"], g["half_y"], g["bottom"], g["top"]
            )
            walled = _footprint_internal_surface(
                mesh,
                cx,
                cy,
                [g["bottom"] + 0.02, (g["bottom"] + g["top"]) / 2.0, g["top"] - 0.02],
                g["half_x"],
                g["half_y"],
            )
            if sibling_clash or floating or perched or poking or walled:
                bad.append(c)
        if not bad:
            continue

        good = [c for c in children if c not in bad]
        floor_z = float(c_mins[2])
        top_z = float(c_maxs[2])
        # 占位以 (obj, transform, z底, z顶) 记录;冲突判定 **同时**要求 XY OBB 相交且 z 区间
        # 相交——货架同一 XY 上不同层的件互不冲突,故必须带 z,否则会把上下层误判为重叠。
        occupants: list[tuple[Mapping[str, Any], RepairedTransform, float, float]] = [
            (obj_by_id[c], out[c], geo[c]["bottom"], geo[c]["top"]) for c in good
        ]

        def _conflict(
            obj: Mapping[str, Any],
            cx: float,
            cy: float,
            rotation: Sequence[float],
            z_bot: float,
            z_top: float,
        ) -> bool:
            proposed = RepairedTransform(
                center=[cx, cy, (z_bot + z_top) / 2.0], rotation=list(rotation)
            )
            for occ_obj, occ_t, ob, ot in occupants:
                if z_top <= ob + 1e-3 or z_bot >= ot - 1e-3:
                    continue
                if pre._xy_push_vector(occ_obj, occ_t, obj, proposed) is not None:
                    return True
            return False

        def _xy_candidates(
            half_x: float, half_y: float, prefer: tuple[float, float]
        ) -> list[tuple[float, float]]:
            lo_x, hi_x = interior[0] + half_x, interior[1] - half_x
            lo_y, hi_y = interior[2] + half_y, interior[3] - half_y
            if lo_x > hi_x or lo_y > hi_y:
                return []
            cands: list[tuple[float, float, float]] = []
            for ix in range(_SLOT_GRID_NX):
                fx = 0.5 if _SLOT_GRID_NX == 1 else ix / (_SLOT_GRID_NX - 1)
                cx = lo_x + (hi_x - lo_x) * fx
                for iy in range(_SLOT_GRID_NY):
                    fy = 0.5 if _SLOT_GRID_NY == 1 else iy / (_SLOT_GRID_NY - 1)
                    cy = lo_y + (hi_y - lo_y) * fy
                    cands.append((math.hypot(cx - prefer[0], cy - prefer[1]), cx, cy))
            cands.sort(key=lambda t: t[0])
            return [(cx, cy) for _d, cx, cy in cands]

        def _local_levels(
            cx: float, cy: float, half_x: float, half_y: float
        ) -> list[float]:
            """该 footprint 列上的候选支撑面:地面(架子无底板时件落地)+ 各局部板顶。

            **逐列**枚举(而非全局聚类):同一架子不同 X 列的隔板高度可能完全不同(本架右列
            是 0.55/0.73 大格,左/中列被细分成 0.24 薄格),全局聚类会把大格按别列的板切碎、
            还漏掉地面层,导致高件找不到真实存在的大格子。
            """
            zs: set[float] = {round(floor_z, 3)}
            for sx, sy in _footprint_samples(cx, cy, half_x, half_y):
                for z in _ray_support_z(mesh, sx, sy, top_z + 0.05):
                    zs.add(round(z, 3))
            levels = sorted(zs)
            clustered: list[float] = []
            for z in levels:
                if not clustered or z - clustered[-1] > _TIER_CLUSTER_EPS:
                    clustered.append(z)
            return clustered

        def _slot_clean(
            c: int, cx: float, cy: float, st: float
        ) -> bool:
            """(cx,cy,支撑面 st) 对件 c 是否是干净落点:有支撑、不扎板、不撞内壁、不与
            已有占位件(同 XY 且 z 相交)重叠。"""
            g = geo[c]
            if abs(st - floor_z) >= 1e-6 and not _tier_board_supports_footprint(
                mesh, cx, cy, g["half_x"], g["half_y"], st
            ):
                return False
            z_top = st + g["height"]
            # 真实净高:从支撑面上方向上射线,取 footprint 上最近的「天花板」(上方隔板底面,
            # 任意朝向面均算),净高 = 天花板 - st。只看「上向板顶」的 _board_crosses_body 会
            # 漏检顶面恰在件顶之上、但底面已扎进件身的隔板(典型矮层),故这里用更稳健的净高。
            ceiling = _ceiling_z(mesh, cx, cy, g["half_x"], g["half_y"], st + 0.01)
            if ceiling - st < g["height"] - _TIER_MIN_CLEARANCE:
                return False
            if _footprint_internal_surface(
                mesh,
                cx,
                cy,
                [st + 0.02, st + max(0.02, g["height"] * 0.5), st + max(0.02, g["height"] - 0.02)],
                g["half_x"],
                g["half_y"],
            ):
                return False
            if _conflict(obj_by_id[c], cx, cy, out[c].rotation, st, z_top):
                return False
            return True

        # 体积大的先放(占大格),小件后放(更易塞缝)。
        bad.sort(key=lambda c: geo[c]["half_x"] * geo[c]["half_y"], reverse=True)
        for c in bad:
            g = geo[c]
            cx0, cy0 = out[c].center[0], out[c].center[1]
            # 原位若本来就干净(常因全局层检测误判 perched),不动它,避免无谓位移/新穿插。
            if _slot_clean(c, cx0, cy0, g["bottom"]):
                occupants.append((obj_by_id[c], out[c], g["bottom"], g["top"]))
                continue
            # 在 (XY 网格 × 该列局部支撑面) 上找**最近的干净落点**;架顶(最高板顶、上方无物)
            # 天然是一个干净候选。找不到干净落点就保留原位——绝不把件挪到另一个扎板/悬空位。
            best: tuple[float, float, float, float] | None = None  # (dist, cx, cy, st)
            for cx, cy in _xy_candidates(g["half_x"], g["half_y"], (cx0, cy0)):
                xy_dist = math.hypot(cx - cx0, cy - cy0)
                if best is not None and xy_dist > best[0] + 1e-9:
                    break  # 候选已按 XY 距离排序,更远的不可能更近
                for st in _local_levels(cx, cy, g["half_x"], g["half_y"]):
                    if not _slot_clean(c, cx, cy, st):
                        continue
                    dist = xy_dist + abs(st - g["bottom"])
                    if best is None or dist < best[0]:
                        best = (dist, cx, cy, st)
            if best is None:
                continue
            _, bx, by_, bst = best
            new_center = pre._round_triplet([bx, by_, bst + g["z_offset"]])
            moved = RepairedTransform(center=new_center, rotation=list(out[c].rotation))
            out[c] = moved
            out_orig[c] = moved
            geoms[c].set_center(new_center)
            occupants.append((obj_by_id[c], moved, bst, bst + g["height"]))

    return out, out_orig
