r"""平移态 XPBD 求解器:在 VHACD 凸块 XY 穿透之上做位置投影去穿插,
并把语义关系(on/within/under/boundary/支撑刚性跟随/限幅)写成同框架的约束。

设计要点:
  - **唯一碰撞原语**:`collision_geometry.pair_contact` 给出每对最深接触
    \((\hat n, d)\);非穿插约束按逆质量(movability)分摊位置修正
    \(\Delta x_i = \pm \tfrac{w_i}{w_i+w_j} d\,\hat n\),Jacobi 同时更新。
  - **去穿插仅作用于 XY 平面**(沿用旧实现:Z 由 bake 沉降给定,贴面交给末轮 Z-snap),
    竖直接触法线在 XY 上分量≈0,天然不推开堆叠/压叠对。
  - **硬豁免**(under/on/within/椅子塞桌下)跳过非穿插约束,合法重叠不被推散;
    其余「贴合」对真实 \(d\le\varepsilon\),推力自然为 0——无需软豁免动物园。
  - **最优轮收敛**:每轮在去穿插前测量 (重叠对数, 最大穿透),快照最优状态返回,
    打破夹角极限环。

旋转沿用 bake 结果固定(v1 只解平移)。本模块不依赖 bpy,可独立单测。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import collision_geometry as cg
import physics_repair_engine as pre
from physics_repair_engine import RepairedTransform
from utils.convex_decomposition import build_transform_matrix as _build_transform_matrix

_XY_PUSH_EPS = 1e-4


@dataclass
class SolveParams:
    max_iters: int = 80
    stiffness: float = 1.0
    converge_eps: float = 1e-4
    max_xy_displacement: float = 0.8
    max_z_adjustment: float = 0.5
    boundary_xy_cap: float = float("inf")
    # 末轮 Z 贴面:对 on/within 子件下射其目标网格取支撑面、贴齐高度。bake 已把 on/within
    # 件锚定为不沉降(只沉降自由件),故其 Z 必须由这里给定。dataclass 默认 False 仅为纯几何
    # 单测(无真实资产网格,贴面会去加载 mesh);生产入口 solve_layout 默认开启。
    snap_z: bool = False
    # 求解前先做货架换层重分配:把摆不下的货架子件挪到同架空层(离散规划,见
    # shelf_redistribute)。默认开启。
    redistribute: bool = True
    # 末轮 within 内壁夹紧:用容器真实网格(保留凹腔,凸分解会把腔体填实而无法用)
    # 侧射,把扎穿容器内壁/隔板的 within 件顶回腔内。依赖真实资产网格,dataclass 默认
    # False 仅为纯几何单测(无网格);生产入口 solve_layout 默认开启。
    clamp_interior: bool = False


def _objects_by_id(objects: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    return {int(o["id"]): o for o in objects}


def _transform_of(obj: Mapping[str, Any]) -> RepairedTransform:
    rotation = obj["rotation"] if "rotation" in obj else [0.0, 0.0, 0.0]
    return RepairedTransform(
        center=[float(v) for v in obj["center"]],
        rotation=[float(v) for v in rotation],
    )


# --------------------------------------------------------------------------- #
# 非穿插约束(唯一碰撞原语)
# --------------------------------------------------------------------------- #
def _non_penetration_deltas(
    objects_by_id: Mapping[int, Mapping[str, Any]],
    geoms: Mapping[int, cg.ObjectGeom],
    transforms: Mapping[int, RepairedTransform],
) -> tuple[dict[int, np.ndarray], dict[int, int], int, float]:
    """测量当前位姿下所有非豁免对的 XY 穿插,返回 Jacobi 位置修正累加量。

    返回 (delta_xy_by_id, contact_count_by_id, overlap_pairs, max_penetration)。
    """
    ids = sorted(geoms.keys())
    for oid in ids:
        geoms[oid].set_center(transforms[oid].center)

    delta: dict[int, np.ndarray] = {oid: np.zeros(3) for oid in ids}
    count: dict[int, int] = {oid: 0 for oid in ids}
    overlap_pairs = 0
    max_pen = 0.0

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            id_a, id_b = ids[i], ids[j]
            obj_a, obj_b = objects_by_id[id_a], objects_by_id[id_b]
            if pre._is_hard_overlap_exempt(
                obj_a, obj_b, transforms[id_a], transforms[id_b]
            ):
                continue
            contact = cg.pair_xy_penetration(geoms[id_a], geoms[id_b])
            if contact is None:
                continue
            direction, depth = contact
            # 整体出口救正:小件嵌入大体块时,逐块 SAT 的 MTV 严重低估(只给沿细方向的
            # 浅层逃逸,如灯杆插床垫只报 ~2cm),无法把件移出。改用「整体移出大件 AABB」
            # 的更深穿透量,保证嵌入件被真正驱逐。
            escape = cg.pair_xy_escape(geoms[id_a], geoms[id_b])
            if escape is not None and escape[1] > depth:
                direction, depth = escape
            overlap_pairs += 1
            max_pen = max(max_pen, depth)

            mtv_xy = np.array([direction[0] * depth, direction[1] * depth, 0.0])
            if float(np.hypot(mtv_xy[0], mtv_xy[1])) < _XY_PUSH_EPS:
                continue

            w_a = pre._xy_separation_movability(obj_a)
            w_b = pre._xy_separation_movability(obj_b)
            total = w_a + w_b
            frac_a = w_a / total
            frac_b = w_b / total
            delta[id_b] += frac_b * mtv_xy
            delta[id_a] -= frac_a * mtv_xy
            count[id_a] += 1
            count[id_b] += 1

    return delta, count, overlap_pairs, max_pen


# --------------------------------------------------------------------------- #
# 语义约束(同框架,取代豁免)
# --------------------------------------------------------------------------- #
def _support_rigid_follow(
    after: dict[int, RepairedTransform],
    before: Mapping[int, RepairedTransform],
    objects_by_id: Mapping[int, Mapping[str, Any]],
) -> dict[int, RepairedTransform]:
    """支撑(桌/柜)本步发生 XY 平移时,把其上 on 子件按支撑的**本步增量**一起平移。

    `before`/`after` 为本轮去穿插步骤前/后的位姿;支撑本步位移取 after-before,叠加到
    子件**当前(after)位置**上(而非复位到原始相对偏移)。这样既让子件刚性跟随支撑,
    又**保留同一支撑上多个子件之间已去穿插解出的相对布局**——同支撑子件同步平移,相对
    位置不变。读取一律用冻结的 before/after,写入 out,避免遍历顺序影响结果。

    旧实现用「原始相对偏移」复位子件,只要支撑被别的件碰一下,就会把同支撑兄弟件已解出的
    分离抹掉、恢复原始互穿(典型:柜面上的音箱/台灯彼此穿插,柜子一被顶就永远分不开)。
    """
    out = dict(after)
    for cid, obj in objects_by_id.items():
        if cid not in after:
            continue
        sup_id = pre._expected_attachment_target_id(obj)
        if sup_id is None or sup_id not in after or sup_id not in before:
            continue
        sdx = after[sup_id].center[0] - before[sup_id].center[0]
        sdy = after[sup_id].center[1] - before[sup_id].center[1]
        if abs(sdx) < 1e-9 and abs(sdy) < 1e-9:
            continue
        c = after[cid].center
        out[cid] = RepairedTransform(
            center=pre._round_triplet([c[0] + sdx, c[1] + sdy, c[2]]),
            rotation=list(out[cid].rotation),
        )
    return out


def _within_on_xy_clamp(
    transforms: dict[int, RepairedTransform],
    objects_by_id: Mapping[int, Mapping[str, Any]],
    originals: Mapping[int, RepairedTransform],
    params: SolveParams,
) -> dict[int, RepairedTransform]:
    """容纳件夹在容器 footprint 内、桌面件夹在支撑 footprint 内(OBB 内壁夹紧)。"""
    out = dict(transforms)
    for oid, obj in objects_by_id.items():
        if oid not in out or oid not in originals:
            continue
        transform = out[oid]
        if pre._expected_containment_target_id(obj) is not None:
            transform = pre._apply_container_constraint(
                transform=transform,
                obj_data=obj,
                objects_by_id=objects_by_id,
                transforms=out,
                original=originals[oid],
                max_xy_displacement=params.max_xy_displacement,
                max_z_adjustment=params.max_z_adjustment,
            )
        elif pre._expected_on_target_id(obj) is not None:
            transform = pre._apply_support_constraint(
                transform=transform,
                obj_data=obj,
                objects_by_id=objects_by_id,
                transforms=out,
                original=originals[oid],
                max_xy_displacement=params.max_xy_displacement,
                max_z_adjustment=params.max_z_adjustment,
            )
        out[oid] = transform
    return out


def _inside_boundary(
    transforms: dict[int, RepairedTransform],
    geoms: Mapping[int, cg.ObjectGeom],
    boundary: Sequence[Sequence[float]] | None,
) -> dict[int, RepairedTransform]:
    if boundary is None or len(boundary) < 3:
        return dict(transforms)
    out = dict(transforms)
    for oid, transform in transforms.items():
        mins, maxs = geoms[oid].aabb_at(transform.center)
        dx, dy = pre.compute_xy_shift_for_axis_aligned_aabb_inside_polygon(
            [float(mins[0]), float(mins[1])],
            [float(maxs[0]), float(maxs[1])],
            boundary,
        )
        if dx == 0.0 and dy == 0.0:
            continue
        out[oid] = RepairedTransform(
            center=pre._round_triplet(
                [transform.center[0] + dx, transform.center[1] + dy, transform.center[2]]
            ),
            rotation=list(transform.rotation),
        )
    return out


def _clamp_to_originals(
    transforms: dict[int, RepairedTransform],
    originals: Mapping[int, RepairedTransform],
    params: SolveParams,
) -> dict[int, RepairedTransform]:
    return pre._clamp_transforms_to_originals(
        transforms=transforms,
        original_transforms=originals,
        max_xy_displacement=params.max_xy_displacement,
        max_z_adjustment=params.max_z_adjustment,
    )


# --------------------------------------------------------------------------- #
# 主求解循环
# --------------------------------------------------------------------------- #
def _max_center_move(
    a: Mapping[int, RepairedTransform], b: Mapping[int, RepairedTransform]
) -> float:
    move = 0.0
    for oid in a:
        ca, cb = a[oid].center, b[oid].center
        move = max(move, abs(ca[0] - cb[0]), abs(ca[1] - cb[1]), abs(ca[2] - cb[2]))
    return move


def _project_semantic(
    transforms: dict[int, RepairedTransform],
    objects_by_id: Mapping[int, Mapping[str, Any]],
    geoms: Mapping[int, cg.ObjectGeom],
    original_transforms: Mapping[int, RepairedTransform],
    boundary: Sequence[Sequence[float]] | None,
    params: SolveParams,
) -> dict[int, RepairedTransform]:
    """语义投影(优先级顺序):容器/支撑内壁夹紧 → 边界内推 → 限幅。

    支撑刚性跟随不在此处:它需要「本步去穿插前/后」的位姿做增量平移,放在主循环里
    紧接去穿插步骤之后调用。
    """
    out = _within_on_xy_clamp(transforms, objects_by_id, original_transforms, params)
    out = _inside_boundary(out, geoms, boundary)
    out = _clamp_to_originals(out, original_transforms, params)
    return out


def _depenetrate_xy(
    transforms: dict[int, RepairedTransform],
    objects_by_id: Mapping[int, Mapping[str, Any]],
    geoms: Mapping[int, cg.ObjectGeom],
    original_transforms: Mapping[int, RepairedTransform],
    boundary: Sequence[Sequence[float]] | None,
    params: SolveParams,
    max_iters: int,
) -> dict[int, RepairedTransform]:
    """XY 去穿插迭代(Jacobi 位置投影 + 支撑跟随 + 语义投影),返回最优轮快照。

    只动 XY、不做 Z 贴面;主求解与「贴地/贴面后补分离」(Fix:贴地可能让件进入他件
    z 区间产生新水平重叠)共用。
    """
    best_transforms = {oid: t for oid, t in transforms.items()}
    best_key: tuple[int, float] | None = None

    for _ in range(max_iters):
        delta, count, overlap_pairs, max_pen = _non_penetration_deltas(
            objects_by_id, geoms, transforms
        )

        key = (overlap_pairs, max_pen)
        if best_key is None or key < best_key:
            best_key = key
            best_transforms = {oid: t for oid, t in transforms.items()}
        if overlap_pairs == 0:
            break

        # 1) 非穿插位置投影(Jacobi 平均后按刚度步进)
        pushed: dict[int, RepairedTransform] = {}
        for oid, t in transforms.items():
            if count[oid] == 0:
                pushed[oid] = t
                continue
            step = params.stiffness * delta[oid] / count[oid]
            pushed[oid] = RepairedTransform(
                center=[t.center[0] + step[0], t.center[1] + step[1], t.center[2]],
                rotation=list(t.rotation),
            )

        # 2) 支撑刚性跟随(增量):支撑本步被推动时,其上 on 子件按同一增量跟随,
        #    保留同支撑子件间已去穿插的相对布局(transforms=步前,pushed=步后)。
        pushed = _support_rigid_follow(pushed, transforms, objects_by_id)

        # 3) 语义投影
        pushed = _project_semantic(
            pushed, objects_by_id, geoms, original_transforms, boundary, params
        )

        if _max_center_move(transforms, pushed) < params.converge_eps:
            transforms = pushed
            # 收敛后再测一次,确保最优轮覆盖末态
            _, _, final_pairs, final_pen = _non_penetration_deltas(
                objects_by_id, geoms, transforms
            )
            final_key = (final_pairs, final_pen)
            if final_key < best_key:
                best_transforms = {oid: t for oid, t in transforms.items()}
            break
        transforms = pushed

    return best_transforms


def solve(
    objects: Sequence[Mapping[str, Any]],
    geoms: Mapping[int, cg.ObjectGeom],
    original_transforms: Mapping[int, RepairedTransform],
    boundary: Sequence[Sequence[float]] | None,
    params: SolveParams,
) -> dict[int, RepairedTransform]:
    """运行 XPBD 求解,返回每件最优轮的 RepairedTransform(id→transform)。"""
    objects_by_id = _objects_by_id(objects)
    transforms: dict[int, RepairedTransform] = {
        oid: RepairedTransform(
            center=list(original_transforms[oid].center),
            rotation=list(original_transforms[oid].rotation),
        )
        for oid in geoms
    }
    # Z 贴面必须在 XY 去穿插之前:先把自由件落地、on/within 件贴支撑面,确定最终高度;
    # 这样后续 XY 去穿插在正确 z-range 上判定,落地/贴面引起的新水平重叠才会被解。
    if params.snap_z:
        transforms = ground_free_z(
            transforms, objects_by_id, params.max_z_adjustment
        )
        transforms = snap_children_z(
            transforms, objects_by_id, params.max_z_adjustment
        )
    # 再把初始态投影到可行域(边界/容器/限幅),保证后续最优轮快照都是可行状态。
    transforms = _project_semantic(
        transforms, objects_by_id, geoms, original_transforms, boundary, params
    )

    best_transforms = _depenetrate_xy(
        transforms, objects_by_id, geoms, original_transforms, boundary, params, params.max_iters
    )

    # 末轮 Z 贴面:循环里 XY 去穿插会把 on/within 子件水平挪动(架子件常被挪 0.3m+),
    # 使循环前贴的 Z 失效而悬空。这里按**最终 XY** 再贴一次 Z(只改 Z 不动 XY,不会引入
    # 新的水平穿插),保证每件落在其当前位置正下方的真实支撑面/地面上。
    if params.snap_z:
        best_transforms = ground_free_z(
            best_transforms, objects_by_id, params.max_z_adjustment
        )
        best_transforms = snap_children_z(
            best_transforms, objects_by_id, params.max_z_adjustment
        )

    # within 内壁夹紧:用容器真实网格顶出扎穿内壁/隔板的 within 件(凸分解填腔无法处理)。
    # 放在 Z 贴面之后(需正确高度做侧射),只动 XY;之后再补一次 Z 贴面跟随其水平微调。
    if params.clamp_interior:
        best_transforms = clamp_within_interior(
            best_transforms, objects_by_id, original_transforms, params.max_xy_displacement
        )
        if params.snap_z:
            best_transforms = snap_children_z(
                best_transforms, objects_by_id, params.max_z_adjustment
            )

    # 贴地/贴面后补一轮 XY 去穿插:ground_free_z/snap_children_z 只改 Z,会把件落进他件的
    # z 区间从而产生**主循环已结束后**才出现的水平重叠(典型:落地灯贴地后灯杆扎进床垫)。
    # 这里按最终高度再去一次穿插,然后重贴 Z(贴 Z 只改高度、幂等,不会再造新水平重叠)。
    if params.snap_z or params.clamp_interior:
        post = _depenetrate_xy(
            best_transforms,
            objects_by_id,
            geoms,
            original_transforms,
            boundary,
            params,
            min(params.max_iters, 40),
        )
        if _max_center_move(best_transforms, post) > 1e-6:
            best_transforms = post
            if params.snap_z:
                best_transforms = ground_free_z(
                    best_transforms, objects_by_id, params.max_z_adjustment
                )
                best_transforms = snap_children_z(
                    best_transforms, objects_by_id, params.max_z_adjustment
                )

    # 货架换层重分配放在**最后**:用求解后(去穿插/贴面/夹紧已定)的**最终**货架位姿,把
    # 「悬空/扎板/骑边/撞兄弟」的货架子件搬进同架真实存在的干净隔间(逐列局部净高判定)。
    # 必须是末步——它精确摆放的子件位置不能再被去穿插循环/容器夹紧/跟随等步骤搅动(那会把
    # 件从干净隔间挤进有隔板的列重新扎板)。它直接给出子件的最终落点(含贴面 z),无需再贴。
    if params.redistribute:
        import shelf_redistribute

        best_transforms, _ = shelf_redistribute.redistribute_container_children(
            objects, geoms, best_transforms, best_transforms
        )

    return best_transforms


# --------------------------------------------------------------------------- #
# 末轮 Z 贴面(可选):自由件落地 + on/within 子件下射目标支撑面
# --------------------------------------------------------------------------- #
def ground_free_z(
    transforms: dict[int, RepairedTransform],
    objects_by_id: Mapping[int, Mapping[str, Any]],
    max_z_adjustment: float,
    floor_z: float = 0.0,
) -> dict[int, RepairedTransform]:
    """无 on/within 目标的自由件(家具/地毯等)落到地面:底部贴 floor_z(限幅)。

    Bullet bake 对"家具压薄地毯"会把地毯竖直弹起、对贴地件也会因碰撞 margin 留约 4cm
    微浮;这里对自由件做确定性落地,消除这类悬空(on/within 件由 snap_children_z 贴支撑面)。
    """
    out = dict(transforms)
    for oid, obj in objects_by_id.items():
        if oid not in out:
            continue
        if pre._expected_on_target_ids(obj) or pre._expected_containment_target_ids(obj):
            continue
        size = obj.get("size")
        if not (
            isinstance(size, Sequence)
            and not isinstance(size, (str, bytes))
            and len(size) == 3
        ):
            continue
        t = out[oid]
        half_z = float(size[2]) / 2.0
        desired = [t.center[0], t.center[1], float(floor_z) + half_z]
        new_center = pre.clamp_center(
            original_center=t.center,
            simulated_center=desired,
            max_xy_displacement=float("inf"),
            max_z_adjustment=max_z_adjustment,
        )
        out[oid] = RepairedTransform(center=new_center, rotation=list(t.rotation))
    return out


def snap_children_z(
    transforms: dict[int, RepairedTransform],
    objects_by_id: Mapping[int, Mapping[str, Any]],
    max_z_adjustment: float,
) -> dict[int, RepairedTransform]:
    """对有显式 on/within 目标的子件,用 trimesh 下射目标世界网格取支撑高度,
    `snap_center_z_to_support` 贴面(限幅)。无命中则保留 bake 高度。"""
    out = dict(transforms)
    world_cache: dict[int, Any] = {}

    def world_mesh(oid: int):
        if oid not in world_cache:
            world_cache[oid] = cg.build_world_trimesh(objects_by_id[oid], out[oid])
        return world_cache[oid]

    for cid, obj in objects_by_id.items():
        if cid not in out:
            continue
        target_ids = pre._expected_on_target_ids(obj) + pre._expected_containment_target_ids(obj)
        target_ids = [t for t in target_ids if t in objects_by_id and t in out]
        if not target_ids:
            continue

        child_t = out[cid]
        child_mesh = world_mesh(cid)
        child_min_z = float(child_mesh.bounds[0][2])
        size = obj.get("size")
        if (
            isinstance(size, Sequence)
            and not isinstance(size, (str, bytes))
            and len(size) == 3
        ):
            half_x, half_y = float(size[0]) / 2.0, float(size[1]) / 2.0
        else:
            half_x = half_y = 0.0
        # 在子件 footprint 上多点(中心+四角内缩)下射:取整个底面下方的**最高**支撑面,
        # 等价于物体落在其底面接触到的最高点。只用中心单射线会被支撑面上的局部低点(桌面
        # 凹陷/缝隙/边缘)误导,把件压沉到比真实落点更低处(如花瓶陷进桌面)。
        ax, ay = _local_xy_axes(child_t.rotation)
        cx, cy = child_t.center[0], child_t.center[1]
        inset = 0.7
        sample_pts = [(cx, cy)]
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                p = (
                    cx + sx * half_x * inset * ax[0] + sy * half_y * inset * ay[0],
                    cy + sx * half_x * inset * ax[1] + sy * half_y * inset * ay[1],
                )
                sample_pts.append(p)
        origins = np.array([[px, py, child_t.center[2] + 5.0] for px, py in sample_pts])
        directions = np.tile(np.array([0.0, 0.0, -1.0]), (len(sample_pts), 1))

        # 只认子件中心以下的支撑面:on/within 件应落到其下方最近的支撑(桌面/货架隔板),
        # 取全局最高命中会把件顶到容器/支撑的最高面(如货架顶板)上。
        support_cap = child_t.center[2]
        support_z: float | None = None
        for tid in target_ids:
            locations, _, _ = world_mesh(tid).ray.intersects_location(origins, directions)
            if len(locations) == 0:
                continue
            below = locations[:, 2][locations[:, 2] <= support_cap]
            if len(below) == 0:
                continue
            top = float(np.max(below))
            support_z = top if support_z is None else max(support_z, top)

        if support_z is None:
            continue
        new_center = pre.snap_center_z_to_support(
            original_center=child_t.center,
            current_center=child_t.center,
            current_bottom_z=child_min_z,
            support_z=support_z,
            max_z_adjustment=max_z_adjustment,
        )
        out[cid] = RepairedTransform(center=new_center, rotation=list(child_t.rotation))
    return out


# --------------------------------------------------------------------------- #
# within 内壁夹紧(真实网格,处理凸分解填腔无法处理的容器内穿插)
# --------------------------------------------------------------------------- #
def _local_xy_axes(rotation: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """物体局部 +x/+y 轴的世界方向(单位向量)。"""
    m = _build_transform_matrix(center=[0.0, 0.0, 0.0], rotation=list(rotation), scale=np.ones(3))
    rot = np.asarray(m[:3, :3], dtype=np.float64)
    return rot[:, 0], rot[:, 1]


def _nearest_ray_hit(mesh: Any, origin: np.ndarray, direction: np.ndarray) -> float | None:
    """从 origin 沿 direction 命中 mesh 的最近正向距离;无命中返回 None。"""
    loc, _, _ = mesh.ray.intersects_location(
        np.array([origin], dtype=np.float64), np.array([direction], dtype=np.float64)
    )
    if len(loc) == 0:
        return None
    dist = (loc - origin) @ np.asarray(direction, dtype=np.float64)
    dist = dist[dist > 1e-4]
    return float(dist.min()) if len(dist) else None


def clamp_within_interior(
    transforms: dict[int, RepairedTransform],
    objects_by_id: Mapping[int, Mapping[str, Any]],
    originals: Mapping[int, RepairedTransform],
    max_xy_displacement: float,
) -> dict[int, RepairedTransform]:
    """把扎穿容器内壁/隔板的 `within` 件,沿其局部 XY 轴顶回腔内(限幅)。

    VHACD 凸分解会把容器的开放腔体近似成实心凸块,故凸块 2D-SAT **无法**区分「件正确
    在腔内」与「件扎穿内壁」(见 README)。这里改用容器**真实网格**(保留凹腔):从件
    中心沿其局部 ±x/±y 各射一条线取最近内壁距离 `d±`,件该轴半宽为 `h`,则需
    `shift∈[h-d_neg, d_pos-h]` 才两侧都不扎墙;取该可行区间内最接近 0 的最小位移;若格子
    比件还窄(区间空)则居中均摊。只调 XY,不动 Z(故不会引入悬空)。
    """
    out = dict(transforms)
    mesh_cache: dict[tuple[int, tuple], Any] = {}

    def container_mesh(tid: int) -> Any:
        key = (tid, tuple(round(v, 4) for v in out[tid].center), tuple(out[tid].rotation))
        if key not in mesh_cache:
            mesh_cache[key] = cg.build_world_trimesh(objects_by_id[tid], out[tid])
        return mesh_cache[key]

    for cid, obj in objects_by_id.items():
        if cid not in out:
            continue
        tid = pre._expected_containment_target_id(obj)
        if tid is None or tid not in objects_by_id or tid not in out:
            continue
        size = obj.get("size")
        if not (
            isinstance(size, Sequence)
            and not isinstance(size, (str, bytes))
            and len(size) == 3
        ):
            continue

        mesh = container_mesh(tid)
        center = np.asarray(out[cid].center, dtype=np.float64)
        axis_x, axis_y = _local_xy_axes(out[cid].rotation)
        shift = np.zeros(3)
        for axis, half in ((axis_x, float(size[0]) / 2.0), (axis_y, float(size[1]) / 2.0)):
            d_pos = _nearest_ray_hit(mesh, center, axis)
            d_neg = _nearest_ray_hit(mesh, center, -axis)
            if d_pos is None or d_neg is None:
                continue
            lo, hi = half - d_neg, d_pos - half
            if lo <= hi:
                s = min(max(0.0, lo), hi)
            else:
                s = (d_pos - d_neg) / 2.0
            shift = shift + s * axis

        if float(np.hypot(shift[0], shift[1])) < 1e-4:
            continue
        new_center = pre.clamp_center(
            original_center=originals[cid].center if cid in originals else list(center),
            simulated_center=[center[0] + shift[0], center[1] + shift[1], center[2]],
            max_xy_displacement=max_xy_displacement,
            max_z_adjustment=0.0,
        )
        out[cid] = RepairedTransform(center=new_center, rotation=list(out[cid].rotation))
    return out


# --------------------------------------------------------------------------- #
# 便捷入口:从 objects 自建几何与原位
# --------------------------------------------------------------------------- #
def solve_scene(
    objects: Sequence[Mapping[str, Any]],
    boundary: Sequence[Sequence[float]] | None,
    params: SolveParams,
) -> dict[int, RepairedTransform]:
    geoms = cg.build_scene_geoms(objects)
    originals = {int(o["id"]): _transform_of(o) for o in objects}
    return solve(objects, geoms, originals, boundary, params)
