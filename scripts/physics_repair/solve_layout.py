"""python 侧几何后处理求解器入口(不依赖 Blender)。

读入一份 layout / bake 后的 JSON(含 `final_scene.objects[].center/rotation/size/jid`
及 `expected_relations`),在 python 环境用 VHACD 凸块 + FCL 穿透 + XPBD 位置投影去穿插
并满足语义关系,只写回 `center/rotation`,其余字段原样保留。

用法:
    python scripts/physics_repair/solve_layout.py \
        --layout output/.../<scene>_baked.json \
        --output-layout output/.../<scene>_repaired.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR.parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_THIS_DIR), str(_SCRIPTS_DIR), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import physics_repair_engine as pre
import xpbd_solver as xs


def _load_layout(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"layout 文件必须是 JSON object: {path}")
    return data


def _scene_data(layout_data: dict[str, Any]) -> dict[str, Any]:
    if "final_scene" in layout_data:
        scene = layout_data["final_scene"]
    elif "objects" in layout_data:
        scene = layout_data
    else:
        raise ValueError("无法识别 layout 格式,缺少 final_scene 或 objects")
    if not isinstance(scene, dict):
        raise ValueError("场景数据必须是 object")
    if "objects" not in scene or not isinstance(scene["objects"], list):
        raise ValueError("场景缺少 objects 列表")
    return scene


def solve_layout(
    layout_path: str,
    output_layout: str,
    *,
    max_xy_displacement: float = 0.8,
    max_z_adjustment: float = 0.5,
    no_boundary: bool = False,
    boundary_max_xy: float = float("inf"),
    iters: int = 80,
    stiffness: float = 1.0,
    snap_z: bool = True,
    redistribute: bool = True,
    clamp_interior: bool = True,
) -> Path:
    validated_output = pre.validate_repaired_layout_output_path(
        layout_path=layout_path,
        output_path=output_layout,
    )

    layout_data = _load_layout(Path(layout_path))
    scene = _scene_data(layout_data)
    objects = scene["objects"]
    boundary = (
        None
        if no_boundary or "boundary" not in scene or not scene["boundary"]
        else scene["boundary"]
    )

    params = xs.SolveParams(
        max_iters=iters,
        stiffness=stiffness,
        max_xy_displacement=max_xy_displacement,
        max_z_adjustment=max_z_adjustment,
        boundary_xy_cap=(
            float("inf") if no_boundary or math.isinf(float(boundary_max_xy)) else float(boundary_max_xy)
        ),
        snap_z=snap_z,
        redistribute=redistribute,
        clamp_interior=clamp_interior,
    )

    print(
        f"\n运行 XPBD 几何求解 (objects={len(objects)}, iters={iters}, "
        f"max_xy={max_xy_displacement}, max_z={max_z_adjustment}, "
        f"boundary={'off' if boundary is None else 'on'}, snap_z={snap_z})..."
    )
    transforms = xs.solve_scene(objects, boundary, params)

    repaired_path = pre.write_repaired_layout(
        layout_data=layout_data,
        repaired_transforms=transforms,
        output_path=validated_output,
    )
    print(f"几何求解 layout 已保存到: {repaired_path}")
    return repaired_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="python 侧 VHACD+FCL+XPBD 几何后处理求解器(不依赖 Blender)"
    )
    parser.add_argument("--layout", "-l", type=str, required=True, help="输入 layout / baked JSON 路径")
    parser.add_argument(
        "--output-layout",
        "-o",
        type=str,
        required=True,
        help="输出的修正后 layout JSON 路径(不能覆盖输入)",
    )
    parser.add_argument("--max-xy-displacement", type=float, default=0.8, help="相对原位最大 XY 偏移(米)")
    parser.add_argument("--max-z-adjustment", type=float, default=0.5, help="相对原位最大 Z 调整(米)")
    parser.add_argument("--no-boundary", action="store_true", help="关闭 boundary XY 出界拉回")
    parser.add_argument(
        "--boundary-max-xy", type=float, default=float("inf"), help="出界拉回相对原位最大 XY 位移(米)"
    )
    parser.add_argument("--iters", type=int, default=80, help="XPBD 最大迭代轮数")
    parser.add_argument("--stiffness", type=float, default=1.0, help="位置投影刚度(0-1,Jacobi 松弛)")
    parser.add_argument(
        "--no-snap-z",
        action="store_true",
        help=(
            "关闭末轮 Z 贴面(默认开启)。bake 只沉降自由件,on/within 件被锚定不沉降,"
            "其 Z 由这里下射目标网格贴支撑面给定;关闭则 on/within 件保持模型原 Z"
        ),
    )
    parser.add_argument(
        "--no-redistribute",
        action="store_true",
        help="关闭求解前的货架换层重分配(默认开启)",
    )
    parser.add_argument(
        "--no-clamp-interior",
        action="store_true",
        help="关闭末轮 within 内壁夹紧(默认开启;用容器真实网格顶出扎穿内壁的 within 件)",
    )

    args = parser.parse_args()

    solve_layout(
        layout_path=args.layout,
        output_layout=args.output_layout,
        max_xy_displacement=args.max_xy_displacement,
        max_z_adjustment=args.max_z_adjustment,
        no_boundary=args.no_boundary,
        boundary_max_xy=args.boundary_max_xy,
        iters=args.iters,
        stiffness=args.stiffness,
        snap_z=not args.no_snap_z,
        redistribute=not args.no_redistribute,
        clamp_interior=not args.no_clamp_interior,
    )


if __name__ == "__main__":
    main()
