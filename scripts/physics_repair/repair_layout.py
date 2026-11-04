"""
仅做物理修正、不渲染的 Blender 入口。

把任意 pipeline（agent / ar_infer 等）产出的统一格式 layout JSON 喂进来，
在 Blender 内导入真实资产、跑 rigid body settle（碰撞代理固定为真实网格 MESH），
然后把修正后的位姿写回一份新的 layout JSON。整个过程不设置灯光/相机、不渲染、
不保存 .blend，因此比走完整可视化流程更快、更聚焦。

与 visualize_layout.py 的关系：
- visualize_layout.py 是“可视化”入口，可选地顺带做物理修正；
- repair_layout.py 是“只修物理”入口，复用前者抽出的 build_scene_from_layout，
  调用 physics_repair_engine.run_physics_repair，得到修正后的 layout。

用法（无界面后台运行）：
    blender -b -P scripts/physics_repair/repair_layout.py -- \
        --layout output/agent/<scene>_<ts>_layout.json \
        --output-layout output/agent/<scene>_<ts>_layout_repaired.json \
        --dataset auto

本入口只做 Bullet bake 沉降,产出 baked layout;几何后处理(去穿插/落地/容器/边界/
货架换层)交给 python 侧 solve_layout.py(VHACD 凸块 2D-SAT + XPBD)。

可选参数：
    --frames                物理模拟帧数（默认 40）
    --substeps              每帧子步数（默认 5）
    --solver-iters          Bullet 每帧约束求解迭代次数（默认 10）
    --max-xy-displacement   相对原位最大 XY 偏移（米，默认 0.8）
    --max-z-adjustment      相对原位最大 Z 调整（米，默认 0.5）
    --no-boundary           关闭根据 layout boundary 的 XY 出界拉回
    --boundary-max-xy       出界拉回相对原位的最大 XY 位移（米，默认不限制）
    --collision-decimate-faces  碰撞网格减面目标面数（默认 2000，保留凹形；<=0 关闭）
    --collision             bake 碰撞代理：vhacd（默认，需安装 vhacdx）或 mesh
"""

import argparse
import math
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
# 自身目录(物理修复子包)+ 父 scripts 目录(visualize_layout 等通用脚本)都需在 path 上
for _path in (script_dir, script_dir.parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from physics_repair_engine import (  # type: ignore
    run_physics_repair,
    validate_repaired_layout_output_path,
    write_repaired_layout,
)
from visualize_layout import build_scene_from_layout  # type: ignore


def repair_layout(
    layout_path: str,
    output_layout: str,
    *,
    dataset: str = "auto",
    frames: int = 40,
    max_xy_displacement: float = 0.8,
    max_z_adjustment: float = 0.5,
    no_boundary: bool = False,
    boundary_max_xy: float = float("inf"),
    collision_decimate_faces: int = 2000,
    substeps_per_frame: int = 5,
    solver_iterations: int = 10,
    collision_mode: str = "vhacd",
    vhacd_max_hulls: int = 16,
    vhacd_resolution: int = 100000,
) -> Path:
    """构建场景 → Bullet bake 沉降 → 写回 baked layout JSON,不渲染。

    几何后处理(去穿插/落地/容器/边界/货架换层)在 python 侧 solve_layout.py 完成。
    """
    validated_output = validate_repaired_layout_output_path(
        layout_path=layout_path,
        output_path=output_layout,
    )

    build = build_scene_from_layout(
        layout_path,
        dataset=dataset,
    )

    decimate_desc = (
        "off" if collision_decimate_faces <= 0 else f"{collision_decimate_faces}faces"
    )
    print(
        "\n运行 Blender rigid body bake 沉降 "
        f"(frames={frames}, max_xy={max_xy_displacement}, "
        f"max_z={max_z_adjustment}, collision={collision_mode}, decimate={decimate_desc})..."
    )
    repaired_transforms = run_physics_repair(
        floor_obj=build.floor,
        asset_objects=build.asset_objects,
        scene_objects=build.objects,
        frame_count=frames,
        max_xy_displacement=max_xy_displacement,
        max_z_adjustment=max_z_adjustment,
        preserve_rotation=True,
        boundary=(
            None
            if no_boundary
            else (build.boundary if isinstance(build.boundary, list) and len(build.boundary) > 0 else None)
        ),
        boundary_max_xy_displacement=(
            None
            if no_boundary or math.isinf(float(boundary_max_xy))
            else float(boundary_max_xy)
        ),
        collision_decimate_target_faces=(
            None if collision_decimate_faces <= 0 else int(collision_decimate_faces)
        ),
        substeps_per_frame=substeps_per_frame,
        solver_iterations=solver_iterations,
        collision_mode=collision_mode,
        vhacd_max_hulls=vhacd_max_hulls,
        vhacd_resolution=vhacd_resolution,
    )

    repaired_layout_path = write_repaired_layout(
        layout_data=build.layout_data,
        repaired_transforms=repaired_transforms,
        output_path=validated_output,
    )
    print(f"bake layout 已保存到: {repaired_layout_path}")
    return repaired_layout_path


def main():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(
        description="仅做物理修正、不渲染的 Blender 入口（碰撞代理固定为真实网格 MESH）"
    )
    parser.add_argument("--layout", "-l", type=str, required=True, help="输入 layout JSON 路径")
    parser.add_argument(
        "--output-layout",
        "-o",
        type=str,
        required=True,
        help="输出的物理修正后 layout JSON 路径（不能覆盖输入）",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="auto",
        choices=["auto", "3dfront", "imaginarium"],
        help="资产数据集类型，默认 auto",
    )
    parser.add_argument("--frames", type=int, default=40, help="rigid body 物理模拟帧数")
    parser.add_argument(
        "--substeps",
        type=int,
        default=5,
        help="rigid body 每帧子步数(越大越稳越慢,默认 5)",
    )
    parser.add_argument(
        "--solver-iters",
        type=int,
        default=10,
        help="rigid body 约束求解迭代次数(越大越稳越慢,默认 10)",
    )
    parser.add_argument(
        "--collision",
        type=str,
        default="vhacd",
        choices=["mesh", "vhacd"],
        help=(
            "bake 碰撞代理:vhacd(V-HACD 凸分解 + COMPOUND 凸-凸,bake 远快于 GImpact,"
            "需安装 vhacdx) 或 mesh(减面真网格 + GImpact);默认 vhacd"
        ),
    )
    parser.add_argument(
        "--vhacd-max-hulls",
        type=int,
        default=16,
        help="vhacd 模式每个物体最大凸包数(默认 16)",
    )
    parser.add_argument(
        "--vhacd-resolution",
        type=int,
        default=100000,
        help="vhacd 体素分辨率(默认 100000)",
    )
    parser.add_argument(
        "--max-xy-displacement",
        type=float,
        default=0.8,
        help="物理微调后允许的最大 XY 平面偏移（米）",
    )
    parser.add_argument(
        "--max-z-adjustment",
        type=float,
        default=0.5,
        help="物理微调后允许的最大 Z 轴调整（米）",
    )
    parser.add_argument(
        "--no-boundary",
        action="store_true",
        help="关闭根据 layout boundary 的 XY 出界拉回",
    )
    parser.add_argument(
        "--boundary-max-xy",
        type=float,
        default=float("inf"),
        help="出界拉回相对推理原位允许的最大 XY 位移（米），默认不限制",
    )
    parser.add_argument(
        "--collision-decimate-faces",
        type=int,
        default=2000,
        help=(
            "碰撞网格减面目标面数（仍用 MESH/COLLAPSE，保留凹形以加速 Bullet 求交、"
            "避免高模 GImpact 递归爆栈段错误）；默认 2000，<=0 表示不减面"
        ),
    )
    args = parser.parse_args(argv)

    repair_layout(
        layout_path=args.layout,
        output_layout=args.output_layout,
        dataset=args.dataset,
        frames=args.frames,
        max_xy_displacement=args.max_xy_displacement,
        max_z_adjustment=args.max_z_adjustment,
        no_boundary=args.no_boundary,
        boundary_max_xy=args.boundary_max_xy,
        collision_decimate_faces=args.collision_decimate_faces,
        substeps_per_frame=args.substeps,
        solver_iterations=args.solver_iters,
        collision_mode=args.collision,
        vhacd_max_hulls=args.vhacd_max_hulls,
        vhacd_resolution=args.vhacd_resolution,
    )


if __name__ == "__main__":
    main()
