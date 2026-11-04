"""启动 Blender 子进程执行 layout 物理修正的辅助工具。

这是普通 Python 模块（不 import bpy），因此可被 agent / ar_infer 等非 Blender 进程 import，用于在生成 layout 后追加一份物理修正版本（_repaired.json）。
物理修正本身在无界面 Blender 中由 scripts/physics_repair/repair_layout.py 执行。
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
_REPAIR_SCRIPT = _PKG_DIR / "repair_layout.py"
_SOLVE_SCRIPT = _PKG_DIR / "solve_layout.py"

# PATH 之外的常见安装位置（macOS 与 Linux）
_FALLBACK_BLENDER_PATHS = (
    # macOS
    "/Applications/Blender.app/Contents/MacOS/Blender",
    "/Applications/Blender.app/Contents/MacOS/blender",
    # Linux
    "/usr/local/bin/blender",
    "/usr/bin/blender",
    "/snap/bin/blender",
    "/opt/blender/blender",
)


def find_blender(blender_bin: str | None = None) -> str:
    """定位 Blender 可执行文件：显式参数 > BLENDER_BIN 环境变量 > PATH > macOS/Linux 默认路径。"""
    if blender_bin:
        return blender_bin
    env_bin = os.environ.get("BLENDER_BIN")
    if env_bin:
        return env_bin
    path_bin = shutil.which("blender")
    if path_bin:
        return path_bin
    for candidate in _FALLBACK_BLENDER_PATHS:
        if Path(candidate).exists():
            return candidate
    return "blender"


def repaired_layout_path(layout_path: str | Path) -> Path:
    """由原始 layout 路径推导 repaired 路径：<stem>_repaired<suffix>。"""
    path = Path(layout_path)
    return path.with_name(f"{path.stem}_repaired{path.suffix}")


def baked_layout_path(layout_path: str | Path) -> Path:
    """由原始 layout 路径推导 baked 路径：<stem>_baked<suffix>。"""
    path = Path(layout_path)
    return path.with_name(f"{path.stem}_baked{path.suffix}")


def find_solve_python(solve_python: str | None = None) -> str:
    """定位跑 solve_layout 的 python：显式参数 > XPBD_PYTHON 环境变量 > 当前解释器。

    几何后处理依赖 fcl/vhacdx/trimesh(python 环境);默认复用当前解释器,允许用
    XPBD_PYTHON 指向独立的 python 环境。
    """
    if solve_python:
        return solve_python
    return os.environ.get("XPBD_PYTHON") or sys.executable


def run_physics_repair_via_blender(
    layout_path: str | Path,
    output_path: str | Path | None = None,
    *,
    dataset: str = "auto",
    blender_bin: str | None = None,
    solve_python: str | None = None,
    extra_args: list[str] | None = None,
) -> Path:
    """两段式几何修正,产出 baked + repaired layout JSON。

    1) 无界面 Blender 子进程跑 repair_layout.py 做 Bullet bake 沉降 → `<stem>_baked.json`;
    2) 当前/python 解释器跑 solve_layout.py 做几何后处理(VHACD 凸块 2D-SAT + XPBD 去穿插 +
       落地/贴面 + 边界拉回 + 货架换层) → `output_path`(默认 `<stem>_repaired.json`)。

    只 bake 不解会让物体穿地/出界(bake 单独不稳),故必须接 solve;这与 run_ar_infer.sh
    的两段式一致。失败时(Blender/资产缺失、脚本报错等)抛出 CalledProcessError,
    由调用方决定是否致命。
    """
    layout_path = Path(layout_path)
    output = Path(output_path) if output_path is not None else repaired_layout_path(layout_path)
    baked = baked_layout_path(layout_path)

    binary = find_blender(blender_bin)
    bake_command = [
        binary,
        "-b",
        "-P",
        str(_REPAIR_SCRIPT),
        "--",
        "--layout",
        str(layout_path),
        "--output-layout",
        str(baked),
        "--dataset",
        dataset,
    ]
    if extra_args:
        bake_command.extend(extra_args)
    subprocess.run(bake_command, check=True)

    solve_command = [
        find_solve_python(solve_python),
        str(_SOLVE_SCRIPT),
        "--layout",
        str(baked),
        "--output-layout",
        str(output),
    ]
    subprocess.run(solve_command, check=True)
    return output
