"""
场景旋转增强脚本

将场景绕 Z 轴旋转 90°, 180°, 270°，生成增强数据。
旋转后保证坐标都是正的，边界最小点为 (0, 0)。

行为：
- 原始 0 度场景保持不变
- 直接在 converted 目录下添加 _rot90, _rot180, _rot270 后缀的旋转场景
- 若旋转文件已存在则直接覆盖

适用于：Imaginarium 等数据量较少的数据集，与 3D-FRONT 混合训练时平衡数据量。
"""

import json
import math
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple
from copy import deepcopy

from tqdm import tqdm

# 添加项目根目录到 Python 路径

from config.paths import ProjectPaths
from dataset_pipeline.utils import translate_scene_to_nonnegative_xy


def rotate_point_2d(x: float, y: float, angle_deg: float) -> Tuple[float, float]:
    """
    将 2D 点绕原点逆时针旋转指定角度
    
    Args:
        x, y: 原始坐标
        angle_deg: 旋转角度（度）
    
    Returns:
        (x', y'): 旋转后的坐标
    """
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    new_x = x * cos_a - y * sin_a
    new_y = x * sin_a + y * cos_a
    
    return new_x, new_y


def _normalize_angle_deg(angle: float) -> float:
    """
    归一化角度到 (-180, 180] 区间。
    """
    while angle > 180:
        angle -= 360
    while angle <= -180:
        angle += 360
    return angle


def rotate_scene(scene_data: Dict[str, Any], angle_deg: float) -> Dict[str, Any]:
    """
    将整个场景绕 Z 轴旋转指定角度
    
    旋转内容：
    1. boundary 的所有点
    2. 所有物体的 center
    3. 所有物体的 rotation（仅更新 yaw）
    
    Args:
        scene_data: 原始场景数据
        angle_deg: 旋转角度（度），正值为逆时针
    
    Returns:
        旋转后的场景数据（深拷贝）
    """
    rotated = deepcopy(scene_data)
    
    # 更新 scene_id，加上旋转角度后缀
    original_id = rotated.get("scene_id", "unknown")
    rotated["scene_id"] = f"{original_id}_rot{int(angle_deg)}"
    
    # 1. 旋转 boundary
    new_boundary = []
    for point in rotated["boundary"]:
        x, y = point[0], point[1]
        new_x, new_y = rotate_point_2d(x, y, angle_deg)
        new_boundary.append([round(new_x, 2), round(new_y, 2)])
    rotated["boundary"] = new_boundary
    
    # 2. 旋转所有物体
    for obj in rotated["objects"]:
        # 旋转位置 (center)
        center = obj.get("center")
        if not center or len(center) < 2:
            raise ValueError(
                f"转换格式错误：缺少 center。scene_id={rotated.get('scene_id')} jid={obj.get('jid')}"
            )
        new_x, new_y = rotate_point_2d(center[0], center[1], angle_deg)
        obj["center"] = [round(new_x, 2), round(new_y, 2), center[2]]

        # 旋转朝向 (rotation) - 只更新 yaw
        rotation = obj.get("rotation")
        if not rotation or len(rotation) < 3:
            raise ValueError(
                f"转换格式错误：缺少 rotation。scene_id={rotated.get('scene_id')} jid={obj.get('jid')}"
            )
        new_yaw = _normalize_angle_deg(float(rotation[2]) + angle_deg)
        obj["rotation"] = [rotation[0], rotation[1], round(new_yaw)]
    
    # 3. 平移到非负坐标，boundary 最小点为 (0, 0)
    translate_scene_to_nonnegative_xy(rotated)
    
    return rotated


def augment_scene_file(
    input_path: Path, 
    output_dir: Path, 
    angles: List[float],
) -> int:
    """
    对单个场景文件进行旋转增强
    
    原始 0 度场景保持不变，只在同目录下添加旋转版本。
    若旋转文件已存在则直接覆盖。
    
    Args:
        input_path: 输入场景文件路径
        output_dir: 输出目录
        angles: 旋转角度列表（如 [90, 180, 270]）
    
    Returns:
        生成的文件数量（仅旋转版本）
    """
    with open(input_path, "r", encoding="utf-8") as f:
        scene_data = json.load(f)
    
    count = 0
    stem = input_path.stem
    
    # 生成旋转增强版本（原始文件不动, 旋转后的文件存在则覆盖）
    for angle in angles:
        output_name = f"{stem}_rot{int(angle)}.json"
        output_path = output_dir / output_name
        
        rotated = rotate_scene(scene_data, angle)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(rotated, f, indent=2, ensure_ascii=False)
        count += 1
    
    return count


def augment_directory(
    input_dir: Path, 
    output_dir: Path, 
    angles: List[float],
):
    """
    批量处理目录中的所有场景文件
    
    原始场景保持不变，只添加旋转版本。
    跳过已有的 _rot 后缀文件，避免递归处理。
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录（可以与输入目录相同）
        angles: 旋转角度列表
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 只处理原始场景，跳过已有的旋转版本（带 _rot 后缀）
    import re
    rot_pattern = re.compile(r"_rot\d+$")
    json_files = sorted([
        f for f in input_path.glob("*.json") 
        if not rot_pattern.search(f.stem)
    ])
    
    print(f"找到 {len(json_files)} 个原始场景文件（跳过 _rot 后缀）")
    print(f"旋转角度: {angles}")
    print(f"预期新增: 最多 {len(json_files)} * {len(angles)} = {len(json_files) * len(angles)} 个文件\n")
    
    total_count = 0
    
    for json_file in tqdm(json_files, desc="增强场景"):
        try:
            count = augment_scene_file(
                json_file, 
                output_path, 
                angles,
            )
            total_count += count
        except Exception as e:
            tqdm.write(f"处理失败 {json_file.name}: {e}")
    
    print(f"\n增强完成: 共新增 {total_count} 个旋转场景文件")


def main():
    parser = argparse.ArgumentParser(
        description="场景旋转增强脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--input-dir", 
        type=str, 
        default=str(ProjectPaths.IMAGINARIUM_DIR / "converted_scenes"),
        help="输入目录（包含 converted 格式的场景 JSON 文件）"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=None,
        help="输出目录（默认与输入目录相同，直接在原目录添加旋转场景）"
    )
    parser.add_argument(
        "--angles", 
        type=float, 
        nargs="+", 
        default=[90, 180, 270],
        help="旋转角度列表（度），默认 90 180 270"
    )
    
    args = parser.parse_args()
    
    # 输出目录默认与输入目录相同
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    
    print("=" * 60)
    print("场景旋转增强")
    print("=" * 60)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"旋转角度: {args.angles}")
    print()
    
    augment_directory(
        input_dir,
        output_dir,
        args.angles,
    )


if __name__ == "__main__":
    main()

# run
# python dataset_pipeline/preprocessing/02_augment_scene_rotation.py