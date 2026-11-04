import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.spatial.transform import Rotation as R

# 添加项目根目录到 Python 路径
from tqdm import tqdm
from config.paths import ProjectPaths
from dataset_pipeline.utils import (
    normalize_boundary_order,
    translate_scene_to_nonnegative_xy,
    get_object_z_bounds,
    short_desc_contains_keyword,
    filter_floating_objects,
)
import argparse


NON_POSITIVE_SIZE_EPS = 1e-6
SCENE_SIZE_DECIMALS = 6


def _has_nonpositive_dimension(size, eps: float = NON_POSITIVE_SIZE_EPS) -> bool:
    return any(float(dim) <= eps for dim in size)


def _round_scene_size(size, decimals: int = SCENE_SIZE_DECIMALS) -> list[float]:
    return [round(float(dim), decimals) for dim in size]


def quaternion_to_euler(q, degrees=True):
    """
    将四元数转换为欧拉角 (roll, pitch, yaw)
    q: [w, x, y, z] 格式的四元数
    返回: (roll, pitch, yaw) 使用 'xyz' 约定
    """
    # scipy 使用 [x, y, z, w] 格式
    rot = R.from_quat([q[1], q[2], q[3], q[0]])
    # 使用 scipy 的小写 "xyz" 欧拉角约定（intrinsic，绕物体自身坐标轴依次旋转）
    euler = rot.as_euler("xyz", degrees=degrees)
    return euler  # [roll, pitch, yaw]


def convert_scene_file(input_path, output_path, new_scene_id=None):
    """
    坐标变换: (x, y, z) -> (x, -z, y)
    new_scene_id: 可选参数，用于覆盖原始的 scene_id（处理重复 scene_id 时使用）
    """
    with open(input_path, "r") as f:
        data = json.load(f)

    converted_data = {}

    converted_data["scene_type"] = data["room_type"]
    converted_data["scene_id"] = new_scene_id if new_scene_id else data["room_id"]

    # 转换房间边界 - bounds_bottom
    if "bounds_bottom" in data:
        converted_data["boundary"] = []
        for point in data["bounds_bottom"]:
            new_point = [point[0], -point[2]]  # 2d points in xy plane
            converted_data["boundary"].append(new_point)
        # 规范化边界点顺序：保证为稳定的环顺序（默认 CCW + 固定起点）
        converted_data["boundary"] = normalize_boundary_order(converted_data["boundary"])

    scene_height = data["bounds_top"][0][1] - data["bounds_bottom"][0][1]
    converted_data["scene_height"] = scene_height

    # 转换物体
    lamp_filtered_count = 0
    lamp_filtered_jids = []  # 记录被过滤的 lamp 物体的 jid
    if "objects" in data:
        converted_data["objects"] = []
        for obj in data["objects"]:
            # 过滤 short_desc 包含 "lamp" 的物体
            if short_desc_contains_keyword(obj, "lamp"):
                lamp_filtered_count += 1
                # 记录被过滤的 jid（去掉 scale 后缀）
                jid = obj["jid"].split("-(")[0]
                lamp_filtered_jids.append(jid)
                continue

            new_obj = {}

            # 转换尺寸 [width, height, depth] -> [width, depth, height]
            size = obj["size"]
            converted_size = [float(size[0]), float(size[2]), float(size[1])]
            if _has_nonpositive_dimension(converted_size):
                continue
            new_obj["size"] = _round_scene_size(converted_size)

            # 转换位置：从底部参考点转为中心点，并变换坐标系
            pos = obj["pos"]
            # 原坐标系中心点 = [x, y + height/2, z]
            center_orig = [pos[0], pos[1] + size[1] / 2, pos[2]]
            # 转换到新坐标系 [x, -z, y]，使用统一格式字段名 center
            new_obj["center"] = [round(center_orig[0], 2), round(-center_orig[2], 2), round(center_orig[1], 2)]

            # 转换四元数旋转
            # 原四元数 [x, y, z, w] (Y-up坐标系)
            quat = obj["rot"]

            # 将四元数转换为旋转矩阵
            rot = R.from_quat([quat[0], quat[1], quat[2], quat[3]])  # [x,y,z,w]
            rot_matrix = rot.as_matrix()

            # 坐标系变换矩阵: [x, y, z] -> [x, -z, y]
            coord_transform = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

            # 新旋转矩阵 = T * R * T^T
            new_rot_matrix = coord_transform @ rot_matrix @ coord_transform.T

            # 转回四元数并转为欧拉角，使用统一格式字段名 rotation
            new_rot = R.from_matrix(new_rot_matrix)
            euler = new_rot.as_euler("xyz", degrees=True).tolist()
            new_obj["rotation"] = [round(x) for x in euler]

            # jid: object id with scale to object id
            new_obj["jid"] = obj["jid"].split("-(")[0]
            
            # 复制其他字段
            new_obj["short_desc"] = obj.get("short_desc", "")
            new_obj["desc"] = obj.get("desc", "")

            converted_data["objects"].append(new_obj)

    # 过滤悬浮物体（没有地板或其他物体支撑的物体）
    filtered_count = 0
    if "objects" in converted_data and converted_data["objects"]:
        original_count = len(converted_data["objects"])
        converted_data["objects"] = filter_floating_objects(
            converted_data["objects"], floor_z=0, z_tolerance=0.05
        )
        filtered_count = original_count - len(converted_data["objects"])

    # 坐标系变换后，将场景平移到 x>=0, y>=0，并保证 boundary 外接矩形的最小点为 (0,0)
    # 平移向量由 boundary 的最小点计算，同时平移所有物体的 pos[x,y]
    translate_scene_to_nonnegative_xy(converted_data)

    # 在过滤悬浮物体后，检查异常物体：pos[2] < 0 或 旋转后 z_bottom < 0
    # 使用容差 -0.01 来过滤浮点精度误差
    NEGATIVE_THRESHOLD = -0.01
    pos_z_negative_count = 0
    z_bottom_negative_count = 0
    object_count = 0
    negative_z_objects = []  # 记录异常物体
    
    if "objects" in converted_data and converted_data["objects"]:
        input_filename = Path(input_path).name
        for obj in converted_data["objects"]:
            object_count += 1
            pos_z = obj["center"][2]
            z_bottom, z_top = get_object_z_bounds(obj)
            
            has_issue = False
            issue_info = {"jid": obj.get("jid")}
            
            if pos_z < NEGATIVE_THRESHOLD:
                pos_z_negative_count += 1
                has_issue = True
                issue_info["pos_z"] = round(pos_z, 3)
                print(f"[{input_filename}] Object pos[2] < 0: jid={obj.get('jid')}, pos[2]={pos_z:.3f}")
            
            if z_bottom < NEGATIVE_THRESHOLD:
                z_bottom_negative_count += 1
                has_issue = True
                issue_info["bbox_z_bottom"] = round(z_bottom, 3)
            
            if has_issue:
                negative_z_objects.append(issue_info)

    with open(output_path, "w") as f:
        json.dump(converted_data, f, indent=2)

    return {
        "lamp_filtered_count": lamp_filtered_count,
        "lamp_filtered_jids": lamp_filtered_jids,
        "filtered_count": filtered_count,
        "pos_z_negative_count": pos_z_negative_count,
        "z_bottom_negative_count": z_bottom_negative_count,
        "object_count": object_count,
        "negative_z_objects": negative_z_objects,
    }


def convert_directory(input_dir, output_dir):
    """
    批量转换目录中的所有场景文件
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if input_path.resolve() == output_path.resolve():
        raise ValueError("输入目录和输出目录不能相同，否则会在清理旧文件时删除源数据")

    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)

    old_json_files = list(output_path.glob("*.json"))
    if old_json_files:
        print(f"清理输出目录旧 JSON 文件: {len(old_json_files)} 个")
        for old_file in old_json_files:
            old_file.unlink()

    generated_log_paths = [
        output_path.parent / "lamp_filtered_jids_log.json",
        output_path.parent / "negative_z_objects_log.json",
        output_path.parent / "filtered_floating_objects_log.json",
    ]
    stale_log_paths = [log_path for log_path in generated_log_paths if log_path.exists()]
    if stale_log_paths:
        print(f"清理旧统计日志: {len(stale_log_paths)} 个")
        for log_path in stale_log_paths:
            log_path.unlink()

    # 查找所有 JSON 文件
    json_files = list(input_path.glob("*.json"))

    print(f"找到 {len(json_files)} 个场景文件")

    # 用于跟踪每个 scene_id 出现的次数，生成顺序编号
    scene_id_counter = defaultdict(int)
    duplicate_count = 0
    success_count = 0
    
    # 记录有悬浮物体被过滤的文件
    filtered_files = []
    
    # 统计异常物体
    total_pos_z_negative = 0
    total_z_bottom_negative = 0
    total_object_count = 0
    total_lamp_filtered = 0
    all_lamp_filtered_jids = []  # 记录所有被过滤的 lamp 物体的 jid
    
    # 记录有异常z值的场景
    negative_z_scenes = []
    
    with tqdm(total=len(json_files), desc="转换场景文件") as pbar:
        for json_file in json_files:
            # 读取文件获取 scene_id
            with open(json_file, "r") as f:
                data = json.load(f)
            
            scene_id = data.get("room_id")
            
            # 获取当前 scene_id 的序号，然后递增计数器
            current_idx = scene_id_counter[scene_id]
            if current_idx > 0:
                duplicate_count += 1
            scene_id_counter[scene_id] += 1

            # 使用顺序编号作为后缀
            new_scene_id = f"{scene_id}_{current_idx}"
            
            # 处理文件
            output_file = output_path / json_file.name
            result = convert_scene_file(json_file, output_file, new_scene_id)
            
            # 收集被过滤的 lamp jid
            all_lamp_filtered_jids.extend(result.get("lamp_filtered_jids", []))
            
            # 累计统计
            total_pos_z_negative += result["pos_z_negative_count"]
            total_z_bottom_negative += result["z_bottom_negative_count"]
            total_object_count += result["object_count"]
            total_lamp_filtered += result.get("lamp_filtered_count", 0)
            
            # 记录有异常z值的场景
            if result["negative_z_objects"]:
                negative_z_scenes.append({
                    "file": json_file.name,
                    "objects": result["negative_z_objects"]
                })
            
            # 记录有过滤的文件
            if result["filtered_count"] > 0:
                filtered_files.append({
                    "file": json_file.name,
                    "filtered_count": result["filtered_count"]
                })
            
            # 记录已处理的 scene_id（只记录原始的，用于检测后续重复）
            success_count += 1
            pbar.update(1)
    
    # 打印异常物体统计
    print(f"\n===== 异常物体统计 =====")
    print(f"pos_z < -0.01 的物体数量: {total_pos_z_negative}")
    print(f"rotated z_bottom < -0.01 的物体数量: {total_z_bottom_negative}")
    print(f"总共物体数量: {total_object_count}")
    print(f'short_desc 包含 "lamp" 被过滤的物体数量: {total_lamp_filtered}')
    
    # 统计并导出被过滤的 lamp 物体的 jid
    if all_lamp_filtered_jids:
        unique_lamp_jids = sorted(set(all_lamp_filtered_jids))
        lamp_jid_counts = defaultdict(int)
        for jid in all_lamp_filtered_jids:
            lamp_jid_counts[jid] += 1
        # 按出现次数降序排序
        sorted_lamp_jids = sorted(lamp_jid_counts.items(), key=lambda x: -x[1])
        
        lamp_log_path = output_path.parent / "lamp_filtered_jids_log.json"
        lamp_log_data = {
            "total_count": len(all_lamp_filtered_jids),
            "unique_count": len(unique_lamp_jids),
            "jid_counts": dict(sorted_lamp_jids),
        }
        with open(lamp_log_path, "w") as f:
            json.dump(lamp_log_data, f, indent=2, ensure_ascii=False)
        print(f'被过滤的 lamp 物体有 {len(unique_lamp_jids)} 种不同的 jid')
        print(f"lamp jid 记录已保存到: {lamp_log_path}")
    
    # 导出有异常z值的场景记录
    if negative_z_scenes:
        negative_z_log_path = output_path.parent / "negative_z_objects_log.json"
        with open(negative_z_log_path, "w") as f:
            json.dump(negative_z_scenes, f, indent=2, ensure_ascii=False)
        print(f"有 {len(negative_z_scenes)} 个场景包含异常z值物体")
        print(f"异常z值记录已保存到: {negative_z_log_path}")
    
    # 导出有悬浮物体被过滤的文件列表
    if filtered_files:
        filter_log_path = output_path.parent / "filtered_floating_objects_log.json"
        with open(filter_log_path, "w") as f:
            json.dump(filtered_files, f, indent=2, ensure_ascii=False)
        total_filtered = sum(item["filtered_count"] for item in filtered_files)
        print(f"\n有 {len(filtered_files)} 个文件包含被过滤的悬浮物体，共过滤 {total_filtered} 个物体")
        print(f"过滤记录已保存到: {filter_log_path}")

    print(
        f"\n全部转换完成, 共处理 {success_count} 个文件,"
        f" 其中 {duplicate_count} 个重复场景已添加顺序编号后缀"
    )


if __name__ == "__main__":
    print("=" * 80)
    print("SSR-3DFRONT 数据集转换工具")
    print("=" * 80)
    
    # 验证必需的外部数据集
    status = ProjectPaths.validate_external_datasets(required=["SSR-3DFRONT"])
    if status is None:
        sys.exit(1)
    
    # 使用统一的路径配置
    ssr_dir = ProjectPaths.SSR_3DFRONT_DIR
    ssr_input_dir = ssr_dir / "raw_scenes_with_short_obj_desc"
    ssr_output_dir = ssr_dir / "converted_scenes"

    parser = argparse.ArgumentParser(description="Convert Respace dataset to our format")
    parser.add_argument("--input-dir", type=str, default=str(ssr_input_dir), help="Input directory")
    parser.add_argument("--output-dir", type=str, default=str(ssr_output_dir), help="Output directory")
    args = parser.parse_args()

    convert_directory(args.input_dir, args.output_dir)

# python dataset_pipeline/preprocessing/01_convert_respace_dataset.py
