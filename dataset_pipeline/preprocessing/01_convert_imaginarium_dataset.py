"""
将 Imaginarium 数据集转换为统一的中间格式，然后可以使用 generate_autoregressive_data.py 生成自回归训练样本。
"""

import json
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
from scipy.spatial.transform import Rotation as R

# 添加项目根目录到 Python 路径

from config.paths import ProjectPaths
from dataset_pipeline.utils import (
    translate_scene_to_nonnegative_xy,
    filter_floating_objects,
)
from utils.spatial_utils import (
    quat_to_rotation_matrix,
    get_obb_world_vertices,
    get_aabb_from_vertices,
)


# 有问题的 FBX 资产黑名单（从 issue_asset_id.txt 读取）
ISSUE_ASSET_FILENAME = "issue_asset_id.txt"
ISSUE_ASSET_CACHE: dict[Path, set[str]] = {}


def load_issue_asset_ids(input_dir: Path) -> set[str]:
    """
    从与 filtered_scene_id.txt 相同目录的 issue_asset_id.txt 读取问题资产列表

    Args:
        input_dir: 输入目录（flux_train_data）

    Returns:
        去掉 .fbx 后缀的资产 id 集合
    """
    issue_file = input_dir.parent / ISSUE_ASSET_FILENAME
    if issue_file in ISSUE_ASSET_CACHE:
        return ISSUE_ASSET_CACHE[issue_file]

    if not issue_file.exists():
        ISSUE_ASSET_CACHE[issue_file] = set()
        return ISSUE_ASSET_CACHE[issue_file]

    issue_assets = set()
    with open(issue_file, "r", encoding="utf-8") as f:
        for line in f:
            asset_id = line.strip()
            if not asset_id:
                continue
            if asset_id.endswith(".fbx"):
                asset_id = asset_id[:-4]
            issue_assets.add(asset_id)

    ISSUE_ASSET_CACHE[issue_file] = issue_assets
    return issue_assets


def is_asset_blacklisted(fbx_name: str | None, blacklisted_assets: set[str]) -> bool:
    """
    检查物体是否在黑名单中（有问题的 FBX：origin 不在 bbx 中心或旋转没重置）

    直接使用原始 JSON 中的 fbx_name 字段进行匹配。

    Args:
        fbx_name: 原始 JSON 中的 fbx_name 字段，如 "d_1000003453475.fbx"

    Returns:
        是否在黑名单中
    """
    if not fbx_name or not blacklisted_assets:
        return False

    # 去掉 .fbx 扩展名
    base_name = fbx_name.replace(".fbx", "")

    return base_name in blacklisted_assets


# fbx_name 到 CSV name_en 的特殊映射（处理不一致的命名）
FBX_NAME_MAPPING = {
    "a_SM_Frame_02_B": "a_SM_Frame_02_B.003",
    "SM_a_SM_CentralTable_01": "a_SM_CentralTable_01",
}


def get_asset_name(fbx_name: str | None) -> str | None:
    """
    从 fbx_name 获取 CSV 中的资产名（去掉 .fbx 扩展名，并处理特殊映射）

    Args:
        fbx_name: 原始 JSON 中的 fbx_name 字段，如 "d_1000003453475.fbx"

    Returns:
        资产名（与 CSV name_en 一致）；若 fbx_name 为空则返回 None
    """
    if not fbx_name:
        return None
    base_name = fbx_name.replace(".fbx", "")
    return FBX_NAME_MAPPING.get(base_name, base_name)


def is_object_in_csv(fbx_name: str | None, asset_info: Dict[str, Dict]) -> bool:
    """
    检查物体是否在 CSV 中存在

    直接使用 fbx_name 匹配 CSV 中的 name_en。

    Args:
        fbx_name: 原始 JSON 中的 fbx_name 字段
        asset_info: 物体尺寸信息字典

    Returns:
        是否在 CSV 中找到
    """
    asset_name = get_asset_name(fbx_name)
    if not asset_name:
        return False
    return asset_name in asset_info


def get_object_size(
    obj_data: Dict[str, Any], asset_info: Dict[str, Dict]
) -> List[float]:
    """
    物体尺寸获取函数

    计算方式: CSV原始尺寸 * JSON中的scale
    直接使用 fbx_name 匹配 CSV。

    Args:
        obj_data: 物体数据（包含 fbx_name 和 scale）
        asset_info: 物体尺寸信息字典

    Returns:
        物体尺寸 [width, depth, height]
    """
    # 获取 JSON 中的缩放比例
    obj_scale = obj_data.get("scale", [1.0, 1.0, 1.0])

    # 从 fbx_name 获取资产名
    asset_name = get_asset_name(obj_data.get("fbx_name"))
    if not asset_name or asset_name not in asset_info:
        raise ValueError(f"物体 fbx_name={obj_data.get('fbx_name')} 在 CSV 中找不到")

    csv_size = asset_info[asset_name]["original_size"]

    # 实际尺寸 = CSV原始尺寸 × JSON缩放比例
    size = [csv_size[i] * abs(obj_scale[i]) for i in range(3)]
    return size


def load_asset_info(csv_path: str) -> Dict[str, Dict[str, Any]]:
    """
    从CSV文件加载物体的原始尺寸信息

    返回:
    - asset_info: 物体尺寸信息字典，key 为 name_en（与 fbx_name 去掉 .fbx 一致）
    """
    asset_info = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name_en"]
            bbx_str = row["bbx"]
            if bbx_str:
                bbx = [float(x) for x in bbx_str.split(",")]
                short_desc = (row.get("short_desc") or "").strip()
                if not short_desc:
                    raise ValueError(f"short_desc 为空: {name}")
                info = {
                    "original_size": bbx,
                    "class_en": row.get("class_en", ""),
                    "caption_en": row.get("caption_en", ""),
                    "short_desc": short_desc,
                }
                asset_info[name] = info

    return asset_info


def matrix_to_position_rotation_quaternion(matrix_4x4: List[List[float]]) -> tuple:
    """从 4x4 变换矩阵提取位置、四元数和欧拉角"""
    matrix = np.array(matrix_4x4)
    position = matrix[:3, 3].tolist()
    rotation_matrix = matrix[:3, :3]

    # 检查并修正镜像变换
    det = np.linalg.det(rotation_matrix)
    if det < 0:
        rotation_matrix[:, 0] *= -1

    # 归一化旋转矩阵
    for i in range(3):
        norm = np.linalg.norm(rotation_matrix[:, i])
        if norm > 1e-6:
            rotation_matrix[:, i] /= norm

    try:
        rot = R.from_matrix(rotation_matrix)
        quat = rot.as_quat()  # [x, y, z, w]
        quaternion = [quat[3], quat[0], quat[1], quat[2]]  # [w, x, y, z]
        euler_angles = rot.as_euler("xyz", degrees=True).tolist()
    except Exception as exc:
        raise ValueError(
            f"旋转矩阵解析失败，无法回退默认旋转: det={det:.6f}, matrix={rotation_matrix.tolist()}"
        ) from exc

    return position, quaternion, euler_angles


def calculate_tight_boundary_from_objects(
    filtered_objects_data: List[Tuple[str, Dict]], asset_info: Dict[str, Dict]
) -> List[List[float]]:
    """
    基于过滤后的家具物体计算紧凑边界（考虑物体旋转）
    """
    if not filtered_objects_data:
        return [[0, 0], [4, 0], [4, 4], [0, 4]]

    min_x = float("inf")
    max_x = float("-inf")
    min_y = float("inf")
    max_y = float("-inf")

    for obj_name, obj_data in filtered_objects_data:
        # 获取物体位置和旋转
        pos, quat, _ = matrix_to_position_rotation_quaternion(
            obj_data["matrix_world_4x4"]
        )

        # 获取物体尺寸
        size = obj_data["size"]
        # size = get_object_size(obj_data, asset_info)

        # 使用旋转后的 OBB 计算 XY 平面的 AABB（考虑物体旋转）
        rot_matrix = quat_to_rotation_matrix(quat)
        world_vertices = get_obb_world_vertices(pos, size, rot_matrix)
        v_min, v_max = get_aabb_from_vertices(world_vertices)

        obj_min_x = float(v_min[0])
        obj_max_x = float(v_max[0])
        obj_min_y = float(v_min[1])
        obj_max_y = float(v_max[1])

        # 更新全局边界
        min_x = min(min_x, obj_min_x)
        max_x = max(max_x, obj_max_x)
        min_y = min(min_y, obj_min_y)
        max_y = max(max_y, obj_max_y)

    # 不添加边距
    margin = 0.0
    min_x -= margin
    max_x += margin
    min_y -= margin
    max_y += margin

    # 逆时针排列四个角点
    boundary = [
        [round(min_x, 2), round(min_y, 2)],
        [round(max_x, 2), round(min_y, 2)],
        [round(max_x, 2), round(max_y, 2)],
        [round(min_x, 2), round(max_y, 2)],
    ]

    return boundary


def calculate_tight_height_from_objects(
    filtered_objects_data: List[Tuple[str, Dict]], asset_info: Dict[str, Dict]
) -> float:
    """
    基于过滤后的地面家具物体计算紧凑高度（考虑物体旋转）
    """
    if not filtered_objects_data:
        return 2.8

    max_height = 0.0

    for obj_name, obj_data in filtered_objects_data:
        # 获取物体位置和旋转
        pos, quat, _ = matrix_to_position_rotation_quaternion(
            obj_data["matrix_world_4x4"]
        )

        # 获取物体尺寸
        size = obj_data["size"]
        # size = get_object_size(obj_data, asset_info)

        # 使用旋转后的 OBB 计算顶部高度（考虑物体旋转/倾斜）
        rot_matrix = quat_to_rotation_matrix(quat)
        world_vertices = get_obb_world_vertices(pos, size, rot_matrix)
        _, v_max = get_aabb_from_vertices(world_vertices)
        obj_top = float(v_max[2])

        max_height = max(max_height, obj_top)

    # 添加一点余量(30cm),让天花板稍微高一点
    scene_height = max_height + 0.3

    # 确保最小高度为2.8米
    scene_height = max(scene_height, 2.8)

    return round(scene_height, 2)


def convert_scene_file(
    input_path: Path,
    output_path: Path,
    asset_info: Dict[str, Dict[str, Any]],
    blacklisted_assets: set[str],
) -> Tuple[int, List[str], int]:
    """转换单个场景文件到中间格式

    Args:
        input_path: 输入JSON文件路径
        output_path: 输出JSON文件路径
        asset_info: 从CSV加载的物体原始尺寸信息
        blacklisted_assets: 黑名单资产 id 集合（去掉 .fbx 后缀）

    Returns:
        (object_count, missing_objects, floating_filtered_count):
        物体数量、CSV中找不到的物体列表、过滤的悬浮物体数量
    """
    with open(input_path, "r", encoding="utf-8") as f:
        scene_data = json.load(f)

    # 若输入文件已经是转换后的中间格式（objects 为 list 且含 boundary），直接写入
    if isinstance(scene_data.get("objects"), list) and "boundary" in scene_data:
        translate_scene_to_nonnegative_xy(scene_data)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(scene_data, f, indent=2, ensure_ascii=False)
        return len(scene_data["objects"]), [], 0

    converted_data = {}

    # 场景基本信息
    scene_name = scene_data.get("scene_name", input_path.stem.replace("_meta", ""))
    scene_type = scene_name.split("_")[0] if "_" in scene_name else "room"

    converted_data["scene_type"] = scene_type
    converted_data["scene_id"] = scene_name

    objects_dict = scene_data.get("objects", {})

    # 第一阶段：收集需要删除的物体名称
    deleted_objects = set()
    csv_missing_objects = set()  # 单独记录因 CSV 中找不到而被删除的物体
    blacklisted_objects = set()  # 记录因黑名单而被删除的物体

    for obj_name, obj_data in objects_dict.items():
        obj_name_lower = obj_name.lower()

        # 过滤非 MESH 类型
        if obj_data.get("type") != "MESH":
            deleted_objects.add(obj_name)
            continue

        # 过滤 wall, ceiling, floor, window, door
        if (
            obj_name_lower.startswith("wall")
            or obj_name_lower.startswith("ceiling")
            or obj_name_lower.startswith("floor")
            or "window" in obj_name_lower
            or "door" in obj_name_lower
        ):
            deleted_objects.add(obj_name)
            continue

        # 排除 parent 包含 'wall', 'ceiling', 'window', 'door' 的物体
        parent = obj_data.get("parent", "")
        if parent:
            parent_lower = parent.lower()
            if (
                "wall" in parent_lower
                or "ceiling" in parent_lower
                or "window" in parent_lower
                or "door" in parent_lower
            ):
                deleted_objects.add(obj_name)
                continue

        # 检查物体是否在黑名单中（有问题的 FBX：origin 不在中心或旋转不为零）
        fbx_name = obj_data.get("fbx_name")
        if is_asset_blacklisted(fbx_name, blacklisted_assets):
            deleted_objects.add(obj_name)
            blacklisted_objects.add(obj_name)
            continue

        # 检查物体是否在 CSV 中存在
        if not is_object_in_csv(fbx_name, asset_info):
            deleted_objects.add(obj_name)
            csv_missing_objects.add(obj_name)  # 记录 CSV 中找不到的物体
            continue

    # 第二阶段：递归删除 - 如果 parent 被删除，其子物体也要删除 (父物体是地板时除外)
    changed = True
    while changed:
        changed = False
        for obj_name, obj_data in objects_dict.items():
            if obj_name in deleted_objects:
                continue
            parent = obj_data.get("parent", "")
            if (
                parent
                and parent in deleted_objects
                and not parent.lower().startswith("floor")
            ):
                deleted_objects.add(obj_name)
                changed = True

    # 第三阶段：构建最终过滤后的物体列表
    filtered_objects_data = []
    skipped_objects = list(csv_missing_objects)  # 仅记录 CSV 中找不到的物体

    for obj_name, obj_data in objects_dict.items():
        if obj_name in deleted_objects:
            continue

        filtered_objects_data.append((obj_name, obj_data))

    # 如果有黑名单物体,打印警告
    if blacklisted_objects:
        print(f"  ⚠️  跳过 {len(blacklisted_objects)} 个黑名单物体（FBX有问题）:")
        for obj in sorted(blacklisted_objects):
            print(f"      - {obj}")

    # 如果有跳过的物体,打印警告
    if skipped_objects:
        print(f"  ⚠️  跳过 {len(skipped_objects)} 个CSV中找不到的物体:")
        for obj in skipped_objects:
            print(f"      - {obj}")

    if len(filtered_objects_data) == 0:
        return 0, skipped_objects, 0

    # 计算边界
    converted_data["boundary"] = calculate_tight_boundary_from_objects(
        filtered_objects_data, asset_info
    )

    # 计算高度
    converted_data["scene_height"] = calculate_tight_height_from_objects(
        filtered_objects_data, asset_info
    )

    # 转换物体
    converted_data["objects"] = []

    for obj_name, obj_data in filtered_objects_data:
        pos, quat, euler = matrix_to_position_rotation_quaternion(
            obj_data["matrix_world_4x4"]
        )

        size = obj_data["size"]
        # 从CSV获取物体的原始尺寸 (CSV原始尺寸 × JSON缩放比例)
        # original_size = get_object_size(obj_data, asset_info)

        # 从 fbx_name 获取资产名（用于 jid 和 CSV 查询）
        fbx_name = obj_data.get("fbx_name")
        asset_name = get_asset_name(fbx_name) or ""

        # desc 优先使用 JSON 的 caption_en；为空时用 jid 去 CSV(asset_info) 里补齐
        caption_en_json = (obj_data.get("caption_en") or "").strip()
        caption_en_csv = (
            (asset_info.get(asset_name, {}) or {}).get("caption_en", "") or ""
        ).strip()
        caption_en_final = caption_en_json or caption_en_csv
        if not caption_en_final:
            print(
                f"  ⚠️  caption_en 为空(含CSV): scene={scene_name} obj_name={obj_name} jid={asset_name}"
            )

        # short_desc 优先使用 JSON；为空时用 jid 去 CSV(asset_info) 里补齐
        short_desc_json = (obj_data.get("short_desc") or "").strip()
        short_desc_csv = (
            (asset_info.get(asset_name, {}) or {}).get("short_desc", "") or ""
        ).strip()
        short_desc_final = short_desc_json or short_desc_csv
        if not short_desc_final:
            print(
                f"  ⚠️  short_desc 为空(含CSV): scene={scene_name} obj_name={obj_name} jid={asset_name}"
            )

        # 使用统一格式：center + rotation（欧拉角）
        new_obj = {
            "desc": caption_en_final,
            "short_desc": short_desc_final,
            "jid": asset_name,
            "center": [
                round(pos[0], 2),
                round(pos[1], 2),
                round(pos[2] - 0.1, 2),
            ],  # 减去地板的厚度
            "rotation": [round(e) for e in euler],
            "size": [round(size[0], 2), round(size[1], 2), round(size[2], 2)],
        }
        converted_data["objects"].append(new_obj)

    # 过滤悬浮物体
    floating_filtered_count = 0
    # if converted_data["objects"]:
    #     original_count = len(converted_data["objects"])
    #     converted_data["objects"] = filter_floating_objects(
    #         converted_data["objects"],
    #         floor_z=0,
    #         z_tolerance=0.05,
    #     )
    #     floating_filtered_count = original_count - len(converted_data["objects"])
    #     if floating_filtered_count > 0:
    #         print(f"  ⚠️  过滤了 {floating_filtered_count} 个悬浮物体")

    # 将场景平移到 x>=0, y>=0，并保证 boundary 外接矩形的最小点为 (0,0)
    # 平移向量由 boundary 的最小点计算，同时平移所有物体的 pos[x,y]
    translate_scene_to_nonnegative_xy(converted_data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)

    return len(converted_data["objects"]), skipped_objects, floating_filtered_count


def load_filtered_scene_ids(input_dir: Path) -> set:
    """
    加载需要过滤的场景 ID 列表

    Args:
        input_dir: 输入目录（flux_train_data）

    Returns:
        需要过滤的场景 ID 集合
    """
    filtered_file = input_dir.parent / "filtered_scene_id.txt"
    if not filtered_file.exists():
        return set()

    filtered_ids = set()
    with open(filtered_file, "r", encoding="utf-8") as f:
        for line in f:
            scene_id = line.strip()
            if scene_id:
                filtered_ids.add(scene_id)

    return filtered_ids


def convert_directory(input_dir: Path, output_dir: Path, csv_path: Path):
    """批量转换目录中的所有场景文件

    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        csv_path: CSV文件路径(包含物体原始尺寸)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if input_path.resolve() == output_path.resolve():
        raise ValueError("输入目录和输出目录不能相同，否则会在清理旧文件时删除源数据")

    output_path.mkdir(parents=True, exist_ok=True)

    old_json_files = list(output_path.glob("*.json"))
    if old_json_files:
        print(f"清理输出目录旧 JSON 文件: {len(old_json_files)} 个")
        for old_file in old_json_files:
            old_file.unlink()

    # 加载CSV中的物体原始尺寸信息
    print(f"加载物体信息: {csv_path}")
    asset_info = load_asset_info(str(csv_path))
    print(f"  加载了 {len(asset_info)} 个物体的原始尺寸\n")

    # 加载需要过滤的场景 ID
    filtered_scene_ids = load_filtered_scene_ids(input_path)
    if filtered_scene_ids:
        print(f"加载过滤列表: {len(filtered_scene_ids)} 个场景将被跳过")
        for scene_id in sorted(filtered_scene_ids):
            print(f"  - {scene_id}")
        print()

    # 加载黑名单资产
    blacklisted_assets = load_issue_asset_ids(input_path)

    meta_files = sorted(input_path.glob("*_meta.json"))  # 排序确保顺序一致
    print(f"找到 {len(meta_files)} 个场景文件\n")

    success_count = 0
    empty_count = 0  # 过滤后没有物体的场景数量
    total_objects = 0
    total_floating_filtered = 0
    missing_objects_map = {}  # 记录CSV中找不到的物体: {obj_name: [scene_names]}
    floating_filtered_files = []  # 记录有悬浮物体被过滤的文件
    failed_files = []  # 记录转换失败文件，避免错误被淹没

    filtered_count = 0  # 被过滤的场景数量
    for meta_file in meta_files:
        # 从文件名提取场景 ID（去掉 _meta.json 后缀）
        scene_id = meta_file.stem.replace("_meta", "")
        
        # 检查是否需要过滤该场景
        if scene_id in filtered_scene_ids:
            filtered_count += 1
            print(f"⊘ {meta_file.name}: 在过滤列表中，跳过")
            continue

        try:
            output_file = output_path / meta_file.name.replace("_meta.json", ".json")
            obj_count, skipped_objects, floating_filtered_count = convert_scene_file(
                meta_file, output_file, asset_info, blacklisted_assets
            )
            if obj_count is not None and obj_count > 0:
                success_count += 1
                total_objects += obj_count
                total_floating_filtered += floating_filtered_count
                print(f"✓ {meta_file.name}: {obj_count} 个物体")
                # 收集缺失的物体及其所在场景
                for obj_name in skipped_objects:
                    if obj_name not in missing_objects_map:
                        missing_objects_map[obj_name] = []
                    missing_objects_map[obj_name].append(meta_file.name)
                # 记录有悬浮物体被过滤的文件
                if floating_filtered_count > 0:
                    floating_filtered_files.append(
                        {
                            "file": meta_file.name,
                            "filtered_count": floating_filtered_count,
                        }
                    )
            else:
                empty_count += 1
                print(f"⊘ {meta_file.name}: 过滤后无物体，跳过")
        except Exception as e:
            print(f"✗ {meta_file.name}: {str(e)}")
            failed_files.append({"file": meta_file.name, "error": str(e)})

    print(
        f"\n转换完成: {success_count}/{len(meta_files)} 个场景, 共 {total_objects} 个物体"
    )
    if filtered_count > 0:
        print(f"  （{filtered_count} 个场景在过滤列表中被跳过）")
    if empty_count > 0:
        print(f"  （{empty_count} 个场景过滤后无物体）")
    if failed_files:
        print(f"  （{len(failed_files)} 个场景转换失败）")
        for item in failed_files[:20]:
            print(f"    - {item['file']}: {item['error']}")

    # 输出悬浮物体过滤统计
    if total_floating_filtered > 0:
        print(
            f"共过滤 {total_floating_filtered} 个悬浮物体（来自 {len(floating_filtered_files)} 个场景）"
        )

    # 汇总CSV中找不到的物体
    if missing_objects_map:
        print(f"\n⚠️  CSV中找不到的物体 (共 {len(missing_objects_map)} 个):")
        for obj_name in sorted(missing_objects_map.keys()):
            scenes = missing_objects_map[obj_name]
            if len(scenes) == 1:
                print(f"  - {obj_name}")
                print(f"      出现在: {scenes[0]}")
            else:
                print(f"  - {obj_name} (出现 {len(scenes)} 次)")
                for scene in scenes:
                    print(f"      - {scene}")
        print(f"\n这些物体使用了JSON中的size作为fallback")


if __name__ == "__main__":
    print("=" * 80)
    print("Imaginarium 数据集转换工具 (转换为中间格式)")
    print("=" * 80)
    
    # 验证必需的外部数据集
    status = ProjectPaths.validate_external_datasets(
        required=["Imaginarium", "Imaginarium assets"]
    )
    if status is None:
        sys.exit(1)
    
    input_dir = ProjectPaths.IMAGINARIUM_DIR / "flux_train_data"
    output_dir = ProjectPaths.IMAGINARIUM_DIR / "converted_scenes"
    csv_path = ProjectPaths.IMAGINARIUM_ASSETS_DIR / "imaginarium_asset_info.csv"

    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"CSV文件: {csv_path}\n")

    if not input_dir.exists():
        print(f"错误: 输入目录不存在: {input_dir}")
        sys.exit(1)

    if not csv_path.exists():
        print(f"错误: CSV文件不存在: {csv_path}")
        sys.exit(1)

    convert_directory(input_dir, output_dir, csv_path)

# run
# python dataset_pipeline/preprocessing/01_convert_imaginarium_dataset.py
