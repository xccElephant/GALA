"""
消息格式化模块

统一管理 system prompt 和消息格式化逻辑
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from config.constants import RELATIVE_RELATION_TEXT
from utils.scene_types import (
    PlacementRequest,
    Scene,
    SceneObject,
    ensure_placement_request,
    ensure_scene,
    scene_object_list_to_dicts,
)

# TODO: 再输出 scale 属性, [x,y,z,roll,pitch,yaw,sw,sd,sh]
SYSTEM_PROMPT = """You are a scene layout assistant. Your task is to predict new object pose based on the current scene layout, geometry of objects and instruction.

## Coordinate System
X: right, Y: forward, Z: up. Ground plane: XY.

## Object Pose
- center: (X, Y, Z) in meters
- rotation: (roll, pitch, yaw) in degrees, range [0, 360). Yaw: rotation around Z, 0° points -Y, positive counterclockwise.
- size: (width, depth, height) in meters

## Physical Constraints
- The new object must rest on a supporting surface. Its bottom must not float in air.
- The new object must not overlap with any existing object. Ensure geometries do not intersect.

## Output Format
Return one list: [x, y, z, roll, pitch, yaw]
"""

def load_internal_spaces_map(spaces_dir: Path) -> Dict[str, List]:
    """
    加载收纳空间数据，返回 {jid: [subspaces]} 字典。
    每个 subspace 只保留 name/center/size。

    Args:
        spaces_dir: imaginarium_assets_internal_placement_space 目录路径

    Returns:
        {jid: [{"name": ..., "center": [...], "size": [...]}]}
    """
    spaces_map = {}
    spaces_dir = Path(spaces_dir)
    if not spaces_dir.exists():
        return spaces_map
    for json_file in spaces_dir.glob("*.json"):
        jid = json_file.stem
        with open(json_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        spaces_map[jid] = [
            {
                "name": sp["name"],
                "center": _round_list(sp["center"]),
                "size": _round_list(sp["size"]),
            }
            for sp in raw
        ]
    return spaces_map


def _normalize_rotation(rotation) -> list:
    """将旋转角度列表 [roll, pitch, yaw] 标准化到 [0, 360) 范围（整数）"""
    return [round(item % 360) for item in rotation]


def _round_list(data, decimals: int = 2):
    """对列表中的数值保留指定小数位数"""
    if isinstance(data, list):
        return [round(item, decimals) if isinstance(item, (int, float)) else item for item in data]
    return data


def _round_boundary(boundary, decimals: int = 2):
    """对边界点列表保留指定小数位数"""
    return [[round(coord, decimals) for coord in point] for point in boundary]


def _transform_space_center(
    local_center: list,
    obj_center: list,
    obj_rotation: list,
) -> list:
    """
    将收纳空间的局部坐标中心变换到世界坐标。

    资产坐标系约定：中心在原点，朝向 -Y，与场景中 yaw=0 对齐。
    变换：c_world = R(roll, pitch, yaw) @ c_local + obj_center
    """
    from utils.spatial_utils import euler_to_rotation_matrix

    roll, pitch, yaw = obj_rotation
    R = euler_to_rotation_matrix(roll, pitch, yaw)
    c_local = np.array(local_center)
    c_world = R @ c_local + np.array(obj_center)
    return _round_list(c_world.tolist())


def _with_spaces(obj_data: dict, jid: str, spaces_map: Optional[Dict]) -> dict:
    """
    若 jid 有收纳空间，将 internal_spaces 变换到世界坐标后附加到 obj_data。
    用底面中心点（bottom_center）+ 净高（height）表示每个子空间，
    便于模型直接判断物体的摆放位置和可用净空。
    """
    if spaces_map and jid in spaces_map:
        obj_center = obj_data["center"]
        obj_rotation = obj_data["rotation"]
        spaces = []
        for sp in spaces_map[jid]:
            world_center = _transform_space_center(sp["center"], obj_center, obj_rotation)
            height = round(sp["size"][2], 2)
            bottom_z = round(world_center[2] - height / 2, 2)
            spaces.append({
                "name": sp["name"],
                "bottom_center": [world_center[0], world_center[1], bottom_z],
                "height": height,
            })
        obj_data["internal_spaces"] = spaces
    return obj_data


def _get_object_label(obj: Dict, include_object_descriptions: bool) -> str:
    """按模式获取物体标签（short_desc 或 category），缺失时显式报错。"""
    key = "short_desc" if include_object_descriptions else "category"
    label = obj.get(key, "")
    if not label:
        raise ValueError(f"Object id={obj.get('id')} missing {key}")
    return label


def _build_existing_object_input(
    obj: Dict,
    internal_spaces_map: Optional[Dict[str, List]],
    include_object_descriptions: bool,
    include_object_pointcloud: bool = False,
    only_pointcloud: bool = False,
) -> Dict:
    label_key = "short_desc" if include_object_descriptions else "category"
    if only_pointcloud:
        obj_data = {
            "id": obj["id"],
            label_key: _get_object_label(obj, include_object_descriptions),
        }
        if include_object_pointcloud:
            obj_data["object_pointcloud"] = "<object_pointcloud>"
        return obj_data

    obj_data = {
        "id": obj["id"],
        label_key: _get_object_label(obj, include_object_descriptions),
        "center": _round_list(obj["center"]),
        "rotation": _normalize_rotation(obj["rotation"]),
        "size": _round_list(obj["size"]),
    }
    if include_object_pointcloud:
        obj_data["object_pointcloud"] = "<object_pointcloud>"

    jid = obj["jid"] if "jid" in obj else None
    return _with_spaces(obj_data, jid, internal_spaces_map)


def _build_new_object_input(
    new_object: Dict,
    include_object_descriptions: bool,
    pointcloud_field: Optional[str] = None,
    pointcloud_placeholder: Optional[str] = None,
    only_pointcloud: bool = False,
) -> Dict:
    label_key = "short_desc" if include_object_descriptions else "category"
    if only_pointcloud:
        obj_data = {
            "id": new_object["id"],
            label_key: _get_object_label(new_object, include_object_descriptions),
        }
        if pointcloud_field is not None:
            obj_data[pointcloud_field] = pointcloud_placeholder
        return obj_data

    obj_data = {
        "id": new_object["id"],
        label_key: _get_object_label(new_object, include_object_descriptions),
        "size": _round_list(new_object["size"]),
    }
    if pointcloud_field is not None:
        obj_data[pointcloud_field] = pointcloud_placeholder
    return obj_data


def build_model_instruction(sample: Dict, include_object_descriptions: bool) -> str:
    """
    构建训练/推理输入中的 instruction。

    优先使用 expected_relations 渲染 instruction；仅当 relations 缺失时，
    才直接返回原始 instruction。
    """
    instruction = sample.get("instruction", "")
    expected_relations = sample.get("expected_relations")
    if expected_relations is None:
        return instruction

    if not isinstance(expected_relations, dict):
        raise TypeError("expected_relations must be a dict")

    absolute_relations = expected_relations.get("absolute_relations")
    relative_relations = expected_relations.get("relative_relations")
    if not isinstance(absolute_relations, list) or not isinstance(relative_relations, list):
        raise TypeError("expected_relations.absolute_relations and relative_relations must be lists")

    new_object = sample.get("new_object", {})
    existing_objects = sample.get("existing_objects", [])
    ref_object_map = {obj["id"]: obj for obj in existing_objects}
    new_obj_label = _get_object_label(new_object, include_object_descriptions)

    instruction_parts: List[str] = [str(rel) for rel in absolute_relations]
    for rel in relative_relations:
        rel_type = rel.get("type")
        target_id = rel.get("target_id")
        if not isinstance(target_id, int) or rel_type is None:
            raise ValueError(
                f"relative_relations[*] must have 'type' (str) and 'target_id' (int), "
                f"got: {rel}"
            )
        rel_desc = RELATIVE_RELATION_TEXT.get(rel_type)
        if rel_desc is None:
            raise ValueError(
                f"Unknown relation type in expected_relations: {rel_type}"
            )
        if target_id not in ref_object_map:
            available_ids = sorted(ref_object_map.keys())
            raise ValueError(
                f"relative_relations references target_id={target_id} "
                f"(type={rel_type}) not found in existing_objects; "
                f"available ids={available_ids}"
            )
        ref_label = _get_object_label(ref_object_map[target_id], include_object_descriptions)
        instruction_parts.append(f"{rel_desc} the {ref_label} (id {target_id})")

    if not instruction_parts:
        return f"place the {new_obj_label} in a suitable position."
    return f"place the {new_obj_label} " + ", and ".join(instruction_parts) + "."


def build_scene_input_sample_from_types(
    scene_info: Union[Scene, Dict],
    placement_task: Union[PlacementRequest, Dict],
    existing_objects_override: Optional[Sequence[Union[SceneObject, Dict]]] = None,
) -> Dict:
    """Build the legacy prompt sample from canonical scene/request models."""
    scene = ensure_scene(scene_info)
    request = ensure_placement_request(placement_task)
    existing_objects = (
        scene.objects if existing_objects_override is None else existing_objects_override
    )

    new_object_data: Dict = {
        "id": request.id,
        "short_desc": request.short_desc,
        "size": list(request.size),
        "category": request.category,
    }
    sample: Dict = {
        "scene_description": scene.scene_description,
        "boundary": scene.boundary,
        "scene_height": scene.scene_height,
        "existing_objects": scene_object_list_to_dicts(existing_objects),
        "new_object": new_object_data,
        "instruction": request.instruction,
    }
    if "expected_relations" in request.extra:
        sample["expected_relations"] = request.extra["expected_relations"]
    return sample


def format_scene_input_from_types(
    scene_info: Union[Scene, Dict],
    placement_task: Union[PlacementRequest, Dict],
    *,
    existing_objects_override: Optional[Sequence[Union[SceneObject, Dict]]] = None,
    use_pointcloud: Optional[bool] = None,
    use_separate_pointcloud: bool = False,
    internal_spaces_map: Optional[Dict[str, List]] = None,
    include_object_descriptions: bool = False,
    only_pointcloud: bool = False,
) -> str:
    """Format a prompt from canonical scene/request models."""
    if use_pointcloud is not None:
        use_separate_pointcloud = bool(use_pointcloud)
    sample = build_scene_input_sample_from_types(
        scene_info,
        placement_task,
        existing_objects_override=existing_objects_override,
    )
    return format_scene_input(
        sample,
        use_separate_pointcloud=use_separate_pointcloud,
        internal_spaces_map=internal_spaces_map,
        include_object_descriptions=include_object_descriptions,
        only_pointcloud=only_pointcloud,
    )


def format_scene_input(
    sample: Dict,
    use_separate_pointcloud: bool = False,
    internal_spaces_map: Optional[Dict[str, List]] = None,
    include_object_descriptions: bool = False,
    only_pointcloud: bool = False,
) -> str:
    """
    将自回归数据样本格式化为输入字符串（用于训练数据生成）
    
    Args:
        sample: 自回归数据样本
        use_separate_pointcloud: 是否使用分离点云占位符
            - floor_pointcloud 放在 boundary 后
            - existing_objects 中每个物体包含按顺序编号的占位符
        include_object_descriptions: 是否写入 existing_objects/new_object 的 short_desc
        only_pointcloud: 是否剔除物体的 center, size, rotation 信息，仅提供点云
    
    Returns:
        格式化的输入字符串
    """
    if only_pointcloud:
        use_separate_pointcloud = True

    scene_height = round(sample["scene_height"], 2)
    instruction_text = build_model_instruction(
        sample,
        include_object_descriptions=include_object_descriptions,
    )

    if use_separate_pointcloud:
        # 分离点云版本：
        # 1) boundary 后紧跟 floor_pointcloud
        # 2) existing_objects 每个物体带一个按顺序编号的点云占位符
        input_data = {
            "boundary": _round_boundary(sample["boundary"]),
            "floor_pointcloud": "<floor_pointcloud>",
            "scene_height": scene_height,
            "existing_objects": [
                _build_existing_object_input(
                    obj,
                    internal_spaces_map=internal_spaces_map,
                    include_object_descriptions=include_object_descriptions,
                    include_object_pointcloud=True,
                    only_pointcloud=only_pointcloud,
                )
                for obj in sample["existing_objects"]
            ],
            "new_object": _build_new_object_input(
                sample["new_object"],
                include_object_descriptions=include_object_descriptions,
                pointcloud_field="new_object_pointcloud",
                pointcloud_placeholder="<new_object_pointcloud>",
                only_pointcloud=only_pointcloud,
            ),
            "instruction": instruction_text,
        }
    else:
        # 纯文本版本：不包含 point_cloud 字段
        input_data = {
            "boundary": _round_boundary(sample["boundary"]),
            "scene_height": scene_height,
            "existing_objects": [
                _build_existing_object_input(
                    obj,
                    internal_spaces_map=internal_spaces_map,
                    include_object_descriptions=include_object_descriptions,
                    only_pointcloud=only_pointcloud,
                )
                for obj in sample["existing_objects"]
            ],
            "new_object": _build_new_object_input(
                sample["new_object"],
                include_object_descriptions=include_object_descriptions,
                only_pointcloud=only_pointcloud,
            ),
            "instruction": instruction_text,
        }

    # 格式化为字符串
    user_prompt = json.dumps(input_data, indent=None, ensure_ascii=False)

    return user_prompt


def format_output(sample: Dict) -> str:
    """
    将目标 pose 格式化为输出字符串
    
    Args:
        sample: 自回归数据样本
    
    Returns:
        格式化的输出字符串：[x, y, z, roll, pitch, yaw]
    """
    center = _round_list(sample["target_pose"]["center"])  # 保留两位小数
    rotation = _normalize_rotation(sample["target_pose"]["rotation"])  # 标准化到 [0, 360)
    
    # 简化格式：[x, y, z, roll, pitch, yaw]
    output_data = center + rotation
    
    return json.dumps(output_data, ensure_ascii=False)

