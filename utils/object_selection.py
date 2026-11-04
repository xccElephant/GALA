"""
参考物体筛选工具

用于训练数据生成和推理管线中，从已有物体中筛选参考物体子集。
"""
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from config.constants import MAX_REF_OBJECTS
from utils.spatial_utils import get_object_footprint_area_xy
from utils.scene_types import SceneObject, scene_object_list_to_dicts


def select_reference_objects(
    existing_objects: Sequence[Union[SceneObject, Dict[str, Any]]],
    referenced_ids: set,
    object_distances: Dict[int, float],
    max_total: int = MAX_REF_OBJECTS,
) -> List[Union[SceneObject, Dict[str, Any]]]:
    """
    从已有物体中筛选出对摆放新物体有参考价值的物体（训练用，基于距离）
    
    筛选准则（按优先级）：
    1. 必须保留：指令中引用的物体（referenced_ids）
    2. 应该保留：距离新物体最近的物体（按边缘距离排序）
    3. 总数不超过 max_total
    
    Args:
        existing_objects: 所有已有物体列表
        referenced_ids: 指令中引用的物体 ID 集合
        object_distances: {obj_id: edge_distance} 所有物体与新物体的边缘距离
        max_total: 参考物体总数上限
    
    Returns:
        筛选后的物体列表（保持原始摆放顺序）
    """
    original_objects = list(existing_objects)
    existing_object_dicts = scene_object_list_to_dicts(original_objects)
    original_by_id = {obj_dict["id"]: original for obj_dict, original in zip(existing_object_dicts, original_objects)}

    if len(existing_object_dicts) <= max_total:
        return original_objects  # 不需要筛选
    
    # 分为两组：必须保留的（指令引用）和候选的
    must_keep = []
    candidates = []
    
    for obj in existing_object_dicts:
        obj_id = obj["id"]
        if obj_id in referenced_ids:
            must_keep.append(obj)
        else:
            dist = object_distances.get(obj_id, float("inf"))
            candidates.append((obj, dist))
    
    # 候选物体按距离排序（近的优先）
    candidates.sort(key=lambda x: x[1])
    
    # 计算还能添加多少个候选物体
    remaining_slots = max(0, max_total - len(must_keep))
    nearest_count = min(remaining_slots, len(candidates))
    
    # 合并：必须保留的 + 最近的 K 个
    selected = must_keep + [obj for obj, _ in candidates[:nearest_count]]
    
    # 按原始 ID 排序，保持摆放顺序
    selected.sort(key=lambda x: x["id"])
    
    return [original_by_id[obj["id"]] for obj in selected]


def select_reference_objects_by_importance(
    existing_objects: Sequence[Union[SceneObject, Dict[str, Any]]],
    max_total: int = MAX_REF_OBJECTS,
    must_keep_ids: Optional[Set[int]] = None,
) -> List[Union[SceneObject, Dict[str, Any]]]:
    """
    从已有物体中按重要性筛选参考物体（推理用，不依赖距离信息）
    
    推理时不知道新物体的目标位置，因此无法按距离筛选。
    使用替代策略：
    1. 强制保留语义依赖的物体（must_keep_ids，来自 depends_on）
    2. 按占地面积排序，保留最大的 K 个锚定物体（如床、桌子、沙发）
    3. 保留最近添加的 M 个物体（提供最新上下文）
    4. 总数不超过 max_total
    
    Args:
        existing_objects: 所有已有物体列表（按摆放顺序排列）
        max_total: 参考物体总数上限
        must_keep_ids: 必须保留的物体 ID 集合（来自语义依赖 depends_on）
    
    Returns:
        筛选后的物体列表（保持原始 ID 顺序）
    """
    original_objects = list(existing_objects)
    existing_object_dicts = scene_object_list_to_dicts(original_objects)
    original_by_id = {obj_dict["id"]: original for obj_dict, original in zip(existing_object_dicts, original_objects)}

    if len(existing_object_dicts) <= max_total:
        return original_objects  # 不需要筛选
    
    if must_keep_ids is None:
        must_keep_ids = set()
    else:
        must_keep_ids = set(must_keep_ids)  # 避免修改调用方的集合
    
    # 保留 sub-object 时，强制保留其 parent_id 对应的物体
    obj_by_id = {obj["id"]: obj for obj in existing_object_dicts}
    expanded = True
    while expanded:
        expanded = False
        for obj in existing_object_dicts:
            if obj["id"] in must_keep_ids:
                pid = obj.get("parent_id")
                if pid is not None and pid not in must_keep_ids and pid in obj_by_id:
                    must_keep_ids.add(pid)
                    expanded = True
    
    # 分离必须保留的物体和候选物体
    must_keep = []
    candidates = []
    for obj in existing_object_dicts:
        if obj["id"] in must_keep_ids:
            must_keep.append(obj)
        else:
            candidates.append(obj)
    
    remaining_slots = max(0, max_total - len(must_keep))
    if remaining_slots == 0 or not candidates:
        must_keep.sort(key=lambda x: x["id"])
        return [original_by_id[obj["id"]] for obj in must_keep[:max_total]]
    
    # 对候选物体计算占地面积
    candidate_areas: List[Tuple[Dict[str, Any], float]] = []
    for obj in candidates:
        try:
            area = get_object_footprint_area_xy(obj)
        except Exception:
            size = obj.get("size") or [0, 0, 0]
            area = abs(size[0] * size[1])
        candidate_areas.append((obj, area))
    
    # 在剩余槽位中分配：60% 给大物体，40% 给最近添加的
    k_large = max(1, int(remaining_slots * 0.6))
    m_recent = max(1, remaining_slots - k_large)
    
    # 按面积降序排列，取前 K 个
    sorted_by_area = sorted(candidate_areas, key=lambda x: -x[1])
    large_objects = {id(obj): obj for obj, _ in sorted_by_area[:k_large]}
    
    # 最近添加的 M 个（candidates 保持原始顺序，取末尾 M 个）
    recent_objects = {id(obj): obj for obj in candidates[-m_recent:]}
    
    # 合并去重
    selected_candidate_ids = set(large_objects.keys()) | set(recent_objects.keys())
    
    if len(selected_candidate_ids) > remaining_slots:
        only_recent = [obj for oid, obj in recent_objects.items() if oid not in large_objects]
        allowed_recent = max(0, remaining_slots - len(large_objects))
        only_recent = only_recent[-allowed_recent:] if allowed_recent > 0 else []
        selected_candidates = list(large_objects.values()) + only_recent
    else:
        selected_candidates = [obj for obj in candidates if id(obj) in selected_candidate_ids]
    
    selected = must_keep + selected_candidates
    selected.sort(key=lambda x: x["id"])
    
    return [original_by_id[obj["id"]] for obj in selected]
