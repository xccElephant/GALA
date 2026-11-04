"""共享的语义关系解析与重叠豁免判定（纯字典逻辑，无几何 / Blender 依赖）。

推理期（``agent.placement_validator``）、后处理（``scripts/physics_repair``）两条链路
原本各自实现一套 ``on / within / under`` 关系解析，导致行为不一致——尤其是后处理对
语义合理的重叠（物在容器内 / 物在支撑面上 / 物压在家具下）做硬豁免，而推理期碰撞检测
没有任何豁免，把这些物理合理的摆放误报为穿模。本模块把关系解析下沉成单一来源。

关系数据约定：``obj["expected_relations"]["relative_relations"]`` 是
``[{"type": "on"|"within"|"under"|..., "target_id": int}, ...]``。
"""

from __future__ import annotations

from typing import Any, Mapping

SUPPORT_RELATION_TYPES = {"on"}
CONTAINMENT_RELATION_TYPES = {"within"}
ATTACHMENT_RELATION_TYPES = SUPPORT_RELATION_TYPES | CONTAINMENT_RELATION_TYPES
UNDER_RELATION_TYPES = {"under"}
# 三类「footprint 本就应重叠」的关系：物在支撑面上 / 物在容器内 / 物压在家具下。
OVERLAP_EXEMPT_RELATION_TYPES = (
    SUPPORT_RELATION_TYPES | CONTAINMENT_RELATION_TYPES | UNDER_RELATION_TYPES
)


def _relative_relations(obj_data: Mapping[str, Any]) -> list:
    relations = obj_data.get("expected_relations")
    if not isinstance(relations, Mapping):
        return []
    relative = relations.get("relative_relations")
    if not isinstance(relative, list):
        return []
    return relative


def has_relation_to_target(
    obj_data: Mapping[str, Any], target_id: int, relation_types: set[str]
) -> bool:
    """``obj_data`` 是否对 ``target_id`` 声明了 ``relation_types`` 中任一关系。"""
    for relation in _relative_relations(obj_data):
        if not isinstance(relation, Mapping):
            continue
        if int(relation.get("target_id", -1)) != int(target_id):
            continue
        if str(relation.get("type", "")).lower() in relation_types:
            return True
    return False


def relation_target_id(
    obj_data: Mapping[str, Any], relation_types: set[str]
) -> int | None:
    """返回首个匹配 ``relation_types`` 的关系目标 id，无则 None。"""
    for relation in _relative_relations(obj_data):
        if not isinstance(relation, Mapping):
            continue
        relation_type = str(relation.get("type", "")).lower()
        if relation_type not in relation_types or "target_id" not in relation:
            continue
        return int(relation["target_id"])
    return None


def all_relation_target_ids(
    obj_data: Mapping[str, Any], relation_types: set[str]
) -> list[int]:
    """返回所有匹配 ``relation_types`` 的目标 id（按出现顺序去重）。"""
    result: list[int] = []
    for relation in _relative_relations(obj_data):
        if not isinstance(relation, Mapping):
            continue
        if str(relation.get("type", "")).lower() not in relation_types:
            continue
        if "target_id" not in relation:
            continue
        target_id = int(relation["target_id"])
        if target_id not in result:
            result.append(target_id)
    return result


def is_overlap_exempt_by_relation(
    obj_a: Mapping[str, Any], obj_b: Mapping[str, Any]
) -> bool:
    """两物体是否因语义关系（on / within / under）本就该 footprint 重叠。

    纯关系判定（不看几何），任一方对另一方声明了上述关系即豁免。用于碰撞检测跳过
    这些语义合理的重叠，避免误报穿模。
    """
    a_id = int(obj_a["id"])
    b_id = int(obj_b["id"])
    return has_relation_to_target(
        obj_a, b_id, OVERLAP_EXEMPT_RELATION_TYPES
    ) or has_relation_to_target(obj_b, a_id, OVERLAP_EXEMPT_RELATION_TYPES)
