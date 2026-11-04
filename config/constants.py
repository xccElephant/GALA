from typing import Dict

# 参考物体数量上限（与训练保持一致，0 表示不筛选）
MAX_REF_OBJECTS: int = 8

# 同一 on/within/under 父物体下强制保留的 sibling 参考物体数量。
# 这些 sibling 会额外加入参考物体，允许样本超过 MAX_REF_OBJECTS。
MAX_SIBLING_REFS: int = 6

# 碰撞检测阈值
COLLISION_RATIO_THRESHOLD: float = 0.2
INTERSECTION_VOLUME_THRESHOLD: float = 0.001

# 推理默认参数
DEFAULT_SCENE_HEIGHT: float = 2.6  # 在未提供 scene_height 时作为默认值
MAX_SCENE_HEIGHT_METERS: float = 5.0  # 房间高度的上限
DEFAULT_TEMPERATURE: float = 0.6
MAX_NEW_TOKENS: int = 4096
FLOOR_SAMPLE_POINTS: int = 2048

# 空间关系 key → 自然语言文本映射（训练数据生成 / 推理指令构建共用）
RELATIVE_RELATION_TEXT: Dict[str, str] = {
    "left": "to the left of",
    "right": "to the right of",
    "in_front": "in front of",
    "behind": "behind",
    "next_to": "next to",
    "face_to": "facing",
    "on": "on",
    "under": "under",
    "within": "within",
}