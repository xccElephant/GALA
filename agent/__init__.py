"""
Agent 系统

双层架构：
- 上层: Agent 控制层（由闭源大模型驱动，负责场景规划、指令生成、结果评估）
- 下层: 执行层（由训练好的小模型负责位姿推理）

主要模块：
- ScenePlannerAgent: 场景规划 Agent, 负责解析用户需求、生成物体清单、生成摆放指令
- AssetRetriever: 资产检索器，根据语义和尺寸从资产库检索匹配的资产
- PlacementValidator: 摆放验证器，验证位姿是否合理（碰撞检测、边界检测等）
- SceneStateManager: 场景状态管理器，管理场景状态和历史（支持回滚）
- SceneGenerationPipeline: 端到端场景生成流程编排
"""

from agent.asset_retriever import AssetRetriever, AssetIndex
from agent.scene_planner import ScenePlannerAgent
from agent.placement_validator import PlacementValidator
from agent.scene_state import SceneStateManager
from agent.pipeline import SceneGenerationPipeline

__all__ = [
    "AssetRetriever",
    "AssetIndex",
    "ScenePlannerAgent",
    "PlacementValidator",
    "SceneStateManager",
    "SceneGenerationPipeline",
]

