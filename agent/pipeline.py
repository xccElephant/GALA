"""
场景生成 Pipeline

端到端的场景生成流程编排，整合：
1. Agent 规划层（ScenePlannerAgent）
2. 资产检索（AssetRetriever）
3. 位姿推理（SceneLayoutInference）
4. 结果验证（PlacementValidator）
5. 状态管理（SceneStateManager）

支持两种模式：
- 生成模式：从用户需求生成完整场景
- 编辑模式：在现有场景上添加/修改物体
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt

from agent.asset_retriever import AssetRetriever
from agent.placement_validator import PlacementValidator
from agent.scene_planner import ScenePlannerAgent, build_instruction_from_relations
from agent.scene_renderer import SceneRenderer
from agent.scene_state import PlacementAttempt, SceneStateManager
from config.constants import COLLISION_RATIO_THRESHOLD, DEFAULT_SCENE_HEIGHT, MAX_REF_OBJECTS
from config.paths import ProjectPaths
from inference.scene_layout_inference import SceneLayoutInference
from utils.logger import Logger
from utils.object_selection import select_reference_objects_by_importance
from utils.scene_types import (
    PlacementRequest,
    Pose,
    SceneObject,
    scene_object_list_to_dicts,
)

# ==================== 数据类定义 ====================


@dataclass
class ObjectPlan:
    """物体规划信息（来自 Agent 规划或用户编辑请求）"""

    description: str  # 详细描述（用于资产检索）
    estimated_size: List[float]  # 估计尺寸
    # 规划/编辑解析阶段的思考过程（编辑模式下来自 edit parser 的原生 think）
    thinking: str = ""
    # 预设空间关系（来自编辑模式解析结果）；None 表示由 LLM 决定
    expected_relations: Optional[Dict] = None
    # 父物体 ID（sub-object 的支撑物体），None 表示主物体
    parent_id: Optional[int] = None
    # 语义依赖的物体（规划阶段的 ID 列表），筛选参考物体时必须保留这些物体
    depends_on: List[int] = field(default_factory=list)


@dataclass
class RetrievedAsset:
    """检索到的资产信息"""

    jid: str  # 资产 ID
    short_desc: str  # 简短描述
    size: List[float]  # 实际尺寸
    category: str  # 物体类别
    score: float = 0.0  # 匹配分数


@dataclass
class InstructionResult:
    """指令生成结果"""

    instruction: str
    thinking: str = ""
    expected_relations: Dict = field(default_factory=dict)
    parent_id: Optional[int] = None


@dataclass
class PlacementResult:
    """最终摆放结果"""

    success: bool
    partial_success: bool = False
    object_id: int = 0
    asset: Optional[RetrievedAsset] = None
    pose: Optional[Pose] = None
    instruction: str = ""
    instruction_thinking: str = ""
    validation: Optional[Dict] = None
    compliance: Optional[Dict] = None
    parent_id: Optional[int] = None
    attempts_count: int = 0
    error: str = ""


@dataclass
class PipelineTimingStats:
    """Pipeline 主要模块耗时统计（秒）。"""

    scene_planner: float = 0.0
    scene_planner_calls: int = 0
    instruction_generator: float = 0.0
    instruction_generator_calls: int = 0
    gala_inference: float = 0.0
    gala_inference_calls: int = 0

    def reset(self) -> None:
        self.scene_planner = 0.0
        self.scene_planner_calls = 0
        self.instruction_generator = 0.0
        self.instruction_generator_calls = 0
        self.gala_inference = 0.0
        self.gala_inference_calls = 0

    def record(self, module: str, duration: float) -> None:
        if module == "scene_planner":
            self.scene_planner += duration
            self.scene_planner_calls += 1
        elif module == "instruction_generator":
            self.instruction_generator += duration
            self.instruction_generator_calls += 1
        elif module == "gala_inference":
            self.gala_inference += duration
            self.gala_inference_calls += 1
        else:
            raise ValueError(f"Unknown timing module: {module}")

    @property
    def tracked_total(self) -> float:
        return self.scene_planner + self.instruction_generator + self.gala_inference

    def to_dict(self, elapsed_time: Optional[float] = None) -> Dict[str, Any]:
        def _module_stats(total: float, calls: int) -> Dict[str, Any]:
            return {
                "total_s": round(total, 3),
                "calls": calls,
                "avg_s": round(total / calls, 3) if calls > 0 else 0.0,
            }

        result = {
            "scene_planner": _module_stats(
                self.scene_planner, self.scene_planner_calls
            ),
            "instruction_generator": _module_stats(
                self.instruction_generator, self.instruction_generator_calls
            ),
            "gala_inference": _module_stats(
                self.gala_inference, self.gala_inference_calls
            ),
            "tracked_total_s": round(self.tracked_total, 3),
        }
        if elapsed_time is not None:
            result["elapsed_time_s"] = round(elapsed_time, 3)
            result["other_s"] = round(
                max(elapsed_time - self.tracked_total, 0.0), 3
            )
        return result


class AssetRetrievalError(RuntimeError):
    """资产检索失败"""


# placement_validator 关闭时使用的占位结果（跳过碰撞/边界/高度/悬空检测）
_SKIPPED_PHYSICAL_VALIDATION: Dict[str, Any] = {
    "valid": True,
    "errors": [],
    "warnings": [],
    "details": {},
}


class SceneGenerationPipeline:
    """
    场景生成 Pipeline
    """

    def __init__(
        self,
        # 资产检索配置
        asset_index_path: Optional[str] = None,
        # 推理模型配置
        inference_model_path: Optional[str] = None,
        asset_pointcloud_dir: Optional[str] = None,
        use_pointcloud: bool = True,
        internal_spaces_dir: Optional[str] = None,
        # 视觉反馈配置
        use_visual_feedback: bool = False,
        save_renders: bool = True,
        # 验证配置
        collision_ratio_threshold: float = COLLISION_RATIO_THRESHOLD,
        use_placement_validator: bool = False,
        # 其他配置
        output_dir: str = "output/agent",
        max_retries_per_object: int = 3,
        max_ref_objects: int = MAX_REF_OBJECTS,
        infer_temp: float = 0.6,
        include_object_descriptions: bool = False,
        verbose: bool = True,
    ):
        """
        初始化 Pipeline

        Args:
            asset_index_path: 资产索引文件路径
            inference_model_path: 推理模型路径
            asset_pointcloud_dir: 资产点云目录
            use_pointcloud: 是否使用点云
            internal_spaces_dir: 收纳空间数据目录路径. None 表示不附加 internal_spaces
            use_visual_feedback: 是否启用视觉反馈（渲染场景图给 Agent 参考）
            save_renders: 是否保存渲染图
            collision_ratio_threshold: 碰撞比例阈值
            use_placement_validator: 是否启用摆放物理验证（碰撞/边界/高度/悬空）
            output_dir: 输出目录
            max_retries_per_object: 每个物体的最大重试次数
            max_ref_objects: 参考物体最大数量（与训练一致, 0=不筛选）
            infer_temp: 推理模型采样温度（默认 0.6, 设为 0.0 为贪婪解码，此时重试无意义）
            include_object_descriptions: 是否在 instruction / 推理 prompt 中暴露物体描述
            verbose: 是否打印详细日志
        """
        self.verbose = verbose
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries_per_object
        self.max_ref_objects = max_ref_objects
        self.infer_temp = infer_temp
        self.include_object_descriptions = include_object_descriptions
        self.use_visual_feedback = use_visual_feedback
        self.save_renders = save_renders
        self.use_placement_validator = use_placement_validator

        # 初始化日志
        self.log = Logger("Pipeline", verbose=verbose)

        # 初始化 Agent
        self.log.info("Initializing ScenePlannerAgent...")
        self.agent = ScenePlannerAgent(
            verbose=verbose,
            include_object_descriptions=include_object_descriptions,
        )

        # 初始化资产检索器
        self.log.info("Initializing AssetRetriever...")
        self.retriever = AssetRetriever(index_path=asset_index_path)

        # 初始化验证器
        self._asset_pointcloud_dir = asset_pointcloud_dir or str(
            ProjectPaths.IMAGINARIUM_ASSETS_DIR
        )
        self.validator = PlacementValidator(
            collision_ratio_threshold=collision_ratio_threshold,
            asset_pointcloud_dir=self._asset_pointcloud_dir,
        )
        if not use_placement_validator:
            self.log.info("PlacementValidator disabled (物理验证已关闭)")

        # 初始化场景渲染器（用于视觉反馈）
        self.renderer: Optional[SceneRenderer] = None
        if use_visual_feedback:
            render_output_dir = self.output_dir / "renders" if save_renders else None
            self.renderer = SceneRenderer(
                output_dir=str(render_output_dir) if render_output_dir else None
            )
            self.log.info("Visual feedback enabled")

        # 推理模型（延迟加载）
        self._inference_model_path = inference_model_path

        self._use_pointcloud = use_pointcloud
        self._internal_spaces_dir = internal_spaces_dir
        self._inferencer = None

        # 状态管理器（每次生成时创建新实例）
        self.state: Optional[SceneStateManager] = None
        self._timing = PipelineTimingStats()

    def _reset_timing(self) -> None:
        self._timing.reset()

    def _log_timing_summary(self, elapsed_time: float) -> None:
        timing = self._timing.to_dict(elapsed_time)
        self.log.newline()
        self.log.info("Timing breakdown:")
        for name, label in (
            ("scene_planner", "Scene Planner"),
            ("instruction_generator", "Instruction Generator"),
            ("gala_inference", "GALA Inference"),
        ):
            stats = timing[name]
            self.log.indent(
                f"{label}: {stats['total_s']:.2f}s "
                f"({stats['calls']} calls, avg {stats['avg_s']:.2f}s)"
            )
        self.log.indent(
            f"Tracked total: {timing['tracked_total_s']:.2f}s / "
            f"{timing['elapsed_time_s']:.2f}s elapsed "
            f"(other: {timing['other_s']:.2f}s)"
        )

    def _get_inferencer(self, session_timestamp: str = None):
        """懒加载推理模型"""
        if self._inferencer is None:
            if self._inference_model_path is None:
                raise ValueError("Inference model path not specified")

            self._inferencer = SceneLayoutInference(
                model_path=self._inference_model_path,
                asset_pointcloud_dir=self._asset_pointcloud_dir,
                output_dir=str(self.output_dir / "pointclouds"),
                use_pointcloud=self._use_pointcloud,
                session_timestamp=session_timestamp,
                internal_spaces_dir=self._internal_spaces_dir,
                include_object_descriptions=self.include_object_descriptions,
            )
        return self._inferencer

    @staticmethod
    def _scene_objects_to_dicts(objects: List[SceneObject]) -> List[Dict[str, Any]]:
        return scene_object_list_to_dicts(objects)

    @staticmethod
    def _build_scene_object(
        object_id: int,
        asset: RetrievedAsset,
        *,
        parent_id: Optional[int] = None,
    ) -> SceneObject:
        return SceneObject(
            id=object_id,
            short_desc=asset.short_desc,
            size=asset.size,
            jid=asset.jid,
            category=asset.category,
            parent_id=parent_id,
        )

    @staticmethod
    def _build_placement_request(
        scene_object: SceneObject,
        instruction: str,
        *,
        request_id: Optional[int] = None,
        expected_relations: Optional[Dict[str, Any]] = None,
    ) -> PlacementRequest:
        extra: Dict[str, Any] = {}
        if expected_relations is not None:
            extra["expected_relations"] = expected_relations
        return PlacementRequest(
            id=scene_object.id if request_id is None else request_id,
            short_desc=scene_object.short_desc,
            size=scene_object.size,
            jid=scene_object.jid,
            category=scene_object.category,
            instruction=instruction,
            extra=extra,
        )

    @staticmethod
    def _build_agent_object_info(
        scene_object: SceneObject,
        plan: ObjectPlan,
        *,
        object_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        return {
            "id": scene_object.id if object_id is None else object_id,
            "short_desc": scene_object.short_desc,
            "description": plan.description,
            "size": scene_object.size,
            "category": scene_object.category,
        }

    @staticmethod
    def _with_pose(scene_object: SceneObject, pose: Pose) -> SceneObject:
        return scene_object.copy_with_pose(pose)

    def _retrieve_asset(self, plan: ObjectPlan) -> RetrievedAsset:
        """
        资产检索

        Args:
            plan: 物体规划信息

        Returns:
            检索到的资产信息
        """
        # 确保 size 是两位小数
        estimated_size = plan.estimated_size
        if isinstance(estimated_size, list) and len(estimated_size) == 3:
            estimated_size = [round(float(s), 2) for s in estimated_size]
        else:
            estimated_size = [0.50, 0.50, 0.50]

        self.log.indent(f"Retrieving asset for: {plan.description}")
        candidates = self.retriever.retrieve(
            description=plan.description,
            size_constraint=estimated_size,
            top_k=5,
        )

        if not candidates:
            raise AssetRetrievalError(
                f"No matching asset found for description='{plan.description}', size={estimated_size}"
            )

        selected = candidates[0]
        asset = RetrievedAsset(
            jid=selected.get("jid"),
            short_desc=selected.get("short_desc"),
            size=selected.get("size"),
            category=selected.get("category"),
            score=selected.get("score"),
        )
        self.log.indent(
            f"Selected: {asset.short_desc} (jid: {asset.jid}, score: {asset.score:.2f})"
        )
        return asset

    def _render_scene_for_feedback(
        self,
        scene_object: SceneObject,
        plan: ObjectPlan,
        session_timestamp: Optional[str] = None,
        filtered_objects: Optional[List[SceneObject]] = None,
    ) -> Optional[str]:
        """
        渲染当前场景用于视觉反馈

        Args:
            scene_object: 待摆放的物体
            plan: 物体规划信息
            session_timestamp: 会话时间戳
            filtered_objects: 筛选后的物体列表，None 时使用全量物体

        Returns:
            base64 编码的场景图像，如果未启用视觉反馈则返回 None
        """
        if not self.use_visual_feedback or not self.renderer:
            return None

        self.log.indent("Rendering scene for visual feedback...")

        existing_objects = (
            filtered_objects
            if filtered_objects is not None
            else self.state.scene_model.objects
        )

        obj_info = self._build_agent_object_info(scene_object, plan)

        save_path = None
        if self.save_renders and session_timestamp:
            render_dir = self.output_dir / "renders"
            render_dir.mkdir(parents=True, exist_ok=True)
            save_path = str(render_dir / f"step_{scene_object.id:02d}_before.png")

        fig, scene_image_base64 = self.renderer.render_scene(
            boundary=self.state.boundary,
            existing_objects=existing_objects,
            new_object_info=obj_info,
            save_path=save_path,
        )
        plt.close(fig)
        self.log.debug(f"Scene rendered (image size: {len(scene_image_base64)} chars)")
        return scene_image_base64

    def _filter_reference_objects(
        self,
        must_keep_ids: Optional[set] = None,
    ) -> Optional[List[SceneObject]]:
        """
        筛选参考物体（保持原始 ID 不变）

        推理时不知道新物体的目标位置，因此使用基于重要性的筛选策略
        （保留最大的锚定家具 + 最近添加的物体），同时强制保留
        语义依赖的物体（must_keep_ids）。

        Args:
            must_keep_ids: 必须保留的物体运行时 ID 集合（来自 depends_on）

        Returns:
            筛选后的物体列表（SceneObject），如果不需要筛选则返回 None
        """
        existing_objects = self.state.scene_model.objects

        if self.max_ref_objects <= 0 or len(existing_objects) <= self.max_ref_objects:
            return None

        filtered_dicts = select_reference_objects_by_importance(
            self._scene_objects_to_dicts(existing_objects),
            max_total=self.max_ref_objects,
            must_keep_ids=must_keep_ids,
        )

        self.log.debug(
            f"Filtered {len(existing_objects)} objects → {len(filtered_dicts)} "
            f"(max_ref={self.max_ref_objects})"
        )

        return [SceneObject.from_dict(obj) for obj in filtered_dicts]

    def _generate_placement_instruction(
        self,
        scene_object: SceneObject,
        plan: ObjectPlan,
        scene_image_base64: Optional[str] = None,
        filtered_objects: Optional[List[SceneObject]] = None,
    ) -> InstructionResult:
        """
        为物体生成摆放指令（在资产检索后调用）

        Args:
            scene_object: 待摆放的物体
            plan: 物体规划信息
            scene_image_base64: 场景渲染图
            filtered_objects: 筛选后的物体列表，None 时使用全量物体

        Returns:
            指令生成结果
        """
        existing_objects = (
            filtered_objects
            if filtered_objects is not None
            else self.state.scene_model.objects
        )

        # 编辑模式：使用预设关系直接拼接指令，跳过 LLM
        if plan.expected_relations is not None:
            return self._build_instruction_from_preset_relations(
                plan, scene_object, existing_objects
            )

        # 生成模式：调用 LLM 生成空间关系，再由代码拼接指令
        self.log.indent("Generating placement instruction...")
        instruction = ""
        thinking = ""
        expected_relations = {}
        parent_id = scene_object.parent_id
        max_retries = 3

        for attempt in range(max_retries):
            t0 = time.perf_counter()
            result = self.agent.generate_instruction(
                scene_object=scene_object,
                scene=self.state.scene_model,
                existing_objects=existing_objects,
                object_description=plan.description,
                scene_image_base64=scene_image_base64,
            )
            self._timing.record("instruction_generator", time.perf_counter() - t0)

            instruction = result.get("instruction")
            thinking = result.get("thinking")
            expected_relations = result.get("expected_relations")
            if "parent_id" in result:
                parent_id = result["parent_id"]

            if instruction and instruction.strip():
                break
            self.log.warning(
                f"Empty instruction returned, retrying ({attempt + 1}/{max_retries})..."
            )

        if not instruction or not instruction.strip():
            raise RuntimeError(
                f"All {max_retries} instruction generation attempts returned empty "
                f"for object '{scene_object.short_desc}' (id={scene_object.id})"
            )

        scene_object.parent_id = parent_id
        self.log.indent(f"Instruction: {instruction}")
        if expected_relations:
            self.log.debug(f"Expected relations: {expected_relations}")

        return InstructionResult(
            instruction=instruction,
            thinking=thinking,
            expected_relations=expected_relations,
            parent_id=parent_id,
        )

    def _build_instruction_from_preset_relations(
        self,
        plan: ObjectPlan,
        scene_object: SceneObject,
        existing_objects: List[SceneObject],
    ) -> InstructionResult:
        """
        编辑模式专用：直接使用预设的 expected_relations 拼接指令，不调用 LLM。

        expected_relations 中引用的 target_id 应已通过 depends_on → must_keep_ids
        在筛选阶段保留，若仍出现悬空引用则说明上游数据不一致。
        """
        self.log.indent("Building instruction from edit plan relations (no LLM)...")
        relations = dict(plan.expected_relations or {})  # shallow copy

        # 校验：所有 target_id 必须在 existing_objects 中
        existing_ids = {obj.id for obj in existing_objects}
        for rel in relations.get("relative_relations", []):
            target_id = rel.get("target_id")
            if target_id not in existing_ids:
                raise ValueError(
                    f"Relation target_id={target_id} not found in existing_objects "
                    f"(ids={sorted(existing_ids)}). "
                    f"The target should have been preserved via depends_on / must_keep_ids."
                )

        normalized = self.agent.normalize_relations(relations)
        instruction = build_instruction_from_relations(
            scene_object.short_desc,
            normalized,
            existing_objects,
            parent_id=scene_object.parent_id,
            include_object_descriptions=self.include_object_descriptions,
            category=scene_object.category,
        )

        self.log.indent(f"Instruction: {instruction}")
        if normalized:
            self.log.debug(f"Expected relations: {normalized}")

        return InstructionResult(
            instruction=instruction,
            thinking=plan.thinking,
            expected_relations=normalized,
            parent_id=scene_object.parent_id,
        )

    def _regenerate_instruction_with_feedback(
        self,
        scene_object: SceneObject,
        plan: ObjectPlan,
        previous_instruction: str,
        validation_result: Dict[str, Any],
        compliance_result: Optional[Dict[str, Any]] = None,
        scene_image_base64: Optional[str] = None,
        filtered_objects: Optional[List[SceneObject]] = None,
    ) -> InstructionResult:
        """
        基于验证失败反馈，重新生成摆放指令

        Args:
            scene_object: 待摆放的物体
            plan: 物体规划信息
            previous_instruction: 之前失败的指令
            validation_result: 物理验证结果（碰撞、边界等）
            compliance_result: 指令符合度结果（空间关系是否满足）
            scene_image_base64: 场景渲染图
            filtered_objects: 筛选后的物体列表，None 时使用全量物体

        Returns:
            新的指令生成结果
        """
        existing_objects = (
            filtered_objects
            if filtered_objects is not None
            else self.state.scene_model.objects
        )

        t0 = time.perf_counter()
        result = self.agent.regenerate_instruction_with_feedback(
            scene_object=scene_object,
            scene=self.state.scene_model,
            existing_objects=existing_objects,
            previous_instruction=previous_instruction,
            validation_result=validation_result,
            compliance_result=compliance_result,
            object_description=plan.description,
            scene_image_base64=scene_image_base64,
        )
        self._timing.record("instruction_generator", time.perf_counter() - t0)

        instruction = result.get("instruction")
        thinking = result.get("thinking")
        expected_relations = result.get("expected_relations", {})
        parent_id = scene_object.parent_id
        if "parent_id" in result:
            parent_id = result["parent_id"]
        scene_object.parent_id = parent_id

        if instruction:
            self.log.indent(f"New instruction: {instruction}")
            if expected_relations:
                self.log.debug(f"New expected relations: {expected_relations}")

        return InstructionResult(
            instruction=instruction,
            thinking=thinking,
            expected_relations=expected_relations,
            parent_id=parent_id,
        )

    def _place_object(
        self,
        plan: ObjectPlan,
        inferencer,
        session_timestamp: Optional[str] = None,
    ) -> PlacementResult:
        """
        统一的物体摆放流程

        流程：检索资产 → 筛选参考物体 → 生成指令 → 推理位姿 → 验证 → 添加到场景

        Args:
            plan: 物体规划信息
            inferencer: 推理器
            session_timestamp: 会话时间戳（用于保存渲染图）

        Returns:
            摆放结果
        """
        object_id = self.state.get_next_object_id()

        # Step 1: 资产检索
        try:
            asset = self._retrieve_asset(plan)
        except AssetRetrievalError as e:
            error_message = str(e)
            self.log.error(error_message)
            self.state.record_failure(
                object_id=object_id,
                short_desc=plan.description,
                jid="",
                size=plan.estimated_size,
                error=error_message,
                parent_id=plan.parent_id,
            )
            return PlacementResult(
                success=False,
                partial_success=False,
                object_id=object_id,
                asset=None,
                parent_id=plan.parent_id,
                attempts_count=0,
                error=error_message,
            )

        # Step 2: 构建摆放任务
        scene_object = self._build_scene_object(
            object_id=object_id,
            asset=asset,
            parent_id=plan.parent_id,
        )

        # Step 3: 筛选参考物体（保持原始 ID 不变）
        filtered_objects = self._filter_reference_objects(
            must_keep_ids=set(plan.depends_on),
        )

        # Step 4: 渲染场景（如果启用视觉反馈，使用与指令一致的筛选后物体）
        scene_image_base64 = self._render_scene_for_feedback(
            scene_object,
            plan,
            session_timestamp,
            filtered_objects=filtered_objects,
        )

        # Step 5: 生成指令（使用筛选后的物体）
        instr_result = self._generate_placement_instruction(
            scene_object,
            plan,
            scene_image_base64,
            filtered_objects=filtered_objects,
        )

        # Step 6: 推理和验证（带重试，使用筛选后的物体）
        return self._infer_and_validate(
            scene_object=scene_object,
            asset=asset,
            plan=plan,
            instruction=instr_result,
            inferencer=inferencer,
            scene_image_base64=scene_image_base64,
            filtered_objects=filtered_objects,
        )

    def _infer_and_validate(
        self,
        scene_object: SceneObject,
        asset: RetrievedAsset,
        plan: ObjectPlan,
        instruction: InstructionResult,
        inferencer,
        scene_image_base64: Optional[str] = None,
        filtered_objects: Optional[List[SceneObject]] = None,
    ) -> PlacementResult:
        """
        位姿推理和验证（带分层重试机制）

        重试策略：
        - 外层循环：指令重试（当指令符合但物理约束失败时，让 Agent 重新生成指令）
        - 内层循环：推理重试（当指令不符合时，使用相同指令重试推理）

        Args:
            scene_object: 待摆放的物体
            asset: 检索到的资产
            plan: 物体规划信息
            instruction: 指令生成结果
            inferencer: 推理器
            scene_image_base64: 场景渲染图（用于指令重试时的视觉反馈）
            filtered_objects: 筛选后的物体列表，None 时使用全量物体

        Returns:
            摆放结果
        """
        placement_candidates = []
        all_attempts = []
        max_instruction_retries = 1
        max_inference_retries = self.max_retries

        # 用于推理和符合度验证的物体列表
        compliance_objects = (
            filtered_objects
            if filtered_objects is not None
            else self.state.scene_model.objects
        )

        infer_task = self._build_placement_request(
            scene_object,
            instruction.instruction,
            expected_relations=instruction.expected_relations,
        )

        current_instruction = instruction.instruction
        current_thinking = instruction.thinking
        current_expected_relations = instruction.expected_relations
        last_validation = None
        last_compliance = None

        for instr_attempt in range(max_instruction_retries):
            if instr_attempt > 0:
                self.log.info(
                    f"Instruction retry {instr_attempt}/{max_instruction_retries - 1}"
                )

            # 内层循环：推理重试
            for infer_attempt in range(max_inference_retries):
                self.log.indent(
                    f"Inference attempt {infer_attempt + 1}/{max_inference_retries}..."
                )

                t0 = time.perf_counter()
                result = inferencer.place_one_object(
                    infer_task,
                    temperature=self.infer_temp,
                    auto_update_scene=False,
                    existing_objects_override=(
                        compliance_objects if filtered_objects is not None else None
                    ),
                )
                self._timing.record("gala_inference", time.perf_counter() - t0)

                if not result.success or result.pose is None:
                    self.log.error(f"Inference failed: {result.error or 'unknown'}")
                    continue

                pose = result.pose

                placed_object = self._with_pose(scene_object, pose)
                # 物理约束验证（使用全量物体检测碰撞）；可经 use_placement_validator 关闭
                if self.use_placement_validator:
                    validation = self.validator.validate(
                        placed_object,
                        self.state.scene_model.objects,
                        self.state.boundary,
                        self.state.scene_height,
                        expected_relations=current_expected_relations,
                    )
                else:
                    validation = dict(_SKIPPED_PHYSICAL_VALIDATION)
                last_validation = validation

                # 指令符合度检测（使用筛选后物体，ID 与 expected_relations 一致）
                compliance = self.validator.check_instruction_compliance(
                    placed_object,
                    compliance_objects,
                    self.state.boundary,
                    current_expected_relations,
                )
                last_compliance = compliance

                # 计算违规分数
                score = self.validator.compute_violation_score(validation)

                # 判断这次尝试是否成功
                is_compliant = compliance.get("compliant", True)
                is_physically_valid = validation.get("valid", False)
                attempt_success = is_compliant and is_physically_valid

                # 记录这次尝试
                attempt = PlacementAttempt(
                    attempt_type="inference",
                    instruction=current_instruction,
                    instruction_thinking=current_thinking,
                    expected_relations=current_expected_relations,
                    parent_id=scene_object.parent_id,
                    pose=pose,
                    raw_output=result.raw_output,
                    validation_result=validation,
                    compliance_result=compliance,
                    success=attempt_success,
                )
                all_attempts.append(attempt)

                # 记录候选结果
                placement_candidates.append(
                    {
                        "pose": pose,
                        "validation": validation,
                        "compliance": compliance,
                        "score": score,
                        "raw_output": result.raw_output,
                        "instruction": current_instruction,
                        "instruction_thinking": current_thinking,
                        "expected_relations": current_expected_relations,
                        "parent_id": scene_object.parent_id,
                    }
                )

                # 判断结果
                if not is_compliant:
                    failed_relations = [
                        r
                        for r in compliance.get("relative_results", [])
                        if not r.get("satisfied")
                    ]
                    failed_relations += [
                        r
                        for r in compliance.get("absolute_results", [])
                        if not r.get("satisfied")
                    ]
                    self.log.warning(
                        f"Instruction non-compliant (compliance: {compliance.get('compliance_ratio', 0):.1%})"
                    )
                    self.log.indent(f"Failed relations: {failed_relations}", level=2)
                    self.log.indent("→ Inference model issue, retrying...", level=2)
                    continue

                # 指令符合，检查物理约束
                if is_physically_valid:
                    # 完全成功
                    self.log.success(
                        f"Validation passed (score: {score:.2f}, compliance: {compliance.get('compliance_ratio', 1.0):.1%})"
                    )
                    inferencer.add_object_to_scene(
                        self._build_placement_request(
                            scene_object,
                            current_instruction,
                            expected_relations=current_expected_relations,
                        ),
                        pose,
                    )

                    self.state.add_object(
                        object_id=scene_object.id,
                        short_desc=scene_object.short_desc,
                        jid=scene_object.jid,
                        center=pose.center,
                        rotation=pose.rotation,
                        size=scene_object.size,
                        category=scene_object.category,
                        instruction=current_instruction,
                        instruction_thinking=current_thinking,
                        raw_output=result.raw_output,
                        validation_result=validation,
                        compliance_result=compliance,
                        expected_relations=current_expected_relations,
                        parent_id=scene_object.parent_id,
                        attempts=all_attempts,
                        partial_success=False,
                    )

                    return PlacementResult(
                        success=True,
                        partial_success=False,
                        object_id=scene_object.id,
                        asset=asset,
                        pose=pose,
                        instruction=current_instruction,
                        instruction_thinking=current_thinking,
                        validation=validation,
                        compliance=compliance,
                        parent_id=scene_object.parent_id,
                        attempts_count=len(all_attempts),
                    )
                else:
                    # 指令符合但物理约束失败
                    self.log.warning(
                        f"Physical validation failed: {validation['errors']} (score: {score:.2f})"
                    )
                    self.log.indent(
                        "Instruction compliant but physical constraint violated",
                        level=2,
                    )
                    self.log.indent("→ Will regenerate instruction...", level=2)
                    break

            # 检查是否需要重新生成指令
            if (
                last_validation
                and last_validation.get("valid")
                and last_compliance
                and last_compliance.get("compliant")
            ):
                break

            instruction_compliant_but_physical_failed = (
                last_compliance
                and last_compliance.get("compliant")
                and last_validation
                and not last_validation.get("valid")
            )
            instruction_never_compliant = last_compliance and not last_compliance.get(
                "compliant"
            )

            should_regenerate = (
                instruction_compliant_but_physical_failed or instruction_never_compliant
            )

            if should_regenerate and instr_attempt < max_instruction_retries - 1:
                if instruction_never_compliant:
                    self.log.warning(
                        f"Inference model failed to comply after {max_inference_retries} attempts"
                    )
                    self.log.indent("→ Will regenerate simpler instruction...")
                else:
                    self.log.info(
                        "Regenerating instruction based on physical constraint feedback..."
                    )

                new_instr = self._regenerate_instruction_with_feedback(
                    scene_object=scene_object,
                    plan=plan,
                    previous_instruction=current_instruction,
                    validation_result=last_validation,
                    compliance_result=last_compliance,
                    scene_image_base64=scene_image_base64,
                    filtered_objects=filtered_objects,
                )

                if (
                    new_instr.instruction
                    and new_instr.instruction != current_instruction
                ):
                    current_instruction = new_instr.instruction
                    current_thinking = new_instr.thinking
                    current_expected_relations = new_instr.expected_relations
                    infer_task.instruction = current_instruction
                    infer_task.extra["expected_relations"] = current_expected_relations
                    last_validation = None
                    last_compliance = None
                else:
                    self.log.warning(
                        "Agent returned same or empty instruction, stopping retry"
                    )
                    break

        # 所有尝试都不完全合法，选择最佳候选放入场景
        if placement_candidates:
            best = max(
                placement_candidates,
                key=lambda x: (
                    x.get("compliance", {}).get("compliance_ratio", 0),
                    x["score"],
                ),
            )
            best_pose = best["pose"]
            best_validation = best["validation"]
            best_compliance = best.get("compliance", {})
            best_score = best["score"]
            best_instruction = best.get("instruction", current_instruction)
            best_thinking = best.get("instruction_thinking", current_thinking)
            best_expected_relations = best.get(
                "expected_relations", current_expected_relations
            )
            best_parent_id = best.get("parent_id", scene_object.parent_id)

            self.log.warning(
                f"No perfect placement. Using best candidate (score: {best_score:.2f}, compliance: {best_compliance.get('compliance_ratio', 0):.1%})"
            )
            self.log.indent(f"Issues: {best_validation.get('errors', [])}", level=2)
            self.log.debug(f"Total attempts: {len(all_attempts)}")

            inferencer.add_object_to_scene(
                self._build_placement_request(
                    scene_object,
                    best_instruction,
                    expected_relations=best_expected_relations,
                ),
                best_pose,
            )

            self.state.add_object(
                object_id=scene_object.id,
                short_desc=scene_object.short_desc,
                jid=scene_object.jid,
                center=best_pose.center,
                rotation=best_pose.rotation,
                size=scene_object.size,
                category=scene_object.category,
                instruction=best_instruction,
                instruction_thinking=best_thinking,
                raw_output=best.get("raw_output", ""),
                validation_result=best_validation,
                compliance_result=best_compliance,
                expected_relations=best_expected_relations,
                parent_id=best_parent_id,
                attempts=all_attempts,
                partial_success=True,
            )

            return PlacementResult(
                success=False,
                partial_success=True,
                object_id=scene_object.id,
                asset=asset,
                pose=best_pose,
                instruction=best_instruction,
                instruction_thinking=best_thinking,
                validation=best_validation,
                compliance=best_compliance,
                parent_id=best_parent_id,
                attempts_count=len(all_attempts),
            )

        # 完全没有候选结果
        self.log.error("All inference attempts failed, no candidates available")
        self.log.debug(f"Total attempts: {len(all_attempts)}")
        self.state.record_failure(
            object_id=scene_object.id,
            short_desc=scene_object.short_desc,
            jid=scene_object.jid,
            size=scene_object.size,
            instruction=current_instruction,
            instruction_thinking=current_thinking,
            parent_id=scene_object.parent_id,
            error="All inference attempts failed (no valid pose returned)",
        )

        return PlacementResult(
            success=False,
            partial_success=False,
            object_id=scene_object.id,
            asset=asset,
            instruction=current_instruction,
            instruction_thinking=current_thinking,
            parent_id=scene_object.parent_id,
            attempts_count=len(all_attempts),
            error="All inference attempts failed",
        )

    def _placement_result_to_dict(self, result: PlacementResult) -> Dict[str, Any]:
        """将 PlacementResult 序列化为字典，供 _generate_layout 和 edit_scene 使用。"""
        return {
            "success": result.success,
            "partial_success": result.partial_success,
            "object_id": result.object_id,
            "short_desc": result.asset.short_desc if result.asset else "",
            "jid": result.asset.jid if result.asset else "",
            "size": result.asset.size if result.asset else [],
            "pose": result.pose.to_dict() if result.pose else None,
            "instruction": result.instruction,
            "instruction_thinking": result.instruction_thinking,
            "parent_id": result.parent_id,
            "validation": result.validation,
            "compliance": result.compliance,
            "attempts_count": result.attempts_count,
            "error": result.error,
            "violation_score": (
                self.validator.compute_violation_score(result.validation)
                if result.validation
                else 0
            ),
        }

    def generate_scene(
        self,
        user_input: str,
        boundary: List[List[float]],
        scene_height: float = DEFAULT_SCENE_HEIGHT,
        scene_id: str = "generated_scene",
    ) -> Dict[str, Any]:
        """
        生成完整场景

        Args:
            user_input: 用户输入的场景需求
            boundary: 房间边界
            scene_height: 房间高度
            scene_id: 场景 ID

        Returns:
            生成结果字典
        """
        start_time = time.time()
        session_timestamp = time.strftime("%Y%m%d_%H%M%S")
        self._reset_timing()

        self.log.newline()
        self.log.separator()
        self.log.info(f"Starting scene generation (session: {session_timestamp})")
        self.log.indent(f"User input: {user_input}")
        self.log.indent(f"Boundary: {boundary}")

        # Step 1: 场景规划
        t0 = time.perf_counter()
        plan = self.agent.plan_scene(user_input, boundary, scene_height)
        self._timing.record("scene_planner", time.perf_counter() - t0)

        self.log.info(
            f"Scene type: {plan.get('scene_type')}, objects: {len(plan.get('objects', []))}"
        )

        # 初始化状态管理器
        self.state = SceneStateManager(
            scene_id=scene_id,
            boundary=boundary,
            scene_height=scene_height,
            scene_description=plan.get("scene_description", user_input),
            scene_type=plan.get("scene_type", "room"),
        )

        # 获取推理器（传递时间戳以避免点云路径覆盖）
        inferencer = self._get_inferencer(session_timestamp=session_timestamp)
        inferencer.init_scene(self.state.scene_model)

        # Step 2: 逐个物体处理
        self.log.newline()
        self.log.info("Placing objects...")
        results = []
        # 规划 ID → 运行时 ID 的映射（物体失败时不会写入）
        plan_id_to_runtime_id: Dict[int, int] = {}

        for idx, obj_plan in enumerate(plan.get("objects", [])):
            self.log.object_header(
                idx, len(plan["objects"]) - 1, obj_plan.get("description")
            )

            # 将 depends_on 中的规划 ID 映射为运行时 ID（跳过映射不到的）
            raw_depends = obj_plan.get("depends_on", [])
            runtime_depends = [
                plan_id_to_runtime_id[pid]
                for pid in raw_depends
                if pid in plan_id_to_runtime_id
            ]

            # 构建 ObjectPlan 并调用统一的摆放流程
            placement_result = self._place_object(
                plan=ObjectPlan(
                    description=obj_plan.get("description"),
                    estimated_size=obj_plan.get("estimated_size"),
                    depends_on=runtime_depends,
                ),
                inferencer=inferencer,
                session_timestamp=session_timestamp,
            )

            # 记录规划 ID → 运行时 ID 映射
            plan_id = obj_plan.get("id", idx)
            if placement_result.success or placement_result.partial_success:
                plan_id_to_runtime_id[plan_id] = placement_result.object_id

            result = self._placement_result_to_dict(placement_result)
            results.append(result)

            if placement_result.success:
                self.log.success(f"Placed at {placement_result.pose.center}")
            elif placement_result.partial_success:
                self.log.warning(
                    f"Placed with issues at {placement_result.pose.center}"
                )
            else:
                self.log.error(f"Failed: {placement_result.error or 'unknown'}")

        # 生成 layout 输出
        elapsed_time = time.time() - start_time
        layout = self._generate_layout(plan, results, elapsed_time, user_input)

        # 保存结果
        self._save_results(layout)

        self.log.newline()
        self.log.separator()
        self.log.info(f"Scene generation completed in {elapsed_time:.2f}s")
        success_rate = layout["successful"] / max(layout["total_tasks"], 1)
        self.log.summary(
            successful=layout["successful"],
            partial=layout["partial_success"],
            failed=layout["failed"],
            total_attempts=layout["total_attempts"],
            success_rate=f"{success_rate:.1%}",
        )
        self._log_timing_summary(elapsed_time)

        return layout

    def edit_scene(
        self,
        user_input: str,
    ) -> Dict[str, Any]:
        """
        编辑模式：在现有场景上添加物体

        Args:
            user_input: 用户的编辑请求（如 "再加一个落地灯放在沙发旁边"，"在床的右边放一个书架"）

        Returns:
            编辑结果
        """
        if self.state is None:
            raise ValueError(
                "No scene loaded. Call generate_scene or load_scene first."
            )

        start_time = time.time()
        self._reset_timing()

        self.log.newline()
        self.log.separator()
        self.log.info("Editing scene")
        self.log.indent(f"User input: {user_input}")

        # 1. 解析编辑请求（获取物体描述）
        edit_info = None
        max_edit_retries = 3
        for edit_attempt in range(max_edit_retries):
            t0 = time.perf_counter()
            edit_info = self.agent.generate_edit_instruction(
                user_input=user_input,
                scene=self.state.scene_model,
                existing_objects=self.state.scene_model.objects,
            )
            self._timing.record("instruction_generator", time.perf_counter() - t0)
            obj_desc = edit_info.get("object_description", "") if edit_info else ""
            if obj_desc and obj_desc.strip():
                break
            self.log.warning(
                f"Empty object description returned, retrying ({edit_attempt + 1}/{max_edit_retries})..."
            )

        if not edit_info or not edit_info.get("object_description", "").strip():
            self.log.warning("All edit parsing attempts failed, using fallback")
            if edit_info is None:
                edit_info = {}
            edit_info["object_description"] = user_input

        self.log.info(f"Parsed object: {edit_info.get('object_description', '')}")

        # 2. 获取推理器并初始化场景
        inferencer = self._get_inferencer()
        inferencer.init_scene(self.state.scene_model)

        # 3. 构建 ObjectPlan 并调用统一的摆放流程
        edit_relations = edit_info.get("expected_relations") or None
        # 编辑模式：从 expected_relations 中提取 target_id 作为语义依赖
        edit_depends_on = []
        if edit_relations:
            for rel in edit_relations.get("relative_relations", []):
                tid = rel.get("target_id")
                if tid is not None:
                    edit_depends_on.append(tid)
        # 提取 parent_id
        edit_parent_id = (
            ScenePlannerAgent._parse_parent_id(edit_info) if edit_info else None
        )
        placement_result = self._place_object(
            plan=ObjectPlan(
                description=edit_info.get("object_description", ""),
                estimated_size=edit_info.get("estimated_size", [0.5, 0.5, 0.5]),
                thinking=edit_info.get("thinking", ""),
                expected_relations=edit_relations,
                parent_id=edit_parent_id,
                depends_on=edit_depends_on,
            ),
            inferencer=inferencer,
            session_timestamp=time.strftime("%Y%m%d_%H%M%S"),
        )

        # 4. 返回结果
        result = self._placement_result_to_dict(placement_result)

        if placement_result.success:
            self.log.success(f"Added {placement_result.asset.short_desc}")
        elif placement_result.partial_success:
            self.log.success(
                f"Added {placement_result.asset.short_desc} (partial success)"
            )
        else:
            self.log.error(
                f"Failed to add object: {placement_result.error or 'unknown'}"
            )

        elapsed_time = time.time() - start_time
        self._log_timing_summary(elapsed_time)
        result["timing_stats"] = self._timing.to_dict(elapsed_time)
        result["elapsed_time"] = elapsed_time

        return result

    def load_scene(self, path: str):
        """加载已保存的场景状态"""
        self.state = SceneStateManager()
        self.state.load(path)
        self.log.info(f"Loaded scene from {path}")

    def _generate_layout(
        self,
        plan: Dict,
        results: List[Dict],
        elapsed_time: float,
        user_input: str = "",
    ) -> Dict[str, Any]:
        """生成 layout JSON，包含摆放结果、场景状态和重试历史。"""
        # 分三类：完全成功、部分成功、完全失败
        successful = [r for r in results if r.get("success")]  # success=True
        partial_success = [
            r for r in results if not r.get("success") and r.get("partial_success")
        ]  # success=False, partial=True
        failed = [
            r for r in results if not r.get("success") and not r.get("partial_success")
        ]  # success=False, partial=False

        # 构建 successful_placements（包括完全成功和部分成功，即所有放入场景的物体）
        placed_objects = successful + partial_success
        successful_placements = []
        for r in placed_objects:
            placement = {
                "object_id": r["object_id"],
                "object_short_desc": r["short_desc"],  # 兼容 batch inference 命名
                "instruction": r.get("instruction", ""),
                "instruction_thinking": r.get("instruction_thinking", ""),
                "parent_id": r.get("parent_id"),
                "jid": r["jid"],
                "pose": r["pose"],
                "size": r.get("size", []),
                "partial_success": r.get(
                    "partial_success", False
                ),  # 标记是否为部分成功
                "attempts_count": r.get("attempts_count", 1),  # 尝试次数
            }
            # 添加验证结果（如果有）
            if r.get("validation"):
                placement["validation_result"] = r["validation"]
            if r.get("compliance"):
                placement["compliance_result"] = r["compliance"]
            successful_placements.append(placement)

        # 构建 failed_placements
        failed_placements = [
            {
                "object_id": r.get("object_id"),
                "object_short_desc": r.get("short_desc"),
                "parent_id": r.get("parent_id"),
                "error": r.get("error", "unknown"),
                "attempts_count": r.get("attempts_count", 0),
            }
            for r in failed
        ]

        # 构建 final_scene（使用统一场景格式）
        final_scene = self.state.scene_model.to_dict(object_field="objects")

        # 构建完整的 placement_history（包括重试记录）
        placement_history = [
            {
                "object_id": r.object_id,
                "short_desc": r.short_desc,
                "jid": r.jid,
                "instruction": r.instruction,
                "instruction_thinking": r.instruction_thinking,
                "expected_relations": r.expected_relations,
                "parent_id": r.parent_id,
                "pose": r.pose.to_dict() if isinstance(r.pose, Pose) else r.pose,
                "size": r.size,
                "success": r.success,
                "partial_success": r.partial_success,
                "timestamp": r.timestamp,
                "raw_output": r.raw_output,
                "validation_result": r.validation_result,
                "compliance_result": r.compliance_result,
                "attempts": [
                    {
                        "attempt_type": a.attempt_type,
                        "instruction": a.instruction,
                        "instruction_thinking": a.instruction_thinking,
                        "expected_relations": a.expected_relations,
                        "parent_id": a.parent_id,
                        "pose": (
                            a.pose.to_dict() if isinstance(a.pose, Pose) else a.pose
                        ),
                        "raw_output": a.raw_output,
                        "validation_result": a.validation_result,
                        "compliance_result": a.compliance_result,
                        "success": a.success,
                        "timestamp": a.timestamp,
                    }
                    for a in r.attempts
                ],
            }
            for r in self.state.placement_history
        ]

        # 统计总重试次数
        total_attempts = sum(r.get("attempts_count", 1) for r in results)

        return {
            # 输入信息
            "user_input": user_input,
            "boundary": self.state.boundary,
            "scene_height": self.state.scene_height,
            "infer_temp": self.infer_temp,
            # 基础统计
            "total_tasks": len(results),
            "successful": len(successful),
            "partial_success": len(partial_success),
            "failed": len(failed),
            "total_attempts": total_attempts,
            # Agent 特有信息
            "scene_type": plan["scene_type"],
            "scene_planner": dict(plan),
            "elapsed_time": elapsed_time,
            "timing_stats": self._timing.to_dict(elapsed_time),
            "inference_model": self._inference_model_path,
            # 摆放记录
            "successful_placements": successful_placements,
            "failed_placements": failed_placements,
            # 详细摆放历史（包括所有重试）
            "placement_history": placement_history,
            # 最终场景（兼容可视化脚本）
            "final_scene": final_scene,
        }

    def _save_results(self, layout: Dict):
        """
        保存结果为统一的 layout.json 格式

        Args:
            layout: 布局结果字典
        """
        scene_id = layout.get("final_scene", {}).get("scene_id", "scene")

        # 保存统一的 layout 文件（每个 run 一个独立文件夹，文件名不再带时间戳）
        layout_path = self.output_dir / f"{scene_id}_layout.json"
        with open(layout_path, "w", encoding="utf-8") as f:
            json.dump(layout, f, indent=2, ensure_ascii=False)

        self.log.info(f"Layout saved to {layout_path}")
