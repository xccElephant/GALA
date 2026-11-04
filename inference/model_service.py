import os
from typing import Dict, List, Optional, Union

from utils.message import (
    SYSTEM_PROMPT,
    build_scene_input_sample_from_types,
    format_scene_input,
)
from utils.logger import Logger
from utils.scene_types import (
    InferenceResult,
    ModelResponse,
    Scene,
    PlacementRequest,
    ensure_scene,
    ensure_placement_request,
    parse_pose,
)

# 模块级别的 logger
_log = Logger("Model", verbose=True)


def _load_pointlm_classes():
    try:
        from llamafactory.model.pointlm.inference import GenerationConfig, PointLMInference
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PointLM inference requires the LLaMA-Factory PointLM extension. "
            "Install the inference extra and the project-specific LLaMA-Factory branch "
            "before constructing ModelService."
        ) from exc
    return PointLMInference, GenerationConfig


class ModelService:
    """
    Model service for spatial language model
    """

    def __init__(
        self,
        model_path: str,
        inference_dtype: str = "bfloat16",
        include_object_descriptions: bool = False,
        only_pointcloud: bool = False,
    ):
        """
        初始化模型服务

        Args:
            model_path: 模型路径
            inference_dtype: 推理数据类型 (bfloat16/float32)
            include_object_descriptions: 是否在推理 prompt 中暴露物体文字描述
            only_pointcloud: 是否使用仅保留点云模式，不给物体的 size, center, rotation 等三维物理特征
        """
        self._include_object_descriptions = include_object_descriptions
        self._only_pointcloud = only_pointcloud
        pointlm_inference_cls, generation_config_cls = _load_pointlm_classes()
        self._generation_config_cls = generation_config_cls
        self._inference = pointlm_inference_cls(
            model_path=model_path,
            inference_dtype=inference_dtype,
            verbose=_log.verbose,
        )

    def load_model(self):
        """加载模型和 tokenizer"""
        if self._inference.is_loaded:
            _log.debug("Model already loaded")
            return

        self._inference.load()

    def generate_response(
        self,
        scene_pointcloud_path: Optional[str],
        prompt: str,
        new_object_pointcloud_path: Optional[str] = None,
        separate_point_cloud_paths: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        top_k: int = 10,
        top_p: float = 0.95,
        temperature: float = 0.6,
        num_beams: int = 1,
        seed: int = -1,
        max_new_tokens: int = 4096,
        use_pointcloud: bool = True,
    ) -> ModelResponse:
        """
        生成响应（自回归推理：支持场景点云和新物体点云）

        Args:
            scene_pointcloud_path: 场景点云文件路径（use_pointcloud=False 时可为 None）
            prompt: 用户提示
            new_object_pointcloud_path: 新物体点云文件路径（可选）
            separate_point_cloud_paths: 分离点云路径列表（可选，与 scene/new object 互斥）
            system_prompt: 系统提示
            top_k: Top-k 采样参数
            top_p: Top-p 采样参数
            temperature: 采样温度
            num_beams: Beam search 数量
            seed: 随机种子
            max_new_tokens: 最大生成 token 数
            use_pointcloud: 是否使用点云输入（默认 True）

        Returns:
            ModelResponse
        """
        if not self._inference.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # 构建生成配置
        config = self._generation_config_cls(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            do_sample=temperature > 0,
            seed=seed,
        )

        # 确定点云路径
        scene_path = None
        object_path = None
        separate_paths = None
        if use_pointcloud:
            if separate_point_cloud_paths:
                separate_paths = separate_point_cloud_paths
            else:
                scene_path = scene_pointcloud_path
                object_path = new_object_pointcloud_path

        # 调用底层推理接口
        result = self._inference.generate(
            prompt=prompt,
            scene_point_cloud_path=scene_path,
            object_point_cloud_path=object_path,
            separate_point_cloud_paths=separate_paths,
            system_prompt=system_prompt,
            generation_config=config,
        )

        return ModelResponse(
            success=result.success,
            response=result.response,
            response_raw=result.response,
            error=result.error,
        )

    def generate_placement(
        self,
        scene_pointcloud_path: Optional[str],
        scene_info: Union[Scene, Dict],
        placement_task: Union[PlacementRequest, Dict],
        new_object_pointcloud_path: Optional[str] = None,
        separate_point_cloud_paths: Optional[List[str]] = None,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 10,
        max_new_tokens: int = 4096,
        seed: int = -1,
        use_pointcloud: bool = True,
        internal_spaces_map: Optional[Dict] = None,
    ) -> InferenceResult:
        """
        生成物体摆放位姿（自回归推理）

        Args:
            scene_pointcloud_path: 场景点云路径（use_pointcloud=False 时可为 None）
            scene_info: 场景信息
            placement_task: 摆放任务
            new_object_pointcloud_path: 新物体点云路径（可选）
            separate_point_cloud_paths: 分离点云路径列表（可选）
            temperature: 采样温度
            top_p: Top-p 参数
            top_k: Top-k 参数
            max_new_tokens: 最大生成 token 数
            seed: 随机种子
            use_pointcloud: 是否使用点云输入（默认 True）
            internal_spaces_map: 收纳空间映射表 {jid: [subspaces]}（None 表示不附加）

        Returns:
            InferenceResult
        """
        scene = ensure_scene(scene_info)
        request = ensure_placement_request(placement_task)

        # 统一构建输入（外部 prompt 层仍使用模型契约要求的 JSON 结构）
        sample = build_scene_input_sample_from_types(scene, request)

        if not use_pointcloud:
            # 纯文本：不包含任何 pointcloud 字段
            system_prompt = SYSTEM_PROMPT
            prompt = format_scene_input(
                sample,
                use_separate_pointcloud=False,
                internal_spaces_map=internal_spaces_map,
                include_object_descriptions=self._include_object_descriptions,
                only_pointcloud=self._only_pointcloud,
            )
        else:
            # 分离点云模式：floor + existing objects + new object
            system_prompt = SYSTEM_PROMPT
            prompt = format_scene_input(
                sample,
                use_separate_pointcloud=True,
                internal_spaces_map=internal_spaces_map,
                include_object_descriptions=self._include_object_descriptions,
                only_pointcloud=self._only_pointcloud,
            )

        # 打印发送给推理模型的 prompt
        _log.info("=" * 60)
        _log.info("Prompt to inference model:")
        _log.info("=" * 60)
        # _log.info(f"[System] {system_prompt}")
        _log.info(f"[User] {prompt}")
        _log.info("=" * 60)

        # 调用生成
        result = self.generate_response(
            scene_pointcloud_path=scene_pointcloud_path,
            prompt=prompt,
            new_object_pointcloud_path=new_object_pointcloud_path,
            separate_point_cloud_paths=separate_point_cloud_paths,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            seed=seed,
            use_pointcloud=use_pointcloud,
        )

        if not result.success:
            return InferenceResult(
                success=False,
                raw_output="",
                error=result.error,
            )

        response_text = result.response
        pose = parse_pose(response_text)

        if pose is None:
            _log.debug("Model raw output (first 500 chars):")
            _log.indent(response_text[:500])
            if len(response_text) > 500:
                _log.indent(f"... (total {len(response_text)} chars)")

        return InferenceResult(
            success=pose is not None,
            raw_output=response_text,
            pose=pose,
            error=None if pose else "Failed to parse pose from model output",
        )

    @property
    def is_loaded(self) -> bool:
        """模型是否已加载"""
        return self._inference.is_loaded

    @property
    def supports_object_point_cloud(self) -> bool:
        """是否支持物体点云"""
        return self._inference.supports_object_point_cloud

