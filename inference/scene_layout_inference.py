"""
场景布局推理脚本, 支持自回归迭代摆放
"""
import argparse
import csv
import json
import os
import re
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from config.paths import ProjectPaths
from inference.model_service import ModelService
from inference.pointcloud_scene_manager import PointCloudSceneManager
from utils.logger import Logger
from utils.message import build_model_instruction, load_internal_spaces_map
from utils.scene_types import (
    InferenceInput,
    InferenceResult,
    LayoutInferenceSummary,
    PlacementRequest,
    PlacementStepResult,
    Pose,
    Scene,
    SceneObject,
    ensure_placement_request,
    ensure_pose,
    ensure_scene,
    ensure_scene_object_list,
)


class SceneLayoutInference:
    """
    场景迭代式摆放推理类, 调用 PointCloudModelService 进行推理
    支持自回归推理：每一步迭代更新场景点云。
    
    点云更新流程：
    1. 初始化场景时，生成初始场景点云（地板 + 已存在物体）
    2. 每步推理成功后，根据推理出的位姿，将新物体点云变换后加入场景点云
    3. 下一步推理使用更新后的场景点云
    """

    def __init__(
        self,
        model_path: str,
        inference_dtype: str = "bfloat16",
        asset_pointcloud_dir: Union[str, List[str]] = None,
        output_dir: str = None,
        use_pointcloud: bool = True,
        session_timestamp: str = None,
        internal_spaces_dir: str = None,
        include_object_descriptions: bool = False,
        only_pointcloud: bool = False,
        seed: int = 42,
    ):
        """
        初始化本地推理器

        Args:
            model_path: 模型路径
            inference_dtype: 推理数据类型
            asset_pointcloud_dir: 资产点云目录
            output_dir: 输出目录（用于保存场景点云）
            use_pointcloud: 是否启用点云输入（默认 True）
            session_timestamp: 会话时间戳（用于点云输出目录，避免覆盖）
            internal_spaces_dir: 收纳空间数据目录路径；None 表示不附加 internal_spaces
            include_object_descriptions: 是否在推理 prompt 中暴露物体文字描述
            only_pointcloud: 是否剔除几何位置、尺寸、角度属性，仅采用点云模式推理
            seed: 推理随机种子；用于控制点云预处理和生成阶段的随机性
        """
        # 初始化日志
        self.log = Logger("Inference", verbose=True)
        
        # 初始化模型服务
        self.log.info(f"Loading model from {model_path}...")
        self.model_service = ModelService(
            model_path=model_path,
            inference_dtype=inference_dtype,
            include_object_descriptions=include_object_descriptions,
            only_pointcloud=only_pointcloud,
        )
        self.model_service.load_model()

        # 当前场景
        self.scene: Optional[Scene] = None
        
        # 点云场景管理器
        self.pointcloud_manager: Optional[PointCloudSceneManager] = None
        self._use_pointcloud = use_pointcloud
        self._asset_pointcloud_dir = asset_pointcloud_dir
        if self._use_pointcloud and self._asset_pointcloud_dir is None:
            self._asset_pointcloud_dir = ProjectPaths.get_asset_pointcloud_dir("3dish")
            self.log.info(
                "asset_pointcloud_dir not provided, use default search dirs: "
                f"{self._asset_pointcloud_dir}"
            )
        self._output_dir = output_dir or "output/scene_pointclouds"
        self._session_timestamp = session_timestamp or time.strftime("%Y%m%d_%H%M%S")
        self._seed = seed
        self._include_object_descriptions = include_object_descriptions
        self.log.info(f"Inference seed: {self._seed}")

        # 收纳空间映射表（{jid: [subspaces]}）
        self._internal_spaces_map: Optional[Dict] = None
        if internal_spaces_dir:
            self._internal_spaces_map = load_internal_spaces_map(Path(internal_spaces_dir))
            self.log.info(f"Loaded internal_spaces for {len(self._internal_spaces_map)} assets")

    @staticmethod
    def _extract_instruction_ids(instruction: str) -> List[int]:
        return [int(x) for x in re.findall(r"\(id (\d+)\)", instruction)]

    def _build_model_visible_instruction(
        self,
        request: PlacementRequest,
        existing_objects: Sequence[Union[SceneObject, Dict]],
    ) -> str:
        existing_object_dicts = [
            obj.to_dict() if isinstance(obj, SceneObject) else dict(obj)
            for obj in existing_objects
        ]
        return build_model_instruction(
            {
                "instruction": request.instruction,
                "expected_relations": request.extra.get("expected_relations"),
                "existing_objects": existing_object_dicts,
                "new_object": request.to_dict(),
            },
            include_object_descriptions=getattr(self, "_include_object_descriptions", False),
        )

    @classmethod
    def _validate_task_reference_context(
        cls,
        task: Union[PlacementRequest, Dict],
    ) -> None:
        request = ensure_placement_request(task)
        if request.new_object_id is None:
            raise ValueError(
                "Each task must contain 'new_object_id' generated from metadata new_object.id."
            )

        referenced_in_instruction = cls._extract_instruction_ids(request.instruction)
        if not request.reference_objects:
            if request.new_object_id != 0:
                raise ValueError(
                    "Each task must contain 'reference_objects' generated from metadata row context."
                )
            if referenced_in_instruction:
                raise ValueError(
                    "Instruction references IDs but reference_objects is empty."
                )
            return

        reference_ids = [obj.id for obj in request.reference_objects]
        if len(reference_ids) != len(set(reference_ids)):
            raise ValueError(
                f"reference_objects IDs must be unique, got {reference_ids}."
            )

        new_object_id = request.new_object_id
        if any(reference_ids[idx] >= reference_ids[idx + 1] for idx in range(len(reference_ids) - 1)):
            raise ValueError(
                "reference_objects IDs must be strictly increasing in placement order, "
                f"got {reference_ids}."
            )

        reference_id_set = set(reference_ids)
        if new_object_id in reference_id_set:
            raise ValueError(
                f"new_object_id must not appear in reference_objects, got {new_object_id}."
            )
        if any(ref_id >= new_object_id for ref_id in reference_ids):
            raise ValueError(
                "reference_objects must only contain earlier objects than new_object_id, "
                f"got new_object_id={new_object_id}, reference_ids={reference_ids}."
            )

        invalid_ids = sorted({rid for rid in referenced_in_instruction if rid not in reference_id_set})
        if invalid_ids:
            raise ValueError(
                f"Instruction references IDs not in reference_objects: {invalid_ids}."
            )

    @property
    def scene_info(self) -> Dict:
        if self.scene is None:
            return {}
        return self.scene.to_dict()


    def init_scene(
        self,
        scene_info: Union[Scene, Dict],
    ) -> Optional[str]:
        """
        设置场景基础信息并初始化场景点云

        Args:
            scene_info: 场景信息字典，包含：
                - scene_id: 场景 ID
                - scene_description: 场景描述
                - boundary: 地板边界点列表
                - scene_height: 房间层高
                - existing_objects: 已存在物体列表（每个物体需要 jid, center, rotation）
        
        Returns:
            初始场景点云文件路径
        """
        self.scene = ensure_scene(scene_info)
        scene_id = self.scene.scene_id
        self.log.info(f"Scene: {scene_id}")
        
        # 初始化点云场景管理器（仅在使用点云时）
        initial_scene_pc_path = None
        if self._use_pointcloud and self._asset_pointcloud_dir is not None:
            self.pointcloud_manager = PointCloudSceneManager(
                asset_pointcloud_dir=self._asset_pointcloud_dir,
                output_dir=self._output_dir,
                session_timestamp=self._session_timestamp,
            )
            
            # 初始化场景点云（地板 + 已存在物体）
            initial_scene_pc_path = self.pointcloud_manager.init_scene(
                scene_id=scene_id,
                boundary=self.scene.boundary,
                existing_objects=self.scene.objects,
            )
            # 注意：pointcloud 路径由 PointCloudSceneManager 打印
        elif not self._use_pointcloud:
            self.log.info("Pointcloud input disabled")
        
        return initial_scene_pc_path

    def place_one_object(
        self,
        task: Union[PlacementRequest, Dict],
        temperature: float = 0.6,
        auto_update_scene: bool = True,
        existing_objects_override: Optional[Sequence[Union[SceneObject, Dict]]] = None,
        seed: Optional[int] = None,
    ) -> InferenceResult:
        """
        摆放一个物体（自回归推理）

        Args:
            task: 摆放任务
            temperature: 采样温度
            auto_update_scene: 是否自动将物体添加到场景
            existing_objects_override: 可选的参考物体列表覆盖
            seed: 本次推理使用的 seed; None 时使用 ``self._seed``。
        """
        if self.scene is None:
            raise RuntimeError("Scene is not initialized. Call init_scene() first.")

        request = ensure_placement_request(task)
        jid = request.jid

        if existing_objects_override is not None:
            effective_scene = self.scene.copy_with_objects(ensure_scene_object_list(existing_objects_override))
        else:
            effective_scene = self.scene
        
        scene_pointcloud_path = None
        new_object_pointcloud_path = None
        separate_point_cloud_paths = None
        
        if self._use_pointcloud:
            scene_pointcloud_path = (
                self.pointcloud_manager.get_current_scene_pointcloud_path()
            )
            if self.model_service.supports_object_point_cloud:
                new_object_pointcloud_path = self.pointcloud_manager.get_new_object_pointcloud_path(
                    jid,
                    size=request.size,
                )

            try:
                separate_point_cloud_paths = self.pointcloud_manager.build_separate_pointcloud_paths(
                    existing_objects=effective_scene.objects,
                    new_object_jid=jid,
                    new_object_size=request.size,
                )
            except Exception as e:
                return InferenceResult(
                    success=False,
                    error=f"Failed to build separate pointcloud paths: {e}",
                    updated_scene_pointcloud_path=scene_pointcloud_path,
                )

            if separate_point_cloud_paths:
                self.log.info(
                    f"[separate pcd mode] {len(separate_point_cloud_paths)} pcds: "
                    + ", ".join(p.split("/")[-1] for p in separate_point_cloud_paths)
                )

            if not scene_pointcloud_path and not separate_point_cloud_paths:
                return InferenceResult(
                    success=False,
                    error=(
                        "scene_pointcloud_path or separate_point_cloud_paths is required "
                        "when use_pointcloud=True"
                    ),
                )

        self.log.newline()
        self.log.info(f"Placing: {request.short_desc} (id={request.id})")
        self.log.indent(
            f"Instruction: {self._build_model_visible_instruction(request, effective_scene.objects)}"
        )

        try:
            inference_start_time = time.time()
            effective_seed = self._seed if seed is None else int(seed)
            result = self.model_service.generate_placement(
                scene_pointcloud_path=scene_pointcloud_path,
                scene_info=effective_scene,
                placement_task=request,
                new_object_pointcloud_path=new_object_pointcloud_path,
                separate_point_cloud_paths=separate_point_cloud_paths,
                temperature=temperature,
                seed=effective_seed,
                use_pointcloud=self._use_pointcloud,
                internal_spaces_map=self._internal_spaces_map,
            )
            inference_time = time.time() - inference_start_time
            result.inference_time = inference_time
            self.log.debug(f"Inference time: {inference_time:.2f}s")
        except Exception as e:
            self.log.error(f"Inference failed: {e}")
            traceback.print_exc()
            return InferenceResult(
                success=False,
                error=str(e),
                updated_scene_pointcloud_path=scene_pointcloud_path,
            )

        updated_scene_pc_path = scene_pointcloud_path

        if result.success and result.pose is not None:
            pose = result.pose

            if auto_update_scene:
                updated_scene_pc_path = self.add_object_to_scene(request, pose)
            else:
                self.log.success(
                    f"{request.short_desc} → {pose.center}, rot={pose.rotation} (awaiting validation)"
                )
            
            result.updated_scene_pointcloud_path = updated_scene_pc_path
        else:
            self.log.error(f"Failed to place {request.short_desc}: {result.error}")
            result.updated_scene_pointcloud_path = scene_pointcloud_path

        return result

    def add_object_to_scene(
        self,
        task: Union[PlacementRequest, Dict],
        pose: Union[Pose, Dict],
    ) -> str:
        """
        将物体添加到场景（在外部验证通过后调用）
        
        Args:
            task: 摆放任务，包含 id, short_desc, size, jid
            pose: 位姿信息，包含 center, rotation
        
        Returns:
            更新后的场景点云路径
        """
        if self.scene is None:
            raise RuntimeError("Scene is not initialized. Call init_scene() first.")

        request = ensure_placement_request(task)
        normalized_pose = ensure_pose(pose)
        new_object = request.to_scene_object(normalized_pose)
        self.scene.add_object(new_object)
        
        # 更新场景点云（仅在使用点云时）
        updated_scene_pc_path = None
        if self._use_pointcloud and self.pointcloud_manager:
            updated_scene_pc_path = self.pointcloud_manager.add_placed_object(
                scene_object=new_object,
            )
        
        self.log.success(
            f"{new_object.short_desc} → {normalized_pose.center}, rot={normalized_pose.rotation}"
        )
        
        return updated_scene_pc_path

    def iterative_placement(
        self,
        placement_tasks: Sequence[Union[PlacementRequest, Dict]],
        temperature: float = 0.6,
    ) -> LayoutInferenceSummary:
        """
        迭代式摆放多个物体（自回归推理）

        自动进行迭代式点云更新：
        1. 使用初始场景点云进行第一步推理
        2. 每步推理成功后，自动更新场景点云
        3. 下一步使用更新后的场景点云

        Args:
            placement_tasks: 摆放任务列表，每个任务包含：
                - id: 物体 ID
                - short_desc: 物体简短描述
                - size: 物体尺寸 [width, depth, height]
                - instruction: 摆放指令
                - jid: 物体资产 ID
            temperature: 采样温度

        Returns:
            包含所有摆放结果的字典，包括最终场景点云路径
        """
        successful_placements = []
        failed_placements = []
        total_inference_time = 0.0
        normalized_tasks = [ensure_placement_request(task) for task in placement_tasks]

        if self.scene is None:
            raise RuntimeError("Scene is not initialized. Call init_scene() first.")

        self.log.info(f"Starting iterative placement for {len(normalized_tasks)} objects...")
        total_start_time = time.time()

        final_scene_pc_path = None

        for i, task in enumerate(normalized_tasks):
            self.log.step(i, len(normalized_tasks) - 1, f"Placing {task.short_desc}")

            # 真实的顺序 ID：当前已有物体数量，保证唯一且单调递增
            existing = self.scene.objects
            real_id = len(existing)

            self._validate_task_reference_context(task)
            model_obj_id = task.new_object_id

            # 将 reference_objects 中的 GT 位姿替换为已推理出的实际位姿
            inferred_by_id: Dict[int, SceneObject] = {
                obj.id: obj for obj in self.scene.objects
            }
            remapped_objects: List[SceneObject] = []
            for ref_obj in task.reference_objects:
                if ref_obj.id not in inferred_by_id:
                    raise ValueError(
                        f"reference object id={ref_obj.id} "
                        f"({ref_obj.short_desc}) not found in placed objects "
                        f"(likely failed in a previous step)."
                    )
                inferred = inferred_by_id[ref_obj.id]
                remapped_objects.append(ref_obj.copy_with_pose(inferred.require_pose()))
            self.log.debug(
                f"reference_objects: {len(remapped_objects)} objects, new_object_id={model_obj_id}"
            )

            infer_task = task.copy_with(id=model_obj_id)

            result = self.place_one_object(
                task=infer_task,
                temperature=temperature,
                auto_update_scene=False,
                existing_objects_override=remapped_objects,
            )

            total_inference_time += result.inference_time

            if result.success and result.pose is not None:
                storage_task = task.copy_with(id=real_id)
                updated_pc = self.add_object_to_scene(storage_task, result.pose)
                result.updated_scene_pointcloud_path = updated_pc

            if result.updated_scene_pointcloud_path:
                final_scene_pc_path = result.updated_scene_pointcloud_path

            model_visible_instruction = self._build_model_visible_instruction(
                task,
                task.reference_objects,
            )
            if result.success:
                successful_placements.append(
                    PlacementStepResult(
                        object_id=real_id,
                        object_short_desc=task.short_desc,
                        instruction=model_visible_instruction,
                        jid=task.jid,
                        pose=result.pose,
                        gt_pose=task.gt_pose,
                        scene_pointcloud_path=result.updated_scene_pointcloud_path,
                        inference_time=result.inference_time,
                    )
                )
            else:
                failed_placements.append(
                    PlacementStepResult(
                        object_id=real_id,
                        object_short_desc=task.short_desc,
                        instruction=model_visible_instruction,
                        raw_output=result.raw_output,
                        jid=task.jid,
                        gt_pose=task.gt_pose,
                        error=result.error,
                        inference_time=result.inference_time,
                    )
                )

        total_elapsed_time = time.time() - total_start_time

        # 汇总结果
        summary = LayoutInferenceSummary(
            total_tasks=len(normalized_tasks),
            successful_placements=successful_placements,
            failed_placements=failed_placements,
            final_scene=self.scene,
            final_scene_pointcloud_path=final_scene_pc_path,
            total_inference_time=total_inference_time,
            total_elapsed_time=total_elapsed_time,
            avg_inference_time=(
                total_inference_time / len(normalized_tasks)
                if normalized_tasks else 0.0
            ),
        )
        summary_data = summary.to_dict()

        self.log.newline()
        self.log.separator()
        self.log.info("Placement Summary")
        self.log.summary(
            total_tasks=summary_data['total_tasks'],
            successful=summary_data['successful'],
            failed=summary_data['failed'],
            total_time=f"{total_elapsed_time:.2f}s",
            avg_inference=f"{summary_data['avg_inference_time']:.2f}s",
        )

        return summary


def _parse_csv_json_field(field_str: str, field_name: str):
    if not field_str or field_str == "[]":
        return []
    try:
        return json.loads(field_str)
    except json.JSONDecodeError as exc:
        raise ValueError(f"字段 {field_name} 不是合法 JSON: {field_str[:120]}") from exc


def _parse_csv_optional_json_field(field_str: Optional[str], field_name: str):
    if field_str is None or field_str == "":
        return None
    try:
        return json.loads(field_str)
    except json.JSONDecodeError as exc:
        raise ValueError(f"字段 {field_name} 不是合法 JSON: {field_str[:120]}") from exc


def build_inference_input_from_metadata(
    scene_id: str,
    dataset: str = "3dish",
    start_step: int = 0,
    metadata_csv: Optional[str] = None,
    include_object_descriptions: bool = False,
    output_file: Optional[str] = None,
) -> InferenceInput:
    """从 metadata.csv 提取指定 scene_id 的场景信息与逐步摆放任务。"""
    resolved_csv = metadata_csv or str(ProjectPaths.get_metadata_path(dataset))
    print(f"[INFO] 读取 metadata.csv: {resolved_csv}")

    scene_rows = []
    with open(resolved_csv, "r", encoding="utf-8") as f:
        first_line = f.readline()
        f.seek(0)
        delimiter = "," if "," in first_line else "\t"
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            if row["scene_id"] == scene_id:
                scene_rows.append(row)

    if not scene_rows:
        raise ValueError(f"在 {resolved_csv} 中未找到 scene_id={scene_id} 的数据")

    scene_rows.sort(key=lambda x: int(x["step"]))
    print(f"[INFO] 找到 {len(scene_rows)} 条数据")

    first_row = scene_rows[0]
    scene_description = first_row["scene_description"]
    boundary = _parse_csv_json_field(first_row["boundary"], "boundary")
    scene_height = float(first_row["scene_height"])

    existing_objects: list[SceneObject] = []
    if start_step > 0:
        start_row = next((r for r in scene_rows if int(r["step"]) == start_step), None)
        if start_row:
            existing_objects_raw = _parse_csv_json_field(
                start_row["existing_objects"], "existing_objects"
            )
            for obj in existing_objects_raw:
                existing_objects.append(
                    SceneObject(
                        id=obj["id"],
                        short_desc=obj["short_desc"],
                        size=obj["size"],
                        jid=obj.get("jid"),
                        category=obj.get("category", ""),
                        pose=Pose(center=obj["center"], rotation=obj["rotation"]),
                    )
                )

    placement_tasks: list[PlacementRequest] = []
    initial_existing_count = len(existing_objects)

    for row in scene_rows:
        step = int(row["step"])
        if step < start_step:
            continue

        new_object_raw = _parse_csv_json_field(row["new_object"], "new_object")
        original_instruction = row.get("instruction", "")
        expected_relations = _parse_csv_optional_json_field(
            row.get("expected_relations"), "expected_relations"
        )
        if expected_relations is not None and not isinstance(expected_relations, dict):
            raise TypeError(f"scene={scene_id}, step={step}: expected_relations 必须是 dict")

        row_existing_objects_raw = _parse_csv_json_field(
            row["existing_objects"], "existing_objects"
        )
        if not isinstance(row_existing_objects_raw, list):
            raise TypeError(f"scene={scene_id}, step={step}: existing_objects 必须是 list")

        reference_objects = [
            SceneObject(
                id=obj["id"],
                short_desc=obj["short_desc"],
                size=obj["size"],
                jid=obj.get("jid"),
                category=obj.get("category", ""),
                pose=Pose(center=obj["center"], rotation=obj["rotation"]),
            )
            for obj in row_existing_objects_raw
        ]

        reference_ids = [obj.id for obj in reference_objects]
        if len(reference_ids) != len(set(reference_ids)):
            raise ValueError(
                f"scene={scene_id}, step={step}: reference_objects ID 存在重复: {reference_ids}"
            )
        if not include_object_descriptions and expected_relations is None:
            raise ValueError(
                f"scene={scene_id}, step={step}: 默认 category 模式要求 metadata 中包含 expected_relations"
            )

        if expected_relations is not None:
            instruction_text = build_model_instruction(
                {
                    "instruction": original_instruction,
                    "expected_relations": expected_relations,
                    "existing_objects": row_existing_objects_raw,
                    "new_object": new_object_raw,
                },
                include_object_descriptions=include_object_descriptions,
            )
        else:
            instruction_text = original_instruction

        instruction_ids = SceneLayoutInference._extract_instruction_ids(instruction_text)
        invalid_ids = sorted({rid for rid in instruction_ids if rid not in set(reference_ids)})
        if invalid_ids:
            raise ValueError(
                f"scene={scene_id}, step={step}: instruction 引用了不存在的 ID: {invalid_ids}"
            )

        sequential_id = initial_existing_count + len(placement_tasks)
        placement_tasks.append(
            PlacementRequest(
                step=step,
                id=sequential_id,
                new_object_id=int(new_object_raw["id"]),
                short_desc=new_object_raw["short_desc"],
                size=new_object_raw.get("size"),
                jid=new_object_raw.get("jid"),
                instruction=instruction_text,
                category=new_object_raw.get("category", ""),
                reference_objects=reference_objects,
                gt_pose=Pose(
                    center=new_object_raw["center"],
                    rotation=new_object_raw["rotation"],
                ),
                extra={} if expected_relations is None else {"expected_relations": expected_relations},
            )
        )

    scene = Scene(
        scene_id=scene_id,
        scene_description=scene_description,
        boundary=boundary,
        scene_height=scene_height,
        objects=existing_objects,
    )
    inference_input = InferenceInput(scene=scene, placement_tasks=placement_tasks)

    if output_file:
        out_dir = os.path.dirname(output_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(inference_input.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"[SUCCESS] 推理输入已保存到: {output_file}")

    return inference_input


def main() -> None:
    parser = argparse.ArgumentParser(description="场景迭代式摆放推理")

    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        default=None,
        help="模型路径（--prepare_input_only 模式下可省略）",
    )
    parser.add_argument(
        "--inference_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float32"],
        help="推理数据类型",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="输入 JSON 文件路径（若指定 --scene_id，也可作为生成的输入 JSON 保存路径）",
    )
    parser.add_argument(
        "--scene_id",
        type=str,
        default=None,
        help="从 metadata.csv 中按 scene_id 直接提取场景进行推理",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="3dish",
        help="数据集名称（配合 --scene_id 使用，默认: 3dish）",
    )
    parser.add_argument(
        "--metadata_csv",
        type=str,
        default=None,
        help="metadata CSV 文件路径（配合 --scene_id 使用，不指定则由 --dataset 自动推断）",
    )
    parser.add_argument(
        "--start_step",
        type=int,
        default=0,
        help="起始 step（配合 --scene_id 使用，之前的物体作为 existing_objects）",
    )
    parser.add_argument(
        "--with-object-descriptions",
        action="store_true",
        default=False,
        help="在推理输入中保留物体的 short_desc 描述（默认只使用 category 标签）",
    )
    parser.add_argument(
        "--prepare_input_only",
        action="store_true",
        default=False,
        help="仅从 metadata.csv 生成推理输入 JSON 文件，不加载模型执行推理",
    )
    parser.add_argument(
        "--no_pcd",
        action="store_true",
        help="关闭点云输入（纯文本推理）",
    )
    parser.add_argument(
        "--asset_pointcloud_dir",
        type=str,
        default=None,
        help="资产点云目录（不传则默认同时搜索 Imaginarium + 3D-FUTURE）",
    )
    parser.add_argument(
        "--pointcloud_output_dir",
        type=str,
        default="output/scene_pointclouds",
        help="场景点云输出目录",
    )
    parser.add_argument(
        "--internal_spaces_dir",
        type=str,
        default=None,
        help="收纳空间数据目录路径（不传则不附加 internal_spaces）",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="采样温度",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="推理随机种子；设为负数可禁用固定 seed",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="output/scene_layout.json",
        help="输出结果文件路径",
    )
    parser.add_argument(
        "--only-pointcloud",
        action="store_true",
        default=False,
        help="不包含物体的 center, size, rotation 几何信息，仅基于点云进行推理",
    )

    args = parser.parse_args()

    if args.scene_id:
        inference_input = build_inference_input_from_metadata(
            scene_id=args.scene_id,
            dataset=args.dataset,
            start_step=args.start_step,
            metadata_csv=args.metadata_csv,
            include_object_descriptions=args.with_object_descriptions,
            output_file=args.input_file,
        )
        if args.prepare_input_only:
            return
    elif args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            input_data = json.load(f)
        inference_input = InferenceInput.from_dict(input_data)
    else:
        parser.error("请指定 --input_file 或 --scene_id")

    if not args.model_path:
        parser.error("执行推理必须指定 -m / --model_path")

    inferencer = SceneLayoutInference(
        model_path=args.model_path,
        inference_dtype=args.inference_dtype,
        asset_pointcloud_dir=args.asset_pointcloud_dir,
        output_dir=args.pointcloud_output_dir,
        use_pointcloud=not args.no_pcd,
        internal_spaces_dir=args.internal_spaces_dir,
        include_object_descriptions=args.with_object_descriptions,
        only_pointcloud=args.only_pointcloud,
        seed=args.seed,
    )

    inferencer.init_scene(scene_info=inference_input.scene)

    results = inferencer.iterative_placement(
        placement_tasks=inference_input.placement_tasks,
        temperature=args.temperature,
    )

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    output_data = results.to_dict() if hasattr(results, "to_dict") else results
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    inferencer.log.success(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
