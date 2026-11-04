"""
摆放验证器

全部检测逻辑均直接复用项目底层共享模块（单一来源）：
1. 碰撞检测：复用 ``utils.pointcloud.compute_pointcloud_collision``（点云体素重叠率，与论文评测一致）
2. 悬空检测：复用 ``utils.pointcloud.compute_pointcloud_floating``（底面 10% 点云 + 2D 支撑高度图，与论文评测一致）
3. 边界检测：复用 ``utils.spatial_utils.get_object_boundary_overlap_metrics``（与数据集过滤一致）
4. 高度检测：复用 ``utils.spatial_utils.get_object_z_bounds``
5. 指令关系符合度：复用 ``utils.spatial_utils.check_relative_relation`` / ``check_absolute_relation``
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from config.constants import COLLISION_RATIO_THRESHOLD, DEFAULT_SCENE_HEIGHT
from config.paths import ProjectPaths
from utils.pointcloud import (
    compute_pointcloud_collision,
    compute_pointcloud_floating,
    load_ply_points_cached,
    place_new_object,
    resolve_asset_ply_path,
    sample_floor_pointcloud,
)
from utils.scene_types import (
    SceneObject,
    scene_object_list_to_dicts,
    scene_object_to_dict,
)
from utils.semantic_relations import (
    OVERLAP_EXEMPT_RELATION_TYPES,
    all_relation_target_ids,
)
from utils.spatial_utils import (
    CONTACT_THRESHOLD,
    check_absolute_relation,
    check_relative_relation,
    get_object_boundary_overlap_metrics,
    get_object_z_bounds,
)

logger = logging.getLogger(__name__)


class PlacementValidator:
    """
    摆放验证器

    在推理结果渲染之前快速验证位姿是否合理，所有底层几何与物理判定均与评测/数据过滤模块共用。
    """

    def __init__(
        self,
        boundary_threshold: float = 0.05,
        floating_threshold: float = 0.05,
        collision_ratio_threshold: float = COLLISION_RATIO_THRESHOLD,
        voxel_size: float = 0.02,
        asset_pointcloud_dir: Optional[Union[str, Path, Sequence[Union[str, Path]]]] = None,
    ):
        self.boundary_threshold = boundary_threshold
        self.floating_threshold = floating_threshold
        self.collision_ratio_threshold = collision_ratio_threshold
        self.voxel_size = voxel_size
        self.asset_pointcloud_dir = (
            asset_pointcloud_dir
            if asset_pointcloud_dir is not None
            else ProjectPaths.get_asset_pointcloud_dir("3dish")
        )

    def _load_world_pointcloud(self, obj: Dict[str, Any]) -> np.ndarray:
        """加载资产点云并按物体的 center / rotation / size 变换到世界坐标系。"""
        jid = obj.get("jid")
        if not jid:
            raise ValueError(f"点云物理检测需要 jid，但物体缺少 jid: {obj}")
        ply_path, _ = resolve_asset_ply_path(str(jid), self.asset_pointcloud_dir)
        raw_pts = load_ply_points_cached(str(ply_path))
        return place_new_object(
            raw_pts,
            predicted_center=obj["center"],
            predicted_rotation_deg=obj["rotation"],
            target_size=obj["size"],
        )

    def validate(
        self,
        new_object: Union[SceneObject, Dict[str, Any]],
        existing_objects: Sequence[Union[SceneObject, Dict[str, Any]]],
        boundary: List[List[float]],
        scene_height: float = DEFAULT_SCENE_HEIGHT,
        expected_relations: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        验证物体摆放。

        Returns:
            验证结果字典：valid, errors, warnings, details
        """
        errors = []
        warnings = []
        details = {}
        new_object = scene_object_to_dict(new_object)
        existing_objects = scene_object_list_to_dicts(existing_objects)

        center = new_object.get("center")
        size = new_object.get("size")
        rotation = new_object.get("rotation")

        if center is None or size is None or rotation is None:
            errors.append("missing_pose_info")
            return {
                "valid": False,
                "errors": errors,
                "warnings": warnings,
                "details": {
                    "error": "Missing center or size or rotation in new_object"
                },
            }

        collision_result = self._check_collision(
            new_object, existing_objects, expected_relations
        )
        details["collision"] = collision_result
        if collision_result["has_collision"]:
            errors.append("collision_detected")

        boundary_result = self._check_boundary(new_object, boundary)
        details["boundary"] = boundary_result
        if not boundary_result["within_boundary"]:
            errors.append("out_of_boundary")

        height_result = self._check_height(new_object, scene_height)
        details["height"] = height_result
        if not height_result["valid"]:
            errors.append(height_result.get("reason", "height_violation"))

        floating_result = self._check_floating(
            new_object, existing_objects, boundary
        )
        details["floating"] = floating_result
        if floating_result["is_floating"]:
            warnings.append("object_floating")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "details": details,
        }

    def _check_collision(
        self,
        new_object: Dict[str, Any],
        existing_objects: List[Dict[str, Any]],
        expected_relations: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """基于点云体素重叠率检测碰撞（复用 utils.pointcloud.compute_pointcloud_collision）。"""
        if not existing_objects:
            return {
                "has_collision": False,
                "colliding_objects": [],
                "collision_ratio": 0.0,
            }

        exempt_ids: set[int] = set()
        if expected_relations:
            exempt_ids = set(
                all_relation_target_ids(
                    {"expected_relations": expected_relations},
                    OVERLAP_EXEMPT_RELATION_TYPES,
                )
            )

        new_pts = self._load_world_pointcloud(new_object)
        active_objects: List[Dict[str, Any]] = []
        existing_pts_list: List[np.ndarray] = []

        for obj in existing_objects:
            obj_raw_id = obj.get("id")
            if obj_raw_id is not None and int(obj_raw_id) in exempt_ids:
                continue
            if obj.get("center") is None or obj.get("size") is None or obj.get("rotation") is None:
                continue
            existing_pts_list.append(self._load_world_pointcloud(obj))
            active_objects.append(obj)

        if not existing_pts_list:
            return {
                "has_collision": False,
                "colliding_objects": [],
                "collision_ratio": 0.0,
            }

        res = compute_pointcloud_collision(
            new_pts=new_pts,
            existing_pts_list=existing_pts_list,
            voxel_size=self.voxel_size,
            overlap_ratio_threshold=self.collision_ratio_threshold,
        )

        new_voxel_count = max(res["new_voxel_count"], 1)
        colliding_objects = []
        for obj, overlap_voxels in zip(active_objects, res["per_object_overlaps"]):
            if overlap_voxels <= 0:
                continue
            colliding_objects.append(
                {
                    "object_id": obj.get("id", "?"),
                    "object_desc": obj.get("short_desc", "unknown"),
                    "collision_ratio": round(overlap_voxels / new_voxel_count, 4),
                    "intersection_volume": round(overlap_voxels * (self.voxel_size ** 3), 6),
                }
            )

        return {
            "has_collision": res["has_collision"],
            "colliding_objects": colliding_objects,
            "collision_ratio": res["overlap_ratio"],
        }

    def _check_boundary(
        self,
        new_object: Dict[str, Any],
        boundary: List[List[float]],
    ) -> Dict[str, Any]:
        """边界检测（复用 utils.spatial_utils.get_object_boundary_overlap_metrics）。"""
        if not boundary or len(boundary) < 3:
            return {"within_boundary": True, "overflow_area": 0.0, "outside_ratio": 0.0}

        metrics = get_object_boundary_overlap_metrics(new_object, boundary)
        outside_ratio = float(metrics["outside_ratio"])
        overflow_area = float(metrics["outside_area"])
        within = outside_ratio <= self.boundary_threshold

        return {
            "within_boundary": bool(within),
            "overflow_area": overflow_area,
            "outside_ratio": outside_ratio,
            "footprint_area": float(metrics["footprint_area"]),
        }

    def _check_height(
        self,
        new_object: Dict[str, Any],
        scene_height: float,
    ) -> Dict[str, Any]:
        """高度检测（复用 utils.spatial_utils.get_object_z_bounds）。"""
        bottom_z, top_z = get_object_z_bounds(new_object)

        if bottom_z < -CONTACT_THRESHOLD:
            return {
                "valid": False,
                "reason": "below_ground",
                "object_top": top_z,
                "object_bottom": bottom_z,
                "scene_height": scene_height,
            }

        if top_z > scene_height + CONTACT_THRESHOLD:
            return {
                "valid": False,
                "reason": "exceeds_ceiling",
                "object_top": top_z,
                "object_bottom": bottom_z,
                "scene_height": scene_height,
            }

        return {
            "valid": True,
            "object_top": top_z,
            "object_bottom": bottom_z,
            "scene_height": scene_height,
        }

    def _check_floating(
        self,
        new_object: Dict[str, Any],
        existing_objects: List[Dict[str, Any]],
        boundary: List[List[float]],
    ) -> Dict[str, Any]:
        """悬空与支撑检测（复用 utils.pointcloud.compute_pointcloud_floating）。"""
        bottom_z, _ = get_object_z_bounds(new_object)
        obj_height = float(new_object["size"][2])
        pred_z = float(new_object["center"][2])

        # 地面接触快速通道（与 reward/floating.py 一致）
        if abs(pred_z - obj_height / 2.0) <= self.floating_threshold:
            return {
                "is_floating": False,
                "is_sinking": False,
                "support_ratio": 1.0,
                "median_gap": 0.0,
                "ground_contact": True,
                "bottom_z": bottom_z,
            }

        new_pts = self._load_world_pointcloud(new_object)
        floor_pts = sample_floor_pointcloud(boundary, seed=42)
        existing_pts_list = [
            self._load_world_pointcloud(obj)
            for obj in existing_objects
            if obj.get("center") is not None
            and obj.get("size") is not None
            and obj.get("rotation") is not None
            and obj.get("jid")
        ]

        res = compute_pointcloud_floating(
            new_pts=new_pts,
            floor_pts=floor_pts,
            existing_pts_list=existing_pts_list,
            voxel_size=self.voxel_size,
            gap_threshold=self.floating_threshold,
        )
        return {
            **res,
            "bottom_z": bottom_z,
        }

    def compute_violation_score(self, validation_result: Dict[str, Any]) -> float:
        """计算违规分数（1.0 = 完全合法，0.0 = 严重违规）。"""
        if validation_result.get("valid"):
            return 1.0

        score = 1.0
        details = validation_result.get("details", {})

        collision = details.get("collision", {})
        collision_ratio = collision.get("collision_ratio", 0.0)
        num_collisions = len(collision.get("colliding_objects", []))

        if collision_ratio > 0:
            score -= min(0.5, collision_ratio * 0.5)
        else:
            score -= num_collisions * 0.15

        boundary = details.get("boundary", {})
        overflow_area = boundary.get("overflow_area", 0.0)
        if overflow_area > 0:
            score -= min(0.3, overflow_area * 0.5)

        height = details.get("height", {})
        if not height.get("valid", True):
            reason = height.get("reason", "")
            if reason == "below_ground":
                bottom_z = height.get("object_bottom", 0)
                score -= min(0.3, abs(bottom_z) * 0.3)
            else:
                object_top = height.get("object_top", 0)
                scene_height = height.get("scene_height", DEFAULT_SCENE_HEIGHT)
                exceed = max(0, object_top - scene_height)
                score -= min(0.3, exceed * 0.3)

        floating = details.get("floating", {})
        if floating.get("is_floating", False):
            score -= 0.05

        return max(0.0, score)

    def check_instruction_compliance(
        self,
        new_object: Union[SceneObject, Dict[str, Any]],
        existing_objects: Sequence[Union[SceneObject, Dict[str, Any]]],
        boundary: List[List[float]],
        expected_relations: Dict[str, Any],
    ) -> Dict[str, Any]:
        """检查摆放结果是否符合指令中的期望关系（复用 utils.spatial_utils）。"""
        if not expected_relations:
            return {
                "compliant": True,
                "relative_results": [],
                "absolute_results": [],
                "compliance_ratio": 1.0,
            }

        new_object = scene_object_to_dict(new_object)
        existing_objects = scene_object_list_to_dicts(existing_objects)

        relative_relations = expected_relations.get("relative_relations", [])
        absolute_relations = expected_relations.get("absolute_relations", [])

        obj_map = {obj.get("id"): obj for obj in existing_objects}

        relative_results = []
        for rel in relative_relations:
            rel_type = rel.get("type", "")
            target_id = rel.get("target_id")

            target_obj = obj_map.get(target_id)
            if target_obj is None:
                relative_results.append(
                    {
                        "type": rel_type,
                        "target_id": target_id,
                        "satisfied": False,
                        "reason": "target_object_not_found",
                    }
                )
                continue

            satisfied = check_relative_relation(new_object, target_obj, rel_type)
            relative_results.append(
                {
                    "type": rel_type,
                    "target_id": target_id,
                    "satisfied": bool(satisfied),
                }
            )

        absolute_results = []
        for rel_type in absolute_relations:
            satisfied = check_absolute_relation(new_object, boundary, rel_type)
            absolute_results.append(
                {
                    "type": rel_type,
                    "satisfied": bool(satisfied),
                }
            )

        total = len(relative_results) + len(absolute_results)
        satisfied_count = sum(1 for r in relative_results if r["satisfied"])
        satisfied_count += sum(1 for r in absolute_results if r["satisfied"])

        compliance_ratio = satisfied_count / total if total > 0 else 1.0
        compliant = satisfied_count == total

        return {
            "compliant": compliant,
            "relative_results": relative_results,
            "absolute_results": absolute_results,
            "compliance_ratio": compliance_ratio,
        }
