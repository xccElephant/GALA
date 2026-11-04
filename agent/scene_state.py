"""
场景状态管理器

管理场景生成过程中的状态，包括：
1. 当前场景信息（boundary, existing_objects 等）
2. 生成历史（支持回滚）
3. 上下文信息（用于生成 prompt）
"""

import json
import copy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.constants import DEFAULT_SCENE_HEIGHT
from utils.scene_types import Pose, Scene, SceneObject, ensure_pose, ensure_scene


@dataclass
class PlacementAttempt:
    """单次摆放尝试记录"""
    attempt_type: str  # "inference" (推理重试) 或 "instruction" (指令重试)
    instruction: str
    instruction_thinking: str
    expected_relations: Optional[Dict] = None
    parent_id: Optional[int] = None
    pose: Optional[Pose] = None
    raw_output: str = ""
    validation_result: Optional[Dict] = None
    compliance_result: Optional[Dict] = None
    success: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class PlacementRecord:
    """单个物体的完整摆放记录"""
    object_id: int
    short_desc: str
    jid: str
    instruction: str
    pose: Optional[Pose]
    size: List[float]
    success: bool
    partial_success: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    raw_output: str = ""
    validation_result: Optional[Dict] = None
    compliance_result: Optional[Dict] = None
    instruction_thinking: str = ""
    expected_relations: Optional[Dict] = None
    parent_id: Optional[int] = None
    attempts: List[PlacementAttempt] = field(default_factory=list)


@dataclass
class SceneSnapshot:
    """场景快照（用于回滚）"""
    scene_info: Dict[str, Any]
    existing_objects: List[SceneObject]
    placement_history: List[PlacementRecord]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class SceneStateManager:
    """
    场景状态管理器
    
    管理场景的当前状态和历史，支持回滚操作。
    """
    
    def __init__(
        self,
        scene_id: str = "scene",
        boundary: Optional[List[List[float]]] = None,
        scene_height: float = DEFAULT_SCENE_HEIGHT,
        scene_description: str = "",
        scene_type: str = "room",
        max_history: int = 500,
    ):
        self.scene_id = scene_id
        self.boundary = boundary or []
        self.scene_height = scene_height
        self.scene_description = scene_description
        self.scene_type = scene_type
        self.max_history = max_history
        
        self.existing_objects: List[SceneObject] = []
        self.placement_history: List[PlacementRecord] = []

        self._snapshots: List[SceneSnapshot] = []
        self._save_snapshot()
    
    @property
    def scene_info(self) -> Dict[str, Any]:
        """获取当前场景信息"""
        return self.scene_model.to_dict()

    @property
    def scene_model(self) -> Scene:
        """获取统一场景模型视图"""
        return Scene(
            scene_id=self.scene_id,
            scene_description=self.scene_description,
            boundary=copy.deepcopy(self.boundary),
            scene_height=self.scene_height,
            objects=list(self.existing_objects),
            scene_type=self.scene_type,
        )
    
    def add_object(
        self,
        object_id: int,
        short_desc: str,
        jid: str,
        center: List[float],
        rotation: List[float],
        size: List[float],
        category: str = "",
        instruction: str = "",
        instruction_thinking: str = "",
        raw_output: str = "",
        validation_result: Optional[Dict] = None,
        compliance_result: Optional[Dict] = None,
        expected_relations: Optional[Dict] = None,
        parent_id: Optional[int] = None,
        attempts: Optional[List[PlacementAttempt]] = None,
        partial_success: bool = False,
        save_snapshot: bool = True,
    ) -> PlacementRecord:
        """添加物体到场景"""
        pose = Pose(center=center, rotation=rotation)
        scene_object = SceneObject(
            id=object_id,
            short_desc=short_desc,
            size=size,
            jid=jid,
            pose=pose,
            category=category,
            parent_id=parent_id,
        )
        
        self.existing_objects.append(scene_object)
        
        record = PlacementRecord(
            object_id=object_id,
            short_desc=short_desc,
            jid=jid,
            instruction=instruction,
            pose=pose,
            size=size,
            success=not partial_success,
            partial_success=partial_success,
            raw_output=raw_output,
            validation_result=validation_result,
            compliance_result=compliance_result,
            instruction_thinking=instruction_thinking,
            expected_relations=expected_relations,
            parent_id=parent_id,
            attempts=attempts or [],
        )
        self.placement_history.append(record)

        if save_snapshot:
            self._save_snapshot()
        
        return record
    
    def record_failure(
        self,
        object_id: int,
        short_desc: str,
        jid: str,
        size: List[float],
        instruction: str = "",
        instruction_thinking: str = "",
        raw_output: str = "",
        parent_id: Optional[int] = None,
        error: str = "",
    ) -> PlacementRecord:
        """记录失败的摆放尝试"""
        record = PlacementRecord(
            object_id=object_id,
            short_desc=short_desc,
            jid=jid,
            instruction=instruction,
            pose=None,
            size=size,
            success=False,
            raw_output=raw_output,
            validation_result={"error": error},
            instruction_thinking=instruction_thinking,
            parent_id=parent_id,
        )
        self.placement_history.append(record)
        
        return record
    
    def get_object_by_id(self, object_id: int) -> Optional[SceneObject]:
        """根据 ID 获取物体"""
        for obj in self.existing_objects:
            if obj.id == object_id:
                return obj
        return None
    
    def get_next_object_id(self) -> int:
        """获取下一个可用的物体 ID"""
        if not self.existing_objects:
            return 0
        return max(obj.id for obj in self.existing_objects) + 1
    
    def _save_snapshot(self):
        """保存当前状态快照"""
        snapshot = SceneSnapshot(
            scene_info={
                "scene_id": self.scene_id,
                "scene_description": self.scene_description,
                "boundary": copy.deepcopy(self.boundary),
                "scene_height": self.scene_height,
            },
            existing_objects=copy.deepcopy(self.existing_objects),
            placement_history=copy.deepcopy(self.placement_history),
        )
        self._snapshots.append(snapshot)
        
        if len(self._snapshots) > self.max_history:
            self._snapshots = self._snapshots[-self.max_history:]
    
    def rollback(self, steps: int = 1) -> bool:
        """回滚到之前的状态"""
        if steps <= 0 or steps >= len(self._snapshots):
            return False
        
        target_idx = len(self._snapshots) - 1 - steps
        snapshot = self._snapshots[target_idx]
        
        self.existing_objects = copy.deepcopy(snapshot.existing_objects)
        self.placement_history = copy.deepcopy(snapshot.placement_history)
        
        self._snapshots = self._snapshots[:target_idx + 1]
        
        return True
    
    def remove_last_object(self) -> Optional[SceneObject]:
        """移除最后添加的物体"""
        if not self.existing_objects:
            return None
        
        removed = self.existing_objects.pop()
        self._save_snapshot()
        return removed
    
    def clear(self):
        """清空场景（保留边界等基本信息）"""
        self.existing_objects = []
        self.placement_history = []
        self._snapshots = []
        self._save_snapshot()
    
    def get_summary(self) -> Dict[str, Any]:
        """获取场景摘要"""
        successful = [r for r in self.placement_history if r.success]
        partial_success = [r for r in self.placement_history if not r.success and r.partial_success]
        failed = [r for r in self.placement_history if not r.success and not r.partial_success]
        
        total_attempts = sum(len(r.attempts) for r in self.placement_history)
        
        return {
            "scene_id": self.scene_id,
            "scene_type": self.scene_type,
            "total_objects": len(self.existing_objects),
            "total_tasks": len(self.placement_history),
            "successful": len(successful),
            "partial_success": len(partial_success),
            "failed": len(failed),
            "total_attempts": total_attempts,
            "success_rate": len(successful) / max(len(self.placement_history), 1),
            "snapshot_count": len(self._snapshots),
        }
    
    def save(self, path: str):
        """保存状态到文件。

        采用与 pipeline 生成的 ``layout.json`` 一致的 envelope：场景放在 ``final_scene`` 下，
        ``placement_history`` 平铺在顶层。这样 generate / edit 产出的 JSON 结构统一，
        可直接被 ``load()``、可视化（``scene_from_external_dict``）与物理修正复用。
        generate 的 layout.json 只是在此基础上额外带了若干 run 统计字段（user_input/总数等）。
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "final_scene": self.scene_model.to_dict(object_field="objects"),
            "placement_history": [
                self._record_to_dict(r) for r in self.placement_history
            ],
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"[SceneStateManager] Saved state to {save_path}")

    @staticmethod
    def _pose_to_dict(pose: Optional[Pose]) -> Optional[Dict[str, Any]]:
        if pose is None:
            return None
        return pose.to_dict()

    @staticmethod
    def _attempt_to_dict(a: PlacementAttempt) -> Dict[str, Any]:
        return {
            "attempt_type": a.attempt_type,
            "instruction": a.instruction,
            "instruction_thinking": a.instruction_thinking,
            "expected_relations": a.expected_relations,
            "parent_id": a.parent_id,
            "pose": SceneStateManager._pose_to_dict(a.pose),
            "raw_output": a.raw_output,
            "validation_result": a.validation_result,
            "compliance_result": a.compliance_result,
            "success": a.success,
            "timestamp": a.timestamp,
        }

    @staticmethod
    def _record_to_dict(r: PlacementRecord) -> Dict[str, Any]:
        return {
            "object_id": r.object_id,
            "short_desc": r.short_desc,
            "jid": r.jid,
            "instruction": r.instruction,
            "instruction_thinking": r.instruction_thinking,
            "expected_relations": r.expected_relations,
            "parent_id": r.parent_id,
            "pose": SceneStateManager._pose_to_dict(r.pose),
            "size": r.size,
            "success": r.success,
            "partial_success": r.partial_success,
            "timestamp": r.timestamp,
            "raw_output": r.raw_output,
            "validation_result": r.validation_result,
            "compliance_result": r.compliance_result,
            "attempts": [
                SceneStateManager._attempt_to_dict(a) for a in r.attempts
            ],
        }

    @staticmethod
    def _pose_from_dict(data: Optional[Dict[str, Any]]) -> Optional[Pose]:
        if data is None:
            return None
        return ensure_pose(data)

    def load(self, path: str):
        """从文件加载状态。

        统一 envelope：场景在 ``final_scene`` 下（save() 与 generate 的 layout.json 均如此）。
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "final_scene" not in data:
            raise KeyError(
                f"场景文件缺少 'final_scene' envelope: {path}"
                "（save() 与 generate 的 layout.json 均使用该结构）"
            )
        scene = ensure_scene(data["final_scene"])
        self.scene_id = scene.scene_id
        self.scene_type = scene.scene_type
        self.scene_description = scene.scene_description
        self.boundary = scene.boundary
        self.scene_height = scene.scene_height
        self.existing_objects = list(scene.objects)

        self.placement_history = []
        for r in data.get("placement_history", []):
            attempts = []
            for a in r.get("attempts", []):
                attempt = PlacementAttempt(
                    attempt_type=a["attempt_type"],
                    instruction=a["instruction"],
                    instruction_thinking=a.get("instruction_thinking", ""),
                    expected_relations=a.get("expected_relations"),
                    parent_id=a.get("parent_id"),
                    pose=self._pose_from_dict(a.get("pose")),
                    raw_output=a.get("raw_output", ""),
                    validation_result=a.get("validation_result"),
                    compliance_result=a.get("compliance_result"),
                    success=a["success"],
                    timestamp=a.get("timestamp", ""),
                )
                attempts.append(attempt)
            
            record = PlacementRecord(
                object_id=r["object_id"],
                short_desc=r["short_desc"],
                jid=r["jid"],
                instruction=r["instruction"],
                pose=self._pose_from_dict(r.get("pose")),
                size=r["size"],
                success=r["success"],
                partial_success=r.get("partial_success", False),
                timestamp=r.get("timestamp", ""),
                raw_output=r.get("raw_output", ""),
                validation_result=r.get("validation_result"),
                compliance_result=r.get("compliance_result"),
                instruction_thinking=r.get("instruction_thinking", ""),
                expected_relations=r.get("expected_relations"),
                parent_id=r.get("parent_id"),
                attempts=attempts,
            )
            self.placement_history.append(record)
        
        self._snapshots = []
        self._save_snapshot()
        
        print(f"[SceneStateManager] Loaded state from {path}")
    
    def get_context_for_agent(self) -> str:
        """生成给 Agent 的上下文描述"""
        lines = [
            f"Scene ID: {self.scene_id}",
            f"Scene Type: {self.scene_type}",
            f"Description: {self.scene_description}",
            f"Room Size: {self._get_room_size_str()}",
            f"Objects Placed: {len(self.existing_objects)}",
            "",
            "Current Objects:",
        ]
        
        if self.existing_objects:
            for obj in self.existing_objects:
                center = obj.center
                lines.append(
                    f"  - ID {obj.id}: {obj.short_desc} "
                    f"at ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})"
                )
        else:
            lines.append("  (empty scene)")
        
        return "\n".join(lines)
    
    def _get_room_size_str(self) -> str:
        """获取房间尺寸描述"""
        if not self.boundary:
            return "unknown"
        
        xs = [p[0] for p in self.boundary]
        ys = [p[1] for p in self.boundary]
        width = max(xs) - min(xs)
        depth = max(ys) - min(ys)
        
        return f"{width:.1f}m x {depth:.1f}m x {self.scene_height:.1f}m"
