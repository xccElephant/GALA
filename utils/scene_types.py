from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union


def _copy_float_list(values: Sequence[float], *, name: str) -> List[float]:
    if len(values) != 3:
        raise ValueError(f"{name} must contain 3 values, got {len(values)}")
    return [float(values[0]), float(values[1]), float(values[2])]


@dataclass
class Pose:
    center: List[float]
    rotation: List[float]

    def __post_init__(self) -> None:
        self.center = _copy_float_list(self.center, name="center")
        self.rotation = _copy_float_list(self.rotation, name="rotation")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Pose":
        if "center" not in data:
            raise KeyError("Pose data missing 'center'")
        if "rotation" not in data:
            raise KeyError("Pose data missing 'rotation'")
        return cls(center=data["center"], rotation=data["rotation"])

    def to_dict(self) -> Dict[str, List[float]]:
        return {
            "center": copy.deepcopy(self.center),
            "rotation": copy.deepcopy(self.rotation),
        }


@dataclass
class SceneObject:
    id: int
    short_desc: str
    size: List[float]
    jid: Optional[str] = None
    pose: Optional[Pose] = None
    desc: str = ""
    category: str = ""
    parent_id: Optional[int] = None
    internal_spaces: Optional[List[Dict[str, Any]]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id = int(self.id)
        self.size = _copy_float_list(self.size, name="size")
        if self.pose is not None and not isinstance(self.pose, Pose):
            raise TypeError("pose must be a Pose or None")
        if self.parent_id is not None:
            self.parent_id = int(self.parent_id)
        self.extra = copy.deepcopy(self.extra)
        if self.internal_spaces is not None:
            self.internal_spaces = copy.deepcopy(self.internal_spaces)

    @property
    def center(self) -> Optional[List[float]]:
        if self.pose is None:
            return None
        return copy.deepcopy(self.pose.center)

    @property
    def rotation(self) -> Optional[List[float]]:
        if self.pose is None:
            return None
        return copy.deepcopy(self.pose.rotation)

    def require_pose(self) -> Pose:
        if self.pose is None:
            raise ValueError(f"SceneObject(id={self.id}, short_desc={self.short_desc}) has no pose")
        return self.pose

    def copy_with_pose(self, pose: Pose) -> "SceneObject":
        return SceneObject(
            id=self.id,
            short_desc=self.short_desc,
            size=self.size,
            jid=self.jid,
            pose=Pose(center=pose.center, rotation=pose.rotation),
            desc=self.desc,
            category=self.category,
            parent_id=self.parent_id,
            internal_spaces=self.internal_spaces,
            extra=self.extra,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SceneObject":
        for field_name in ("id", "short_desc", "size"):
            if field_name not in data:
                raise KeyError(f"SceneObject data missing '{field_name}'")

        pose = None
        if "pose" in data and data["pose"] is not None:
            pose = Pose.from_dict(data["pose"])
        else:
            has_center = "center" in data
            has_rotation = "rotation" in data
            if has_center != has_rotation:
                raise KeyError("SceneObject pose must include both 'center' and 'rotation'")
            if has_center:
                pose = Pose(center=data["center"], rotation=data["rotation"])

        known_keys = {
            "id",
            "short_desc",
            "size",
            "jid",
            "pose",
            "center",
            "rotation",
            "desc",
            "category",
            "parent_id",
            "internal_spaces",
        }
        extra = {key: copy.deepcopy(value) for key, value in data.items() if key not in known_keys}
        return cls(
            id=data["id"],
            short_desc=data["short_desc"],
            size=data["size"],
            jid=data["jid"] if "jid" in data else None,
            pose=pose,
            desc=data["desc"] if "desc" in data else "",
            category=data["category"] if "category" in data else "",
            parent_id=int(data["parent_id"]) if data.get("parent_id") is not None else None,
            internal_spaces=data["internal_spaces"] if "internal_spaces" in data else None,
            extra=extra,
        )

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "id": self.id,
            "short_desc": self.short_desc,
            "size": copy.deepcopy(self.size),
        }
        if self.jid is not None:
            data["jid"] = self.jid
        if self.pose is not None:
            data.update(self.pose.to_dict())
        if self.desc:
            data["desc"] = self.desc
        if self.category:
            data["category"] = self.category
        if self.parent_id is not None:
            data["parent_id"] = self.parent_id
        if self.internal_spaces is not None:
            data["internal_spaces"] = copy.deepcopy(self.internal_spaces)
        data.update(copy.deepcopy(self.extra))
        return data


@dataclass
class Scene:
    scene_id: str
    boundary: List[List[float]]
    scene_height: float
    scene_description: str = ""
    objects: List[SceneObject] = field(default_factory=list)
    scene_type: str = "room"
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.boundary = copy.deepcopy(self.boundary)
        self.scene_height = float(self.scene_height)
        self.objects = [ensure_scene_object(obj) for obj in self.objects]
        self.extra = copy.deepcopy(self.extra)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Scene":
        if "scene_id" not in data:
            raise KeyError("Scene data missing 'scene_id'")
        if "boundary" not in data:
            raise KeyError("Scene data missing 'boundary'")
        if "existing_objects" in data:
            raw_objects = data["existing_objects"]
        elif "objects" in data:
            raw_objects = data["objects"]
        else:
            raw_objects = []

        known_keys = {
            "scene_id",
            "boundary",
            "scene_height",
            "scene_description",
            "existing_objects",
            "objects",
            "scene_type",
        }
        extra = {key: copy.deepcopy(value) for key, value in data.items() if key not in known_keys}
        return cls(
            scene_id=data["scene_id"],
            boundary=data["boundary"],
            scene_height=data["scene_height"] if "scene_height" in data else 2.6,
            scene_description=data["scene_description"] if "scene_description" in data else "",
            objects=ensure_scene_object_list(raw_objects),
            scene_type=data["scene_type"] if "scene_type" in data else "room",
            extra=extra,
        )

    def to_dict(self, *, object_field: str = "existing_objects") -> Dict[str, Any]:
        if object_field not in {"existing_objects", "objects"}:
            raise ValueError(f"Unsupported object_field: {object_field}")
        data: Dict[str, Any] = {
            "scene_id": self.scene_id,
            "scene_description": self.scene_description,
            "boundary": copy.deepcopy(self.boundary),
            "scene_height": self.scene_height,
            object_field: [obj.to_dict() for obj in self.objects],
        }
        if self.scene_type:
            data["scene_type"] = self.scene_type
        data.update(copy.deepcopy(self.extra))
        return data

    def copy_with_objects(self, objects: Sequence[Union["SceneObject", Dict[str, Any]]]) -> "Scene":
        return Scene(
            scene_id=self.scene_id,
            boundary=self.boundary,
            scene_height=self.scene_height,
            scene_description=self.scene_description,
            objects=ensure_scene_object_list(objects),
            scene_type=self.scene_type,
            extra=self.extra,
        )

    def add_object(self, scene_object: Union["SceneObject", Dict[str, Any]]) -> None:
        self.objects.append(ensure_scene_object(scene_object))


@dataclass
class PlacementRequest:
    id: int
    short_desc: str
    size: List[float]
    jid: Optional[str]
    instruction: str
    category: str = ""
    step: Optional[int] = None
    new_object_id: Optional[int] = None
    reference_objects: List[SceneObject] = field(default_factory=list)
    gt_pose: Optional[Pose] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id = int(self.id)
        self.size = _copy_float_list(self.size, name="size")
        if self.new_object_id is not None:
            self.new_object_id = int(self.new_object_id)
        if self.step is not None:
            self.step = int(self.step)
        self.reference_objects = [ensure_scene_object(obj) for obj in self.reference_objects]
        self.extra = copy.deepcopy(self.extra)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlacementRequest":
        for field_name in ("id", "short_desc", "size", "instruction"):
            if field_name not in data:
                raise KeyError(f"PlacementRequest data missing '{field_name}'")

        gt_pose = None
        if "gt_pose" in data and data["gt_pose"] is not None:
            gt_pose = Pose.from_dict(data["gt_pose"])

        known_keys = {
            "id",
            "short_desc",
            "size",
            "jid",
            "instruction",
            "category",
            "step",
            "new_object_id",
            "reference_objects",
            "gt_pose",
        }
        extra = {key: copy.deepcopy(value) for key, value in data.items() if key not in known_keys}
        return cls(
            id=data["id"],
            short_desc=data["short_desc"],
            size=data["size"],
            jid=data["jid"] if "jid" in data else None,
            instruction=data["instruction"],
            category=data["category"] if "category" in data else "",
            step=data["step"] if "step" in data else None,
            new_object_id=data["new_object_id"] if "new_object_id" in data else None,
            reference_objects=ensure_scene_object_list(data["reference_objects"]) if "reference_objects" in data else [],
            gt_pose=gt_pose,
            extra=extra,
        )

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "id": self.id,
            "short_desc": self.short_desc,
            "size": copy.deepcopy(self.size),
            "instruction": self.instruction,
        }
        if self.jid is not None:
            data["jid"] = self.jid
        if self.category:
            data["category"] = self.category
        if self.step is not None:
            data["step"] = self.step
        if self.new_object_id is not None:
            data["new_object_id"] = self.new_object_id
        if self.reference_objects:
            data["reference_objects"] = [obj.to_dict() for obj in self.reference_objects]
        if self.gt_pose is not None:
            data["gt_pose"] = self.gt_pose.to_dict()
        data.update(copy.deepcopy(self.extra))
        return data

    def copy_with(
        self,
        *,
        id: Optional[int] = None,
        new_object_id: Optional[int] = None,
        reference_objects: Optional[Sequence[Union[SceneObject, Dict[str, Any]]]] = None,
    ) -> "PlacementRequest":
        return PlacementRequest(
            id=self.id if id is None else id,
            short_desc=self.short_desc,
            size=self.size,
            jid=self.jid,
            instruction=self.instruction,
            category=self.category,
            step=self.step,
            new_object_id=self.new_object_id if new_object_id is None else new_object_id,
            reference_objects=self.reference_objects if reference_objects is None else list(reference_objects),
            gt_pose=None if self.gt_pose is None else Pose(
                center=self.gt_pose.center,
                rotation=self.gt_pose.rotation,
            ),
            extra=self.extra,
        )

    def to_scene_object(self, pose: Union[Pose, Dict[str, Any]], *, object_id: Optional[int] = None) -> SceneObject:
        normalized_pose = ensure_pose(pose)
        return SceneObject(
            id=self.id if object_id is None else object_id,
            short_desc=self.short_desc,
            size=self.size,
            jid=self.jid,
            pose=normalized_pose,
            category=self.category,
            extra=self.extra,
        )


def ensure_pose(pose: Union[Pose, Dict[str, Any]]) -> Pose:
    if isinstance(pose, Pose):
        return pose
    return Pose.from_dict(pose)


def parse_pose(
    text: Any, *, raise_json_error: bool = False
) -> Optional[Pose]:
    """从模型输出的文本中解析 6D pose [x, y, z, roll, pitch, yaw]。"""
    try:
        data = text if isinstance(text, (list, dict)) else json.loads(text)
        if isinstance(data, list) and len(data) == 6:
            return Pose(center=data[:3], rotation=data[3:])
        return None
    except json.JSONDecodeError:
        if raise_json_error:
            raise
        print("[WARNING] Failed to parse pose from text: invalid JSON")
        return None
    except Exception as e:
        print(f"[WARNING] Failed to parse pose from text: {e}")
        return None


def ensure_scene_object(scene_object: Union[SceneObject, Dict[str, Any]]) -> SceneObject:
    if isinstance(scene_object, SceneObject):
        return scene_object
    return SceneObject.from_dict(scene_object)


def scene_object_to_dict(scene_object: Union[SceneObject, Dict[str, Any]]) -> Dict[str, Any]:
    """Return a detached legacy object dict from a SceneObject or compatible dict."""
    return ensure_scene_object(scene_object).to_dict()


def ensure_scene_object_list(
    objects: Optional[Sequence[Union[SceneObject, Dict[str, Any]]]]
) -> List[SceneObject]:
    if objects is None:
        return []
    return [ensure_scene_object(obj) for obj in objects]


def scene_object_list_to_dicts(
    objects: Optional[Sequence[Union[SceneObject, Dict[str, Any]]]]
) -> List[Dict[str, Any]]:
    if objects is None:
        return []
    return [scene_object_to_dict(obj) for obj in objects]


def ensure_scene(scene: Union[Scene, Dict[str, Any]]) -> Scene:
    if isinstance(scene, Scene):
        return scene
    return Scene.from_dict(scene)


def ensure_placement_request(
    request: Union[PlacementRequest, Dict[str, Any]]
) -> PlacementRequest:
    if isinstance(request, PlacementRequest):
        return request
    return PlacementRequest.from_dict(request)


# ---------------------------------------------------------------------------
# Inference result types
# ---------------------------------------------------------------------------

@dataclass
class ModelResponse:
    """底层模型 generate_response 的返回类型。"""
    success: bool
    response: str = ""
    response_raw: str = ""
    error: Optional[str] = None


@dataclass
class InferenceResult:
    """单物体自回归推理的返回类型。"""
    success: bool
    raw_output: str = ""
    pose: Optional[Pose] = None
    error: Optional[str] = None
    inference_time: float = 0.0
    updated_scene_pointcloud_path: Optional[str] = None


# ---------------------------------------------------------------------------
# External I/O envelopes and summaries
# ---------------------------------------------------------------------------

def scene_from_external_dict(data: Dict[str, Any]) -> Scene:
    """Parse any supported external scene envelope into the canonical Scene model.

    Supported envelopes:
    - Agent/layout output: {"final_scene": {...}}
    - Inference input: {"scene_info": {...}}
    - Converted/layout scene: {"scene_id": ..., "objects": [...]}
    - Training/inference scene: {"scene_id": ..., "existing_objects": [...]}
    """
    if "final_scene" in data:
        scene_data = data["final_scene"]
        if not isinstance(scene_data, dict):
            raise TypeError("final_scene must be a JSON object")
        return Scene.from_dict(scene_data)

    if "scene_info" in data:
        scene_data = data["scene_info"]
        if not isinstance(scene_data, dict):
            raise TypeError("scene_info must be a JSON object")
        return Scene.from_dict(scene_data)

    has_scene_identity = "scene_id" in data and "boundary" in data
    has_objects = "objects" in data or "existing_objects" in data
    if has_scene_identity and has_objects:
        return Scene.from_dict(data)

    raise ValueError(
        "Unsupported scene envelope: expected final_scene, scene_info, "
        "or a scene object with scene_id/boundary and objects/existing_objects"
    )


@dataclass
class InferenceInput:
    """External inference JSON envelope normalized to typed fields."""
    scene: Scene
    placement_tasks: List[PlacementRequest] = field(default_factory=list)
    layout_mode: str = "autoregressive"
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.scene = ensure_scene(self.scene)
        self.placement_tasks = [
            ensure_placement_request(task) for task in self.placement_tasks
        ]
        self.layout_mode = str(self.layout_mode)
        self.extra = copy.deepcopy(self.extra)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InferenceInput":
        if "scene_info" not in data:
            raise KeyError("InferenceInput data missing 'scene_info'")

        known_keys = {"scene_info", "placement_tasks", "layout_mode"}
        extra = {key: copy.deepcopy(value) for key, value in data.items() if key not in known_keys}
        raw_tasks = data["placement_tasks"] if "placement_tasks" in data else []
        if raw_tasks is None:
            raw_tasks = []
        if not isinstance(raw_tasks, list):
            raise TypeError("placement_tasks must be a list")

        return cls(
            scene=Scene.from_dict(data["scene_info"]),
            placement_tasks=[ensure_placement_request(task) for task in raw_tasks],
            layout_mode=data["layout_mode"] if "layout_mode" in data else "autoregressive",
            extra=extra,
        )

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "scene_info": self.scene.to_dict(object_field="existing_objects"),
            "placement_tasks": [task.to_dict() for task in self.placement_tasks],
            "layout_mode": self.layout_mode,
        }
        data.update(copy.deepcopy(self.extra))
        return data


@dataclass
class AutoregressiveMetadata:
    """Typed metadata row that serializes to the existing metadata CSV schema."""
    sample_id: int
    scene_id: str
    scene_description: str
    boundary: List[List[float]]
    scene_height: float
    step: int
    new_object_id: int
    expected_relations: Dict[str, Any]
    existing_objects: List[SceneObject]
    new_object: SceneObject
    instruction: str = ""
    scene_pointcloud_path: Optional[str] = None
    new_object_pointcloud_path: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sample_id = int(self.sample_id)
        self.scene_height = float(self.scene_height)
        self.step = int(self.step)
        self.new_object_id = int(self.new_object_id)
        self.boundary = copy.deepcopy(self.boundary)
        self.expected_relations = copy.deepcopy(self.expected_relations)
        self.existing_objects = ensure_scene_object_list(self.existing_objects)
        self.new_object = ensure_scene_object(self.new_object)
        self.extra = copy.deepcopy(self.extra)

    @classmethod
    def from_sample(
        cls,
        sample: Dict[str, Any],
        sample_id: int,
        scene_pointcloud_path: Optional[str] = None,
        new_object_pointcloud_path: Optional[str] = None,
    ) -> "AutoregressiveMetadata":
        new_object_data = copy.deepcopy(sample["new_object"])
        target_pose = sample["target_pose"]
        new_object_data["center"] = target_pose["center"]
        new_object_data["rotation"] = target_pose["rotation"]

        return cls(
            sample_id=sample_id,
            scene_id=sample["scene_id"],
            scene_description=sample["scene_description"],
            boundary=sample["boundary"],
            scene_height=sample["scene_height"],
            step=sample["step"],
            new_object_id=sample["new_object"]["id"],
            instruction=sample["instruction"],
            expected_relations=(
                copy.deepcopy(sample["expected_relations"])
                if "expected_relations" in sample else {}
            ),
            existing_objects=ensure_scene_object_list(sample["existing_objects"]),
            new_object=SceneObject.from_dict(new_object_data),
            scene_pointcloud_path=scene_pointcloud_path,
            new_object_pointcloud_path=new_object_pointcloud_path,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AutoregressiveMetadata":
        required = (
            "sample_id",
            "scene_id",
            "scene_description",
            "boundary",
            "scene_height",
            "step",
            "new_object_id",
            "existing_objects",
            "new_object",
        )
        for key in required:
            if key not in data:
                raise KeyError(f"AutoregressiveMetadata data missing '{key}'")
        known_keys = set(required) | {
            "instruction",
            "expected_relations",
            "scene_pointcloud_path",
            "new_object_pointcloud_path",
        }
        extra = {key: copy.deepcopy(value) for key, value in data.items() if key not in known_keys}
        return cls(
            sample_id=data["sample_id"],
            scene_id=data["scene_id"],
            scene_description=data["scene_description"],
            boundary=data["boundary"],
            scene_height=data["scene_height"],
            step=data["step"],
            new_object_id=data["new_object_id"],
            instruction=data.get("instruction", ""),
            expected_relations=(
                copy.deepcopy(data["expected_relations"])
                if "expected_relations" in data else {}
            ),
            existing_objects=ensure_scene_object_list(data["existing_objects"]),
            new_object=SceneObject.from_dict(data["new_object"]),
            scene_pointcloud_path=(
                data["scene_pointcloud_path"] if "scene_pointcloud_path" in data else None
            ),
            new_object_pointcloud_path=(
                data["new_object_pointcloud_path"]
                if "new_object_pointcloud_path" in data else None
            ),
            extra=extra,
        )

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "sample_id": self.sample_id,
            "scene_id": self.scene_id,
            "scene_description": self.scene_description,
            "boundary": copy.deepcopy(self.boundary),
            "scene_height": self.scene_height,
            "step": self.step,
            "new_object_id": self.new_object_id,
            "instruction": self.instruction,
            "expected_relations": copy.deepcopy(self.expected_relations),
            "existing_objects": [obj.to_dict() for obj in self.existing_objects],
            "new_object": self.new_object.to_dict(),
        }
        if self.scene_pointcloud_path is not None or self.new_object_pointcloud_path is not None:
            data["scene_pointcloud_path"] = self.scene_pointcloud_path or ""
            data["new_object_pointcloud_path"] = self.new_object_pointcloud_path or ""
        data.update(copy.deepcopy(self.extra))
        return data


@dataclass
class PlacementStepResult:
    """Typed representation of one placement entry in layout/inference outputs."""
    object_id: int
    object_short_desc: str
    instruction: str
    jid: Optional[str] = None
    pose: Optional[Pose] = None
    gt_pose: Optional[Pose] = None
    scene_pointcloud_path: Optional[str] = None
    inference_time: float = 0.0
    raw_output: str = ""
    error: Optional[str] = None
    rejection_sampling: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.object_id = int(self.object_id)
        if self.pose is not None:
            self.pose = ensure_pose(self.pose)
        if self.gt_pose is not None:
            self.gt_pose = ensure_pose(self.gt_pose)
        self.inference_time = float(self.inference_time)
        self.rejection_sampling = copy.deepcopy(self.rejection_sampling)
        self.extra = copy.deepcopy(self.extra)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlacementStepResult":
        if "object_id" not in data:
            raise KeyError("PlacementStepResult data missing 'object_id'")
        if "object_short_desc" not in data:
            raise KeyError("PlacementStepResult data missing 'object_short_desc'")

        known_keys = {
            "object_id",
            "object_short_desc",
            "instruction",
            "jid",
            "pose",
            "gt_pose",
            "scene_pointcloud_path",
            "inference_time",
            "raw_output",
            "error",
            "rejection_sampling",
        }
        extra = {key: copy.deepcopy(value) for key, value in data.items() if key not in known_keys}
        return cls(
            object_id=data["object_id"],
            object_short_desc=data["object_short_desc"],
            instruction=data["instruction"] if "instruction" in data else "",
            jid=data["jid"] if "jid" in data else None,
            pose=ensure_pose(data["pose"]) if "pose" in data and data["pose"] is not None else None,
            gt_pose=ensure_pose(data["gt_pose"]) if "gt_pose" in data and data["gt_pose"] is not None else None,
            scene_pointcloud_path=data["scene_pointcloud_path"] if "scene_pointcloud_path" in data else None,
            inference_time=data["inference_time"] if "inference_time" in data else 0.0,
            raw_output=data["raw_output"] if "raw_output" in data else "",
            error=data["error"] if "error" in data else None,
            rejection_sampling=(
                copy.deepcopy(data["rejection_sampling"]) if "rejection_sampling" in data else None
            ),
            extra=extra,
        )

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "object_id": self.object_id,
            "object_short_desc": self.object_short_desc,
            "instruction": self.instruction,
            "jid": self.jid,
            "pose": self.pose.to_dict() if self.pose is not None else None,
            "gt_pose": self.gt_pose.to_dict() if self.gt_pose is not None else None,
            "scene_pointcloud_path": self.scene_pointcloud_path,
            "inference_time": self.inference_time,
            "raw_output": self.raw_output,
        }
        if self.error is not None:
            data["error"] = self.error
        if self.rejection_sampling is not None:
            data["rejection_sampling"] = copy.deepcopy(self.rejection_sampling)
        data.update(copy.deepcopy(self.extra))
        return data


@dataclass
class LayoutInferenceSummary:
    """Typed batch/joint layout summary with legacy JSON serialization."""
    total_tasks: int
    successful_placements: List[PlacementStepResult]
    failed_placements: List[PlacementStepResult]
    final_scene: Scene
    final_scene_pointcloud_path: Optional[str] = None
    total_inference_time: float = 0.0
    total_elapsed_time: float = 0.0
    avg_inference_time: Optional[float] = None
    layout_mode: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.total_tasks = int(self.total_tasks)
        self.successful_placements = [
            result if isinstance(result, PlacementStepResult) else PlacementStepResult.from_dict(result)
            for result in self.successful_placements
        ]
        self.failed_placements = [
            result if isinstance(result, PlacementStepResult) else PlacementStepResult.from_dict(result)
            for result in self.failed_placements
        ]
        self.final_scene = ensure_scene(self.final_scene)
        self.total_inference_time = float(self.total_inference_time)
        self.total_elapsed_time = float(self.total_elapsed_time)
        if self.avg_inference_time is not None:
            self.avg_inference_time = float(self.avg_inference_time)
        self.extra = copy.deepcopy(self.extra)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LayoutInferenceSummary":
        if "final_scene" not in data:
            raise KeyError("LayoutInferenceSummary data missing 'final_scene'")

        known_keys = {
            "layout_mode",
            "total_tasks",
            "successful",
            "failed",
            "successful_placements",
            "failed_placements",
            "final_scene",
            "final_scene_pointcloud_path",
            "total_inference_time",
            "total_elapsed_time",
            "avg_inference_time",
        }
        extra = {key: copy.deepcopy(value) for key, value in data.items() if key not in known_keys}
        return cls(
            total_tasks=data["total_tasks"] if "total_tasks" in data else 0,
            successful_placements=[
                PlacementStepResult.from_dict(item)
                for item in data["successful_placements"]
            ] if "successful_placements" in data else [],
            failed_placements=[
                PlacementStepResult.from_dict(item)
                for item in data["failed_placements"]
            ] if "failed_placements" in data else [],
            final_scene=Scene.from_dict(data["final_scene"]),
            final_scene_pointcloud_path=(
                data["final_scene_pointcloud_path"]
                if "final_scene_pointcloud_path" in data else None
            ),
            total_inference_time=(
                data["total_inference_time"] if "total_inference_time" in data else 0.0
            ),
            total_elapsed_time=(
                data["total_elapsed_time"] if "total_elapsed_time" in data else 0.0
            ),
            avg_inference_time=(
                data["avg_inference_time"] if "avg_inference_time" in data else None
            ),
            layout_mode=data["layout_mode"] if "layout_mode" in data else None,
            extra=extra,
        )

    def to_dict(self) -> Dict[str, Any]:
        if self.avg_inference_time is None:
            avg_inference_time = (
                self.total_inference_time / self.total_tasks
                if self.total_tasks else 0.0
            )
        else:
            avg_inference_time = self.avg_inference_time

        data: Dict[str, Any] = {
            "total_tasks": self.total_tasks,
            "successful": len(self.successful_placements),
            "failed": len(self.failed_placements),
            "successful_placements": [
                result.to_dict() for result in self.successful_placements
            ],
            "failed_placements": [
                result.to_dict() for result in self.failed_placements
            ],
            "final_scene": self.final_scene.to_dict(object_field="objects"),
            "final_scene_pointcloud_path": self.final_scene_pointcloud_path,
            "total_inference_time": self.total_inference_time,
            "total_elapsed_time": self.total_elapsed_time,
            "avg_inference_time": avg_inference_time,
        }
        if self.layout_mode is not None:
            data["layout_mode"] = self.layout_mode
        data.update(copy.deepcopy(self.extra))
        return data
