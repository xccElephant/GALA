"""
点云场景管理器

用于自回归推理时动态管理场景点云：
- 初始化场景点云（地板 + 已存在物体）
- 每步推理成功后，将新物体点云变换并加入场景
- 保存/加载场景点云
"""
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from utils.pointcloud import (
    build_pointcloud,
    load_asset_pointcloud,
    sample_floor_pointcloud,
    save_pointcloud_as_ply,
    translate_pointcloud_to_nonnegative,
    transform_normals,
    transform_pointcloud,
)
from utils.logger import Logger
from utils.scene_types import Pose, SceneObject


class PointCloudSceneManager:
    """
    点云场景管理器
    
    用于自回归推理时动态管理场景点云
    """
    
    def __init__(
        self,
        asset_pointcloud_dir: Union[str, Path, Sequence[Union[str, Path]]],
        output_dir: str = None,
        floor_sample_points: int = 2048,
        session_timestamp: str = None,
        floor_random_seed: Optional[int] = 42,
    ):
        """
        初始化点云场景管理器
        
        Args:
            asset_pointcloud_dir: 资产点云目录（单目录或多目录）
            output_dir: 输出目录（用于保存场景点云），如果为 None 则使用临时目录
            floor_sample_points: 地板点云采样点数
            session_timestamp: 会话时间戳（用于区分不同运行），如果为 None 则自动生成
            floor_random_seed: 地板点云采样随机种子；固定后可保证多次运行输入一致
        """
        self.asset_pointcloud_dirs = self._normalize_asset_dirs(asset_pointcloud_dir)
        self.output_dir = Path(output_dir) if output_dir else Path("/tmp/scene_pointclouds")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.floor_sample_points = floor_sample_points
        self.floor_random_seed = floor_random_seed
        
        # 会话时间戳，用于区分不同的推理运行，避免覆盖
        self._session_timestamp = session_timestamp or time.strftime("%Y%m%d_%H%M%S")
        
        # 点云缓存: (points, colors, normals, metadata)
        self._pointcloud_cache: Dict[
            str, Optional[Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[Dict[str, Any]]]]
        ] = {}
        
        # 当前场景点云（xyz + rgb + normals）
        self._scene_points: List[np.ndarray] = []  # List of (N, 3)
        self._scene_colors: List[np.ndarray] = []  # List of (N, 3)
        self._scene_normals: List[np.ndarray] = []  # List of (N, 3)
        
        # 场景信息
        self._scene_id: str = "unknown"
        self._step: int = 0
        self._scene_boundary: List[List[float]] = []
        
        # 初始化日志
        self.log = Logger("PointCloud", verbose=True)
        
        # 验证资产目录是否存在，不存在直接报错
        existing_dirs = [d for d in self.asset_pointcloud_dirs if d.exists() and d.is_dir()]
        if len(existing_dirs) == 0:
            raise FileNotFoundError(
                f"Asset pointcloud directories do NOT exist: {self.asset_pointcloud_dirs}\n"
                f"Please check --asset_pointcloud_dir parameter!"
            )

        self._search_dirs = self._build_search_dirs(existing_dirs)
        has_any_asset_entries = False
        for d in self._search_dirs:
            if any(child.is_dir() or child.suffix == ".ply" for child in d.iterdir()):
                has_any_asset_entries = True
                break
        if not has_any_asset_entries:
            raise FileNotFoundError(
                f"Asset pointcloud directories are empty: {self._search_dirs}\n"
                f"Expected subdirectories like {{jid}}/{{jid}}.ply or .ply files."
            )

    @staticmethod
    def _normalize_asset_dirs(
        asset_pointcloud_dir: Union[str, Path, Sequence[Union[str, Path]]],
    ) -> List[Path]:
        if isinstance(asset_pointcloud_dir, (str, Path)):
            dirs = [Path(asset_pointcloud_dir)]
        else:
            dirs = [Path(p) for p in asset_pointcloud_dir]

        unique_dirs: List[Path] = []
        seen = set()
        for d in dirs:
            key = str(d)
            if key in seen:
                continue
            unique_dirs.append(d)
            seen.add(key)
        return unique_dirs

    @staticmethod
    def _build_search_dirs(base_dirs: Sequence[Path]) -> List[Path]:
        search_dirs: List[Path] = []
        seen = set()

        def _add_dir(path: Path) -> None:
            key = str(path)
            if key in seen:
                return
            search_dirs.append(path)
            seen.add(key)

        for base_dir in base_dirs:
            _add_dir(base_dir)
            for dataset_subdir in ("3D-FUTURE-model", "imaginarium_assets"):
                subdir_path = base_dir / dataset_subdir
                if subdir_path.exists() and subdir_path.is_dir():
                    _add_dir(subdir_path)

        return search_dirs
    
    def init_scene(
        self,
        scene_id: str,
        boundary: List[List[float]],
        existing_objects: Optional[Sequence[SceneObject]] = None,
    ) -> str:
        """
        初始化场景点云（地板 + 已存在物体）
        
        Args:
            scene_id: 场景 ID
            boundary: 地板边界点列表 [[x1, y1], [x2, y2], ...]
            existing_objects: 已存在物体列表，每个物体包含:
                - jid: 资产 ID
                - center: 中心位置 [x, y, z]
                - rotation: 旋转角度 [roll, pitch, yaw]
                - size: 物体尺寸 [width, depth, height]
        
        Returns:
            初始场景点云文件路径
        """
        self._scene_id = scene_id
        self._step = 0
        self._scene_boundary = boundary or []
        self._scene_points = []
        self._scene_colors = []
        self._scene_normals = []
        existing_scene_objects = list(existing_objects or [])
                
        # 1. 添加地板点云
        if boundary:
            floor_pc = sample_floor_pointcloud(
                boundary,
                num_points=self.floor_sample_points,
                seed=self.floor_random_seed,
            )
            if len(floor_pc) > 0:
                self._scene_points.append(floor_pc)
                # 地板颜色使用中性灰色
                floor_colors = np.full((len(floor_pc), 3), 0.5, dtype=np.float32)
                self._scene_colors.append(floor_colors)
                # 地板法向量向上 [0, 0, 1]
                floor_normals = np.zeros((len(floor_pc), 3), dtype=np.float32)
                floor_normals[:, 2] = 1.0
                self._scene_normals.append(floor_normals)
                self.log.indent(f"Floor: {len(floor_pc)} points")
        
        # 2. 添加已存在物体的点云
        if existing_scene_objects:
            for obj in existing_scene_objects:
                pose = obj.require_pose()
                self._add_object_pointcloud(
                    jid=obj.jid,
                    center=pose.center,
                    rotation=pose.rotation,
                    size=obj.size,
                )
                self.log.indent(f"Existing: {obj.short_desc}")
        
        # 3. 保存初始场景点云
        scene_pc_path = self._save_current_scene()
        self.log.indent(f"Saved: {scene_pc_path}")
        
        return scene_pc_path

    def _transform_asset_pointcloud(
        self,
        jid: str,
        center: List[float],
        rotation: List[float],
        size: Optional[List[float]],
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """加载并按目标位姿/尺寸变换资产点云"""
        asset_pc = self._load_asset_pointcloud(jid)
        if asset_pc is None:
            raise FileNotFoundError(f"Pointcloud not found for jid: {jid}")
        asset_points, asset_colors, asset_normals, asset_metadata = asset_pc

        scale_vec = None
        if (
            size is not None
            and isinstance(size, (list, tuple))
            and len(size) == 3
            and asset_metadata is not None
        ):
            obj_size = asset_metadata.get("transformed_size")
            if isinstance(obj_size, list) and len(obj_size) == 3:
                scale_vec = [1.0, 1.0, 1.0]
                for i in range(3):
                    if obj_size[i] and obj_size[i] > 0:
                        scale_vec[i] = float(size[i]) / float(obj_size[i])

        transformed_points = transform_pointcloud(asset_points, center, rotation, scale=scale_vec)
        transformed_normals = (
            transform_normals(asset_normals, rotation) if asset_normals is not None else None
        )
        return transformed_points, asset_colors, transformed_normals

    def build_separate_pointcloud_paths(
        self,
        existing_objects: Sequence[SceneObject],
        new_object_jid: str,
        new_object_size: Optional[List[float]] = None,
    ) -> List[str]:
        """
        构建分离点云路径列表，顺序严格对应 prompt 占位符：
        [floor, existing_0, existing_1, ..., new_object]
        """
        if not self._scene_boundary:
            raise ValueError("Scene boundary is empty, cannot build floor pointcloud")
        if not new_object_jid:
            raise ValueError("new_object_jid is empty")

        scene_dir = self.output_dir / f"{self._scene_id}_{self._session_timestamp}"
        separate_dir = scene_dir / f"separate_step_{self._step:03d}"
        separate_dir.mkdir(parents=True, exist_ok=True)

        paths: List[str] = []

        floor_pc = sample_floor_pointcloud(
            self._scene_boundary,
            num_points=self.floor_sample_points,
            seed=self.floor_random_seed,
        )
        if floor_pc is None or len(floor_pc) == 0:
            raise ValueError("Failed to sample floor pointcloud from boundary")
        floor_colors = np.full((len(floor_pc), 3), 0.5, dtype=np.float32)
        floor_normals = np.zeros((len(floor_pc), 3), dtype=np.float32)
        floor_normals[:, 2] = 1.0
        floor_pointcloud = build_pointcloud(floor_pc, floor_colors, floor_normals)
        floor_path = separate_dir / "floor.ply"
        save_pointcloud_as_ply(floor_pointcloud, floor_path)
        paths.append(str(floor_path))

        normalized_existing_objects = list(existing_objects)
        for obj_idx, obj in enumerate(normalized_existing_objects):
            if obj.jid is None:
                raise KeyError(f"Missing 'jid' in existing_objects[{obj_idx}]")
            pose = obj.require_pose()

            transformed_points, colors, normals = self._transform_asset_pointcloud(
                jid=obj.jid,
                center=pose.center,
                rotation=pose.rotation,
                size=obj.size,
            )
            object_pointcloud = build_pointcloud(transformed_points, colors, normals)
            object_path = separate_dir / f"object_{obj_idx:03d}.ply"
            save_pointcloud_as_ply(object_pointcloud, object_path)
            paths.append(str(object_path))

        new_pc = self._load_asset_pointcloud(new_object_jid)
        if new_pc is None:
            raise FileNotFoundError(f"Pointcloud not found for new object jid: {new_object_jid}")
        new_points, new_colors, new_normals, new_metadata = new_pc

        if (
            new_object_size is not None
            and isinstance(new_object_size, (list, tuple))
            and len(new_object_size) == 3
            and new_metadata is not None
        ):
            obj_size = new_metadata.get("transformed_size")
            if isinstance(obj_size, list) and len(obj_size) == 3:
                scale_vec = [1.0, 1.0, 1.0]
                for i in range(3):
                    if obj_size[i] and obj_size[i] > 0:
                        scale_vec[i] = float(new_object_size[i]) / float(obj_size[i])
                new_points = transform_pointcloud(
                    new_points, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], scale=scale_vec,
                )

        new_points = translate_pointcloud_to_nonnegative(new_points)

        new_pointcloud = build_pointcloud(new_points, new_colors, new_normals)
        new_object_path = separate_dir / "new_object.ply"
        save_pointcloud_as_ply(new_pointcloud, new_object_path)
        paths.append(str(new_object_path))

        return paths
    
    def add_placed_object(
        self,
        scene_object: Optional[SceneObject] = None,
        *,
        jid: Optional[str] = None,
        center: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None,
        size: Optional[List[float]] = None,
        object_id: int = 0,
        short_desc: Optional[str] = None,
    ) -> str:
        """
        添加新摆放的物体到场景点云
        
        在每步推理成功后调用，将新物体点云变换后加入场景
        
        Args:
            jid: 物体资产 ID
            center: 推理出的中心位置 [x, y, z]
            rotation: 推理出的旋转角度 [roll, pitch, yaw]
            size: 物体尺寸 [width, depth, height]
            object_id: 物体 ID（保存在场景对象中）
            short_desc: 物体描述（用于日志）
        
        Returns:
            更新后的场景点云文件路径
        """
        if scene_object is not None:
            normalized_object = scene_object
        else:
            if jid is None:
                raise ValueError("jid is required when scene_object is not provided")
            if center is None or rotation is None or size is None:
                raise ValueError("center/rotation/size are required when scene_object is not provided")
            normalized_object = SceneObject(
                id=object_id,
                short_desc=short_desc or jid,
                size=size,
                jid=jid,
                pose=Pose(center=center, rotation=rotation),
            )

        if normalized_object.jid is None:
            raise ValueError("Placed object is missing jid")
        pose = normalized_object.require_pose()

        self._step += 1
        desc = normalized_object.short_desc
        
        self.log.info(f"Step {self._step}: Adding '{desc}'")
        
        # 添加物体点云
        self._add_object_pointcloud(
            normalized_object.jid,
            pose.center,
            pose.rotation,
            size=normalized_object.size,
        )
        
        # 保存更新后的场景点云
        scene_pc_path = self._save_current_scene()
        self.log.indent(f"Saved: {scene_pc_path}")
        
        return scene_pc_path
    
    def get_new_object_pointcloud_path(
        self,
        jid: str,
        size: Optional[List[float]] = None,
    ) -> Optional[Path]:
        """
        获取推理用的新物体点云文件路径。

        与训练数据保持一致：
        - 若提供 size，则按目标尺寸缩放
        - 无论是否缩放，都会平移到非负空间
        
        Args:
            jid: 物体资产 ID
            size: 目标尺寸 [width, depth, height]
        
        Returns:
            点云文件路径，如果不存在则返回 None
        """

        asset_pc = self._load_asset_pointcloud(jid)
        if asset_pc is None:
            return None
        points, colors, normals, metadata = asset_pc

        transformed_points = points
        if (
            size is not None
            and isinstance(size, (list, tuple))
            and len(size) == 3
            and metadata is not None
        ):
            obj_size = metadata.get("transformed_size")
            if isinstance(obj_size, list) and len(obj_size) == 3:
                scale_vec = [1.0, 1.0, 1.0]
                for i in range(3):
                    if obj_size[i] and obj_size[i] > 0:
                        scale_vec[i] = float(size[i]) / float(obj_size[i])
                transformed_points = transform_pointcloud(
                    points, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], scale=scale_vec,
                )

        transformed_points = translate_pointcloud_to_nonnegative(transformed_points)

        scene_dir = self.output_dir / f"{self._scene_id}_{self._session_timestamp}"
        input_dir = scene_dir / "new_object_inputs"
        input_dir.mkdir(parents=True, exist_ok=True)

        size_tag = "raw"
        if size is not None and isinstance(size, (list, tuple)) and len(size) == 3:
            size_tag = "_".join(f"{float(dim):.3f}".replace(".", "p") for dim in size)

        output_path = input_dir / f"{jid}_{size_tag}.ply"
        pointcloud = build_pointcloud(
            transformed_points,
            colors,
            normals,
        )
        save_pointcloud_as_ply(pointcloud, output_path)
        return output_path
    
    def get_current_scene_pointcloud(self) -> Optional[np.ndarray]:
        """
        获取当前场景点云数据
        
        Returns:
            场景点云 (N, 9) [x, y, z, r, g, b, nx, ny, nz]
        """
        scene_xyz = np.vstack(self._scene_points).astype(np.float32)
        scene_rgb = np.vstack(self._scene_colors).astype(np.float32)
        scene_nrm = np.vstack(self._scene_normals).astype(np.float32)
        return np.concatenate([scene_xyz, scene_rgb, scene_nrm], axis=1)
    
    def get_current_scene_pointcloud_path(self) -> str:
        """
        获取当前场景点云文件路径
        
        Returns:
            当前场景点云文件路径
        """
        return str(self.output_dir / f"{self._scene_id}_{self._session_timestamp}" / f"scene_step_{self._step:03d}.ply")
    
    def _add_object_pointcloud(
        self,
        jid: str,
        center: List[float],
        rotation: List[float],
        size: [List[float]],
    ) -> bool:
        """
        添加物体点云到场景（内部方法）
        
        Args:
            jid: 资产 ID
            center: 中心位置 [x, y, z]
            rotation: 旋转角度 [roll, pitch, yaw]
            size: 物体尺寸 [width, depth, height]
        """
        transformed_pc, asset_colors, transformed_normals = self._transform_asset_pointcloud(
            jid=jid,
            center=center,
            rotation=rotation,
            size=size,
        )
        self._scene_points.append(transformed_pc)
        
        # 处理颜色
        if asset_colors is None:
            asset_colors = np.full((len(transformed_pc), 3), 0.5, dtype=np.float32)
        self._scene_colors.append(asset_colors.astype(np.float32))
        
        # 处理法向量（只旋转，不缩放、不平移）
        if transformed_normals is None:
            transformed_normals = np.zeros((len(transformed_pc), 3), dtype=np.float32)
            transformed_normals[:, 2] = 1.0
        self._scene_normals.append(transformed_normals)
    
    def _load_asset_pointcloud(
        self,
        jid: str,
    ) -> Optional[Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[Dict[str, Any]]]]:
        """
        加载资产点云（带缓存）
        
        Args:
            jid: 资产 ID
        
        Returns:
            (points, colors, normals, metadata) 或 None
        """
        # 检查缓存
        if jid in self._pointcloud_cache:
            return self._pointcloud_cache[jid]

        try:
            result = load_asset_pointcloud(jid, self._search_dirs)
            self._pointcloud_cache[jid] = result
            return result

        except Exception as e:
            self.log.warning(f"Failed to load pointcloud {jid}: {e}")
            self._pointcloud_cache[jid] = None
            return None
    
    def _save_current_scene(self) -> str:
        """
        保存当前场景点云
        
        Returns:
            保存的文件路径
        """
        # 创建场景子目录（包含时间戳以避免覆盖）
        scene_dir = self.output_dir / f"{self._scene_id}_{self._session_timestamp}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存文件
        output_path = scene_dir / f"scene_step_{self._step:03d}.ply"
        
        scene_pc = self.get_current_scene_pointcloud()
        save_pointcloud_as_ply(scene_pc, output_path)
        
        return str(output_path)
