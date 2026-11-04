"""
将 layout 结果导出为点云或网格文件，便于在服务器上快速预览。

默认导出格式:
- glb: 场景网格（适合浏览器或任意 GLTF viewer）

可选格式:
- ply: 场景点云
- obj: 合并后的场景网格

物体上色方式 (--color-mode):
- texture (默认): 保留资产原始材质/贴图
- flat: 按调色板给每个物体分配一个纯色，便于按物体区分（不含贴图）

示例:
    python scripts/export_layout_scene.py \
        --layout output/benchmark/20260401_133816/pcd/within_book_tvtable03.json \
        --formats glb
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import trimesh
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import triangulate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.paths import ProjectPaths
from inference.pointcloud_scene_manager import PointCloudSceneManager
from utils.convex_decomposition import (
    build_transform_matrix as _build_transform_matrix,
    load_canonical_mesh as _load_base_mesh,
)
from utils.scene_types import ensure_scene_object_list, scene_from_external_dict


_PALETTE = np.array(
    [
        [213, 94, 0, 255],
        [0, 114, 178, 255],
        [0, 158, 115, 255],
        [204, 121, 167, 255],
        [230, 159, 0, 255],
        [86, 180, 233, 255],
        [240, 228, 66, 255],
        [148, 103, 189, 255],
    ],
    dtype=np.uint8,
)


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"layout 文件必须是 JSON object: {path}")
    return data


def _extract_scene_data(layout_data: dict[str, Any]) -> dict[str, Any]:
    scene = scene_from_external_dict(layout_data)
    return scene.to_dict(object_field="objects")


def _resolve_output_dir(layout_path: Path, output_dir_arg: str | None) -> Path:
    if output_dir_arg is not None:
        return Path(output_dir_arg)
    return layout_path.parent / f"{layout_path.stem}_exports"


def _resolve_output_path(layout_path: Path, output_dir: Path, file_type: str) -> Path:
    return output_dir / f"{layout_path.stem}.{file_type}"


def _resolve_path_from_project(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _apply_object_flat_color(mesh: trimesh.Trimesh, object_index: int) -> None:
    color = _PALETTE[object_index % len(_PALETTE)]
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh,
        face_colors=np.repeat(color[None, :], len(mesh.faces), axis=0),
    )


def _build_object_mesh(
    obj_data: dict[str, Any], object_index: int, color_mode: str
) -> tuple[str, trimesh.Trimesh]:
    required_keys = ["id", "jid", "size", "center"]
    for key in required_keys:
        if key not in obj_data:
            raise KeyError(f"object 缺少字段 {key}: {obj_data}")

    obj_id = obj_data["id"]
    jid = obj_data["jid"]
    size = obj_data["size"]
    center = obj_data["center"]
    rotation = obj_data["rotation"] if "rotation" in obj_data else [0.0, 0.0, 0.0]
    short_desc = obj_data["short_desc"] if "short_desc" in obj_data else f"object_{obj_id}"

    base_mesh = _load_base_mesh(str(jid))
    mesh = base_mesh.copy()

    mesh_size = mesh.bounds[1] - mesh.bounds[0]
    scale = np.asarray(size, dtype=np.float64) / np.maximum(mesh_size, 1e-6)
    transform = _build_transform_matrix(center=center, rotation=rotation, scale=scale)
    mesh.apply_transform(transform)
    if color_mode == "flat":
        _apply_object_flat_color(mesh, object_index)

    safe_desc = str(short_desc).replace(" ", "_")
    name = f"{int(obj_id):03d}_{safe_desc}"
    return name, mesh


def _normalize_polygon(boundary: Sequence[Sequence[float]]) -> Polygon:
    polygon = Polygon(boundary)
    if polygon.is_empty:
        raise ValueError("boundary 为空，无法创建地板")
    if not polygon.is_valid:
        polygon = polygon.buffer(0)

    if isinstance(polygon, MultiPolygon):
        polygon = max(polygon.geoms, key=lambda geom: geom.area)

    if not isinstance(polygon, Polygon):
        raise TypeError("boundary 无法转换为单个 Polygon")
    return polygon


def _build_floor_mesh(boundary: Sequence[Sequence[float]]) -> trimesh.Trimesh | None:
    if len(boundary) < 3:
        return None

    polygon = _normalize_polygon(boundary)
    triangles = triangulate(polygon)

    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for triangle in triangles:
        rep_point = triangle.representative_point()
        if not polygon.covers(rep_point):
            continue

        coords = list(triangle.exterior.coords)[:-1]
        if len(coords) != 3:
            continue

        base_index = len(vertices)
        for x_coord, y_coord in coords:
            vertices.append([float(x_coord), float(y_coord), 0.0])
        faces.append([base_index, base_index + 1, base_index + 2])

    if not faces:
        return None

    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh,
        face_colors=np.repeat(
            np.asarray([[180, 180, 180, 255]], dtype=np.uint8),
            len(mesh.faces),
            axis=0,
        ),
    )
    return mesh


def _build_scene_mesh(scene_data: dict[str, Any], color_mode: str) -> trimesh.Scene:
    scene = trimesh.Scene()

    boundary = scene_data["boundary"] if "boundary" in scene_data else []
    floor_mesh = _build_floor_mesh(boundary)
    if floor_mesh is not None:
        scene.add_geometry(floor_mesh, node_name="floor", geom_name="floor")

    objects = scene_data["objects"] if "objects" in scene_data else None
    if not isinstance(objects, list):
        raise ValueError("final_scene.objects 必须是 list")

    for object_index, obj_data in enumerate(objects):
        if not isinstance(obj_data, dict):
            raise TypeError(f"objects[{object_index}] 必须是 object")
        name, mesh = _build_object_mesh(obj_data, object_index, color_mode)
        scene.add_geometry(mesh, node_name=name, geom_name=name)

    return scene


def _export_scene_mesh(
    scene_data: dict[str, Any], output_path: Path, file_type: str, color_mode: str
) -> Path:
    scene = _build_scene_mesh(scene_data, color_mode)

    if file_type == "obj":
        mesh_list = []
        for geometry in scene.geometry.values():
            if not isinstance(geometry, trimesh.Trimesh):
                raise TypeError("scene 中包含非 Trimesh geometry，无法导出 obj")
            mesh_list.append(geometry)
        combined = trimesh.util.concatenate(mesh_list)
        combined.export(output_path)
        return output_path

    if file_type == "glb":
        scene.export(output_path)
        return output_path

    raise ValueError(f"不支持的网格格式: {file_type}")


def _export_pointcloud(
    layout_data: dict[str, Any],
    scene_data: dict[str, Any],
    output_path: Path,
    asset_pointcloud_dirs: Sequence[Path],
) -> Path:
    if "final_scene_pointcloud_path" in layout_data:
        source_path_value = layout_data["final_scene_pointcloud_path"]
        if isinstance(source_path_value, str) and source_path_value:
            source_path = _resolve_path_from_project(source_path_value)
            if source_path.exists():
                shutil.copy2(source_path, output_path)
                return output_path

    if "scene_id" not in scene_data:
        raise KeyError("final_scene 缺少 scene_id")
    if "boundary" not in scene_data:
        raise KeyError("final_scene 缺少 boundary")
    if "objects" not in scene_data:
        raise KeyError("final_scene 缺少 objects")

    manager = PointCloudSceneManager(
        asset_pointcloud_dir=list(asset_pointcloud_dirs),
        output_dir=output_path.parent,
        session_timestamp="export_scene",
    )
    generated_path = manager.init_scene(
        scene_id=str(scene_data["scene_id"]),
        boundary=scene_data["boundary"],
        existing_objects=ensure_scene_object_list(scene_data["objects"]),
    )
    shutil.copy2(generated_path, output_path)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出 layout 为 ply / obj / glb")
    parser.add_argument("--layout", type=str, required=True, help="layout JSON 文件路径")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["glb"],
        choices=["ply", "obj", "glb"],
        help="导出格式，可多选；默认仅导出 glb",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录，默认在 layout 同级生成 *_exports 目录",
    )
    parser.add_argument(
        "--asset-pointcloud-dirs",
        nargs="+",
        default=[
            str(ProjectPaths.IMAGINARIUM_ASSETS_DIR),
            str(ProjectPaths.THREED_FUTURE_MODEL_DIR),
        ],
        help="导出 ply 时使用的资产点云目录列表",
    )
    parser.add_argument(
        "--color-mode",
        type=str,
        default="texture",
        choices=["texture", "flat"],
        help=(
            "obj/glb 物体网格上色方式: texture(默认，保留资产原始材质/贴图) / "
            "flat(按调色板给每个物体分配一个纯色，便于按物体区分)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    layout_path = Path(args.layout)
    if not layout_path.is_absolute():
        layout_path = PROJECT_ROOT / layout_path
    if not layout_path.exists():
        raise FileNotFoundError(f"layout 文件不存在: {layout_path}")

    output_dir = _resolve_output_dir(layout_path, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layout_data = _load_json(layout_path)
    scene_data = _extract_scene_data(layout_data)

    if "scene_id" not in scene_data:
        raise KeyError("final_scene 缺少 scene_id")
    scene_id = str(scene_data["scene_id"])

    asset_pointcloud_dirs = [Path(path_str) for path_str in args.asset_pointcloud_dirs]

    exported_paths: list[Path] = []
    for fmt in args.formats:
        output_path = _resolve_output_path(
            layout_path=layout_path,
            output_dir=output_dir,
            file_type=fmt,
        )
        if fmt == "ply":
            exported_paths.append(
                _export_pointcloud(
                    layout_data=layout_data,
                    scene_data=scene_data,
                    output_path=output_path,
                    asset_pointcloud_dirs=asset_pointcloud_dirs,
                )
            )
            continue

        exported_paths.append(
            _export_scene_mesh(scene_data, output_path, file_type=fmt, color_mode=args.color_mode)
        )

    print("=" * 60)
    print(f"layout: {layout_path}")
    print(f"scene_id: {scene_id}")
    print("导出完成:")
    for path in exported_paths:
        print(f"  - {path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
