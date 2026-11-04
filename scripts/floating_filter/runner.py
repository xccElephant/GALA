#!/usr/bin/env python3
"""Batch-filter floating or below-floor objects from converted scene JSON files.

This script is intended to run inside Blender:

    blender -b -P scripts/floating_filter/runner.py -- \
        --input-dir /path/to/converted_scenes \
        --output-dir /path/to/filtered_scenes \
        --log-path /path/to/floating_filter_log.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PKG_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = PKG_DIR.parent
PROJECT_ROOT = SCRIPTS_DIR.parent
for path in (SCRIPTS_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bpy  # type: ignore
from mathutils import Vector  # type: ignore
from mathutils.bvhtree import BVHTree  # type: ignore

from floating_filter.detection import (
    detect_floating_objects_in_current_scene,
    partition_floating_issues,
)
from floating_filter.checkpoint import (
    aggregate_checkpoint_results,
    checkpoint_is_complete,
    load_checkpoint_result,
    prune_stale_checkpoints,
    select_shard_files,
    write_checkpoint_result,
)
from visualize_layout import clear_scene, import_asset

DEFAULT_COLLISION_MIN_OVERLAP_TRIANGLES = 100


def _scene_for_blender(scene_data: dict[str, Any]) -> dict[str, Any]:
    """Return a scene copy whose object ids are stable list indices."""
    objects = []
    for index, obj in enumerate(scene_data.get("objects", [])):
        blender_obj = dict(obj)
        blender_obj["id"] = index
        objects.append(blender_obj)

    return {
        "scene_id": scene_data.get("scene_id", "unknown"),
        "scene_type": scene_data.get("scene_type", ""),
        "boundary": scene_data.get("boundary", []),
        "scene_height": scene_data.get("scene_height"),
        "objects": objects,
    }


def _build_blender_scene(
    scene_data: dict[str, Any],
    dataset: str,
    asset_cache_size: int,
) -> None:
    boundary = scene_data.get("boundary") or []
    if len(boundary) < 3:
        raise ValueError("场景缺少有效 boundary，无法创建地板并检测悬浮")

    clear_scene()
    bpy.context.scene.tool_settings.transform_pivot_point = "BOUNDING_BOX_CENTER"
    _create_detection_floor(boundary)

    objects = scene_data.get("objects", [])
    print(f"物体数量: {len(objects)}")
    for obj_data in objects:
        obj_id = int(obj_data["id"])
        short_desc = obj_data.get("short_desc", f"Object_{obj_id}")
        center = obj_data["center"]
        size = obj_data["size"]
        rotation = obj_data.get("rotation", [0, 0, 0])
        jid = obj_data.get("jid", "")

        obj_name = f"{obj_id:02d}_{short_desc.replace(' ', '_')}"
        if not jid:
            raise ValueError(f"Object {obj_name} 缺少 jid，无法进行悬浮/穿地检测")

        obj = import_asset(
            jid,
            obj_name,
            center,
            size,
            rotation,
            dataset=dataset,
            asset_cache_size=asset_cache_size,
        )
        if obj is None:
            raise RuntimeError(
                f"Object {obj_name} 的资产导入失败 (jid={jid}, dataset={dataset})"
            )

        obj["short_desc"] = short_desc
        obj["object_id"] = obj_id
        obj["jid"] = jid
        instruction = obj_data.get("instruction")
        if instruction:
            obj["instruction"] = str(instruction)

    bpy.context.view_layer.update()


def _create_detection_floor(boundary: list, height: float = 0.0):
    mesh = bpy.data.meshes.new("Floor_Mesh")
    ordered_boundary = list(boundary)
    signed_area = 0.0
    for index, point in enumerate(ordered_boundary):
        next_point = ordered_boundary[(index + 1) % len(ordered_boundary)]
        signed_area += float(point[0]) * float(next_point[1])
        signed_area -= float(next_point[0]) * float(point[1])
    if signed_area < 0.0:
        ordered_boundary.reverse()

    vertices = [
        (float(point[0]), float(point[1]), float(height))
        for point in ordered_boundary
    ]
    mesh.from_pydata(vertices, [], [list(range(len(vertices)))])
    mesh.update(calc_edges=True)
    floor = bpy.data.objects.new("Floor", mesh)
    bpy.context.collection.objects.link(floor)
    return floor


def _remove_blender_asset_objects(indices_to_remove: set[int]) -> None:
    if not indices_to_remove:
        return
    for obj in list(bpy.context.scene.objects):
        if obj.type == "MESH" and "object_id" in obj:
            try:
                obj_id = int(obj["object_id"])
            except (TypeError, ValueError):
                continue
            if obj_id in indices_to_remove:
                bpy.data.objects.remove(obj, do_unlink=True)
    bpy.context.view_layer.update()


def _world_bbox_bounds(obj) -> tuple[Vector, Vector]:
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    mins = Vector((min(float(corner[i]) for corner in corners) for i in range(3)))
    maxs = Vector((max(float(corner[i]) for corner in corners) for i in range(3)))
    return mins, maxs


def _bbox_overlaps(a_obj, b_obj) -> bool:
    a_min, a_max = _world_bbox_bounds(a_obj)
    b_min, b_max = _world_bbox_bounds(b_obj)
    for axis in range(3):
        if a_max[axis] < b_min[axis]:
            return False
        if b_max[axis] < a_min[axis]:
            return False
    return True


def _build_world_bvh(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        vertices = [tuple(obj.matrix_world @ vertex.co) for vertex in mesh.vertices]
        polygons = [
            tuple(int(vertex_index) for vertex_index in polygon.vertices)
            for polygon in mesh.polygons
            if len(polygon.vertices) >= 3
        ]
        if not vertices or not polygons:
            return None
        return BVHTree.FromPolygons(vertices, polygons, all_triangles=False, epsilon=0.0)
    finally:
        evaluated.to_mesh_clear()


def _asset_objects_by_id() -> dict[int, Any]:
    result = {}
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH" and "object_id" in obj:
            result[int(obj["object_id"])] = obj
    return result


def _collision_detail(
    index: int,
    obj_data: dict[str, Any],
    volume: float,
    worst_collision: dict[str, Any],
) -> dict[str, Any]:
    return {
        "removed_idx": index,
        "removed_jid": obj_data.get("jid", ""),
        "removed_desc": obj_data.get("short_desc", ""),
        "removed_volume": round(volume, 6),
        **worst_collision,
    }


def _compute_blender_collision_removals(
    original_objects: list[dict[str, Any]],
    min_overlap_triangles: int,
) -> list[dict[str, Any]]:
    object_by_id = _asset_objects_by_id()
    indexed_objects = []
    for index, obj_data in enumerate(original_objects):
        blender_obj = object_by_id.get(index)
        if blender_obj is None:
            continue
        size = obj_data["size"]
        volume = float(size[0]) * float(size[1]) * float(size[2])
        indexed_objects.append((volume, index, obj_data, blender_obj))

    indexed_objects.sort(reverse=True, key=lambda item: item[0])
    placed: list[tuple[int, dict[str, Any], Any, Any]] = []
    removals: list[dict[str, Any]] = []

    for volume, index, obj_data, blender_obj in indexed_objects:
        candidate_bvh = _build_world_bvh(blender_obj)
        if candidate_bvh is None:
            placed.append((index, obj_data, blender_obj, None))
            continue

        worst_collision = None
        for placed_index, placed_data, placed_obj, placed_bvh in placed:
            if placed_bvh is None:
                continue
            if not _bbox_overlaps(blender_obj, placed_obj):
                continue
            overlap_count = len(candidate_bvh.overlap(placed_bvh))
            if overlap_count < min_overlap_triangles:
                continue
            if worst_collision is None or overlap_count > worst_collision["overlap_triangles"]:
                worst_collision = {
                    "collided_with_idx": placed_index,
                    "collided_with_desc": placed_data.get("short_desc", ""),
                    "overlap_triangles": overlap_count,
                }

        if worst_collision is None:
            placed.append((index, obj_data, blender_obj, candidate_bvh))
            continue

        removals.append(_collision_detail(index, obj_data, volume, worst_collision))

    return removals


def _floating_result_to_detail(
    result: dict[str, Any],
    original_objects: list[dict[str, Any]],
    *,
    removal_round: int | None = None,
    ignored_reason: str | None = None,
) -> dict[str, Any]:
    index = int(result["id"])
    obj = original_objects[index]
    detail = {
        "index": index,
        "id": obj.get("id"),
        "jid": obj.get("jid"),
        "short_desc": obj.get("short_desc", ""),
        "kind": result.get("kind"),
        "bottom_z": round(float(result["bottom_z"]), 4),
        "gap": round(float(result["gap"]), 4),
        "support_name": result.get("support_name"),
        "severity": result.get("severity"),
    }
    if removal_round is not None:
        detail["removal_round"] = removal_round
    if ignored_reason is not None:
        detail["ignored_reason"] = ignored_reason
    return detail


def filter_scene_file(
    json_file: Path,
    output_file: Path,
    dataset: str,
    min_object_count: int,
    tolerance: float,
    asset_cache_size: int,
    collision_min_overlap_triangles: int,
    disable_collision: bool,
    disable_floating: bool,
) -> dict[str, Any]:
    with open(json_file, "r", encoding="utf-8") as f:
        scene_data = json.load(f)

    original_objects = list(scene_data.get("objects", []))
    blender_scene = _scene_for_blender(scene_data)
    _build_blender_scene(
        blender_scene,
        dataset=dataset,
        asset_cache_size=asset_cache_size,
    )

    collision_details = []
    collision_indices_to_remove: set[int] = set()
    if not disable_collision:
        collision_details = _compute_blender_collision_removals(
            original_objects,
            min_overlap_triangles=collision_min_overlap_triangles,
        )
        collision_indices_to_remove = {
            int(result["removed_idx"]) for result in collision_details
        }
        _remove_blender_asset_objects(collision_indices_to_remove)

    floating_removal_details = []
    floating_indices_to_remove: set[int] = set()
    removal_round = 0
    ignored_results = []
    if not disable_floating:
        while True:
            removal_round += 1
            floating_results = detect_floating_objects_in_current_scene(
                blender_scene,
                tolerance=tolerance,
            )
            removal_results, ignored_results = partition_floating_issues(floating_results)
            round_indices = {int(result["id"]) for result in removal_results}
            round_indices -= collision_indices_to_remove
            round_indices -= floating_indices_to_remove
            if not round_indices:
                break

            for result in removal_results:
                index = int(result["id"])
                if index in round_indices:
                    floating_removal_details.append(
                        _floating_result_to_detail(
                            result,
                            original_objects,
                            removal_round=removal_round,
                        )
                    )
            floating_indices_to_remove.update(round_indices)
            _remove_blender_asset_objects(round_indices)

    ignored_details = []
    for result in ignored_results:
        ignored_details.append(
            _floating_result_to_detail(
                result,
                original_objects,
                ignored_reason="slight_floating",
            )
        )

    filtered_objects = [
        obj
        for index, obj in enumerate(original_objects)
        if index not in collision_indices_to_remove
        and index not in floating_indices_to_remove
    ]
    scene_data["objects"] = filtered_objects

    remaining_object_count = len(filtered_objects)
    skipped = min_object_count > 0 and remaining_object_count < min_object_count
    if not skipped:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(scene_data, f, indent=2, ensure_ascii=False)

    return {
        "file": json_file.name,
        "removed_collision_count": len(collision_indices_to_remove),
        "removed_floating_count": len(floating_indices_to_remove),
        "ignored_floating_count": len(ignored_details),
        "floating_removal_rounds": removal_round - 1,
        "remaining_object_count": remaining_object_count,
        "skipped": skipped,
        "reason": (
            f"object_count ({remaining_object_count}) < {min_object_count} after Blender filter"
            if skipped
            else None
        ),
        "collision_min_overlap_triangles": collision_min_overlap_triangles,
        "collision_objects": collision_details,
        "floating_objects": floating_removal_details,
        "ignored_floating_objects": ignored_details,
    }


def filter_directory(
    input_dir: Path,
    output_dir: Path,
    log_path: Path,
    dataset: str,
    min_object_count: int,
    tolerance: float,
    asset_cache_size: int,
    collision_min_overlap_triangles: int,
    disable_collision: bool,
    disable_floating: bool,
    checkpoint_log_dir: Path | None = None,
    num_shards: int = 1,
    shard_index: int = 0,
    resume: bool = True,
) -> dict[str, Any]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    checkpoint_log_dir = (
        Path(checkpoint_log_dir)
        if checkpoint_log_dir is not None
        else log_path.parent / f"{log_path.stem}_scene_logs"
    )
    json_files = sorted(input_dir.glob("*.json"))
    selected_files = select_shard_files(json_files, num_shards=num_shards, shard_index=shard_index)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_log_dir.mkdir(parents=True, exist_ok=True)
    prune_stale_checkpoints(json_files, output_dir=output_dir, log_dir=checkpoint_log_dir)

    scene_logs = []

    for index, json_file in enumerate(selected_files, start=1):
        if resume and checkpoint_is_complete(
            json_file,
            output_dir=output_dir,
            log_dir=checkpoint_log_dir,
        ):
            result = load_checkpoint_result(json_file, checkpoint_log_dir)
            scene_logs.append(result)
            print(
                f"[shard {shard_index + 1}/{num_shards}] "
                f"[{index}/{len(selected_files)}] 跳过已完成: {json_file.name}"
            )
            continue

        if not resume:
            (output_dir / json_file.name).unlink(missing_ok=True)
            (checkpoint_log_dir / json_file.name).unlink(missing_ok=True)

        print(
            f"[shard {shard_index + 1}/{num_shards}] "
            f"[{index}/{len(selected_files)}] 检测碰撞 + 悬浮/穿地: {json_file.name}"
        )
        result = filter_scene_file(
            json_file=json_file,
            output_file=output_dir / json_file.name,
            dataset=dataset,
            min_object_count=min_object_count,
            tolerance=tolerance,
            asset_cache_size=asset_cache_size,
            collision_min_overlap_triangles=collision_min_overlap_triangles,
            disable_collision=disable_collision,
            disable_floating=disable_floating,
        )
        write_checkpoint_result(json_file, checkpoint_log_dir, result)
        scene_logs.append(result)

    payload = aggregate_checkpoint_results(
        selected_files,
        output_dir=output_dir,
        log_dir=checkpoint_log_dir,
        min_object_count=min_object_count,
        tolerance=tolerance,
        dataset=dataset,
    )
    payload["summary"]["input_dir"] = str(input_dir)
    payload["summary"]["asset_cache_size"] = asset_cache_size
    payload["summary"]["collision_min_overlap_triangles"] = collision_min_overlap_triangles
    payload["summary"]["disable_collision"] = disable_collision
    payload["summary"]["disable_floating"] = disable_floating

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    summary = payload["summary"]
    print(f"\n===== 悬浮/穿地过滤完成 shard {shard_index + 1}/{num_shards} =====")
    print(f"本分片场景数: {len(selected_files)}")
    print(f"保留场景数: {summary['written_scenes']}")
    print(f"跳过场景数: {summary['skipped_scenes']}")
    print(f"删除悬浮/穿地物体总数: {summary['total_floating_removed_objects']}")
    print(f"发生悬浮/穿地删除的场景数: {summary['scenes_with_floating_removal']}")
    print(f"删除碰撞物体总数: {summary.get('total_collision_removed_objects', 0)}")
    print(f"发生碰撞删除的场景数: {summary.get('scenes_with_collision_removal', 0)}")
    print(f"过滤日志已保存: {log_path}")
    return payload


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量过滤 converted_scenes 中的悬浮/穿地物体")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--log-path", required=True, type=Path)
    parser.add_argument("--dataset", default="auto", choices=["auto", "3dfront", "imaginarium"])
    parser.add_argument("--min-object-count", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=0.015)
    parser.add_argument("--asset-cache-size", type=int, default=512)
    parser.add_argument(
        "--collision-min-overlap-triangles",
        type=int,
        default=100,
    )
    parser.add_argument("--disable-collision", action="store_true")
    parser.add_argument("--disable-floating", action="store_true")
    parser.add_argument("--checkpoint-log-dir", type=Path, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    args = parse_args(argv)
    filter_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        log_path=args.log_path,
        dataset=args.dataset,
        min_object_count=args.min_object_count,
        tolerance=args.tolerance,
        asset_cache_size=max(0, args.asset_cache_size),
        collision_min_overlap_triangles=max(1, args.collision_min_overlap_triangles),
        disable_collision=args.disable_collision,
        disable_floating=args.disable_floating,
        checkpoint_log_dir=args.checkpoint_log_dir,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
