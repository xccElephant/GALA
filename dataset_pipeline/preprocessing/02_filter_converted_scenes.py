import argparse
import json
import subprocess
import sys
import tempfile
import shutil
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict
from multiprocessing import cpu_count

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.paths import ProjectPaths
from dataset_pipeline.utils import (
    remove_objects_by_indices,
    translate_scene_to_nonnegative_xy,
)
from scripts.floating_filter.checkpoint import (
    aggregate_checkpoint_results,
    count_completed_checkpoints,
    prune_stale_checkpoints,
)
from utils.spatial_utils import (
    get_obb_world_vertices_from_obj,
    get_object_boundary_overlap_metrics,
)


DEFAULT_MIN_OBJECT_COUNT = 4
DEFAULT_MAX_OUTSIDE_RATIO = 0.05
DEFAULT_CENTER_OUTSIDE_MAX_RATIO = 0.02
DEFAULT_COLLISION_LOG_FILENAME = "collision_removal_log.json"
DEFAULT_TIGHT_BOUNDARY_MARGIN = 0.05
DEFAULT_COLLISION_VOLUME_SAMPLE_COUNT = 10000
SPATIALLM_COLLISION_VOLUME_SAMPLE_COUNT = 2000
NON_POSITIVE_SIZE_EPS = 1e-6
DEFAULT_FLOATING_FILTER_LOG_FILENAME = "floating_filter_log.json"
DEFAULT_FLOATING_CHECKPOINT_DIRNAME = "blender_collision_floating_checkpoint"
DEFAULT_FLOATING_WORKERS = 4
DEFAULT_FLOATING_ASSET_CACHE_SIZE = 512
DEFAULT_FLOATING_TOLERANCE = 0.015
DEFAULT_BLENDER_COLLISION_MIN_OVERLAP_TRIANGLES = 100


def resolve_paths(
    dataset: str,
    input_dir: str | None = None,
    output_dir: str | None = None,
    collision_log_path: str | None = None,
) -> tuple[Path, Path, Path]:
    if dataset == "custom":
        if not input_dir or not output_dir:
            raise ValueError("custom 模式必须提供 --input-dir 和 --output-dir")
        resolved_input_dir = Path(input_dir)
        resolved_output_dir = Path(output_dir)
        resolved_log_path = (
            Path(collision_log_path)
            if collision_log_path
            else resolved_output_dir.parent / DEFAULT_COLLISION_LOG_FILENAME
        )
        return resolved_input_dir, resolved_output_dir, resolved_log_path

    if input_dir or output_dir or collision_log_path:
        raise ValueError(
            "预定义数据集模式下不需要传 --input-dir/--output-dir/--collision-log-path"
        )

    dataset_root_map = {
        "3dfront": ProjectPaths.SSR_3DFRONT_DIR,
        "imaginarium": ProjectPaths.IMAGINARIUM_DIR,
        "spatiallm": ProjectPaths.SPATIALLM_DIR,
        "handcrafted": ProjectPaths.HANDCRAFTED_DIR,
    }
    return (
        ProjectPaths.get_converted_scene_dir(dataset),
        ProjectPaths.get_filtered_scene_dir(dataset),
        dataset_root_map[dataset] / DEFAULT_COLLISION_LOG_FILENAME,
    )


def find_out_of_bounds_objects(
    scene_data: Dict[str, Any],
    max_outside_ratio: float = DEFAULT_MAX_OUTSIDE_RATIO,
    center_outside_max_ratio: float = DEFAULT_CENTER_OUTSIDE_MAX_RATIO,
) -> tuple[set[int], list[Dict[str, Any]], list[Dict[str, Any]]]:
    boundary = scene_data.get("boundary") or []
    if not boundary:
        return set(), [], []

    indices_to_remove: set[int] = set()
    removal_details: list[Dict[str, Any]] = []
    degenerate_details: list[Dict[str, Any]] = []
    for idx, obj in enumerate(scene_data.get("objects", [])):
        metrics = get_object_boundary_overlap_metrics(obj, boundary)
        outside_ratio = float(metrics["outside_ratio"])
        center_inside_boundary = bool(metrics["center_inside_boundary"])
        degenerate_footprint = bool(metrics.get("degenerate_footprint", False))

        if degenerate_footprint:
            degenerate_details.append(
                {
                    "index": idx,
                    "jid": obj.get("jid"),
                    "short_desc": obj.get("short_desc", ""),
                    "degenerate_reason": metrics.get("degenerate_reason"),
                    "center_inside_boundary": center_inside_boundary,
                    "outside_ratio": round(outside_ratio, 4),
                }
            )

        should_remove = (
            outside_ratio > max_outside_ratio
            or (not center_inside_boundary and outside_ratio > center_outside_max_ratio)
        )
        if not should_remove:
            continue

        indices_to_remove.add(idx)
        removal_details.append(
            {
                "index": idx,
                "jid": obj.get("jid"),
                "short_desc": obj.get("short_desc", ""),
                "outside_ratio": round(outside_ratio, 4),
                "outside_area": round(float(metrics["outside_area"]), 4),
                "footprint_area": round(float(metrics["footprint_area"]), 4),
                "center_inside_boundary": center_inside_boundary,
                "degenerate_footprint": degenerate_footprint,
            }
        )

    return indices_to_remove, removal_details, degenerate_details


def compute_tight_boundary_from_objects(
    objects: list[Dict[str, Any]],
    margin: float = DEFAULT_TIGHT_BOUNDARY_MARGIN,
) -> list[list[float]]:
    """根据所有物体 OBB 的 XY 投影计算贴合矩形边界。"""
    if not objects:
        return []

    all_xy = []
    for obj in objects:
        vertices = get_obb_world_vertices_from_obj(obj)
        all_xy.append(vertices[:, :2])
    all_xy = np.vstack(all_xy)

    min_x = float(all_xy[:, 0].min()) - margin
    max_x = float(all_xy[:, 0].max()) + margin
    min_y = float(all_xy[:, 1].min()) - margin
    max_y = float(all_xy[:, 1].max()) + margin

    return [
        [round(min_x, 4), round(min_y, 4)],
        [round(max_x, 4), round(min_y, 4)],
        [round(max_x, 4), round(max_y, 4)],
        [round(min_x, 4), round(max_y, 4)],
    ]


def recompute_tight_boundary_and_translate(scene_data: Dict[str, Any]) -> None:
    """按当前物体重算矩形边界，并将场景平移到非负 XY 空间。"""
    objects = scene_data.get("objects") or []
    if not objects:
        return

    scene_data["boundary"] = compute_tight_boundary_from_objects(objects)
    translate_scene_to_nonnegative_xy(scene_data)


def remove_nonpositive_size_objects(
    scene_data: Dict[str, Any],
    eps: float = NON_POSITIVE_SIZE_EPS,
) -> int:
    objects = scene_data.get("objects") or []
    filtered_objects = []
    removed_count = 0
    for obj in objects:
        size = obj.get("size") or []
        if len(size) != 3:
            filtered_objects.append(obj)
            continue
        try:
            dims = [float(dim) for dim in size]
        except (TypeError, ValueError):
            filtered_objects.append(obj)
            continue
        if any(dim <= eps for dim in dims):
            removed_count += 1
            continue
        filtered_objects.append(obj)
    scene_data["objects"] = filtered_objects
    return removed_count


def preprocess_spatiallm_scene(
    scene_data: Dict[str, Any],
) -> tuple[bool, str | None]:
    """
    SpatialLM 专用预处理：
    1. 过滤非矩形边界（boundary 顶点数 > 4）
    2. 用所有物体 OBB 重算贴合矩形边界
    3. 平移到非负 XY 空间
    """
    boundary = scene_data.get("boundary") or []
    if len(boundary) > 4:
        return False, f"boundary_point_count ({len(boundary)}) > 4"

    remove_nonpositive_size_objects(scene_data)
    recompute_tight_boundary_and_translate(scene_data)
    return True, None


def resolve_collision_volume_sample_count(dataset: str | None) -> int:
    if dataset == "spatiallm":
        return SPATIALLM_COLLISION_VOLUME_SAMPLE_COUNT
    return DEFAULT_COLLISION_VOLUME_SAMPLE_COUNT


def resolve_floating_filter_blender_dataset(dataset: str | None) -> str:
    if dataset == "3dfront":
        return "3dfront"
    if dataset in {"imaginarium", "spatiallm"}:
        return "imaginarium"
    return "auto"


def run_floating_filter(
    input_dir: Path,
    output_dir: Path,
    log_path: Path,
    dataset: str | None,
    min_object_count: int,
    tolerance: float,
    blender_bin: str,
    floating_workers: int,
    floating_asset_cache_size: int,
    collision_min_overlap_triangles: int,
    disable_collision: bool,
    disable_floating: bool,
    checkpoint_dir: Path,
    force_floating: bool = False,
) -> Dict[str, Any]:
    script_path = ProjectPaths.PROJECT_ROOT / "scripts" / "floating_filter" / "runner.py"
    blender_dataset = resolve_floating_filter_blender_dataset(dataset)
    input_files = sorted(input_dir.glob("*.json"))
    worker_count = max(1, min(int(floating_workers), len(input_files) or 1))
    checkpoint_dir = Path(checkpoint_dir)
    scene_checkpoint_dir = checkpoint_dir / "scenes"
    scene_log_dir = checkpoint_dir / "scene_logs"
    shard_log_dir = checkpoint_dir / "blender_logs"

    if force_floating and checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    scene_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    scene_log_dir.mkdir(parents=True, exist_ok=True)
    shard_log_dir.mkdir(parents=True, exist_ok=True)
    prune_stale_checkpoints(input_files, output_dir=scene_checkpoint_dir, log_dir=scene_log_dir)

    print("\n===== Blender 碰撞 + 悬浮/穿地过滤 =====")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {scene_checkpoint_dir}")
    print(f"checkpoint 目录: {checkpoint_dir}")
    print(f"日志路径: {log_path}")
    print(f"Blender: {blender_bin}")
    print(f"Blender workers: {worker_count}")
    print(f"Blender asset cache size/worker: {floating_asset_cache_size}")
    print(f"Blender collision min overlap triangles: {collision_min_overlap_triangles}")
    print(f"Blender 详细输出目录: {shard_log_dir}")

    processes: list[tuple[int, subprocess.Popen, Any, Path]] = []
    for shard_index in range(worker_count):
        shard_log_path = shard_log_dir / f"shard_{shard_index}.log"
        shard_payload_path = shard_log_dir / f"shard_{shard_index}_summary.json"
        command = [
            blender_bin,
            "-b",
            "-P",
            str(script_path),
            "--",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(scene_checkpoint_dir),
            "--log-path",
            str(shard_payload_path),
            "--dataset",
            blender_dataset,
            "--min-object-count",
            str(min_object_count),
            "--tolerance",
            str(tolerance),
            "--asset-cache-size",
            str(max(0, int(floating_asset_cache_size))),
            "--collision-min-overlap-triangles",
            str(max(1, int(collision_min_overlap_triangles))),
            "--checkpoint-log-dir",
            str(scene_log_dir),
            "--num-shards",
            str(worker_count),
            "--shard-index",
            str(shard_index),
        ]
        if disable_collision:
            command.append("--disable-collision")
        if disable_floating:
            command.append("--disable-floating")
        if force_floating:
            command.append("--no-resume")

        shard_log = open(shard_log_path, "w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            stdout=shard_log,
            stderr=subprocess.STDOUT,
        )
        processes.append((shard_index, process, shard_log, shard_log_path))

    failed: list[tuple[int, int, Path]] = []
    try:
        with tqdm(total=len(input_files), desc="Blender 碰撞+悬浮/穿地", unit="scene") as progress:
            completed = count_completed_checkpoints(
                input_files,
                output_dir=scene_checkpoint_dir,
                log_dir=scene_log_dir,
            )
            progress.update(completed)
            while True:
                if all(process.poll() is not None for _idx, process, _log, _path in processes):
                    break
                time.sleep(2.0)
                completed = count_completed_checkpoints(
                    input_files,
                    output_dir=scene_checkpoint_dir,
                    log_dir=scene_log_dir,
                )
                if completed > progress.n:
                    progress.update(completed - progress.n)

            completed = count_completed_checkpoints(
                input_files,
                output_dir=scene_checkpoint_dir,
                log_dir=scene_log_dir,
            )
            if completed > progress.n:
                progress.update(completed - progress.n)

        for shard_index, process, shard_log, shard_log_path in processes:
            return_code = process.returncode
            shard_log.close()
            if return_code != 0:
                failed.append((shard_index, return_code, shard_log_path))
    except KeyboardInterrupt:
        for _shard_index, process, shard_log, _shard_log_path in processes:
            if process.poll() is None:
                process.terminate()
            shard_log.close()
        raise

    if failed:
        for shard_index, return_code, shard_log_path in failed:
            print(
                f"Blender shard {shard_index} 失败 (exit code: {return_code})，"
                f"日志末尾如下: {shard_log_path}"
            )
            with open(shard_log_path, "r", encoding="utf-8", errors="replace") as blender_log:
                for line in deque(blender_log, maxlen=80):
                    print(line.rstrip())
        raise subprocess.CalledProcessError(failed[0][1], command)

    payload = aggregate_checkpoint_results(
        input_files,
        output_dir=scene_checkpoint_dir,
        log_dir=scene_log_dir,
        min_object_count=min_object_count,
        tolerance=tolerance,
        dataset=blender_dataset,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_file in output_dir.glob("*.json"):
        old_file.unlink()
    for scene_file in sorted(scene_checkpoint_dir.glob("*.json")):
        shutil.copy2(scene_file, output_dir / scene_file.name)

    payload["summary"]["input_dir"] = str(input_dir)
    payload["summary"]["output_dir"] = str(output_dir)
    payload["summary"]["checkpoint_scene_dir"] = str(scene_checkpoint_dir)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    with open(log_path, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_scene_data(
    scene_data: Dict[str, Any],
    indices_to_remove: set[int] | None = None,
    min_object_count: int = DEFAULT_MIN_OBJECT_COUNT,
) -> tuple[Dict[str, Any] | None, Dict[str, int]]:
    removed_collision_count = 0
    if indices_to_remove:
        removed_collision_count = remove_objects_by_indices(scene_data, indices_to_remove)

    object_count = len(scene_data.get("objects", []))
    return (
        scene_data if min_object_count <= 0 or object_count >= min_object_count else None,
        {
            "removed_collision_count": removed_collision_count,
            "removed_count": removed_collision_count,
            "remaining_object_count": object_count,
        },
    )


def prepare_boundary_filtered_inputs(
    input_dir: Path,
    prepared_dir: Path,
    dataset: str | None = None,
    min_object_count: int = DEFAULT_MIN_OBJECT_COUNT,
    max_outside_ratio: float = DEFAULT_MAX_OUTSIDE_RATIO,
    center_outside_max_ratio: float = DEFAULT_CENTER_OUTSIDE_MAX_RATIO,
) -> Dict[str, Any]:
    prepared_dir.mkdir(parents=True, exist_ok=True)
    json_files = sorted(input_dir.glob("*.json"))

    total_removed_objects = 0
    scenes_with_removal = 0
    total_degenerate_objects = 0
    scenes_with_degenerate_objects = 0
    non_rectangular_boundary_scenes = 0
    degenerate_scenes: list[Dict[str, Any]] = []
    skipped_scenes: list[Dict[str, Any]] = []

    for json_file in tqdm(json_files, desc="过滤出界物体"):
        with open(json_file, "r", encoding="utf-8") as f:
            scene_data = json.load(f)

        if dataset == "spatiallm":
            keep_scene, skip_reason = preprocess_spatiallm_scene(scene_data)
            if not keep_scene:
                non_rectangular_boundary_scenes += 1
                skipped_scenes.append(
                    {
                        "file": json_file.name,
                        "remaining_object_count": len(scene_data.get("objects", [])),
                        "removed_out_of_bounds_count": 0,
                        "removed_collision_count": 0,
                        "reason": skip_reason,
                        "out_of_bounds_objects": [],
                        "degenerate_objects": [],
                    }
                )
                continue

            remaining_object_count = len(scene_data.get("objects", []))
            if min_object_count > 0 and remaining_object_count < min_object_count:
                skipped_scenes.append(
                    {
                        "file": json_file.name,
                        "remaining_object_count": remaining_object_count,
                        "removed_out_of_bounds_count": 0,
                        "removed_collision_count": 0,
                        "reason": f"object_count ({remaining_object_count}) < {min_object_count}",
                        "out_of_bounds_objects": [],
                        "degenerate_objects": [],
                    }
                )
                continue

            with open(prepared_dir / json_file.name, "w", encoding="utf-8") as f:
                json.dump(scene_data, f, indent=2, ensure_ascii=False)
            continue

        indices_to_remove, removal_details, degenerate_details = find_out_of_bounds_objects(
            scene_data,
            max_outside_ratio=max_outside_ratio,
            center_outside_max_ratio=center_outside_max_ratio,
        )
        total_degenerate_objects += len(degenerate_details)
        if degenerate_details:
            scenes_with_degenerate_objects += 1
            degenerate_scenes.append(
                {
                    "file": json_file.name,
                    "degenerate_object_count": len(degenerate_details),
                    "degenerate_objects": degenerate_details,
                }
            )
        removed_count = remove_objects_by_indices(scene_data, indices_to_remove)
        total_removed_objects += removed_count
        if removed_count > 0:
            scenes_with_removal += 1

        remaining_object_count = len(scene_data.get("objects", []))
        if min_object_count > 0 and remaining_object_count < min_object_count:
            skipped_scenes.append(
                {
                    "file": json_file.name,
                    "remaining_object_count": remaining_object_count,
                    "removed_out_of_bounds_count": removed_count,
                    "removed_collision_count": 0,
                    "reason": (
                        f"object_count ({remaining_object_count}) < {min_object_count} after boundary filter"
                    ),
                    "out_of_bounds_objects": removal_details,
                    "degenerate_objects": degenerate_details,
                }
            )
            continue

        with open(prepared_dir / json_file.name, "w", encoding="utf-8") as f:
            json.dump(scene_data, f, indent=2, ensure_ascii=False)

    return {
        "total_out_of_bounds_removed_objects": total_removed_objects,
        "scenes_with_out_of_bounds_removal": scenes_with_removal,
        "total_degenerate_objects": total_degenerate_objects,
        "scenes_with_degenerate_objects": scenes_with_degenerate_objects,
        "non_rectangular_boundary_scenes": non_rectangular_boundary_scenes,
        "degenerate_scenes": degenerate_scenes,
        "skipped_scenes": skipped_scenes,
    }


def filter_directory(
    input_dir: Path,
    output_dir: Path,
    collision_log_path: Path,
    dataset: str | None,
    collision_workers: int,
    force_collision: bool,
    disable_collision: bool,
    disable_floating: bool,
    min_object_count: int = DEFAULT_MIN_OBJECT_COUNT,
    log_path: Path | None = None,
    floating_log_path: Path | None = None,
    floating_checkpoint_dir: Path | None = None,
    max_outside_ratio: float = DEFAULT_MAX_OUTSIDE_RATIO,
    center_outside_max_ratio: float = DEFAULT_CENTER_OUTSIDE_MAX_RATIO,
    floating_tolerance: float = DEFAULT_FLOATING_TOLERANCE,
    blender_bin: str = "blender",
    floating_workers: int = DEFAULT_FLOATING_WORKERS,
    floating_asset_cache_size: int = DEFAULT_FLOATING_ASSET_CACHE_SIZE,
    collision_min_overlap_triangles: int = DEFAULT_BLENDER_COLLISION_MIN_OVERLAP_TRIANGLES,
    force_floating: bool = False,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if not input_dir.exists():
        raise ValueError(f"输入目录不存在: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    old_json_files = list(output_dir.glob("*.json"))
    if old_json_files:
        print(f"清理输出目录旧 JSON 文件: {len(old_json_files)} 个")
        for old_file in old_json_files:
            old_file.unlink()

    json_files = sorted(input_dir.glob("*.json"))
    print(f"找到 {len(json_files)} 个场景文件")
    if not json_files:
        return

    with tempfile.TemporaryDirectory(prefix="filter_stages_") as temp_dir:
        temp_root = Path(temp_dir)
        prepared_dir = temp_root / "boundary_filter"
        boundary_stats = prepare_boundary_filtered_inputs(
            input_dir,
            prepared_dir,
            dataset=dataset,
            min_object_count=min_object_count,
            max_outside_ratio=max_outside_ratio,
            center_outside_max_ratio=center_outside_max_ratio,
        )

        floating_stats: Dict[str, Any] = {
            "summary": {
                "total_floating_removed_objects": 0,
                "scenes_with_floating_removal": 0,
                "written_scenes": 0,
                "skipped_scenes": 0,
            },
            "floating_scenes": [],
            "skipped_scenes": [],
        }

        scenes_skipped = len(boundary_stats["skipped_scenes"])
        skipped_scenes = list(boundary_stats["skipped_scenes"])

        if not (disable_collision and disable_floating):
            resolved_floating_checkpoint_dir = (
                floating_checkpoint_dir
                if floating_checkpoint_dir is not None
                else output_dir.parent / DEFAULT_FLOATING_CHECKPOINT_DIRNAME
            )
            resolved_floating_log_path = (
                floating_log_path
                if floating_log_path is not None
                else output_dir.parent / DEFAULT_FLOATING_FILTER_LOG_FILENAME
            )
            floating_stats = run_floating_filter(
                input_dir=prepared_dir,
                output_dir=output_dir,
                log_path=resolved_floating_log_path,
                dataset=dataset,
                min_object_count=min_object_count,
                tolerance=floating_tolerance,
                blender_bin=blender_bin,
                floating_workers=floating_workers,
                floating_asset_cache_size=floating_asset_cache_size,
                collision_min_overlap_triangles=collision_min_overlap_triangles,
                disable_collision=disable_collision,
                disable_floating=disable_floating,
                checkpoint_dir=resolved_floating_checkpoint_dir,
                force_floating=force_floating or force_collision,
            )
            skipped_scenes.extend(floating_stats.get("skipped_scenes", []))
            scenes_skipped += len(floating_stats.get("skipped_scenes", []))
        else:
            for json_file in tqdm(sorted(prepared_dir.glob("*.json")), desc="复制边界过滤结果"):
                shutil.copy2(json_file, output_dir / json_file.name)

        floating_summary = floating_stats.get("summary", {})
        total_collision_removed_objects = int(
            floating_summary.get("total_collision_removed_objects", 0)
        )
        scenes_with_collision_removal = int(
            floating_summary.get("scenes_with_collision_removal", 0)
        )
        scenes_written = (
            int(floating_summary.get("written_scenes", 0))
            if not (disable_collision and disable_floating)
            else len(list(output_dir.glob("*.json")))
        )

    print("\n===== 过滤完成 =====")
    print(f"保留场景数: {scenes_written}")
    print(f"跳过场景数: {scenes_skipped}")
    print(f"删除出界物体总数: {boundary_stats['total_out_of_bounds_removed_objects']}")
    print(f"发生出界删除的场景数: {boundary_stats['scenes_with_out_of_bounds_removal']}")
    print(f"退化物体总数: {boundary_stats['total_degenerate_objects']}")
    print(f"包含退化物体的场景数: {boundary_stats['scenes_with_degenerate_objects']}")
    print(f"非矩形边界跳过场景数: {boundary_stats['non_rectangular_boundary_scenes']}")
    print(f"删除悬浮/穿地物体总数: {floating_summary.get('total_floating_removed_objects', 0)}")
    print(f"发生悬浮/穿地删除的场景数: {floating_summary.get('scenes_with_floating_removal', 0)}")
    print(
        "保留轻微漂浮物体总数: "
        f"{floating_summary.get('total_ignored_slight_floating_objects', 0)}"
    )
    print(
        "包含轻微漂浮保留的场景数: "
        f"{floating_summary.get('scenes_with_ignored_slight_floating', 0)}"
    )
    print(f"删除碰撞物体总数: {total_collision_removed_objects}")
    print(f"发生碰撞删除的场景数: {scenes_with_collision_removal}")

    if log_path is not None:
        log_payload = {
            "summary": {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "total_scenes": len(json_files),
                "written_scenes": scenes_written,
                "skipped_scenes": scenes_skipped,
                "total_out_of_bounds_removed_objects": boundary_stats["total_out_of_bounds_removed_objects"],
                "scenes_with_out_of_bounds_removal": boundary_stats["scenes_with_out_of_bounds_removal"],
                "total_degenerate_objects": boundary_stats["total_degenerate_objects"],
                "scenes_with_degenerate_objects": boundary_stats["scenes_with_degenerate_objects"],
                "non_rectangular_boundary_scenes": boundary_stats["non_rectangular_boundary_scenes"],
                "total_floating_removed_objects": floating_summary.get(
                    "total_floating_removed_objects", 0
                ),
                "scenes_with_floating_removal": floating_summary.get(
                    "scenes_with_floating_removal", 0
                ),
                "total_ignored_slight_floating_objects": floating_summary.get(
                    "total_ignored_slight_floating_objects", 0
                ),
                "scenes_with_ignored_slight_floating": floating_summary.get(
                    "scenes_with_ignored_slight_floating", 0
                ),
                "total_collision_removed_objects": total_collision_removed_objects,
                "scenes_with_collision_removal": scenes_with_collision_removal,
                "min_object_count": min_object_count,
                "max_outside_ratio": max_outside_ratio,
                "center_outside_max_ratio": center_outside_max_ratio,
                "floating_tolerance": floating_tolerance,
                "disable_floating": disable_floating,
            },
            "degenerate_scenes": boundary_stats["degenerate_scenes"],
            "floating_scenes": floating_stats.get("floating_scenes", []),
            "ignored_slight_floating_scenes": floating_stats.get(
                "ignored_slight_floating_scenes", []
            ),
            "collision_scenes": floating_stats.get("collision_scenes", []),
            "skipped_scenes": skipped_scenes,
        }
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_payload, f, indent=2, ensure_ascii=False)
        print(f"过滤日志已保存: {log_path}")


def copy_converted_scenes(
    input_dir: Path,
    output_dir: Path,
    log_path: Path | None = None,
) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if not input_dir.exists():
        raise ValueError(f"输入目录不存在: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    old_json_files = list(output_dir.glob("*.json"))
    if old_json_files:
        print(f"清理输出目录旧 JSON 文件: {len(old_json_files)} 个")
        for old_file in old_json_files:
            old_file.unlink()

    json_files = sorted(input_dir.glob("*.json"))
    for json_file in tqdm(json_files, desc="复制 converted_scenes"):
        shutil.copy2(json_file, output_dir / json_file.name)

    print("\n===== 复制完成 =====")
    print(f"复制场景数: {len(json_files)}")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")

    if log_path is not None:
        log_payload = {
            "summary": {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "total_scenes": len(json_files),
                "written_scenes": len(json_files),
                "skipped_scenes": 0,
                "copy_only": True,
            },
            "skipped_scenes": [],
        }
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_payload, f, indent=2, ensure_ascii=False)
        print(f"复制日志已保存: {log_path}")


def run_for_dataset(
    dataset: str,
    min_object_count: int,
    collision_workers: int,
    force_collision: bool,
    force_floating: bool,
    disable_collision: bool,
    disable_floating: bool,
    floating_tolerance: float,
    floating_workers: int,
    floating_asset_cache_size: int,
    collision_min_overlap_triangles: int,
    blender_bin: str,
    input_dir: str | None = None,
    output_dir: str | None = None,
    collision_log_path: str | None = None,
):
    resolved_input_dir, resolved_output_dir, resolved_collision_log_path = resolve_paths(
        dataset,
        input_dir=input_dir,
        output_dir=output_dir,
        collision_log_path=collision_log_path,
    )
    filter_log_path = resolved_output_dir.parent / "filtered_scenes_log.json"
    floating_log_path = resolved_output_dir.parent / DEFAULT_FLOATING_FILTER_LOG_FILENAME

    print("=" * 80)
    print(f"过滤数据集: {dataset}")
    print(f"输入目录: {resolved_input_dir}")
    print(f"输出目录: {resolved_output_dir}")
    print(f"最少物体数: {min_object_count}")
    print(f"出界阈值: outside_ratio > {DEFAULT_MAX_OUTSIDE_RATIO}")
    if dataset in {"imaginarium", "handcrafted"}:
        print("过滤模式: copy-only")
    else:
        print(f"Blender 碰撞过滤: {'关闭' if disable_collision else '开启'}")
        print(f"悬浮/穿地过滤: {'关闭' if disable_floating else '开启'}")
    if not (disable_collision and disable_floating) and dataset not in {"imaginarium", "handcrafted"}:
        print(f"Blender 过滤进程数: {floating_workers}")
        print(f"Blender 资产缓存/进程: {floating_asset_cache_size}")
        print(f"碰撞面片阈值: {collision_min_overlap_triangles}")
        print(f"Blender 过滤 checkpoint: {resolved_output_dir.parent / DEFAULT_FLOATING_CHECKPOINT_DIRNAME}")
    print("=" * 80)

    if dataset in {"imaginarium", "handcrafted"}:
        print("该数据集暂时不做出界/悬浮/碰撞过滤，仅复制 converted_scenes 到 filtered_scenes")
        copy_converted_scenes(
            input_dir=resolved_input_dir,
            output_dir=resolved_output_dir,
            log_path=filter_log_path,
        )
        return

    filter_directory(
        resolved_input_dir,
        resolved_output_dir,
        collision_log_path=resolved_collision_log_path,
        dataset=dataset,
        collision_workers=collision_workers,
        force_collision=force_collision,
        disable_collision=disable_collision,
        disable_floating=disable_floating,
        min_object_count=min_object_count,
        log_path=filter_log_path,
        floating_log_path=floating_log_path,
        max_outside_ratio=DEFAULT_MAX_OUTSIDE_RATIO,
        center_outside_max_ratio=DEFAULT_CENTER_OUTSIDE_MAX_RATIO,
        floating_tolerance=floating_tolerance,
        blender_bin=blender_bin,
        floating_workers=floating_workers,
        floating_asset_cache_size=floating_asset_cache_size,
        collision_min_overlap_triangles=collision_min_overlap_triangles,
        force_floating=force_floating,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统一过滤 converted_scenes 数据")
    parser.add_argument(
        "--dataset",
        type=str,
        default="3dfront",
        choices=["3dfront", "imaginarium", "spatiallm", "handcrafted", "custom"],
        help="数据集名称",
    )
    parser.add_argument("--input-dir", type=str, default=None, help="输入目录（custom 模式必填）")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录（custom 模式必填）")
    parser.add_argument(
        "--collision-log-path",
        type=str,
        default=None,
        help="碰撞日志路径（custom 模式可选）",
    )
    parser.add_argument(
        "--collision-workers",
        "-j",
        type=int,
        default=max(1, cpu_count() // 2),
        help="保留兼容旧命令；Blender 碰撞/悬浮过滤并行数请使用 --floating-workers",
    )
    parser.add_argument(
        "--min-object-count",
        type=int,
        default=DEFAULT_MIN_OBJECT_COUNT,
        help=f"过滤后最少保留的物体数（默认 {DEFAULT_MIN_OBJECT_COUNT}）",
    )
    parser.add_argument(
        "--disable-collision",
        action="store_true",
        help="仅按边界/物体数过滤，不执行碰撞过滤",
    )
    parser.add_argument(
        "--disable-floating",
        action="store_true",
        help="关闭 Blender 悬浮/穿地过滤",
    )
    parser.add_argument(
        "--floating-tolerance",
        type=float,
        default=DEFAULT_FLOATING_TOLERANCE,
        help=f"悬浮/穿地检测容差（米，默认 {DEFAULT_FLOATING_TOLERANCE}）",
    )
    parser.add_argument(
        "--blender-bin",
        type=str,
        default="blender",
        help="Blender 可执行文件路径（默认 blender）",
    )
    parser.add_argument(
        "--floating-workers",
        type=int,
        default=DEFAULT_FLOATING_WORKERS,
        help=f"悬浮/穿地过滤并行 Blender 进程数（默认 {DEFAULT_FLOATING_WORKERS}）",
    )
    parser.add_argument(
        "--floating-asset-cache-size",
        type=int,
        default=DEFAULT_FLOATING_ASSET_CACHE_SIZE,
        help=(
            "每个 Blender 进程缓存的资产 mesh 模板数量 "
            f"（默认 {DEFAULT_FLOATING_ASSET_CACHE_SIZE}，0 表示关闭）"
        ),
    )
    parser.add_argument(
        "--collision-min-overlap-triangles",
        type=int,
        default=DEFAULT_BLENDER_COLLISION_MIN_OVERLAP_TRIANGLES,
        help=(
            "Blender BVH 碰撞过滤的最少相交三角面数量 "
            f"（默认 {DEFAULT_BLENDER_COLLISION_MIN_OVERLAP_TRIANGLES}）"
        ),
    )
    parser.add_argument(
        "--force-collision",
        action="store_true",
        help="强制重新计算碰撞日志",
    )
    parser.add_argument(
        "--force-floating",
        action="store_true",
        help="强制清空悬浮/穿地过滤 checkpoint 并重新计算",
    )
    args = parser.parse_args()

    run_for_dataset(
        dataset=args.dataset,
        min_object_count=args.min_object_count,
        collision_workers=args.collision_workers,
        force_collision=args.force_collision,
        force_floating=args.force_floating,
        disable_collision=args.disable_collision,
        disable_floating=args.disable_floating,
        floating_tolerance=args.floating_tolerance,
        floating_workers=args.floating_workers,
        floating_asset_cache_size=args.floating_asset_cache_size,
        collision_min_overlap_triangles=args.collision_min_overlap_triangles,
        blender_bin=args.blender_bin,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        collision_log_path=args.collision_log_path,
    )
