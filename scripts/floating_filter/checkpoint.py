#!/usr/bin/env python3
"""Checkpoint helpers for Blender-based floating object filtering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


def scene_checkpoint_path(log_dir: Path, json_file: Path) -> Path:
    return Path(log_dir) / Path(json_file).name


def select_shard_files(
    json_files: Sequence[Path],
    num_shards: int,
    shard_index: int,
) -> list[Path]:
    if num_shards < 1:
        raise ValueError("--num-shards 必须 >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index 必须在 [0, num_shards) 范围内")
    sorted_files = sorted(Path(path) for path in json_files)
    return [path for index, path in enumerate(sorted_files) if index % num_shards == shard_index]


def load_checkpoint_result(json_file: Path, log_dir: Path) -> dict[str, Any]:
    checkpoint_path = scene_checkpoint_path(log_dir, json_file)
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        return json.load(f)


def checkpoint_is_complete(
    json_file: Path,
    output_dir: Path,
    log_dir: Path,
) -> bool:
    checkpoint_path = scene_checkpoint_path(log_dir, json_file)
    if not checkpoint_path.exists():
        return False

    try:
        result = load_checkpoint_result(json_file, log_dir)
    except (OSError, json.JSONDecodeError):
        return False
    if result.get("skipped"):
        return True
    return (Path(output_dir) / Path(json_file).name).exists()


def count_completed_checkpoints(
    input_files: Sequence[Path],
    output_dir: Path,
    log_dir: Path,
) -> int:
    return sum(
        1
        for json_file in input_files
        if checkpoint_is_complete(json_file, output_dir=output_dir, log_dir=log_dir)
    )


def write_checkpoint_result(
    json_file: Path,
    log_dir: Path,
    result: dict[str, Any],
) -> None:
    checkpoint_path = scene_checkpoint_path(log_dir, json_file)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    temp_path.replace(checkpoint_path)


def prune_stale_checkpoints(
    input_files: Sequence[Path],
    output_dir: Path,
    log_dir: Path,
) -> None:
    current_names = {Path(path).name for path in input_files}
    for directory in (Path(output_dir), Path(log_dir)):
        if not directory.exists():
            continue
        for path in directory.glob("*.json"):
            if path.name not in current_names:
                path.unlink()


def aggregate_checkpoint_results(
    input_files: Sequence[Path],
    output_dir: Path,
    log_dir: Path,
    min_object_count: int,
    tolerance: float,
    dataset: str,
) -> dict[str, Any]:
    scene_logs: list[dict[str, Any]] = []
    skipped_scenes: list[dict[str, Any]] = []
    written_scenes = 0
    total_removed = 0
    scenes_with_removal = 0
    total_collision_removed = 0
    scenes_with_collision_removal = 0
    total_ignored = 0
    scenes_with_ignored = 0

    for json_file in sorted(Path(path) for path in input_files):
        if not checkpoint_is_complete(json_file, output_dir=output_dir, log_dir=log_dir):
            raise RuntimeError(f"悬浮过滤 checkpoint 不完整: {json_file.name}")

        result = load_checkpoint_result(json_file, log_dir)
        scene_logs.append(result)
        removed_count = int(result["removed_floating_count"])
        total_removed += removed_count
        if removed_count > 0:
            scenes_with_removal += 1
        collision_removed_count = int(result.get("removed_collision_count", 0))
        total_collision_removed += collision_removed_count
        if collision_removed_count > 0:
            scenes_with_collision_removal += 1
        ignored_count = int(result.get("ignored_floating_count", 0))
        total_ignored += ignored_count
        if ignored_count > 0:
            scenes_with_ignored += 1
        if result["skipped"]:
            skipped_scenes.append(result)
        else:
            written_scenes += 1

    return {
        "summary": {
            "input_dir": "",
            "output_dir": str(output_dir),
            "total_scenes": len(input_files),
            "written_scenes": written_scenes,
            "skipped_scenes": len(skipped_scenes),
            "total_floating_removed_objects": total_removed,
            "scenes_with_floating_removal": scenes_with_removal,
            "total_collision_removed_objects": total_collision_removed,
            "scenes_with_collision_removal": scenes_with_collision_removal,
            "total_ignored_slight_floating_objects": total_ignored,
            "scenes_with_ignored_slight_floating": scenes_with_ignored,
            "min_object_count": min_object_count,
            "tolerance": tolerance,
            "dataset": dataset,
        },
        "floating_scenes": [
            result for result in scene_logs if result["removed_floating_count"] > 0
        ],
        "collision_scenes": [
            result for result in scene_logs if result.get("removed_collision_count", 0) > 0
        ],
        "ignored_slight_floating_scenes": [
            result for result in scene_logs if result.get("ignored_floating_count", 0) > 0
        ],
        "skipped_scenes": skipped_scenes,
    }
