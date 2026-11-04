"""
生成分离点云训练数据：
1) 为每个样本导出地板点云（floor_pointcloud）
2) 为每个已存在物体导出“完整变换后”的独立点云（按 existing_objects 顺序）
3) 导出新物体局部点云（按 gt size 缩放后平移到非负空间）
4) 生成训练 JSON，使用分离点云占位符格式
"""

import json
import random
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

# 添加项目根目录到 Python 路径

from config import ProjectPaths
from dataset_pipeline.utils import (
    cleanup_output_files,
    count_samples_in_file,
    iter_samples_from_json_file,
    parse_asset_pc_dir,
    recreate_output_directory,
    resolve_autoregressive_dir,
    resolve_num_workers,
)
from utils.message import SYSTEM_PROMPT, format_output, format_scene_input, load_internal_spaces_map
from utils.pointcloud import (
    build_pointcloud,
    load_asset_pointcloud,
    save_pointcloud_as_ply,
    sample_floor_pointcloud,
    translate_pointcloud_to_nonnegative,
    transform_normals,
    transform_pointcloud,
)


def _to_float3(values: Sequence[float]) -> np.ndarray:
    if values is None or len(values) != 3:
        raise ValueError(f"Expected 3 values, got {len(values)}")
    return np.asarray([float(values[0]), float(values[1]), float(values[2])], dtype=np.float32)


def _compute_scale_vec(asset_metadata: Dict[str, Any], target_size: Sequence[float]) -> np.ndarray:
    transformed_size = asset_metadata["transformed_size"]
    target = _to_float3(target_size)
    source = _to_float3(transformed_size)

    scale_vec = np.ones(3, dtype=np.float32)
    for i in range(3):
        if source[i] > 1e-8:
            scale_vec[i] = target[i] / source[i]
    return scale_vec


def export_sample_pointclouds(
    sample: Dict[str, Any],
    sample_id: int,
    asset_pointcloud_dir: Path | List[Path],
    output_dir: Path,
    floor_sample_points: int = 2048,
) -> Dict[str, Any]:
    """
    导出单个样本的分离点云文件，并返回构建训练样本所需路径
    """
    sample_dir = output_dir / f"sample_{sample_id:06d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    existing_objects = sample["existing_objects"]
    result: Dict[str, Any] = {
        "existing_total": len(existing_objects),
        "existing_exported": 0,
        "existing_missing": 0,
        "floor_exported": 0,
        "floor_missing": 0,
        "new_object_exported": 0,
        "new_object_missing": 0,
        "floor_pointcloud_path": "",
        "existing_pointcloud_paths": [],
        "new_object_pointcloud_path": "",
    }

    manifest: Dict[str, Any] = {
        "sample_id": sample_id,
        "scene_id": sample["scene_id"],
        "floor_pointcloud": None,
        "existing_objects": [],
        "new_object": None,
    }

    # 1) floor pointcloud
    boundary = sample["boundary"]
    floor_pc = sample_floor_pointcloud(boundary, num_points=floor_sample_points)
    if floor_pc is None or len(floor_pc) == 0:
        result["floor_missing"] = 1
        manifest["floor_pointcloud"] = {"status": "missing_or_empty_boundary"}
    else:
        floor_colors = np.full((len(floor_pc), 3), 0.5, dtype=np.float32)
        floor_normals = np.zeros((len(floor_pc), 3), dtype=np.float32)
        floor_normals[:, 2] = 1.0
        floor_pointcloud = build_pointcloud(floor_pc, floor_colors, floor_normals)
        floor_file = sample_dir / "floor.ply"
        save_pointcloud_as_ply(floor_pointcloud, floor_file)
        result["floor_exported"] = 1
        result["floor_pointcloud_path"] = str(floor_file)
        manifest["floor_pointcloud"] = {"status": "ok", "ply_file": floor_file.name}

    # 2) existing objects (按顺序)
    for obj_idx, obj in enumerate(existing_objects):
        jid = obj["jid"]
        center = _to_float3(obj["center"])
        rotation = _to_float3(obj["rotation"])
        size = _to_float3(obj["size"])

        try:
            points, colors, normals, metadata = load_asset_pointcloud(jid, asset_pointcloud_dir)
        except ValueError:
            result["existing_missing"] += 1
            manifest["existing_objects"].append(
                {
                    "object_idx": obj_idx,
                    "status": "missing_asset_pointcloud",
                }
            )
            continue

        scale_vec = _compute_scale_vec(metadata, size)
        points_transformed = transform_pointcloud(
            points,
            center.tolist(),
            rotation.tolist(),
            scale=scale_vec,
        )
        normals_transformed = transform_normals(normals, rotation.tolist())

        transformed_file = sample_dir / f"object_{obj_idx:03d}.ply"
        save_pointcloud_as_ply(
            build_pointcloud(points_transformed, colors, normals_transformed),
            transformed_file,
        )

        result["existing_exported"] += 1
        result["existing_pointcloud_paths"].append(str(transformed_file))
        manifest["existing_objects"].append(
            {
                "object_idx": obj_idx,
                "center": center.tolist(),
                "rotation": rotation.tolist(),
                "size": size.tolist(),
                "scale_vec": scale_vec.tolist(),
                "status": "ok",
                "ply_file": transformed_file.name,
            }
        )

    # 3) new object pointcloud (按 gt size 缩放，并平移到非负空间)
    new_object = sample["new_object"]
    new_jid = new_object["jid"]
    if not new_jid:
        result["new_object_missing"] = 1
        manifest["new_object"] = {"status": "missing_jid"}
    else:
        try:
            new_points, new_colors, new_normals, new_metadata = load_asset_pointcloud(
                new_jid, asset_pointcloud_dir
            )
            new_size = _to_float3(new_object["size"])
            new_scale_vec = _compute_scale_vec(new_metadata, new_size)
            new_points = transform_pointcloud(
                new_points, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], scale=new_scale_vec,
            )
            new_points = translate_pointcloud_to_nonnegative(new_points)
            new_object_id = len(existing_objects)
            new_pointcloud = build_pointcloud(new_points, new_colors, new_normals)
            new_file = sample_dir / "new_object.ply"
            save_pointcloud_as_ply(new_pointcloud, new_file)
            result["new_object_exported"] = 1
            result["new_object_pointcloud_path"] = str(new_file)
            manifest["new_object"] = {
                "object_id": new_object_id,
                "size": new_size.tolist(),
                "scale_vec": new_scale_vec.tolist(),
                "status": "ok",
                "initial_ply": new_file.name,
            }
        except ValueError:
            result["new_object_missing"] = 1
            manifest["new_object"] = {"status": "missing_asset_pointcloud"}

    # completeness: floor + all existing + new object 都成功
    result["is_complete"] = (
        result["floor_exported"] == 1
        and result["existing_missing"] == 0
        and result["new_object_exported"] == 1
    )

    with open(sample_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return result


def convert_to_training_format_with_separate_pointcloud(
    autoregressive_data: List[Dict[str, Any]],
    start_index: int = 0,
    asset_pointcloud_dir: Optional[Path | List[Path]] = None,
    training_pointcloud_dir: Optional[Path] = None,
    floor_sample_points: int = 2048,
    internal_spaces_map: Optional[Dict] = None,
    include_object_descriptions: bool = False,
    only_pointcloud: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    生成训练样本：
    - instruction 中使用分离占位符（floor + 每个 existing object + new object）
    - point_clouds 路径顺序严格对应占位符顺序
    """
    if asset_pointcloud_dir is None:
        raise ValueError("asset_pointcloud_dir 不能为空")
    if training_pointcloud_dir is None:
        raise ValueError("training_pointcloud_dir 不能为空")

    training_data: List[Dict[str, Any]] = []
    stats = {
        "total_samples": 0,
        "training_samples": 0,
        "skipped_incomplete_samples": 0,
        "existing_total": 0,
        "existing_exported": 0,
        "existing_missing": 0,
        "floor_exported": 0,
        "floor_missing": 0,
        "new_object_exported": 0,
        "new_object_missing": 0,
    }

    for idx, sample in enumerate(autoregressive_data):
        sample_id = start_index + idx
        stats["total_samples"] += 1

        export_result = export_sample_pointclouds(
            sample=sample,
            sample_id=sample_id,
            asset_pointcloud_dir=asset_pointcloud_dir,
            output_dir=training_pointcloud_dir,
            floor_sample_points=floor_sample_points,
        )

        stats["existing_total"] += export_result["existing_total"]
        stats["existing_exported"] += export_result["existing_exported"]
        stats["existing_missing"] += export_result["existing_missing"]
        stats["floor_exported"] += export_result["floor_exported"]
        stats["floor_missing"] += export_result["floor_missing"]
        stats["new_object_exported"] += export_result["new_object_exported"]
        stats["new_object_missing"] += export_result["new_object_missing"]

        if not export_result["is_complete"]:
            stats["skipped_incomplete_samples"] += 1
            continue

        existing_paths = export_result["existing_pointcloud_paths"]
        point_cloud_paths = [export_result["floor_pointcloud_path"]]
        point_cloud_paths.extend(existing_paths)
        point_cloud_paths.append(export_result["new_object_pointcloud_path"])

        training_sample = {
            "scene_id": sample["scene_id"],
            "system": SYSTEM_PROMPT,
            "instruction": format_scene_input(
                sample,
                use_separate_pointcloud=True,
                internal_spaces_map=internal_spaces_map,
                include_object_descriptions=include_object_descriptions,
                only_pointcloud=only_pointcloud,
            ),
            "output": format_output(sample),
            "point_clouds": point_cloud_paths,
            "expected_relations": sample["expected_relations"],
        }
        training_data.append(training_sample)
        stats["training_samples"] += 1

    return training_data, stats


def _process_single_file_worker(
    task: Tuple[Path, int],
    asset_pointcloud_dir: Path | List[Path],
    training_pointcloud_dir: Path,
    floor_sample_points: int,
    internal_spaces_map: Optional[Dict] = None,
    include_object_descriptions: bool = False,
    only_pointcloud: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    json_file, start_index = task
    autoregressive_data = iter_samples_from_json_file(json_file)
    return convert_to_training_format_with_separate_pointcloud(
        autoregressive_data=autoregressive_data,
        start_index=start_index,
        asset_pointcloud_dir=asset_pointcloud_dir,
        training_pointcloud_dir=training_pointcloud_dir,
        floor_sample_points=floor_sample_points,
        internal_spaces_map=internal_spaces_map,
        include_object_descriptions=include_object_descriptions,
        only_pointcloud=only_pointcloud,
    )


def _merge_stats(total: Dict[str, Any], current: Dict[str, Any]):
    for key in total.keys():
        total[key] += current[key]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="生成分离点云训练数据")
    parser.add_argument(
        "--dataset",
        type=str,
        default="3dish",
        help="数据集类型 (3dfront, imaginarium, spatiallm, 3dis, handcrafted, 3dish, custom 等)",
    )
    parser.add_argument("--dataset-dir", type=str, default=None, help="数据集目录")
    parser.add_argument(
        "--input-file",
        "-f",
        type=str,
        default=None,
        help="单个 autoregressive json 文件（优先于 --input-dir）",
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        default=None,
        help="autoregressive json 目录",
    )
    parser.add_argument(
        "--asset-pc-dir",
        type=str,
        default=None,
        help="资产点云目录；多目录可用逗号分隔",
    )
    parser.add_argument(
        "--training-pc-dir",
        type=str,
        default=None,
        help="分离点云导出目录",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="训练数据 JSON 输出目录",
    )
    parser.add_argument(
        "--output-json-name",
        type=str,
        default=None,
        help="训练数据 JSON 文件名",
    )
    parser.add_argument(
        "--floor-points",
        type=int,
        default=2048,
        help="地板采样点数（默认 2048）",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="并行工作进程数（默认为 CPU 核心数的一半）",
    )
    parser.add_argument(
        "--with-internal-spaces",
        action="store_true",
        default=False,
        help="在 instruction 中写入 internal_spaces（默认关闭）",
    )
    parser.add_argument(
        "--with-object-descriptions",
        action="store_true",
        default=False,
        help="在训练输入中写入 existing_objects/new_object 的 short_desc，并保留 instruction 中的物体描述（默认只使用 category 标签）",
    )
    parser.add_argument(
        "--only-pointcloud",
        action="store_true",
        default=False,
        help="不包含物体的 center, size, rotation 信息，仅提供点云占位符并剔除几何属性",
    )
    args = parser.parse_args()
    dataset = args.dataset.lower()
    include_object_descriptions = args.with_object_descriptions
    only_pointcloud = args.only_pointcloud
    ProjectPaths.ensure_dirs()

    asset_pointcloud_dir = parse_asset_pc_dir(dataset, args.asset_pc_dir)

    if args.training_pc_dir:
        training_pointcloud_dir = Path(args.training_pc_dir)
    else:
        training_pointcloud_dir = ProjectPaths.get_separate_training_pointcloud_dir(dataset)
    training_pointcloud_dir.mkdir(parents=True, exist_ok=True)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = ProjectPaths.DATA_PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_file:
        input_file = Path(args.input_file)
        if not input_file.exists():
            raise ValueError(f"输入文件不存在: {input_file}")
        json_files = [input_file]
        default_json_name = f"{input_file.stem}_only_pointcloud.json" if only_pointcloud else f"{input_file.stem}_with_separate_pointcloud.json"
    else:
        autoregressive_dir = resolve_autoregressive_dir(dataset, args.dataset_dir, args.input_dir)
        json_files = sorted(autoregressive_dir.glob("**/*.json"))
        random.seed(42)  # 与主生成脚本保持一致，固定种子可复现
        random.shuffle(json_files)
        if dataset == "custom" and args.input_dir:
            suffix = "_only_pointcloud.json" if only_pointcloud else "_separate_pointcloud.json"
            default_json_name = f"{Path(args.input_dir).name}{suffix}"
        else:
            suffix = "_only_pointcloud.json" if only_pointcloud else "_separate_pointcloud.json"
            default_json_name = f"{dataset}{suffix}"

    output_json_name = args.output_json_name or default_json_name
    output_json_file = output_dir / output_json_name
    cleanup_output_files(
        [output_json_file],
        description="训练数据 JSON 文件",
    )
    recreate_output_directory(training_pointcloud_dir, description="分离点云导出目录")

    print("=" * 60)
    print("生成分离点云训练数据")
    print("=" * 60)
    print(f"数据集: {dataset}")
    print(f"输入文件数: {len(json_files)}")
    if isinstance(asset_pointcloud_dir, list):
        print(f"资产点云目录: {[str(p) for p in asset_pointcloud_dir]}")
    else:
        print(f"资产点云目录: {asset_pointcloud_dir}")
    print(f"点云导出目录: {training_pointcloud_dir}")
    print(f"训练数据输出: {output_json_file}")
    print(f"地板采样点数: {args.floor_points}")
    print(f"并行进程数: {args.num_workers if args.num_workers is not None else 'auto'}")
    print(f"写入 internal_spaces: {'是' if args.with_internal_spaces else '否'}")
    print(f"写入物体描述: {'是' if include_object_descriptions else '否'}")
    print(f"仅保留点云模式: {'是' if only_pointcloud else '否'}")
    print("=" * 60)

    # 按需加载收纳空间数据（使用 Imaginarium 资产的数据集支持）
    internal_spaces_map = None
    if args.with_internal_spaces and dataset in ("imaginarium", "spatiallm", "3dis", "handcrafted", "3dish"):
        internal_spaces_map = load_internal_spaces_map(ProjectPaths.IMAGINARIUM_INTERNAL_SPACES_DIR)
        print(f"已加载 {len(internal_spaces_map)} 个资产的收纳空间数据")

    all_training_data: List[Dict[str, Any]] = []
    global_stats = {
        "total_samples": 0,
        "training_samples": 0,
        "skipped_incomplete_samples": 0,
        "existing_total": 0,
        "existing_exported": 0,
        "existing_missing": 0,
        "floor_exported": 0,
        "floor_missing": 0,
        "new_object_exported": 0,
        "new_object_missing": 0,
    }

    total_files = len(json_files)
    if total_files > 0:
        num_workers = resolve_num_workers(args.num_workers, total_files)

        print("扫描文件统计样本数量...")
        file_sample_counts: List[Tuple[Path, int, int]] = []
        cumulative_count = 0
        for json_file in tqdm(json_files, desc="统计样本数"):
            count = count_samples_in_file(json_file)
            file_sample_counts.append((json_file, cumulative_count, count))
            cumulative_count += count

        tasks = [(json_file, start_index) for json_file, start_index, _ in file_sample_counts]
        worker_func = partial(
            _process_single_file_worker,
            asset_pointcloud_dir=asset_pointcloud_dir,
            training_pointcloud_dir=training_pointcloud_dir,
            floor_sample_points=args.floor_points,
            internal_spaces_map=internal_spaces_map,
            include_object_descriptions=include_object_descriptions,
            only_pointcloud=only_pointcloud,
        )

        print(f"文件处理并行进程数: {num_workers}")
        with Pool(processes=num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(worker_func, tasks),
                    total=total_files,
                    desc="处理文件",
                )
            )

        for training_data, stats in results:
            all_training_data.extend(training_data)
            _merge_stats(global_stats, stats)

    with open(output_json_file, "w", encoding="utf-8") as f:
        json.dump(all_training_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("训练数据生成完成")
    print("=" * 60)
    print(f"总输入样本: {global_stats['total_samples']}")
    print(f"有效训练样本: {global_stats['training_samples']}")
    print(f"跳过样本(点云不完整): {global_stats['skipped_incomplete_samples']}")
    print(f"已存在物体总数: {global_stats['existing_total']}")
    print(f"已存在物体导出: {global_stats['existing_exported']}")
    print(f"已存在物体缺失: {global_stats['existing_missing']}")
    print(f"地板点云导出: {global_stats['floor_exported']}")
    print(f"地板点云缺失: {global_stats['floor_missing']}")
    print(f"新物体导出: {global_stats['new_object_exported']}")
    print(f"新物体缺失: {global_stats['new_object_missing']}")
    print(f"训练数据文件: {output_json_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()

