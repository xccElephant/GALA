"""
生成 Stage-1 对齐训练数据（基于分离点云）。

复用 04_ 已导出的点云文件，生成用于预训练 point_proj 的对齐数据。
任务的答案只能从点云中获得，迫使 projector 学会将点云特征
映射到 LLM 嵌入空间中有意义的位置。

任务类型:
  1. captioning      — 点云 → 物体外观描述
  2. grounded_bbox   — 点云 → 空间属性 (center + size)
  3. center          — 点云 → 中心坐标
  4. obj_bbox        — 点云 → 完整 bbox
  5. rotation        — 点云 → 旋转角度
  6. size            — 点云 → 尺寸
  7. room_boundary   — 地面点云 → 房间边界
  8. room_bbox       — 地面点云 → 房间 AABB
  9. pairwise_dist   — 两物体点云 → 距离
  10. pairwise_dir   — 两物体点云 → 相对方位

数据源:
  - autoregressive JSON (与 04_ 相同): 包含 jid, desc, center, size, rotation
  - Imaginarium asset CSV: name_en → caption_en (作为描述备选)

Usage:
    # 复用已有点云（02_ 已导出），只生成对齐训练 JSON
    python dataset_pipeline/generation/03_generate_alignment_data.py \\
        --dataset 3dish --max-samples 500000
"""

import csv
import json
import math
import random
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from config import ProjectPaths
from dataset_pipeline.utils import (
    cleanup_output_files,
    count_samples_in_file,
    iter_samples_from_json_file,
    resolve_autoregressive_dir,
    resolve_num_workers,
)
from utils.message import _round_boundary, _round_list, _normalize_rotation


# ──────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────

ALIGNMENT_SYSTEM_PROMPT = (
    "You are a Point Cloud Understanding assistant."
)


# ──────────────────────────────────────────────────────────────────
# Description loading
# ──────────────────────────────────────────────────────────────────

def load_imaginarium_captions(asset_csv: Path) -> Dict[str, str]:
    """Load name_en → caption_en from Imaginarium asset CSV."""
    mapping: Dict[str, str] = {}
    if not asset_csv.exists():
        return mapping
    with open(asset_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("name_en", "").strip()
            cap = row.get("caption_en", "").strip()
            if name and cap:
                mapping[name] = cap
    return mapping


def _get_desc(obj: Dict[str, Any], imaginarium_captions: Dict[str, str]) -> str:
    """获取物体描述。优先级: obj["desc"] > imaginarium caption > ""."""
    desc = obj.get("desc", "")
    if desc:
        return desc
    else:
        print(f"Warning: no desc for object {obj['id']}, jid: {obj['jid']}")
    jid = obj.get("jid", "")
    if jid and jid in imaginarium_captions:
        return imaginarium_captions[jid]
    else:
        print(f"Warning: no caption for jid {jid}")
    return ""


# ──────────────────────────────────────────────────────────────────
# Task generators (每个返回 List[Dict])
# ──────────────────────────────────────────────────────────────────

_CAPTION_PROMPTS = [
    "Describe this 3D object based on its point cloud. Include shape, structure, and any notable geometric features.",

    "What is this object? Describe its appearance and geometry based on the point cloud.",

    "Based on the point cloud, describe this 3D object. Focus on its overall shape and key structural details.",

    "Analyze this point cloud and provide a detailed description of the 3D object it represents.",

    "Look at this point cloud and tell me what object it depicts. Describe its form, structure, and visual characteristics.",
]


def _is_captioning_instruction(instruction: str) -> bool:
    """判断 instruction 是否为 captioning 任务。"""
    return any(prompt in instruction for prompt in _CAPTION_PROMPTS)


def _gen_captioning(pcd_path: str, desc: str) -> List[Dict]:
    prompt = random.choice(_CAPTION_PROMPTS)
    return [{
        "system": ALIGNMENT_SYSTEM_PROMPT,
        "instruction": (
            '{"object_pointcloud": "<object_pointcloud>"}\n\n' + prompt
        ),
        "output": desc,
        "point_clouds": [pcd_path],
    }]


def _gen_grounded_bbox(obj: Dict, pcd_path: str) -> List[Dict]:
    """点云 → 空间属性 (center + size)。"""
    center = obj.get("center")
    size = obj.get("size")
    if not center or not size:
        return []
    return [{
        "system": ALIGNMENT_SYSTEM_PROMPT,
        "instruction": (
            '{"object_pointcloud": "<object_pointcloud>"}\n\n'
            "Based on the point cloud of this object, return its "
            "center position (x, y, z) and size (width, depth, height) "
            "as [x, y, z, w, d, h]."
        ),
        "output": json.dumps(_round_list(center) + _round_list(size)),
        "point_clouds": [pcd_path],
    }]


def _gen_spatial(obj: Dict, pcd_path: str) -> List[Dict]:
    tasks = []
    center = obj.get("center")
    size = obj.get("size")
    rotation = obj.get("rotation")

    if center is None or size is None:
        return tasks

    tasks.append({
        "system": ALIGNMENT_SYSTEM_PROMPT,
        "instruction": (
            '{"object_pointcloud": "<object_pointcloud>"}\n\n'
            "What is the center position (x, y, z) of this object?"
        ),
        "output": json.dumps(_round_list(center)),
        "point_clouds": [pcd_path],
    })

    if rotation is not None:
        tasks.append({
            "system": ALIGNMENT_SYSTEM_PROMPT,
            "instruction": (
                '{"object_pointcloud": "<object_pointcloud>"}\n\n'
                "What is the rotation (roll, pitch, yaw) of this object in degrees?"
            ),
            "output": json.dumps(_normalize_rotation(rotation)),
            "point_clouds": [pcd_path],
        })

    return tasks


def _gen_floor(boundary: List, floor_pcd_path: str) -> List[Dict]:
    tasks = []
    rounded_boundary = _round_boundary(boundary)

    tasks.append({
        "system": ALIGNMENT_SYSTEM_PROMPT,
        "instruction": (
            '{"floor_pointcloud": "<floor_pointcloud>"}\n\n'
            "What is the room boundary polygon? "
            "Return a list of [x, y] vertex coordinates."
        ),
        "output": json.dumps(rounded_boundary),
        "point_clouds": [floor_pcd_path],
    })

    xs = [v[0] for v in boundary]
    ys = [v[1] for v in boundary]
    room_bbox = [round(min(xs), 2), round(min(ys), 2),
                 round(max(xs), 2), round(max(ys), 2)]
    tasks.append({
        "system": ALIGNMENT_SYSTEM_PROMPT,
        "instruction": (
            '{"floor_pointcloud": "<floor_pointcloud>"}\n\n'
            "What is the axis-aligned bounding box of this room? "
            "Return [x_min, y_min, x_max, y_max]."
        ),
        "output": json.dumps(room_bbox),
        "point_clouds": [floor_pcd_path],
    })

    return tasks


def _gen_pairwise(
    obj_a: Dict, pcd_a: str,
    obj_b: Dict, pcd_b: str,
) -> List[Dict]:
    tasks = []
    ca = obj_a.get("center")
    cb = obj_b.get("center")
    if ca is None or cb is None:
        return tasks

    dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(ca, cb)))
    tasks.append({
        "system": ALIGNMENT_SYSTEM_PROMPT,
        "instruction": (
            '{"object_A": {"object_pointcloud": "<object_pointcloud>"}, '
            '"object_B": {"object_pointcloud": "<object_pointcloud>"}}\n\n'
            "What is the Euclidean distance between these two objects in meters?"
        ),
        "output": json.dumps(round(dist, 2)),
        "point_clouds": [pcd_a, pcd_b],
    })

    tasks.append({
        "system": ALIGNMENT_SYSTEM_PROMPT,
        "instruction": (
            '{"object_A": {"object_pointcloud": "<object_pointcloud>"}, '
            '"object_B": {"object_pointcloud": "<object_pointcloud>"}}\n\n'
            "What is the relative position of object A to object B? "
            "Return [dx, dy, dz] in meters."
        ),
        "output": json.dumps(_round_list([ca[i] - cb[i] for i in range(3)])),
        "point_clouds": [pcd_a, pcd_b],
    })

    return tasks



# ──────────────────────────────────────────────────────────────────
# Per-sample processing
# ──────────────────────────────────────────────────────────────────

def _generate_tasks_for_sample(
    sample: Dict[str, Any],
    sample_id: int,
    training_pc_dir: Path,
    imaginarium_captions: Dict[str, str],
) -> List[Dict]:
    """Generate all alignment tasks for one autoregressive sample."""
    sample_dir = training_pc_dir / f"sample_{sample_id:06d}"
    scene_id = sample["scene_id"]
    existing_objects = sample["existing_objects"]
    new_object = sample["new_object"]
    boundary = sample["boundary"]

    floor_pcd = str(sample_dir / "floor.ply")
    existing_pcds = [str(sample_dir / f"object_{i:03d}.ply") for i in range(len(existing_objects))]
    new_obj_pcd = str(sample_dir / "new_object.ply")

    if not Path(floor_pcd).exists():
        return []
    for p in existing_pcds:
        if not Path(p).exists():
            return []
    if not Path(new_obj_pcd).exists():
        return []

    all_tasks: List[Dict] = []

    # Floor tasks — 每个 sample 只产生 1 个（随机选）
    floor_tasks = _gen_floor(boundary, floor_pcd)
    if floor_tasks:
        all_tasks.append(random.choice(floor_tasks))

    # Per-object tasks — captioning 全保留
    for i, obj in enumerate(existing_objects):
        if i >= len(existing_pcds):
            break
        desc = _get_desc(obj, imaginarium_captions)
        if desc:
            all_tasks.extend(_gen_captioning(existing_pcds[i], desc))

    # spatial + grounded_bbox 任务：只对 1 个随机物体生成（避免每个物体都生成多条 spatial）
    spatial_indices = list(range(min(len(existing_objects), len(existing_pcds))))
    if spatial_indices:
        si = random.choice(spatial_indices)
        all_tasks.extend(_gen_spatial(existing_objects[si], existing_pcds[si]))
        all_tasks.extend(_gen_grounded_bbox(existing_objects[si], existing_pcds[si]))

    # New object tasks
    new_desc = _get_desc(new_object, imaginarium_captions)
    if new_desc:
        all_tasks.extend(_gen_captioning(new_obj_pcd, new_desc))
    all_tasks.extend(_gen_grounded_bbox(new_object, new_obj_pcd))

    new_size = new_object.get("size")
    if new_size:
        all_tasks.append({
            "system": ALIGNMENT_SYSTEM_PROMPT,
            "instruction": (
                '{"new_object_pointcloud": "<new_object_pointcloud>"}\n\n'
                "What is the size (width, depth, height) of this object?"
            ),
            "output": json.dumps(_round_list(new_size)),
            "point_clouds": [new_obj_pcd],
        })

    # Pairwise tasks — 最多 1 对
    if len(existing_objects) >= 2 and len(existing_pcds) >= 2:
        pairs = [
            (i, j)
            for i in range(len(existing_objects))
            for j in range(i + 1, len(existing_objects))
            if i < len(existing_pcds) and j < len(existing_pcds)
        ]
        if pairs:
            ia, ib = random.choice(pairs)
            all_tasks.extend(_gen_pairwise(
                existing_objects[ia], existing_pcds[ia],
                existing_objects[ib], existing_pcds[ib],
            ))

    for task in all_tasks:
        task["scene_id"] = scene_id

    return all_tasks


# ──────────────────────────────────────────────────────────────────
# File-level worker (for multiprocessing)
# ──────────────────────────────────────────────────────────────────

def _process_file_worker(
    task: Tuple[Path, int],
    training_pc_dir: Path,
    imaginarium_captions: Dict[str, str],
) -> Tuple[List[Dict], int, int]:
    """Process one autoregressive JSON file.

    Returns (tasks, n_with_desc, n_without_desc).
    """
    json_file, start_index = task
    data = iter_samples_from_json_file(json_file)

    all_tasks: List[Dict] = []
    n_with_desc = 0
    n_without_desc = 0

    for idx, sample in enumerate(data):
        sample_id = start_index + idx
        tasks = _generate_tasks_for_sample(
            sample=sample,
            sample_id=sample_id,
            training_pc_dir=training_pc_dir,
            imaginarium_captions=imaginarium_captions,
        )

        has_caption = any(_is_captioning_instruction(t["instruction"]) for t in tasks)
        if has_caption:
            n_with_desc += 1
        elif tasks:
            n_without_desc += 1

        all_tasks.extend(tasks)

    return all_tasks, n_with_desc, n_without_desc


def assign_alignment_sample_ids(tasks: List[Dict]) -> None:
    """Assign numeric sample IDs to final alignment tasks."""
    for i, task in enumerate(tasks):
        task["sample_id"] = i


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="生成 Stage-1 对齐训练数据（分离点云）"
    )
    parser.add_argument(
        "--dataset", type=str,
        default="3dish",
        help="数据集类型 (3dfront, imaginarium, spatiallm, 3dis, handcrafted, 3dish, custom 等)",
    )
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--input-file", "-f", type=str, default=None,
                        help="单个 autoregressive JSON 文件")
    parser.add_argument("--input-dir", "-i", type=str, default=None,
                        help="autoregressive JSON 目录")
    parser.add_argument("--training-pc-dir", type=str, default=None,
                        help="分离点云目录（02_ 的输出）")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出 JSON 目录")
    parser.add_argument("--output-json-name", type=str, default=None)
    parser.add_argument("--asset-csv", type=str, default=None,
                        help="Imaginarium asset_info.csv 路径")
    parser.add_argument("--max-samples", type=int, default=500000,
                        help="最大输出任务数")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="并行工作进程数（默认为 CPU 核心数的一半）",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    dataset = args.dataset.lower()
    ProjectPaths.ensure_dirs()

    # ---- Resolve input files ----
    if args.input_file:
        input_file = Path(args.input_file)
        if not input_file.exists():
            raise ValueError(f"输入文件不存在: {input_file}")
        json_files = [input_file]
        default_json_name = f"{input_file.stem}_alignment.json"
    else:
        autoregressive_dir = resolve_autoregressive_dir(
            dataset, args.dataset_dir, args.input_dir
        )
        json_files = sorted(autoregressive_dir.glob("**/*.json"))
        random.seed(42)
        random.shuffle(json_files)
        default_json_name = f"{dataset}_separate_pointcloud_alignment.json"

    # ---- Resolve dirs ----
    training_pc_dir = (
        Path(args.training_pc_dir) if args.training_pc_dir
        else ProjectPaths.get_separate_training_pointcloud_dir(dataset)
    )
    training_pc_dir.mkdir(parents=True, exist_ok=True)

    output_dir = Path(args.output_dir) if args.output_dir else ProjectPaths.DATA_PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json_file = output_dir / (args.output_json_name or default_json_name)
    cleanup_output_files(
        [output_json_file],
        description="对齐训练数据 JSON 文件",
    )

    asset_csv_path = (
        Path(args.asset_csv) if args.asset_csv
        else ProjectPaths.IMAGINARIUM_ASSETS_DIR / "imaginarium_asset_info.csv"
    )

    # ---- Load captions ----
    print("加载 Imaginarium 资产描述...")
    imaginarium_captions = load_imaginarium_captions(asset_csv_path)
    print(f"  已加载 {len(imaginarium_captions)} 条 name_en → caption_en 映射")

    # ---- Summary ----
    print("=" * 60)
    print("生成 Stage-1 对齐训练数据")
    print("=" * 60)
    print(f"数据集: {dataset}")
    print(f"输入文件数: {len(json_files)}")
    print(f"点云目录: {training_pc_dir}")
    print("点云策略: 复用 02_ 已导出的点云")
    print(f"输出文件: {output_json_file}")
    print("=" * 60)

    # ---- Count samples for start_index ----
    print("扫描文件统计样本数量...")
    file_meta: List[Tuple[Path, int, int]] = []
    cumulative = 0
    for jf in tqdm(json_files, desc="统计样本数"):
        count = count_samples_in_file(jf)
        file_meta.append((jf, cumulative, count))
        cumulative += count
    print(f"总输入样本: {cumulative}")

    # ---- Process ----
    tasks_input = [(jf, start) for jf, start, _ in file_meta]

    num_workers = resolve_num_workers(args.num_workers, len(tasks_input))

    worker_func = partial(
        _process_file_worker,
        training_pc_dir=training_pc_dir,
        imaginarium_captions=imaginarium_captions,
    )

    print(f"处理文件（{num_workers} 进程）...")
    if num_workers > 1:
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(worker_func, tasks_input),
                total=len(tasks_input),
                desc="处理文件",
            ))
    else:
        results = [worker_func(t) for t in tqdm(tasks_input, desc="处理文件")]

    all_tasks: List[Dict] = []
    total_with_desc = 0
    total_without_desc = 0
    for tasks_batch, n_with, n_without in results:
        all_tasks.extend(tasks_batch)
        total_with_desc += n_with
        total_without_desc += n_without

    print(f"\n总任务数: {len(all_tasks)}")
    print(f"有描述的样本: {total_with_desc}, 无描述: {total_without_desc} "
          f"({total_with_desc / max(1, total_with_desc + total_without_desc) * 100:.1f}% 命中)")

    # ---- Subsample ----
    if args.max_samples and len(all_tasks) > args.max_samples:
        random.shuffle(all_tasks)
        all_tasks = all_tasks[:args.max_samples]
        print(f"采样至 {args.max_samples} 条")

    # ---- Assign ID ----
    assign_alignment_sample_ids(all_tasks)

    # ---- Save ----
    with open(output_json_file, "w", encoding="utf-8") as f:
        json.dump(all_tasks, f, ensure_ascii=False, indent=2)

    # ---- Print distribution ----
    dist: Dict[str, int] = {}
    for t in all_tasks:
        instr = t["instruction"]
        if _is_captioning_instruction(instr):
            key = "captioning"
        elif "Based on the point cloud" in instr:
            key = "grounded_bbox"
        elif "center position" in instr:
            key = "center"
        elif "bounding box of this object" in instr:
            key = "obj_bbox"
        elif "size (width" in instr:
            key = "size"
        elif "rotation" in instr:
            key = "rotation"
        elif "boundary polygon" in instr:
            key = "room_boundary"
        elif "bounding box of this room" in instr:
            key = "room_bbox"
        elif "distance" in instr:
            key = "pairwise_dist"
        elif "relative position" in instr:
            key = "pairwise_dir"
        else:
            key = "other"
        dist[key] = dist.get(key, 0) + 1

    print("\n任务分布:")
    for k, v in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {k:20s}: {v:6d} ({v / len(all_tasks) * 100:.1f}%)")

    print(f"\n已保存至 {output_json_file}")


if __name__ == "__main__":
    main()
