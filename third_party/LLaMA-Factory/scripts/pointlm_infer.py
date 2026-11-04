"""
PointLM 多进程并行推理脚本

每个 GPU 运行一个完全独立的进程，无 NCCL 通信开销。

Usage:
    # 8 卡并行推理
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/pointlm_infer.py \
        -d ../../datasets/processed/3dish_separate_pointcloud_331k_test.json \
        -m saves/train_full_sft/qwen3-0_6b/3dish_separate_pointcloud_331k_train \
        -o results/eval/predictions.json \
        --num_gpus 8

    # 消融实验：随机噪声点云
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/pointlm_infer.py \
        -d /path/to/test.json \
        -m saves/train_full_sft/qwen3-0_6b/.../checkpoint-1400 \
        -o results/predictions_noise.json \
        --num_gpus 8 --pcd_random_noise
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llamafactory.data.pcd_preprocess import resolve_point_cloud_path
from llamafactory.model.pointlm import GenerationConfig, PointLMInference

def _parse_sample(sample: dict):
    """Extract (prompt, gt, system, point_clouds) from a sample dict. Returns None if invalid."""
    sample_id = sample.get("sample_id")
    if sample_id is None:
        return None

    point_clouds = sample.get("point_clouds", [])
    if not isinstance(point_clouds, list):
        raise TypeError(
            f"Sample {sample_id} has invalid point_clouds: "
            f"expected list, got {type(point_clouds).__name__}"
        )
    point_clouds = [resolve_point_cloud_path(p) for p in point_clouds]

    if "instruction" in sample:
        prompt = sample["instruction"]
        gt = sample.get("output", "")
        system = sample.get("system", "You are a helpful assistant.")
    elif "conversations" in sample:
        prompt = sample["conversations"][0]["value"]
        gt = sample["conversations"][1]["value"] if len(sample["conversations"]) > 1 else ""
        system = "You are a helpful assistant."
    else:
        return None

    return prompt, gt, system, point_clouds


def run_worker(args, data: list[dict]):
    rank = args.distributed_rank if args.distributed_rank is not None else 0
    is_main = rank == 0
    batch_size = args.batch_size

    if is_main:
        print(f"[Rank {rank}] Loading model from {args.model_path} (batch_size={batch_size}) ...")

    inference = PointLMInference(
        model_path=args.model_path,
        inference_dtype=args.inference_dtype
    )
    inference.load()

    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_beams=args.num_beams,
    )

    # Pre-parse and validate all samples
    parsed_data: list[tuple] = []
    for sample in data:
        parsed = _parse_sample(sample)
        if parsed is None:
            continue
        prompt, gt, system, point_clouds = parsed
        if not args.pcd_random_noise:
            missing_paths = [p for p in point_clouds if not os.path.exists(p)]
            if missing_paths:
                raise FileNotFoundError(
                    f"Sample {sample['sample_id']} has missing point cloud files: {missing_paths}"
                )
        parsed_data.append((sample, prompt, gt, system, point_clouds))

    # Sort by estimated token count so samples with similar sequence lengths
    # land in the same batch, minimising padding waste.  When .npy caches
    # exist we use their shape[0] (= post-voxelisation point count, monotonic
    # with backbone output tokens); otherwise fall back to point cloud count.
    backbone_type = inference.object_backbone_type

    def _sort_key(item):
        _sample, prompt, _gt, _system, point_clouds = item
        total_pts = 0
        for p in point_clouds:
            npy = os.path.splitext(p)[0] + f".{backbone_type}.npy"
            if os.path.isfile(npy):
                total_pts += np.load(npy, mmap_mode="r").shape[0]
            else:
                total_pts += 1
        return (total_pts, len(prompt))

    parsed_data.sort(key=_sort_key)

    results = []
    num_batches = (len(parsed_data) + batch_size - 1) // batch_size
    progress = tqdm(total=len(parsed_data), desc=f"[Rank {rank}]") if is_main else None

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_items = parsed_data[batch_start : batch_start + batch_size]

        prompts = [item[1] for item in batch_items]
        pcd_paths_list = [item[4] if item[4] else None for item in batch_items]

        # Use the system prompt from the first sample (they are almost always
        # identical within a dataset; mixed system prompts are rare).
        system_prompt = batch_items[0][3]

        batch_results = inference.generate_batch(
            prompts=prompts,
            separate_point_cloud_paths_list=pcd_paths_list,
            system_prompt=system_prompt,
            generation_config=gen_config,
            pcd_random_noise=args.pcd_random_noise,
        )

        for (sample, prompt, gt, _system, point_clouds), result in zip(batch_items, batch_results):
            results.append({
                "sample_id": sample["sample_id"],
                "prompt": prompt,
                "label": gt,
                "predict": result.response,
                "point_clouds": point_clouds,
            })

        if progress is not None:
            progress.update(len(batch_items))
        elif (batch_start + len(batch_items)) % 200 < batch_size:
            done = batch_start + len(batch_items)
            print(f"[Rank {rank}] {done}/{len(parsed_data)} ({done / len(parsed_data) * 100:.1f}%)")

    if progress is not None:
        progress.close()

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[Rank {rank}] Done, {len(results)} results saved to {args.output}")


def cleanup_file(path: str):
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def run_distributed(args, total: int) -> bool:
    world_size = args.num_gpus
    visible_str = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible_ids = (
        [v.strip() for v in visible_str.split(",") if v.strip()]
        if visible_str.strip()
        else [str(i) for i in range(world_size)]
    )

    if len(visible_ids) < world_size:
        print(f"[WARN] Requested {world_size} GPUs but only {len(visible_ids)} visible, using {len(visible_ids)}")
        world_size = len(visible_ids)

    if world_size <= 1:
        return False

    print(f"[INFO] Launching {world_size} worker processes for {total} samples")
    cleanup_file(args.output)

    base = total // world_size
    rest = total % world_size
    processes = []
    shard_outputs = []
    offset = 0

    for rank in range(world_size):
        shard_len = base + (1 if rank < rest else 0)
        if shard_len <= 0:
            continue

        shard_start = args.start_idx + offset
        offset += shard_len

        root, ext = os.path.splitext(args.output)
        shard_out = f"{root}_shard{rank}{ext or '.json'}"
        cleanup_file(shard_out)
        shard_outputs.append(shard_out)

        child_args = [
            sys.executable, __file__,
            "-d", args.data_file,
            "-m", args.model_path,
            "-o", shard_out,
            "--start_idx", str(shard_start),
            "--num_samples", str(shard_len),
            "--num_gpus", "1",
            "--distributed_rank", str(rank),
            "--inference_dtype", args.inference_dtype,
            "--max_new_tokens", str(args.max_new_tokens),
            "--top_k", str(args.top_k),
            "--top_p", str(args.top_p),
            "--temperature", str(args.temperature),
            "--num_beams", str(args.num_beams),
            "--batch_size", str(args.batch_size),
        ]
        if args.pcd_random_noise:
            child_args.append("--pcd_random_noise")
        if args.do_sample:
            child_args.append("--do_sample")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = visible_ids[rank]

        print(
            f"  rank={rank}, gpu={visible_ids[rank]}, "
            f"samples=[{shard_start}:{shard_start + shard_len}], output={shard_out}"
        )
        p = subprocess.Popen(child_args, env=env)
        processes.append(p)

    exit_codes = [p.wait() for p in processes]
    if any(c != 0 for c in exit_codes):
        print(f"[ERROR] Worker exit codes: {exit_codes}")
        sys.exit(1)

    # Merge shards
    merged = []
    for shard_out in shard_outputs:
        if not os.path.exists(shard_out):
            print(f"[WARN] Missing shard: {shard_out}")
            continue
        with open(shard_out, "r", encoding="utf-8") as f:
            merged.extend(json.load(f))

    merged.sort(key=lambda x: x.get("sample_id", 0))
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] Merged {len(merged)} results -> {args.output}")

    for shard_out in shard_outputs:
        cleanup_file(shard_out)

    return True

def main():
    parser = argparse.ArgumentParser(description="PointLM parallel inference")
    parser.add_argument("-d", "--data_file", type=str, required=True)
    parser.add_argument("-m", "--model_path", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, default="predictions.json")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--distributed_rank", type=int, default=None)

    parser.add_argument("--inference_dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float32"])
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--pcd_random_noise", action="store_true", default=False,
                        help="Replace real point clouds with random noise (ablation)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Number of samples per batch for inference (default: 4)")

    args = parser.parse_args()

    print(f"Loading data from {args.data_file} ...")
    with open(args.data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    available = max(len(data) - args.start_idx, 0)
    total = min(args.num_samples, available) if args.num_samples is not None else available
    print(f"Total samples: {total} (index {args.start_idx} to {args.start_idx + total - 1})")

    if args.num_gpus > 1 and args.distributed_rank is None:
        if run_distributed(args, total):
            return

    shard = data[args.start_idx : args.start_idx + total]
    rank = args.distributed_rank if args.distributed_rank is not None else 0
    print(f"[Rank {rank}] Processing {len(shard)} samples")
    run_worker(args, shard)


if __name__ == "__main__":
    main()
