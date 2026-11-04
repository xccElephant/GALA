#!/usr/bin/env python3
"""Pre-compute point cloud features and save as .npy for faster training.

Scans one or more dataset JSON files, collects all unique .ply paths,
runs the preprocessing pipeline (Open3D load + grid-sample + normalise),
and saves the resulting float32 arrays as .npy files next to each .ply.

Naming convention:
    <original>.ply  ->  <original>.<backbone_type>.npy

The training-time code in ``PointLMPlugin.get_mm_inputs`` will detect
these .npy caches automatically and skip expensive on-the-fly processing.

Usage:
    # Scan a whole directory of JSON files
    find ../../datasets/pointclouds/ -name "*.utonia.npy" -delete
    python scripts/pointlm_cache_pcd.py \
        --data_files ../../datasets/processed/ \
        --backbone_type utonia \
        --utonia_grid_size 0.02 \
        --workers 32

    # Or specify individual files
    python scripts/pointlm_cache_pcd.py \
        --data_files ../../datasets/processed/3dish_separate_pointcloud_331k_train.json \
        --workers 16

    # Dry-run: only report how many files need caching
    python scripts/pointlm_cache_pcd.py \
        --data_files ../../datasets/processed \
        --dry_run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from llamafactory.data.pcd_preprocess import preprocess_object_point_cloud, resolve_point_cloud_path


def npy_cache_path(pcd_path: str, backbone_type: str) -> str:
    base, _ = os.path.splitext(pcd_path)
    return f"{base}.{backbone_type}.npy"


def _resolve_data_files(raw_paths: list[str]) -> list[str]:
    """Expand directories to all .json files inside them."""
    resolved: list[str] = []
    for p in raw_paths:
        if os.path.isdir(p):
            for fname in sorted(os.listdir(p)):
                if fname.endswith(".json"):
                    resolved.append(os.path.join(p, fname))
        else:
            resolved.append(p)
    return resolved


def _collect_ply_paths(data_files: list[str], pcd_column: str) -> tuple[set[str], int]:
    """Scan dataset JSON files and return (deduplicated .ply paths, skipped_file_count)."""
    paths: set[str] = set()
    skipped = 0
    for data_file in data_files:
        file_has_pcd = False
        with open(data_file) as f:
            records = json.load(f)
        for record in records:
            pcd_list = record.get(pcd_column)
            if not pcd_list:
                continue
            if isinstance(pcd_list, str):
                pcd_list = [pcd_list]
            for p in pcd_list:
                p = str(p).strip()
                if p:
                    paths.add(resolve_point_cloud_path(p))
                    file_has_pcd = True
        if not file_has_pcd:
            skipped += 1
            print(f"  Skipped (no point clouds): {data_file}")
    return paths, skipped


_worker_args: dict = {}


def _init_worker(args: dict) -> None:
    global _worker_args
    _worker_args = args


def _process_one(pcd_path: str) -> tuple[str, bool, str]:
    """Process a single .ply -> .npy. Returns (path, success, message)."""
    args = _worker_args
    cache = npy_cache_path(pcd_path, args["backbone_type"])

    if not args["force"] and os.path.isfile(cache):
        return pcd_path, True, "skipped (cached)"

    try:
        result = preprocess_object_point_cloud(
            pcd_path,
            backbone_type=args["backbone_type"],
            floor_grid_scale=args["floor_grid_scale"],
            utonia_scale=args["utonia_scale"],
            utonia_grid_size=args["utonia_grid_size"],
            utonia_normalize_coord=args["utonia_normalize_coord"],
            utonia_center_shift=args["utonia_center_shift"],
        )
        np.save(cache, result)
        return pcd_path, True, f"ok ({result.shape[0]} pts, {result.shape[1]} dims)"
    except Exception as e:
        return pcd_path, False, str(e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-cache point cloud features as .npy")
    parser.add_argument("--data_files", nargs="+", required=True,
                        help="Dataset JSON files or directories containing JSON files")
    parser.add_argument("--pcd_column", default="point_clouds", help="JSON key for point cloud paths")
    parser.add_argument("--backbone_type", default="utonia", choices=["utonia"],
                        help="Backbone type; also used as the .npy cache filename suffix")
    parser.add_argument("--workers", type=int, default=min(32, cpu_count()), help="Number of parallel workers")
    parser.add_argument("--force", action="store_true", help="Overwrite existing .npy caches")
    parser.add_argument("--dry_run", action="store_true", help="Only report counts, don't process")

    parser.add_argument("--floor_grid_scale", type=float, default=2.0)
    parser.add_argument("--utonia_scale", type=float, default=1.0)
    parser.add_argument("--utonia_grid_size", type=float, default=0.02)
    parser.add_argument("--utonia_normalize_coord", action="store_true")
    parser.add_argument("--utonia_center_shift", action="store_true")
    args = parser.parse_args()

    data_files = _resolve_data_files(args.data_files)
    print(f"Scanning {len(data_files)} JSON file(s) ...")
    all_paths, skipped = _collect_ply_paths(data_files, args.pcd_column)
    print(f"Found {len(all_paths)} unique point cloud paths ({skipped} JSON file(s) had no point clouds)")

    if not args.force:
        need_cache = [p for p in all_paths if not os.path.isfile(npy_cache_path(p, args.backbone_type))]
        already = len(all_paths) - len(need_cache)
        print(f"  Already cached: {already}, need processing: {len(need_cache)}")
    else:
        need_cache = list(all_paths)

    if args.dry_run:
        print("Dry run, exiting.")
        return

    if not need_cache:
        print("All files are already cached, nothing to do.")
        return

    worker_args = {
        "backbone_type": args.backbone_type,
        "floor_grid_scale": args.floor_grid_scale,
        "utonia_scale": args.utonia_scale,
        "utonia_grid_size": args.utonia_grid_size,
        "utonia_normalize_coord": args.utonia_normalize_coord,
        "utonia_center_shift": args.utonia_center_shift,
        "force": args.force,
    }

    t0 = time.time()
    ok_count, fail_count = 0, 0

    with Pool(args.workers, initializer=_init_worker, initargs=(worker_args,)) as pool:
        for i, (path, success, msg) in enumerate(pool.imap_unordered(_process_one, need_cache), 1):
            if success:
                ok_count += 1
            else:
                fail_count += 1
                print(f"  FAIL: {path} — {msg}")
            if i % 5000 == 0 or i == len(need_cache):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                print(f"  [{i}/{len(need_cache)}] {rate:.0f} files/s, ok={ok_count}, fail={fail_count}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. OK: {ok_count}, Failed: {fail_count}")


if __name__ == "__main__":
    main()
