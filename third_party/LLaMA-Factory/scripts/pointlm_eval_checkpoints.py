"""
Run automatic evaluation for PointLM checkpoints (all checkpoints or latest only).

For each evaluated checkpoint this script:
1. runs ``pointlm_infer.py`` on a fixed evaluation dataset
2. runs ``pointlm_eval_score.py`` to compute physical and semantic metrics
3. aggregates checkpoint summaries into JSON and CSV leaderboards (in ``all`` mode)
4. optionally uploads the evaluation leaderboard to Weights & Biases

Usage:
    python scripts/pointlm_eval_checkpoints.py \
        --model_dir saves/train_full_sft/qwen3-0_6b/3dish_separate_pointcloud_331k_train \
        --train_yaml examples/train_full/qwen3-0_6b/3dish_separate_pointcloud_331k_train/base.yaml \
        --output_dir results/eval/qwen3-0_6b/3dish_separate_pointcloud_331k_train \
        --eval_mode all \
        --num_gpus 8 \
        --batch_size 128 \
        --score_workers 32
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys

import yaml


def _checkpoint_sort_key(path: str):
    name = os.path.basename(path.rstrip("/"))
    match = re.search(r"checkpoint-(\d+)$", name)
    if match:
        return (0, int(match.group(1)))
    return (1, name)


def _discover_checkpoints(model_dir: str) -> list[str]:
    pattern = os.path.join(model_dir, "checkpoint-*")
    checkpoints = [p for p in glob.glob(pattern) if os.path.isdir(p)]
    checkpoints.sort(key=_checkpoint_sort_key)
    return checkpoints


def _resolve_data_file(args: argparse.Namespace) -> str:
    if args.data_file:
        return os.path.abspath(args.data_file)

    if not args.train_yaml:
        raise ValueError("Must provide either --data_file or --train_yaml")

    with open(args.train_yaml, "r", encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)

    eval_dataset = train_cfg.get("eval_dataset")
    if not eval_dataset:
        raise ValueError(f"No eval_dataset found in {args.train_yaml}")

    dataset_info_path = os.path.abspath(args.dataset_info_json)
    with open(dataset_info_path, "r", encoding="utf-8") as f:
        dataset_info = json.load(f)

    entry = dataset_info.get(eval_dataset)
    if not entry or not entry.get("file_name"):
        raise ValueError(
            f"Dataset {eval_dataset!r} not found in dataset info {dataset_info_path}"
        )

    file_name = entry["file_name"]
    if not os.path.isabs(file_name):
        file_name = os.path.join(os.path.dirname(dataset_info_path), file_name)
    return os.path.abspath(file_name)


def _run_command(command: list[str], env: dict | None = None):
    print("\n[RUN]", " ".join(command))
    subprocess.run(command, check=True, env=env)


def _write_leaderboard(rows: list[dict], output_dir: str) -> tuple[str, str]:
    leaderboard_json = os.path.join(output_dir, "leaderboard.json")
    leaderboard_csv = os.path.join(output_dir, "leaderboard.csv")

    sorted_rows = sorted(
        rows,
        key=lambda row: (
            -(row.get("strict_valid_rate") or -1.0),
            -(row.get("within_strict_valid_rate") or -1.0),
            row.get("checkpoint_step") or 10**18,
        ),
    )

    with open(leaderboard_json, "w", encoding="utf-8") as f:
        json.dump(sorted_rows, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "checkpoint_name",
        "checkpoint_step",
        "model_path",
        "matched_samples",
        "errors",
        "strict_valid",
        "strict_valid_rate",
        "within_samples",
        "within_expected",
        "within_matched",
        "within_match_rate",
        "within_strict_valid",
        "within_strict_valid_rate",
        "format_failures",
        "format_failure_rate",
        "collisions",
        "collision_rate",
        "floating",
        "floating_rate",
        "support_ratio_mean",
        "semantic_all_satisfied",
        "semantic_constrained",
        "semantic_pass_rate",
        "semantic_matched",
        "semantic_expected",
        "semantic_match_rate",
        "predictions_path",
        "summary_path",
    ]
    with open(leaderboard_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    print(f"\nSaved leaderboard to {leaderboard_json}")
    print(f"Saved leaderboard to {leaderboard_csv}")

    if sorted_rows:
        best = sorted_rows[0]
        within_valid = best.get("within_strict_valid_rate", 0.0) * 100
        print(
            "Best checkpoint: "
            f"{best['checkpoint_name']} | "
            f"strict_valid={best.get('strict_valid_rate', 0.0) * 100:.2f}% | "
            f"within_valid={within_valid:.2f}%"
        )

    return leaderboard_json, leaderboard_csv


def _upload_leaderboard_to_wandb(
    leaderboard_json: str,
    leaderboard_csv: str,
    *,
    project: str,
    run_name: str,
    entity: str | None = None,
    job_type: str = "eval",
    model_dir: str | None = None,
    train_yaml: str | None = None,
    eval_dataset: str | None = None,
) -> None:
    """Upload evaluation leaderboard metrics and artifacts to Weights & Biases."""
    import wandb

    with open(leaderboard_json, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not rows:
        raise ValueError(f"No rows found in {leaderboard_json}")

    rows_by_step = sorted(rows, key=lambda row: row["checkpoint_step"] or 0)
    best = rows[0]

    init_kwargs = {
        "project": project,
        "name": run_name,
        "job_type": job_type,
        "config": {
            "leaderboard_json": leaderboard_json,
            "leaderboard_csv": leaderboard_csv,
            "model_dir": model_dir,
            "train_yaml": train_yaml,
            "eval_dataset": eval_dataset,
            "num_checkpoints": len(rows_by_step),
            "num_samples": rows_by_step[0].get("matched_samples"),
        },
    }
    if entity:
        init_kwargs["entity"] = entity

    run = wandb.init(**init_kwargs)
    if getattr(wandb, "define_metric", None):
        wandb.define_metric("train/global_step")
        wandb.define_metric("*", step_metric="train/global_step", step_sync=True)

    for row in rows_by_step:
        wandb.log(
            {
                "train/global_step": row["checkpoint_step"],
                "eval/strict_valid_rate": row["strict_valid_rate"],
                "eval/within_strict_valid_rate": row.get("within_strict_valid_rate"),
                "eval/within_match_rate": row.get("within_match_rate"),
                "eval/semantic_pass_rate": row["semantic_pass_rate"],
                "eval/semantic_match_rate": row["semantic_match_rate"],
                "eval/collision_rate": row["collision_rate"],
                "eval/floating_rate": row["floating_rate"],
                "eval/support_ratio_mean": row.get("support_ratio_mean"),
                "eval/format_failure_rate": row["format_failure_rate"],
            }
        )

    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", run_name).strip("._-")
    artifact = wandb.Artifact(f"{sanitized or 'evaluation'}_artifact", type="evaluation")
    artifact.add_file(leaderboard_json, name="leaderboard.json")
    if os.path.exists(leaderboard_csv):
        artifact.add_file(leaderboard_csv, name="leaderboard.csv")
    run.log_artifact(artifact)

    run.summary["best_checkpoint"] = best["checkpoint_name"]
    run.summary["best_checkpoint_step"] = best["checkpoint_step"]
    run.summary["best_strict_valid_rate"] = best["strict_valid_rate"]
    run.summary["best_within_strict_valid_rate"] = best.get("within_strict_valid_rate")
    run.summary["best_within_match_rate"] = best.get("within_match_rate")
    run.summary["best_semantic_pass_rate"] = best["semantic_pass_rate"]
    run.summary["best_collision_rate"] = best["collision_rate"]
    run.summary["best_floating_rate"] = best["floating_rate"]
    run.summary["best_support_ratio_mean"] = best.get("support_ratio_mean")
    run.summary["best_summary_path"] = best.get("summary_path")

    print(f"W&B run URL: {run.url}")
    run.finish()


def main():
    parser = argparse.ArgumentParser(description="Automatically evaluate PointLM checkpoints")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing checkpoint-* subdirs")
    parser.add_argument("--output_dir", type=str, required=True, help="Root directory for eval outputs")
    parser.add_argument("--data_file", type=str, default=None, help="Eval dataset JSON")
    parser.add_argument("--score_data_file", type=str, default=None, help="Optional scoring dataset JSON with point clouds")
    parser.add_argument("--train_yaml", type=str, default=None, help="Training YAML to resolve eval_dataset")
    parser.add_argument("--dataset_info_json", type=str, default="data/dataset_info.json")
    parser.add_argument(
        "--eval_mode",
        type=str,
        default="all",
        choices=["all", "latest"],
        help="Evaluate all checkpoints or only the latest checkpoint (default: all)",
    )
    parser.add_argument("--max_checkpoints", type=int, default=None, help="Optionally limit evaluated checkpoints")
    parser.add_argument("--force", action="store_true", default=False, help="Re-run even if summary already exists")

    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--inference_dtype", type=str, default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--pcd_random_noise", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=4)

    parser.add_argument("--score_workers", type=int, default=8)
    parser.add_argument("--voxel_size", type=float, default=0.02)
    parser.add_argument("--overlap_ratio_threshold", type=float, default=0.20)
    parser.add_argument("--height_tolerance", type=float, default=0.05)

    # Optional W&B upload arguments
    parser.add_argument("--upload_wandb", action="store_true", default=False, help="Upload leaderboard to W&B")
    parser.add_argument("--wandb_project", type=str, default="llamafactory")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_job_type", type=str, default="eval")
    args = parser.parse_args()

    model_dir = os.path.abspath(args.model_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    data_file = _resolve_data_file(args)
    checkpoints = _discover_checkpoints(model_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint-* directories found under {model_dir}")

    if args.eval_mode == "latest":
        checkpoints = [checkpoints[-1]]
    elif args.max_checkpoints is not None:
        checkpoints = checkpoints[: args.max_checkpoints]

    infer_script = os.path.join(os.path.dirname(__file__), "pointlm_infer.py")
    score_script = os.path.join(os.path.dirname(__file__), "pointlm_eval_score.py")

    print(f"Model dir:      {model_dir}")
    print(f"Eval dataset:   {data_file}")
    if args.score_data_file:
        print(f"Score dataset:  {os.path.abspath(args.score_data_file)}")
    print(f"Output dir:     {output_dir}")
    print(f"Eval mode:      {args.eval_mode} ({len(checkpoints)} checkpoint(s))")

    rows = []
    for checkpoint_path in checkpoints:
        checkpoint_name = os.path.basename(checkpoint_path.rstrip("/"))
        checkpoint_match = re.search(r"checkpoint-(\d+)$", checkpoint_name)
        checkpoint_step = int(checkpoint_match.group(1)) if checkpoint_match else None

        ckpt_output_dir = os.path.join(output_dir, checkpoint_name)
        os.makedirs(ckpt_output_dir, exist_ok=True)
        predictions_path = os.path.join(ckpt_output_dir, "predictions.json")
        scored_path = os.path.join(ckpt_output_dir, "scored.json")
        summary_path = os.path.join(ckpt_output_dir, "summary.json")

        print(f"\n{'=' * 60}")
        print(f"Evaluating {checkpoint_name}")
        print(f"{'=' * 60}")

        if args.force or not os.path.exists(summary_path):
            infer_cmd = [
                sys.executable,
                infer_script,
                "-d",
                data_file,
                "-m",
                checkpoint_path,
                "-o",
                predictions_path,
                "--num_gpus",
                str(args.num_gpus),
                "--start_idx",
                str(args.start_idx),
                "--inference_dtype",
                args.inference_dtype,
                "--max_new_tokens",
                str(args.max_new_tokens),
                "--temperature",
                str(args.temperature),
                "--top_p",
                str(args.top_p),
                "--top_k",
                str(args.top_k),
                "--num_beams",
                str(args.num_beams),
                "--batch_size",
                str(args.batch_size),
            ]
            if args.num_samples is not None:
                infer_cmd.extend(["--num_samples", str(args.num_samples)])
            if args.do_sample:
                infer_cmd.append("--do_sample")
            if args.pcd_random_noise:
                infer_cmd.append("--pcd_random_noise")
            _run_command(infer_cmd)

            score_cmd = [
                sys.executable,
                score_script,
                "--predictions",
                predictions_path,
                "--data_file",
                data_file,
                "-o",
                scored_path,
                "--summary_output",
                summary_path,
                "--num_workers",
                str(args.score_workers),
                "--voxel_size",
                str(args.voxel_size),
                "--overlap_ratio_threshold",
                str(args.overlap_ratio_threshold),
                "--height_tolerance",
                str(args.height_tolerance),
            ]
            if args.score_data_file:
                score_cmd.extend(["--score_data_file", args.score_data_file])
            _run_command(score_cmd)
        else:
            print(f"Skipping {checkpoint_name}: found existing {summary_path}")

        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        summary["checkpoint_name"] = checkpoint_name
        summary["checkpoint_step"] = checkpoint_step
        summary["model_path"] = os.path.abspath(checkpoint_path)
        summary["summary_path"] = os.path.abspath(summary_path)
        rows.append(summary)

    if args.eval_mode == "all":
        leaderboard_json, leaderboard_csv = _write_leaderboard(rows, output_dir)
        if args.upload_wandb:
            run_name = args.wandb_run_name or f"eval_{os.path.basename(model_dir)}"
            _upload_leaderboard_to_wandb(
                leaderboard_json,
                leaderboard_csv,
                project=args.wandb_project,
                run_name=run_name,
                entity=args.wandb_entity,
                job_type=args.wandb_job_type,
                model_dir=model_dir,
                train_yaml=args.train_yaml,
                eval_dataset=data_file,
            )


if __name__ == "__main__":
    main()
