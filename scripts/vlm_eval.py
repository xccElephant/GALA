#!/usr/bin/env python3
"""
Batch score scene renderings with VLM.

Features:
1. Async parallel scoring with bounded concurrency.
2. Retry with exponential backoff for transient API errors.
3. Save intermediate results every N scenes to avoid data loss.
4. Compute summary statistics after scoring.
"""

import argparse
import asyncio
import base64
import json
import random
import re
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from agent.llm import call_vlm


SCORE_DIMENSIONS = (
    "aesthetic",
    "layout_reasonableness",
    "physical_correctness",
    "object_density",
)

SYSTEM_PROMPT = """You are a strict evaluator for indoor 3D scene renderings.
You must be conservative, not generous.
Penalize sparse, incomplete, or implausible scenes aggressively.
Your job is to score scene quality and return JSON only.
Do not output markdown or any extra text.
"""

USER_PROMPT_TEMPLATE = """Please evaluate this indoor 3D scene rendering and score it from 1 to 10 on each dimension:

1) aesthetic:
- Visual appeal, color/material harmony, composition quality.

2) layout_reasonableness:
- Whether object arrangement is functionally reasonable and spatially coherent.

3) physical_correctness:
- Whether placements look physically plausible (support/contact, no obvious floating/interpenetration).

4) object_density:
- Whether object amount is appropriate (neither too sparse nor too cluttered for a usable room).

Rules:
- Use integer scores in [1, 10].
- Be strict and consistent, and prefer lower scores when uncertain.
- Judge only visible content in this single image. Do not assume hidden quality.
- If the scene is very sparse / mostly empty (large unused area, very few objects), score low.
- If the room is functionally incomplete for its scene type, layout_reasonableness should usually be low.
- If there are obvious floating/interpenetration/support errors, physical_correctness should usually be low.
- Scoring guide:
  - 9-10: excellent, complete, coherent, physically convincing.
  - 7-8: good quality with minor issues.
  - 5-6: mediocre, clear shortcomings.
  - 3-4: poor, obvious functional/visual problems.
  - 1-2: very poor or almost empty/useless scene.
- Return ONLY this JSON schema:
{{
  "aesthetic": <int 1-10>,
  "layout_reasonableness": <int 1-10>,
  "physical_correctness": <int 1-10>,
  "object_density": <int 1-10>,
  "brief_comment": "<one short sentence>"
}}

Scene type: {scene_type}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use VLM to score scene renderings in batch."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/path/to/datasets/visualizations/batch_render_3dfront",
        help="Directory containing rendered scene images.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="output/vlm_scene_scores_3dfront.json",
        help="Path to save scoring results.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="instruction_generator",
        help="Task name in config/llm_key.json for selecting the VLM model.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Max number of concurrent VLM requests.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max retries per image on API/parsing failures.",
    )
    parser.add_argument(
        "--retry-base-wait",
        type=float,
        default=2.0,
        help="Base waiting seconds for exponential backoff.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Save progress every N processed scenes.",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=0,
        help="Limit number of scenes to process (0 means all).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for VLM.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only scan top-level files in input-dir.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore existing output file and start from scratch.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose logs from model calls.",
    )
    parser.add_argument(
        "--plot-histogram",
        action="store_true",
        help="Plot score histograms after scoring finishes.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip scoring and only plot histograms from existing output-json.",
    )
    parser.add_argument(
        "--hist-output",
        type=str,
        default="",
        help="Output image path for histogram plot (default: <output-json stem>_hist.png).",
    )
    parser.add_argument(
        "--hist-dpi",
        type=int,
        default=180,
        help="DPI for histogram image output.",
    )
    return parser.parse_args()


def now_str() -> str:
    return datetime.now().isoformat(timespec="seconds")


def discover_scene_images(input_dir: Path, recursive: bool) -> Dict[str, Path]:
    extensions = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    if recursive:
        all_files = [p for p in input_dir.rglob("*") if p.is_file()]
    else:
        all_files = [p for p in input_dir.iterdir() if p.is_file()]

    image_files = [p for p in all_files if p.suffix.lower() in extensions]
    image_files.sort()

    scene_to_image: Dict[str, Path] = {}
    for image_path in image_files:
        scene_id = image_path.stem
        if scene_id in scene_to_image:
            raise ValueError(
                f"Duplicated scene_id from file name stem: {scene_id}. "
                f"Conflicting files: {scene_to_image[scene_id]} and {image_path}"
            )
        scene_to_image[scene_id] = image_path
    return scene_to_image


def extract_scene_type(scene_id: str) -> str:
    scene_type = scene_id.split("-", 1)[0].strip().lower()
    if not scene_type:
        raise ValueError(f"Cannot extract scene_type from scene_id: {scene_id}")
    return scene_type


def image_to_base64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_json_object(text: str) -> Dict[str, Any]:
    try:
        loaded = json.loads(text)
        if isinstance(loaded, dict):
            return loaded
    except json.JSONDecodeError:
        pass

    fenced_patterns = [
        r"```json\s*([\s\S]*?)\s*```",
        r"```\s*([\s\S]*?)\s*```",
    ]
    for pattern in fenced_patterns:
        match = re.search(pattern, text)
        if match:
            candidate = match.group(1)
            loaded = json.loads(candidate)
            if isinstance(loaded, dict):
                return loaded

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        loaded = json.loads(match.group(0))
        if isinstance(loaded, dict):
            return loaded

    raise ValueError("Cannot parse JSON object from VLM response.")


def normalize_score(value: Any, key: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Invalid score type for {key}: bool")
    numeric = float(value)
    rounded = int(round(numeric))
    if rounded < 1:
        return 1
    if rounded > 10:
        return 10
    return rounded


def parse_score_response(text: str) -> Tuple[Dict[str, int], str]:
    payload = extract_json_object(text)

    scores: Dict[str, int] = {}
    for dim in SCORE_DIMENSIONS:
        if dim not in payload:
            raise ValueError(f"Missing key in VLM response: {dim}")
        scores[dim] = normalize_score(payload[dim], dim)

    if "brief_comment" not in payload:
        raise ValueError("Missing key in VLM response: brief_comment")
    brief_comment = str(payload["brief_comment"]).strip()
    if not brief_comment:
        brief_comment = "No comment."

    return scores, brief_comment


def compute_retry_wait_seconds(error_message: str, attempt: int, base_wait: float) -> float:
    lowered = error_message.lower()
    has_rate_signal = (
        "429" in lowered
        or "rate" in lowered
        or "quota" in lowered
        or "too many requests" in lowered
        or "timeout" in lowered
        or "timed out" in lowered
        or "temporarily unavailable" in lowered
    )
    multiplier = 2.0 if has_rate_signal else 1.0
    wait = base_wait * (2 ** (attempt - 1)) * multiplier
    wait = min(wait, 120.0)
    return wait + random.uniform(0.0, 1.0)


async def score_one_image(
    scene_id: str,
    scene_type: str,
    image_path: Path,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    retries = args.max_retries
    for attempt in range(1, retries + 1):
        try:
            image_base64 = await asyncio.to_thread(image_to_base64, image_path)
            prompt = USER_PROMPT_TEMPLATE.format(scene_type=scene_type)
            async with semaphore:
                response_text = await asyncio.to_thread(
                    call_vlm,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=prompt,
                    image_base64=image_base64,
                    task=args.task,
                    temperature=args.temperature,
                    max_retries=1,
                    verbose=args.verbose,
                )
            scores, brief_comment = parse_score_response(response_text)
            return {
                "image_path": str(image_path),
                "scores": scores,
                "brief_comment": brief_comment,
            }
        except Exception as exc:
            if attempt == retries:
                raise RuntimeError(
                    f"Failed after {retries} attempts for {image_path}: {exc}"
                ) from exc
            wait_seconds = compute_retry_wait_seconds(
                error_message=str(exc),
                attempt=attempt,
                base_wait=args.retry_base_wait,
            )
            print(
                f"[Retry] scene={scene_id}, image={image_path.name}, "
                f"attempt={attempt}/{retries}, wait={wait_seconds:.1f}s, err={exc}"
            )
            await asyncio.sleep(wait_seconds)

    raise RuntimeError(f"Unexpected retry logic exit for image: {image_path}")


async def score_one_scene(
    scene_id: str,
    scene_type: str,
    image_path: Path,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    image_result = await score_one_image(
        scene_id=scene_id,
        scene_type=scene_type,
        image_path=image_path,
        args=args,
        semaphore=semaphore,
    )

    return {
        "scene_type": scene_type,
        "scores": image_result["scores"],
        "brief_comment": image_result["brief_comment"],
        "image_path": image_result["image_path"],
        "updated_at": now_str(),
    }


def compute_statistics(scene_scores: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if not scene_scores:
        return {
            "scene_count": 0,
            "dimensions": {},
            "overall": {},
        }

    per_dimension_values, per_scene_overall = collect_score_values(scene_scores)

    dim_stats: Dict[str, Dict[str, float]] = {}
    for dim, values in per_dimension_values.items():
        if not values:
            continue
        if len(values) == 1:
            std_val = 0.0
        else:
            std_val = statistics.pstdev(values)
        dim_stats[dim] = {
            "count": len(values),
            "mean": round(statistics.fmean(values), 4),
            "median": round(statistics.median(values), 4),
            "std": round(std_val, 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
        }

    if len(per_scene_overall) == 1:
        overall_std = 0.0
    else:
        overall_std = statistics.pstdev(per_scene_overall)

    overall_stats = {
        "count": len(per_scene_overall),
        "mean": round(statistics.fmean(per_scene_overall), 4),
        "median": round(statistics.median(per_scene_overall), 4),
        "std": round(overall_std, 4),
        "min": round(min(per_scene_overall), 4),
        "max": round(max(per_scene_overall), 4),
    }

    return {
        "scene_count": len(per_scene_overall),
        "dimensions": dim_stats,
        "overall": overall_stats,
    }


def collect_score_values(
    scene_scores: Dict[str, Dict[str, Any]]
) -> Tuple[Dict[str, List[float]], List[float]]:
    per_dimension_values: Dict[str, List[float]] = {dim: [] for dim in SCORE_DIMENSIONS}
    per_scene_overall: List[float] = []

    for scene_data in scene_scores.values():
        if "scores" not in scene_data:
            continue
        score_map = scene_data["scores"]
        scene_vals = []
        for dim in SCORE_DIMENSIONS:
            if dim not in score_map:
                raise ValueError(f"Missing score dimension in saved data: {dim}")
            val = float(score_map[dim])
            per_dimension_values[dim].append(val)
            scene_vals.append(val)
        per_scene_overall.append(sum(scene_vals) / len(scene_vals))
    return per_dimension_values, per_scene_overall


def resolve_hist_output_path(output_json: Path, hist_output: str) -> Path:
    if hist_output:
        return Path(hist_output)
    return output_json.with_name(output_json.stem + "_hist.png")


def plot_score_histograms(
    scene_scores: Dict[str, Dict[str, Any]],
    output_path: Path,
    dpi: int,
) -> None:
    if dpi < 1:
        raise ValueError("--hist-dpi must be >= 1")

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for histogram plotting. "
            "Install it in the current environment first."
        ) from exc

    per_dimension_values, per_scene_overall = collect_score_values(scene_scores)
    if not per_scene_overall:
        raise ValueError("No valid scene scores found; cannot plot histogram.")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    axes_flat = list(axes.flat)
    plot_items = list(SCORE_DIMENSIONS) + ["overall"]

    for idx, key in enumerate(plot_items):
        ax = axes_flat[idx]
        if key == "overall":
            values = per_scene_overall
            color = "#DD8452"
            bins = 20
            hist_range = (1.0, 10.0)
            title = "overall (avg of 4 dimensions)"
        else:
            values = per_dimension_values[key]
            color = "#4C72B0"
            bins = [x - 0.5 for x in range(1, 12)]
            hist_range = None
            title = key

        if hist_range is None:
            ax.hist(values, bins=bins, color=color, edgecolor="black", alpha=0.85)
        else:
            ax.hist(
                values,
                bins=bins,
                range=hist_range,
                color=color,
                edgecolor="black",
                alpha=0.85,
            )

        mean_val = statistics.fmean(values)
        median_val = statistics.median(values)
        mean_color = "red"
        median_color = "green"
        ax.axvline(mean_val, color=mean_color, linestyle="--", linewidth=1.2)
        ax.axvline(median_val, color=median_color, linestyle="-.", linewidth=1.2)
        ax.text(
            0.98,
            0.95,
            f"n={len(values)}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )
        ax.text(
            0.98,
            0.89,
            f"mean={mean_val:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            color=mean_color,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )
        ax.text(
            0.98,
            0.83,
            f"median={median_val:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            color=median_color,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )
        ax.set_title(title)
        ax.set_xlabel("score")
        ax.set_ylabel("count")
        ax.set_xlim(1, 10)
        ax.set_xticks(range(1, 11))
        ax.grid(axis="y", alpha=0.25)

    axes_flat[-1].axis("off")
    fig.suptitle(f"VLM scene score histograms (n={len(per_scene_overall)})", fontsize=16)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def save_result_file(
    output_json: Path,
    meta: Dict[str, Any],
    scene_scores: Dict[str, Dict[str, Any]],
    failed_scenes: Dict[str, Dict[str, Any]],
) -> None:
    stats = compute_statistics(scene_scores)
    payload = {
        "meta": meta,
        "scene_scores": scene_scores,
        "failed_scenes": failed_scenes,
        "stats": stats,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_json.with_suffix(output_json.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp_path.replace(output_json)


def load_existing_result(output_json: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    with open(output_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"Invalid existing result format: {output_json}")
    if "scene_scores" not in payload:
        raise ValueError(f"Missing 'scene_scores' in existing result: {output_json}")
    if "failed_scenes" not in payload:
        raise ValueError(f"Missing 'failed_scenes' in existing result: {output_json}")

    scene_scores = payload["scene_scores"]
    failed_scenes = payload["failed_scenes"]
    if not isinstance(scene_scores, dict) or not isinstance(failed_scenes, dict):
        raise ValueError(f"Invalid existing result content: {output_json}")
    return scene_scores, failed_scenes


def print_stats(stats: Dict[str, Any]) -> None:
    scene_count = stats["scene_count"]
    print("\n" + "=" * 72)
    print("Scoring Statistics")
    print("=" * 72)
    print(f"Scored scenes: {scene_count}")

    dimensions = stats["dimensions"]
    for dim in SCORE_DIMENSIONS:
        if dim not in dimensions:
            continue
        info = dimensions[dim]
        print(
            f"{dim:24s} mean={info['mean']:.3f}  median={info['median']:.3f}  "
            f"std={info['std']:.3f}  min={info['min']:.1f}  max={info['max']:.1f}"
        )

    overall = stats["overall"]
    if overall:
        print(
            f"{'overall':24s} mean={overall['mean']:.3f}  median={overall['median']:.3f}  "
            f"std={overall['std']:.3f}  min={overall['min']:.3f}  max={overall['max']:.3f}"
        )
    print("=" * 72)


async def run_scoring(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_json = Path(args.output_json)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    if args.max_retries < 1:
        raise ValueError("--max-retries must be >= 1")
    if args.save_every < 1:
        raise ValueError("--save-every must be >= 1")
    scene_to_image = discover_scene_images(input_dir, recursive=not args.no_recursive)
    all_scene_ids = sorted(scene_to_image.keys())

    if not all_scene_ids:
        raise RuntimeError(f"No image files found under: {input_dir}")

    if args.max_scenes > 0:
        all_scene_ids = all_scene_ids[: args.max_scenes]

    scene_scores: Dict[str, Dict[str, Any]] = {}
    failed_scenes: Dict[str, Dict[str, Any]] = {}
    if output_json.exists() and not args.overwrite:
        print(f"[Resume] Loading existing results from: {output_json}")
        loaded_scores, loaded_failed = load_existing_result(output_json)
        scene_scores = loaded_scores
        failed_scenes = loaded_failed

    pending_scene_ids = [sid for sid in all_scene_ids if sid not in scene_scores]
    total_scenes = len(all_scene_ids)
    print(f"Discovered scenes: {len(all_scene_ids)}")
    print(f"Already scored: {len(scene_scores)}")
    print(f"Pending scenes: {len(pending_scene_ids)}")
    print(f"Output json: {output_json}")

    meta = {
        "input_dir": str(input_dir),
        "output_json": str(output_json),
        "task": args.task,
        "concurrency": args.concurrency,
        "max_retries": args.max_retries,
        "retry_base_wait": args.retry_base_wait,
        "save_every": args.save_every,
        "temperature": args.temperature,
        "recursive_scan": not args.no_recursive,
        "started_at": now_str(),
        "updated_at": now_str(),
    }

    if not pending_scene_ids:
        print("No pending scenes. Recomputing statistics and exiting.")
        meta["updated_at"] = now_str()
        save_result_file(output_json, meta, scene_scores, failed_scenes)
        print_stats(compute_statistics(scene_scores))
        return

    semaphore = asyncio.Semaphore(args.concurrency)

    async def task_wrapper(scene_id: str) -> Tuple[str, Dict[str, Any], str]:
        try:
            result = await score_one_scene(
                scene_id=scene_id,
                scene_type=extract_scene_type(scene_id),
                image_path=scene_to_image[scene_id],
                args=args,
                semaphore=semaphore,
            )
            return scene_id, result, ""
        except Exception as exc:
            return scene_id, {}, str(exc)

    tasks = [asyncio.create_task(task_wrapper(scene_id)) for scene_id in pending_scene_ids]

    start_time = time.time()
    processed_since_save = 0
    processed_total = 0

    for finished in asyncio.as_completed(tasks):
        processed_total += 1
        scene_id, result, error_text = await finished
        if not error_text:
            scene_scores[scene_id] = result
            if scene_id in failed_scenes:
                del failed_scenes[scene_id]
            print(
                f"[OK] {scene_id} | scores={result['scores']} "
                f"| progress={processed_total}/{len(pending_scene_ids)}"
            )
        else:
            failed_scenes[scene_id] = {
                "error": error_text,
                "updated_at": now_str(),
            }
            print(
                f"[FAIL] scene={scene_id} | err={error_text} "
                f"| progress={processed_total}/{len(pending_scene_ids)}"
            )

        processed_since_save += 1
        if processed_since_save >= args.save_every:
            meta["updated_at"] = now_str()
            save_result_file(output_json, meta, scene_scores, failed_scenes)
            print(f"[Save] Intermediate results written to: {output_json}")
            processed_since_save = 0

    meta["updated_at"] = now_str()
    save_result_file(output_json, meta, scene_scores, failed_scenes)

    elapsed = time.time() - start_time
    print(f"\nDone. elapsed={elapsed:.2f}s")
    print(f"Total discovered scenes: {total_scenes}")
    print(f"Scored scenes: {len(scene_scores)}")
    print(f"Failed scenes: {len(failed_scenes)}")
    print(f"Saved to: {output_json}")

    stats = compute_statistics(scene_scores)
    print_stats(stats)


def main() -> None:
    args = parse_args()
    output_json = Path(args.output_json)

    if args.plot_only:
        if not output_json.exists():
            raise FileNotFoundError(
                f"--plot-only is set but output json does not exist: {output_json}"
            )
        scene_scores, _ = load_existing_result(output_json)
        hist_output_path = resolve_hist_output_path(output_json, args.hist_output)
        plot_score_histograms(scene_scores, hist_output_path, args.hist_dpi)
        print(f"Histogram saved to: {hist_output_path}")
        return

    asyncio.run(run_scoring(args))
    if args.plot_histogram:
        scene_scores, _ = load_existing_result(output_json)
        hist_output_path = resolve_hist_output_path(output_json, args.hist_output)
        plot_score_histograms(scene_scores, hist_output_path, args.hist_dpi)
        print(f"Histogram saved to: {hist_output_path}")


if __name__ == "__main__":
    main()

# run
"""
python scripts/vlm_eval.py \
  --input-dir /path/to/datasets/visualizations/batch_render_3dfront \
  --output-json output/vlm_scene_scores_3dfront.json \
  --task instruction_generator \
  --max-scenes 20000 \
  --concurrency 8 \
  --max-retries 5 \
  --save-every 100
"""