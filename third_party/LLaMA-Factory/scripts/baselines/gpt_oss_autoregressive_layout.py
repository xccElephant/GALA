"""Autoregressive GPT-OSS layout inference for IntelliScene3 input scenes.

The script reads IntelliScene3 ``experiments/input`` JSON files, places each
object sequentially with a text-only GPT-OSS model, and writes layout JSON files
compatible with ``scripts/visualize_layout_blender.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from tqdm import tqdm
from transformers import AutoTokenizer


SYSTEM_PROMPT = (
    "You are a scene layout assistant. Your task is to predict new object pose "
    "based on the current scene layout, geometry of objects and instruction.\n\n"
    "## Coordinate System\n"
    "X: right, Y: forward, Z: up. Ground plane: XY.\n\n"
    "## Object Pose\n"
    "- center: (X, Y, Z) in meters\n"
    "- rotation: (roll, pitch, yaw) in degrees, range [0, 360). Yaw: rotation "
    "around Z, 0\u00b0 points -Y, positive counterclockwise.\n"
    "- size: (width, depth, height) in meters\n\n"
    "## Physical Constraints\n"
    "- The new object must rest on a supporting surface. Its bottom must not float in air.\n"
    "- The new object must not overlap with any existing object. Ensure geometries do not intersect.\n\n"
    "## Output Format\n"
    "Return one list: [x, y, z, roll, pitch, yaw]\n"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPT-OSS autoregressive layout inference")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="../../input",
        help="Directory containing inference input JSON files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/eval/gpt-oss-120b/intelliscene3_autoregressive",
        help="Directory where visualization-compatible layout JSON files are written.",
    )
    parser.add_argument("--model_name_or_path", type=str, default="openai/gpt-oss-120b")
    parser.add_argument("--scene_glob", type=str, default="*.json")
    parser.add_argument("--limit_scenes", type=int, default=None)
    parser.add_argument("--max_objects_per_scene", type=int, default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=8)
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument(
        "--max_retries",
        type=int,
        default=2,
        help="Number of stricter-format retries after a pose parsing failure.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        choices=["low", "medium", "high"],
        default="low",
    )
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def iter_scene_files(input_dir: Path, scene_glob: str, limit: int | None) -> list[Path]:
    files = sorted(input_dir.glob(scene_glob))
    if limit is not None:
        files = files[:limit]
    return files


def _round_list(values: Iterable[float], ndigits: int = 4) -> list[float]:
    return [round(float(v), ndigits) for v in values]


def _scene_type(scene_info: dict[str, Any]) -> str:
    scene_type = scene_info.get("scene_type")
    if scene_type:
        return str(scene_type)
    description = str(scene_info.get("scene_description", ""))
    return description.removeprefix("A ").strip() or "room"


def _object_for_prompt(obj: dict[str, Any]) -> dict[str, Any]:
    data = {
        "id": int(obj["id"]),
        "category": obj.get("category") or obj.get("short_desc", "object"),
        "short_desc": obj.get("short_desc", obj.get("category", "object")),
        "center": _round_list(obj["center"]),
        "rotation": _round_list(obj.get("rotation", [0.0, 0.0, 0.0])),
        "size": _round_list(obj["size"]),
    }
    if obj.get("jid"):
        data["jid"] = obj["jid"]
    return data


def _task_new_object(task: dict[str, Any]) -> dict[str, Any]:
    obj_id = task.get("new_object_id", task.get("id"))
    data = {
        "id": int(obj_id),
        "category": task.get("category") or task.get("short_desc", "object"),
        "short_desc": task.get("short_desc", task.get("category", "object")),
        "size": _round_list(task["size"]),
    }
    if task.get("jid"):
        data["jid"] = task["jid"]
    return data


def build_user_prompt(
    scene_info: dict[str, Any],
    placed_objects: list[dict[str, Any]],
    task: dict[str, Any],
) -> str:
    prompt_data = {
        "scene_id": scene_info.get("scene_id"),
        "scene_type": _scene_type(scene_info),
        "boundary": scene_info["boundary"],
        "scene_height": round(float(scene_info.get("scene_height", 2.8)), 4),
        "existing_objects": [_object_for_prompt(obj) for obj in placed_objects],
        "new_object": _task_new_object(task),
        "instruction": task["instruction"],
    }
    return json.dumps(prompt_data, ensure_ascii=False, separators=(",", ":"))


def build_retry_user_prompt(user_prompt: str) -> str:
    return (
        "Previous response was invalid because it was not exactly one JSON list of six numbers.\n"
        "Return exactly one JSON list [x, y, z, roll, pitch, yaw].\n"
        "No analysis, no markdown, no prose, no labels.\n"
        "Scene JSON:\n"
        f"{user_prompt}"
    )


def build_chat_prompt(
    tokenizer,
    user_prompt: str,
    enable_thinking: bool,
    reasoning_effort: str | None,
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    chat_template_kwargs = {"enable_thinking": enable_thinking}
    if reasoning_effort is not None:
        chat_template_kwargs["reasoning_effort"] = reasoning_effort
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **chat_template_kwargs,
    )


def extract_final_response(text: str, raw_text: str | None = None) -> str:
    if raw_text:
        final_marker = "<|channel|>final<|message|>"
        if final_marker in raw_text:
            final_text = raw_text.rsplit(final_marker, maxsplit=1)[1]
            for end_marker in ("<|return|>", "<|end|>"):
                final_text = final_text.split(end_marker, maxsplit=1)[0]
            return final_text.strip()

    final_match = re.search(r"(?:^|\n)final\s*(.*)\Z", text.strip(), flags=re.DOTALL)
    if final_match:
        return final_match.group(1).strip()
    return text.strip()


def parse_pose(response: str) -> list[float]:
    text = response.strip()
    candidates = [text]
    candidates.extend(match.group(0) for match in re.finditer(r"\[[^\[\]]+\]", text, flags=re.DOTALL))

    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, list) and len(value) == 6:
            pose = [float(v) for v in value]
            if all(v == v for v in pose):
                return pose
    raise ValueError(f"Cannot parse 6D pose from response: {response!r}")


def object_from_task_and_pose(task: dict[str, Any], pose: list[float], object_id: int) -> dict[str, Any]:
    return {
        "id": int(object_id),
        "short_desc": task.get("short_desc", task.get("category", "object")),
        "category": task.get("category", task.get("short_desc", "object")),
        "center": _round_list(pose[:3]),
        "size": _round_list(task["size"]),
        "rotation": _round_list(pose[3:6]),
        "jid": task.get("jid", ""),
        "instruction": task.get("instruction", ""),
    }


def gt_pose_dict(task: dict[str, Any]) -> dict[str, list[float]] | None:
    gt_pose = task.get("gt_pose")
    if not isinstance(gt_pose, dict):
        return None
    center = gt_pose.get("center")
    rotation = gt_pose.get("rotation")
    if center is None or rotation is None:
        return None
    return {"center": _round_list(center), "rotation": _round_list(rotation)}


def infer_one_pose(
    llm,
    tokenizer,
    sampling_params,
    user_prompt: str,
    enable_thinking: bool,
    reasoning_effort: str | None,
    max_retries: int = 0,
) -> tuple[list[float], str, str]:
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")

    last_final_text = ""
    last_raw_text = ""
    for attempt in range(max_retries + 1):
        active_prompt = user_prompt if attempt == 0 else build_retry_user_prompt(user_prompt)
        chat_prompt = build_chat_prompt(tokenizer, active_prompt, enable_thinking, reasoning_effort)
        outputs = llm.generate([chat_prompt], sampling_params)
        completion = outputs[0].outputs[0]
        last_raw_text = tokenizer.decode(completion.token_ids, skip_special_tokens=False)
        last_final_text = extract_final_response(completion.text, last_raw_text)
        try:
            return parse_pose(last_final_text), last_final_text, last_raw_text
        except ValueError:
            if attempt == max_retries:
                break

    raise ValueError(f"Cannot parse 6D pose after {max_retries + 1} attempt(s): {last_final_text!r}")


def run_scene(
    scene_path: Path,
    output_path: Path,
    llm,
    tokenizer,
    sampling_params,
    args: argparse.Namespace,
) -> dict[str, Any]:
    with open(scene_path, "r", encoding="utf-8") as f:
        scene_input = json.load(f)

    scene_info = scene_input["scene_info"]
    tasks = sorted(scene_input.get("placement_tasks", []), key=lambda item: item.get("step", item.get("id", 0)))
    if args.max_objects_per_scene is not None:
        tasks = tasks[: args.max_objects_per_scene]

    placed_objects = [_object_for_prompt(obj) for obj in scene_info.get("existing_objects", [])]
    successful_placements = []
    failed_placements = []
    total_inference_time = 0.0

    for task in tqdm(tasks, desc=scene_info.get("scene_id", scene_path.stem), leave=False):
        storage_id = int(task.get("new_object_id", task.get("id", len(placed_objects))))
        user_prompt = build_user_prompt(scene_info, placed_objects, task)
        start_time = time.time()
        try:
            pose, final_text, raw_text = infer_one_pose(
                llm,
                tokenizer,
                sampling_params,
                user_prompt,
                args.enable_thinking,
                args.reasoning_effort,
                args.max_retries,
            )
            inference_time = time.time() - start_time
            total_inference_time += inference_time
            scene_obj = object_from_task_and_pose(task, pose, storage_id)
            placed_objects.append(scene_obj)
            successful_placements.append(
                {
                    "object_id": storage_id,
                    "object_short_desc": scene_obj["short_desc"],
                    "instruction": task["instruction"],
                    "jid": scene_obj["jid"],
                    "pose": {"center": scene_obj["center"], "rotation": scene_obj["rotation"]},
                    "gt_pose": gt_pose_dict(task),
                    "raw_output": final_text,
                    "inference_time": inference_time,
                }
            )
        except Exception as exc:
            inference_time = time.time() - start_time
            total_inference_time += inference_time
            failed_placements.append(
                {
                    "object_id": storage_id,
                    "object_short_desc": task.get("short_desc", task.get("category", "object")),
                    "instruction": task.get("instruction", ""),
                    "jid": task.get("jid", ""),
                    "gt_pose": gt_pose_dict(task),
                    "error": str(exc),
                    "inference_time": inference_time,
                }
            )

    result = {
        "layout_mode": "gpt_oss_autoregressive",
        "model_name_or_path": args.model_name_or_path,
        "source_input": str(scene_path),
        "total_tasks": len(tasks),
        "successful": len(successful_placements),
        "failed": len(failed_placements),
        "successful_placements": successful_placements,
        "failed_placements": failed_placements,
        "final_scene": {
            "scene_id": scene_info.get("scene_id", scene_path.stem),
            "scene_type": _scene_type(scene_info),
            "scene_description": scene_info.get("scene_description", ""),
            "scene_height": scene_info.get("scene_height", 2.8),
            "boundary": scene_info.get("boundary", []),
            "objects": placed_objects,
        },
        "total_inference_time": total_inference_time,
        "avg_inference_time": total_inference_time / len(tasks) if tasks else 0.0,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    scene_files = iter_scene_files(input_dir, args.scene_glob, args.limit_scenes)
    if not scene_files:
        raise FileNotFoundError(f"No scene files matched {args.scene_glob!r} under {input_dir}")

    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    llm = LLM(
        model=args.model_name_or_path,
        trust_remote_code=True,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    summary = []
    for scene_path in tqdm(scene_files, desc="Scenes"):
        output_path = output_dir / f"{scene_path.stem}_gpt_oss_layout.json"
        if output_path.exists() and not args.overwrite:
            print(f"Skip existing output: {output_path}")
            continue
        result = run_scene(scene_path, output_path, llm, tokenizer, sampling_params, args)
        summary.append(
            {
                "scene_id": result["final_scene"]["scene_id"],
                "output": str(output_path),
                "total_tasks": result["total_tasks"],
                "successful": result["successful"],
                "failed": result["failed"],
            }
        )

    if summary:
        with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(summary)} scene layouts to {output_dir}")


if __name__ == "__main__":
    main()
