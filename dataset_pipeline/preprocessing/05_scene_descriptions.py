#!/usr/bin/env python3
"""
Generate and judge English scene descriptions for the autoregressive dataset using Qwen3.6-27B on vLLM.

Subcommands:
  - generate: Generate English scene descriptions directly from structured 3D ground truth.
  - judge:    Use LLM-as-judge with majority voting and verbatim evidence grounding to verify description quality.

Example vLLM server launch:
    VLLM_USE_PRECOMPILED=1 \
    vllm serve Qwen/Qwen3.6-27B \
        --tensor-parallel-size 8 \
        --port 8000 \
        --host localhost \
        --trust-remote-code \
        --enforce-eager

Usage:
    python dataset_pipeline/preprocessing/05_scene_descriptions.py generate \
        --input-dir datasets/3dish/autoregressive

    python dataset_pipeline/preprocessing/05_scene_descriptions.py judge \
        --input-dir datasets/3dish/autoregressive
"""

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PROMPT_CHAR_LIMIT = 18000

JUDGE_SYSTEM = (
    "You are a strict data-quality checker for an English 3D-scene captioning dataset. "
    "You are given a 'scene_description' that should be ONE coherent, natural English paragraph "
    "describing the layout of a room. Decide whether it is VALID or INVALID.\n\n"
    "A description is INVALID only when it is clearly broken in one of these ways:\n"
    "- It is written wholly or partly in a language other than English.\n"
    "- It contains the model's own reasoning, planning notes, or analysis instead of (or in addition to) "
    "the final paragraph.\n"
    "- It contains stray markup or control text that is not part of natural prose.\n"
    "- A word or short phrase is repeated many times in a row (looping / degenerate output).\n"
    "- It is empty, or it is obviously cut off / incomplete.\n\n"
    "Do NOT mark a description invalid for minor typos, small grammatical mistakes, awkward phrasing, or "
    "for naturally repeating words (e.g. many 'black' or 'modern' items, or repeated prepositions like "
    "'next to'). Such descriptions are VALID.\n\n"
    "CRITICAL EVIDENCE RULE: If you judge it INVALID, you MUST copy the EXACT offending substring, "
    "verbatim and character-for-character, from the description into the \"evidence\" field (about 3 to 12 "
    "words). If you cannot quote a concrete offending substring that literally appears in the text, then "
    "the description is VALID. Never invent or paraphrase evidence.\n\n"
    "Respond with ONLY a compact JSON object and nothing else:\n"
    '{"valid": true|false, "evidence": "<verbatim quote from the text, or empty if valid>", '
    '"reason": "<short reason>"}'
)


# ──────────────────────────────────────────────────────────────────
# Shared target resolution
# ──────────────────────────────────────────────────────────────────


def _resolve_scene_targets(
    input_dir: Path,
    scene_list_path: Optional[str],
    skip_paths: Optional[Set[str]] = None,
) -> Tuple[List[Path], int, int]:
    """Resolve target scene JSON files from --scene-list or by scanning input_dir."""
    skip_set = skip_paths or set()

    if scene_list_path:
        with open(scene_list_path, "r", encoding="utf-8") as f:
            raw_list = (
                json.load(f)
                if scene_list_path.endswith(".json")
                else [line.strip() for line in f if line.strip()]
            )
        targets: List[Path] = []
        missing = 0
        for entry in raw_list:
            p = Path(entry)
            if p.is_file() and p.suffix == ".json":
                json_path = p
            elif p.is_dir():
                json_path = p / f"{p.name}.json"
            else:
                json_path = input_dir / entry / f"{entry}.json"
            if not json_path.exists():
                cands = list(json_path.parent.glob("*.json")) if json_path.parent.exists() else []
                if not cands:
                    missing += 1
                    continue
                json_path = cands[0]
            if str(json_path) in skip_set:
                continue
            targets.append(json_path)
        return targets, len(raw_list), missing

    scene_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    targets = []
    for scene_dir in scene_dirs:
        json_path = scene_dir / f"{scene_dir.name}.json"
        if not json_path.exists():
            json_files = [p for p in scene_dir.iterdir() if p.suffix == ".json"]
            if not json_files:
                continue
            json_path = json_files[0]
        if str(json_path) in skip_set:
            continue
        targets.append(json_path)
    return targets, len(scene_dirs), 0


# ──────────────────────────────────────────────────────────────────
# 1. Description Generation
# ──────────────────────────────────────────────────────────────────


def _construct_prompt_compact(scene_data: Dict[str, Any]) -> str:
    """Compact prompt for object-dense scenes: all objects, minimal per-object verbosity."""
    scene_type = scene_data.get("scene_type", "room").replace("_", " ")
    scene_height = scene_data.get("scene_height", 2.8)
    objects = scene_data.get("objects", [])

    lines = []
    for obj in objects:
        name = obj.get("short_desc") or obj.get("category", "object")
        instruction = (obj.get("instruction") or "").strip()
        instruction = re.sub(r"^place the\s*", "", instruction).strip()
        lines.append(f"- {name}: {instruction}" if instruction else f"- {name}")

    objects_text = "\n".join(lines) if lines else "(no objects listed)"
    return (
        f"This is a {scene_type} with a ceiling height of {scene_height:.1f} meters. "
        f"It contains {len(objects)} items.\n\n"
        f"Objects and their placement:\n{objects_text}\n\n"
        f"Write a detailed, cohesive paragraph in natural English prose describing the layout "
        f"of this {scene_type}, faithfully reflecting the objects and their placement listed above. "
        f"Do not use bullet points or lists."
    )


def construct_prompt(scene_data: Dict[str, Any]) -> str:
    """Construct a prompt directly from the scene's structured ground truth."""
    scene_type = scene_data.get("scene_type", "room").replace("_", " ")
    scene_height = scene_data.get("scene_height", 2.8)
    objects = scene_data.get("objects", [])

    obj_lines = []
    for obj in objects:
        obj_id = obj.get("id")
        category = obj.get("category", "object")
        desc = obj.get("desc") or obj.get("short_desc") or category
        size = obj.get("size", [0.0, 0.0, 0.0])
        size_str = f"{size[0]:.2f}m x {size[1]:.2f}m x {size[2]:.2f}m"
        instruction = (obj.get("instruction") or "").strip()

        line = f"- {desc} (id {obj_id}, category: {category}, size: {size_str})"
        if instruction:
            line += f"; placement: {instruction}"
        obj_lines.append(line)

    objects_text = "\n".join(obj_lines) if obj_lines else "(no objects listed)"

    prompt = (
        f"This is a {scene_type} with a ceiling height of {scene_height:.1f} meters.\n\n"
        f"Objects in the room and their placement:\n{objects_text}\n\n"
        f"Write a detailed, cohesive paragraph in natural English prose describing the layout "
        f"of this {scene_type}, faithfully reflecting the objects and their spatial placement listed above. "
        f"Do not use bullet points or lists."
    )

    if len(prompt) > PROMPT_CHAR_LIMIT:
        return _construct_prompt_compact(scene_data)
    return prompt


async def _generate_single_scene(
    client: AsyncOpenAI,
    json_path: Path,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_retries: int,
    enable_thinking: bool,
    overwrite: bool,
    semaphore: asyncio.Semaphore,
) -> Tuple[bool, str, str, Optional[str]]:
    async with semaphore:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                scene_data = json.load(f)

            if (
                not overwrite
                and scene_data.get("scene_description")
                and str(scene_data["scene_description"]).strip()
            ):
                return True, f"Skipped (already exists): {json_path.name}", str(json_path), None

            prompt_text = construct_prompt(scene_data)

            description = ""
            last_err: Optional[str] = None
            for attempt in range(max_retries):
                try:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt_text}],
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
                    )
                    candidate = (response.choices[0].message.content or "").strip()
                    if candidate:
                        description = candidate
                        break
                    last_err = "empty response"
                except Exception as api_err:
                    last_err = str(api_err)
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 * (attempt + 1))

            if not description:
                return False, f"Failed ({last_err}): {json_path.name}", str(json_path), last_err or "unknown"

            scene_data["scene_description"] = description
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(scene_data, f, indent=2, ensure_ascii=False)

            return True, f"Success: {json_path.name}", str(json_path), None

        except Exception as exc:
            return False, f"Error processing {json_path.name}: {exc}", str(json_path), str(exc)


async def run_generate_async(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return

    overwrite = args.overwrite or args.mode == "overwrite" or bool(args.scene_list)
    candidates, total_scanned, missing = _resolve_scene_targets(input_dir, args.scene_list)

    tasks_to_run: List[Path] = []
    skipped_count = 0
    for json_path in candidates:
        if not overwrite:
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("scene_description") and str(data["scene_description"]).strip():
                    skipped_count += 1
                    continue
            except Exception:
                pass
        tasks_to_run.append(json_path)

    if args.scene_list:
        print(f"Scene-list mode: {len(tasks_to_run)} scenes to regenerate ({missing} not found).")
    else:
        print(f"Found {total_scanned} scene directories.")
        print(f"Already processed (skipped): {skipped_count}")
        print(f"Scenes to process: {len(tasks_to_run)}")

    if args.limit:
        tasks_to_run = tasks_to_run[: args.limit]
        print(f"Limited run to first {args.limit} scenes.")

    if not tasks_to_run:
        print("No scenes to process. Exiting.")
        return

    client = AsyncOpenAI(base_url=args.api_url, api_key=args.api_key)
    semaphore = asyncio.Semaphore(args.concurrency)
    error_log_path = Path(args.error_log) if args.error_log else input_dir / "generation_errors.jsonl"

    print(f"Starting generation with concurrency={args.concurrency} using model={args.model}...")
    print(f"Failures (if any) will be logged to: {error_log_path}")

    async_tasks = [
        _generate_single_scene(
            client=client,
            json_path=json_path,
            model=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
            enable_thinking=args.enable_thinking,
            overwrite=overwrite,
            semaphore=semaphore,
        )
        for json_path in tasks_to_run
    ]

    success_count = 0
    failure_count = 0
    run_started = datetime.now().isoformat()
    with open(error_log_path, "a", encoding="utf-8") as error_log_file:
        for future in tqdm(asyncio.as_completed(async_tasks), total=len(async_tasks), desc="Generating Descriptions"):
            success, message, scene_path, error_reason = await future
            if success:
                success_count += 1
            else:
                failure_count += 1
                print(f"\n[FAILURE] {message}")
                record = {
                    "timestamp": datetime.now().isoformat(),
                    "run_started": run_started,
                    "scene_path": scene_path,
                    "error": error_reason,
                    "message": message,
                }
                error_log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                error_log_file.flush()

    print("\n" + "=" * 40)
    print("Generation Completed Summary:")
    print(f"Total Scenes Attempted: {len(tasks_to_run)}")
    print(f"Successfully Annotated: {success_count}")
    print(f"Failed: {failure_count}")
    if failure_count:
        print(f"Failure details logged to: {error_log_path}")
    print("=" * 40)


# ──────────────────────────────────────────────────────────────────
# 2. LLM-as-Judge Quality Verification
# ──────────────────────────────────────────────────────────────────


def _normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def parse_verdict(text: str) -> Tuple[Optional[bool], str, str]:
    """Extract (valid, evidence, reason) from the model output."""
    if not text:
        return None, "", "empty judge output"
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None, "", f"no json in judge output: {text[:120]}"
    try:
        obj = json.loads(m.group(0))
        return bool(obj.get("valid")), str(obj.get("evidence", "")), str(obj.get("reason", ""))[:300]
    except Exception:
        low = text.lower()
        if '"valid": true' in low or '"valid":true' in low:
            return True, "", "parsed-fallback"
        if '"valid": false' in low or '"valid":false' in low:
            return None, "", f"unparseable-false judge output: {text[:120]}"
        return None, "", f"unparseable judge output: {text[:120]}"


async def _single_verdict(
    client: AsyncOpenAI,
    model: str,
    user_content: str,
    desc: str,
    max_retries: int,
    temperature: float = 0.6,
) -> Tuple[Optional[bool], str]:
    desc_norm = _normalize_whitespace(desc)
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": user_content},
                ],
                temperature=temperature,
                max_tokens=200,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            out = (resp.choices[0].message.content or "").strip()
            valid, evidence, reason = parse_verdict(out)
            if valid is None:
                last_err = reason
                continue
            if valid is False:
                ev = _normalize_whitespace(evidence)
                if len(ev) < 3 or ev not in desc_norm:
                    return True, f"ungrounded-evidence->valid (claimed: {reason[:80]})"
                return False, f"{reason} | evidence: {evidence[:80]}"
            return True, reason
        except Exception as exc:
            last_err = str(exc)
            if attempt < max_retries - 1:
                await asyncio.sleep(2 * (attempt + 1))
    return None, f"judge_failed: {last_err}"


async def _judge_single_scene(
    client: AsyncOpenAI,
    json_path: Path,
    model: str,
    max_retries: int,
    semaphore: asyncio.Semaphore,
    votes: int = 3,
    temperature: float = 0.6,
) -> Dict[str, Any]:
    async with semaphore:
        scene_id = json_path.parent.name
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            scene_id = data.get("scene_id", scene_id)
            desc = (data.get("scene_description") or "").strip()
        except Exception as exc:
            return {
                "scene_id": scene_id,
                "scene_path": str(json_path),
                "valid": False,
                "reason": f"load_error: {exc}",
                "judge_error": True,
            }

        if not desc:
            return {
                "scene_id": scene_id,
                "scene_path": str(json_path),
                "valid": False,
                "reason": "empty scene_description",
            }

        shown = desc if len(desc) <= 4000 else desc[:4000] + " ...[truncated]"
        user_content = f'scene_description:\n"""\n{shown}\n"""\n\nReturn the JSON verdict.'

        valid_count = invalid_count = 0
        reasons = []
        for _ in range(max(1, votes)):
            v, r = await _single_verdict(client, model, user_content, desc, max_retries, temperature)
            if v is True:
                valid_count += 1
            elif v is False:
                invalid_count += 1
                reasons.append(r)

        decided = valid_count + invalid_count
        if decided == 0:
            return {
                "scene_id": scene_id,
                "scene_path": str(json_path),
                "valid": None,
                "reason": "judge_failed: no parseable verdict",
                "judge_error": True,
            }

        is_valid = valid_count >= invalid_count
        return {
            "scene_id": scene_id,
            "scene_path": str(json_path),
            "valid": is_valid,
            "votes": {"valid": valid_count, "invalid": invalid_count},
            "reason": "" if is_valid else (reasons[0] if reasons else "majority invalid"),
        }


async def run_judge_async(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return

    out_path = Path(args.out) if args.out else input_dir / "judge_results.jsonl"

    done_ids: Set[str] = set()
    if out_path.exists() and not args.overwrite:
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["scene_path"])
                except Exception:
                    pass
        print(f"Resuming: {len(done_ids)} scenes already judged.")
    elif args.overwrite and out_path.exists():
        out_path.unlink()

    targets, total_scanned, _ = _resolve_scene_targets(input_dir, args.scene_list, skip_paths=done_ids)
    if args.scene_list:
        print(f"Scene-list mode. To judge: {len(targets)}")
    else:
        print(f"Found {total_scanned} scenes. To judge: {len(targets)}")

    if args.limit:
        targets = targets[: args.limit]
        print(f"Limited to first {args.limit}.")
    if not targets:
        print("Nothing to judge.")
        return

    client = AsyncOpenAI(base_url=args.api_url, api_key=args.api_key)
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        _judge_single_scene(client, p, args.model, args.max_retries, semaphore, args.votes, args.temperature)
        for p in targets
    ]

    n_valid = n_invalid = n_error = 0
    with open(out_path, "a", encoding="utf-8") as out_file:
        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Judging"):
            rec = await fut
            out_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_file.flush()
            if rec.get("judge_error") or rec.get("valid") is None:
                n_error += 1
            elif rec["valid"]:
                n_valid += 1
            else:
                n_invalid += 1

    print("\n" + "=" * 40)
    print("Judging complete:")
    print(f"  VALID:        {n_valid}")
    print(f"  INVALID:      {n_invalid}")
    print(f"  JUDGE ERRORS: {n_error}")
    print(f"  Results -> {out_path}")
    print("=" * 40)
    print("Tip: list invalid scenes with:")
    print(
        f'  python -c "import json;[print(json.loads(l)[\'scene_path\']) '
        f"for l in open('{out_path}') if json.loads(l).get('valid') is False]\""
    )


# ──────────────────────────────────────────────────────────────────
# CLI Entrypoint
# ──────────────────────────────────────────────────────────────────


def _add_shared_args(parser: argparse.ArgumentParser, default_concurrency: int) -> None:
    parser.add_argument(
        "--input-dir",
        type=str,
        default="datasets/3dish/autoregressive",
        help="Path to the autoregressive dataset directory.",
    )
    parser.add_argument("--api-url", type=str, default="http://127.0.0.1:8000/v1", help="vLLM server API base URL.")
    parser.add_argument("--api-key", type=str, default="EMPTY", help="API key for vLLM server.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.6-27B", help="Model name registered in vLLM.")
    parser.add_argument("--concurrency", type=int, default=default_concurrency, help="Max concurrent API requests.")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries on transient API errors.")
    parser.add_argument(
        "--scene-list",
        type=str,
        default=None,
        help="JSON array (or newline list) of scene json paths / dirs / ids to process.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of scenes to process.")


def main() -> None:
    for k in list(os.environ.keys()):
        if "proxy" in k.lower():
            del os.environ[k]

    parser = argparse.ArgumentParser(
        description="Generate and judge English scene descriptions for autoregressive datasets."
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Subcommands: generate (default) or judge")

    gen_parser = subparsers.add_parser("generate", help="Generate scene descriptions")
    _add_shared_args(gen_parser, default_concurrency=16)
    gen_parser.add_argument("--top-p", type=float, default=0.95, help="Top-p nucleus sampling.")
    gen_parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens to generate.")
    gen_parser.add_argument("--enable-thinking", action="store_true", help="Enable model thinking mode.")
    gen_parser.add_argument("--error-log", type=str, default=None, help="Path to JSONL failure log.")
    gen_parser.add_argument(
        "--mode",
        type=str,
        choices=["continue", "overwrite"],
        default="continue",
        help="'continue': only generate for scenes without a description; 'overwrite': regenerate all.",
    )

    judge_parser = subparsers.add_parser("judge", help="LLM-as-judge quality check for scene descriptions")
    _add_shared_args(judge_parser, default_concurrency=32)
    judge_parser.add_argument("--votes", type=int, default=3, help="Independent judge votes per scene.")
    judge_parser.add_argument("--out", type=str, default=None, help="Output JSONL path.")

    # Default to 'generate' if no subcommand is explicitly passed
    argv = sys.argv[1:]
    if not argv or argv[0] not in ("generate", "judge", "-h", "--help"):
        argv = ["generate"] + argv

    args = parser.parse_args(argv)
    if args.subcommand == "judge":
        asyncio.run(run_judge_async(args))
    else:
        asyncio.run(run_generate_async(args))


if __name__ == "__main__":
    main()
