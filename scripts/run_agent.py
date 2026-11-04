#!/usr/bin/env python3
"""
Agent 场景生成命令行入口

支持两种模式：
1. 生成模式：从用户需求生成完整场景
2. 编辑模式：在现有场景上添加物体

LLM 配置从 config/llm_key.json 读取，支持为不同任务指定不同模型。

用法:
    # 生成完整场景
    python scripts/run_agent.py generate \
        --model /path/to/model \
        --input "I want a modern bedroom with a double bed and nightstands" \
        --boundary "[[0,0],[4,0],[4,3],[0,3]]"

    # 编辑场景
    python scripts/run_agent.py edit \
        --model /path/to/model \
        --scene output/agent/scene_state.json \
        --input "Add a floor lamp next to the sofa"

    # 交互式模式
    python scripts/run_agent.py interactive \
        --model /path/to/model
"""

# 抑制 spconv 的 FutureWarning（PyTorch AMP API 废弃警告）
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="spconv")

import os

# 关闭 HF/transformers 权重加载进度条：重定向到日志文件时，tqdm 的 \r 无法覆盖，
# 会把单行进度条打成成千上万行（Loading weights ... Materializing param=...）。
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
from transformers.utils import logging as _hf_logging

_hf_logging.disable_progress_bar()

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
for _p in (str(SCRIPTS_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent.pipeline import SceneGenerationPipeline
from agent.scene_state import SceneStateManager
from config.paths import ProjectPaths
from scripts.physics_repair import repaired_layout_path, run_physics_repair_via_blender


@dataclass(frozen=True)
class BatchJob:
    """一次 batch generate 子进程任务。"""

    base_scene_id: str
    scene_id: str
    run_index: int
    input_text: str
    boundary: object
    height: float


def parse_boundary(boundary_str: str):
    """解析边界字符串"""
    try:
        return json.loads(boundary_str)
    except json.JSONDecodeError:
        print(f"Error: Invalid boundary format: {boundary_str}")
        print("Expected format: [[x1,y1],[x2,y2],...]")
        sys.exit(1)


def _normalize_boundary(boundary):
    if isinstance(boundary, str):
        try:
            return json.loads(boundary)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid boundary JSON: {boundary}") from exc
    return boundary


def _boundary_to_arg(boundary) -> str:
    return json.dumps(boundary, ensure_ascii=False, separators=(",", ":"))


def load_batch_jobs(jobs_path: str, runs_per_input: int) -> List[BatchJob]:
    if runs_per_input < 1:
        raise ValueError("--runs-per-input must be >= 1")

    with open(jobs_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if not isinstance(config, dict):
        raise ValueError("Batch jobs file must contain a JSON object")
    if "scenes" not in config:
        raise ValueError("Batch jobs file must contain a 'scenes' list")

    defaults = config["defaults"] if "defaults" in config else {}
    scenes = config["scenes"]
    if not isinstance(defaults, dict):
        raise ValueError("'defaults' must be a JSON object")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("'scenes' must be a non-empty list")

    jobs: List[BatchJob] = []
    for scene_index, scene in enumerate(scenes):
        if not isinstance(scene, dict):
            raise ValueError(f"Scene at index {scene_index} must be a JSON object")

        payload = dict(defaults)
        payload.update(scene)
        for key in ("scene_id", "input", "boundary"):
            if key not in payload:
                raise ValueError(f"Scene at index {scene_index} missing required key: {key}")

        base_scene_id = str(payload["scene_id"])
        height = float(payload["height"]) if "height" in payload else 2.6
        boundary = _normalize_boundary(payload["boundary"])
        for run_index in range(1, runs_per_input + 1):
            jobs.append(
                BatchJob(
                    base_scene_id=base_scene_id,
                    scene_id=f"{base_scene_id}_run{run_index:02d}",
                    run_index=run_index,
                    input_text=str(payload["input"]),
                    boundary=boundary,
                    height=height,
                )
            )
    return jobs


def parse_gpu_ids(gpus: str) -> List[str]:
    parts = [part.strip() for part in gpus.split(",")] if gpus is not None else []
    if not parts or any(part == "" for part in parts):
        raise ValueError("--gpus must be a comma-separated list of non-empty GPU ids")
    return parts


def resolve_max_parallel(max_parallel: Optional[int], gpu_ids: List[str]) -> int:
    if not gpu_ids:
        raise ValueError("--gpus must contain at least one GPU id")
    if max_parallel is None:
        return len(gpu_ids)
    if max_parallel < 1:
        raise ValueError("--max-parallel must be >= 1")
    if max_parallel > len(gpu_ids):
        raise ValueError("--max-parallel cannot exceed the number of GPU ids")
    return max_parallel


def build_generate_command(args, job: BatchJob, run_dirname: str) -> List[str]:
    script_path = Path(__file__).resolve()
    command = [sys.executable, str(script_path), "generate"]
    if args.model is not None:
        command.extend(["-m", args.model])

    command.extend(
        [
            "--asset-index",
            args.asset_index,
            "--asset-dir",
            args.asset_dir,
            "--output-dir",
            args.output_dir,
            "--max-retries",
            str(args.max_retries),
            "--infer-temp",
            str(args.infer_temp),
            "-i",
            job.input_text,
            "-b",
            _boundary_to_arg(job.boundary),
            "--height",
            str(job.height),
            "--scene-id",
            job.scene_id,
            "--run-dirname",
            run_dirname,
        ]
    )
    if args.no_pcd:
        command.append("--no-pcd")
    if args.include_desc:
        command.append("--include-desc")
    if args.visual_feedback:
        command.append("--visual-feedback")
    else:
        command.append("--no-visual-feedback")
    if args.no_save_renders:
        command.append("--no-save-renders")
    if args.internal_spaces_dir is not None:
        command.extend(["--internal-spaces-dir", args.internal_spaces_dir])
    if not args.repair_physics:
        command.append("--no-repair-physics")
    if not args.export_glb:
        command.append("--no-export-glb")
    if args.blender_bin is not None:
        command.extend(["--blender-bin", args.blender_bin])
    return command


def build_job_env(gpu_id: str, base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    return env


def unique_run_dirname(scene_id: str) -> str:
    """为每个 run 生成带时间戳的唯一目录名，避免重跑覆盖。"""
    return f"{scene_id}_{time.strftime('%Y%m%d_%H%M%S')}"


def _export_glb(layout_path: Path, output_dir: Path) -> int:
    export_script = Path(__file__).resolve().parent / "export_layout_scene.py"
    command = [
        sys.executable,
        str(export_script),
        "--layout",
        str(layout_path),
        "--output-dir",
        str(output_dir),
    ]
    completed = subprocess.run(command, check=False)
    return completed.returncode


def _run_batch_jobs(args, jobs: List[BatchJob], gpu_ids: List[str], max_parallel: int) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # 默认每个 run 的日志放进自己的场景目录（run.log）；显式 --log-dir 时则集中存放
    explicit_log_dir = Path(args.log_dir) if args.log_dir is not None else None
    if explicit_log_dir is not None:
        explicit_log_dir.mkdir(parents=True, exist_ok=True)

    active_gpu_ids = list(gpu_ids[:max_parallel])
    available_gpu_ids = list(active_gpu_ids)
    pending_jobs = list(jobs)
    running = []
    results = []
    batch_timestamp = time.strftime("%Y%m%d_%H%M%S")
    batch_started_at = time.time()

    while pending_jobs or running:
        while pending_jobs and available_gpu_ids:
            job = pending_jobs.pop(0)
            gpu_id = available_gpu_ids.pop(0)
            # 每个 job 一个带时间戳的唯一产物目录，父子进程共用同一目录名
            run_dirname = unique_run_dirname(job.scene_id)
            if explicit_log_dir is not None:
                log_path = explicit_log_dir / f"{job.scene_id}.log"
            else:
                job_dir = output_dir / run_dirname
                job_dir.mkdir(parents=True, exist_ok=True)
                log_path = job_dir / "run.log"
            log_file = open(log_path, "w", encoding="utf-8")
            command = build_generate_command(args, job, run_dirname)
            started_at = time.time()

            print(f"[batch] start {job.scene_id} on GPU {gpu_id}; log={log_path}")
            try:
                process = subprocess.Popen(
                    command,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=build_job_env(gpu_id),
                )
            except Exception as exc:
                log_file.write(f"Failed to start job: {exc}\n")
                log_file.close()
                available_gpu_ids.append(gpu_id)
                results.append(
                    {
                        "job": asdict(job),
                        "gpu_id": gpu_id,
                        "return_code": 127,
                        "layout_path": None,
                        "log_path": str(log_path),
                        "elapsed_time": 0.0,
                        "error": str(exc),
                    }
                )
                continue

            running.append(
                {
                    "job": job,
                    "gpu_id": gpu_id,
                    "run_dirname": run_dirname,
                    "process": process,
                    "log_file": log_file,
                    "log_path": log_path,
                    "started_at": started_at,
                }
            )

        if not running:
            continue

        completed_index = None
        while completed_index is None:
            for index, state in enumerate(running):
                return_code = state["process"].poll()
                if return_code is not None:
                    completed_index = index
                    break
            if completed_index is None:
                time.sleep(1.0)

        state = running.pop(completed_index)
        job = state["job"]
        gpu_id = state["gpu_id"]
        return_code = state["process"].returncode
        state["log_file"].close()
        available_gpu_ids.append(gpu_id)

        # GLB 导出/物理修正已在子进程内完成；这里只记录确定的 layout 路径
        run_dirname = state["run_dirname"]
        layout_path = (
            output_dir / run_dirname / f"{job.scene_id}_layout.json"
            if return_code == 0
            else None
        )
        if layout_path is not None and not layout_path.exists():
            layout_path = None

        elapsed_time = time.time() - state["started_at"]
        print(f"[batch] done {job.scene_id} on GPU {gpu_id}; return_code={return_code}")
        results.append(
            {
                "job": asdict(job),
                "gpu_id": gpu_id,
                "return_code": return_code,
                "layout_path": str(layout_path) if layout_path is not None else None,
                "log_path": str(state["log_path"]),
                "elapsed_time": elapsed_time,
                "error": None,
            }
        )

    failed = [result for result in results if result["return_code"] != 0]
    summary = {
        "started_at": batch_timestamp,
        "finished_at": time.strftime("%Y%m%d_%H%M%S"),
        "elapsed_time": time.time() - batch_started_at,
        "gpus": active_gpu_ids,
        "total": len(results),
        "successful": len(results) - len(failed),
        "failed": len(failed),
        "jobs": results,
    }
    summary_path = output_dir / f"batch_summary_{batch_timestamp}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary_path


def cmd_batch(args):
    """批量并发生成场景命令"""
    try:
        gpu_ids = parse_gpu_ids(args.gpus)
        max_parallel = resolve_max_parallel(args.max_parallel, gpu_ids)
        jobs = load_batch_jobs(args.jobs, runs_per_input=args.runs_per_input)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    active_gpu_ids = gpu_ids[:max_parallel]
    print(
        f"[batch] jobs={len(jobs)}, runs_per_input={args.runs_per_input}, "
        f"gpus={','.join(active_gpu_ids)}"
    )

    if args.dry_run:
        for index, job in enumerate(jobs):
            gpu_id = active_gpu_ids[index % len(active_gpu_ids)]
            print(f"[dry-run] {job.scene_id} -> GPU {gpu_id}")
        return

    summary_path = _run_batch_jobs(args, jobs, gpu_ids, max_parallel)
    print(f"[batch] summary saved to {summary_path}")
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    if summary["failed"] > 0:
        sys.exit(1)


def _print_timing_stats(timing_stats: Dict) -> None:
    """打印 pipeline 三模块耗时统计。"""
    if not timing_stats:
        return
    print("\nTiming Breakdown")
    print("-" * 60)
    labels = {
        "scene_planner": "Scene Planner",
        "instruction_generator": "Instruction Generator",
        "gala_inference": "GALA Inference",
    }
    for key, label in labels.items():
        stats = timing_stats[key]
        print(
            f"  {label:<24} {stats['total_s']:>8.2f}s  "
            f"({stats['calls']} calls, avg {stats['avg_s']:.2f}s)"
        )
    print(
        f"  {'Tracked total':<24} {timing_stats['tracked_total_s']:>8.2f}s  "
        f"/ {timing_stats['elapsed_time_s']:.2f}s elapsed "
        f"(other: {timing_stats['other_s']:.2f}s)"
    )


def cmd_generate(args):
    """生成场景命令"""
    # 解析边界
    boundary = parse_boundary(args.boundary)

    # 每个 run 的产物落到独立目录 <output_dir>/<scene_id>_<时间戳>/，避免重跑覆盖
    scene_id = args.scene_id
    run_dirname = args.run_dirname or unique_run_dirname(scene_id)
    out_dir = Path(args.output_dir) / run_dirname

    # 创建 Pipeline
    pipeline = SceneGenerationPipeline(
        asset_index_path=args.asset_index,
        inference_model_path=args.model,
        asset_pointcloud_dir=args.asset_dir,
        use_pointcloud=not args.no_pcd,
        internal_spaces_dir=args.internal_spaces_dir,
        use_visual_feedback=args.visual_feedback,
        save_renders=not args.no_save_renders,
        output_dir=str(out_dir),
        max_retries_per_object=args.max_retries,
        infer_temp=args.infer_temp,
        include_object_descriptions=args.include_desc,
        use_placement_validator=args.placement_validator,
    )

    # 生成场景
    result = pipeline.generate_scene(
        user_input=args.input,
        boundary=boundary,
        scene_height=args.height,
        scene_id=scene_id,
    )

    print("\n" + "=" * 60)
    print("Generation Result")
    print("=" * 60)
    print(f"Scene ID: {result['final_scene']['scene_id']}")
    print(f"Scene Type: {result['scene_type']}")
    print(f"Total Objects: {result['total_tasks']}")
    print(f"Successful: {result['successful']}")
    print(f"Failed: {result['failed']}")
    success_rate = result["successful"] / max(result["total_tasks"], 1)
    print(f"Success Rate: {success_rate:.1%}")
    print(f"Time: {result['elapsed_time']:.2f}s")
    _print_timing_stats(result.get("timing_stats"))
    print("=" * 60)

    layout_path = out_dir / f"{scene_id}_layout.json"

    # 导出原始 layout 的 GLB（次级产物，失败仅告警）
    if args.export_glb:
        _export_glb_safe(layout_path, out_dir)

    # 物理修正 + 导出 repaired 的 GLB
    if args.repair_physics:
        repaired = _repair_generated_layout(layout_path, args.blender_bin)
        if repaired is not None and args.export_glb:
            _export_glb_safe(repaired, out_dir)


def _export_glb_safe(layout_path: Path, output_dir: Path) -> None:
    """导出 GLB，失败仅告警（次级产物，不影响生成结果）。"""
    if not layout_path.exists():
        print(f"[glb] 未找到 layout 文件，跳过导出: {layout_path}", file=sys.stderr)
        return
    print(f"[glb] 导出中: {layout_path}")
    code = _export_glb(layout_path, output_dir)
    if code == 0:
        print(f"[glb] 已导出: {layout_path.with_suffix('.glb')}")
    else:
        print(f"[glb] 导出失败（return_code={code}），不影响生成结果", file=sys.stderr)


def _repair_generated_layout(layout_path: Path, blender_bin: Optional[str]) -> Optional[Path]:
    """生成后用 Blender 物理修正，额外保存 <stem>_repaired.json，返回其路径。

    这是次级产物：原始 layout 已保存，物理修正失败时只告警、不影响生成结果，
    以免在 batch 中因 Blender/资产问题拖垮整条流程。
    """
    if not layout_path.exists():
        print(f"[repair] 未找到 layout 文件，跳过物理修正: {layout_path}", file=sys.stderr)
        return None
    repaired = repaired_layout_path(layout_path)
    print(f"[repair] 物理修正中: {layout_path} -> {repaired}")
    try:
        run_physics_repair_via_blender(
            layout_path,
            repaired,
            dataset="auto",
            blender_bin=blender_bin,
        )
        print(f"[repair] 已保存物理修正 layout: {repaired}")
        return repaired
    except Exception as exc:
        print(
            f"[repair] 物理修正失败（原始 layout 已保存，不影响生成结果）: {exc}",
            file=sys.stderr,
        )
        return None


def cmd_edit(args):
    """编辑场景命令：在现有场景上添加物体。

    输入 -s 既支持 SceneStateManager.save() 的状态文件，也支持 pipeline 生成的 layout.json。
    编辑后的场景另存到 <output_dir>/<scene_id>_edit_<时间戳>/ 下，避免覆盖原始 run 的产物。
    """
    if not args.scene:
        print("Error: --scene is required for edit mode")
        sys.exit(1)

    # 先读出 scene_id 以确定每次编辑的独立产物目录
    with open(args.scene, "r", encoding="utf-8") as f:
        scene_source = json.load(f)
    scene_block = scene_source["final_scene"] if "final_scene" in scene_source else scene_source
    scene_id = scene_block["scene_id"]
    run_dirname = args.run_dirname or unique_run_dirname(f"{scene_id}_edit")
    out_dir = Path(args.output_dir) / run_dirname

    # 创建 Pipeline
    pipeline = SceneGenerationPipeline(
        asset_index_path=args.asset_index,
        inference_model_path=args.model,
        asset_pointcloud_dir=args.asset_dir,
        use_pointcloud=not args.no_pcd,
        internal_spaces_dir=args.internal_spaces_dir,
        use_visual_feedback=args.visual_feedback,
        save_renders=not args.no_save_renders,
        output_dir=str(out_dir),
        max_retries_per_object=args.max_retries,
        infer_temp=args.infer_temp,
        include_object_descriptions=args.include_desc,
        use_placement_validator=args.placement_validator,
    )

    # 加载场景
    pipeline.load_scene(args.scene)

    # 编辑场景
    result = pipeline.edit_scene(user_input=args.input)

    if result["success"]:
        print(f"\n✓ Successfully added: {result['short_desc']}")
        print(f"  Position: {result['pose']['center']}")
        print(f"  Rotation: {result['pose']['rotation']}")
    elif result.get("partial_success"):
        print(f"\n⚠ Added with issues: {result['short_desc']} (score: {result.get('violation_score', 0):.2f})")
        print(f"  Position: {result['pose']['center']}")
        print(f"  Rotation: {result['pose']['rotation']}")
        validation = result.get("validation", {})
        if validation.get("errors"):
            print(f"  Issues: {validation['errors']}")
    else:
        print(f"\n✗ Failed: {result.get('error', 'Unknown error')}")
        return

    _print_timing_stats(result.get("timing_stats"))

    # 持久化编辑后的场景（新 layout + 可选 glb / 物理修正），便于查看与回归
    layout_path = out_dir / f"{scene_id}_edit_layout.json"
    pipeline.state.save(str(layout_path))

    if args.export_glb:
        _export_glb_safe(layout_path, out_dir)
    if args.repair_physics:
        repaired = _repair_generated_layout(layout_path, args.blender_bin)
        if repaired is not None and args.export_glb:
            _export_glb_safe(repaired, out_dir)


def cmd_interactive(args):
    """交互式模式"""
    print("\n" + "=" * 60)
    print("Scene Agent - Interactive Mode")
    print("=" * 60)
    def print_help():
        print("Commands:")
        print("  new <boundary>  - Start a new scene (e.g., new [[0,0],[4,0],[4,3],[0,3]])")
        print("  gen <request>   - Generate scene from request")
        print("  add <request>   - Add object to current scene")
        print("  show            - Show current scene status")
        print("  undo            - Undo last placement")
        print("  save <path>     - Save current scene")
        print("  load <path>     - Load saved scene")
        print("  help            - Show this help message")
        print("  quit            - Exit")

    print_help()
    print("=" * 60 + "\n")

    # 创建 Pipeline
    pipeline = SceneGenerationPipeline(
        asset_index_path=args.asset_index,
        inference_model_path=args.model,
        asset_pointcloud_dir=args.asset_dir,
        use_pointcloud=not args.no_pcd,
        internal_spaces_dir=args.internal_spaces_dir,
        use_visual_feedback=args.visual_feedback,
        save_renders=not args.no_save_renders,
        output_dir=args.output_dir,
        max_retries_per_object=args.max_retries,
        infer_temp=args.infer_temp,
        include_object_descriptions=args.include_desc,
        use_placement_validator=args.placement_validator,
    )

    current_boundary = None

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        parts = user_input.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in ("quit", "exit", "q"):
            print("Bye!")
            break

        elif cmd == "new":
            if not arg:
                print("Usage: new [[x1,y1],[x2,y2],...]")
                continue
            try:
                current_boundary = json.loads(arg)
                # 创建空场景状态，这样可以直接使用 add 命令
                pipeline.state = SceneStateManager(
                    scene_id="interactive_scene",
                    boundary=current_boundary,
                    scene_height=2.6,
                    scene_description="Interactive scene",
                    scene_type="room",
                )
                print(f"New empty scene initialized with boundary: {current_boundary}")
                print("You can now use 'add' to add objects or 'gen' to generate a full scene.")
            except json.JSONDecodeError:
                print("Invalid boundary format")

        elif cmd == "gen":
            if not arg:
                print("Usage: gen <your scene request>")
                continue
            if not current_boundary:
                print("Please set boundary first with: new [[x1,y1],...]")
                continue

            print("\nGenerating scene... (this may take a while)")
            result = pipeline.generate_scene(
                user_input=arg,
                boundary=current_boundary,
            )

        elif cmd == "add":
            if not arg:
                print("Usage: add <description and placement>")
                continue
            if pipeline.state is None:
                print("No scene loaded. Use 'gen' or 'load' first.")
                continue

            result = pipeline.edit_scene(user_input=arg)
            if result["success"]:
                print(f"✓ Added: {result['short_desc']}")
            elif result.get("partial_success"):
                print(f"⚠ Added with issues: {result['short_desc']} (score: {result.get('violation_score', 0):.2f})")
                validation = result.get("validation", {})
                if validation.get("errors"):
                    print(f"  Issues: {validation['errors']}")
            else:
                print(f"✗ Failed: {result.get('error', 'Unknown error')}")

        elif cmd == "show":
            if pipeline.state is None:
                print("No scene loaded.")
            else:
                print(pipeline.state.get_context_for_agent())

        elif cmd == "undo":
            if pipeline.state is None:
                print("No scene loaded.")
            elif pipeline.state.rollback(1):
                print("Undone last placement")
            else:
                print("Nothing to undo")

        elif cmd == "save":
            if not arg:
                print("Usage: save <path>")
                continue
            if pipeline.state is None:
                print("No scene to save.")
            else:
                pipeline.state.save(arg)
                print(f"Saved to {arg}")

        elif cmd == "load":
            if not arg:
                print("Usage: load <path>")
                continue
            try:
                pipeline.load_scene(arg)
                current_boundary = pipeline.state.boundary
                print(
                    f"Loaded scene with {len(pipeline.state.existing_objects)} objects"
                )
            except Exception as e:
                print(f"Failed to load: {e}")

        elif cmd in ("help", "h", "?"):
            print_help()

        else:
            print(f"Unknown command: {cmd}")
            print("Type 'help' to see available commands.")


def cmd_build_index(args):
    """构建资产索引命令"""
    from agent.asset_retriever import build_asset_index_from_asset_info

    output_path = args.output or str(
        ProjectPaths.OUTPUT_DIR / f"asset_index_{args.dataset}"
    )
    asset_info_csv = ProjectPaths.get_asset_info_csv_path(args.dataset)
    if args.dataset == "3dfront":
        asset_dir = str(ProjectPaths.THREED_FUTURE_MODEL_DIR)
        if not asset_info_csv.exists():
            raise FileNotFoundError(
                f"3D-FRONT asset info 不存在: {asset_info_csv}\n"
                "请先运行: python scripts/run_agent.py build-asset-info --dataset 3dfront"
            )
    else:
        asset_dir = str(ProjectPaths.IMAGINARIUM_ASSETS_DIR)

    print(f"Building asset index for {args.dataset}...")
    print(f"  Asset Info: {asset_info_csv}")
    print(f"  Assets: {asset_dir}")
    print(f"  Output: {output_path}")
    build_asset_index_from_asset_info(
        asset_info_csv_path=str(asset_info_csv),
        asset_dir=asset_dir,
        output_path=output_path,
    )

    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Scene Agent - 智能场景生成系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # 通用参数
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="推理模型路径",
    )
    common_parser.add_argument(
        "--asset-index",
        type=str,
        default=str(ProjectPaths.OUTPUT_DIR / "asset_index_imaginarium"),
        help="资产索引文件路径",
    )
    common_parser.add_argument(
        "--asset-dir",
        type=str,
        default=str(ProjectPaths.IMAGINARIUM_ASSETS_DIR),
        help="资产目录",
    )
    common_parser.add_argument(
        "--no-pcd",
        action="store_true",
        help="不使用点云输入",
    )
    common_parser.add_argument(
        "--output-dir",
        type=str,
        default="output/agent",
        help="输出目录 (default: output/agent)",
    )
    common_parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="每个物体摆放失败时的最大重试次数 (default: 3, 设为 1 表示不重试)",
    )
    common_parser.add_argument(
        "--infer-temp",
        type=float,
        default=0.0,
        help="推理模型采样温度 (default: 0.0, 贪婪解码；调大以获得多样性，配合 --max-retries 使用)",
    )
    common_parser.add_argument(
        "--visual-feedback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="视觉反馈（渲染场景图给 Agent 参考，需要 VLM 支持），默认启用，可用 --no-visual-feedback 关闭",
    )
    common_parser.add_argument(
        "--no-save-renders",
        action="store_true",
        help="不保存渲染图（仅在启用视觉反馈时有效）",
    )
    common_parser.add_argument(
        "--internal-spaces-dir",
        type=str,
        default=None,
        help="收纳空间数据目录路径（传入即启用，不传则不附加；默认路径：ProjectPaths.IMAGINARIUM_INTERNAL_SPACES_DIR）",
    )
    common_parser.add_argument(
        "--include-desc",
        action="store_true",
        help="在推理指令中保留物体 short_desc（默认匿名化为 object）",
    )
    common_parser.add_argument(
        "--placement-validator",
        dest="placement_validator",
        action="store_true",
        help="开启摆放物理验证（碰撞/边界/高度/悬空）；默认关闭",
    )
    common_parser.set_defaults(placement_validator=False)
    common_parser.add_argument(
        "--no-repair-physics",
        dest="repair_physics",
        action="store_false",
        help="关闭：生成后不再用 Blender 物理修正额外保存 _repaired.json（默认开启）",
    )
    common_parser.set_defaults(repair_physics=True)
    common_parser.add_argument(
        "--no-export-glb",
        dest="export_glb",
        action="store_false",
        help="关闭：生成（及物理修正）成功后不再导出 GLB（默认导出 layout.glb / repaired.glb）",
    )
    common_parser.set_defaults(export_glb=True)
    common_parser.add_argument(
        "--blender-bin",
        type=str,
        default=None,
        help="Blender 可执行文件路径（默认自动查找 / 读 BLENDER_BIN 环境变量）",
    )

    # generate 子命令
    gen_parser = subparsers.add_parser(
        "generate",
        parents=[common_parser],
        help="从用户需求生成完整场景",
    )
    gen_parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="用户输入的场景需求描述",
    )
    gen_parser.add_argument(
        "-b",
        "--boundary",
        type=str,
        required=True,
        help='房间边界 JSON (e.g., "[[0,0],[4,0],[4,3],[0,3]]")',
    )
    gen_parser.add_argument(
        "--height",
        type=float,
        default=2.6,
        help="房间高度 (default: 2.6)",
    )
    gen_parser.add_argument(
        "--scene-id",
        type=str,
        default="generated_scene",
        help="场景 ID (default: generated_scene)",
    )
    gen_parser.add_argument(
        "--run-dirname",
        type=str,
        default=None,
        help="产物子目录名（默认在 scene-id 后追加时间戳，避免重跑覆盖）",
    )
    gen_parser.set_defaults(func=cmd_generate)

    # batch 子命令
    batch_parser = subparsers.add_parser(
        "batch",
        parents=[common_parser],
        help="从 JSON 配置批量并发生成场景",
    )
    batch_parser.add_argument(
        "--jobs",
        type=str,
        required=True,
        help="batch JSON 配置路径，包含 defaults 和 scenes",
    )
    batch_parser.add_argument(
        "--gpus",
        type=str,
        required=True,
        help='逗号分隔的 GPU id，例如 "4,5,6"',
    )
    batch_parser.add_argument(
        "--runs-per-input",
        type=int,
        default=1,
        help="每个输入重复生成的次数 (default: 1)",
    )
    batch_parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="最大并发 job 数，默认等于 GPU 数，不能超过 GPU 数",
    )
    batch_parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="batch job 日志目录，默认 output-dir/logs",
    )
    batch_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印展开后的任务和 GPU 分配，不启动 generate",
    )
    batch_parser.set_defaults(func=cmd_batch)

    # edit 子命令
    edit_parser = subparsers.add_parser(
        "edit",
        parents=[common_parser],
        help="在现有场景上添加物体",
    )
    edit_parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="编辑请求 (e.g., '再加一个落地灯放在沙发旁边')",
    )
    edit_parser.add_argument(
        "-s",
        "--scene",
        type=str,
        required=True,
        help="已保存的场景状态文件路径",
    )
    edit_parser.add_argument(
        "--run-dirname",
        type=str,
        default=None,
        help="产物子目录名（默认在 <scene-id>_edit 后追加时间戳，避免重跑覆盖）",
    )
    edit_parser.set_defaults(func=cmd_edit)

    # interactive 子命令
    int_parser = subparsers.add_parser(
        "interactive",
        parents=[common_parser],
        help="交互式模式",
    )
    int_parser.set_defaults(func=cmd_interactive)

    # build-index 子命令
    idx_parser = subparsers.add_parser(
        "build-index",
        help="构建资产索引",
    )
    idx_parser.add_argument(
        "--dataset",
        type=str,
        choices=["imaginarium", "3dfront"],
        default="imaginarium",
        help="数据集类型",
    )
    idx_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="输出索引文件路径",
    )
    idx_parser.set_defaults(func=cmd_build_index)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
