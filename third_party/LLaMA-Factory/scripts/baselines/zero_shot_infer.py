"""Zero-shot LLM batch inference (vLLM / SGLang) for JSON instruction datasets.

Used to evaluate large zero-shot language model baselines (e.g., Qwen3, DeepSeek)
on the single-object placement test split. Outputs a predictions JSON file
compatible with ``scripts/pointlm_eval_score.py``.

Examples:
    # vLLM backend (default)
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/baselines/zero_shot_infer.py \
        --backend vllm \
        --data_file ../../datasets/processed/3dish_331k_test.json \
        --model_name_or_path Qwen/Qwen3-32B \
        --output results/eval/qwen3-32b/3dish_331k_test/predictions.json \
        --tensor_parallel_size 8

    # SGLang backend
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/baselines/zero_shot_infer.py \
        --backend sglang \
        --data_file ../../datasets/processed/3dish_331k_test.json \
        --model_name_or_path PhalaCloud/GLM-5.2-W4AFP8 \
        --output results/eval/glm-5.2-w4afp8/3dish_331k_test/predictions.json \
        --tensor_parallel_size 8 \
        --quantization w4afp8
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
import os
import re

from tqdm import tqdm
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-shot LLM batch inference (vLLM / SGLang)")
    parser.add_argument("--backend", type=str, default="vllm", choices=["vllm", "sglang"])
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-32B")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--tensor_parallel_size", type=int, default=8)
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max_model_len", "--context_length", dest="max_model_len", type=int, default=8192)
    parser.add_argument(
        "--gpu_memory_utilization",
        "--mem_fraction_static",
        dest="gpu_memory_utilization",
        type=float,
        default=0.85,
    )
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        choices=["low", "medium", "high"],
        default=None,
    )
    parser.add_argument("--enforce_eager", action="store_true")
    # SGLang-specific options
    parser.add_argument("--quantization", type=str, default=None)
    parser.add_argument("--kv_cache_dtype", type=str, default=None)
    parser.add_argument("--disable_shared_experts_fusion", action="store_true")
    parser.add_argument("--reasoning_parser", type=str, default=None)
    return parser.parse_args()


def iter_batches(items: list[dict], batch_size: int) -> Iterable[list[dict]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def build_prompt(
    tokenizer,
    sample: dict,
    enable_thinking: bool,
    reasoning_effort: str | None,
) -> str:
    messages = [
        {"role": "system", "content": sample["system"]},
        {"role": "user", "content": sample["instruction"]},
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


def extract_final_response(text: str, raw_or_meta: str | dict | None = None) -> str:
    """Strip reasoning/thinking segments and return the final answer text."""
    candidate = text
    if isinstance(raw_or_meta, str) and raw_or_meta:
        candidate = raw_or_meta
    elif isinstance(raw_or_meta, dict):
        for key in ("text", "content", "completion"):
            if key in raw_or_meta and isinstance(raw_or_meta[key], str) and raw_or_meta[key].strip():
                candidate = raw_or_meta[key]
                break

    final_marker = "<|channel|>final<|message|>"
    if final_marker in candidate:
        final_text = candidate.rsplit(final_marker, maxsplit=1)[1]
        for end_marker in ("<|return|>", "<|end|>"):
            final_text = final_text.split(end_marker, maxsplit=1)[0]
        return final_text.strip()

    thinking_stripped = re.sub(
        r"<think>.*?</think>\s*",
        "",
        text,
        count=1,
        flags=re.DOTALL,
    )
    if thinking_stripped != text:
        return thinking_stripped.strip()

    final_match = re.search(r"(?:^|\n)final\s*(.*)\Z", text.strip(), flags=re.DOTALL)
    if final_match:
        return final_match.group(1).strip()

    return text.strip()


def _run_vllm(args: argparse.Namespace, tokenizer, samples: list[dict], prompts: list[str]) -> list[dict]:
    from vllm import LLM, SamplingParams

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

    predictions = []
    for batch_samples in tqdm(list(iter_batches(samples, args.batch_size)), desc="vLLM inference"):
        batch_prompts = prompts[len(predictions) : len(predictions) + len(batch_samples)]
        outputs = llm.generate(batch_prompts, sampling_params)

        for sample, output in zip(batch_samples, outputs, strict=True):
            completion = output.outputs[0]
            raw_text = tokenizer.decode(completion.token_ids, skip_special_tokens=False)
            predictions.append(
                {
                    "sample_id": sample["sample_id"],
                    "prompt": sample["instruction"],
                    "label": sample["output"],
                    "predict": extract_final_response(completion.text, raw_text),
                    "point_clouds": sample.get("point_clouds", []),
                }
            )
    return predictions


def _run_sglang(args: argparse.Namespace, samples: list[dict], prompts: list[str]) -> list[dict]:
    import sglang as sgl

    engine_kwargs = {
        "model_path": args.model_name_or_path,
        "tp_size": args.tensor_parallel_size,
        "trust_remote_code": True,
        "context_length": args.max_model_len,
        "mem_fraction_static": args.gpu_memory_utilization,
    }
    if args.kv_cache_dtype:
        engine_kwargs["kv_cache_dtype"] = args.kv_cache_dtype
    if args.quantization:
        engine_kwargs["quantization"] = args.quantization
    if args.disable_shared_experts_fusion:
        engine_kwargs["disable_shared_experts_fusion"] = True
    if args.reasoning_parser:
        engine_kwargs["reasoning_parser"] = args.reasoning_parser

    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
    }
    if args.top_k >= 0:
        sampling_params["top_k"] = args.top_k

    llm = sgl.Engine(**engine_kwargs)
    predictions = []
    try:
        for batch_samples in tqdm(list(iter_batches(samples, args.batch_size)), desc="SGLang inference"):
            batch_prompts = prompts[len(predictions) : len(predictions) + len(batch_samples)]
            raw_outputs = llm.generate(batch_prompts, sampling_params)
            outputs = raw_outputs if isinstance(raw_outputs, list) else [raw_outputs]

            for sample, output in zip(batch_samples, outputs, strict=True):
                meta = output if isinstance(output, dict) else {}
                raw_text = meta.get("text", "") if isinstance(output, dict) else str(output)
                predictions.append(
                    {
                        "sample_id": sample["sample_id"],
                        "prompt": sample["instruction"],
                        "label": sample["output"],
                        "predict": extract_final_response(raw_text, meta),
                        "point_clouds": sample.get("point_clouds", []),
                    }
                )
    finally:
        llm.shutdown()
    return predictions


def main() -> None:
    args = parse_args()

    with open(args.data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    end_idx = None if args.num_samples is None else args.start_idx + args.num_samples
    samples = data[args.start_idx : end_idx]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    prompts = [
        build_prompt(tokenizer, sample, args.enable_thinking, args.reasoning_effort)
        for sample in samples
    ]

    if args.backend == "sglang":
        predictions = _run_sglang(args, samples, prompts)
    else:
        predictions = _run_vllm(args, tokenizer, samples, prompts)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
