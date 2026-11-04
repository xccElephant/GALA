"""
PointLM Attention 分析脚本

分析模型在生成输出 token 时，对点云 token vs 文本 token 的注意力分配。
用于诊断模型是否真正利用了点云信息。

Usage:
    python scripts/pointlm_attention_analysis.py \
        -d ../../datasets/processed/3dish_separate_pointcloud_331k_test.json \
        -m saves/train_full_sft/qwen3-0_6b/3dish_separate_pointcloud_331k_train/checkpoint-6240 \
        --num_samples 20

    # 对比两个模型（例如有/无 Stage-1 预训练）
    python scripts/pointlm_attention_analysis.py \
        -d ../../datasets/processed/3dish_separate_pointcloud_331k_test.json \
        -m saves/train_full_sft/qwen3-0_6b/3dish_separate_pointcloud_331k_train/checkpoint-6240 \
        -m2 saves/train_full_sft/qwen3-0_6b/3dish_separate_pointcloud_331k_train_without_pretrain/checkpoint-6240 \
        --num_samples 20
"""

import argparse
import json
import os
import sys
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llamafactory.data.pcd_preprocess import make_random_noise_pcd_batch, preprocess_point_cloud_batch
from llamafactory.model.pointlm.utils import build_chatml_prompt, replace_point_cloud_placeholders


def load_model(model_path: str, dtype_str: str = "bfloat16"):
    from llamafactory.model.pointlm import register as register_pointlm

    register_pointlm()
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    dtype_map = {"bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map.get(dtype_str, torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="right")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.attn_implementation = "eager"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, torch_dtype=dtype, trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval().requires_grad_(False)
    if torch.cuda.is_available():
        model = model.cuda()

    point_config = getattr(config, "point_config", None)
    if not point_config:
        raise ValueError(f"{model_path} config.json has no 'point_config'.")

    return model, tokenizer, point_config["backbone_type"]


def find_pcd_token_ranges_from_pos_info(
    pos_info: list[tuple[int, int, int]],
) -> list[tuple[int, int]]:
    """Compute feature-token ranges in the expanded sequence from mixin's pos_info.

    pos_info is a list of (orig_start, orig_end, n_patches) tuples captured from
    _process_separate_point_clouds. Each tuple records the point_start / point_end
    positions in the *original* (pre-expansion) sequence and the exact number of
    feature tokens injected for that cloud.

    Returns list of (feat_start, feat_end) in the *expanded* sequence (Python slice
    convention: feat_start inclusive, feat_end exclusive).
    """
    ranges = []
    offset = 0  # cumulative shift from all previous injections
    for orig_start, orig_end, n_patches in pos_info:
        orig_placeholder_len = orig_end - orig_start + 1  # start + pad(s) + end
        expanded_ps_pos = orig_start + offset
        feat_start = expanded_ps_pos + 1
        feat_end = feat_start + n_patches
        ranges.append((feat_start, feat_end))
        # This cloud replaced orig_placeholder_len tokens with (1 + n_patches + 1).
        offset += (1 + n_patches + 1) - orig_placeholder_len
    return ranges


@torch.inference_mode()
def extract_generation_attention(
    model,
    tokenizer,
    prompt: str,
    pcd_paths: list[str],
    system_prompt: str,
    backbone_type: str,
    pcd_random_noise: bool = False,
) -> Optional[dict]:
    """Step-by-step generation, collecting per-token attention to pcd keys.

    Generates tokens one by one and records how much each newly generated token
    attends to point-cloud keys vs text keys across layers.
    """
    user_content = replace_point_cloud_placeholders(prompt)
    full_prompt = build_chatml_prompt(user_content, system_prompt)

    encoding = tokenizer(full_prompt, return_tensors="pt")
    input_ids = encoding["input_ids"].to(model.device)
    attention_mask = encoding["attention_mask"].to(model.device)

    captured_pos_info: list[list[tuple[int, int, int]]] = []

    pcd_tensor = None
    pcd_counts = None
    if pcd_paths:
        if pcd_random_noise:
            pcd_tensor = torch.as_tensor(make_random_noise_pcd_batch(len(pcd_paths)))
        else:
            pcd_tensor = torch.as_tensor(preprocess_point_cloud_batch(
                pcd_paths, backbone_type=backbone_type,
            ))
        pcd_tensor = pcd_tensor.to(model.device)
        pcd_counts = torch.tensor([len(pcd_paths)], dtype=torch.long, device=model.device)

        _orig_process = model._process_separate_point_clouds

        def _hooked_process(*args, **kwargs):
            result = _orig_process(*args, **kwargs)
            _ie, _am, all_pos_info, _af = result
            captured_pos_info.extend(all_pos_info)
            return result

        model._process_separate_point_clouds = _hooked_process

    # --- Prefill: one forward to fill KV-cache and get expanded-sequence layout ---
    try:
        prefill_out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True,
            use_cache=True,
            separate_point_clouds=pcd_tensor,
            separate_point_clouds_counts=pcd_counts,
        )
    finally:
        if pcd_paths:
            model._process_separate_point_clouds = _orig_process

    prefill_attentions = prefill_out.attentions
    if prefill_attentions is None or len(prefill_attentions) == 0:
        print("[WARN] No attention weights returned during prefill.")
        return None

    n_layers = len(prefill_attentions)
    prefill_len = prefill_attentions[0].shape[-1]

    pos_info_sample0 = captured_pos_info[0] if captured_pos_info else []
    pcd_ranges = find_pcd_token_ranges_from_pos_info(pos_info_sample0)

    pcd_key_mask = torch.zeros(prefill_len, dtype=torch.bool, device=model.device)
    for start, end in pcd_ranges:
        pcd_key_mask[start:end] = True
    n_pcd_keys = pcd_key_mask.sum().item()
    if n_pcd_keys == 0:
        return None

    past_key_values = prefill_out.past_key_values
    next_token = prefill_out.logits[:, -1:, :].argmax(dim=-1)

    cur_attention_mask = torch.ones(
        1, prefill_len + 1, dtype=attention_mask.dtype, device=model.device
    )
    eos_id = tokenizer.eos_token_id

    # Per-generated-token, per-layer attention-to-pcd ratio
    gen_token_stats: list[dict] = []
    generated_ids: list[int] = []

    max_pos = getattr(model.config, "max_position_embeddings", None)
    # Analyze all generated tokens until EOS, constrained by context length.
    max_steps = max(0, int(max_pos) - int(prefill_len)) if max_pos is not None else 1024

    for step in range(max_steps):
        token_id = next_token[0, 0].item()
        if token_id == eos_id:
            break
        generated_ids.append(token_id)

        cache_len = past_key_values.get_seq_length() if hasattr(past_key_values, "get_seq_length") else prefill_len + step
        cache_pos = torch.tensor([cache_len], device=model.device)
        position_ids = cache_pos.unsqueeze(0)

        step_out = model.model(
            input_ids=None,
            inputs_embeds=model.model.embed_tokens(next_token),
            attention_mask=cur_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=True,
            return_dict=True,
            cache_position=cache_pos,
        )

        past_key_values = step_out.past_key_values
        step_attentions = step_out.attentions  # tuple of (1, n_heads, 1, total_key_len)

        if step_attentions and len(step_attentions) == n_layers:
            layer_attn_to_pcd = []
            for layer_idx, aw in enumerate(step_attentions):
                # aw: (1, n_heads, 1, key_len)
                aw_f = aw[0, :, 0, :].float()  # (n_heads, key_len)
                key_len = aw_f.shape[-1]
                if key_len > pcd_key_mask.shape[0]:
                    extended_mask = torch.zeros(key_len, dtype=torch.bool, device=model.device)
                    extended_mask[:pcd_key_mask.shape[0]] = pcd_key_mask
                else:
                    extended_mask = pcd_key_mask[:key_len]
                attn_pcd = aw_f[:, extended_mask].sum(dim=-1).mean().item()
                layer_attn_to_pcd.append(attn_pcd)
            gen_token_stats.append({
                "step": step,
                "token_id": token_id,
                "token_str": tokenizer.decode([token_id]),
                "layer_attn_to_pcd": layer_attn_to_pcd,
                "avg_attn_to_pcd": float(np.mean(layer_attn_to_pcd)),
            })

        logits = model.lm_head(step_out.last_hidden_state[:, -1:, :])
        next_token = logits.argmax(dim=-1)
        cur_attention_mask = torch.ones(
            1, cache_len + 2, dtype=attention_mask.dtype, device=model.device
        )

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    pcd_key_ratio = n_pcd_keys / prefill_len

    per_layer_avg = []
    for layer_idx in range(n_layers):
        vals = [s["layer_attn_to_pcd"][layer_idx] for s in gen_token_stats]
        per_layer_avg.append(float(np.mean(vals)) if vals else 0.0)

    return {
        "n_layers": n_layers,
        "prefill_len": prefill_len,
        "n_generated": len(generated_ids),
        "generated_text": generated_text,
        "n_pcd_keys": n_pcd_keys,
        "pcd_key_ratio": pcd_key_ratio,
        "pcd_ranges": pcd_ranges,
        "gen_token_stats": gen_token_stats,
        "gen_per_layer_avg_attn_to_pcd": per_layer_avg,
        "gen_overall_avg_attn_to_pcd": float(np.mean(per_layer_avg)) if per_layer_avg else 0.0,
    }


def print_summary(gen_results: list[dict], model_name: str):
    """Print aggregated generation-phase attention statistics."""
    if not gen_results:
        print(f"\n[{model_name}] No valid generation results to summarize.")
        return

    n_layers = gen_results[0]["n_layers"]
    n_samples = len(gen_results)
    avg_pcd_ratio = np.mean([r["pcd_key_ratio"] for r in gen_results])
    avg_n_pcd = np.mean([r["n_pcd_keys"] for r in gen_results])
    avg_n_gen = np.mean([r["n_generated"] for r in gen_results])
    avg_prefill = np.mean([r["prefill_len"] for r in gen_results])

    print(f"\n{'='*72}")
    print(f"  Generation Attention Analysis: {model_name}")
    print(f"  Samples: {n_samples}, Layers: {n_layers}")
    print(f"  Avg prefill length: {avg_prefill:.0f}, Avg PCD keys: {avg_n_pcd:.0f}")
    print(f"  Avg generated tokens: {avg_n_gen:.1f}")
    print(f"  PCD key ratio (baseline): {avg_pcd_ratio:.2%}")
    print(f"{'='*72}")

    print(f"\n{'Layer':>6} | {'Gen→PCD(%)':>10} | {'Attn/Base':>10} | {'Verdict':>10}")
    print(f"{'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    for layer in range(n_layers):
        g_vals = [r["gen_per_layer_avg_attn_to_pcd"][layer] for r in gen_results]
        g2p = float(np.mean(g_vals))
        gen_ratio = g2p / avg_pcd_ratio if avg_pcd_ratio > 0 else 0.0

        if gen_ratio > 1.5:
            verdict = "✓ uses"
        elif gen_ratio > 1.0:
            verdict = "~ weak"
        else:
            verdict = "✗ ignores"

        print(f"{layer:>6} | {g2p * 100:>8.2f}% | {gen_ratio:>9.2f}x | {verdict:>10}")

    overall_gen = np.mean([r["gen_overall_avg_attn_to_pcd"] for r in gen_results])
    overall_gen_ratio = overall_gen / avg_pcd_ratio if avg_pcd_ratio > 0 else 0.0
    print(f"\n  Ratio = (avg attention to PCD keys) / (PCD key ratio).")
    print(f"  >1 = over-attending to PCD; ≈1 = uniform; <1 = under-attending.")
    print(f"  Overall: Gen→PCD avg = {overall_gen:.2%}, baseline = {avg_pcd_ratio:.2%}, ratio = {overall_gen_ratio:.2f}x")

    # Per-token detail for first sample
    first = gen_results[0]
    if first["gen_token_stats"]:
        print(f"\n  Token-level detail (sample 0, {len(first['gen_token_stats'])} tokens):")
        print(f"  {'Step':>5} | {'Token':>15} | {'Avg→PCD(%)':>10} | {'Attn/Base':>12}")
        print(f"  {'-'*5}-+-{'-'*15}-+-{'-'*10}-+-{'-'*12}")
        for s in first["gen_token_stats"]:
            tok = repr(s["token_str"])[:15]
            r = s["avg_attn_to_pcd"] / avg_pcd_ratio if avg_pcd_ratio > 0 else 0
            print(f"  {s['step']:>5} | {tok:>15} | {s['avg_attn_to_pcd'] * 100:>8.2f}% | {r:>11.2f}x")


def _parse_sample(sample: dict, idx: int, pcd_random_noise: bool):
    """Extract prompt, system, pcd_paths from a data sample. Returns None if invalid."""
    sample_id = sample.get("sample_id", idx)
    raw_pcds = sample.get("point_clouds", [])
    pcd_paths = []
    for p in raw_pcds:
        if os.path.exists(p):
            pcd_paths.append(p)
        else:
            pcd_paths = []
            break

    if not pcd_paths and not pcd_random_noise:
        return None

    if "instruction" in sample:
        prompt = sample["instruction"]
        system = sample.get("system", "You are a helpful assistant.")
    elif "conversations" in sample:
        prompt = sample["conversations"][0]["value"]
        system = "You are a helpful assistant."
    else:
        return None

    return sample_id, prompt, system, pcd_paths


def run_analysis(
    model_path: str,
    data: list[dict],
    num_samples: int,
    dtype_str: str,
    pcd_random_noise: bool = False,
) -> list[dict]:
    print(f"\nLoading model: {model_path}")
    model, tokenizer, backbone_type = load_model(model_path, dtype_str)

    model_name = os.path.basename(model_path)
    gen_results = []

    for idx, sample in enumerate(data[:num_samples]):
        parsed = _parse_sample(sample, idx, pcd_random_noise)
        if parsed is None:
            continue
        sample_id, prompt, system, pcd_paths = parsed

        print(f"  [{idx+1}/{num_samples}] sample_id={sample_id}, "
              f"n_pcds={len(pcd_paths)}", end="", flush=True)

        try:
            gen_result = extract_generation_attention(
                model, tokenizer, prompt, pcd_paths, system,
                pcd_random_noise=pcd_random_noise,
                backbone_type=backbone_type,
            )
            if gen_result is not None:
                gen_result["sample_id"] = sample_id
                gen_results.append(gen_result)
                print(f" → pcd_keys={gen_result['n_pcd_keys']}, "
                      f"gen={gen_result['n_generated']}tok", end="")
            else:
                print(" → skipped (no pcd tokens)", end="")
        except torch.cuda.OutOfMemoryError:
            print(" → OOM, skipped")
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f" → error: {e}")
            continue

        print()

    print_summary(gen_results, model_name)

    del model
    torch.cuda.empty_cache()

    return gen_results


def main():
    parser = argparse.ArgumentParser(description="PointLM attention analysis")
    parser.add_argument("-d", "--data_file", type=str, required=True)
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Primary model")
    parser.add_argument("-m2", "--model_path_2", type=str, default=None, help="Comparison model")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--inference_dtype", type=str, default="bfloat16")
    parser.add_argument("--pcd_random_noise", action="store_true", default=False)
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Save raw results as JSON")
    args = parser.parse_args()

    print(f"Loading data from {args.data_file} ...")
    with open(args.data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Total samples available: {len(data)}")

    results_1 = run_analysis(
        args.model_path, data, args.num_samples,
        args.inference_dtype, args.pcd_random_noise,
    )

    results_2 = None
    if args.model_path_2:
        results_2 = run_analysis(
            args.model_path_2, data, args.num_samples,
            args.inference_dtype, args.pcd_random_noise,
        )

    if args.output:
        output_data = {
            "model_1": {"path": args.model_path, "generation": results_1},
        }
        if results_2 is not None:
            output_data["model_2"] = {"path": args.model_path_2, "generation": results_2}

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\nRaw results saved to {args.output}")


if __name__ == "__main__":
    main()
