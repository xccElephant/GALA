#!/usr/bin/env python3
"""
Transfer Utonia point encoder weights into a PointLM (LLaMA-Factory) checkpoint.

Reads a Utonia pretrained checkpoint (.pth from
https://github.com/Pointcept/Utonia), which carries both ``config`` and
``state_dict``; decoder weights are discarded if present.  The LLM half comes
from a Qwen3 backbone, and the MLP projector is always randomly initialised.

Usage:
    python scripts/transfer_point_encoder.py \
        --point_encoder_source https://huggingface.co/Pointcept/Utonia/blob/main/utonia.pth \
        --llm_backbone Qwen/Qwen3-0.6B \
        --output models/Utonia-Qwen3-0.6B
"""

import argparse
import json
import os
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.llamafactory.model.pointlm import PointLMQwen3Config, PointLMQwen3ForCausalLM  # noqa: E402


# ------------------------------------------------------------------
# Tokenizer setup & save
# ------------------------------------------------------------------

def setup_tokenizer_and_save(
    model: PointLMQwen3ForCausalLM,
    llm_backbone: str,
    output: str,
    summary: dict,
) -> None:
    """Add point tokens, update config IDs, and save model + tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(llm_backbone, trust_remote_code=True)

    point_tokens = ["<|point_start|>", "<|point_end|>", "<|point_pad|>"]
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": point_tokens})
    print(f"   Added {num_added} special tokens")

    new_start = tokenizer.convert_tokens_to_ids("<|point_start|>")
    new_end = tokenizer.convert_tokens_to_ids("<|point_end|>")
    new_pad = tokenizer.convert_tokens_to_ids("<|point_pad|>")

    model.config.point_start_token_id = new_start
    model.config.point_end_token_id = new_end
    model.config.point_token_id = new_pad

    print(f"   Token IDs: start={new_start}, end={new_end}, pad={new_pad}")

    model.point_backbone.to(torch.float32)
    model.point_proj.to(torch.bfloat16)
    model.model.to(torch.bfloat16)
    model.lm_head.to(torch.bfloat16)
    model.config.torch_dtype = torch.bfloat16

    os.makedirs(output, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)

    summary["vocab_size"] = len(tokenizer)
    summary["token_ids"] = {
        "point_start_token_id": new_start,
        "point_end_token_id": new_end,
        "point_token_id": new_pad,
    }
    summary["dtype_strategy"] = {
        "point_backbone": "float32",
        "point_proj": "bfloat16",
        "llm": "bfloat16",
    }
    with open(os.path.join(output, "transfer_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main_utonia(args) -> None:
    sep = "=" * 80
    print(sep)
    print("PointLM  Utonia Encoder Transfer Script (Qwen3)")
    print(sep)
    print(f"  Utonia Checkpoint   : {args.point_encoder_source}")
    print(f"  LLM Backbone        : {args.llm_backbone}")
    print(f"  Output               : {args.output}")
    print(sep)

    # ── Step 1: Load Utonia .pth checkpoint ────────────────────────────
    print("\n[1/6] Loading Utonia checkpoint ...")
    pth_path = args.point_encoder_source
    if not os.path.isfile(pth_path):
        # Try downloading from HuggingFace
        from huggingface_hub import hf_hub_download
        print(f"   Downloading {args.utonia_ckpt_name} from {args.utonia_repo_id} ...")
        pth_path = hf_hub_download(
            repo_id=args.utonia_repo_id,
            filename=args.utonia_ckpt_name,
            repo_type="model",
        )
        print(f"   Downloaded to: {pth_path}")

    import packaging.version
    if packaging.version.parse(torch.__version__) >= packaging.version.parse("2.4"):
        ckpt = torch.load(pth_path, map_location="cpu", weights_only=True)
    else:
        ckpt = torch.load(pth_path, map_location="cpu")

    if "config" not in ckpt or "state_dict" not in ckpt:
        raise ValueError(
            "Utonia checkpoint must contain 'config' and 'state_dict' keys. "
            f"Got: {list(ckpt.keys())}"
        )
    utonia_config = ckpt["config"]
    utonia_state_dict = ckpt["state_dict"]
    print(f"   {len(utonia_state_dict)} tensors loaded")
    print(f"   Utonia config: {json.dumps(utonia_config, indent=4, default=str)}")

    # ── Step 2: Build point_config from Utonia checkpoint config ──────
    print("\n[2/6] Building point_config ...")
    point_config = {
        "backbone_type": "utonia",
        "in_channels": utonia_config.get("in_channels", 6),
        "order": utonia_config.get("order", ["z", "z-trans"]),
        "stride": list(utonia_config.get("stride", [2, 2, 2, 2])),
        "enc_depths": list(utonia_config.get("enc_depths", [3, 3, 3, 12, 3])),
        "enc_channels": list(utonia_config.get("enc_channels", [48, 96, 192, 384, 512])),
        "enc_num_head": list(utonia_config.get("enc_num_head", [3, 6, 12, 24, 32])),
        "enc_patch_size": list(utonia_config.get("enc_patch_size", [1024, 1024, 1024, 1024, 1024])),
        "mlp_ratio": utonia_config.get("mlp_ratio", 4),
        "mask_token": utonia_config.get("mask_token", False),
        "enc_mode": True,
        "enable_flash": utonia_config.get("enable_flash", True),
        "rope_base": utonia_config.get("rope_base", 10),
        "drop_path": utonia_config.get("drop_path", 0.3),
        "utonia_grid_size": utonia_config.get("grid_size", 0.02),
    }
    print(f"   point_config: {json.dumps(point_config, indent=4)}")

    # ── Step 3: Filter state_dict to encoder-only keys ────────────────
    print("\n[3/6] Filtering to encoder-only keys ...")
    enc_only = utonia_config.get("enc_mode", False)
    if not enc_only:
        # Original checkpoint has both encoder and decoder; filter out decoder
        encoder_keys = {}
        skipped_prefixes = set()
        for k, v in utonia_state_dict.items():
            if k.startswith("dec.") or k.startswith("dec_"):
                skipped_prefixes.add(k.split(".")[0])
            else:
                encoder_keys[k] = v
        if skipped_prefixes:
            print(f"   Skipped decoder prefixes: {sorted(skipped_prefixes)}")
        utonia_state_dict = encoder_keys
    print(f"   {len(utonia_state_dict)} encoder tensors")

    # ── Step 4: Build PointLMQwen3 config ─────────────────────────────
    print("\n[4/6] Building PointLMQwen3 config ...")
    llm_config = AutoConfig.from_pretrained(args.llm_backbone, trust_remote_code=True)
    new_hidden_size = llm_config.hidden_size
    print(f"   LLM hidden_size: {new_hidden_size}")

    pointlm_config = PointLMQwen3Config(**llm_config.to_dict())
    pointlm_config.model_type = "pointlm_qwen3"
    pointlm_config.point_config = point_config
    pointlm_config.point_start_token_id = 0
    pointlm_config.point_end_token_id = 0
    pointlm_config.point_token_id = 0

    # ── Step 5: Initialise model & load weights ───────────────────────
    print("\n[5/6] Initialising PointLMQwen3ForCausalLM ...")
    model = PointLMQwen3ForCausalLM(pointlm_config)

    # 5a. LLM backbone
    print(f"   Loading LLM weights from {args.llm_backbone} ...")
    llm_model = AutoModelForCausalLM.from_pretrained(
        args.llm_backbone, trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    missing, unexpected = model.load_state_dict(llm_model.state_dict(), strict=False)
    print(f"   LLM weights loaded  (missing: {len(missing)}, unexpected: {len(unexpected)})")
    del llm_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 5b. Utonia encoder (fp32)
    print("   Loading Utonia encoder weights (fp32) ...")
    fp32_weights = {k: v.to(torch.float32) for k, v in utonia_state_dict.items()}
    load_result = model.point_backbone.load_state_dict(fp32_weights, strict=False)
    print(f"   point_backbone loaded ({len(fp32_weights)} tensors)")
    if load_result.missing_keys:
        print(f"   WARNING: missing keys ({len(load_result.missing_keys)}): {load_result.missing_keys[:10]}...")
    if load_result.unexpected_keys:
        print(f"   WARNING: unexpected keys ({len(load_result.unexpected_keys)}): {load_result.unexpected_keys[:10]}...")

    # 5c. Point projector — always randomly initialised for Utonia
    enc_channels = point_config["enc_channels"]
    embed_channels = enc_channels[-1]
    print(f"   point_proj: randomly initialized (embed_channels={embed_channels} -> hidden_size={new_hidden_size})")
    model.point_proj.to(torch.bfloat16)

    # ── Step 6: Tokenizer & save ──────────────────────────────────────
    print("\n[6/6] Handling tokenizer & saving ...")
    summary = {
        "encoder_type": "utonia",
        "point_encoder_source": args.point_encoder_source,
        "llm_backbone": args.llm_backbone,
        "model_type": "pointlm_qwen3",
        "new_hidden_size": new_hidden_size,
        "embed_channels": embed_channels,
        "projector_reinitialized": True,
        "utonia_config": utonia_config,
    }
    setup_tokenizer_and_save(model, args.llm_backbone, args.output, summary)

    print(f"\n{sep}")
    print("Transfer completed!")
    print(sep)
    print(f"  Output      : {args.output}")
    print(f"  Encoder     : Utonia (PTv3 Mode3)")
    print(f"  Model class : PointLMQwen3ForCausalLM")
    print(f"  Precision   : point_backbone=fp32, point_proj=bf16, LLM=bf16")
    print(f"  Embed dim   : {embed_channels} (encoder output -> projector input)")
    print("  Note: point_proj was randomly initialized – train it to adapt!")
    print(sep)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transfer a Utonia point encoder into a PointLM (Qwen3) checkpoint.",
    )
    parser.add_argument(
        "--point_encoder_source", type=str, required=True,
        help="Utonia .pth checkpoint path; downloaded from the HF repo when absent",
    )
    parser.add_argument(
        "--llm_backbone", type=str, required=True,
        help="Qwen3 LLM backbone path or HF Hub name",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for the assembled PointLM model",
    )
    parser.add_argument(
        "--utonia_repo_id", type=str, default="Pointcept/Utonia",
        help="HuggingFace repo ID for Utonia model (default: Pointcept/Utonia)",
    )
    parser.add_argument(
        "--utonia_ckpt_name", type=str, default="utonia.pth",
        help="Checkpoint filename in the Utonia HF repo (default: utonia.pth)",
    )

    args = parser.parse_args()
    main_utonia(args)


if __name__ == "__main__":
    main()
