import json
import os
import csv
import re
import argparse
from pathlib import Path
from typing import Optional

from tqdm import tqdm

DEFAULT_MODEL = "Qwen/Qwen3-VL-32B-Instruct"
MODEL_PATH = os.environ.get("SHORT_DESC_MODEL", DEFAULT_MODEL)
DEFAULT_3DFRONT_CSV = "datasets/3D-FRONT/3D-FUTURE-model/3dfront_asset_info.csv"
DEFAULT_MAX_MODEL_LEN = 4096
DEFAULT_MAX_NUM_SEQS = 64
DEFAULT_GPU_MEMORY_UTILIZATION = 0.80
DEFAULT_ENFORCE_EAGER = True

_llm = None
_tokenizer = None
_sampling_params = None
_loaded_model_path: Optional[str] = None


def _resolve_num_gpus() -> int:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible_devices:
        return len([device for device in visible_devices.split(",") if device.strip()])

    import torch

    return torch.cuda.device_count()


def _ensure_model(
    model_path: Optional[str] = None,
    *,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    max_num_seqs: int = DEFAULT_MAX_NUM_SEQS,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    enforce_eager: bool = DEFAULT_ENFORCE_EAGER,
):
    global _llm, _tokenizer, _sampling_params, _loaded_model_path

    resolved_model_path = model_path or MODEL_PATH
    if _llm is not None:
        if _loaded_model_path != resolved_model_path:
            raise RuntimeError(
                f"Model already loaded as {_loaded_model_path}; "
                f"cannot switch to {resolved_model_path} in the same process"
            )
        return _llm, _tokenizer, _sampling_params

    from vllm import LLM, SamplingParams  # type: ignore[import-not-found]

    num_gpus = _resolve_num_gpus()
    print(f"Loading model: {resolved_model_path}")
    _llm = LLM(
        model=resolved_model_path,
        tensor_parallel_size=num_gpus,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
    )
    _tokenizer = _llm.get_tokenizer()
    _sampling_params = SamplingParams(temperature=0.2, top_p=0.95, max_tokens=50)
    _loaded_model_path = resolved_model_path
    return _llm, _tokenizer, _sampling_params


def create_prompt(full_desc):
    """Create prompt for generating short description"""
    _, tokenizer, _ = _ensure_model()
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that generates concise object descriptions.",
        },
        {
            "role": "user",
            "content": f"""Generate a short description (3-5 words) from the detailed description below. 

Requirements:
1. Use ONLY English words
2. Format: [adjective(s)] + [specific object type]
3. End with a SPECIFIC object category (e.g., sofa, table, chair, lamp, nightstand) NOT generic words like "furniture"
4. Do NOT use articles (a, an, the)
5. Use spaces only, NO punctuation (no commas, hyphens, semicolons)
6. Words should be in lowercase.

Examples:
- Input: "Modern gray loveseat sofa with fabric upholstery and metal legs"
  Output: gray fabric loveseat

- Input: "Modern minimalist coffee table with black glass top"
  Output: black glass coffee table

- Input: "Contemporary gray upholstered dining chair"
  Output: gray upholstered dining chair

Detailed description: {full_desc}

Short description:""",
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return prompt


def create_category_prompt(desc):
    """Create prompt for generating object category (a single lowercase word)."""
    _, tokenizer, _ = _ensure_model()
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that categorizes objects.",
        },
        {
            "role": "user",
            "content": f"""Given the object description below, output a SINGLE lowercase English word that represents the object category.

Requirements:
1. Output exactly ONE word, lowercase, no punctuation
2. Use a specific but common category noun (e.g., sofa, table, chair, lamp, nightstand, desk, shelf, cabinet, bed, rug, mirror, curtain, stool, dresser, ottoman, bench, bookcase, wardrobe, vase, clock)
3. Do NOT output adjectives, articles, or multiple words

Examples:
- Input: "gray fabric loveseat"
  Output: sofa

- Input: "black glass coffee table"
  Output: table

- Input: "gray upholstered dining chair"
  Output: chair

- Input: "modern minimalist floor lamp with brass finish"
  Output: lamp

- Input: "wooden bedside nightstand with two drawers"
  Output: nightstand

Description: {desc}

Category:""",
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return prompt


def _normalize_category(text: str) -> str:
    """Extract a single lowercase word as category."""
    if not isinstance(text, str):
        return ""
    line = text.strip().splitlines()[0].strip().lower()
    line = re.sub(r"[^a-z]+", " ", line).strip()
    # Take only the first word
    return line.split()[0] if line else ""


def generate_short_descriptions_batch(descriptions, *, verbose: bool = False):
    """Generate short descriptions in batch"""
    llm, _, sampling_params = _ensure_model()
    prompts = [create_prompt(desc) for desc in descriptions]
    outputs = llm.generate(prompts, sampling_params)

    results = []
    for output in outputs:
        short_desc = output.outputs[0].text
        if verbose:
            print("\n" + short_desc)

        results.append(short_desc)

    return results


def generate_categories_batch(descriptions, *, verbose: bool = False):
    """Generate categories in batch from descriptions (short_desc or full desc)."""
    llm, _, sampling_params = _ensure_model()
    prompts = [create_category_prompt(desc) for desc in descriptions]
    outputs = llm.generate(prompts, sampling_params)

    results = []
    for output in outputs:
        cat = output.outputs[0].text
        if verbose:
            print("\n" + cat)
        results.append(cat)

    return results


def _iter_captioned_objects(scene_data):
    """
    Return a list of object dicts that contain a non-empty English caption.

    Supports both formats:
    - scene_data["objects"] is a list[dict]
    - scene_data["objects"] is a dict[str, dict] (Imaginarium *_meta.json)
    """
    objects = scene_data.get("objects", None)
    if isinstance(objects, list):
        candidates = [obj for obj in objects if isinstance(obj, dict)]
    elif isinstance(objects, dict):
        candidates = [obj for obj in objects.values() if isinstance(obj, dict)]
    else:
        raise TypeError(
            f'Unsupported "objects" type: {type(objects)}. Expected list or dict.'
        )

    captioned = []
    for obj in candidates:
        cap = obj.get("caption_en", None)
        if isinstance(cap, str) and cap.strip():
            captioned.append(obj)
    return captioned


def _normalize_short_desc(text: str) -> str:
    # Keep only the first line, strip whitespace, enforce: lowercase + spaces only.
    if not isinstance(text, str):
        return ""
    line = text.strip().splitlines()[0].strip()
    line = line.lower()
    # Replace any non-letter with space (keep ASCII letters only to match "ONLY English words")
    line = re.sub(r"[^a-z]+", " ", line)
    line = re.sub(r"\s+", " ", line).strip()
    return line


def process_scene_file(input_path):
    """Process a single scene file"""
    with open(input_path, "r", encoding="utf-8") as f:
        scene_data = json.load(f)

    objs = _iter_captioned_objects(scene_data)
    if not objs:
        print(f"No valid caption_en found in {input_path}, skip.")
        return

    descriptions = [obj["caption_en"].strip() for obj in objs]
    print("Example captions:")
    for d in descriptions[:20]:
        print("-", d)

    short_descriptions = generate_short_descriptions_batch(descriptions, verbose=True)
    sd_normed = [_normalize_short_desc(sd) for sd in short_descriptions]
    for obj, sd in zip(objs, sd_normed):
        obj["short_desc"] = sd

    categories = generate_categories_batch(sd_normed, verbose=True)
    for obj, cat in zip(objs, categories):
        obj["category"] = _normalize_category(cat)

    # Save back to the original file
    with open(input_path, "w", encoding="utf-8") as f:
        json.dump(scene_data, f, indent=2, ensure_ascii=False)


def process_all_scenes(input_dir):
    """Process all scene files"""
    input_dir = Path(input_dir)

    scene_files = sorted(list(input_dir.glob("*.json")))
    print(f"Found {len(scene_files)} scene files\n")

    for scene_file in tqdm(scene_files):
        try:
            process_scene_file(scene_file)
        except Exception as e:
            print(f"Error processing {scene_file}: {e}")
            continue


def process_csv(
    csv_path: str,
    *,
    desc_col: str = "desc",
    short_desc_col: str = "short_desc",
    category_col: str = "category",
    output_path: Optional[str] = None,
    batch_size: int = 256,
    overwrite: bool = False,
):
    """
    Read a CSV and generate missing columns (short_desc / category).

    Auto-detect mode based on existing columns:
    - If CSV has desc but no short_desc → generate both short_desc and category
    - If CSV already has short_desc → only generate category

    Category is a single lowercase word representing the object type.
    """
    in_path = Path(csv_path)
    out_path = Path(output_path) if output_path else in_path
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    with open(in_path, "r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {csv_path}")

        fieldnames = list(reader.fieldnames)
        if desc_col not in fieldnames:
            raise KeyError(f'Missing column "{desc_col}" in {csv_path}')

        if short_desc_col not in fieldnames:
            fieldnames.append(short_desc_col)
            print(f"Column '{short_desc_col}' not found → will add it")
        else:
            print(f"Column '{short_desc_col}' found → will fill missing rows")

        if category_col not in fieldnames:
            fieldnames.append(category_col)

        processed_rows = 0
        gen_short_desc_rows = 0
        gen_category_rows = 0

        # Pending rows that need short_desc generation (mode: need both)
        pending_sd_rows = []
        pending_sd_descs = []

        # Pending rows that need category generation only (mode: already have short_desc)
        pending_cat_rows = []
        pending_cat_descs = []

        with open(tmp_path, "w", encoding="utf-8", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()

            pbar = tqdm(desc="Processing", unit="rows")

            def _flush_short_desc():
                nonlocal pending_sd_rows, pending_sd_descs, gen_short_desc_rows
                if not pending_sd_descs:
                    return
                # Step 1: generate short_desc from full desc
                sd_results = generate_short_descriptions_batch(pending_sd_descs, verbose=False)
                sd_normed = [_normalize_short_desc(x) for x in sd_results]
                for r, sd in zip(pending_sd_rows, sd_normed):
                    r[short_desc_col] = sd
                gen_short_desc_rows += len(pending_sd_rows)

                # Step 2: generate category from short_desc
                cat_results = generate_categories_batch(sd_normed, verbose=False)
                cat_normed = [_normalize_category(x) for x in cat_results]
                for r, cat in zip(pending_sd_rows, cat_normed):
                    r[category_col] = cat
                    writer.writerow(r)
                pbar.update(len(pending_sd_rows))
                pending_sd_rows = []
                pending_sd_descs = []

            def _flush_category():
                nonlocal pending_cat_rows, pending_cat_descs, gen_category_rows
                if not pending_cat_descs:
                    return
                cat_results = generate_categories_batch(pending_cat_descs, verbose=False)
                cat_normed = [_normalize_category(x) for x in cat_results]
                for r, cat in zip(pending_cat_rows, cat_normed):
                    r[category_col] = cat
                    writer.writerow(r)
                gen_category_rows += len(pending_cat_rows)
                pbar.update(len(pending_cat_rows))
                pending_cat_rows = []
                pending_cat_descs = []

            for row in reader:
                processed_rows += 1
                desc = row.get(desc_col, "")
                desc = desc.strip() if isinstance(desc, str) else ""

                existing_sd = row.get(short_desc_col, "")
                existing_sd = existing_sd.strip() if isinstance(existing_sd, str) else ""
                existing_cat = row.get(category_col, "")
                existing_cat = existing_cat.strip() if isinstance(existing_cat, str) else ""

                if (not desc) or ((not overwrite) and existing_sd and existing_cat):
                    writer.writerow(row)
                    pbar.update(1)
                    continue

                if (not overwrite) and existing_sd:
                    pending_cat_rows.append(row)
                    pending_cat_descs.append(existing_sd)
                    if len(pending_cat_descs) >= batch_size:
                        _flush_category()
                else:
                    pending_sd_rows.append(row)
                    pending_sd_descs.append(desc)
                    if len(pending_sd_descs) >= batch_size:
                        _flush_short_desc()

            _flush_short_desc()
            _flush_category()
            pbar.close()

    os.replace(tmp_path, out_path)
    print(f"Done. Processed rows: {processed_rows}")
    if gen_short_desc_rows:
        print(f"  Generated short_desc: {gen_short_desc_rows}")
    if gen_category_rows:
        print(f"  Generated category: {gen_category_rows}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate missing short_desc/category columns for asset CSVs."
    )
    parser.add_argument(
        "--csv",
        default=DEFAULT_3DFRONT_CSV,
        help="Input CSV path. Defaults to the 3D-FRONT asset info CSV.",
    )
    parser.add_argument(
        "--desc-col",
        default="desc",
        help='Description column name used as the source text. Default: "desc".',
    )
    parser.add_argument(
        "--short-desc-col",
        default="short_desc",
        help='Short description column name. Default: "short_desc".',
    )
    parser.add_argument(
        "--category-col",
        default="category",
        help='Category column name. Default: "category".',
    )
    parser.add_argument("--output", default=None, help="Optional output CSV path.")
    parser.add_argument("--batch-size", type=int, default=256, help="vLLM batch size.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate short_desc/category even when existing values are present.",
    )
    parser.add_argument(
        "--model",
        default=MODEL_PATH,
        help=(
            "Model path or Hugging Face repo id. "
            "Can also be set with SHORT_DESC_MODEL."
        ),
    )
    args = parser.parse_args()

    _ensure_model(args.model)
    process_csv(
        args.csv,
        desc_col=args.desc_col,
        short_desc_col=args.short_desc_col,
        category_col=args.category_col,
        output_path=args.output,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()