#!/bin/bash

set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:."

export DATASETS_ROOT="./datasets"

SCENE_ID="${SCENE_ID:-imaginarium_bedroom_14}"

START_STEP="${START_STEP:-0}"
INCLUDE_OBJECT_DESCRIPTIONS="${INCLUDE_OBJECT_DESCRIPTIONS:-false}"
ONLY_POINTCLOUD="${ONLY_POINTCLOUD:-false}"
METADATA_CSV="${METADATA_CSV:-$DATASETS_ROOT/processed/3dish_metadata.csv}"
PCD_MODEL_PATH="${PCD_MODEL_PATH:-$(pwd)/checkpoints/gala-0.6b}"
PCD_OUTPUT_DIR="output/qwen3-0_6b-separate_pcd"
TEMPERATURE="${TEMPERATURE:-0.0}"
SEED="${SEED:-25}"
RUN_TAG="${RUN_TAG:-}"  # 输出 file 后缀, 比如 "baseline"
BLENDER_BIN="${BLENDER_BIN:-blender}"
REPAIR_PHYSICS="${REPAIR_PHYSICS:-true}"  # 推理后是否额外保存几何修正版 _repaired.json
XPBD_PYTHON="${XPBD_PYTHON:-python}"  # 跑 solve_layout 的 python(需含 fcl/vhacdx/trimesh,即 python 环境)

ONLY_POINTCLOUD_ARG=""
if [[ "$ONLY_POINTCLOUD" == "1" || "$ONLY_POINTCLOUD" == "true" ]]; then
  ONLY_POINTCLOUD_ARG="--only-pointcloud"
  PCD_OUTPUT_DIR="output/qwen3-0_6b-only_pcd"
fi

mkdir -p "$PCD_OUTPUT_DIR"

export_glb() {
  local layout_json="$1"
  local export_dir="$2"

  python scripts/export_layout_scene.py \
    --layout "$layout_json" \
    --output-dir "$export_dir"
}

repair_layout() {
  # 几何修正，额外保存 <stem>_repaired.json。两段式:
  #   1) Blender 无界面只做 Bullet bake,产出 <stem>_baked.json;
  #   2) python 侧 solve_layout.py 做几何后处理(VHACD 凸块 2D-SAT + XPBD + 货架换层)。
  local layout_json="$1"
  local repaired_json="${layout_json%.json}_repaired.json"
  local baked_json="${layout_json%.json}_baked.json"

  "$BLENDER_BIN" -b -P scripts/physics_repair/repair_layout.py -- \
    --layout "$layout_json" \
    --output-layout "$baked_json" \
    --dataset auto
  "$XPBD_PYTHON" scripts/physics_repair/solve_layout.py \
    --layout "$baked_json" \
    --output-layout "$repaired_json"
}

OBJECT_DESCRIPTIONS_ARG=""
INPUT_STEM="${SCENE_ID}_inference_input"
if [[ "$INCLUDE_OBJECT_DESCRIPTIONS" == "1" || "$INCLUDE_OBJECT_DESCRIPTIONS" == "true" ]]; then
  OBJECT_DESCRIPTIONS_ARG="--with-object-descriptions"
  INPUT_STEM="${SCENE_ID}_with_desc_inference_input"
fi

TEXT_INPUT_FILE="input/${INPUT_STEM}.json"
if [[ "$START_STEP" -gt 0 ]]; then
  TEXT_INPUT_FILE="input/${INPUT_STEM}_from_step${START_STEP}.json"
fi

SUFFIX="temp_${TEMPERATURE}"
if [[ -n "$RUN_TAG" ]]; then
  SUFFIX="${SUFFIX}_${RUN_TAG}"
fi
# 每个场景一个独立文件夹,最终含 6 个文件:layout/baked/repaired 各自的 .json + .glb
PCD_SCENE_DIR="$PCD_OUTPUT_DIR/${SCENE_ID}_layout_${SUFFIX}"
mkdir -p "$PCD_SCENE_DIR"
PCD_OUTPUT_FILE="$PCD_SCENE_DIR/${SCENE_ID}_layout_${SUFFIX}.json"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" python inference/scene_layout_inference.py \
  --model_path "$PCD_MODEL_PATH" \
  --scene_id "$SCENE_ID" \
  --metadata_csv "$METADATA_CSV" \
  --start_step "$START_STEP" \
  --input_file "$TEXT_INPUT_FILE" \
  --output_file "$PCD_OUTPUT_FILE" \
  --inference_dtype float32 \
  --temperature "$TEMPERATURE" \
  $OBJECT_DESCRIPTIONS_ARG \
  $ONLY_POINTCLOUD_ARG \
  --seed "$SEED"

echo "[pcd->glb] $PCD_OUTPUT_FILE"
export_glb \
  "$PCD_OUTPUT_FILE" \
  "$PCD_SCENE_DIR"

if [[ "$REPAIR_PHYSICS" == "1" || "$REPAIR_PHYSICS" == "true" ]]; then
  echo "[pcd->repair] $PCD_OUTPUT_FILE"
  repair_layout "$PCD_OUTPUT_FILE"

  BAKED_OUTPUT_FILE="${PCD_OUTPUT_FILE%.json}_baked.json"
  echo "[baked->glb] $BAKED_OUTPUT_FILE"
  export_glb \
    "$BAKED_OUTPUT_FILE" \
    "$PCD_SCENE_DIR"

  REPAIRED_OUTPUT_FILE="${PCD_OUTPUT_FILE%.json}_repaired.json"
  echo "[repair->glb] $REPAIRED_OUTPUT_FILE"
  export_glb \
    "$REPAIRED_OUTPUT_FILE" \
    "$PCD_SCENE_DIR"
fi