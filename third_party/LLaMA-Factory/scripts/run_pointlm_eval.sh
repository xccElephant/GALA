#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# PointLM checkpoint 自动评估脚本（薄封装，核心逻辑位于 scripts/pointlm_eval_checkpoints.py）

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_INFO_JSON="${DATASET_INFO_JSON:-data/dataset_info.json}"
DATA_FILE="${DATA_FILE:-../../datasets/processed/3dish_separate_pointcloud_331k_test.json}"
SCORE_DATA_FILE="${SCORE_DATA_FILE:-../../datasets/processed/3dish_separate_pointcloud_331k_test.json}"

EXPERIMENT="${EXPERIMENT:-qwen3-0_6b/3dish_separate_pointcloud_331k_train}"
TRAIN_YAML="${TRAIN_YAML:-examples/train_full/qwen3-0_6b/3dish_separate_pointcloud_331k_train/base.yaml}"
MODEL_DIR="${MODEL_DIR:-saves/train_full_sft/${EXPERIMENT}}"
OUTPUT_DIR="${OUTPUT_DIR:-results/eval/${EXPERIMENT}}"

EVAL_MODE="${EVAL_MODE:-all}"          # all | latest
FORCE="${FORCE:-false}"

NUM_GPUS="${NUM_GPUS:-8}"
NUM_SAMPLES="${NUM_SAMPLES:-}"
START_IDX="${START_IDX:-0}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
INFERENCE_DTYPE="${INFERENCE_DTYPE:-bfloat16}"
DO_SAMPLE="${DO_SAMPLE:-false}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-10}"
NUM_BEAMS="${NUM_BEAMS:-1}"
PCD_RANDOM_NOISE="${PCD_RANDOM_NOISE:-false}"

SCORE_WORKERS="${SCORE_WORKERS:-32}"
VOXEL_SIZE="${VOXEL_SIZE:-0.02}"
OVERLAP_RATIO_THRESHOLD="${OVERLAP_RATIO_THRESHOLD:-0.20}"
HEIGHT_TOLERANCE="${HEIGHT_TOLERANCE:-0.05}"

UPLOAD_WANDB="${UPLOAD_WANDB:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-llamafactory}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-eval_${EXPERIMENT}}"
WANDB_JOB_TYPE="${WANDB_JOB_TYPE:-eval}"

cmd=(
  "${PYTHON_BIN}" scripts/pointlm_eval_checkpoints.py
  --model_dir "${MODEL_DIR}"
  --train_yaml "${TRAIN_YAML}"
  --output_dir "${OUTPUT_DIR}"
  --dataset_info_json "${DATASET_INFO_JSON}"
  --eval_mode "${EVAL_MODE}"
  --num_gpus "${NUM_GPUS}"
  --start_idx "${START_IDX}"
  --batch_size "${BATCH_SIZE}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --inference_dtype "${INFERENCE_DTYPE}"
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P}"
  --top_k "${TOP_K}"
  --num_beams "${NUM_BEAMS}"
  --score_workers "${SCORE_WORKERS}"
  --voxel_size "${VOXEL_SIZE}"
  --overlap_ratio_threshold "${OVERLAP_RATIO_THRESHOLD}"
  --height_tolerance "${HEIGHT_TOLERANCE}"
  --wandb_project "${WANDB_PROJECT}"
  --wandb_run_name "${WANDB_RUN_NAME}"
  --wandb_job_type "${WANDB_JOB_TYPE}"
)

[[ -n "${DATA_FILE}" ]] && cmd+=(--data_file "${DATA_FILE}")
[[ -n "${SCORE_DATA_FILE}" ]] && cmd+=(--score_data_file "${SCORE_DATA_FILE}")
[[ -n "${NUM_SAMPLES}" ]] && cmd+=(--num_samples "${NUM_SAMPLES}")
[[ "${FORCE}" == "true" ]] && cmd+=(--force)
[[ "${DO_SAMPLE}" == "true" ]] && cmd+=(--do_sample)
[[ "${PCD_RANDOM_NOISE}" == "true" ]] && cmd+=(--pcd_random_noise)
[[ "${UPLOAD_WANDB}" == "true" ]] && cmd+=(--upload_wandb)
[[ -n "${WANDB_ENTITY}" ]] && cmd+=(--wandb_entity "${WANDB_ENTITY}")

"${cmd[@]}"
