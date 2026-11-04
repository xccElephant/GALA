#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Two-stage GALA training + evaluation pipeline (supports optional ablations via RUN_ABLATIONS=true).
# Usage:
#   conda activate gala
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/run_train_sequence.sh

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-8}"
RUN_ABLATIONS="${RUN_ABLATIONS:-false}"
export CUDA_VISIBLE_DEVICES

mkdir -p logs/eval

echo "Using GPUs: ${CUDA_VISIBLE_DEVICES} (NUM_GPUS=${NUM_GPUS})"

echo "========== Stage 1: Projector Alignment (3dish_alignment_500k_train) =========="
mkdir -p saves/train_full_sft/qwen3-0_6b/3dish_alignment_500k_train
llamafactory-cli train examples/train_full/qwen3-0_6b/3dish_alignment_500k_train/base.yaml \
  2>&1 | tee saves/train_full_sft/qwen3-0_6b/3dish_alignment_500k_train/train.log

echo "========== Stage 2: Full GALA Model (3dish_separate_pointcloud_331k_train) =========="
mkdir -p saves/train_full_sft/qwen3-0_6b/3dish_separate_pointcloud_331k_train
llamafactory-cli train examples/train_full/qwen3-0_6b/3dish_separate_pointcloud_331k_train/base.yaml \
  2>&1 | tee saves/train_full_sft/qwen3-0_6b/3dish_separate_pointcloud_331k_train/train.log

echo "========== Eval: Full GALA Model =========="
NUM_GPUS="${NUM_GPUS}" EVAL_MODE=all \
EXPERIMENT=qwen3-0_6b/3dish_separate_pointcloud_331k_train \
TRAIN_YAML=examples/train_full/qwen3-0_6b/3dish_separate_pointcloud_331k_train/base.yaml \
DATA_FILE=../../datasets/processed/3dish_separate_pointcloud_331k_test.json \
SCORE_DATA_FILE=../../datasets/processed/3dish_separate_pointcloud_331k_test.json \
bash scripts/run_pointlm_eval.sh 2>&1 | tee logs/eval/3dish_separate_pointcloud_331k_test.log

if [[ "${RUN_ABLATIONS}" == "true" ]]; then
  echo "========== Ablation 1: w/o point cloud (3dish_331k_train) =========="
  mkdir -p saves/train_full_sft/qwen3-0_6b/3dish_331k_train
  llamafactory-cli train examples/train_full/qwen3-0_6b/3dish_331k_train/base.yaml \
    2>&1 | tee saves/train_full_sft/qwen3-0_6b/3dish_331k_train/train.log

  NUM_GPUS="${NUM_GPUS}" EVAL_MODE=all \
  EXPERIMENT=qwen3-0_6b/3dish_331k_train \
  TRAIN_YAML=examples/train_full/qwen3-0_6b/3dish_331k_train/base.yaml \
  DATA_FILE=../../datasets/processed/3dish_331k_test.json \
  SCORE_DATA_FILE=../../datasets/processed/3dish_separate_pointcloud_331k_test.json \
  bash scripts/run_pointlm_eval.sh 2>&1 | tee logs/eval/3dish_331k_test.log

  echo "========== Ablation 2: w/o pretraining (without_pretrain.yaml) =========="
  mkdir -p saves/train_full_sft/qwen3-0_6b/3dish_separate_pointcloud_331k_train_without_pretrain
  llamafactory-cli train examples/train_full/qwen3-0_6b/3dish_separate_pointcloud_331k_train/without_pretrain.yaml \
    2>&1 | tee saves/train_full_sft/qwen3-0_6b/3dish_separate_pointcloud_331k_train_without_pretrain/train.log

  NUM_GPUS="${NUM_GPUS}" EVAL_MODE=all \
  EXPERIMENT=qwen3-0_6b/3dish_separate_pointcloud_331k_train_without_pretrain \
  TRAIN_YAML=examples/train_full/qwen3-0_6b/3dish_separate_pointcloud_331k_train/without_pretrain.yaml \
  DATA_FILE=../../datasets/processed/3dish_separate_pointcloud_331k_test.json \
  SCORE_DATA_FILE=../../datasets/processed/3dish_separate_pointcloud_331k_test.json \
  bash scripts/run_pointlm_eval.sh 2>&1 | tee logs/eval/3dish_separate_pointcloud_331k_without_pretrain_test.log

  echo "========== Ablation 3: w/o artist scenes (3dish_no_imaginarium_handcrafted) =========="
  mkdir -p saves/train_full_sft/qwen3-0_6b/3dish_no_imaginarium_handcrafted_separate_pointcloud_train
  llamafactory-cli train examples/train_full/qwen3-0_6b/3dish_no_imaginarium_handcrafted_separate_pointcloud_train/base.yaml \
    2>&1 | tee saves/train_full_sft/qwen3-0_6b/3dish_no_imaginarium_handcrafted_separate_pointcloud_train/train.log

  NUM_GPUS="${NUM_GPUS}" EVAL_MODE=all \
  EXPERIMENT=qwen3-0_6b/3dish_no_imaginarium_handcrafted_separate_pointcloud_train \
  TRAIN_YAML=examples/train_full/qwen3-0_6b/3dish_no_imaginarium_handcrafted_separate_pointcloud_train/base.yaml \
  DATA_FILE=../../datasets/processed/3dish_separate_pointcloud_331k_test.json \
  SCORE_DATA_FILE=../../datasets/processed/3dish_separate_pointcloud_331k_test.json \
  bash scripts/run_pointlm_eval.sh 2>&1 | tee logs/eval/3dish_no_imaginarium_handcrafted_test.log
fi
