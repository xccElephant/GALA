#!/bin/bash
#
# 数据处理全流程脚本
#
# 按顺序执行 dataset_pipeline/preprocessing 和 dataset_pipeline/generation 中的所有步骤。
#
# 使用方法:
#   bash scripts/run_data_pipeline.sh --tag ""
#
# 选项:
#   --step <N>          从第 N 步开始执行 (默认从第 1 步开始)
#   --only <N>          只执行第 N 步
#   --dataset <name>    生成阶段的数据集类型: 3dfront, imaginarium, spatiallm, 3dis, handcrafted, 3dish (默认: 3dish)
#                       3dis  = 3D-FRONT + Imaginarium + SpatialLM 三数据集融合
#                       3dish = 3D-FRONT + Imaginarium + SpatialLM + handcrafted 四数据集融合
#   --tag <str>         输出文件名后缀，用于区分不同版本 (默认: 当天日期，如 _20260301)
#                       示例: --tag _with_space → 3dish_with_space.json
#                             --tag "" → 3dish.json (不加后缀，会覆盖旧数据)
#   --num-workers <N>   并行工作进程数 (默认: CPU 核心数的一半，至少 1)
#   --val-interval <N>  兼容旧划分脚本的验证集间隔参数 (默认: 10)
#                       当前默认按 scene family 总数计算 train/val/test = 9:0.5:0.5
#   --group-by-scene-family
#                       按场景家族划分（避免旋转增强样本跨 train/test，默认: 开启）
#   --with-internal-spaces
#                       在生成阶段写入 internal_spaces 字段（默认: 关闭）
#   --with-object-descriptions, --include-object-descriptions
#                       在生成阶段写入物体 short_desc（默认: 使用 category，不写入 short_desc）；
#                       给产物名追加后缀 _with_desc
#   --disable-collision 关闭 converted_scenes 的 Blender BVH 碰撞过滤（默认: 开启）
#   --disable-floating  关闭 converted_scenes 的 Blender 悬浮/穿地过滤（默认: 开启）
#   --floating-workers <N>
#                       3dfront/spatiallm 悬浮/穿地过滤并行 Blender 进程数
#                       (默认: 4，可用环境变量 FLOATING_WORKERS 覆盖)
#   --floating-asset-cache-size <N>
#                       每个 Blender 进程缓存的资产 mesh 数量
#                       (默认: 512，可用环境变量 FLOATING_ASSET_CACHE_SIZE 覆盖；0=关闭)
#   --collision-min-overlap-triangles <N>
#                       Blender BVH 碰撞过滤的最少相交三角面数量
#                       (默认: 100，可用环境变量 COLLISION_MIN_OVERLAP_TRIANGLES 覆盖)
#   --force-floating    清空悬浮/穿地过滤 checkpoint 并重新计算
#   --dry-run           只打印命令，不实际执行
#   -h, --help          显示帮助信息
#
# 步骤列表:
#   === 预处理阶段 (dataset_pipeline/preprocessing) ===
#   1  - 预处理资产点云 (3D-FUTURE-model)
#   2  - 预处理资产点云 (Imaginarium assets)
#   3  - 转换 Imaginarium 数据集
#   4  - 转换 ReSpace/3DFront 数据集
#   5  - 场景旋转增强 (Imaginarium + handcrafted)
#   6  - 生成 filtered_scenes (3dfront/spatiallm 过滤；Imaginarium/handcrafted 复制)
#   7  - 生成自回归数据 (3DFront)
#   8  - 生成自回归数据 (Imaginarium)
#   9  - 生成自回归数据 (SpatialLM)
#   10 - 生成自回归数据 (handcrafted)
#   11 - 合并四数据集 (→ 3dish)
#   === 生成阶段 (dataset_pipeline/generation) ===
#   12 - 生成纯文本训练数据
#   13 - 生成分离点云训练数据 (floor/existing/new object)
#   14 - 生成对齐训练数据 (Stage-1 point_proj 预训练)
#   === 划分阶段 (dataset_pipeline/tools, 默认按 scene family 总数 9:0.5:0.5 计算) ===
#   15 - 划分纯文本数据集 (train/test/val)
#   16 - 划分分离点云数据集 (train/test/val)
#   17 - 划分对齐数据集 (train/test/val)

set -euo pipefail

# 确保通过 "python path/to/script.py" 执行时也能导入项目根目录下的包（如 config）
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# ============================================================
# 默认配置
# ============================================================
START_STEP=1
ONLY_STEP=""
DATASET="3dish"
TAG="__auto__"   # __auto__ 表示使用当天日期自动生成
NUM_WORKERS=""
ASSET_NUM_POINTS=8192
VAL_INTERVAL=10
ALIGNMENT_VAL_INTERVAL=10
GROUP_BY_SCENE_FAMILY=true
WITH_INTERNAL_SPACES=false
INCLUDE_OBJECT_DESCRIPTIONS=false
ONLY_POINTCLOUD=false
DISABLE_COLLISION=false
DISABLE_FLOATING=false
FLOATING_WORKERS="${FLOATING_WORKERS:-4}"
FLOATING_ASSET_CACHE_SIZE="${FLOATING_ASSET_CACHE_SIZE:-512}"
COLLISION_MIN_OVERLAP_TRIANGLES="${COLLISION_MIN_OVERLAP_TRIANGLES:-100}"
FORCE_FLOATING=false
DRY_RUN=false

# ============================================================
# 参数解析
# ============================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --step)
            START_STEP="$2"; shift 2 ;;
        --only)
            ONLY_STEP="$2"; shift 2 ;;
        --dataset)
            DATASET="$2"; shift 2 ;;
        --tag)
            TAG="$2"; shift 2 ;;
        --num-workers)
            NUM_WORKERS="$2"; shift 2 ;;
        --val-interval)
            VAL_INTERVAL="$2"; shift 2 ;;
        --group-by-scene-family)
            GROUP_BY_SCENE_FAMILY=true; shift ;;
        --with-internal-spaces)
            WITH_INTERNAL_SPACES=true; shift ;;
        --with-object-descriptions|--include-object-descriptions)
            INCLUDE_OBJECT_DESCRIPTIONS=true; shift ;;
        --without-object-descriptions)
            INCLUDE_OBJECT_DESCRIPTIONS=false; shift ;;
        --only-pointcloud)
            ONLY_POINTCLOUD=true; shift ;;
        --disable-collision)
            DISABLE_COLLISION=true; shift ;;
        --disable-floating)
            DISABLE_FLOATING=true; shift ;;
        --floating-workers)
            FLOATING_WORKERS="$2"; shift 2 ;;
        --floating-asset-cache-size)
            FLOATING_ASSET_CACHE_SIZE="$2"; shift 2 ;;
        --collision-min-overlap-triangles)
            COLLISION_MIN_OVERLAP_TRIANGLES="$2"; shift 2 ;;
        --force-floating)
            FORCE_FLOATING=true; shift ;;
        --dry-run)
            DRY_RUN=true; shift ;;
        -h|--help)
            # 打印脚本头部注释作为帮助信息
            sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "错误: 未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1 ;;
    esac
done

case "$DATASET" in
    3dfront|imaginarium|spatiallm|3dis|handcrafted|3dish) ;;
    *)
        echo "错误: 不支持的数据集: $DATASET"
        echo "支持的数据集: 3dfront, imaginarium, spatiallm, 3dis, handcrafted, 3dish"
        exit 1
        ;;
esac

AUTO_NUM_WORKERS="$(python3 -c 'import os; print(max(1, (os.cpu_count() or 1) // 2))')"
EFFECTIVE_NUM_WORKERS="${NUM_WORKERS:-$AUTO_NUM_WORKERS}"
WORKER_ARG="--num-workers $EFFECTIVE_NUM_WORKERS"
FILTER_COLLISION_WORKERS="$EFFECTIVE_NUM_WORKERS"
if (( FILTER_COLLISION_WORKERS > 32 )); then
    FILTER_COLLISION_WORKERS=192
fi

BLENDER_BIN="${BLENDER_BIN:-blender}"
FLOATING_FILTER_ARG=""
if [[ "$DISABLE_COLLISION" == true ]]; then
    FLOATING_FILTER_ARG="$FLOATING_FILTER_ARG --disable-collision"
fi
if [[ "$DISABLE_FLOATING" == true ]]; then
    FLOATING_FILTER_ARG="$FLOATING_FILTER_ARG --disable-floating"
fi
if [[ "$FORCE_FLOATING" == true ]]; then
    FLOATING_FILTER_ARG="$FLOATING_FILTER_ARG --force-floating"
fi

INTERNAL_SPACES_ARG=""
if [[ "$WITH_INTERNAL_SPACES" == true ]]; then
    INTERNAL_SPACES_ARG="--with-internal-spaces"
fi

OBJECT_DESCRIPTIONS_ARG=""
OBJECT_DESCRIPTIONS_SUFFIX=""
if [[ "$INCLUDE_OBJECT_DESCRIPTIONS" == true ]]; then
    OBJECT_DESCRIPTIONS_ARG="--with-object-descriptions"
    OBJECT_DESCRIPTIONS_SUFFIX="_with_desc"
fi

ONLY_POINTCLOUD_ARG=""
JSON_SUFFIX="_separate_pointcloud.json"
if [[ "$ONLY_POINTCLOUD" == true ]]; then
    ONLY_POINTCLOUD_ARG="--only-pointcloud"
    JSON_SUFFIX="_only_pointcloud.json"
fi

# TAG 处理：__auto__ 时使用当天日期
if [[ "$TAG" == "__auto__" ]]; then
    TAG="_$(date +%Y%m%d)"
fi

# ============================================================
# 检测数据集根目录 (与 config/paths.py 保持一致)
# ============================================================
if [[ -z "${DATASETS_ROOT:-}" ]]; then
    DATASETS_ROOT="$PROJECT_ROOT/datasets"
fi

# 各数据集路径
SSR_3DFRONT_DIR="$DATASETS_ROOT/SSR-3DFRONT"
SSR_3DFRONT_FILTERED_SCENES_DIR="$SSR_3DFRONT_DIR/filtered_scenes"
IMAGINARIUM_DIR="$DATASETS_ROOT/Imaginarium"
IMAGINARIUM_FILTERED_SCENES_DIR="$IMAGINARIUM_DIR/filtered_scenes"
IMAGINARIUM_ASSETS_DIR="$IMAGINARIUM_DIR/imaginarium_assets"
IMAGINARIUM_CONVERTED_SCENES_DIR="$IMAGINARIUM_DIR/converted_scenes"
THREED_FUTURE_MODEL_DIR="$DATASETS_ROOT/3D-FRONT/3D-FUTURE-model"
SPATIALLM_DIR="$DATASETS_ROOT/spatiallm"
SPATIALLM_CONVERTED_SCENES_DIR="$SPATIALLM_DIR/converted_scenes"
SPATIALLM_FILTERED_SCENES_DIR="$SPATIALLM_DIR/filtered_scenes"
SPATIALLM_AUTOREGRESSIVE_DIR="$SPATIALLM_DIR/autoregressive_data"
THREEDIS_AUTOREGRESSIVE_DIR="$DATASETS_ROOT/3dis/autoregressive"
HANDCRAFTED_DIR="$DATASETS_ROOT/handcrafted"
HANDCRAFTED_CONVERTED_SCENES_DIR="$HANDCRAFTED_DIR/converted_scenes"
HANDCRAFTED_FILTERED_SCENES_DIR="$HANDCRAFTED_DIR/filtered_scenes"
HANDCRAFTED_AUTOREGRESSIVE_DIR="$HANDCRAFTED_DIR/autoregressive_data"
THREEDISH_AUTOREGRESSIVE_DIR="$DATASETS_ROOT/3dish/autoregressive"

MERGED_DATASET_NAME="3dish"
MERGED_AUTOREGRESSIVE_DIR="$THREEDISH_AUTOREGRESSIVE_DIR"
MERGE_AUTOREGRESSIVE_SOURCES="3dfront:$SSR_3DFRONT_DIR/autoregressive_data imaginarium:$IMAGINARIUM_DIR/autoregressive_data spatiallm:$SPATIALLM_AUTOREGRESSIVE_DIR handcrafted:$HANDCRAFTED_AUTOREGRESSIVE_DIR"

if [[ "$DATASET" == "3dis" ]]; then
    MERGED_DATASET_NAME="3dis"
    MERGED_AUTOREGRESSIVE_DIR="$THREEDIS_AUTOREGRESSIVE_DIR"
    MERGE_AUTOREGRESSIVE_SOURCES="3dfront:$SSR_3DFRONT_DIR/autoregressive_data imaginarium:$IMAGINARIUM_DIR/autoregressive_data spatiallm:$SPATIALLM_AUTOREGRESSIVE_DIR"
fi

# 资产信息 CSV（用于生成 category 映射）
THREED_FRONT_ASSET_INFO_CSV="$THREED_FUTURE_MODEL_DIR/3dfront_asset_info.csv"
IMAGINARIUM_ASSET_INFO_CSV="$IMAGINARIUM_ASSETS_DIR/imaginarium_asset_info.csv"

# 生成产物目录（与 config/paths.py DATA_PROCESSED_DIR 保持一致）
DATA_PROCESSED_DIR="$DATASETS_ROOT/processed"

# ============================================================
# 辅助函数
# ============================================================
STEP_COUNT=0
FAIL_COUNT=0

# 划分阶段专用辅助函数: 调用 04_divide_dataset.py（默认按 90:5:5 自动计算 train/val/test 大小）
# 用法: run_divide_step <step_num> <desc> <suffix> [val_interval] [group_arg]
run_divide_step() {
    local step_num=$1
    local step_desc=$2
    local suffix=$3
    local interval=${4:-$VAL_INTERVAL}
    local group_arg=${5-$GROUP_BY_SCENE_FAMILY_ARG}   # 未传则用全局默认

    should_run "$step_num" || return 0

    # shellcheck disable=SC2086
    run_step "$step_num" "$step_desc" \
        python dataset_pipeline/generation/04_divide_dataset.py \
            --input "${DATASET}${TAG}${suffix}.json" \
            --val-interval "$interval" \
            $group_arg
}

should_run() {
    local step_num=$1
    if [[ -n "$ONLY_STEP" && "$step_num" != "$ONLY_STEP" ]]; then
        return 1
    fi
    if [[ "$step_num" -lt "$START_STEP" ]]; then
        return 1
    fi
    return 0
}

run_step() {
    local step_num=$1
    local step_desc=$2
    shift 2

    should_run "$step_num" || return 0

    echo ""
    echo "=========================================="
    echo "  步骤 $step_num: $step_desc"
    echo "=========================================="
    echo "  命令: $*"
    echo "------------------------------------------"

    if $DRY_RUN; then
        echo "  [DRY-RUN] 跳过执行"
        return 0
    fi

    STEP_COUNT=$((STEP_COUNT + 1))
    local start_time
    start_time=$(date +%s)

    if "$@"; then
        local end_time
        end_time=$(date +%s)
        local elapsed=$((end_time - start_time))
        echo ""
        echo "  [OK] 步骤 $step_num 完成 (耗时 ${elapsed}s)"
    else
        local exit_code=$?
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo ""
        echo "  [FAIL] 步骤 $step_num 失败 (exit code: $exit_code)"
        echo "  如需跳过此步骤继续执行，请使用: --step $((step_num + 1))"
        exit "$exit_code"
    fi
}

# ============================================================
# 打印配置信息
# ============================================================
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║       数据处理全流程                    ║"
echo "╚══════════════════════════════════════════════════╝"
echo "  数据集根目录:  $DATASETS_ROOT"
echo "  生成数据集:    $DATASET"
echo "  输出文件后缀:  ${TAG:-(无，直接覆盖)}"
echo "  并行工作进程:  $EFFECTIVE_NUM_WORKERS$([[ -n "$NUM_WORKERS" ]] && echo " (手动指定)" || echo " (自动=CPU 一半)")"
echo "  碰撞检测进程:  $FILTER_COLLISION_WORKERS"
echo "  资产点云点数:  $ASSET_NUM_POINTS (固定)"
echo "  划分大小:      自动 ($([[ "$GROUP_BY_SCENE_FAMILY" == true ]] && echo "按 scene family 总数" || echo "按样本数万位取整"), train:val:test = 9:0.5:0.5)"
echo "  场景家族划分:  $GROUP_BY_SCENE_FAMILY"
echo "  internal_spaces: $WITH_INTERNAL_SPACES"
echo "  物体描述:      $INCLUDE_OBJECT_DESCRIPTIONS"
echo "  悬浮过滤:      $([[ "$DISABLE_FLOATING" == true ]] && echo "关闭" || echo "开启")"
echo "  悬浮过滤进程:  $FLOATING_WORKERS"
echo "  悬浮资产缓存:  $FLOATING_ASSET_CACHE_SIZE / Blender 进程"
echo "  碰撞面片阈值:  $COLLISION_MIN_OVERLAP_TRIANGLES"
echo "  Blender:       $BLENDER_BIN"
echo "  起始步骤:      ${ONLY_STEP:-$START_STEP}"
if $DRY_RUN; then
    echo "  模式:          DRY-RUN (仅打印命令)"
fi

# ============================================================
# 预处理阶段 (dataset_pipeline/preprocessing)
# ============================================================

# --- Step 1: 预处理资产点云 (3D-FUTURE-model) ---
# shellcheck disable=SC2086
run_step 1 "预处理资产点云 (3D-FUTURE-model)" \
    python dataset_pipeline/preprocessing/00_preprocess_asset_pointclouds.py \
        --dir "$THREED_FUTURE_MODEL_DIR" \
        --num-points "$ASSET_NUM_POINTS" \
        --overwrite \
        $WORKER_ARG

# --- Step 2: 预处理资产点云 (Imaginarium assets) ---
# shellcheck disable=SC2086
run_step 2 "预处理资产点云 (Imaginarium assets)" \
    python dataset_pipeline/preprocessing/00_preprocess_asset_pointclouds.py \
        --dir "$IMAGINARIUM_ASSETS_DIR" \
        --num-points "$ASSET_NUM_POINTS" \
        --overwrite \
        $WORKER_ARG

# --- Step 3: 转换 Imaginarium 数据集 ---
run_step 3 "转换 Imaginarium 数据集" \
    python dataset_pipeline/preprocessing/01_convert_imaginarium_dataset.py

# --- Step 4: 转换 ReSpace/3DFront 数据集 ---
run_step 4 "转换 ReSpace/3DFront 数据集" \
    python dataset_pipeline/preprocessing/01_convert_respace_dataset.py

# --- Step 5: 场景旋转增强 (Imaginarium + handcrafted) ---
run_step 5 "场景旋转增强 (Imaginarium + handcrafted)" \
    bash -c "python dataset_pipeline/preprocessing/02_augment_scene_rotation.py --angles 90 180 270 && \
      python dataset_pipeline/preprocessing/02_augment_scene_rotation.py --input-dir \"$HANDCRAFTED_CONVERTED_SCENES_DIR\" --output-dir \"$HANDCRAFTED_CONVERTED_SCENES_DIR\" --angles 90 180 270"

# --- Step 6: 生成 filtered_scenes (3dfront/spatiallm 过滤；Imaginarium/handcrafted 复制) ---
# shellcheck disable=SC2086
run_step 6 "生成 filtered_scenes (3dfront/spatiallm 过滤；Imaginarium/handcrafted 复制)" \
    bash -c "python dataset_pipeline/preprocessing/02_filter_converted_scenes.py --dataset 3dfront --collision-workers \"$FILTER_COLLISION_WORKERS\" --min-object-count 4 --blender-bin \"$BLENDER_BIN\" --floating-workers \"$FLOATING_WORKERS\" --floating-asset-cache-size \"$FLOATING_ASSET_CACHE_SIZE\" --collision-min-overlap-triangles \"$COLLISION_MIN_OVERLAP_TRIANGLES\" $FLOATING_FILTER_ARG && \
      python dataset_pipeline/preprocessing/02_filter_converted_scenes.py --dataset imaginarium && \
      python dataset_pipeline/preprocessing/02_filter_converted_scenes.py --dataset spatiallm --collision-workers \"$FILTER_COLLISION_WORKERS\" --min-object-count 4 --blender-bin \"$BLENDER_BIN\" --floating-workers \"$FLOATING_WORKERS\" --floating-asset-cache-size \"$FLOATING_ASSET_CACHE_SIZE\" --collision-min-overlap-triangles \"$COLLISION_MIN_OVERLAP_TRIANGLES\" $FLOATING_FILTER_ARG && \
      python dataset_pipeline/preprocessing/02_filter_converted_scenes.py --dataset handcrafted"

# --- Step 7: 生成自回归数据 (3DFront) ---
# shellcheck disable=SC2086
run_step 7 "生成自回归数据 (3DFront)" \
    python dataset_pipeline/preprocessing/03_generate_autoregressive_data.py \
        --input-dir "$SSR_3DFRONT_FILTERED_SCENES_DIR" \
        --output-dir "$SSR_3DFRONT_DIR/autoregressive_data" \
        --skip-within \
        --asset-csv "$THREED_FRONT_ASSET_INFO_CSV" \
        $WORKER_ARG

# --- Step 8: 生成自回归数据 (Imaginarium) ---
# shellcheck disable=SC2086
run_step 8 "生成自回归数据 (Imaginarium)" \
    python dataset_pipeline/preprocessing/03_generate_autoregressive_data.py \
        --input-dir "$IMAGINARIUM_FILTERED_SCENES_DIR" \
        --output-dir "$IMAGINARIUM_DIR/autoregressive_data" \
        --asset-csv "$IMAGINARIUM_ASSET_INFO_CSV" \
        $WORKER_ARG

# --- Step 9: 生成自回归数据 (SpatialLM) ---
# shellcheck disable=SC2086
run_step 9 "生成自回归数据 (SpatialLM)" \
    python dataset_pipeline/preprocessing/03_generate_autoregressive_data.py \
        --input-dir "$SPATIALLM_FILTERED_SCENES_DIR" \
        --output-dir "$SPATIALLM_AUTOREGRESSIVE_DIR" \
        --asset-csv "$IMAGINARIUM_ASSET_INFO_CSV" \
        $WORKER_ARG

# --- Step 10: 生成自回归数据 (handcrafted，资产来自 Imaginarium + 3D-FRONT) ---
# shellcheck disable=SC2086
run_step 10 "生成自回归数据 (handcrafted)" \
    python dataset_pipeline/preprocessing/03_generate_autoregressive_data.py \
        --input-dir "$HANDCRAFTED_FILTERED_SCENES_DIR" \
        --output-dir "$HANDCRAFTED_AUTOREGRESSIVE_DIR" \
        --asset-csv "$IMAGINARIUM_ASSET_INFO_CSV" "$THREED_FRONT_ASSET_INFO_CSV" \
        $WORKER_ARG

# --- Step 11: 合并数据集 ---
# shellcheck disable=SC2086
run_step 11 "合并数据集 (→ $MERGED_DATASET_NAME)" \
    python dataset_pipeline/preprocessing/04_merge_datasets.py \
        --output-dir "$MERGED_AUTOREGRESSIVE_DIR" \
        --sources $MERGE_AUTOREGRESSIVE_SOURCES

# ============================================================
# 生成阶段 (dataset_pipeline/generation)
# ============================================================

# --- Step 12: 生成纯文本训练数据 ---
# shellcheck disable=SC2086
run_step 12 "生成纯文本训练数据" \
    python dataset_pipeline/generation/01_generate_training_data.py \
        --dataset "$DATASET" \
        $INTERNAL_SPACES_ARG \
        $OBJECT_DESCRIPTIONS_ARG \
        --output-json-name "${DATASET}${TAG}${OBJECT_DESCRIPTIONS_SUFFIX}.json" \
        --output-csv-name "${DATASET}${TAG}${OBJECT_DESCRIPTIONS_SUFFIX}_metadata.csv"

# --- Step 13: 生成分离点云训练数据 (floor/existing/new object) ---
# shellcheck disable=SC2086
run_step 13 "生成分离点云训练数据 (floor/existing/new object)" \
    python dataset_pipeline/generation/02_generate_pointcloud_training_data.py \
        --dataset "$DATASET" \
        $INTERNAL_SPACES_ARG \
        $OBJECT_DESCRIPTIONS_ARG \
        $ONLY_POINTCLOUD_ARG \
        --output-json-name "${DATASET}${TAG}${OBJECT_DESCRIPTIONS_SUFFIX}${JSON_SUFFIX}" \
        $WORKER_ARG

# --- Step 14: 生成对齐训练数据 (Stage-1 point_proj 预训练) ---
# shellcheck disable=SC2086
run_step 14 "生成对齐训练数据 (Stage-1 point_proj 预训练)" \
    python dataset_pipeline/generation/03_generate_alignment_data.py \
        --dataset "$DATASET" \
        --output-json-name "${DATASET}${TAG}_alignment.json" \
        $WORKER_ARG

# ============================================================
# 划分阶段 (dataset_pipeline/generation/04_divide_dataset.py)
# 划分大小动态计算: 默认按 scene family 总数, 9:0.5:0.5 分为 train/val/test
# ============================================================

GROUP_BY_SCENE_FAMILY_ARG=""
if [[ "$GROUP_BY_SCENE_FAMILY" == true ]]; then
    GROUP_BY_SCENE_FAMILY_ARG="--group-by-scene-family"
fi

# Steps 15-17: 各类型数据集划分 (带场景家族分组)
run_divide_step 15 "划分纯文本数据集 (train/test/val)" "${OBJECT_DESCRIPTIONS_SUFFIX}"
run_divide_step 16 "划分分离点云数据集 (train/test/val)" "${OBJECT_DESCRIPTIONS_SUFFIX}${JSON_SUFFIX%.json}"
run_divide_step 17 "划分对齐数据集 (train/test/val)" "_alignment" "$ALIGNMENT_VAL_INTERVAL"

# ============================================================
# 完成
# ============================================================
echo ""
echo "=========================================="
if $DRY_RUN; then
    echo "  全部步骤扫描完成 (dry-run 模式)"
else
    echo "  全部步骤执行完成!"
    echo "  成功: $STEP_COUNT 步, 失败: $FAIL_COUNT 步"
fi
echo "=========================================="
