#!/bin/bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-$(pwd)/checkpoints/gala-0.6b}"

ASSET_INDEX_PATH="${ASSET_INDEX_PATH:-output/asset_index_imaginarium}"
ASSET_DIR="${ASSET_DIR:-datasets/Imaginarium/imaginarium_assets}"
MAX_RETRIES=1           # 推理层重试次数（指令不符合时重试）
INFER_TEMP=0.0          # 采样温度（>0 才能让重试产生不同结果）
USE_POINTCLOUD=1        # 点云模式开关：1=使用点云输入，0=纯文本推理
INCLUDE_DESC=0          # 物体描述开关：1=指令保留 short_desc，0=匿名化为 object
VISUAL_FEEDBACK=1       # 视觉反馈开关：1=启用 --visual-feedback，0=关闭

# 并发配置
# 每个 job 独占一张可见 GPU；要 5 并发就填 5 张卡，例如：GPU_IDS="2,3,4,5,6"
GPU_IDS="${GPU_IDS:-0}"
RUNS_PER_INPUT="${RUNS_PER_INPUT:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-}"  # 留空时由 run_agent.py batch 默认等于 GPU 数
EXPORT_GLB="${EXPORT_GLB:-1}"     # 1=导出 GLB（默认），0=传 --no-export-glb 关闭
DRY_RUN="${DRY_RUN:-0}"           # 1=只打印展开后的任务和 GPU 分配，不启动模型

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

OUTPUT_DIR="output/agent"
BATCH_SCENES_JSONL=$(mktemp /tmp/scene_agent_scenes.XXXXXX.jsonl)
BATCH_JOBS_JSON=$(mktemp /tmp/scene_agent_batch.XXXXXX.json)
trap 'rm -f "$BATCH_SCENES_JSONL" "$BATCH_JOBS_JSON"' EXIT

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Scene Agent 批量测试${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

run_scene() {
    local scene_id=$1
    local boundary=$2
    local height=$3
    local input=$4
    
    echo -e "${YELLOW}----------------------------------------${NC}"
    echo -e "${GREEN}[queue] [$scene_id]${NC}"
    echo -e "边界: $boundary"
    echo -e "需求: $input"
    echo -e "${YELLOW}----------------------------------------${NC}"

    python - "$BATCH_SCENES_JSONL" "$scene_id" "$boundary" "$height" "$input" <<'PY'
import json
import sys

jsonl_path, scene_id, boundary, height, input_text = sys.argv[1:]
scene = {
    "scene_id": scene_id,
    "boundary": json.loads(boundary),
    "height": float(height),
    "input": input_text,
}
with open(jsonl_path, "a", encoding="utf-8") as f:
    f.write(json.dumps(scene, ensure_ascii=False) + "\n")
PY
    
    echo ""
    echo ""
}

# ============================================================
# 1. 单人卧室
# ============================================================
run_scene \
    "test_01_single_bedroom" \
    "[[0,0],[5.148,0],[5.148,1.8],[6.449,1.8],[6.449,0],[7.6,0],[7.6,3.6],[0,3.6]]" \
    "2.8" \
    "Design this single bedroom and furnish it with appropriate bedroom items. Include one bed, and place a bedside table next to the head of the bed with a table lamp on top."

# # ============================================================
# # 2. L形客厅
# # ============================================================
# run_scene \
#     "test_02_l_shaped_living_room" \
#     "[[0,0],[11.6,0],[11.6,7.6],[4,7.6],[4,3.6],[0,3.6],[0,2.451],[4,2.451],[4,1.15],[0,1.15]]" \
#     "2.8" \
#     "This is a relatively large L-shaped living room. Design and furnish it with appropriate living room items. Include a multi-seat sofa, a coffee table with food placed on it, a TV stand with a television on top, and other relevant living room objects."

# # ============================================================
# # 3. 客卧一体
# # ============================================================
# run_scene \
#     "test_03_living_bedroom_combo" \
#     "[[0,0],[11.6,0],[11.6,7.6],[0,7.6],[0,6.45],[2,6.45],[2,5.149],[0,5.149]]" \
#     "2.8" \
#     "This is a large combined living room and bedroom space. Design and furnish it with appropriate items for both living and sleeping areas. Include several multi-tier shelving units filled with relevant household objects."

# # ============================================================
# # 4. 住宅走廊
# # ============================================================
# run_scene \
#     "test_04_residential_corridor" \
#     "[[0,0],[1.151,0],[1.151,1.788],[2.452,1.788],[2.452,0],[9.151,0],[9.151,1.788],[10.452,1.788],[10.452,0],[19.8,0],[19.8,3.6],[18.451,3.6],[18.451,1.2],[17.149,1.2],[17.149,3.6],[10.452,3.6],[10.452,1.814],[9.151,1.814],[9.151,3.6],[2.452,3.6],[2.452,1.814],[1.151,1.814],[1.151,3.6],[0,3.6]]" \
#     "2.8" \
#     "This is a corridor inside a residential building. Design it and place appropriate corridor items. Add several coffee tables and several large potted green plants, then place some food and drinks on the coffee tables."

# # ============================================================
# # 5. 仓库
# # ============================================================
# run_scene \
#     "test_05_storage_warehouse" \
#     "[[0,0],[7.6,0],[7.6,1.149],[5.603,1.149],[5.603,2.45],[7.6,2.45],[7.6,7.6],[0,7.6]]" \
#     "3.0" \
#     "This is a small storage warehouse. Place storage cabinets and shelving units filled with stored items, along with other appropriate warehouse-related objects."

# # ============================================================
# # 6. 破旧小巷
# # ============================================================
# run_scene \
#     "test_06_rundown_alley" \
#     "[[0,0],[16,0],[16,3.8],[0,3.8]]" \
#     "3.0" \
#     "This is a rundown alley. Place one large dumpster, several trash bins, several old worn shelving units filled with garbage and clutter, and a large amount of additional trash and miscellaneous debris throughout the alley."

# # ============================================================
# # 7. 天台售货角
# # ============================================================
# run_scene \
#     "test_07_rooftop_vending_corner" \
#     "[[0,0],[7.7,0],[7.7,7.4],[0,7.4]]" \
#     "4.0" \
#     "This is a vending corner on a rooftop. Design and arrange the area with several vending machines. Place a table-and-chair set near the vending machines, and put some food and drinks on the table."

# # ============================================================
# # 8. 天台休息处
# # ============================================================
# run_scene \
#     "test_08_rooftop_relaxation_area" \
#     "[[0.5,0],[11.5,0],[11.5,0.5],[12,0.5],[12,7.7],[0,7.7],[0,0.5],[0.5,0.5]]" \
#     "3.0" \
#     "This is a relaxation area on a rooftop terrace for residents to rest and socialize. Design the space with tables and chairs, place food and drinks on the tables, and add other relevant leisure and rooftop-area objects."

# # ============================================================
# # 9. 停车场维修角
# # ============================================================
# run_scene \
#     "test_09_parking_lot_maintenance_corner" \
#     "[[0,0],[8,0],[8,12],[0,12]]" \
#     "2.8" \
#     "This is a maintenance corner inside a parking lot. Design the area and include a workbench covered with repair tools."

# # ============================================================
# # 10. 室外垃圾角
# # ============================================================
# run_scene \
#     "test_10_outdoor_garbage_corner" \
#     "[[0,0],[4,0],[4,6],[0,6]]" \
#     "4.0" \
#     "This is an outdoor garbage collection corner. Fill the area with a large amount of trash and miscellaneous waste, then add several garbage bags and broken shelving units piled with garbage."

# # ============================================================
# # 2. 中等卧室 (Medium) - 床、床头柜x2、衣柜
# # ============================================================
# run_scene \
#     "test_02_medium_bedroom" \
#     "[[0,0],[4,0],[4,3.5],[0,3.5]]" \
#     "2.8" \
#     "A modern bedroom with a double bed against the wall, two nightstands on each side of the bed, and a wardrobe at the corner."

# # ============================================================
# # 3. 复杂主卧 (Hard) - 床、床头柜、衣柜、梳妆台、单椅
# # ============================================================
# run_scene \
#     "test_03_master_bedroom" \
#     "[[0,0],[5,0],[5,4],[0,4]]" \
#     "2.8" \
#     "A master bedroom with a king-size bed, two matching nightstands, a large wardrobe, a dressing table with stool, and a comfortable armchair."

# # ============================================================
# # 4. 简单客厅 (Easy) - 沙发和茶几
# # ============================================================
# run_scene \
#     "test_04_simple_living" \
#     "[[0,0],[4,0],[4,3],[0,3]]" \
#     "2.6" \
#     "A living room with a sofa and a coffee table."

# # ============================================================
# # 5. 中等客厅 (Medium) - 沙发、茶几、电视柜、单椅
# # ============================================================
# run_scene \
#     "test_05_medium_living" \
#     "[[0,0],[5,0],[5,4],[0,4]]" \
#     "2.8" \
#     "A living room with a three-seater sofa facing a TV cabinet, a coffee table in the middle, and two armchairs on the sides."

# ============================================================
# 6. 复杂客餐厅 (Hard) - 开放式客餐厅
# ============================================================
# run_scene \
#     "test_06_living_dining" \
#     "[[0,0],[7,0],[7,5],[0,5]]" \
#     "2.9" \
#     "An open-plan living and dining room. The living area has an L-shaped sofa, coffee table, and TV cabinet. The dining area has a rectangular dining table with six chairs. Add a bookshelf against the wall."

# # ============================================================
# # 7. 简单餐厅 (Easy) - 餐桌和椅子
# # ============================================================
# run_scene \
#     "test_07_simple_dining" \
#     "[[0,0],[3.5,0],[3.5,3],[0,3]]" \
#     "2.6" \
#     "A dining room with a round dining table and four chairs."

# # ============================================================
# # 8. 中等书房 (Medium) - 书桌、椅子、书架、落地灯
# # ============================================================
# run_scene \
#     "test_08_study_room" \
#     "[[0,0],[3.5,0],[3.5,3],[0,3]]" \
#     "2.6" \
#     "A study room with a desk against the wall, an office chair, a tall bookshelf, and an armchair with a floor lamp for reading."

# # ============================================================
# # 9. L形客厅 (Hard) - 不规则房间形状
# # ============================================================
# run_scene \
#     "test_09_l_shaped_living" \
#     "[[0,0],[6,0],[6,3],[3,3],[3,5],[0,5]]" \
#     "2.8" \
#     "An L-shaped living room. Place a sofa set with coffee table in the larger area, and a small reading corner with an armchair and side table in the smaller section. Include a TV cabinet."

# ============================================================
# 10. 紧凑卧室 (Hard) - 床、双床头柜、衣柜、书桌、角落灯具
# ============================================================
# run_scene \
#     "test_10_compact_bedroom_layout" \
#     "[[0,0],[5,0],[5,4],[0,4]]" \
#     "2.8" \
#     "A bed stands against the north wall. A rug lies beneath the bed. There's a bedside table on each side of the bed, with a lamp and alarm clock on each table. A wardrobe stands against the west wall, and there is a box and some clothes within the wardrobe. A desk stands against the east wall, with a chair underneath. A computer, keyboard, and mouse are on the desk. A trash can sits to the left of the desk and next to the desk. A potted plant is to the right of the desk. A floor lamp occupies the corner."

# ============================================================
# 11. 紧凑书房客厅 (Hard) - 书架、长桌、沙发、单椅、茶几
# ============================================================
# run_scene \
#     "test_11_compact_study_lounge" \
#     "[[0,0],[6,0],[6,4],[0,4]]" \
#     "2.8" \
#     "A large bookshelf stands against the west wall. Two storage boxes, and two photo albums are placed as separate visible items within the bookshelf. Next to the bookshelf is a long table, positioned in front of the bookshelf and also against the wall. A chair sits under the table, on which rests a water glass, a laptop, and a desk lamp. A sofa stands against the east wall, with a rug underneath it, on its left side. A coffee table faces the sofa, holding a fruit stand and a water pitcher. An armchair faces the coffee table, next to which is a large potted plant. A trash can sits to the south of the sofa. A floor lamp occupies a corner of the room."

# ============================================================
# 12. 简洁卧室 (Medium) - 中央床位、衣柜、书桌、角落绿植
# ============================================================
# run_scene \
#     "test_12_minimal_bedroom" \
#     "[[0,0],[4,0],[4,4],[0,4]]" \
#     "2.8" \
#     "A bed stands in the center of the room. A lamp and telephone sit on the bedside table. A suitcase is inside the wardrobe against the west wall. A chair sits under the desk against the east wall. A laptop and mouse are on the desk. A rug covers the floor under the bed. A potted plant sits in the corner of the room."

# ============================================================
# 13. 紧凑客厅 (Medium) - 沙发、电视柜、餐桌、角落绿植
# ============================================================
# run_scene \
#     "test_13_compact_living_room" \
#     "[[0,0],[5,0],[5,4],[0,4]]" \
#     "2.8" \
#     "The sofa is placed against the south wall, with cushions on it. In front of the sofa is a coffee table with a tissue box on it and a rug underneath. Opposite the sofa is a TV cabinet with a television set on it and a router on top. Next to the sofa is a dining table with a chair and a vase on it. A potted plant sits on the floor in the corner."

# ============================================================
# 14. 豪华开放式公寓 (Very Hard) - 约20个物体
# ============================================================
# run_scene \
#     "test_14_luxury_apartment" \
#     "[[0,0],[7,0],[7,4],[4,4],[4,7],[0,7]]" \
#     "3.0" \
#     "A luxurious apartment with multiple functional zones. Living area: a large sofa, two armchairs, a rectangular coffee table, a TV cabinet, and a floor lamp. Dining area: a long dining table with six chairs. Work corner: a desk with an office chair and a tall bookshelf. Lounge corner: a chaise lounge with a side table and a reading lamp. Decor: two potted plants, and a decorative rug under the sofa set."

# ============================================================
# 15. 高物体丰富度多功能套间 (Very Hard) - 高密度家具与小物件组合
# ============================================================
# run_scene \
#     "test_15_high_density_multifunction_suite" \
#     "[[0,0],[5,0],[5,8],[0,8]]" \
#     "3.0" \
#     "A highly detailed multifunction suite with dense object placement. Sleeping area: a queen bed against the north wall, two nightstands on both sides, a table lamp and an alarm clock on each nightstand, a bench at the foot of the bed, and a large rug underneath. Storage area: a wardrobe and a chest of drawers against the west wall. The wardrobe contains storage boxes, folded clothes, and a suitcase. Work area: a long desk against the east wall with an office chair tucked under it, a laptop, monitor, keyboard, mouse, desk lamp, books, and a water glass on the desk, plus a trash can beside it. Lounge area: a sofa, a coffee table, an armchair, and a side table in the south part of the room, with a floor lamp nearby and books plus a decorative vase on the tables. Dining corner: a round dining table with four chairs and a fruit bowl in the remaining open area. Decor: two potted plants in different corners and wall-side shelving filled with books and storage baskets."

# ============================================================
# 完成
# ============================================================
if [[ ! -s "$BATCH_SCENES_JSONL" ]]; then
    echo -e "${YELLOW}[batch] No scenes queued. Uncomment at least one run_scene block.${NC}"
    exit 1
fi

python - "$BATCH_SCENES_JSONL" "$BATCH_JOBS_JSON" <<'PY'
import json
import sys

jsonl_path, jobs_path = sys.argv[1:]
with open(jsonl_path, "r", encoding="utf-8") as f:
    scenes = [json.loads(line) for line in f if line.strip()]

with open(jobs_path, "w", encoding="utf-8") as f:
    json.dump({"scenes": scenes}, f, indent=2, ensure_ascii=False)
PY

batch_args=(
    batch
    -m "$MODEL_PATH"
    --asset-index "$ASSET_INDEX_PATH"
    --asset-dir "$ASSET_DIR"
    --output-dir "$OUTPUT_DIR"
    --max-retries "$MAX_RETRIES"
    --infer-temp "$INFER_TEMP"
    --jobs "$BATCH_JOBS_JSON"
    --gpus "$GPU_IDS"
    --runs-per-input "$RUNS_PER_INPUT"
)

if [[ -n "$MAX_PARALLEL" ]]; then
    batch_args+=(--max-parallel "$MAX_PARALLEL")
fi
if [ "$USE_POINTCLOUD" -ne 1 ]; then
    batch_args+=(--no-pcd)
fi
if [ "$INCLUDE_DESC" -eq 1 ]; then
    batch_args+=(--include-desc)
fi
if [ "$VISUAL_FEEDBACK" -eq 1 ]; then
    batch_args+=(--visual-feedback)
fi
if [ "$EXPORT_GLB" -ne 1 ]; then
    batch_args+=(--no-export-glb)
fi
if [ "$DRY_RUN" -eq 1 ]; then
    batch_args+=(--dry-run)
fi

echo -e "${BLUE}[batch] GPU_IDS=$GPU_IDS RUNS_PER_INPUT=$RUNS_PER_INPUT${NC}"
if [[ -n "$MAX_PARALLEL" ]]; then
    echo -e "${BLUE}[batch] MAX_PARALLEL=$MAX_PARALLEL${NC}"
fi
echo -e "${BLUE}[batch] Jobs config: $BATCH_JOBS_JSON${NC}"

python scripts/run_agent.py "${batch_args[@]}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  批量测试完成！${NC}"
echo -e "${BLUE}  输出目录: output/agent/${NC}"
echo -e "${BLUE}========================================${NC}"
