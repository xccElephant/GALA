import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from multiprocessing import Pool
from functools import partial

from tqdm import tqdm
import argparse
# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.paths import ProjectPaths
from config.constants import (
    MAX_REF_OBJECTS,
    MAX_SCENE_HEIGHT_METERS,
    MAX_SIBLING_REFS,
    RELATIVE_RELATION_TEXT,
)
# 导入公共空间关系计算函数
from utils.spatial_utils import (
    calculate_absolute_relation,
    calculate_relative_relation,
    get_boundary_info,
    get_aabb_from_obj,
    get_object_footprint_area_xy,
)
# 导入公共物体筛选函数
from utils.object_selection import select_reference_objects
from dataset_pipeline.utils import (
    is_floor_covering,
    prefer_under_or_within,
    resolve_num_workers,
)



def clamp_scene_height(scene_height: Any, max_height: float = MAX_SCENE_HEIGHT_METERS) -> float:
    """统一限制样本中的 scene_height 上限。"""
    return min(float(scene_height), max_height)


def load_category_map(*csv_paths: Path) -> Dict[str, str]:
    """
    从资产信息 CSV 加载 jid/name_en → category 的映射。

    按 category > class_en > retrieval_class_en 优先级取值，
    与 agent/asset_retriever.py 中的取值逻辑保持一致。
    支持传入多个 CSV（如融合数据集同时使用 Imaginarium + 3D-FRONT 资产）。
    """
    mapping: Dict[str, str] = {}
    for csv_path in csv_paths:
        if not csv_path.exists():
            print(f"[WARN] 资产 CSV 不存在，跳过: {csv_path}")
            continue
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                jid = (row.get("jid") or row.get("name_en") or "").strip()
                if not jid:
                    continue
                cat = (
                    row.get("category") or row.get("class_en") or row.get("retrieval_class_en") or ""
                ).strip().lower().replace("_", " ")
                if cat and jid not in mapping:
                    mapping[jid] = cat
    return mapping


def require_category(obj: Dict[str, Any], category_map: Optional[Dict[str, str]]) -> str:
    """从资产映射中获取 category；缺失时显式报错，避免写入脏数据。"""
    jid = obj.get("jid")
    if not jid:
        raise ValueError(
            f"Missing category: object id={obj.get('id')} has no jid"
        )
    if category_map is None:
        raise ValueError(
            f"Missing category map for jid={jid}, object id={obj.get('id')}"
        )

    category = category_map.get(jid, "")
    if not category:
        raise ValueError(
            f"Missing category for jid={jid}, object id={obj.get('id')}, "
            f"short_desc={obj.get('short_desc')}"
        )
    return category


# ==================== 自回归数据生成 ====================


def generate_instruction(
    new_object: Dict[str, Any],
    existing_objects: List[Dict[str, Any]],
    boundary: List[List[float]],
    skip_within: bool = False,
) -> tuple[str, List[str], List[Dict[str, Any]], Dict[int, float]]:
    """
    根据新物体与已有物体的关系以及绝对位置信息生成摆放指令
    
    Returns:
        tuple: (instruction, absolute_relations, relative_relations, object_distances)
            - instruction: 生成的摆放指令字符串
            - absolute_relations: 实际使用的绝对关系列表
            - relative_relations: 实际使用的相对关系列表（格式与 expected_relations 一致）
            - object_distances: {obj_id: edge_distance} 所有已有物体与新物体的边缘距离
    """
    instruction_parts = []

    # 计算绝对位置关系
    boundary_info = get_boundary_info(boundary)
    abs_relations = calculate_absolute_relation(new_object, boundary_info, True)

    short_desc = new_object.get("short_desc", "object")

    # 生成基于绝对位置的指令（只保留关系描述）
    for abs_rel in abs_relations:
        instruction_parts.append(abs_rel)

    # 收集所有关系候选
    all_relations = []
    object_distances: Dict[int, float] = {}  # {obj_id: edge_distance}

    # 与每个已有物体计算关系
    for existing_obj in existing_objects:
        relation_info = calculate_relative_relation(new_object, existing_obj)
        # 记录边缘距离（用于后续筛选参考物体）
        object_distances[existing_obj["id"]] = relation_info["edge_distance"]
        # 如果有非空关系
        if "none" not in relation_info["relations"]:
            dist = relation_info["edge_distance"]
            rels = list(relation_info["relations"])
            # 3DFront 数据跳过 within 指令（该数据集中 within 关系不可靠）
            if skip_within:
                rels = [r for r in rels if r != "within"]
            if not rels:
                continue
            # within / under 同时成立时，按 reference 的 short_desc 做偏好选择，避免不自然的指令
            if "within" in rels and "under" in rels:
                preferred = prefer_under_or_within(existing_obj.get("short_desc", ""))
                if preferred == "within":
                    rels = [r for r in rels if r != "under"]
                else:
                    rels = [r for r in rels if r != "within"]

            for relation_type in rels:
                # 计算优先级：on/under/within (3) > next_to (2) > others (1)
                priority = 1
                if relation_type in ["on", "under", "within"]:
                    priority = 3
                elif relation_type == "next_to":
                    priority = 2

                all_relations.append({
                    "type": relation_type,
                    "obj": existing_obj,
                    "dist": dist,
                    "priority": priority
                })

    # 排序并筛选：优先级高 > 距离近
    # 如果关系太多，优先保留物理支撑(on/in)和近距离关系，这对于定位更有帮助
    all_relations.sort(key=lambda x: (-x["priority"], x["dist"]))

    # 最多保留3条最关键的关系，避免Prompt过长造成干扰
    selected_relations = all_relations[:3]

    # 构建实际使用的相对关系列表
    relative_relations_list: List[Dict[str, Any]] = []
    for rel in selected_relations:
        relation_type = rel["type"]
        existing_obj = rel["obj"]
        ref_id = existing_obj["id"]
        ref_short_desc = existing_obj.get("short_desc", "object")
        
        rel_desc = RELATIVE_RELATION_TEXT.get(relation_type, "")
        if rel_desc:
            instruction_parts.append(
                f"{rel_desc} the {ref_short_desc} (id {ref_id})"
            )
        
        relative_relations_list.append({
            "type": relation_type,
            "target_id": ref_id,
        })

    # 如果没有任何指令，返回通用描述
    if not instruction_parts:
        return f"place the {short_desc} in a suitable position.", [], [], object_distances

    # 组合指令：第一部分加上 "place the {short_desc}"，后续部分用 " and " 连接
    full_instruction = (
        f"place the {short_desc} " + ", and ".join(instruction_parts) + "."
    )
    return full_instruction, abs_relations, relative_relations_list, object_distances


def _build_parent_relationships(
    objects: List[Dict[str, Any]],
) -> Tuple[Dict[int, int | None], Dict[int, List[int]], Dict[int, float], Dict[int, float]]:
    """基于 on/within/under 关系构建父子关系，并返回排序所需的几何缓存。"""
    n = len(objects)

    # 预计算每个物体的占地面积和底部高度
    footprint_area: Dict[int, float] = {}
    bottom_z: Dict[int, float] = {}
    for i in range(n):
        try:
            footprint_area[i] = get_object_footprint_area_xy(objects[i])
        except Exception as exc:
            obj_id = objects[i].get("id", i)
            raise ValueError(f"对象占地面积计算失败: index={i}, id={obj_id}") from exc
        
        try:
            obj_min, _ = get_aabb_from_obj(objects[i])
            bottom_z[i] = float(obj_min[2])
        except Exception as exc:
            obj_id = objects[i].get("id", i)
            raise ValueError(f"对象底部高度计算失败: index={i}, id={obj_id}") from exc

    # 构建支撑关系树：找出每个物体的"父物体"（支撑物）
    # 规则：如果 A on B 或 A within B 或 A under B，则 B 是 A 的父物体
    parent: Dict[int, int | None] = {i: None for i in range(n)}
    children: Dict[int, List[int]] = {i: [] for i in range(n)}
    
    for i in range(n):
        best_parent: int | None = None
        best_priority = -1  # on > within > under
        best_area = float("inf")  # 选择最小的父物体（更精确的支撑）
        
        for j in range(n):
            if i == j:
                continue
            relation_info = calculate_relative_relation(objects[i], objects[j])
            relations = relation_info["relations"]
            
            # 检查支撑关系
            # - "on": A 在 B 上面 → B 是支撑物，B 先放，B 是 A 的父物体 ✓
            # - "within": A 在 B 内部 → B 是容器，B 先放，B 是 A 的父物体 ✓
            # - "under": A 在 B 下方 → B 是上层/容纳结构，B 先放，B 是 A 的父物体 ✓
            has_on = "on" in relations
            has_within = "within" in relations
            has_under = "under" in relations
            
            # 地毯是地面装饰，不作为 on 关系的支撑物
            if has_on and is_floor_covering(objects[j]):
                has_on = False
            
            if has_on or has_within or has_under:
                # on 优先级高于 within，within 高于 under
                if has_on:
                    priority = 3
                elif has_within:
                    priority = 2
                else:
                    priority = 1
                area_j = footprint_area[j]
                
                # 选择优先级更高的，或同优先级下面积更小的
                if priority > best_priority or (priority == best_priority and area_j < best_area):
                    best_parent = j
                    best_priority = priority
                    best_area = area_j
        
        parent[i] = best_parent
    
    # 检测并打破循环依赖
    # 策略：对于循环中的物体，选择底部高度最低的作为"根"（无父物体）
    def find_cycle_root(start: int) -> int | None:
        """检测从 start 开始是否存在循环，返回循环中底部高度最低的物体索引"""
        visited = set()
        path = []
        current = start
        while current is not None:
            if current in visited:
                # 发现循环，找出循环中底部最低的物体
                cycle_start_idx = path.index(current)
                cycle = path[cycle_start_idx:]
                # 返回循环中底部高度最低的物体
                return min(cycle, key=lambda x: (bottom_z.get(x, 0.0), -footprint_area.get(x, 0.0), x))
            visited.add(current)
            path.append(current)
            current = parent[current]
        return None
    
    # 打破所有循环
    for i in range(n):
        cycle_root = find_cycle_root(i)
        if cycle_root is not None and parent[cycle_root] is not None:
            # 将循环中底部最低的物体设为根节点
            parent[cycle_root] = None
    
    # 重新构建子节点列表（因为可能打破了一些父子关系）
    children = {i: [] for i in range(n)}
    for i in range(n):
        if parent[i] is not None:
            children[parent[i]].append(i)

    return parent, children, footprint_area, bottom_z


def determine_object_order(
    objects: List[Dict[str, Any]], boundary: List[List[float]]
) -> List[int]:
    """
    确定物体添加顺序（使用物体索引）
    
    分层排序策略：
    1. 第一层：摆放所有地面物体（按占地面积从大到小）
    2. 第二层及更深：按地面物体的顺序，依次摆放各自区域的子物体
    
    例如：床 -> 衣柜 -> 桌子 -> 床上物品 -> 衣柜内物品 -> 桌上物品
    
    注意：处理循环依赖的情况（如 A on B 且 B under A），通过打破循环确保不丢失物体
    """
    n = len(objects)
    if n == 0:
        return []

    parent, children, footprint_area, bottom_z = _build_parent_relationships(objects)
    
    # 找出所有根节点（地面物体）
    ground_objects = [i for i in range(n) if parent[i] is None]
    
    # 对地面物体按占地面积从大到小排序
    ground_objects.sort(
        key=lambda x: (
            -footprint_area.get(x, 0.0),  # 面积大的先放
            bottom_z.get(x, 0.0),          # 同面积时，更低的先放
            x,                              # 稳定排序
        )
    )
    
    # 分层摆放：先摆所有地面物体，再按地面物体顺序摆放子物体
    result: List[int] = []
    visited_in_result: set[int] = set()
    
    # 第一层：所有地面物体
    result.extend(ground_objects)
    visited_in_result.update(ground_objects)
    
    # 第二层及更深：BFS 按层遍历，每层按父物体顺序处理
    current_layer = ground_objects[:]
    while current_layer:
        next_layer: List[int] = []
        for p in current_layer:
            # 对每个父物体的子物体排序：面积大的先放
            kids = children[p]
            kids.sort(
                key=lambda x: (
                    -footprint_area.get(x, 0.0),
                    bottom_z.get(x, 0.0),
                    x,
                )
            )
            for kid in kids:
                if kid not in visited_in_result:
                    next_layer.append(kid)
                    visited_in_result.add(kid)
        result.extend(next_layer)
        current_layer = next_layer
    
    # 安全检查：确保所有物体都被包含
    # （理论上打破循环后不应该有遗漏，但以防万一）
    missing = [i for i in range(n) if i not in visited_in_result]
    if missing:
        # 按底部高度和面积排序后添加
        missing.sort(
            key=lambda x: (
                bottom_z.get(x, 0.0),
                -footprint_area.get(x, 0.0),
                x,
            )
        )
        result.extend(missing)
    
    return result


def _select_sibling_reference_ids(
    new_obj: Dict[str, Any],
    existing_objects: List[Dict[str, Any]],
    parent_by_object_id: Dict[int, int | None],
    object_distances: Dict[int, float],
    max_sibling_refs: int,
) -> set[int]:
    """保留同一 on/within/under 父物体下的已放 sibling，避免局部空间重叠。"""
    if max_sibling_refs <= 0:
        return set()

    parent_id = parent_by_object_id.get(new_obj["id"])
    if parent_id is None:
        return set()

    sibling_candidates = [
        obj for obj in existing_objects
        if parent_by_object_id.get(obj["id"]) == parent_id
    ]
    sibling_candidates.sort(
        key=lambda obj: (
            object_distances.get(obj["id"], float("inf")),
            obj["id"],
        )
    )

    return {obj["id"] for obj in sibling_candidates[:max_sibling_refs]}


def generate_autoregressive_dataset(
    scene_data: Dict[str, Any],
    max_ref_objects: int = MAX_REF_OBJECTS,
    skip_within: bool = False,
    category_map: Optional[Dict[str, str]] = None,
    max_sibling_refs: int = MAX_SIBLING_REFS,
) -> List[Dict[str, Any]]:
    """
    生成自回归训练数据集
    按摆放顺序分配id (0, 1, 2, ...). 并实时计算空间关系
    
    Args:
        scene_data: 场景数据
        max_ref_objects: 每个样本最多保留的参考物体数量 (0=不筛选)
        max_sibling_refs: 同一父物体下额外强制保留的 sibling 数量
        skip_within: 是否跳过 within 关系（3DFront 数据中该关系不可靠）
        category_map: jid → category 映射字典（从资产 CSV 加载）
    """
    training_samples = []

    scene_type = scene_data.get("scene_type", "room")
    scene_id = scene_data.get("scene_id", "unknown")
    boundary = scene_data.get("boundary", [])
    scene_height = clamp_scene_height(scene_data.get("scene_height", 2.8))
    original_objects = scene_data.get("objects", [])

    # 跳过没有 desc 的物体，不参与数据生成
    original_objects = [obj for obj in original_objects if obj.get("desc")]

    if len(original_objects) == 0:
        return training_samples

    # 确定物体添加顺序（返回原始物体列表的索引）
    object_order = determine_object_order(original_objects, boundary)
    parent_by_original_index, _, _, _ = _build_parent_relationships(original_objects)
    original_to_ordered_id = {
        original_index: ordered_id
        for ordered_id, original_index in enumerate(object_order)
    }
    parent_by_ordered_id: Dict[int, int | None] = {}
    for ordered_id, original_index in enumerate(object_order):
        parent_original_index = parent_by_original_index[original_index]
        if parent_original_index is None:
            parent_by_ordered_id[ordered_id] = None
        else:
            parent_by_ordered_id[ordered_id] = original_to_ordered_id[parent_original_index]

    # 创建按摆放顺序重新组织的物体列表，并分配 id
    ordered_objects = []
    for id, original_index in enumerate(object_order):
        obj = original_objects[original_index].copy()
        obj["id"] = id  # 按摆放顺序分配id
        ordered_objects.append(obj)

    # 逐步添加物体，生成训练样本
    filter_objects = max_ref_objects > 0

    for step in range(len(ordered_objects)):
        new_obj = ordered_objects[step]

        # 已有物体列表（已经按顺序摆放的物体）
        existing_objects = ordered_objects[:step]

        # 生成摆放指令（用全量物体计算关系），同时返回实际使用的关系和距离
        instruction, absolute_relations, relative_relations, object_distances = (
            generate_instruction(new_obj, existing_objects, boundary, skip_within=skip_within)
        )

        # 筛选参考物体（保持原始 ID 不变）
        if filter_objects and len(existing_objects) > max_ref_objects:
            referenced_ids = {r["target_id"] for r in relative_relations}
            referenced_ids.update(_select_sibling_reference_ids(
                new_obj,
                existing_objects,
                parent_by_ordered_id,
                object_distances,
                max_sibling_refs,
            ))
            filtered_objects = select_reference_objects(
                existing_objects, referenced_ids, object_distances,
                max_total=max_ref_objects,
            )
        else:
            filtered_objects = existing_objects

        scene_description = f"A {scene_id.replace('_', '-').split('-')[0].lower()}"
        # 构建训练样本
        sample = {
            "scene_id": scene_id,
            "scene_type": scene_type,
            "step": step,
            "scene_description": scene_description,
            "boundary": boundary,
            "scene_height": scene_height,
            "existing_objects": [
                {
                    "id": obj["id"],
                    "desc": obj.get("desc"),
                    "short_desc": obj.get("short_desc"),
                    "category": require_category(obj, category_map),
                    "center": [round(x, 2) for x in obj["center"]],
                    "rotation": [round(x) for x in obj["rotation"]],
                    "size": [round(x, 2) for x in obj["size"]],
                    "jid": obj.get("jid"),
                }
                for obj in filtered_objects
            ],
            "new_object": {
                "id": new_obj["id"],
                "desc": new_obj.get("desc"),
                "short_desc": new_obj.get("short_desc"),
                "category": require_category(new_obj, category_map),
                "size": [round(x, 2) for x in new_obj["size"]],
                "jid": new_obj.get("jid"),
            },
            "instruction": instruction,
            "target_pose": {
                "center": [round(x, 2) for x in new_obj["center"]],
                "rotation": [round(x) for x in new_obj["rotation"]],
            },
            "expected_relations": {
                "absolute_relations": absolute_relations,
                "relative_relations": relative_relations,
            },
        }

        training_samples.append(sample)

    return training_samples


def process_single_scene(input_path: str, output_path: str):
    """
    处理单个场景文件
    """
    print(f"正在处理: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        scene_data = json.load(f)

    training_samples = generate_autoregressive_dataset(scene_data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(training_samples, f, indent=2, ensure_ascii=False)

    print(f"已生成 {len(training_samples)} 个训练样本，保存到: {output_path}")


def _process_single_file(
    json_file: Path,
    output_dir: Path,
    max_ref_objects: int = MAX_REF_OBJECTS,
    skip_within: bool = False,
    max_sibling_refs: int = MAX_SIBLING_REFS,
    category_map: Optional[Dict[str, str]] = None,
) -> Tuple[bool, str]:
    """
    处理单个文件的 worker 函数（用于多进程）
    
    Returns:
        Tuple[bool, str]: (是否成功, 文件名或错误信息)
    """
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            scene_data = json.load(f)

        samples = generate_autoregressive_dataset(
            scene_data,
            max_ref_objects=max_ref_objects,
            skip_within=skip_within,
            max_sibling_refs=max_sibling_refs,
            category_map=category_map,
        )

        # 跳过空场景
        if not samples:
            return False, f"空场景: {json_file.name}"

        # 保存到文件
        output_file = output_dir / f"{json_file.stem}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)

        return True, json_file.name
    except (ValueError, KeyError, IndexError, TypeError) as e:
        return False, f"跳过文件 {json_file.name}: {str(e)}"


def process_directory(
    input_dir: str,
    output_dir: str,
    num_workers: int = None,
    max_ref_objects: int = MAX_REF_OBJECTS,
    skip_within: bool = False,
    max_sibling_refs: int = MAX_SIBLING_REFS,
    category_map: Optional[Dict[str, str]] = None,
):
    """
    批量处理目录中的所有场景文件（支持多进程并行）
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        num_workers: 并行工作进程数，默认为 CPU 核心数
        max_sibling_refs: 同一父物体下额外强制保留的 sibling 数量
        skip_within: 是否跳过 within 关系（3DFront 数据中该关系不可靠）
        category_map: jid → category 映射字典
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if input_path.resolve() == output_path.resolve():
        raise ValueError("输入目录和输出目录不能相同，否则会在清理旧文件时删除输入场景")

    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)

    # 默认清理输出目录中旧的 JSON，避免残留文件污染统计结果
    old_json_files = list(output_path.glob("*.json"))
    if old_json_files:
        print(f"清理输出目录旧 JSON 文件: {len(old_json_files)} 个")
        for old_file in old_json_files:
            old_file.unlink()

    # 查找所有 JSON 文件
    json_files = list(input_path.glob("*.json"))
    total_files = len(json_files)

    print(f"找到 {total_files} 个场景文件")

    if total_files == 0:
        return

    # 确定工作进程数
    num_workers = resolve_num_workers(num_workers, total_files)

    print(f"使用 {num_workers} 个并行进程处理")

    skipped_count = 0
    success_count = 0

    # 使用多进程池并行处理
    worker_func = partial(
        _process_single_file,
        output_dir=output_path,
        max_ref_objects=max_ref_objects,
        skip_within=skip_within,
        max_sibling_refs=max_sibling_refs,
        category_map=category_map,
    )

    if num_workers == 1:
        results = []
        for json_file in tqdm(json_files, total=total_files, desc="处理场景文件"):
            results.append(worker_func(json_file))
    else:
        with Pool(processes=num_workers) as pool:
            # 使用 imap_unordered 以获得更好的进度显示
            results = list(
                tqdm(
                    pool.imap_unordered(worker_func, json_files),
                    total=total_files,
                    desc="处理场景文件",
                )
            )

    # 统计结果
    for success, msg in results:
        if success:
            success_count += 1
        else:
            skipped_count += 1
            if "跳过文件" in msg:
                tqdm.write(msg)

    print(f"\n处理完成: 成功 {success_count} 个, 跳过 {skipped_count} 个")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成自回归数据")
    parser.add_argument("--input-dir", type=str, required=True, help="输入目录")
    parser.add_argument("--output-dir", type=str, required=True, help="输出目录")
    parser.add_argument(
        "--num-workers", "-j", type=int, default=None,
        help="并行工作进程数（默认为 CPU 核心数的一半）"
    )
    parser.add_argument(
        "--max-ref-objects", type=int, default=MAX_REF_OBJECTS,
        help=f"每个样本最多保留的参考物体数量 (0=不筛选, 默认 {MAX_REF_OBJECTS})",
    )
    parser.add_argument(
        "--max-sibling-refs", type=int, default=MAX_SIBLING_REFS,
        help=(
            "同一 on/within/under 父物体下额外强制保留的 sibling 数量 "
            f"(默认 {MAX_SIBLING_REFS})"
        ),
    )
    parser.add_argument(
        "--skip-within", action="store_true", default=False,
        help="跳过 within 关系（3DFront 数据中该关系不可靠）",
    )
    parser.add_argument(
        "--asset-csv",
        type=str,
        nargs="+",
        default=None,
        help="资产信息 CSV 路径（用于读取 jid → category 映射），可指定多个",
    )
    args = parser.parse_args()

    # 加载 category 映射
    category_map = None
    if args.asset_csv:
        csv_paths = [Path(p) for p in args.asset_csv]
        category_map = load_category_map(*csv_paths)
        print(f"已加载 {len(category_map)} 个资产的 category 映射")

    # 批量处理整个目录
    process_directory(
        args.input_dir,
        args.output_dir,
        args.num_workers,
        args.max_ref_objects,
        args.skip_within,
        args.max_sibling_refs,
        category_map=category_map,
    )

    # 处理单个场景文件
    # input_file = ProjectPaths.SSR_3DFRONT_DIR / "converted_scenes" / "0003d406-5f27-4bbf-94cd-1cff7c310ba1-c441536f-6867-4eb9-9dfd-354a7d3bdcef.json"
    # output_file = ProjectPaths.SSR_3DFRONT_AUTOREGRESSIVE_DIR / "0003d406-5f27-4bbf-94cd-1cff7c310ba1-c441536f-6867-4eb9-9dfd-354a7d3bdcef.json"
    # process_single_scene(input_file, output_file)

# run
# python dataset_pipeline/preprocessing/03_generate_autoregressive_data.py