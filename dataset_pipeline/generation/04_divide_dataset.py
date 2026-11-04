"""
将生成的数据集划分为训练集、测试集和验证集（默认按 90:5:5 / 18:1:1 比例划分）

用法:
    # 默认按场景家族分层 90:5:5 自动划分（推荐）
    python dataset_pipeline/generation/04_divide_dataset.py --input 3dish_separate_pointcloud.json --group-by-scene-family

    # 手动指定各集合大小
    python dataset_pipeline/generation/04_divide_dataset.py --input 3dish.json --train-size 90000 --val-size 5000 --test-size 5000
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


# 添加项目根目录到 Python 路径

from config import ProjectPaths


ROTATION_SUFFIX_PATTERN = re.compile(r"_rot\d+$")
SOURCE_PREFIX_PATTERN = re.compile(r"^([A-Za-z0-9]+)_")


def _to_scene_family(scene_id: str) -> str:
    """将 scene_id 归一化为场景家族 ID（去掉结尾旋转后缀）。"""
    family_id = ROTATION_SUFFIX_PATTERN.sub("", scene_id)
    if family_id == "":
        raise ValueError(f"无效 scene_id，无法提取 scene family: {scene_id}")
    return family_id


def _to_scene_source(scene_id: str) -> str:
    """从 scene_id 提取来源标签（如 3dfront / imaginarium）。"""
    match = SOURCE_PREFIX_PATTERN.match(scene_id)
    if match:
        return match.group(1).lower()
    return "unknown"


def _extract_scene_id(item: Dict[str, Any], sample_idx: int) -> str:
    """
    从样本中提取 scene_id。

    优先读取 item["scene_id"]，其次尝试从 instruction JSON 中读取 scene_id 字段。
    若都不存在则直接报错，避免静默退化为随机切分。
    """
    if "scene_id" in item:
        scene_id = item["scene_id"]
        if isinstance(scene_id, str) and scene_id != "":
            return scene_id
        raise ValueError(
            f"样本 #{sample_idx} 的 `scene_id` 不是有效非空字符串: {scene_id}"
        )

    if "instruction" in item and isinstance(item["instruction"], str):
        try:
            instruction_obj = json.loads(item["instruction"])
        except json.JSONDecodeError:
            instruction_obj = None

        if isinstance(instruction_obj, dict) and "scene_id" in instruction_obj:
            scene_id = instruction_obj["scene_id"]
            if isinstance(scene_id, str) and scene_id != "":
                return scene_id
            raise ValueError(
                f"样本 #{sample_idx} 的 instruction.scene_id 不是有效非空字符串: {scene_id}"
            )

    raise ValueError(
        f"样本 #{sample_idx} 缺少 `scene_id`，无法按场景家族分组切分。"
        "请使用包含 scene_id 的输入数据（例如 autoregressive 或 metadata 数据）。"
    )


def _allocate_source_quota(
    target_size: int,
    source_capacity: Dict[str, int],
) -> Dict[str, int]:
    """
    按来源容量比例分配配额（最大余数法），并保证不超过各来源容量。

    target_size: 目标总数（样本数或 scene family 数）
    source_capacity: 每个来源可用容量
    """
    quotas = {source: 0 for source in source_capacity}
    if target_size <= 0:
        return quotas

    total_capacity = sum(source_capacity.values())
    if total_capacity <= 0:
        return quotas

    effective_target = min(target_size, total_capacity)

    remainders: List[tuple[float, str]] = []
    assigned = 0
    for source, capacity in source_capacity.items():
        exact = effective_target * capacity / total_capacity
        base = int(exact)
        if base > capacity:
            base = capacity
        quotas[source] = base
        assigned += base
        remainders.append((exact - base, source))

    remaining = effective_target - assigned
    remainders.sort(key=lambda item: (-item[0], item[1]))
    idx = 0
    while remaining > 0 and idx < len(remainders):
        _, source = remainders[idx]
        if quotas[source] < source_capacity[source]:
            quotas[source] += 1
            remaining -= 1
        idx += 1

    # 若出现容量限制导致剩余配额未分完，按来源顺序补齐。
    if remaining > 0:
        for source in sorted(source_capacity.keys()):
            if remaining <= 0:
                break
            spare = source_capacity[source] - quotas[source]
            if spare <= 0:
                continue
            take = min(spare, remaining)
            quotas[source] += take
            remaining -= take

    return quotas


def _take_families_for_quota(
    family_ids: List[str],
    quota: int,
) -> tuple[List[str], List[str]]:
    """按 scene family 数量配额切分有序 family 列表。"""
    if quota <= 0:
        return [], list(family_ids)
    split_at = min(quota, len(family_ids))
    return family_ids[:split_at], family_ids[split_at:]


def _split_by_scene_family(
    data: List[Dict[str, Any]],
    train_size: int,
    val_size: int,
    test_size: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    按 scene family 切分数据，保证同一家族不会跨 train/val/test。

    train_size / val_size / test_size 在该模式下表示 scene family 数。
    在来源(source)维度做分层配额，再在各来源内按 family 顺序填充。
    样本数由所选 family 内的样本数自然决定，可能与样本级比例不同。
    """
    family_to_samples: Dict[str, List[Dict[str, Any]]] = {}
    family_to_source: Dict[str, str] = {}
    source_to_families: Dict[str, List[str]] = {}
    ordered_sources: List[str] = []

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"样本 #{idx} 不是 dict，实际类型: {type(item)}")

        scene_id = _extract_scene_id(item, sample_idx=idx)
        family_id = _to_scene_family(scene_id)
        source = _to_scene_source(scene_id)

        if source not in source_to_families:
            source_to_families[source] = []
            ordered_sources.append(source)

        if family_id not in family_to_samples:
            family_to_samples[family_id] = []
            family_to_source[family_id] = source
            source_to_families[source].append(family_id)
        elif family_to_source[family_id] != source:
            raise ValueError(
                f"scene family 来源不一致: family={family_id}, "
                f"old_source={family_to_source[family_id]}, new_source={source}"
            )
        family_to_samples[family_id].append(item)

    total_families = sum(len(source_to_families[source]) for source in ordered_sources)
    print(f"Detected scene families: {total_families}")

    split_targets = {
        "train": train_size,
        "val": val_size,
        "test": test_size,
    }
    remaining_families = {
        source: list(source_to_families[source]) for source in ordered_sources
    }
    selected_families: Dict[str, List[str]] = {name: [] for name in split_targets}
    source_family_actual: Dict[str, Dict[str, int]] = {
        name: {source: 0 for source in ordered_sources} for name in split_targets
    }

    for split_name, target_size in split_targets.items():
        source_capacity = {
            source: len(remaining_families[source])
            for source in ordered_sources
        }
        quotas = _allocate_source_quota(
            target_size=target_size,
            source_capacity=source_capacity,
        )
        print(f"{split_name.capitalize()} source quotas: {quotas}")

        for source in ordered_sources:
            chosen, remaining = _take_families_for_quota(
                family_ids=remaining_families[source],
                quota=quotas[source],
            )
            selected_families[split_name].extend(chosen)
            remaining_families[source] = remaining
            source_family_actual[split_name][source] = len(chosen)

    split_data: Dict[str, List[Dict[str, Any]]] = {}
    for split_name, family_ids in selected_families.items():
        split_data[split_name] = [
            sample
            for family_id in family_ids
            for sample in family_to_samples[family_id]
        ]

    print(f"Train source families: {source_family_actual['train']}")
    print(f"Val source families: {source_family_actual['val']}")
    print(f"Test source families: {source_family_actual['test']}")

    unused_families = sum(
        1
        for source in ordered_sources
        for family_id in remaining_families[source]
    )
    if unused_families > 0:
        print(f"Unused scene families after requested split sizes: {unused_families}")

    return split_data["train"], split_data["val"], split_data["test"]


def _attach_sample_ids(data: List[Any]) -> List[Dict[str, Any]]:
    """
    为样本写入 sample_id（原始数据中的索引）。

    返回新列表，不修改原始输入对象。
    """
    data_with_ids: List[Dict[str, Any]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"样本 #{idx} 不是 dict，实际类型: {type(item)}")
        data_with_ids.append({**item, "sample_id": idx})
    return data_with_ids


def _count_scene_families(data: List[Dict[str, Any]]) -> int:
    """统计输入数据中的 scene family 数量。"""
    families = {
        _to_scene_family(_extract_scene_id(item, sample_idx=idx))
        for idx, item in enumerate(data)
    }
    return len(families)


def compute_default_split_sizes(
    data: List[Dict[str, Any]],
    group_by_scene_family: bool,
) -> tuple[int, int, int]:
    """按 90:5:5 (18:1:1) 计算默认的 (train_size, val_size, test_size)。"""
    if group_by_scene_family:
        total = _count_scene_families(data)
    else:
        total = len(data)
        rounded = (total // 10000) * 10000
        if rounded != 0:
            total = rounded

    weights = [("train", 90), ("val", 5), ("test", 5)]
    values: Dict[str, int] = {}
    remainders: List[tuple[int, str]] = []
    assigned = 0
    for name, weight in weights:
        raw = total * weight
        values[name] = raw // 100
        assigned += values[name]
        remainders.append((raw % 100, name))
    remaining = total - assigned
    for _, name in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if remaining <= 0:
            break
        values[name] += 1
        remaining -= 1
    return values["train"], values["val"], values["test"]


def divide_dataset(
    input_file: Path,
    train_size: int | None = None,
    test_size: int | None = None,
    val_interval: int = 10,
    val_size: int | None = None,
    output_dir: Path | None = None,
    group_by_scene_family: bool = False,
):
    """
    将数据集划分为训练集、测试集和验证集

    Args:
        input_file: 输入 JSON 文件路径
        train_size: 训练集大小；为 None 时按 90:5:5 自动计算
        test_size: 测试集大小；为 None 时按 90:5:5 自动计算
        val_interval: 未显式传 val_size 时，从原 holdout 中估算验证集大小
        val_size: 验证集大小；为 None 时从 test_size 和 val_interval 兼容推导或自动计算
        output_dir: 输出目录（默认与输入文件同目录）
        group_by_scene_family: 是否按 scene family 分组切分（避免 rot 增强数据泄漏）
    """
    if val_interval <= 0:
        raise ValueError(f"val_interval 必须大于 0，当前: {val_interval}")

    # 读取数据
    print(f"Reading data from {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    data = _attach_sample_ids(data)

    total = len(data)
    print(f"Total samples: {total}")

    if train_size is None and test_size is None and val_size is None:
        train_size, val_size, test_size = compute_default_split_sizes(
            data, group_by_scene_family=group_by_scene_family
        )
        unit_label = "scene_families" if group_by_scene_family else "samples"
        print(
            f"Auto-computed split sizes ({unit_label}, 90:5:5): "
            f"train={train_size}, val={val_size}, test={test_size}"
        )
    else:
        if train_size is None or test_size is None:
            raise ValueError("手动指定大小时，必须同时提供 --train-size 和 --test-size")
        if val_size is None:
            holdout_size = test_size
            val_size = holdout_size // val_interval
            if holdout_size > 0 and val_size == 0:
                val_size = 1
            test_size = holdout_size - val_size

    if val_size < 0:
        raise ValueError(f"val_size 不能为负数，当前: {val_size}")
    if test_size < 0:
        raise ValueError(f"test_size 不能为负数，当前: {test_size}")

    if group_by_scene_family:
        max_requested_total = _count_scene_families(data)
        size_unit = "scene families"
    else:
        max_requested_total = total
        size_unit = "samples"

    requested_total = train_size + val_size + test_size
    if requested_total > max_requested_total:
        print(
            "警告: "
            f"train_size({train_size}) + val_size({val_size}) + test_size({test_size}) "
            f"= {requested_total} > total {size_unit}({max_requested_total})"
        )
        overflow = requested_total - max_requested_total
        reduce_test = min(test_size, overflow)
        test_size -= reduce_test
        overflow -= reduce_test
        reduce_val = min(val_size, overflow)
        val_size -= reduce_val
        overflow -= reduce_val
        if overflow > 0:
            train_size = max(0, train_size - overflow)
        print(
            f"  调整后: train_size={train_size}, val_size={val_size}, "
            f"test_size={test_size}"
        )

    # 划分数据
    if group_by_scene_family:
        print("Split mode: group by scene family (scene_id)")
        train_data, val_data, test_data = _split_by_scene_family(
            data=data,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
        )
    else:
        print("Split mode: plain index slicing")
        train_data = data[:train_size]
        val_start = train_size
        test_start = train_size + val_size
        val_data = data[val_start:test_start]
        test_data = data[test_start : test_start + test_size]

    print(
        f"Split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}"
    )
    if group_by_scene_family:
        print(
            "注意: 按场景家族分组时，实际大小可能因 family 边界略偏离目标，"
            "同家族样本不会跨集合出现。"
        )

    # 确定输出路径
    if output_dir is None:
        output_dir = input_file.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 输出前缀固定自动生成: "{stem}_{total_k}k" (e.g. "3dima_with_pointcloud_100k")
    stem = input_file.stem  # e.g. "3dima_with_only_scene_pointcloud"
    total_k = (len(train_data) + len(val_data) + len(test_data)) // 1000
    output_prefix = f"{stem}_{total_k}k"

    # 保存
    splits = {"train": train_data, "val": val_data, "test": test_data}
    for split_name, split_data in splits.items():
        output_file = output_dir / f"{output_prefix}_{split_name}.json"
        print(f"Saving {split_name} data ({len(split_data)} samples) to {output_file}")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)

    # 当划分分离点云数据集时，自动导出论文消融与分层评测所需的子集 JSON
    if stem.endswith("_separate_pointcloud"):
        base_prefix = stem[: -len("_separate_pointcloud")]
        hq_sources = {"imaginarium", "handcrafted"}
        no_hq_train = [
            s for idx, s in enumerate(train_data)
            if _to_scene_source(_extract_scene_id(s, idx)) not in hq_sources
        ]
        no_hq_val = [
            s for idx, s in enumerate(val_data)
            if _to_scene_source(_extract_scene_id(s, idx)) not in hq_sources
        ]
        hq_test = [
            s for idx, s in enumerate(test_data)
            if _to_scene_source(_extract_scene_id(s, idx)) in hq_sources
        ]
        non_hq_test = [
            s for idx, s in enumerate(test_data)
            if _to_scene_source(_extract_scene_id(s, idx)) not in hq_sources
        ]
        if hq_test and non_hq_test:
            subset_files = [
                (f"{base_prefix}_no_imaginarium_handcrafted_separate_pointcloud_train.json", no_hq_train),
                (f"{base_prefix}_no_imaginarium_handcrafted_separate_pointcloud_val.json", no_hq_val),
                (f"{base_prefix}_imaginarium_handcrafted_separate_pointcloud_test.json", hq_test),
                (f"{base_prefix}_3dfront_spatiallm_separate_pointcloud_test.json", non_hq_test),
            ]
            for filename, subset_data in subset_files:
                subset_path = output_dir / filename
                print(f"Saving subset ({len(subset_data)} samples) to {subset_path}")
                with open(subset_path, "w", encoding="utf-8") as f:
                    json.dump(subset_data, f, indent=2, ensure_ascii=False)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="将生成的数据集划分为训练集、测试集和验证集"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="输入 JSON 文件名或完整路径（文件名会在 DATA_PROCESSED_DIR 中查找）",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=None,
        help="训练集大小（未指定时默认按 90:5:5 自动计算）",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=None,
        help="测试集大小（未指定时默认按 90:5:5 自动计算）",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=None,
        help="验证集大小；未指定时根据 --test-size 和 --val-interval 兼容推导",
    )
    parser.add_argument(
        "--val-interval",
        type=int,
        default=10,
        help="未指定 --val-size 时，从 holdout 中按间隔估算验证集大小 (默认: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录（默认与输入文件同目录）",
    )
    parser.add_argument(
        "--group-by-scene-family",
        action="store_true",
        help="按场景家族分组切分，避免同一 scene 的旋转增强样本跨 train/test",
    )

    args = parser.parse_args()

    # 解析输入文件路径: 如果是文件名则在 DATA_PROCESSED_DIR 中查找
    input_path = Path(args.input)
    if not input_path.is_absolute() and not input_path.exists():
        input_path = ProjectPaths.DATA_PROCESSED_DIR / args.input

    output_dir = Path(args.output_dir) if args.output_dir else None

    divide_dataset(
        input_file=input_path,
        train_size=args.train_size,
        test_size=args.test_size,
        val_interval=args.val_interval,
        val_size=args.val_size,
        output_dir=output_dir,
        group_by_scene_family=args.group_by_scene_family,
    )
