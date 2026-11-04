import json
import random
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

# 添加项目根目录到 Python 路径

from config import ProjectPaths
from dataset_pipeline.utils import build_metadata, export_metadata_to_csv
from utils.message import SYSTEM_PROMPT, format_output, format_scene_input, load_internal_spaces_map


def convert_to_training_format(
    autoregressive_data: List[Dict[str, Any]],
    start_index: int = 0,
    internal_spaces_map: Dict = None,
    include_object_descriptions: bool = False,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    将自回归数据转换为 LLM 训练格式
    
    Returns:
        (training_data, metadata_list): 训练数据和元数据列表
    """
    training_data = []
    metadata_list = []

    for idx, sample in enumerate(autoregressive_data):
        sample_id = start_index + idx
        
        # 使用公共模块构建元数据
        metadata = build_metadata(sample, sample_id)
        
        training_sample = {
            "scene_id": sample["scene_id"],
            "system": SYSTEM_PROMPT,
            "instruction": format_scene_input(
                sample,
                use_separate_pointcloud=False,
                internal_spaces_map=internal_spaces_map,
                include_object_descriptions=include_object_descriptions,
            ),
            "output": format_output(sample),
            "expected_relations": sample["expected_relations"],
        }

        training_data.append(training_sample)
        metadata_list.append(metadata)

    return training_data, metadata_list


def process_single_file(
    input_file: str,
    output_file: str,
    csv_file: str = None,
    include_object_descriptions: bool = False,
):
    """
    处理单个自回归数据文件，生成训练数据和元数据CSV
    """
    print(f"正在处理: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        autoregressive_data = json.load(f)

    training_data, metadata_list = convert_to_training_format(
        autoregressive_data,
        include_object_descriptions=include_object_descriptions,
    )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)

    print(f"已生成 {len(training_data)} 条训练样本，保存到: {output_file}")
    
    # 导出元数据到 CSV（纯文本版本不包含点云路径）
    if csv_file:
        export_metadata_to_csv(metadata_list, csv_file, include_pointcloud_paths=False)
        print(f"元数据已保存到: {csv_file}")


def process_directory(
    input_dir: str,
    output_dir: str,
    dataset: str = "3dfront",
    samples_max_count: int = None,
    output_json_name: str = "3dfront.json",
    output_csv_name: str = "3dfront_metadata.csv",
    with_internal_spaces: bool = False,
    include_object_descriptions: bool = False,
):
    """
    批量处理目录中的所有自回归数据文件
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)

    # 按需加载收纳空间数据（使用 Imaginarium 资产的数据集支持）
    internal_spaces_map = None
    if with_internal_spaces and dataset in ("imaginarium", "spatiallm", "3dis", "handcrafted", "3dish"):
        internal_spaces_map = load_internal_spaces_map(ProjectPaths.IMAGINARIUM_INTERNAL_SPACES_DIR)
        print(f"已加载 {len(internal_spaces_map)} 个资产的收纳空间数据")

    # 查找所有 JSON 文件并打乱顺序（递归查找以支持子目录结构）
    # 打乱文件顺序可以确保生成的训练数据中不同场景类型混合分布
    # 这对于 Imaginarium 等按场景名排序的数据集尤为重要
    json_files = sorted(list(input_path.glob("**/*.json")))
    random.seed(42)  # 固定随机种子，保证可复现
    random.shuffle(json_files)

    print(f"数据集: {dataset.upper()}")
    print(f"找到 {len(json_files)} 个自回归数据文件")

    all_training_data = []
    all_metadata = []

    samples_count = 0

    for json_file in tqdm(json_files):
        # print(f"正在处理: {json_file.name}")

        with open(json_file, "r", encoding="utf-8") as f:
            autoregressive_data = json.load(f)
        
        # 兼容单个对象和列表格式
        if isinstance(autoregressive_data, dict):
            autoregressive_data = [autoregressive_data]

        training_data, metadata_list = convert_to_training_format(
            autoregressive_data,
            start_index=samples_count,
            internal_spaces_map=internal_spaces_map,
            include_object_descriptions=include_object_descriptions,
        )
        all_training_data.extend(training_data)
        all_metadata.extend(metadata_list)
        samples_count += len(training_data)
        if samples_max_count and samples_count >= samples_max_count:
            break

    # 保存所有训练数据到一个文件
    all_data_file = output_path / output_json_name
    with open(all_data_file, "w", encoding="utf-8") as f:
        json.dump(all_training_data, f, indent=2, ensure_ascii=False)

    print(f"\n总共生成 {samples_count} 条训练样本")
    print(f"所有数据已保存到: {all_data_file}")
    
    # 导出元数据到 CSV
    metadata_csv_file = output_path / output_csv_name
    export_metadata_to_csv(all_metadata, metadata_csv_file, include_pointcloud_paths=False)
    print(f"元数据已保存到: {metadata_csv_file}")


if __name__ == "__main__":    
    import argparse

    parser = argparse.ArgumentParser(description="生成训练数据（纯文本）")
    parser.add_argument(
        "--dataset",
        type=str,
        default="3dish",
        help="数据集类型 (3dfront, imaginarium, spatiallm, 3dis, handcrafted, 3dish 等, 默认: 3dish)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最大样本数量（用于调试）",
    )
    parser.add_argument(
        "--output-json-name",
        '-json',
        type=str,
        default=None,
        help="输出训练数据 JSON 文件名（默认根据数据集自动设置）",
    )
    parser.add_argument(
        "--output-csv-name",
        '-csv',
        type=str,
        default=None,
        help="输出元数据 CSV 文件名（默认根据数据集自动设置）",
    )
    parser.add_argument(
        "--input-dir",
        '-i',
        type=str,
        default=None,
        help="自定义输入目录（自回归数据文件夹），若指定则忽略 --dataset 参数",
    )
    parser.add_argument(
        "--with-internal-spaces",
        action="store_true",
        default=False,
        help="在 instruction 中写入 internal_spaces（默认关闭）",
    )
    parser.add_argument(
        "--with-object-descriptions",
        action="store_true",
        default=False,
        help="在训练输入中写入 existing_objects/new_object 的 short_desc，并保留 instruction 中的物体描述（默认只使用 category 标签）",
    )

    args = parser.parse_args()

    dataset = args.dataset.lower()
    include_object_descriptions = args.with_object_descriptions
    
    # 如果指定了自定义输入目录，则使用自定义目录
    if args.input_dir:
        autoregressive_dir = Path(args.input_dir)
        if not autoregressive_dir.exists():
            raise ValueError(f"输入目录不存在: {autoregressive_dir}")
        # 自定义目录时，使用目录名作为默认文件名前缀
        dir_name = autoregressive_dir.name
        default_json_name = f"{dir_name}.json"
        default_csv_name = f"{dir_name}_metadata.csv"
    else:
        autoregressive_dir = ProjectPaths.get_autoregressive_dir(dataset)
        default_json_name = ProjectPaths.get_training_data_path(dataset).name
        default_csv_name = ProjectPaths.get_metadata_path(dataset).name

    output_dir = ProjectPaths.DATA_PROCESSED_DIR
    samples_max_count = args.max_samples
    output_json_name = args.output_json_name or default_json_name
    output_csv_name = args.output_csv_name or default_csv_name
    
    print("=" * 60)
    print("生成训练数据（纯文本）")
    print("=" * 60)
    
    # 确保输出目录存在
    ProjectPaths.ensure_dirs()
    
    if args.input_dir:
        print(f"自定义输入目录: {autoregressive_dir}")
    else:
        print(f"数据集: {dataset.upper()}")
        print(f"输入目录: {autoregressive_dir}")
    print(f"输出文件: {output_dir / output_json_name}")
    print(f"元数据文件: {output_dir / output_csv_name}")
    print(f"写入 internal_spaces: {'是' if args.with_internal_spaces else '否'}")
    print(f"写入物体描述: {'是' if include_object_descriptions else '否'}")
    if samples_max_count:
        print(f"最大样本数: {samples_max_count}")
    print("=" * 60)
    print()

    # 批量处理整个目录
    process_directory(
        str(autoregressive_dir),
        str(output_dir),
        dataset=dataset,
        samples_max_count=samples_max_count,
        output_json_name=output_json_name,
        output_csv_name=output_csv_name,
        with_internal_spaces=args.with_internal_spaces,
        include_object_descriptions=include_object_descriptions,
    )


# python dataset_pipeline/generation/01_generate_training_data.py --dataset 3dish
# python dataset_pipeline/generation/01_generate_training_data.py --dataset imaginarium
# python dataset_pipeline/generation/01_generate_training_data.py --input-dir /path/to/custom/autoregressive_dir