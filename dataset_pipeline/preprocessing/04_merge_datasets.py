"""
数据集合并脚本

将多个数据集的 autoregressive 训练数据合并为一个统一的数据集。
支持按数据源添加前缀、设置采样权重等。
"""

import json
import csv
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

from tqdm import tqdm

# 添加项目根目录到 Python 路径

from config.paths import ProjectPaths


def parse_source(source_str: str) -> Tuple[str, Path]:
    """
    解析数据源字符串
    
    格式: "name:path" 或 "path"（name 默认为目录名）
    
    Returns:
        (name, path)
    """
    if ":" in source_str:
        name, path_str = source_str.split(":", 1)
    else:
        path_str = source_str
        name = Path(path_str).name
    
    return name, Path(path_str)


def collect_files(source_dir: Path, source_name: str) -> List[Dict]:
    """
    收集数据源中的所有文件信息
    
    Returns:
        文件信息列表: [{"path": Path, "source": str, "scene_id": str}, ...]
    """
    files = []
    json_files = sorted(source_dir.glob("*.json"))
    
    for f in json_files:
        files.append({
            "path": f,
            "source": source_name,
            "original_name": f.name,
        })
    
    return files


def merge_autoregressive_datasets(
    sources: List[Tuple[str, Path]],
    output_dir: Path,
    add_source_prefix: bool = True,
):
    """
    合并多个 autoregressive 数据集
    
    Args:
        sources: [(source_name, source_path), ...]
        output_dir: 输出目录
        add_source_prefix: 是否为 scene_id 添加数据源前缀
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 清理输出目录中的旧 JSON 文件
    old_json_files = list(output_dir.glob("*.json"))
    if old_json_files:
        print(f"清理输出目录旧 JSON 文件: {len(old_json_files)} 个")
        for f in old_json_files:
            f.unlink()

    # 统计信息
    stats = defaultdict(lambda: {"files": 0, "samples": 0})
    
    # 用于检测重名冲突
    output_names = {}  # {output_name: source_path}
    
    print("收集文件...")
    all_files = []
    for name, path in sources:
        if not path.exists():
            print(f"⚠️  跳过不存在的目录: {path}")
            continue
        
        files = collect_files(path, name)
        all_files.extend(files)
        print(f"  {name}: {len(files)} 个文件")
    
    print(f"\n合并 {len(all_files)} 个文件...")
    
    for file_info in tqdm(all_files, desc="合并"):
        src_path = file_info["path"]
        source_name = file_info["source"]
        original_name = file_info["original_name"]
        
        # 构建输出文件名
        if add_source_prefix:
            output_name = f"{source_name}_{original_name}"
        else:
            output_name = original_name
        
        # 检测重名冲突
        if output_name in output_names:
            tqdm.write(f"⚠️  文件名冲突: {output_name}")
            tqdm.write(f"    来源1: {output_names[output_name]}")
            tqdm.write(f"    来源2: {src_path}")
            # 添加额外后缀避免覆盖
            stem = Path(output_name).stem
            suffix = Path(output_name).suffix
            counter = 1
            while output_name in output_names:
                output_name = f"{stem}_{counter}{suffix}"
                counter += 1
        
        output_names[output_name] = src_path
        
        try:
            # 读取并处理数据
            with open(src_path, "r", encoding="utf-8") as f:
                samples = json.load(f)
            
            sample_count = 0
            if isinstance(samples, dict):
                sample_count = 1
                if add_source_prefix:
                    original_scene_id = samples.get("scene_id", "")
                    samples["scene_id"] = f"{source_name}_{original_scene_id}"
                    samples["data_source"] = source_name
            else:
                sample_count = len(samples)
                # 更新 scene_id（添加数据源前缀）
                if add_source_prefix and samples:
                    for sample in samples:
                        original_scene_id = sample.get("scene_id", "")
                        sample["scene_id"] = f"{source_name}_{original_scene_id}"
                        # 同时更新 scene_description
                        if "scene_description" in sample:
                            sample["data_source"] = source_name
            
            # 写入输出文件
            output_path = output_dir / output_name
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(samples, f, indent=2, ensure_ascii=False)
            
            # 更新统计
            stats[source_name]["files"] += 1
            stats[source_name]["samples"] += sample_count
            
        except Exception as e:
            tqdm.write(f"处理失败 {src_path}: {e}")
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("合并统计")
    print("=" * 60)
    
    total_files = 0
    total_samples = 0
    
    for source_name, stat in stats.items():
        print(f"{source_name}:")
        print(f"  文件数: {stat['files']}")
        print(f"  样本数: {stat['samples']}")
        total_files += stat["files"]
        total_samples += stat["samples"]
    
    print("-" * 60)
    print(f"总计: {total_files} 个文件, {total_samples} 个样本")
    
    # 保存统计信息
    stats_path = output_dir.parent / "merge_stats.json"
    stats_data = {
        "sources": [{"name": name, "path": str(path)} for name, path in sources],
        "stats": dict(stats),
        "total_files": total_files,
        "total_samples": total_samples,
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_data, f, indent=2, ensure_ascii=False)
    print(f"\n统计信息已保存到: {stats_path}")


def main():
    parser = argparse.ArgumentParser(
        description="合并多个 autoregressive 数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
示例:
  # 使用默认路径（自动检测环境）
    python dataset_pipeline/preprocessing/04_merge_datasets.py --default

  # 手动指定路径
    python dataset_pipeline/preprocessing/04_merge_datasets.py \\
        --output-dir {ProjectPaths.DATASETS_ROOT}/3dima/autoregressive \\
        --sources 3dfront:{ProjectPaths.SSR_3DFRONT_AUTOREGRESSIVE_DIR} \\
                imaginarium:{ProjectPaths.IMAGINARIUM_AUTOREGRESSIVE_DIR}

  # 使用环境变量切换环境（无需修改代码）
    export DATASETS_ROOT=/your/custom/path
    python dataset_pipeline/preprocessing/04_merge_datasets.py --default

当前检测到的环境:
  DATASETS_ROOT: {ProjectPaths.DATASETS_ROOT}
"""
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="输出目录（使用 --default 时可省略）"
    )
    parser.add_argument(
        "--sources",
        type=str,
        nargs="+",
        help="数据源列表，格式: 'name:path' 或 'path'（使用 --default 时可省略）"
    )
    parser.add_argument(
        "--no-prefix",
        action="store_true",
        help="不为 scene_id 添加数据源前缀"
    )
    parser.add_argument(
        "--default",
        action="store_true",
        help="使用默认配置（基于 config/paths.py 自动检测的环境）"
    )
    
    args = parser.parse_args()
    
    # 使用默认配置
    if args.default:
        output_dir = ProjectPaths.THREEDISH_AUTOREGRESSIVE_DIR
        sources = [
            ("3dfront", ProjectPaths.SSR_3DFRONT_AUTOREGRESSIVE_DIR),
            ("imaginarium", ProjectPaths.IMAGINARIUM_AUTOREGRESSIVE_DIR),
            ("spatiallm", ProjectPaths.SPATIALLM_AUTOREGRESSIVE_DIR),
            ("handcrafted", ProjectPaths.HANDCRAFTED_AUTOREGRESSIVE_DIR),
        ]
    else:
        if not args.output_dir or not args.sources:
            parser.error("必须指定 --output-dir 和 --sources，或使用 --default")
        output_dir = Path(args.output_dir)
        sources = [parse_source(s) for s in args.sources]
    
    print("=" * 60)
    print("数据集合并")
    print("=" * 60)
    print(f"环境: {ProjectPaths.DATASETS_ROOT}")
    print(f"输出目录: {output_dir}")
    print(f"数据源:")
    for name, path in sources:
        print(f"  - {name}: {path}")
    print(f"添加前缀: {not args.no_prefix}")
    print()
    
    merge_autoregressive_datasets(
        sources=sources,
        output_dir=output_dir,
        add_source_prefix=not args.no_prefix,
    )


if __name__ == "__main__":
    main()
