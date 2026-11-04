import os
from pathlib import Path


def _detect_datasets_root() -> Path:
    """数据集根目录默认为 项目根/datasets，也可由环境变量 DATASETS_ROOT 指定。"""
    env_path = os.environ.get("DATASETS_ROOT")
    if env_path:
        return Path(env_path).expanduser().resolve()

    return Path(__file__).parent.parent.resolve() / "datasets"


class ProjectPaths:
    """项目路径管理类"""
    
    # ==================== 项目根目录 ====================
    PROJECT_ROOT = Path(__file__).parent.parent.resolve()
    
    # ==================== 数据集路径（外部数据） ====================
    DATASETS_ROOT = _detect_datasets_root()
    
    # 3D-FRONT 数据集
    THREED_FRONT_DIR = DATASETS_ROOT / "3D-FRONT" / "3D-FRONT"
    THREED_FUTURE_MODEL_DIR = DATASETS_ROOT / "3D-FRONT" / "3D-FUTURE-model"
    THREED_FRONT_ASSET_INFO_CSV = THREED_FUTURE_MODEL_DIR / "3dfront_asset_info.csv"
    
    # SSR-3DFRONT 数据集
    SSR_3DFRONT_DIR = DATASETS_ROOT / "SSR-3DFRONT"
    SSR_3DFRONT_AUTOREGRESSIVE_DIR = SSR_3DFRONT_DIR / "autoregressive_data"
    SSR_3DFRONT_CONVERTED_SCENES_DIR = SSR_3DFRONT_DIR / "converted_scenes"
    SSR_3DFRONT_FILTERED_SCENES_DIR = SSR_3DFRONT_DIR / "filtered_scenes"

    # Imaginarium 数据集
    IMAGINARIUM_DIR = DATASETS_ROOT / "Imaginarium"
    IMAGINARIUM_ASSETS_DIR = IMAGINARIUM_DIR / "imaginarium_assets"
    IMAGINARIUM_ASSET_INFO_CSV = IMAGINARIUM_ASSETS_DIR / "imaginarium_asset_info.csv"
    IMAGINARIUM_AUTOREGRESSIVE_DIR = IMAGINARIUM_DIR / "autoregressive_data"
    IMAGINARIUM_FILTERED_SCENES_DIR = IMAGINARIUM_DIR / "filtered_scenes"
    IMAGINARIUM_INTERNAL_SPACES_DIR = IMAGINARIUM_DIR / "imaginarium_assets_internal_placement_space"

    # SpatialLM 数据集（场景来自 SpatialLM，资产来自 Imaginarium）
    SPATIALLM_DIR = DATASETS_ROOT / "spatiallm"
    SPATIALLM_CONVERTED_SCENES_DIR = SPATIALLM_DIR / "converted_scenes"
    SPATIALLM_FILTERED_SCENES_DIR = SPATIALLM_DIR / "filtered_scenes"
    SPATIALLM_AUTOREGRESSIVE_DIR = SPATIALLM_DIR / "autoregressive_data"

    # handcrafted 数据集（手工制作，资产混合 3D-FRONT + Imaginarium）
    HANDCRAFTED_DIR = DATASETS_ROOT / "handcrafted"
    HANDCRAFTED_CONVERTED_SCENES_DIR = HANDCRAFTED_DIR / "converted_scenes"
    HANDCRAFTED_FILTERED_SCENES_DIR = HANDCRAFTED_DIR / "filtered_scenes"
    HANDCRAFTED_AUTOREGRESSIVE_DIR = HANDCRAFTED_DIR / "autoregressive_data"

    # 3dish 混合数据集（3D-FRONT + Imaginarium + SpatialLM + handcrafted 合并）
    THREEDISH_DIR = DATASETS_ROOT / "3dish"
    THREEDISH_AUTOREGRESSIVE_DIR = THREEDISH_DIR / "autoregressive"

    # 处理后的数据
    DATA_PROCESSED_DIR = DATASETS_ROOT / "processed"
    DATA_POINTCLOUDS_DIR = DATASETS_ROOT / "pointclouds"

    # ==================== 项目内部目录 ====================
    DATASET_PIPELINE_DIR = PROJECT_ROOT / "dataset_pipeline"
    INPUT_DIR = PROJECT_ROOT / "input"
    OUTPUT_DIR = PROJECT_ROOT / "output"
    
    # ==================== 辅助方法 ====================

    @classmethod
    def get_autoregressive_dir(cls, dataset: str) -> Path:
        """
        获取数据集的 autoregressive 数据目录
        
        Args:
            dataset: 数据集名称
        """
        mapping = {
            "3dfront": cls.SSR_3DFRONT_AUTOREGRESSIVE_DIR,
            "imaginarium": cls.IMAGINARIUM_AUTOREGRESSIVE_DIR,
            "spatiallm": cls.SPATIALLM_AUTOREGRESSIVE_DIR,
            "handcrafted": cls.HANDCRAFTED_AUTOREGRESSIVE_DIR,
            "3dish": cls.THREEDISH_AUTOREGRESSIVE_DIR,
        }
        dataset = dataset.lower()
        if dataset not in mapping:
            raise ValueError(
                f"未知的数据集: {dataset}，支持: {', '.join(mapping)}"
            )
        return mapping[dataset]

    @classmethod
    def get_converted_scene_dir(cls, dataset: str) -> Path:
        """获取数据集的 converted_scenes 目录。"""
        mapping = {
            "3dfront": cls.SSR_3DFRONT_CONVERTED_SCENES_DIR,
            "imaginarium": cls.IMAGINARIUM_DIR / "converted_scenes",
            "spatiallm": cls.SPATIALLM_CONVERTED_SCENES_DIR,
            "handcrafted": cls.HANDCRAFTED_CONVERTED_SCENES_DIR,
        }
        dataset = dataset.lower()
        if dataset not in mapping:
            raise ValueError(
                f"未知的数据集: {dataset}，支持: {', '.join(mapping)}"
            )
        return mapping[dataset]

    @classmethod
    def get_filtered_scene_dir(cls, dataset: str) -> Path:
        """获取数据集的过滤后场景目录。"""
        mapping = {
            "3dfront": cls.SSR_3DFRONT_FILTERED_SCENES_DIR,
            "imaginarium": cls.IMAGINARIUM_FILTERED_SCENES_DIR,
            "spatiallm": cls.SPATIALLM_FILTERED_SCENES_DIR,
            "handcrafted": cls.HANDCRAFTED_FILTERED_SCENES_DIR,
        }
        dataset = dataset.lower()
        if dataset not in mapping:
            raise ValueError(
                f"未知的数据集: {dataset}，支持: {', '.join(mapping)}"
            )
        return mapping[dataset]

    @classmethod
    def ensure_dirs(cls):
        """确保所有必要的目录存在"""
        dirs = [
            cls.DATA_PROCESSED_DIR,
            cls.DATA_POINTCLOUDS_DIR,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_training_data_path(cls, dataset: str = "3dfront") -> Path:
        """
        获取训练数据文件路径
        
        Args:
            dataset: 数据集名称
        """
        dataset = dataset.lower()
        return cls.DATA_PROCESSED_DIR / f"{dataset}.json"
    
    @classmethod
    def get_metadata_path(cls, dataset: str = "3dfront") -> Path:
        """
        获取元数据文件路径
        
        Args:
            dataset: 数据集名称
        """
        dataset = dataset.lower()
        return cls.DATA_PROCESSED_DIR / f"{dataset}_metadata.csv"
    
    @classmethod
    def get_asset_pointcloud_dir(cls, dataset: str = "3dfront") -> Path | list[Path]:
        """
        获取资产点云数据目录
        
        Args:
            dataset: 数据集名称
            
        Returns:
            单个 Path 或 Path 列表（对于融合数据集如 3dima）
        """
        dataset = dataset.lower()
        if dataset == "3dfront":
            return cls.THREED_FUTURE_MODEL_DIR
        elif dataset in ("imaginarium", "spatiallm"):
            return cls.IMAGINARIUM_ASSETS_DIR
        elif dataset in ("3dish", "handcrafted"):
            # 融合数据集，资产来自两个目录
            # 优先搜索 Imaginarium 资产，再搜索 3D-FUTURE 资产
            return [cls.IMAGINARIUM_ASSETS_DIR, cls.THREED_FUTURE_MODEL_DIR]
        else:
            raise ValueError(f"未知的数据集: {dataset}")

    @classmethod
    def get_asset_info_csv_path(cls, dataset: str) -> Path:
        """获取资产信息 CSV 路径"""
        dataset = dataset.lower()
        mapping = {
            "3dfront": cls.THREED_FRONT_ASSET_INFO_CSV,
            "imaginarium": cls.IMAGINARIUM_ASSET_INFO_CSV,
        }
        if dataset not in mapping:
            raise ValueError(f"未知的数据集: {dataset}")
        return mapping[dataset]

    @classmethod
    def get_separate_training_pointcloud_dir(cls, dataset: str = "3dfront") -> Path:
        """
        获取分离点云训练用数据目录
        Args:
            dataset: 数据集名称

        Returns:
            分离点云训练用数据目录
        """
        dataset = dataset.lower()
        return cls.DATA_POINTCLOUDS_DIR / "separate_pointcloud" / dataset
    
    @classmethod
    def validate_external_datasets(cls, required: list[str] = None, verbose: bool = True) -> dict:
        """
        验证外部数据集是否存在
        
        Args:
            required: 必须存在的数据集名称列表（如 ["Imaginarium", "Imaginarium assets"]）
                      如果指定且某个数据集不存在，会打印错误信息
            verbose: 是否打印验证结果
        
        Returns:
            status: {dataset_name: exists}
        """
        status = {
            "3D-FRONT": cls.THREED_FRONT_DIR.exists(),
            "3D-FUTURE-model": cls.THREED_FUTURE_MODEL_DIR.exists(),
            "SSR-3DFRONT": cls.SSR_3DFRONT_DIR.exists(),
            "Imaginarium": cls.IMAGINARIUM_DIR.exists(),
            "Imaginarium assets": cls.IMAGINARIUM_ASSETS_DIR.exists(),
            "SpatialLM": cls.SPATIALLM_DIR.exists(),
            "handcrafted": cls.HANDCRAFTED_DIR.exists(),
        }
        
        if verbose:
            print("数据集状态:")
            for name, exists in status.items():
                mark = "✓" if exists else "✗"
                print(f"  {mark} {name}")
            print()
        
        # 检查必需的数据集
        if required:
            missing = [name for name in required if not status.get(name, False)]
            if missing:
                print(f"错误: 以下必需的数据集不存在: {', '.join(missing)}")
                return None
        
        return status

    
    @classmethod
    def print_paths(cls):
        """打印所有路径信息"""
        from_env = os.environ.get("DATASETS_ROOT") is not None
        
        print("=" * 70)
        source = "环境变量 DATASETS_ROOT" if from_env else "仓库内 datasets/ 占位目录"
        print(f"【数据根目录来源】 {source}")
        print("=" * 70)
        print(f"PROJECT_ROOT:    {cls.PROJECT_ROOT}")
        print(f"DATASETS_ROOT:   {cls.DATASETS_ROOT}")
        print("-" * 70)
        print(f"3D-FRONT:        {cls.THREED_FRONT_DIR}")
        print(f"3D-FUTURE-model: {cls.THREED_FUTURE_MODEL_DIR}")
        print(f"SSR-3DFRONT:     {cls.SSR_3DFRONT_DIR}")
        print(f"Imaginarium:     {cls.IMAGINARIUM_DIR}")
        print(f"SpatialLM:       {cls.SPATIALLM_DIR}")
        print(f"3dish:           {cls.THREEDISH_DIR}")
        print(f"handcrafted:     {cls.HANDCRAFTED_DIR}")
        print("=" * 70)


if __name__ == "__main__":
    ProjectPaths.print_paths()
    ProjectPaths.validate_external_datasets(verbose=True)
