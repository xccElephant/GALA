"""
资产检索模块

根据语义描述和尺寸约束从资产库中检索匹配的资产。
支持：
1. 文本语义检索（基于 embedding 相似度）
2. 尺寸软打分
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from config.paths import ProjectPaths


# ---- 尺寸打分权重 ----
_SIZE_SCORE_CLOSE = 0.08      # 尺寸在 1x tolerance 内
_SIZE_SCORE_MODERATE = 0.03   # 尺寸在 2x tolerance 内
_SIZE_PENALTY_FAR = -0.02     # 尺寸在 3x tolerance 内
_SIZE_PENALTY_VERY_FAR = -0.08  # 尺寸超过 3x tolerance


def _compute_size_score(
    asset_size: Optional[List[float]],
    target_size: Optional[List[float]],
    tolerance: float,
) -> float:
    """基于尺寸相近程度返回软分数，避免前置硬过滤导致误杀。"""
    if not asset_size or not target_size or len(asset_size) != 3 or len(target_size) != 3:
        return 0.0

    valid_diffs = []
    for asset_dim, target_dim in zip(asset_size, target_size):
        if target_dim <= 0 or asset_dim <= 0:
            continue
        valid_diffs.append(abs(np.log(float(asset_dim) / float(target_dim))))

    if not valid_diffs:
        return 0.0

    mean_diff = float(np.mean(valid_diffs))
    if mean_diff <= np.log(1 + tolerance):
        return _SIZE_SCORE_CLOSE
    if mean_diff <= np.log(1 + tolerance * 2):
        return _SIZE_SCORE_MODERATE
    if mean_diff <= np.log(1 + tolerance * 3):
        return _SIZE_PENALTY_FAR
    return _SIZE_PENALTY_VERY_FAR


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _parse_size_field(value: Any) -> Optional[List[float]]:
    text = _clean_text(value)
    if not text:
        return None

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list) and len(parsed) == 3:
            size = [float(x) for x in parsed]
            if all(dim > 0 for dim in size):
                return size
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    try:
        size = [float(x) for x in text.split(",")]
        if len(size) == 3 and all(dim > 0 for dim in size):
            return size
    except ValueError:
        pass
    return None


class AssetIndex:
    """
    资产索引
    
    存储资产的元数据和 embedding，用于快速检索。
    """
    
    def __init__(self):
        self.assets: Dict[str, Dict[str, Any]] = {}  # jid -> asset info
        self.embeddings: Optional[np.ndarray] = None  # (N, D) embedding matrix, 调用 finalize() 后可用
        self.jid_list: List[str] = []  # jid 列表，与 embeddings 行对应
        self._embedding_buffer: List[np.ndarray] = []  # 构建期间暂存，finalize() 后清空
    
    def add_asset(
        self,
        jid: str,
        short_desc: str,
        size: List[float],
        category: str = "",
        description: str = "",
        embedding: Optional[np.ndarray] = None,
    ):
        """添加资产到索引（构建期间调用，最后需调用 finalize()）"""
        self.assets[jid] = {
            "jid": jid,
            "short_desc": short_desc,
            "size": size,
            "category": category,
            "description": description or short_desc,
        }
        
        if jid not in self.jid_list:
            self.jid_list.append(jid)
            if embedding is not None:
                self._embedding_buffer.append(embedding.reshape(1, -1))

    def finalize(self):
        """将暂存的 embedding 一次性合并为矩阵。构建完成后必须调用。"""
        if self._embedding_buffer:
            self.embeddings = np.vstack(self._embedding_buffer)
            self._embedding_buffer.clear()
    
    def get_asset(self, jid: str) -> Optional[Dict[str, Any]]:
        return self.assets.get(jid)
    
    def save(self, path: str):
        """保存索引到文件"""
        self.finalize()
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        assets_file = save_path.with_suffix(".json")
        with open(assets_file, "w", encoding="utf-8") as f:
            json.dump({
                "assets": self.assets,
                "jid_list": self.jid_list,
            }, f, ensure_ascii=False, indent=2)
        
        if self.embeddings is not None:
            embeddings_file = save_path.with_suffix(".npy")
            np.save(embeddings_file, self.embeddings)
        
        print(f"[AssetIndex] Saved {len(self.assets)} assets to {save_path}")
    
    def load(self, path: str):
        """从文件加载索引"""
        load_path = Path(path)
        
        assets_file = load_path.with_suffix(".json")
        with open(assets_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assets = data["assets"]
            self.jid_list = data["jid_list"]
        
        embeddings_file = load_path.with_suffix(".npy")
        if embeddings_file.exists():
            self.embeddings = np.load(embeddings_file)
        
        print(f"[AssetIndex] Loaded {len(self.assets)} assets from {load_path}")
    
    def __len__(self) -> int:
        return len(self.assets)
    
    def __contains__(self, jid: str) -> bool:
        return jid in self.assets


class AssetRetriever:
    """
    资产检索器
    
    根据语义描述和约束条件检索匹配的资产。
    """
    
    def __init__(
        self,
        index_path: Optional[str] = None,
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        device: Optional[str] = None,
        use_flash_attention: bool = False,
    ):
        """
        初始化检索器
        
        Args:
            index_path: 索引文件路径
            embedding_model: sentence-transformers 模型名称
            device: 计算设备（默认自动检测，优先使用 GPU）
            use_flash_attention: 是否使用 flash_attention_2 加速
        """
        self.index = AssetIndex()
        self.embedding_model_name = embedding_model
        
        # 自动检测设备：优先使用第一个可见的 GPU
        if device is None:
            import torch
            if torch.cuda.is_available():
                device = "cuda:0"  # 使用第一个可见的 GPU
            else:
                device = "cpu"
        self.device = device
        
        self.use_flash_attention = use_flash_attention
        self._model = None
        
        if index_path:
            self.index.load(index_path)
    
    def _get_embedding_model(self):
        """懒加载 embedding 模型"""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"[AssetRetriever] Loading embedding model: {self.embedding_model_name}")
            
            if self.use_flash_attention and "cuda" in self.device:
                import torch
                # 使用 flash_attention_2 加速，需要 GPU 支持
                # FlashAttention 只支持 fp16/bf16，且要求模型初始化时就在 GPU 上
                # 必须通过 device_map 指定，而非 device 参数（后者是先 CPU 加载再移动）
                self._model = SentenceTransformer(
                    self.embedding_model_name,
                    model_kwargs={
                        "attn_implementation": "flash_attention_2",
                        "torch_dtype": torch.bfloat16,
                        "device_map": self.device,
                    },
                    tokenizer_kwargs={"padding_side": "left"},
                )
            else:
                self._model = SentenceTransformer(
                    self.embedding_model_name,
                    device=self.device,
                )
        return self._model
    
    def encode_text(self, text: str, is_query: bool = True) -> np.ndarray:
        """
        编码文本为 embedding
        
        Args:
            text: 输入文本
            is_query: 是否为查询文本。查询文本使用 prompt_name="query"，
                      文档/资产描述则不使用 prompt。
        
        Returns:
            文本的 embedding 向量
        """
        model = self._get_embedding_model()
        
        if is_query:
            # 查询文本使用 query prompt
            return model.encode(text, prompt_name="query", convert_to_numpy=True)
        else:
            # 文档/资产描述不使用 prompt
            return model.encode(text, convert_to_numpy=True)
    
    def retrieve(
        self,
        description: str,
        category: Optional[str] = None,
        size_constraint: Optional[List[float]] = None,
        size_tolerance: float = 0.5,
        top_k: int = 5,
        min_score: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        检索匹配的资产
        
        Args:
            description: 物体描述（用于语义匹配）
            category: 类别名称（可选，拼入查询文本增强语义匹配）
            size_constraint: 尺寸约束 [width, depth, height]（可选）
            size_tolerance: 尺寸容差比例（默认 0.5，即 50%）
            top_k: 返回的最大结果数
            min_score: 最小相似度分数（低于此分数的结果仅在无更好候选时作为保底）
        
        Returns:
            匹配的资产列表，按相似度降序排列，至少返回 1 个结果（除非索引为空）
        """
        if len(self.index) == 0:
            print("[WARNING] Asset index is empty")
            return []

        if self.index.embeddings is None:
            raise ValueError("Asset index does not contain embeddings. Please rebuild the index.")

        query_text = f"{category} {description}".strip() if category else description
        query_embedding = self.encode_text(query_text, is_query=True)
        norms = np.linalg.norm(self.index.embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-8
        cosine_scores = self.index.embeddings @ query_embedding / norms

        # 语义相似度 + 尺寸奖惩
        all_scored: List[Dict[str, Any]] = []
        for i, jid in enumerate(self.index.jid_list):
            asset = self.index.assets[jid]
            score = float(cosine_scores[i])
            score += _compute_size_score(asset.get("size"), size_constraint, size_tolerance)
            all_scored.append({**asset, "score": score})

        all_scored.sort(key=lambda x: x["score"], reverse=True)

        results = [r for r in all_scored if r["score"] >= min_score]
        if results:
            return results[:top_k]

        if all_scored:
            fallback = all_scored[0]
            print(
                f"[WARNING] No asset above min_score={min_score} for '{description}', "
                f"using best match: {fallback.get('short_desc')} (score: {fallback['score']:.3f})"
            )
            return [fallback]

        return []

def build_asset_index_from_asset_info(
    asset_info_csv_path: str,
    asset_dir: str,
    output_path: str,
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
) -> AssetIndex:
    """
    从资产目录的 asset_info.csv 构建资产索引
    
    适用于：
    - Imaginarium 的完整 asset_info.csv
    - 3D-FRONT 的精简 asset_info.csv
    
    Args:
        asset_info_csv_path: 资产信息 CSV 文件路径
        asset_dir: 资产目录路径
        output_path: 输出索引文件路径
        embedding_model: embedding 模型名称
    
    Returns:
        构建好的资产索引
    """
    import csv
    
    index = AssetIndex()
    
    # 加载 embedding 模型
    from sentence_transformers import SentenceTransformer
    print(f"[build_asset_index] Loading embedding model: {embedding_model}")
    model = SentenceTransformer(embedding_model)
    
    print(f"[build_asset_index] Reading asset info from {asset_info_csv_path}")
    
    with open(asset_info_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # 从 CSV 提取信息
            jid = _clean_text(row.get("jid") or row.get("name_en"))
            if not jid:
                continue

            # 检查资产目录是否存在
            asset_path = Path(asset_dir) / jid
            if not asset_path.exists():
                continue

            row_short_desc = _clean_text(row.get("short_desc"))
            row_desc = _clean_text(row.get("desc") or row.get("caption_en"))
            row_category = _clean_text(row.get("category")).lower().replace("_", " ")
            size = _parse_size_field(row.get("size") or row.get("bbx"))

            # 尝试从 metadata.json 获取更精确的尺寸
            metadata_file = asset_path / f"{jid}_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, "r") as mf:
                    meta = json.load(mf)
                    if "transformed_size" in meta:
                        size = meta["transformed_size"]

            if size is None or len(size) != 3:
                raise ValueError(
                    f"Asset {jid} is missing valid size in asset_info and metadata: {asset_path}"
                )
            size = [max(float(d), 0.01) for d in size]

            category = row_category

            if not row_short_desc:
                raise ValueError(f"Asset {jid} is missing short_desc in asset_info CSV")
            short_desc = row_short_desc

            description = row_desc or short_desc

            embedding = model.encode(short_desc, convert_to_numpy=True)

            index.add_asset(
                jid=jid,
                short_desc=short_desc,
                size=size,
                category=category,
                description=description,
                embedding=embedding,
            )
    
    print(f"[build_asset_index] Indexed {len(index)} unique assets")
    
    # 保存索引
    index.save(output_path)
    
    return index


# ==================== 命令行工具 ====================

def main():
    """构建资产索引的命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="构建资产索引")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["imaginarium", "3dfront"],
        default="imaginarium",
        help="数据集类型",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出索引文件路径",
    )
    
    args = parser.parse_args()
    
    output_path = args.output or str(
        ProjectPaths.OUTPUT_DIR / f"asset_index_{args.dataset}"
    )

    if args.dataset == "3dfront":
        asset_info_csv = str(ProjectPaths.get_asset_info_csv_path("3dfront"))
        asset_dir = str(ProjectPaths.THREED_FUTURE_MODEL_DIR)
    else:
        asset_info_csv = str(ProjectPaths.get_asset_info_csv_path("imaginarium"))
        asset_dir = str(ProjectPaths.IMAGINARIUM_ASSETS_DIR)

    build_asset_index_from_asset_info(
        asset_info_csv_path=str(asset_info_csv),
        asset_dir=asset_dir,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()

# python agent/asset_retriever.py --dataset imaginarium
