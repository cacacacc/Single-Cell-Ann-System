"""vector_store.py — ChromaDB 向量数据库封装层

本模块将 ChromaDB 作为轻量级持久化向量数据库，
将单细胞数据（PCA 降维向量 + 元数据）写入 Collection，
并提供统一的向量检索接口供 RAG 模块调用。

架构角色
---------
DataLoader (data_reader.py)
    │  get_vectors / get_cell_info
    ▼
CellVectorStore (本模块)
    │  populate_from_loader()  ← 批量写入
    │  query_similar()         ← 向量相似检索
    │  query_by_text()         ← 元数据文本过滤
    ▼
ChromaDB  (持久化到 chroma_dir/)

使用示例
--------
    from backend.vector_store import CellVectorStore
    from backend.data_reader import DataLoader

    store = CellVectorStore(collection_name="liver", persist_dir="chroma_db")
    loader = DataLoader("data/liver.h5ad")
    store.populate_from_loader(loader, use_rep="X_pca", batch_size=512)

    results = store.query_similar(query_vector, n_results=5)
    for r in results:
        print(r["cell_type"], r["distance"])
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    import chromadb  # type: ignore
    from chromadb.config import Settings  # type: ignore
    _CHROMA_AVAILABLE = True
except ImportError:
    chromadb = None  # type: ignore
    Settings = None  # type: ignore
    _CHROMA_AVAILABLE = False

logger = logging.getLogger(__name__)

# ChromaDB 默认持久化目录（相对于项目根）
DEFAULT_CHROMA_DIR = "chroma_db"

# 单次批量写入的最大条数（ChromaDB 建议 <= 5461，保守取 512）
DEFAULT_BATCH_SIZE = 512


def is_chroma_available() -> bool:
    """返回 ChromaDB 是否已安装。"""
    return _CHROMA_AVAILABLE


# ---------------------------------------------------------------------------
# 核心封装类
# ---------------------------------------------------------------------------

class CellVectorStore:
    """ChromaDB 单细胞向量数据库封装。

    Parameters
    ----------
    collection_name:
        ChromaDB Collection 名称，通常对应数据集名称（如 ``"liver"``）。
    persist_dir:
        ChromaDB 持久化目录路径。若为相对路径，相对于项目根目录。
    distance_metric:
        向量距离函数，支持 ``"cosine"``（余弦距离）、``"l2"``（欧氏距离）、
        ``"ip"``（内积，越大越相似）。默认 ``"cosine"``。
    """

    def __init__(
        self,
        collection_name: str,
        persist_dir: Union[str, Path] = DEFAULT_CHROMA_DIR,
        distance_metric: str = "cosine",
    ) -> None:
        if not _CHROMA_AVAILABLE:
            raise ImportError(
                "chromadb 未安装，请执行: pip install chromadb"
            )

        self._collection_name = collection_name
        self._persist_dir = Path(persist_dir).resolve()
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        _valid_metrics = {"cosine", "l2", "ip"}
        if distance_metric not in _valid_metrics:
            raise ValueError(
                f"distance_metric 必须是 {_valid_metrics} 之一，得到：{distance_metric}"
            )
        self._distance_metric = distance_metric

        # 初始化 ChromaDB 客户端（持久化模式）
        self._client = chromadb.PersistentClient(
            path=str(self._persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        # 获取或创建 Collection
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": self._distance_metric},
        )

        logger.info(
            "CellVectorStore 初始化完成: collection=%s, dir=%s, metric=%s, "
            "已有文档数=%d",
            self._collection_name,
            self._persist_dir,
            self._distance_metric,
            self._collection.count(),
        )

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def persist_dir(self) -> Path:
        return self._persist_dir

    @property
    def distance_metric(self) -> str:
        return self._distance_metric

    def count(self) -> int:
        """返回 Collection 中已存储的向量条数。"""
        return int(self._collection.count())

    def is_populated(self) -> bool:
        """返回 Collection 是否已有数据（用于跳过重复写入）。"""
        return self.count() > 0

    def populate_from_loader(
        self,
        loader: Any,  # DataLoader，避免循环导入故用 Any
        use_rep: Optional[str] = "X_pca",
        batch_size: int = DEFAULT_BATCH_SIZE,
        force: bool = False,
        obs_fields: Optional[List[str]] = None,
        top_genes: int = 20,
    ) -> int:
        """将 DataLoader 中的细胞数据批量写入 ChromaDB。

        Parameters
        ----------
        loader:
            ``DataLoader`` 实例（来自 ``data_reader.py``）。
        use_rep:
            向量表示键，默认 ``"X_pca"``（50 维 PCA 降维结果）。
            若数据集中不存在该表示，自动回退到 ``"X"``（原始基因表达矩阵，
            维度较高，可能较慢）。
        batch_size:
            每批次写入的最大细胞数，默认 512。
        force:
            ``True`` 时即使 Collection 已有数据也强制重写（会先清空）。
        obs_fields:
            需要写入 ChromaDB 元数据的 obs 字段列表。
            ``None`` 表示写入全部 obs 字段（字符串化）。
        top_genes:
            额外在元数据中记录表达量最高的前 N 个基因名称，
            默认 20，用于后续 Prompt 组装。

        Returns
        -------
        int
            写入的细胞总数。
        """
        if self.is_populated() and not force:
            logger.info(
                "Collection '%s' 已有 %d 条数据，跳过写入（传入 force=True 可强制重写）",
                self._collection_name,
                self.count(),
            )
            return self.count()

        if force and self.is_populated():
            logger.info("force=True，清空 Collection '%s'...", self._collection_name)
            self._client.delete_collection(self._collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={
                    "hnsw:space": self._distance_metric,
                    "use_rep": str(use_rep or "X_pca"),
                },
            )

        # ------ 确定向量表示 ------
        actual_rep = self._resolve_use_rep(loader, use_rep)
        vectors = loader.get_vectors(actual_rep)  # (n_cells, dim)
        n_cells = vectors.shape[0]

        # ------ 提取高表达基因（用于 Prompt 元数据）------
        gene_names = self._get_gene_names(loader)
        has_gene_info = gene_names is not None and len(gene_names) > 0

        # ------ 确定要写入的 obs 字段 ------
        all_obs_cols = loader.obs_columns
        if obs_fields is None:
            write_fields = all_obs_cols
        else:
            write_fields = [f for f in obs_fields if f in all_obs_cols]

        logger.info(
            "开始写入 %d 个细胞到 ChromaDB (use_rep=%s, batch_size=%d)...",
            n_cells,
            actual_rep,
            batch_size,
        )

        total_written = 0
        n_batches = (n_cells + batch_size - 1) // batch_size
        for batch_idx, batch_start in enumerate(range(0, n_cells, batch_size)):
            batch_end = min(batch_start + batch_size, n_cells)
            batch_vectors = vectors[batch_start:batch_end]
            batch_top_genes: List[List[str]] = [[] for _ in range(batch_end - batch_start)]
            if has_gene_info:
                try:
                    raw_block = loader.get_X_block(batch_start, batch_end)
                    batch_top_genes = self._top_expressed_genes_block(
                        raw_block,
                        gene_names,
                        top_n=top_genes,
                    )
                except Exception as exc:
                    logger.warning(
                        "批次 %d/%d 高表达基因计算失败，top_genes 将置空: %s",
                        batch_idx + 1,
                        n_batches,
                        exc,
                    )

            ids = []
            embeddings = []
            metadatas = []
            documents = []

            for local_idx in range(batch_end - batch_start):
                global_idx = batch_start + local_idx
                cell_info = loader.get_cell_info(global_idx)
                cell_id = cell_info.get("cell_id", str(global_idx))

                # 构建元数据字典（仅保留可序列化的标量字段）
                meta: Dict[str, Any] = {
                    "cell_index": global_idx,
                    "cell_id": cell_id,
                    "use_rep": actual_rep,
                    "dataset": self._collection_name,
                }
                for field in write_fields:
                    val = cell_info.get(field)
                    if val is None:
                        meta[field] = ""
                    elif isinstance(val, (int, float, bool, str)):
                        meta[field] = val
                    else:
                        meta[field] = str(val)

                # 高表达基因列表（字符串形式存入元数据）
                meta["top_genes"] = ",".join(batch_top_genes[local_idx])

                # ChromaDB document 字段（可作为文本内容用于混合检索）
                doc_text = self._build_document_text(cell_info, meta.get("top_genes", ""))

                ids.append(cell_id)
                embeddings.append(batch_vectors[local_idx].tolist())
                metadatas.append(meta)
                documents.append(doc_text)

            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
            )
            total_written += len(ids)
            pct = round(total_written / n_cells * 100)
            logger.info(
                "  批次 %d/%d — 已写入 %d / %d 个细胞（%d%%）",
                batch_idx + 1, n_batches, total_written, n_cells, pct,
            )

        logger.info(
            "写入完成：共 %d 个细胞写入 Collection '%s'（use_rep=%s）",
            total_written,
            self._collection_name,
            actual_rep,
        )
        return total_written

    def query_similar(
        self,
        query_vector: Union[np.ndarray, List[float]],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """按向量相似度检索最近邻细胞。

        Parameters
        ----------
        query_vector:
            查询向量，形状 ``(dim,)``，维度须与写入时一致。
        n_results:
            返回结果数量，默认 10。
        where:
            ChromaDB 元数据过滤条件，例如 ``{"cell_type": {"$eq": "Hepatocyte"}}``。

        Returns
        -------
        list of dict
            每个 dict 包含 ``cell_id``、``cell_index``、``cell_type``、
            ``distance``、``top_genes``、``document``、``metadata`` 等字段。
        """
        if self.count() == 0:
            raise RuntimeError(
                f"Collection '{self._collection_name}' 尚未写入数据，"
                "请先调用 populate_from_loader()"
            )

        vec = _to_float_list(query_vector)
        n_results = min(n_results, self.count())

        query_kwargs: Dict[str, Any] = {
            "query_embeddings": [vec],
            "n_results": n_results,
            "include": ["metadatas", "documents", "distances"],
        }
        if where:
            query_kwargs["where"] = where

        raw = self._collection.query(**query_kwargs)

        results: List[Dict[str, Any]] = []
        ids_list = raw.get("ids", [[]])[0]
        distances_list = raw.get("distances", [[]])[0]
        metadatas_list = raw.get("metadatas", [[]])[0]
        documents_list = raw.get("documents", [[]])[0]

        for rank, (chroma_id, dist, meta, doc) in enumerate(
            zip(ids_list, distances_list, metadatas_list, documents_list), start=1
        ):
            results.append(
                {
                    "rank": rank,
                    "cell_id": chroma_id,
                    "cell_index": meta.get("cell_index", -1),
                    "cell_type": meta.get("cell_type", "unknown"),
                    "distance": float(dist),
                    "top_genes": meta.get("top_genes", ""),
                    "document": doc,
                    "metadata": meta,
                }
            )
        return results

    def query_by_metadata(
        self,
        where: Dict[str, Any],
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """纯元数据过滤查询（不使用向量，仅按字段筛选）。

        Parameters
        ----------
        where:
            ChromaDB ``where`` 过滤条件字典。
            例如 ``{"cell_type": {"$eq": "T cell"}}``。
        limit:
            最多返回条数，默认 20。

        Returns
        -------
        list of dict
        """
        get_kwargs: Dict[str, Any] = {
            "limit": limit,
            "include": ["metadatas", "documents"],
        }
        if where:  # ChromaDB 不接受空 where={}
            get_kwargs["where"] = where
        raw = self._collection.get(**get_kwargs)
        ids_list = raw.get("ids", [])
        metadatas_list = raw.get("metadatas", [])
        documents_list = raw.get("documents", [])

        results: List[Dict[str, Any]] = []
        for chroma_id, meta, doc in zip(ids_list, metadatas_list, documents_list):
            results.append(
                {
                    "cell_id": chroma_id,
                    "cell_index": meta.get("cell_index", -1),
                    "cell_type": meta.get("cell_type", "unknown"),
                    "top_genes": meta.get("top_genes", ""),
                    "document": doc,
                    "metadata": meta,
                }
            )
        return results

    def get_by_cell_ids(self, cell_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch stored ChromaDB records by cell ID.

        This is used to enrich deterministic metadata query results with fields
        that only exist in the vector store, such as ``top_genes`` and
        ``document``.
        """
        ids = [str(cell_id) for cell_id in cell_ids if str(cell_id).strip()]
        if not ids or self.count() == 0:
            return {}

        raw = self._collection.get(
            ids=ids,
            include=["metadatas", "documents"],
        )
        found_ids = raw.get("ids", [])
        metadatas = raw.get("metadatas", [])
        documents = raw.get("documents", [])

        records: Dict[str, Dict[str, Any]] = {}
        for chroma_id, meta, doc in zip(found_ids, metadatas, documents):
            meta = meta or {}
            records[str(chroma_id)] = {
                "cell_id": chroma_id,
                "cell_index": meta.get("cell_index", -1),
                "cell_type": meta.get("cell_type", "unknown"),
                "top_genes": meta.get("top_genes", ""),
                "document": doc,
                "metadata": meta,
            }
        return records

    def query_by_keywords(
        self,
        keywords: List[str],
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """基于关键词搜索细胞（用于 RAG 自然语言问题）。

        在 ChromaDB documents 字段和 top_genes 元数据中匹配关键词，
        按命中数降序排列后返回 top-N。

        Parameters
        ----------
        keywords:
            待匹配的关键词列表（如基因名、细胞类型等）。
        n_results:
            最多返回的细胞数量。

        Returns
        -------
        list of dict
            与 ``query_similar()`` 返回格式一致。
        """
        if not keywords:
            return []

        if self.count() == 0:
            return []

        # 获取所有细胞的 documents 和 metadatas（ChromaDB get 可批量拉取）
        total = self.count()
        # 分批获取全部数据（ChromaDB get 默认 limit 可能较小）
        batch_size = 1000
        all_ids: List[str] = []
        all_metas: List[Dict[str, Any]] = []
        all_docs: List[str] = []

        for offset in range(0, total, batch_size):
            raw = self._collection.get(
                offset=offset,
                limit=min(batch_size, total - offset),
                include=["metadatas", "documents"],
            )
            all_ids.extend(raw.get("ids", []))
            all_metas.extend(raw.get("metadatas", []))
            all_docs.extend(raw.get("documents", []))

        # 对每个细胞计算关键词命中数
        keywords_lower = [k.lower() for k in keywords]
        scored: List[tuple] = []  # (score, cell_id, meta, doc, search_text)
        for cid, meta, doc in zip(all_ids, all_metas, all_docs):
            meta = meta or {}
            search_text = (
                (doc or "").lower() + " " + (meta.get("top_genes", "") or "").lower()
            )
            score = sum(1 for kw in keywords_lower if kw in search_text)
            if score > 0:
                scored.append((score, cid, meta, doc, search_text))

        # 按命中数降序
        scored.sort(key=lambda x: -x[0])
        top = scored[:n_results]

        results: List[Dict[str, Any]] = []
        for rank, (score, cid, meta, doc, search_text) in enumerate(top, start=1):
            results.append(
                {
                    "rank": rank,
                    "cell_id": cid,
                    "cell_index": meta.get("cell_index", -1),
                    "cell_type": meta.get("cell_type", "unknown"),
                    "score": int(score),
                    "top_genes": meta.get("top_genes", ""),
                    "document": doc,
                    "metadata": meta,
                    "match_reasons": [
                        f"keyword={kw}" for kw in keywords_lower if kw in search_text
                    ],
                }
            )
        return results

    def get_collection_info(self) -> Dict[str, Any]:
        """返回 Collection 的基础信息，用于状态接口。"""
        # 尝试从 Collection 元数据中读取写入时记录的 use_rep
        try:
            col_meta = self._collection.metadata or {}
            use_rep_recorded = col_meta.get("use_rep", "unknown")
        except Exception:
            use_rep_recorded = "unknown"

        top_genes_nonempty = 0
        try:
            if self.count() > 0:
                sample = self._collection.get(limit=min(self.count(), 1000), include=["metadatas"])
                for meta in sample.get("metadatas", []):
                    if meta and str(meta.get("top_genes") or "").strip():
                        top_genes_nonempty += 1
        except Exception:
            top_genes_nonempty = 0

        return {
            "collection_name": self._collection_name,
            "persist_dir": str(self._persist_dir),
            "distance_metric": self._distance_metric,
            "count": self.count(),
            "is_populated": self.is_populated(),
            "chroma_available": _CHROMA_AVAILABLE,
            "use_rep": use_rep_recorded,
            "top_genes_sampled": min(self.count(), 1000),
            "top_genes_nonempty_sampled": top_genes_nonempty,
            "top_genes_available": top_genes_nonempty > 0,
        }

    def delete_collection(self) -> None:
        """删除当前 Collection（不可逆）。"""
        self._client.delete_collection(self._collection_name)
        # 重新创建空 collection
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": self._distance_metric},
        )
        logger.info("Collection '%s' 已删除并重建（空）", self._collection_name)

    # ------------------------------------------------------------------
    # 内部工具方法
    # ------------------------------------------------------------------

    def _resolve_use_rep(self, loader: Any, use_rep: Optional[str]) -> str:
        """确定实际使用的向量表示，自动回退。"""
        if use_rep is None or use_rep == "X":
            return "X"
        available = loader.available_reps
        if use_rep in available:
            return use_rep
        # 优先尝试 X_pca，其次 X_umap，最终退回 X
        for fallback in ("X_pca", "X_umap", "X"):
            if fallback in available or fallback == "X":
                logger.warning(
                    "use_rep '%s' 不在数据集中（可用: %s），自动回退到 '%s'",
                    use_rep,
                    available,
                    fallback,
                )
                return fallback
        return "X"

    def _get_gene_names(self, loader: Any) -> Optional[List[str]]:
        """从 AnnData 获取基因名称列表，用于高表达基因标注。"""
        try:
            if hasattr(loader, "get_gene_names"):
                return list(loader.get_gene_names())
            adata = loader.adata
            if hasattr(adata, "var") and "feature_name" in adata.var.columns:
                return list(adata.var["feature_name"].astype(str))
            return list(adata.var_names)
        except Exception:
            return None

    def _top_expressed_genes(
        self,
        expression_vector: np.ndarray,
        gene_names: List[str],
        top_n: int = 20,
    ) -> List[str]:
        """返回表达量最高的前 top_n 个基因名称。"""
        arr = np.asarray(expression_vector, dtype=np.float32).flatten()
        if arr.shape[0] != len(gene_names):
            return []
        top_n = min(top_n, len(gene_names))
        top_indices = np.argpartition(arr, -top_n)[-top_n:]
        top_indices = top_indices[np.argsort(arr[top_indices])[::-1]]
        return [gene_names[int(i)] for i in top_indices]

    def _top_expressed_genes_block(
        self,
        expression_block: np.ndarray,
        gene_names: List[str],
        top_n: int = 20,
    ) -> List[List[str]]:
        """Return top expressed genes for each row in a dense expression block."""
        arr = np.asarray(expression_block, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != len(gene_names) or arr.shape[0] == 0:
            return [[] for _ in range(arr.shape[0] if arr.ndim >= 1 else 0)]

        top_n = max(0, min(int(top_n), arr.shape[1]))
        if top_n == 0:
            return [[] for _ in range(arr.shape[0])]

        top_indices = np.argpartition(arr, -top_n, axis=1)[:, -top_n:]
        row_values = np.take_along_axis(arr, top_indices, axis=1)
        order = np.argsort(row_values, axis=1)[:, ::-1]
        sorted_indices = np.take_along_axis(top_indices, order, axis=1)

        results: List[List[str]] = []
        for row_idx, row_indices in enumerate(sorted_indices):
            values = arr[row_idx, row_indices]
            genes = [
                gene_names[int(gene_idx)]
                for gene_idx, value in zip(row_indices, values)
                if float(value) > 0.0
            ]
            results.append(genes)
        return results

    def _build_document_text(
        self,
        cell_info: Dict[str, Any],
        top_genes_str: str,
    ) -> str:
        """构建存入 ChromaDB documents 字段的纯文本描述，便于后续文本检索。

        格式示例：
            Cell type: Hepatocyte | Tissue: liver | Top genes: ALB, APOA1, CYP3A4, ...
        """
        parts: List[str] = []
        cell_type = cell_info.get("cell_type")
        if cell_type:
            parts.append(f"Cell type: {cell_type}")
        for key in ("tissue", "organ", "sample", "donor", "condition", "leiden", "louvain"):
            val = cell_info.get(key)
            if val is not None and str(val).strip():
                parts.append(f"{key.capitalize()}: {val}")
        if top_genes_str:
            parts.append(f"Top genes: {top_genes_str}")
        return " | ".join(parts) if parts else f"Cell ID: {cell_info.get('cell_id', 'unknown')}"


# ---------------------------------------------------------------------------
# 全局单例管理（供 Flask app 使用）
# ---------------------------------------------------------------------------

_STORE_REGISTRY: Dict[str, CellVectorStore] = {}


def get_or_create_store(
    collection_name: str,
    persist_dir: Union[str, Path] = DEFAULT_CHROMA_DIR,
    distance_metric: str = "cosine",
) -> CellVectorStore:
    """获取或创建指定 collection_name 的 CellVectorStore 单例。

    在 Flask 进程生命周期内复用同一 ChromaDB 客户端和 Collection 连接，
    避免重复初始化开销。

    Parameters
    ----------
    collection_name:
        Collection 名称（通常为 dataset_id 或数据集显示名）。
    persist_dir:
        ChromaDB 持久化目录。
    distance_metric:
        向量距离度量。

    Returns
    -------
    CellVectorStore
    """
    key = f"{persist_dir}::{collection_name}"
    if key not in _STORE_REGISTRY:
        _STORE_REGISTRY[key] = CellVectorStore(
            collection_name=collection_name,
            persist_dir=persist_dir,
            distance_metric=distance_metric,
        )
    return _STORE_REGISTRY[key]


def clear_store_registry() -> None:
    """清空全局单例注册表（主要用于测试场景）。"""
    _STORE_REGISTRY.clear()


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _to_float_list(vec: Union[np.ndarray, List[float]]) -> List[float]:
    """将向量转为 Python float 列表（ChromaDB API 要求）。"""
    if isinstance(vec, np.ndarray):
        return vec.flatten().astype(np.float64).tolist()
    return [float(v) for v in vec]
