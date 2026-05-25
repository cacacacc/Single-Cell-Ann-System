from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData


class DataLoader:
    """从 .h5ad 单细胞数据文件中加载向量矩阵与细胞元数据。

    使用方式::

        loader = DataLoader("data/liver.h5ad")
        vectors = loader.get_vectors()          # shape: (n_cells, n_dims), float32
        info = loader.get_cell_info(0)          # dict，包含该细胞的所有 obs 字段
    """

    def __init__(self, file_path: Union[str, Path]) -> None:
        """加载 .h5ad 文件。

        Parameters
        ----------
        file_path:
            h5ad 文件路径。
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"数据文件不存在：{path}")
        if path.suffix.lower() != ".h5ad":
            raise ValueError(f"文件格式不支持，需要 .h5ad 文件，得到：{path.suffix}")

        self._adata: AnnData = sc.read_h5ad(str(path))
        self._file_path = path
        self._cell_id_to_index: Optional[Dict[str, int]] = None

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def get_vectors(self, use_rep: Optional[str] = None) -> np.ndarray:
        """提取细胞向量矩阵，返回 float32 的 2D NumPy 数组。

        Parameters
        ----------
        use_rep:
            指定使用哪种表示：
            - ``None`` 或 ``"X"``：使用原始基因表达矩阵 ``adata.X``（默认）。
            - ``"X_pca"``、``"X_umap"`` 等：使用 ``adata.obsm`` 中对应的降维结果。

        Returns
        -------
        np.ndarray
            形状 ``(n_cells, n_dims)``，数据类型 ``float32``。
        """
        if use_rep is None or use_rep == "X":
            raw = self._adata.X
            # X 可能是稀疏矩阵，统一转为稠密数组
            if hasattr(raw, "toarray"):
                arr = raw.toarray()
            else:
                arr = np.asarray(raw)
        else:
            if use_rep not in self._adata.obsm:
                available = list(self._adata.obsm.keys())
                raise KeyError(
                    f"obsm 中不存在 '{use_rep}'，可用的表示为：{available}"
                )
            arr = np.asarray(self._adata.obsm[use_rep])

        return np.ascontiguousarray(arr, dtype=np.float32)

    def get_vector(self, cell_index: int, use_rep: Optional[str] = None) -> np.ndarray:
        """获取单个细胞的向量，用于传入 ANNIndexer.search()。

        Parameters
        ----------
        cell_index:
            细胞在数据集中的整数索引（从 0 开始）。
        use_rep:
            同 ``get_vectors()``，指定向量来源，默认使用 ``X``。

        Returns
        -------
        np.ndarray
            形状 ``(n_dims,)``，数据类型 ``float32``。
        """
        n_cells = self._adata.n_obs
        if not isinstance(cell_index, (int, np.integer)):
            raise TypeError(f"cell_index 必须是整数，得到 {type(cell_index).__name__}")
        cell_index = int(cell_index)
        if cell_index < 0 or cell_index >= n_cells:
            raise IndexError(
                f"cell_index 超出范围：共 {n_cells} 个细胞，索引需在 [0, {n_cells - 1}] 之间，得到 {cell_index}"
            )
        # 直接按行切片，避免对整个矩阵做全量转换
        if use_rep is None or use_rep == "X":
            raw = self._adata.X
            if hasattr(raw, "getrow"):
                # 稀疏矩阵：取单行后转稠密
                row_arr = raw.getrow(cell_index).toarray().flatten()
            elif hasattr(raw, "toarray"):
                row_arr = raw[cell_index].toarray().flatten()
            else:
                row_arr = np.asarray(raw[cell_index]).flatten()
        else:
            if use_rep not in self._adata.obsm:
                available = list(self._adata.obsm.keys())
                raise KeyError(
                    f"obsm 中不存在 '{use_rep}'，可用的表示为：{available}"
                )
            row_arr = np.asarray(self._adata.obsm[use_rep][cell_index]).flatten()

        return np.ascontiguousarray(row_arr, dtype=np.float32)

    def get_cell_info(self, cell_index: int) -> Dict[str, Any]:
        """获取指定细胞的元数据（obs 字段）。

        Parameters
        ----------
        cell_index:
            细胞在数据集中的整数索引（从 0 开始）。

        Returns
        -------
        dict
            包含该细胞所有 obs 字段的字典，额外附带 ``cell_id`` 键（obs_names 中的原始字符串 ID）。
        """
        n_cells = self._adata.n_obs
        if not isinstance(cell_index, (int, np.integer)):
            raise TypeError(f"cell_index 必须是整数，得到 {type(cell_index).__name__}")
        cell_index = int(cell_index)
        if cell_index < 0 or cell_index >= n_cells:
            raise IndexError(
                f"cell_index 超出范围：共 {n_cells} 个细胞，索引需在 [0, {n_cells - 1}] 之间，得到 {cell_index}"
            )

        row: pd.Series = self._adata.obs.iloc[cell_index]
        info: Dict[str, Any] = {"cell_id": str(self._adata.obs_names[cell_index])}
        for col in self._adata.obs.columns:
            val = row[col]
            # 将 pandas/numpy 的标量类型转为 Python 原生类型，方便序列化
            if isinstance(val, (np.integer,)):
                val = int(val)
            elif isinstance(val, (np.floating,)):
                val = float(val)
            elif isinstance(val, (np.bool_,)):
                val = bool(val)
            elif hasattr(val, "item") and callable(val.item):
                # 捕获其他 numpy 标量
                val = val.item()
            elif not isinstance(val, (int, float, bool, str, type(None))):
                # categorical 单元素、其余未知类型统一转 str
                val = str(val)
            info[col] = val
        return info

    def cell_index_from_id(self, cell_id: str) -> int:
        """通过细胞 ID 获取索引位置。"""
        if not isinstance(cell_id, str) or not cell_id.strip():
            raise ValueError("cell_id must be a non-empty string")
        if self._cell_id_to_index is None:
            self._cell_id_to_index = {
                str(item): idx for idx, item in enumerate(self._adata.obs_names)
            }
        key = cell_id.strip()
        if key not in self._cell_id_to_index:
            raise KeyError(f"cell_id not found: {cell_id}")
        return int(self._cell_id_to_index[key])

    # ------------------------------------------------------------------
    # 便捷属性
    # ------------------------------------------------------------------

    @property
    def n_cells(self) -> int:
        """数据集中的细胞数量。"""
        return int(self._adata.n_obs)

    @property
    def n_genes(self) -> int:
        """数据集中的基因数量（特征维度）。"""
        return int(self._adata.n_vars)

    def vector_dim(self, use_rep: Optional[str] = None) -> int:
        """返回当前向量表示的维度，直接用于初始化 ANNIndexer(dim=...)。

        Parameters
        ----------
        use_rep:
            同 ``get_vectors()``，默认为 ``X``（基因表达维度即 n_genes）。

        Returns
        -------
        int
            向量维度数。
        """
        if use_rep is None or use_rep == "X":
            return self.n_genes
        if use_rep not in self._adata.obsm:
            available = list(self._adata.obsm.keys())
            raise KeyError(
                f"obsm 中不存在 '{use_rep}'，可用的表示为：{available}"
            )
        return int(self._adata.obsm[use_rep].shape[1])

    @property
    def obs_columns(self) -> List[str]:
        """obs 元数据的所有列名。"""
        return list(self._adata.obs.columns)

    @property
    def available_reps(self) -> List[str]:
        """obsm 中可用的降维表示键名列表（如 X_pca、X_umap 等）。"""
        return list(self._adata.obsm.keys())

    @property
    def adata(self) -> AnnData:
        """返回底层 AnnData 对象，供高级用户直接操作。"""
        return self._adata

    def __repr__(self) -> str:
        return (
            f"DataLoader("
            f"file='{self._file_path.name}', "
            f"n_cells={self.n_cells}, "
            f"n_genes={self.n_genes}, "
            f"obs_columns={self.obs_columns}, "
            f"available_reps={self.available_reps}"
            f")"
        )
