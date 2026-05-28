from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

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


# ---------------------------------------------------------------------------
# 多数据集管理
# ---------------------------------------------------------------------------

class DatasetManager:
    """多数据集管理器：支持 .h5ad 数据集的增删查、上传校验和元信息维护。

    目录结构::

        data_dir/
            <dataset_id>.h5ad          # 数据文件
        index_dir/
            <dataset_id>/              # 该数据集的索引目录（由算法模块写入）
        meta_dir/
            <dataset_id>.json          # 元信息缓存

    使用方式::

        manager = DatasetManager()
        dataset_id = manager.register("data/liver.h5ad", name="Liver Atlas")
        loader     = manager.get_loader(dataset_id)
        manager.list_datasets()
        manager.delete_dataset(dataset_id)
    """

    _META_SUFFIX = ".json"
    _DATA_SUFFIX = ".h5ad"

    def __init__(
        self,
        data_dir: Union[str, Path] = "data",
        index_dir: Union[str, Path] = "indexes",
        meta_dir: Union[str, Path] = "data/.meta",
    ) -> None:
        self._data_dir = Path(data_dir)
        self._index_dir = Path(index_dir)
        self._meta_dir = Path(meta_dir)

        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._meta_dir.mkdir(parents=True, exist_ok=True)

        # 内存缓存：dataset_id -> DataLoader
        self._loader_cache: Dict[str, DataLoader] = {}

    # ------------------------------------------------------------------
    # 核心增删查接口
    # ------------------------------------------------------------------

    def register(
        self,
        source_path: Union[str, Path],
        name: Optional[str] = None,
        copy: bool = True,
    ) -> str:
        """注册一个已有的 .h5ad 文件到管理器。

        Parameters
        ----------
        source_path:
            源文件路径。
        name:
            数据集展示名称，默认取文件名（不含扩展名）。
        copy:
            True（默认）：将文件复制到 data_dir；False：原地注册（文件必须已在 data_dir 内）。

        Returns
        -------
        str
            分配的 dataset_id（不可变唯一标识）。
        """
        source = Path(source_path)
        _validate_h5ad_file(source)

        dataset_id = _make_dataset_id(source)

        # 目标路径
        dest = self._data_dir / f"{dataset_id}{self._DATA_SUFFIX}"
        if not dest.exists():
            if copy:
                shutil.copy2(str(source), str(dest))
            else:
                if source.resolve() != dest.resolve():
                    raise ValueError(
                        f"copy=False 时文件必须已经位于 data_dir 内，"
                        f"期望路径：{dest}，实际路径：{source}"
                    )

        meta = self._load_meta(dataset_id) or {}
        meta.setdefault("dataset_id", dataset_id)
        meta.setdefault("name", name or source.stem)
        meta.setdefault("filename", dest.name)
        meta.setdefault("registered_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
        meta["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._save_meta(dataset_id, meta)

        return dataset_id

    def upload(
        self,
        file_bytes: bytes,
        filename: str,
        name: Optional[str] = None,
    ) -> str:
        """校验并上传 .h5ad 文件（字节流形式，适合 Flask 接口调用）。

        Parameters
        ----------
        file_bytes:
            文件原始字节内容（``request.files["file"].read()``）。
        filename:
            原始文件名，用于后缀校验。
        name:
            数据集展示名称，默认取文件名（不含扩展名）。

        Returns
        -------
        str
            分配的 dataset_id。

        Raises
        ------
        ValueError
            文件后缀不是 .h5ad，或文件内容校验不通过。
        """
        if not filename.lower().endswith(self._DATA_SUFFIX):
            raise ValueError(f"仅支持 .h5ad 文件，收到：{filename}")
        if len(file_bytes) == 0:
            raise ValueError("上传的文件内容为空")

        # 先写到临时位置做校验
        tmp_path = self._data_dir / f"_tmp_{int(time.time()*1000)}.h5ad"
        try:
            tmp_path.write_bytes(file_bytes)
            _validate_h5ad_file(tmp_path)  # 内容校验
            dataset_id = _make_dataset_id_from_bytes(file_bytes)
            dest = self._data_dir / f"{dataset_id}{self._DATA_SUFFIX}"
            if not dest.exists():
                shutil.move(str(tmp_path), str(dest))
            else:
                tmp_path.unlink(missing_ok=True)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

        stem = Path(filename).stem
        meta = self._load_meta(dataset_id) or {}
        meta.setdefault("dataset_id", dataset_id)
        meta.setdefault("name", name or stem)
        meta.setdefault("original_filename", filename)
        meta.setdefault("filename", dest.name)
        meta.setdefault("registered_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
        meta["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._save_meta(dataset_id, meta)

        return dataset_id

    def delete_dataset(self, dataset_id: str) -> None:
        """删除指定数据集的文件、索引目录和元信息缓存。

        Parameters
        ----------
        dataset_id:
            要删除的数据集 ID。

        Raises
        ------
        KeyError
            数据集不存在。
        """
        _validate_dataset_id(dataset_id)
        data_file = self._data_path(dataset_id)
        if not data_file.exists():
            raise KeyError(f"数据集不存在：{dataset_id}")

        # 1. 移除内存缓存
        self._loader_cache.pop(dataset_id, None)

        # 2. 删除数据文件
        data_file.unlink()

        # 3. 删除索引目录（如果存在）
        index_dir = self._index_dir / dataset_id
        if index_dir.exists():
            shutil.rmtree(str(index_dir))

        # 4. 删除元信息文件
        meta_file = self._meta_path(dataset_id)
        if meta_file.exists():
            meta_file.unlink()

    def get_loader(self, dataset_id: str) -> DataLoader:
        """获取指定数据集的 DataLoader（带内存缓存，不重复读文件）。

        Parameters
        ----------
        dataset_id:
            数据集 ID。

        Returns
        -------
        DataLoader
        """
        _validate_dataset_id(dataset_id)
        if dataset_id not in self._loader_cache:
            path = self._data_path(dataset_id)
            if not path.exists():
                raise KeyError(f"数据集不存在：{dataset_id}")
            self._loader_cache[dataset_id] = DataLoader(path)
        return self._loader_cache[dataset_id]

    def list_datasets(self) -> List[Dict[str, Any]]:
        """列出所有已注册的数据集及其元信息。

        Returns
        -------
        list of dict
            每个 dict 包含 ``dataset_id``、``name``、``filename`` 等字段。
        """
        result: List[Dict[str, Any]] = []
        for data_file in sorted(self._data_dir.glob(f"*{self._DATA_SUFFIX}")):
            dataset_id = data_file.stem
            if dataset_id.startswith("_"):
                continue  # 跳过临时文件
            meta = self._load_meta(dataset_id) or {
                "dataset_id": dataset_id,
                "name": dataset_id,
                "filename": data_file.name,
            }
            meta["file_size_bytes"] = data_file.stat().st_size
            result.append(meta)
        return result

    def get_meta(self, dataset_id: str) -> Dict[str, Any]:
        """获取指定数据集的元信息字典。

        Raises
        ------
        KeyError
            数据集不存在。
        """
        _validate_dataset_id(dataset_id)
        if not self._data_path(dataset_id).exists():
            raise KeyError(f"数据集不存在：{dataset_id}")
        meta = self._load_meta(dataset_id) or {"dataset_id": dataset_id}
        meta["file_size_bytes"] = self._data_path(dataset_id).stat().st_size
        return meta

    def update_meta(self, dataset_id: str, **kwargs: Any) -> None:
        """更新指定数据集的元信息字段（只允许改展示性字段，不影响文件）。

        Parameters
        ----------
        **kwargs:
            要更新的键值对，例如 ``name="新名称"``。
        """
        _validate_dataset_id(dataset_id)
        if not self._data_path(dataset_id).exists():
            raise KeyError(f"数据集不存在：{dataset_id}")
        _IMMUTABLE_META_KEYS = {"dataset_id", "filename", "registered_at"}
        for key in kwargs:
            if key in _IMMUTABLE_META_KEYS:
                raise ValueError(f"字段 '{key}' 不可修改")
        meta = self._load_meta(dataset_id) or {"dataset_id": dataset_id}
        meta.update(kwargs)
        meta["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._save_meta(dataset_id, meta)

    def index_path_for(self, dataset_id: str, filename: str = "cell_index.index") -> Path:
        """返回该数据集对应的索引文件路径，供算法模块使用。

        Parameters
        ----------
        dataset_id:
            数据集 ID。
        filename:
            索引文件名，默认 ``cell_index.index``。

        Returns
        -------
        Path
            索引文件的完整路径（目录会自动创建）。
        """
        _validate_dataset_id(dataset_id)
        idx_dir = self._index_dir / dataset_id
        idx_dir.mkdir(parents=True, exist_ok=True)
        return idx_dir / filename

    def get_available_reps(self, dataset_id: str) -> List[str]:
        """返回指定数据集可用的向量表示列表（如 X_pca、X_umap 等）。

        供 API 层在构建索引时展示可选的 ``use_rep`` 选项。

        Parameters
        ----------
        dataset_id:
            数据集 ID。

        Returns
        -------
        list of str
            ``obsm`` 中所有可用的降维表示键名，例如 ``["X_pca", "X_umap", "X_tsne"]``。

        Raises
        ------
        KeyError
            数据集不存在。
        """
        return self.get_loader(dataset_id).available_reps

    def get_dataset_info(self, dataset_id: str) -> Dict[str, Any]:
        """返回数据集完整信息：元信息 + 运行时统计（细胞数、基因数、可用向量类型等）。

        供 ``GET /api/metadata`` 接口一次性返回完整元数据，无需调用方再单独 get_loader。

        Parameters
        ----------
        dataset_id:
            数据集 ID。

        Returns
        -------
        dict
            包含以下字段：

            - 所有 ``get_meta()`` 字段（dataset_id、name、filename、registered_at 等）
            - ``n_cells``：细胞总数
            - ``n_genes``：基因数量
            - ``available_reps``：可用向量表示列表（如 ``["X_pca", "X_umap"]``）
            - ``obs_columns``：细胞元数据字段名列表

        Raises
        ------
        KeyError
            数据集不存在。
        """
        info = self.get_meta(dataset_id)
        loader = self.get_loader(dataset_id)
        info["n_cells"] = loader.n_cells
        info["n_genes"] = loader.n_genes
        info["available_reps"] = loader.available_reps
        info["obs_columns"] = loader.obs_columns
        return info

    def __iter__(self) -> Iterator[str]:
        """遍历所有 dataset_id。"""
        for data_file in self._data_dir.glob(f"*{self._DATA_SUFFIX}"):
            if not data_file.stem.startswith("_"):
                yield data_file.stem

    def __len__(self) -> int:
        return sum(
            1 for f in self._data_dir.glob(f"*{self._DATA_SUFFIX}")
            if not f.stem.startswith("_")
        )

    def __repr__(self) -> str:
        return (
            f"DatasetManager("
            f"data_dir='{self._data_dir}', "
            f"n_datasets={len(self)}"
            f")"
        )

    # ------------------------------------------------------------------
    # 内部工具方法
    # ------------------------------------------------------------------

    def _data_path(self, dataset_id: str) -> Path:
        return self._data_dir / f"{dataset_id}{self._DATA_SUFFIX}"

    def _meta_path(self, dataset_id: str) -> Path:
        return self._meta_dir / f"{dataset_id}{self._META_SUFFIX}"

    def _load_meta(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        path = self._meta_path(dataset_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _save_meta(self, dataset_id: str, meta: Dict[str, Any]) -> None:
        self._meta_path(dataset_id).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# 模块级工具函数
# ---------------------------------------------------------------------------

def _validate_h5ad_file(path: Path) -> None:
    """校验文件是否是合法的 .h5ad 文件（格式 + 内容双重校验）。"""
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")
    if path.suffix.lower() != ".h5ad":
        raise ValueError(f"文件格式不支持，需要 .h5ad，得到：{path.suffix}")
    if path.stat().st_size == 0:
        raise ValueError(f"文件内容为空：{path.name}")
    # 尝试用 scanpy 读取，确认内容合法
    try:
        adata = sc.read_h5ad(str(path))
    except Exception as exc:
        raise ValueError(f"文件内容校验失败，不是有效的 .h5ad 文件：{exc}") from exc
    if adata.n_obs == 0:
        raise ValueError(f"数据集中细胞数量为 0，文件可能损坏：{path.name}")
    if adata.n_vars == 0:
        raise ValueError(f"数据集中基因数量为 0，文件可能损坏：{path.name}")


def _make_dataset_id(path: Path) -> str:
    """根据文件路径（绝对路径 + 修改时间）生成稳定的 dataset_id。"""
    key = f"{path.resolve()}:{path.stat().st_mtime}"
    return hashlib.md5(key.encode()).hexdigest()[:16]


def _make_dataset_id_from_bytes(data: bytes) -> str:
    """根据文件内容生成 dataset_id（用于上传场景）。"""
    return hashlib.md5(data).hexdigest()[:16]


def _validate_dataset_id(dataset_id: str) -> None:
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("dataset_id 必须是非空字符串")
