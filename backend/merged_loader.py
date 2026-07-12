"""In-memory loader facade for cross-dataset joint search.

``MergedDataLoader`` does not create a physical merged .h5ad file. Instead, it
stitches several existing ``DataLoader`` instances into one duck-typed loader so
ANN indexing, vector search and metadata lookup can treat multiple datasets as a
single search space.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


MERGED_DIR_NAME = ".merged"


def _make_merged_id(source_ids: List[str]) -> str:
    """Create a stable id for the unordered set of source dataset ids."""
    key = "-".join(sorted(source_ids))
    return "merged_" + hashlib.md5(key.encode()).hexdigest()[:12]


@dataclass
class MergedDatasetConfig:
    """Persisted metadata describing one logical merged dataset."""

    merged_id: str
    name: str
    source_datasets: List[str]
    use_rep: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MergedDatasetConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save(self, merged_dir: Path) -> None:
        merged_dir.mkdir(parents=True, exist_ok=True)
        path = merged_dir / f"{self.merged_id}.json"
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, merged_dir: Path, merged_id: str) -> "MergedDatasetConfig":
        path = merged_dir / f"{merged_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"合并数据集配置不存在：{path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)


def get_merged_dir(data_dir: Path) -> Path:
    d = data_dir / MERGED_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_merged_configs(data_dir: Path) -> List[MergedDatasetConfig]:
    merged_dir = data_dir / MERGED_DIR_NAME
    if not merged_dir.exists():
        return []
    configs = []
    for path in sorted(merged_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            configs.append(MergedDatasetConfig.from_dict(data))
        except (json.JSONDecodeError, TypeError, KeyError):
            continue
    return configs


def load_merged_config(data_dir: Path, merged_id: str) -> MergedDatasetConfig:
    return MergedDatasetConfig.load(data_dir / MERGED_DIR_NAME, merged_id)


def save_merged_config(data_dir: Path, config: MergedDatasetConfig) -> None:
    config.save(data_dir / MERGED_DIR_NAME)


def delete_merged_config(data_dir: Path, merged_id: str) -> None:
    path = data_dir / MERGED_DIR_NAME / f"{merged_id}.json"
    if path.exists():
        path.unlink()


class _MergedLightAnnDataProxy:
    """合并数据集的 AnnData 兼容代理，仅暴露下游需要的轻量属性。"""

    def __init__(
        self,
        obsm: Dict[str, np.ndarray],
        obs: pd.DataFrame,
        obs_names: List[str],
        n_obs: int,
        n_vars: int,
    ) -> None:
        self.obsm = obsm
        self.obs = obs
        self.obs_names = obs_names
        self.n_obs = n_obs
        self.n_vars = n_vars

    @property
    def X(self):
        return None


class MergedDataLoader:
    """将多个 DataLoader 的向量合并为统一接口，实现与 DataLoader 相同的鸭子类型。

    全局索引是每个源数据集索引区间的拼接；返回给前端的 cell_id 会加上
    ``<dataset_id>:`` 前缀，避免不同数据集的原始细胞名冲突。
    """

    def __init__(self, config: MergedDatasetConfig, source_loaders: Dict[str, Any]) -> None:
        if len(config.source_datasets) < 2:
            raise ValueError("合并数据集至少需要 2 个源数据集")

        missing = [ds for ds in config.source_datasets if ds not in source_loaders]
        if missing:
            raise ValueError(f"源数据集缺失：{missing}")

        self._config = config
        self._use_rep = config.use_rep
        self._sources: List[Tuple[str, Any]] = []
        for ds_id in config.source_datasets:
            self._sources.append((ds_id, source_loaders[ds_id]))

        dims = set()
        for ds_id, loader in self._sources:
            if config.use_rep not in loader.available_reps:
                raise ValueError(
                    f"源数据集 '{ds_id}' 不包含表示 '{config.use_rep}'，"
                    f"可用：{loader.available_reps}"
                )
            dims.add(loader.vector_dim(config.use_rep))
        if len(dims) > 1:
            dim_detail = {ds: ld.vector_dim(config.use_rep) for ds, ld in self._sources}
            raise ValueError(
                f"源数据集在 '{config.use_rep}' 下维度不一致：{dim_detail}"
            )

        self._offsets: List[int] = []
        offset = 0
        for _, loader in self._sources:
            self._offsets.append(offset)
            offset += loader.n_cells
        self._total_cells = offset

        self._cell_id_to_index: Optional[Dict[str, int]] = None
        self._obs_names: Optional[List[str]] = None
        self._adata_proxy: Optional[_MergedLightAnnDataProxy] = None

    def _resolve(self, global_index: int) -> Tuple[str, Any, int]:
        """Map a merged/global row index back to ``(dataset_id, loader, local_idx)``."""
        if global_index < 0 or global_index >= self._total_cells:
            raise IndexError(
                f"global_index {global_index} 超出范围 [0, {self._total_cells})"
            )
        for i, (ds_id, loader) in enumerate(self._sources):
            end = self._offsets[i] + loader.n_cells
            if global_index < end:
                return ds_id, loader, global_index - self._offsets[i]
        raise IndexError(f"global_index {global_index} 超出范围")

    @property
    def n_cells(self) -> int:
        return self._total_cells

    @property
    def n_genes(self) -> int:
        return self._sources[0][1].n_genes

    def vector_dim(self, use_rep: Optional[str] = None) -> int:
        rep = use_rep or self._use_rep
        return self._sources[0][1].vector_dim(rep)

    @property
    def available_reps(self) -> List[str]:
        if not self._sources:
            return []
        common = set(self._sources[0][1].available_reps)
        for _, loader in self._sources[1:]:
            common &= set(loader.available_reps)
        return sorted(common)

    @property
    def obs_columns(self) -> List[str]:
        all_cols: set = set()
        for _, loader in self._sources:
            all_cols |= set(loader.obs_columns)
        return sorted(all_cols)

    @property
    def obs_names(self) -> List[str]:
        if self._obs_names is None:
            names = []
            for ds_id, loader in self._sources:
                for orig_name in loader.adata.obs_names:
                    names.append(f"{ds_id}:{orig_name}")
            self._obs_names = names
        return self._obs_names

    def get_vectors(self, use_rep: Optional[str] = None) -> np.ndarray:
        """Stack the selected representation from every source loader."""
        rep = use_rep or self._use_rep
        arrays = []
        for _, loader in self._sources:
            arrays.append(loader.get_vectors(rep))
        result = np.vstack(arrays)
        return np.ascontiguousarray(result, dtype=np.float32)

    def get_vector(self, cell_index: int, use_rep: Optional[str] = None) -> np.ndarray:
        rep = use_rep or self._use_rep
        if rep == "X" or rep is None:
            raise ValueError("合并数据集不支持原始表达矩阵 X，请使用 obsm 中的表示（如 X_pca）")
        ds_id, loader, local_idx = self._resolve(cell_index)
        return loader.get_vector(local_idx, use_rep=rep)

    def get_cell_info(self, cell_index: int) -> Dict[str, Any]:
        """Return metadata for a global cell index with source dataset attached."""
        ds_id, loader, local_idx = self._resolve(cell_index)
        info = loader.get_cell_info(local_idx)
        orig_id = info.get("cell_id", "")
        info["cell_id"] = f"{ds_id}:{orig_id}"
        info["source_dataset"] = ds_id
        return info

    def cell_index_from_id(self, cell_id: str) -> int:
        """Resolve prefixed cell ids such as ``dataset_a:AAAC...``."""
        if self._cell_id_to_index is None:
            self._cell_id_to_index = {}
            for idx, name in enumerate(self.obs_names):
                self._cell_id_to_index[name] = idx
        if cell_id not in self._cell_id_to_index:
            raise KeyError(f"Cell ID 不存在：{cell_id}")
        return self._cell_id_to_index[cell_id]

    @property
    def adata(self) -> _MergedLightAnnDataProxy:
        """Build and cache a lightweight AnnData-like view for query code."""
        if self._adata_proxy is None:
            obsm: Dict[str, np.ndarray] = {}
            common_reps = self.available_reps
            for rep in common_reps:
                arrays = [loader.adata.obsm[rep] for _, loader in self._sources]
                obsm[rep] = np.ascontiguousarray(np.vstack(arrays), dtype=np.float32)

            obs_frames = []
            for ds_id, loader in self._sources:
                df = loader.adata.obs.copy()
                df["source_dataset"] = ds_id
                df.index = [f"{ds_id}:{idx}" for idx in df.index]
                obs_frames.append(df)
            merged_obs = pd.concat(obs_frames, axis=0, ignore_index=False)

            self._adata_proxy = _MergedLightAnnDataProxy(
                obsm=obsm,
                obs=merged_obs,
                obs_names=self.obs_names,
                n_obs=self._total_cells,
                n_vars=self.n_genes,
            )
        return self._adata_proxy

    @property
    def source_config(self) -> MergedDatasetConfig:
        return self._config
