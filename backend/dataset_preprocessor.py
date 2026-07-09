from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Union

import anndata as ad
import numpy as np
import pandas as pd


GeneJoinMode = Literal["inner", "outer"]


@dataclass(frozen=True)
class DatasetSummary:
    dataset_id: str
    source_path: str
    n_cells_before: int
    n_genes_before: int
    n_cells_after: int
    n_genes_after: int


@dataclass(frozen=True)
class JointDatasetReport:
    output_path: str
    join: GeneJoinMode
    n_datasets: int
    total_cells: int
    aligned_genes: int
    created_at: str
    datasets: List[DatasetSummary]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["datasets"] = [asdict(item) for item in self.datasets]
        return data

    def write_json(self, path: Union[str, Path]) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def prepare_joint_dataset(
    input_paths: Sequence[Union[str, Path]],
    output_path: Union[str, Path],
    *,
    join: GeneJoinMode = "inner",
    dataset_ids: Optional[Sequence[str]] = None,
    min_cells: int = 1,
    min_genes: int = 1,
    normalize_total: bool = False,
    log1p: bool = False,
    report_path: Optional[Union[str, Path]] = None,
) -> JointDatasetReport:
    """Load, clean and align multiple .h5ad datasets into one AnnData file.

    Parameters
    ----------
    input_paths:
        Two or more source .h5ad files.
    output_path:
        Destination path for the merged .h5ad file.
    join:
        ``"inner"`` keeps genes shared by every dataset. ``"outer"`` keeps the
        gene union and fills missing values with zero.
    dataset_ids:
        Optional labels written to ``obs["dataset_id"]``.
    min_cells, min_genes:
        Basic quality filters. Genes seen in fewer than ``min_cells`` cells and
        cells with fewer than ``min_genes`` detected genes are removed per dataset.
    normalize_total, log1p:
        Optional Scanpy-style normalization flags. They are implemented locally
        to keep this module easy to test.
    report_path:
        Optional JSON report destination.
    """
    paths = _validate_inputs(input_paths)
    if join not in ("inner", "outer"):
        raise ValueError("join must be either 'inner' or 'outer'")
    if len(paths) < 2:
        raise ValueError("at least two datasets are required for joint processing")
    if dataset_ids is not None and len(dataset_ids) != len(paths):
        raise ValueError("dataset_ids length must match input_paths length")

    ids = list(dataset_ids) if dataset_ids is not None else [_dataset_id(path) for path in paths]
    adatas: List[ad.AnnData] = []
    summaries: List[DatasetSummary] = []

    for path, dataset_id in zip(paths, ids):
        raw = ad.read_h5ad(path)
        n_cells_before = int(raw.n_obs)
        n_genes_before = int(raw.n_vars)
        cleaned = clean_adata(
            raw,
            dataset_id=dataset_id,
            min_cells=min_cells,
            min_genes=min_genes,
            normalize_total=normalize_total,
            log1p=log1p,
        )
        adatas.append(cleaned)
        summaries.append(
            DatasetSummary(
                dataset_id=dataset_id,
                source_path=str(path),
                n_cells_before=n_cells_before,
                n_genes_before=n_genes_before,
                n_cells_after=int(cleaned.n_obs),
                n_genes_after=int(cleaned.n_vars),
            )
        )

    merged = align_and_concat(adatas, dataset_ids=ids, join=join)
    merged.obs = _sanitize_dataframe_for_h5ad(merged.obs)
    merged.var = _sanitize_dataframe_for_h5ad(merged.var)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.write_h5ad(output)

    report = JointDatasetReport(
        output_path=str(output),
        join=join,
        n_datasets=len(adatas),
        total_cells=int(merged.n_obs),
        aligned_genes=int(merged.n_vars),
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        datasets=summaries,
    )
    if report_path is not None:
        report_file = Path(report_path)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report.write_json(report_file)
    return report


def clean_adata(
    adata: ad.AnnData,
    *,
    dataset_id: str,
    min_cells: int = 1,
    min_genes: int = 1,
    normalize_total: bool = False,
    log1p: bool = False,
) -> ad.AnnData:
    cleaned = adata.copy()
    cleaned.obs_names = pd.Index(cleaned.obs_names).astype(str)
    cleaned.var_names = pd.Index(cleaned.var_names).astype(str)
    cleaned.obs = _sanitize_dataframe_for_h5ad(cleaned.obs)
    cleaned.var = _sanitize_dataframe_for_h5ad(cleaned.var)
    cleaned.var_names_make_unique()
    cleaned.obs_names_make_unique()

    if min_genes > 1:
        cell_mask = np.asarray((_to_dense_bool(cleaned.X > 0).sum(axis=1) >= min_genes)).ravel()
        cleaned = cleaned[cell_mask, :].copy()
    if min_cells > 1:
        gene_mask = np.asarray((_to_dense_bool(cleaned.X > 0).sum(axis=0) >= min_cells)).ravel()
        cleaned = cleaned[:, gene_mask].copy()

    if normalize_total:
        cleaned.X = _normalize_total(cleaned.X)
    if log1p:
        cleaned.X = _log1p(cleaned.X)

    cleaned.obs["dataset_id"] = dataset_id
    cleaned.obs["source_cell_id"] = cleaned.obs_names.astype(str)
    cleaned.obs_names = [f"{dataset_id}:{cell_id}" for cell_id in cleaned.obs_names]
    return cleaned


def _sanitize_dataframe_for_h5ad(frame: pd.DataFrame) -> pd.DataFrame:
    sanitized = frame.copy()
    for column in sanitized.columns:
        series = sanitized[column]
        if pd.api.types.is_object_dtype(series.dtype) or isinstance(series.dtype, pd.CategoricalDtype):
            values = series.astype(object)
            sanitized[column] = values.map(lambda value: "" if pd.isna(value) else str(value))
    return sanitized


def align_and_concat(
    adatas: Sequence[ad.AnnData],
    *,
    dataset_ids: Sequence[str],
    join: GeneJoinMode = "inner",
) -> ad.AnnData:
    if not adatas:
        raise ValueError("adatas cannot be empty")
    if len(adatas) != len(dataset_ids):
        raise ValueError("dataset_ids length must match adatas length")
    if join not in ("inner", "outer"):
        raise ValueError("join must be either 'inner' or 'outer'")

    aligned_genes = _aligned_gene_names(adatas, join=join)
    aligned = [_reindex_genes(item, aligned_genes) for item in adatas]
    merged = ad.concat(
        aligned,
        join="outer",
        label="dataset_id",
        keys=list(dataset_ids),
        index_unique=None,
        fill_value=0,
    )
    merged.var_names = pd.Index(aligned_genes)
    merged.uns["joint_dataset"] = {
        "join": join,
        "n_datasets": len(adatas),
        "dataset_ids": list(dataset_ids),
    }
    return merged


def _validate_inputs(input_paths: Sequence[Union[str, Path]]) -> List[Path]:
    paths = [Path(item) for item in input_paths]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"dataset not found: {path}")
        if path.suffix.lower() != ".h5ad":
            raise ValueError(f"only .h5ad files are supported: {path}")
    return paths


def _dataset_id(path: Path) -> str:
    return path.stem.replace(" ", "_")


def _aligned_gene_names(adatas: Sequence[ad.AnnData], *, join: GeneJoinMode) -> List[str]:
    gene_sets = [set(map(str, item.var_names)) for item in adatas]
    if join == "inner":
        genes = set.intersection(*gene_sets)
    else:
        genes = set.union(*gene_sets)
    if not genes:
        raise ValueError("no aligned genes remain after applying join mode")
    return sorted(genes)


def _reindex_genes(adata: ad.AnnData, genes: Sequence[str]) -> ad.AnnData:
    current = pd.Index(map(str, adata.var_names))
    positions = current.get_indexer(genes)
    existing_mask = positions >= 0
    existing_positions = positions[existing_mask]

    x = _as_dense_array(adata.X)
    aligned_x = np.zeros((adata.n_obs, len(genes)), dtype=np.float32)
    if len(existing_positions):
        aligned_x[:, existing_mask] = x[:, existing_positions].astype(np.float32, copy=False)

    aligned = ad.AnnData(
        X=aligned_x,
        obs=adata.obs.copy(),
        var=pd.DataFrame(index=pd.Index(genes, name=adata.var_names.name)),
    )
    for key, value in adata.obsm.items():
        aligned.obsm[key] = value.copy() if hasattr(value, "copy") else value
    return aligned


def _as_dense_array(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return np.asarray(matrix.toarray())
    return np.asarray(matrix)


def _to_dense_bool(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return np.asarray(matrix.toarray(), dtype=bool)
    return np.asarray(matrix, dtype=bool)


def _normalize_total(matrix: Any, target_sum: float = 1e4) -> np.ndarray:
    arr = _as_dense_array(matrix).astype(np.float32, copy=True)
    totals = arr.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return arr / totals * target_sum


def _log1p(matrix: Any) -> np.ndarray:
    return np.log1p(_as_dense_array(matrix).astype(np.float32, copy=False))
