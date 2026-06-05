from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

from backend.ann_indexer import ANNIndexer, IndexConfig
from backend.data_reader import DataLoader

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "indexes"

DEFAULT_DATA_PATH = "data/liver.h5ad"
DEFAULT_USE_REP = "X_pca"
DEFAULT_TOP_K = 10
MAX_TOP_K = 100
ALLOWED_EXTENSIONS = {".h5ad"}

_DATASET_CACHE: Dict[str, DataLoader] = {}
_INDEX_CACHE: Dict[Tuple[str, str], Tuple[ANNIndexer, Tuple[Any, ...]]] = {}
_BENCHMARK_INDEX_CACHE: Dict[Tuple[str, str, Tuple[Any, ...]], ANNIndexer] = {}
_BENCHMARK_HISTORY_PATH = DATA_DIR / "benchmark_history.json"


def _normalize_dataset_id(dataset_id: Optional[str]) -> Optional[str]:
    if dataset_id is None:
        return None
    cleaned = dataset_id.strip()
    if not cleaned or "/" in cleaned or "\\" in cleaned:
        raise ValueError("dataset_id is invalid")
    return cleaned


def _resolve_dataset_path(dataset_id: Optional[str]) -> Path:
    if dataset_id:
        name = _normalize_dataset_id(dataset_id)
        assert name is not None
        candidate = DATA_DIR / name
        if candidate.suffix == "":
            candidate = candidate.with_suffix(".h5ad")
        if not candidate.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_id}")
        return candidate

    default_path = Path(DEFAULT_DATA_PATH)
    if not default_path.is_absolute():
        default_path = BASE_DIR / default_path
    if not default_path.exists():
        raise FileNotFoundError(f"Default dataset not found: {default_path}")
    return default_path


def _dataset_id_from_path(path: Path) -> str:
    return path.stem


def _backup_index_path(index_path: Path) -> Path:
    return Path(f"{index_path}.npz")


def _index_config_from_env() -> IndexConfig:
    return IndexConfig.from_env()


def _config_cache_key(config: IndexConfig) -> Tuple[Any, ...]:
    cfg = config.normalized()
    return (
        cfg.backend,
        cfg.index_type,
        cfg.metric,
        cfg.nlist,
        cfg.nprobe,
        cfg.m,
        cfg.ef_construction,
        cfg.ef_search,
        cfg.pq_m,
        cfg.pq_nbits,
    )


def _index_config_from_payload(payload: Dict[str, Any]) -> IndexConfig:
    config = _index_config_from_env()
    overrides: Dict[str, Any] = {}

    backend = payload.get("index_backend")
    if backend not in (None, ""):
        overrides["backend"] = backend

    index_type = payload.get("index_type")
    if index_type not in (None, ""):
        overrides["index_type"] = index_type

    metric = payload.get("index_metric")
    if metric not in (None, ""):
        overrides["metric"] = metric

    nlist = _parse_optional_int(payload, "nlist")
    if nlist is not None:
        overrides["nlist"] = nlist

    nprobe = _parse_optional_int(payload, "nprobe")
    if nprobe is not None:
        overrides["nprobe"] = nprobe

    m = _parse_optional_int(payload, "m")
    if m is not None:
        overrides["m"] = m

    ef_construction = _parse_optional_int(payload, "ef_construction")
    if ef_construction is not None:
        overrides["ef_construction"] = ef_construction

    ef_search = _parse_optional_int(payload, "ef_search")
    if ef_search is not None:
        overrides["ef_search"] = ef_search

    pq_m = _parse_optional_int(payload, "pq_m")
    if pq_m is not None:
        overrides["pq_m"] = pq_m

    pq_nbits = _parse_optional_int(payload, "pq_nbits")
    if pq_nbits is not None:
        overrides["pq_nbits"] = pq_nbits

    if not overrides:
        return config
    return config.update(**overrides)


def _index_path(dataset_id: str, use_rep: str) -> Path:
    return INDEX_DIR / f"{dataset_id}_{use_rep}.index"


def _read_stored_index_config(dataset_id: str, use_rep: str) -> Optional[Dict[str, Any]]:
    """Read index config from an existing index file without loading vectors."""
    index_path = _index_path(dataset_id, use_rep)
    backup_path = _backup_index_path(index_path)

    # Prefer reading from the .npz backup (contains full config_json)
    npz_path = backup_path if backup_path.exists() else None
    if npz_path is None and index_path.exists():
        # FAISS binary index without backup — try to find config from cache
        cached = _INDEX_CACHE.get((dataset_id, use_rep))
        if cached is not None:
            cached_indexer, _ = cached
            return cached_indexer.config_summary
        return None

    if npz_path is None:
        return None

    try:
        with np.load(npz_path, allow_pickle=False) as data:
            config_json = (
                str(np.asarray(data["config_json"]).item())
                if "config_json" in data
                else None
            )
            if config_json:
                return json.loads(config_json)
            # Fallback: assemble from individual fields
            return {
                "backend": str(np.asarray(data.get("backend", "auto")).item()),
                "index_type": str(np.asarray(data.get("index_type", "flat")).item()),
                "metric": str(np.asarray(data.get("metric", "l2")).item()),
            }
    except (KeyError, ValueError, OSError):
        return None


def _get_loader(dataset_id: str, dataset_path: Optional[Path] = None) -> DataLoader:
    if dataset_id in _DATASET_CACHE:
        return _DATASET_CACHE[dataset_id]
    if dataset_path is None:
        dataset_path = _resolve_dataset_path(dataset_id)
    loader = DataLoader(dataset_path)
    _DATASET_CACHE[dataset_id] = loader
    return loader


def _get_indexer(
    dataset_id: str,
    use_rep: str,
    index_config: IndexConfig,
    build_if_missing: bool = True,
) -> ANNIndexer:
    cache_key = (dataset_id, use_rep)
    config_key = _config_cache_key(index_config)
    cached = _INDEX_CACHE.get(cache_key)
    if cached is not None:
        cached_indexer, cached_key = cached
        if cached_key == config_key:
            return cached_indexer

    loader = _get_loader(dataset_id)
    indexer = ANNIndexer(dim=loader.vector_dim(use_rep), config=index_config)
    index_path = _index_path(dataset_id, use_rep)
    backup_path = _backup_index_path(index_path)

    if index_path.exists() or backup_path.exists():
        try:
            indexer.load_index(index_path)
        except ValueError as exc:
            if "config mismatch" in str(exc).lower() and build_if_missing:
                vectors = loader.get_vectors(use_rep)
                indexer.build_index(vectors)
                indexer.save_index(index_path)
            else:
                raise
    elif build_if_missing:
        vectors = loader.get_vectors(use_rep)
        indexer.build_index(vectors)
        indexer.save_index(index_path)
    else:
        raise FileNotFoundError("Index not found")

    _INDEX_CACHE[cache_key] = (indexer, config_key)
    return indexer


def _get_benchmark_indexer(
    dataset_id: str,
    use_rep: str,
    vectors: np.ndarray,
    config: IndexConfig,
) -> ANNIndexer:
    cache_key = (dataset_id, use_rep, _config_cache_key(config))
    cached = _BENCHMARK_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached
    indexer = ANNIndexer(dim=vectors.shape[1], config=config)
    indexer.build_index(vectors)
    _BENCHMARK_INDEX_CACHE[cache_key] = indexer
    return indexer


def _load_benchmark_history() -> List[Dict[str, Any]]:
    if not _BENCHMARK_HISTORY_PATH.exists():
        return []
    try:
        return json.loads(_BENCHMARK_HISTORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_benchmark_history(history: List[Dict[str, Any]]) -> None:
    _BENCHMARK_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _BENCHMARK_HISTORY_PATH.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _append_benchmark_history(entry: Dict[str, Any], keep_last: int = 50) -> int:
    history = _load_benchmark_history()
    history.append(entry)
    if keep_last > 0 and len(history) > keep_last:
        history = history[-keep_last:]
    _save_benchmark_history(history)
    return len(history) - 1


def _suggest_pq_m(dim: int, preferred: int = 16) -> int:
    if dim <= 0:
        return 1
    preferred = min(preferred, dim)
    for candidate in range(preferred, 0, -1):
        if dim % candidate == 0:
            return candidate
    return 1


def _pq_m_options(dim: int) -> List[int]:
    if dim <= 0:
        return [1]
    return [candidate for candidate in range(dim, 0, -1) if dim % candidate == 0]


def _normalize_pq_config_for_dim(config: IndexConfig, dim: int) -> IndexConfig:
    normalized = config.normalized()
    if normalized.index_type != "pq":
        return normalized
    if dim > 0 and dim % int(normalized.pq_m) != 0:
        return normalized.update(pq_m=_suggest_pq_m(dim, int(normalized.pq_m)))
    return normalized


def _dataset_payload(dataset_id: str, dataset_path: Path) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "id": dataset_id,
        "filename": dataset_path.name,
    }

    try:
        loader = _get_loader(dataset_id, dataset_path)
        payload.update(
            {
                "n_cells": loader.n_cells,
                "n_genes": loader.n_genes,
                "available_reps": loader.available_reps,
                "obs_columns": loader.obs_columns,
            }
        )
    except Exception as exc:
        payload["error"] = str(exc)

    stat = dataset_path.stat()
    payload["size_mb"] = round(stat.st_size / (1024 * 1024), 2)
    payload["modified_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))

    index_path = _index_path(dataset_id, DEFAULT_USE_REP)
    index_ready = index_path.exists() or _backup_index_path(index_path).exists()
    payload["index_status"] = "ready" if index_ready else "missing"
    payload["index_backend"] = None
    cached_entry = _INDEX_CACHE.get((dataset_id, DEFAULT_USE_REP))
    if cached_entry is not None:
        cached_indexer, _ = cached_entry
        payload["index_backend"] = cached_indexer.backend

    return payload


def _metadata_payload(
    dataset_id: str, use_rep: str, index_config: IndexConfig
) -> Dict[str, Any]:
    loader = _get_loader(dataset_id)
    vector_dim = loader.vector_dim(use_rep)
    index_config = _normalize_pq_config_for_dim(index_config, vector_dim)
    index_path = _index_path(dataset_id, use_rep)
    backup_path = _backup_index_path(index_path)
    index_ready = index_path.exists() or backup_path.exists()
    payload: Dict[str, Any] = {
        "dataset_id": dataset_id,
        "data_path": str(_resolve_dataset_path(dataset_id)),
        "index_path": str(index_path),
        "use_rep": use_rep,
        "ready": index_ready,
        "n_cells": loader.n_cells,
        "n_genes": loader.n_genes,
        "vector_dim": vector_dim,
        "pq_m_options": _pq_m_options(vector_dim),
        "suggested_pq_m": _suggest_pq_m(vector_dim),
        "available_reps": loader.available_reps,
        "obs_columns": loader.obs_columns,
        "index_config": index_config.to_dict(),
    }
    cached_entry = _INDEX_CACHE.get((dataset_id, use_rep))
    if cached_entry is not None:
        cached_indexer, _ = cached_entry
        payload["index_backend"] = cached_indexer.backend
        payload["index_type"] = cached_indexer.index_type
        payload["index_metric"] = cached_indexer.metric
        payload["index_config"] = cached_indexer.config_summary
    else:
        payload["index_backend"] = None
        payload["index_type"] = index_config.index_type
        payload["index_metric"] = index_config.metric
    return payload


app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)
CORS(app)
DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)


def create_app() -> Flask:
    return app


# 路由 1：系统大屏首页 (登录后看到的)
@app.route('/')
def index():
    return render_template('index.html')

# 路由 2：登录/注册页 (独立页面，不带侧边栏)
@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/data')
def data_manage():
    return render_template('data_manage.html')

# ================================
# 相似检索页路由
# ================================
@app.route('/search')
def similarity_search():
    return render_template('search.html')

# ================================
# 用户管理页路由 (仅管理员可见)
# ================================
@app.route('/users')
def user_management():
    return render_template('users.html')

@app.route('/profile')
def profile():
    return render_template('profile.html')

# ================================
# 性能评测页路由
# ================================
@app.route('/benchmark')
def benchmark():
    return render_template('benchmark.html')


@app.get("/api")
def api_root():
    return _json_response(
        {
            "message": "Single-Cell ANN API",
            "endpoints": ["/api/health", "/api/metadata", "/api/search"],
            "datasets": "/api/datasets",
        }
    )


@app.get("/api/health")
def health():
    try:
        dataset_path = _resolve_dataset_path(None)
        dataset_id = _dataset_id_from_path(dataset_path)
        payload = _metadata_payload(dataset_id, DEFAULT_USE_REP, _index_config_from_env())
        return _json_response(payload, 200)
    except Exception as exc:
        return _json_response(
            {
                "ready": False,
                "data_path": DEFAULT_DATA_PATH,
                "use_rep": DEFAULT_USE_REP,
                "error": str(exc),
            },
            503,
        )


@app.get("/api/metadata")
def metadata():
    try:
        payload = dict(request.args)
        dataset_id = _normalize_dataset_id(payload.get("dataset_id"))
        dataset_path = _resolve_dataset_path(dataset_id)
        resolved_id = _dataset_id_from_path(dataset_path)
        use_rep = payload.get("use_rep", DEFAULT_USE_REP)
        index_config = _index_config_from_payload(payload)
        return _json_response(_metadata_payload(resolved_id, use_rep, index_config))
    except Exception as exc:
        return _json_response({"error": str(exc)}, 400)


@app.get("/api/datasets")
def list_datasets():
    datasets = []
    for path in sorted(DATA_DIR.glob("*.h5ad")):
        dataset_id = _dataset_id_from_path(path)
        datasets.append(_dataset_payload(dataset_id, path))
    return _json_response({"datasets": datasets})


@app.get("/api/datasets/<dataset_id>/indices")
def dataset_indices(dataset_id: str):
    """Return pre-built index info for a dataset so the search page can auto-fill."""
    try:
        dataset_id = _normalize_dataset_id(dataset_id)
        use_rep = request.args.get("use_rep", DEFAULT_USE_REP)

        # Check in-memory cache first
        cached = _INDEX_CACHE.get((dataset_id, use_rep))
        if cached is not None:
            cached_indexer, _ = cached
            return _json_response({
                "ready": True,
                "index_config": cached_indexer.config_summary,
            })

        # Check on-disk index
        stored = _read_stored_index_config(dataset_id, use_rep)
        if stored is not None:
            return _json_response({
                "ready": True,
                "index_config": stored,
            })

        return _json_response({"ready": False, "index_config": None})
    except Exception as exc:
        return _json_response({"error": str(exc)}, 400)


@app.get("/api/cells")
def list_cells():
    try:
        payload = dict(request.args)
        dataset_id = _normalize_dataset_id(payload.get("dataset_id"))
        dataset_path = _resolve_dataset_path(dataset_id)
        resolved_id = _dataset_id_from_path(dataset_path)
        offset = _parse_int(payload, "offset", default=0)
        limit = _parse_int(payload, "limit", default=50)

        if offset < 0:
            return _json_response({"error": "offset must be >= 0"}, 400)
        if limit <= 0:
            return _json_response({"error": "limit must be a positive integer"}, 400)
        limit = min(limit, 200)

        loader = _get_loader(resolved_id, dataset_path)
        total = loader.n_cells
        start = min(offset, total)
        end = min(start + limit, total)

        cells = []
        obs_names = loader.adata.obs_names
        for idx in range(start, end):
            cell_id = str(obs_names[idx])
            cell_type = None
            if "cell_type" in loader.adata.obs.columns:
                cell_type = loader.adata.obs.iloc[idx]["cell_type"]
                if not isinstance(cell_type, (str, int, float, bool, type(None))):
                    cell_type = str(cell_type)
            cells.append(
                {
                    "cell_index": idx,
                    "cell_id": cell_id,
                    "cell_type": cell_type,
                }
            )

        return _json_response(
            {
                "dataset_id": resolved_id,
                "offset": start,
                "limit": limit,
                "total": total,
                "next_offset": end if end < total else None,
                "prev_offset": max(start - limit, 0) if start > 0 else None,
                "cells": cells,
            }
        )
    except Exception as exc:
        return _json_response({"error": str(exc)}, 400)


@app.get("/api/umap")
def umap_points():
    try:
        payload = dict(request.args)
        dataset_id = _normalize_dataset_id(payload.get("dataset_id"))
        dataset_path = _resolve_dataset_path(dataset_id)
        resolved_id = _dataset_id_from_path(dataset_path)
        limit = _parse_int(payload, "limit", default=3000)
        seed = _parse_int(payload, "seed", default=42)
        color_by = payload.get("color_by") or "cell_type"

        if limit <= 0:
            return _json_response({"error": "limit must be a positive integer"}, 400)
        limit = min(limit, 10000)

        loader = _get_loader(resolved_id, dataset_path)
        if "X_umap" not in loader.adata.obsm:
            return _json_response({"error": "X_umap is not available for this dataset"}, 400)

        coords = np.asarray(loader.adata.obsm["X_umap"])
        if coords.ndim != 2 or coords.shape[1] < 2:
            return _json_response({"error": "X_umap must have at least two dimensions"}, 400)

        total = loader.n_cells
        if limit >= total:
            indices = np.arange(total, dtype=np.int64)
        else:
            rng = np.random.default_rng(seed)
            indices = np.sort(rng.choice(total, size=limit, replace=False))

        obs = loader.adata.obs
        if color_by not in obs.columns:
            color_by = "cell_type" if "cell_type" in obs.columns else None

        points = []
        category_counts: Dict[str, int] = {}
        obs_names = loader.adata.obs_names
        for idx_value in indices.tolist():
            idx = int(idx_value)
            category = "unknown"
            if color_by is not None:
                value = obs.iloc[idx][color_by]
                category = str(value) if value is not None else "unknown"
            category_counts[category] = category_counts.get(category, 0) + 1
            points.append(
                {
                    "cell_index": idx,
                    "cell_id": str(obs_names[idx]),
                    "x": float(coords[idx, 0]),
                    "y": float(coords[idx, 1]),
                    "category": category,
                }
            )

        return _json_response(
            {
                "dataset_id": resolved_id,
                "use_rep": "X_umap",
                "color_by": color_by,
                "total": total,
                "sampled": len(points),
                "points": points,
                "categories": [
                    {"name": name, "count": count}
                    for name, count in sorted(
                        category_counts.items(), key=lambda item: item[1], reverse=True
                    )
                ],
            }
        )
    except Exception as exc:
        return _json_response({"error": str(exc)}, 400)


@app.post("/api/umap/cells")
def umap_cells():
    try:
        payload = _request_payload()
        dataset_id = _normalize_dataset_id(payload.get("dataset_id"))
        dataset_path = _resolve_dataset_path(dataset_id)
        resolved_id = _dataset_id_from_path(dataset_path)
        cell_ids = payload.get("cell_ids") or []
        color_by = payload.get("color_by") or "cell_type"

        if not isinstance(cell_ids, list):
            return _json_response({"error": "cell_ids must be a list"}, 400)
        if len(cell_ids) > 500:
            return _json_response({"error": "cell_ids must not exceed 500"}, 400)

        loader = _get_loader(resolved_id, dataset_path)
        if "X_umap" not in loader.adata.obsm:
            return _json_response({"error": "X_umap is not available for this dataset"}, 400)

        coords = np.asarray(loader.adata.obsm["X_umap"])
        if coords.ndim != 2 or coords.shape[1] < 2:
            return _json_response({"error": "X_umap must have at least two dimensions"}, 400)

        obs = loader.adata.obs
        if color_by not in obs.columns:
            color_by = "cell_type" if "cell_type" in obs.columns else None

        points = []
        missing = []
        for raw_cell_id in cell_ids:
            cell_id = str(raw_cell_id)
            try:
                idx = loader.cell_index_from_id(cell_id)
            except KeyError:
                missing.append(cell_id)
                continue

            category = "unknown"
            if color_by is not None:
                value = obs.iloc[idx][color_by]
                category = str(value) if value is not None else "unknown"
            points.append(
                {
                    "cell_index": idx,
                    "cell_id": cell_id,
                    "x": float(coords[idx, 0]),
                    "y": float(coords[idx, 1]),
                    "category": category,
                }
            )

        return _json_response(
            {
                "dataset_id": resolved_id,
                "use_rep": "X_umap",
                "color_by": color_by,
                "points": points,
                "missing": missing,
            }
        )
    except Exception as exc:
        return _json_response({"error": str(exc)}, 400)


@app.post("/api/datasets/upload")
def upload_dataset():
    if "file" not in request.files:
        return _json_response({"error": "file is required"}, 400)
    handle = request.files["file"]
    if not handle.filename:
        return _json_response({"error": "filename is required"}, 400)
    filename = secure_filename(handle.filename)
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return _json_response({"error": "only .h5ad is supported"}, 400)
    target_path = DATA_DIR / filename
    if target_path.exists():
        return _json_response({"error": "dataset already exists"}, 409)
    handle.save(str(target_path))
    dataset_id = _dataset_id_from_path(target_path)
    _DATASET_CACHE.pop(dataset_id, None)
    return _json_response({"status": "uploaded", "dataset_id": dataset_id}, 201)


@app.delete("/api/datasets/<dataset_id>")
def delete_dataset(dataset_id: str):
    try:
        dataset_path = _resolve_dataset_path(dataset_id)
    except Exception as exc:
        return _json_response({"error": str(exc)}, 404)

    dataset_id = _dataset_id_from_path(dataset_path)
    dataset_path.unlink(missing_ok=True)

    _DATASET_CACHE.pop(dataset_id, None)
    for key in list(_INDEX_CACHE.keys()):
        if key[0] == dataset_id:
            _INDEX_CACHE.pop(key, None)

    for index_file in INDEX_DIR.glob(f"{dataset_id}_*.index*"):
        index_file.unlink(missing_ok=True)

    return _json_response({"status": "deleted", "dataset_id": dataset_id})


@app.post("/api/index/build")
def build_index():
    try:
        payload = _request_payload()
        dataset_id = _normalize_dataset_id(payload.get("dataset_id"))
        dataset_path = _resolve_dataset_path(dataset_id)
        resolved_id = _dataset_id_from_path(dataset_path)
        use_rep = payload.get("use_rep") or DEFAULT_USE_REP
        index_config = _index_config_from_payload(payload)
        loader = _get_loader(resolved_id, dataset_path)
        index_config = _normalize_pq_config_for_dim(
            index_config, loader.vector_dim(use_rep)
        )
        indexer = _get_indexer(
            resolved_id, use_rep, index_config, build_if_missing=True
        )
        return _json_response(
            {
                "status": "built",
                "dataset_id": resolved_id,
                "use_rep": use_rep,
                "backend": indexer.backend,
                "index_type": indexer.index_type,
                "index_metric": indexer.metric,
                "index_config": indexer.config_summary,
            }
        )
    except ImportError as exc:
        return _json_response({"error": str(exc)}, 400)
    except Exception as exc:
        return _json_response({"error": str(exc)}, 400)


@app.route("/api/search", methods=["GET", "POST"])
def search():
    try:
        total_start_time = time.perf_counter()
        payload = _request_payload()
        dataset_id = _normalize_dataset_id(payload.get("dataset_id"))
        dataset_path = _resolve_dataset_path(dataset_id)
        resolved_id = _dataset_id_from_path(dataset_path)
        use_rep = payload.get("use_rep") or DEFAULT_USE_REP
        index_config = _index_config_from_payload(payload)

        cell_index_value = payload.get("cell_index")
        cell_id_value = payload.get("cell_id")
        k = _parse_int(payload, "k", default=DEFAULT_TOP_K)
        include_self = _parse_bool(payload.get("include_self"), default=False)

        loader = _get_loader(resolved_id, dataset_path)
        index_config = _normalize_pq_config_for_dim(
            index_config, loader.vector_dim(use_rep)
        )
        if cell_index_value is None or cell_index_value == "":
            if cell_id_value is None or str(cell_id_value).strip() == "":
                raise ValueError("cell_index or cell_id is required")
            cell_index = loader.cell_index_from_id(str(cell_id_value))
        else:
            cell_index = _parse_int(payload, "cell_index", required=True)

        if k <= 0:
            return _json_response({"error": "k must be a positive integer"}, 400)
        if k > MAX_TOP_K:
            return _json_response({"error": f"k must not exceed {MAX_TOP_K}"}, 400)

        index_prepare_start_time = time.perf_counter()
        indexer = _get_indexer(
            resolved_id, use_rep, index_config, build_if_missing=True
        )
        index_prepare_ms = round(
            (time.perf_counter() - index_prepare_start_time) * 1000.0, 2
        )
        query_vector = loader.get_vector(cell_index, use_rep=use_rep)
        search_k = min(k + 1 if not include_self else k, loader.n_cells)
        start_time = time.perf_counter()
        distances, indices = indexer.search(query_vector, search_k)
        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        results = []
        metric = indexer.metric
        for idx, dist in zip(indices.tolist(), distances.tolist()):
            idx = int(idx)
            distance = float(dist)
            if not include_self and idx == cell_index:
                continue

            cell_info = loader.get_cell_info(idx)
            results.append(
                {
                    "rank": len(results) + 1,
                    "cell_index": idx,
                    "cell_id": cell_info.get("cell_id"),
                    "cell_type": cell_info.get("cell_type", "unknown"),
                    "distance": round(distance, 6),
                    "similarity_score": round(
                        _similarity_from_distance(distance, metric), 6
                    ),
                    "metadata": cell_info,
                }
            )

            if len(results) == k:
                break

        return _json_response(
            {
                "dataset_id": resolved_id,
                "query_cell": cell_index,
                "cell_id": loader.get_cell_info(cell_index).get("cell_id"),
                "k": k,
                "include_self": include_self,
                "use_rep": use_rep,
                "index_backend": indexer.backend,
                "index_type": indexer.index_type,
                "index_metric": metric,
                "index_config": indexer.config_summary,
                "elapsed_ms": elapsed_ms,
                "search_elapsed_ms": elapsed_ms,
                "index_prepare_ms": index_prepare_ms,
                "total_elapsed_ms": round(
                    (time.perf_counter() - total_start_time) * 1000.0, 2
                ),
                "results": results,
            }
        )
    except FileNotFoundError as exc:
        return _json_response(
            {
                "error": str(exc),
                "action": "select_dataset",
                "message": "数据集不存在，请先上传或选择已有数据集",
            },
            404,
        )
    except ImportError as exc:
        return _json_response({"error": str(exc)}, 400)
    except (TypeError, ValueError, KeyError, IndexError) as exc:
        return _json_response({"error": str(exc)}, 400)
    except RuntimeError as exc:
        return _json_response({"error": str(exc)}, 503)


@app.post("/api/benchmark")
def benchmark_api():
    try:
        payload = _request_payload()
        dataset_id = _normalize_dataset_id(payload.get("dataset_id"))
        dataset_path = _resolve_dataset_path(dataset_id)
        resolved_id = _dataset_id_from_path(dataset_path)
        use_rep = payload.get("use_rep") or DEFAULT_USE_REP

        query_count = _parse_int(payload, "query_count", default=500)
        k = _parse_int(payload, "k", default=DEFAULT_TOP_K)

        if query_count <= 0:
            return _json_response({"error": "query_count must be a positive integer"}, 400)
        if k <= 0:
            return _json_response({"error": "k must be a positive integer"}, 400)
        if k > MAX_TOP_K:
            return _json_response({"error": f"k must not exceed {MAX_TOP_K}"}, 400)

        loader = _get_loader(resolved_id, dataset_path)
        vectors = loader.get_vectors(use_rep)
        n_cells = loader.n_cells
        pq_m = _suggest_pq_m(vectors.shape[1])

        query_count = min(query_count, n_cells)
        if query_count <= 0:
            return _json_response({"error": "dataset has no cells"}, 400)

        max_k = min(k, n_cells, 100)
        if max_k <= 0:
            return _json_response({"error": "k must be <= cell count"}, 400)

        rng = np.random.default_rng()
        if query_count == n_cells:
            query_indices = np.arange(n_cells)
        else:
            query_indices = rng.choice(n_cells, size=query_count, replace=False)
        queries = vectors[query_indices]

        standard_ks = [10, 20, 50, 100]
        ks = [value for value in standard_ks if value <= max_k]

        algorithms = [
            {
                "id": "brute",
                "label": "精确搜索 (暴力)",
                "config": IndexConfig(backend="numpy", index_type="brute", metric="l2"),
            },
            {
                "id": "faiss_hnsw",
                "label": "FAISS-HNSW",
                "config": IndexConfig(
                    backend="faiss",
                    index_type="hnsw",
                    metric="l2",
                    m=16,
                    ef_construction=200,
                    ef_search=50,
                ),
            },
            {
                "id": "hnswlib",
                "label": "HNSWLIB",
                "config": IndexConfig(
                    backend="hnswlib",
                    index_type="hnsw",
                    metric="l2",
                    m=16,
                    ef_construction=200,
                    ef_search=50,
                ),
            },
            {
                "id": "faiss_ivf",
                "label": "FAISS-IVF",
                "config": IndexConfig(
                    backend="faiss",
                    index_type="ivf_flat",
                    metric="l2",
                    nlist=100,
                    nprobe=10,
                ),
            },
            {
                "id": "faiss_pq",
                "label": "FAISS-PQ",
                "config": IndexConfig(
                    backend="faiss",
                    index_type="pq",
                    metric="l2",
                    pq_m=pq_m,
                    pq_nbits=8,
                ),
            },
        ]

        # Baseline: brute-force for ground truth
        brute_algo = algorithms[0]
        brute_indexer = _get_benchmark_indexer(
            resolved_id, use_rep, vectors, brute_algo["config"]
        )

        truth_indices: List[np.ndarray] = []
        start_time = time.perf_counter()
        for query in queries:
            _, idx = brute_indexer.search(query, max_k)
            truth_indices.append(idx)
        brute_elapsed = time.perf_counter() - start_time

        results: List[Dict[str, Any]] = []

        brute_recalls = []
        for k_value in standard_ks:
            if k_value <= max_k:
                brute_recalls.append(100.0)
            else:
                brute_recalls.append(None)

        results.append(
            {
                "id": brute_algo["id"],
                "label": brute_algo["label"],
                "available": True,
                "avg_ms": round(brute_elapsed / query_count * 1000.0, 3),
                "recall_curve": brute_recalls,
            }
        )

        # Evaluate approximate algorithms
        for algo in algorithms[1:]:
            config = algo["config"]
            try:
                indexer = _get_benchmark_indexer(
                    resolved_id, use_rep, vectors, config
                )
            except ImportError as exc:
                results.append(
                    {
                        "id": algo["id"],
                        "label": algo["label"],
                        "available": False,
                        "error": str(exc),
                        "avg_ms": None,
                        "recall_curve": [None for _ in standard_ks],
                    }
                )
                continue

            recall_sums = {k_value: 0.0 for k_value in ks}
            start_time = time.perf_counter()
            for idx, query in enumerate(queries):
                _, pred_indices = indexer.search(query, max_k)
                truth = truth_indices[idx]
                for k_value in ks:
                    pred_set = set(pred_indices[:k_value].tolist())
                    truth_set = set(truth[:k_value].tolist())
                    recall_sums[k_value] += len(pred_set.intersection(truth_set)) / k_value
            elapsed = time.perf_counter() - start_time

            recall_curve: List[Optional[float]] = []
            for k_value in standard_ks:
                if k_value <= max_k:
                    recall_value = recall_sums[k_value] / query_count * 100.0
                    recall_curve.append(round(recall_value, 2))
                else:
                    recall_curve.append(None)

            results.append(
                {
                    "id": algo["id"],
                    "label": algo["label"],
                    "available": True,
                    "avg_ms": round(elapsed / query_count * 1000.0, 3),
                    "recall_curve": recall_curve,
                }
            )

        created_at = time.strftime("%Y-%m-%d %H:%M:%S")
        entry = {
            "created_at": created_at,
            "dataset_id": resolved_id,
            "use_rep": use_rep,
            "query_count": query_count,
            "k": k,
            "k_values": standard_ks,
            "algorithms": results,
        }
        history_id = _append_benchmark_history(entry)

        return _json_response({
            "dataset_id": resolved_id,
            "use_rep": use_rep,
            "query_count": query_count,
            "k": k,
            "k_values": standard_ks,
            "algorithms": results,
            "created_at": created_at,
            "history_id": history_id,
        })
    except ImportError as exc:
        return _json_response({"error": str(exc)}, 400)
    except (TypeError, ValueError, KeyError, IndexError) as exc:
        return _json_response({"error": str(exc)}, 400)
    except RuntimeError as exc:
        return _json_response({"error": str(exc)}, 503)


@app.get("/api/benchmark/history")
def benchmark_history():
    try:
        payload = dict(request.args)
        limit = payload.get("limit")
        history = _load_benchmark_history()
        if limit:
            try:
                limit_value = int(limit)
                if limit_value > 0:
                    history = history[-limit_value:]
            except ValueError:
                pass
        return _json_response({"history": history})
    except Exception as exc:
        return _json_response({"error": str(exc)}, 400)


def _json_response(payload: Any, status_code: int = 200):
    return jsonify(_json_safe(payload)), status_code


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _request_payload() -> Dict[str, Any]:
    if request.method == "POST":
        return dict(request.get_json(silent=True) or {})
    return dict(request.args)


def _parse_int(
    payload: Dict[str, Any], key: str, required: bool = False, default: Optional[int] = None
) -> int:
    value = payload.get(key)
    if value is None or value == "":
        if required:
            raise ValueError(f"{key} is required")
        if default is None:
            raise ValueError(f"{key} is required")
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer") from exc


def _parse_optional_int(payload: Dict[str, Any], key: str) -> Optional[int]:
    value = payload.get(key)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer") from exc


def _parse_bool(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError("include_self must be a boolean")


def _similarity_from_distance(distance: float, metric: str) -> float:
    if metric == "cosine":
        return 1.0 - distance
    if metric == "ip":
        return -distance
    return 1.0 / (1.0 + distance)


def main() -> None:
    port = int(os.getenv("PORT", "5000"))
    app.run(debug=True, port=port)


if __name__ == "__main__":
    main()
