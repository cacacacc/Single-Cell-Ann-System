from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

    if not overrides:
        return config
    return config.update(**overrides)


def _index_path(dataset_id: str, use_rep: str) -> Path:
    return INDEX_DIR / f"{dataset_id}_{use_rep}.index"


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
        "vector_dim": loader.vector_dim(use_rep),
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
        return _json_response({"error": str(exc)}, 503)


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

        indexer = _get_indexer(
            resolved_id, use_rep, index_config, build_if_missing=True
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
                "results": results,
            }
        )
    except ImportError as exc:
        return _json_response({"error": str(exc)}, 400)
    except (TypeError, ValueError, KeyError, IndexError) as exc:
        return _json_response({"error": str(exc)}, 400)
    except RuntimeError as exc:
        return _json_response({"error": str(exc)}, 503)


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

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(debug=True, port=port)