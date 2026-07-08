from __future__ import annotations

import os

# ── macOS 并行库冲突防护（仅 macOS 生效）────────────────────────────────────
# macOS 上 FAISS(OpenMP) + NumPy(OpenBLAS) 会同时向系统注册多线程运行时，
# 互相争夺 pthread 资源，导致随机 Segmentation Fault。
# 以下变量必须在 numpy / faiss 等任何科学计算库导入之前设置。
# 使用 sys.platform 判断，Windows / Linux 不受影响，保留多核并行性能。
import sys as _sys
if _sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # 允许多份 OpenMP 共存
    os.environ.setdefault("OMP_NUM_THREADS", "1")            # 限制 OpenMP 只用单线程
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")       # 限制 OpenBLAS 只用单线程
    os.environ.setdefault("MKL_NUM_THREADS", "1")            # 限制 MKL 只用单线程
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")     # 限制 macOS Accelerate 框架
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")        # 限制 numexpr
# ─────────────────────────────────────────────────────────────────────────────

import json
import logging
import math
import time
from datetime import timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

# 自动加载项目根目录的 .env 文件（Windows/Mac/Linux 通用）
# 需安装：pip install python-dotenv
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
except ImportError:
    pass  # python-dotenv 未安装时跳过，不影响正常运行

logger = logging.getLogger(__name__)

import numpy as np
from anndata import read_h5ad
from flask import Flask, Response, jsonify, redirect, render_template, request, session, stream_with_context, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename

from backend.ann_indexer import ANNIndexer, IndexConfig, PCAReducer
from backend.data_reader import DataLoader
from backend.vector_store import (
    CellVectorStore,
    get_or_create_store,
    is_chroma_available,
    DEFAULT_CHROMA_DIR,
)
from backend.llm_client import LLMClient, get_llm_client, reset_llm_client
from backend.rag_engine import (
    RAGEngine,
    get_or_create_engine,
    _CHAT_HISTORY,
)
from backend.natural_language_query import (
    DEFAULT_LIMIT as NLQ_DEFAULT_LIMIT,
    execute_natural_cell_query,
    parse_natural_cell_query,
)
from backend.merged_loader import (
    MergedDataLoader,
    MergedDatasetConfig,
    get_merged_dir,
    list_merged_configs,
    load_merged_config,
    save_merged_config,
    delete_merged_config,
    _make_merged_id,
)
from backend.user_store import (
    AuthenticationError,
    DuplicateUserError,
    LastAdminError,
    UserStore,
    UserStoreError,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "indexes"
CHROMA_DIR = BASE_DIR / DEFAULT_CHROMA_DIR
MERGED_DIR = DATA_DIR / ".merged"

DEFAULT_DATA_PATH = "data/liver.h5ad"
DEFAULT_USE_REP = "X_pca"
DEFAULT_TOP_K = 10
MAX_TOP_K = 100
FILTER_SEARCH_MULTIPLIER = 10
ALLOWED_EXTENSIONS = {".h5ad"}

_DATASET_CACHE: Dict[str, DataLoader] = {}
_INDEX_CACHE: Dict[Tuple[str, str], Tuple[ANNIndexer, Tuple[Any, ...]]] = {}
_BENCHMARK_INDEX_CACHE: Dict[Tuple[str, str, Tuple[Any, ...]], ANNIndexer] = {}
_MERGED_CACHE: Dict[str, MergedDataLoader] = {}
_BENCHMARK_HISTORY_PATH = DATA_DIR / "benchmark_history.json"
USER_DB_PATH: Optional[Path] = None
DEFAULT_ADMIN_USERNAME = os.getenv("DEFAULT_ADMIN_USERNAME", "admin")
DEFAULT_ADMIN_PASSWORD = os.getenv("DEFAULT_ADMIN_PASSWORD", "Admin@123456")
DEFAULT_ADMIN_NAME = os.getenv("DEFAULT_ADMIN_NAME", "系统管理员")
DEFAULT_ADMIN_EMAIL = os.getenv("DEFAULT_ADMIN_EMAIL", "admin@example.com")
DEFAULT_NEW_USER_PASSWORD = os.getenv("DEFAULT_NEW_USER_PASSWORD", "Nankai@123")
_USER_STORE: Optional[UserStore] = None
_USER_STORE_PATH: Optional[Path] = None
ViewFunc = TypeVar("ViewFunc", bound=Callable[..., Any])


def _normalize_dataset_id(dataset_id: Optional[str]) -> Optional[str]:
    if dataset_id is None:
        return None
    cleaned = dataset_id.strip()
    if not cleaned or "/" in cleaned or "\\" in cleaned:
        raise ValueError("dataset_id is invalid")
    return cleaned


def _is_merged_dataset(dataset_id: str) -> bool:
    return (MERGED_DIR / f"{dataset_id}.json").exists()


def _get_merged_loader(merged_id: str) -> MergedDataLoader:
    if merged_id in _MERGED_CACHE:
        return _MERGED_CACHE[merged_id]
    config = load_merged_config(DATA_DIR, merged_id)
    source_loaders = {}
    for ds_id in config.source_datasets:
        source_loaders[ds_id] = _get_loader(ds_id)
    merged = MergedDataLoader(config, source_loaders)
    _MERGED_CACHE[merged_id] = merged
    return merged


def _merged_dataset_payload(config: MergedDatasetConfig) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "id": config.merged_id,
        "filename": f"[合并] {config.name}",
        "is_merged": True,
        "source_datasets": config.source_datasets,
        "use_rep": config.use_rep,
        "created_at": config.created_at,
        "size_mb": 0,
        "modified_at": config.created_at,
    }
    if config.merged_id in _MERGED_CACHE:
        merged = _MERGED_CACHE[config.merged_id]
        payload["n_cells"] = merged.n_cells
        payload["n_genes"] = merged.n_genes
        payload["available_reps"] = merged.available_reps
        payload["obs_columns"] = merged.obs_columns
    else:
        try:
            merged = _get_merged_loader(config.merged_id)
            payload["n_cells"] = merged.n_cells
            payload["n_genes"] = merged.n_genes
            payload["available_reps"] = merged.available_reps
            payload["obs_columns"] = merged.obs_columns
        except Exception as exc:
            payload["error"] = str(exc)
    _migrate_old_index(config.merged_id)
    indices = _list_index_names(config.merged_id)
    payload["index_status"] = "ready" if indices else "missing"
    return payload


def _resolve_dataset_path(dataset_id: Optional[str]) -> Path:
    if dataset_id:
        name = _normalize_dataset_id(dataset_id)
        assert name is not None
        if _is_merged_dataset(name):
            return MERGED_DIR / f"{name}.json"
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


def _normalize_filter(payload: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    field = payload.get("filter_field") or payload.get("condition_field")
    value = payload.get("filter_value") or payload.get("condition_value")
    if field is None or str(field).strip() in {"", "all", "__none__"}:
        return None, None
    field_text = str(field).strip()
    if "/" in field_text or "\\" in field_text:
        raise ValueError("filter_field is invalid")
    if value is None or str(value).strip() == "":
        raise ValueError("filter_value is required when filter_field is set")
    return field_text, str(value).strip()


def _metadata_matches_filter(cell_info: Dict[str, Any], field: Optional[str], value: Optional[str]) -> bool:
    if not field:
        return True
    actual = cell_info.get(field)
    if actual is None:
        return False
    return str(actual).strip().lower() == str(value).strip().lower()


def _index_dir(dataset_id: str) -> Path:
    return INDEX_DIR / dataset_id


def _index_path(dataset_id: str, index_name: str) -> Path:
    return _index_dir(dataset_id) / f"{index_name}.index"


def _default_index_name(use_rep: str, index_type: str, metric: str) -> str:
    """Auto-generate a default index name from its key characteristics."""
    return f"{use_rep}_{index_type}_{metric}"


def _list_index_names(dataset_id: str) -> List[Dict[str, Any]]:
    """List all saved indices for a dataset. Returns [{name, path, config}, ...]."""
    index_dir = _index_dir(dataset_id)
    if not index_dir.exists():
        return []

    results: List[Dict[str, Any]] = []
    for idx_path in sorted(index_dir.glob("*.index")):
        name = idx_path.stem
        backup_path = _backup_index_path(idx_path)
        # Try .npz backup first, then the .index file itself (numpy backend)
        npz_path = backup_path if backup_path.exists() else None
        if npz_path is None and idx_path.exists():
            npz_path = idx_path  # numpy backend stores archive as the .index file
        config = None

        if npz_path is not None:
            try:
                with np.load(npz_path, allow_pickle=False) as data:
                    if "config_json" in data:
                        config = json.loads(str(np.asarray(data["config_json"]).item()))
                    elif "vectors" in data:
                        # Valid archive with vectors but no config_json (legacy)
                        config = {
                            "backend": str(np.asarray(data.get("backend", "auto")).item()),
                            "index_type": str(np.asarray(data.get("index_type", "flat")).item()),
                            "metric": str(np.asarray(data.get("metric", "l2")).item()),
                        }
            except (KeyError, ValueError, OSError):
                pass

        results.append({
            "name": name,
            "config": config,
            "ready": True,
        })
    return results


def _migrate_old_index(dataset_id: str) -> None:
    """Migrate old flat-format index files to new subdirectory layout."""
    index_dir = _index_dir(dataset_id)
    new_path = index_dir / "default.index"
    if new_path.exists() or _backup_index_path(new_path).exists():
        return  # already migrated

    for old_file in INDEX_DIR.glob(f"{dataset_id}_*.index*"):
        if old_file.is_dir():
            continue
        index_dir.mkdir(parents=True, exist_ok=True)
        if old_file.suffix == ".npz":
            # {dataset_id}_{use_rep}.index.npz
            stem = old_file.name[:-4]  # remove .npz
            if stem.endswith(".index"):
                target = _backup_index_path(new_path)
            else:
                continue
        elif old_file.suffix == ".index":
            target = new_path
        else:
            continue
        if not target.exists():
            old_file.rename(target)


def _read_stored_index_config(dataset_id: str, index_name: str) -> Optional[Dict[str, Any]]:
    """Read index config from an existing index file without loading vectors."""
    index_path = _index_path(dataset_id, index_name)
    backup_path = _backup_index_path(index_path)

    # Prefer reading from the .npz backup (contains full config_json)
    npz_path = backup_path if backup_path.exists() else None
    if npz_path is None and index_path.exists():
        # Try the .index file itself (numpy backend stores archive directly)
        npz_path = index_path
    if npz_path is None and index_path.exists():
        # FAISS binary index without backup — try to find config from cache
        cached = _INDEX_CACHE.get((dataset_id, index_name))
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


def _get_loader(dataset_id: str, dataset_path: Optional[Path] = None):
    if dataset_id in _DATASET_CACHE:
        return _DATASET_CACHE[dataset_id]
    if _is_merged_dataset(dataset_id):
        merged = _get_merged_loader(dataset_id)
        _DATASET_CACHE[dataset_id] = merged
        return merged
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
    index_name: Optional[str] = None,
) -> ANNIndexer:
    """Return a ready-to-search ANNIndexer.

    If *index_name* is given, the index is loaded from (or saved to) the
    per-dataset subdirectory ``INDEX_DIR/<dataset_id>/<index_name>.index``.
    Otherwise the index is built on-the-fly without persisting to disk.
    """
    if index_name is not None:
        # --- named (persisted) path ---
        cache_key = (dataset_id, index_name)
        config_key = _config_cache_key(index_config)
        cached = _INDEX_CACHE.get(cache_key)
        if cached is not None:
            cached_indexer, cached_key = cached
            if cached_key == config_key:
                return cached_indexer

        loader = _get_loader(dataset_id)
        indexer = ANNIndexer(dim=loader.vector_dim(use_rep), config=index_config)
        index_path = _index_path(dataset_id, index_name)
        backup_path = _backup_index_path(index_path)

        if index_path.exists() or backup_path.exists():
            try:
                indexer.load_index(index_path)
            except ValueError as exc:
                if "config mismatch" in str(exc).lower() and build_if_missing:
                    vectors = loader.get_vectors(use_rep)
                    indexer.build_index(vectors)
                    indexer.save_index(index_path, use_rep=use_rep)
                else:
                    raise
        elif build_if_missing:
            vectors = loader.get_vectors(use_rep)
            indexer.build_index(vectors)
            indexer.save_index(index_path, use_rep=use_rep)
        else:
            raise FileNotFoundError("Index not found")

        _INDEX_CACHE[cache_key] = (indexer, config_key)
        return indexer

    # --- unnamed (on-the-fly) path — don't persist, but cache in memory ---
    config_key = _config_cache_key(index_config)
    anon_cache_key = (dataset_id, use_rep, config_key)
    cached_anon = _BENCHMARK_INDEX_CACHE.get(anon_cache_key)
    if cached_anon is not None:
        return cached_anon
    loader = _get_loader(dataset_id)
    indexer = ANNIndexer(dim=loader.vector_dim(use_rep), config=index_config)
    vectors = loader.get_vectors(use_rep)
    indexer.build_index(vectors)
    _BENCHMARK_INDEX_CACHE[anon_cache_key] = indexer
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
        # 优先复用已缓存的 DataLoader，避免重复打开 h5ad 文件（磁盘 IO）
        if dataset_id in _DATASET_CACHE:
            loader = _DATASET_CACHE[dataset_id]
            payload.update(
                {
                    "n_cells": int(loader.n_cells),
                    "n_genes": int(loader.adata.n_vars),
                    "available_reps": list(loader.adata.obsm.keys()),
                    "obs_columns": list(loader.adata.obs.columns),
                }
            )
        else:
            adata = read_h5ad(dataset_path, backed="r")
            payload.update(
                {
                    "n_cells": int(adata.n_obs),
                    "n_genes": int(adata.n_vars),
                    "available_reps": list(adata.obsm.keys()),
                    "obs_columns": list(adata.obs.columns),
                }
            )
            if getattr(adata, "file", None) is not None:
                adata.file.close()
    except Exception as exc:
        payload["error"] = str(exc)

    stat = dataset_path.stat()
    payload["size_mb"] = round(stat.st_size / (1024 * 1024), 2)
    payload["modified_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))

    # Check for indices in new subdirectory layout + migrate old flat files
    _migrate_old_index(dataset_id)
    indices = _list_index_names(dataset_id)
    payload["index_status"] = "ready" if indices else "missing"
    payload["index_backend"] = None
    if indices:
        first = indices[0]
        if first.get("config") and first["config"].get("backend"):
            payload["index_backend"] = first["config"]["backend"]

    return payload


def _metadata_payload(
    dataset_id: str,
    use_rep: str,
    index_config: IndexConfig,
    index_name: Optional[str] = None,
) -> Dict[str, Any]:
    loader = _get_loader(dataset_id)
    vector_dim = loader.vector_dim(use_rep)
    index_config = _normalize_pq_config_for_dim(index_config, vector_dim)
    _migrate_old_index(dataset_id)
    indices = _list_index_names(dataset_id)
    index_ready = len(indices) > 0
    effective_name = index_name or (indices[0]["name"] if indices else None)
    index_path = _index_path(dataset_id, effective_name) if effective_name else _index_path(dataset_id, "none")
    payload: Dict[str, Any] = {
        "dataset_id": dataset_id,
        "data_path": str(_resolve_dataset_path(dataset_id)),
        "index_path": str(index_path) if effective_name else "",
        "use_rep": use_rep,
        "ready": index_ready,
        "indices": indices,
        "n_cells": loader.n_cells,
        "n_genes": loader.n_genes,
        "vector_dim": vector_dim,
        "pq_m_options": _pq_m_options(vector_dim),
        "suggested_pq_m": _suggest_pq_m(vector_dim),
        "available_reps": loader.available_reps,
        "obs_columns": loader.obs_columns,
        "index_config": index_config.to_dict(),
    }
    if effective_name:
        cache_key = (dataset_id, effective_name)
        cached_entry = _INDEX_CACHE.get(cache_key)
        if cached_entry is not None:
            cached_indexer, _ = cached_entry
            payload["index_backend"] = cached_indexer.backend
            payload["index_type"] = cached_indexer.index_type
            payload["index_metric"] = cached_indexer.metric
            payload["index_config"] = cached_indexer.config_summary
            return payload
        saved_config = _read_stored_index_config(dataset_id, effective_name)
        if saved_config is not None:
            payload["index_backend"] = saved_config.get("backend")
            payload["index_type"] = saved_config.get("index_type")
            payload["index_metric"] = saved_config.get("metric")
            payload["index_config"] = saved_config
            return payload

    payload["index_backend"] = None
    payload["index_type"] = index_config.index_type
    payload["index_metric"] = index_config.metric
    return payload


app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-change-me")
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    PERMANENT_SESSION_LIFETIME=timedelta(days=7),
)
CORS(app)
DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)


def create_app() -> Flask:
    return app


def _configured_user_db_path() -> Path:
    if USER_DB_PATH is not None:
        return Path(USER_DB_PATH)
    raw_path = os.getenv("USER_DB_PATH")
    if raw_path:
        return Path(raw_path)
    return DATA_DIR / "users.sqlite3"


def _user_store() -> UserStore:
    global _USER_STORE, _USER_STORE_PATH
    db_path = _configured_user_db_path()
    if _USER_STORE is None or _USER_STORE_PATH != db_path:
        store = UserStore(db_path)
        store.init_db(
            default_admin_username=DEFAULT_ADMIN_USERNAME,
            default_admin_password=DEFAULT_ADMIN_PASSWORD,
            default_admin_name=DEFAULT_ADMIN_NAME,
            default_admin_email=DEFAULT_ADMIN_EMAIL,
        )
        _USER_STORE = store
        _USER_STORE_PATH = db_path
    return _USER_STORE


def _current_user() -> Optional[Dict[str, Any]]:
    user_id = session.get("user_id")
    if user_id is None:
        return None
    try:
        user = _user_store().get_user(int(user_id))
    except (TypeError, ValueError):
        session.clear()
        return None
    if user is None or not user["is_active"]:
        session.clear()
        return None
    return user


def _set_session_user(user: Dict[str, Any], remember: bool = False) -> None:
    session.clear()
    session.permanent = bool(remember)
    session["user_id"] = user["id"]
    session["role"] = user["role"]


def _auth_required_response():
    if request.path.startswith("/api/"):
        return _json_response({"error": "请先登录"}, 401)
    return redirect(url_for("login", next=request.path))


def _forbidden_response():
    if request.path.startswith("/api/"):
        return _json_response({"error": "需要管理员权限"}, 403)
    return "需要管理员权限", 403


def _user_error_response(exc: Exception, default_status: int = 400):
    if isinstance(exc, AuthenticationError):
        status = 401
    elif isinstance(exc, DuplicateUserError):
        status = 409
    elif isinstance(exc, LastAdminError):
        status = 403
    else:
        status = default_status
    return _json_response({"error": str(exc)}, status)


def _payload_bool(value: Any, default: bool = True) -> bool:
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
    return bool(value)


def login_required(view: ViewFunc) -> ViewFunc:
    @wraps(view)
    def wrapped(*args: Any, **kwargs: Any):
        if _current_user() is None:
            return _auth_required_response()
        return view(*args, **kwargs)

    return wrapped  # type: ignore[return-value]


def admin_required(view: ViewFunc) -> ViewFunc:
    @wraps(view)
    def wrapped(*args: Any, **kwargs: Any):
        user = _current_user()
        if user is None:
            return _auth_required_response()
        if user["role"] != "admin":
            return _forbidden_response()
        return view(*args, **kwargs)

    return wrapped  # type: ignore[return-value]


@app.before_request
def ensure_user_store_ready() -> None:
    _user_store()


@app.context_processor
def inject_current_user() -> Dict[str, Any]:
    return {"current_user": _current_user()}


# 路由 1：系统大屏首页 (登录后看到的)
@app.route('/')
@login_required
def index():
    return render_template('index.html')

# 路由 2：登录/注册页 (独立页面，不带侧边栏)
@app.route('/login')
def login():
    if _current_user() is not None:
        next_path = request.args.get("next") or url_for("index")
        return redirect(next_path)
    return render_template('login.html')

@app.route('/data')
@login_required
def data_manage():
    return render_template('data_manage.html')

# ================================
# 相似检索页路由
# ================================
@app.route('/search')
@login_required
def similarity_search():
    return render_template('search.html')

# ================================
# 用户管理页路由 (仅管理员可见)
# ================================
@app.route('/users')
@admin_required
def user_management():
    return render_template('users.html')

@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html')

# ================================
# 性能评测页路由
# ================================
@app.route('/benchmark')
@login_required
def benchmark():
    return render_template('benchmark.html')


# ================================
# AI 细胞助手聊天页路由
# ================================
@app.route('/chat')
@login_required
def chat_page():
    return render_template('chat.html')


@app.post("/api/auth/register")
def register_api():
    try:
        payload = _request_payload()
        user = _user_store().create_user(
            username=str(payload.get("username") or ""),
            password=str(payload.get("password") or ""),
            full_name=str(payload.get("full_name") or ""),
            email=str(payload.get("email") or ""),
            role="user",
            is_active=True,
        )
        _set_session_user(user, remember=bool(payload.get("remember")))
        _notify_admins("info", "新用户注册", f"用户 {user['username']} 已注册，请等待管理员分配权限")
        return _json_response({"status": "registered", "user": user}, 201)
    except UserStoreError as exc:
        return _user_error_response(exc)


@app.post("/api/auth/login")
def login_api():
    try:
        payload = _request_payload()
        user = _user_store().authenticate(
            str(payload.get("username") or ""),
            str(payload.get("password") or ""),
        )
        _set_session_user(user, remember=bool(payload.get("remember")))
        return _json_response({"status": "logged_in", "user": user})
    except UserStoreError as exc:
        return _user_error_response(exc)


@app.post("/api/auth/logout")
def logout_api():
    session.clear()
    return _json_response({"status": "logged_out"})


@app.get("/api/auth/me")
def me_api():
    user = _current_user()
    return _json_response({"authenticated": user is not None, "user": user})


@app.patch("/api/profile")
@login_required
def update_profile_api():
    user = _current_user()
    assert user is not None
    try:
        payload = _request_payload()
        updated = _user_store().update_user(
            user["id"],
            full_name=str(payload.get("full_name") or ""),
            email=str(payload.get("email") or ""),
        )
        return _json_response({"status": "updated", "user": updated})
    except UserStoreError as exc:
        return _user_error_response(exc)


@app.post("/api/profile/password")
@login_required
def change_password_api():
    user = _current_user()
    assert user is not None
    try:
        payload = _request_payload()
        _user_store().change_password(
            user["id"],
            str(payload.get("old_password") or ""),
            str(payload.get("new_password") or ""),
        )
        return _json_response({"status": "password_changed"})
    except UserStoreError as exc:
        return _user_error_response(exc)


@app.get("/api/profile/search-snapshots")
@login_required
def list_search_snapshots_api():
    user = _current_user()
    assert user is not None
    limit = _parse_int(_request_payload(), "limit", default=20)
    snapshots = _user_store().list_search_snapshots(user["id"], limit=limit)
    return _json_response({"snapshots": snapshots, "total": len(snapshots)})


@app.delete("/api/profile/search-snapshots/<int:snapshot_id>")
@login_required
def delete_search_snapshot_api(snapshot_id: int):
    user = _current_user()
    assert user is not None
    try:
        _user_store().delete_search_snapshot(user["id"], snapshot_id)
        return _json_response({"status": "deleted", "snapshot_id": snapshot_id})
    except UserStoreError as exc:
        return _user_error_response(exc)


@app.get("/api/users")
@admin_required
def list_users_api():
    query = request.args.get("q", "")
    users = _user_store().list_users(query=query)
    return _json_response({"users": users, "total": len(users)})


@app.post("/api/users")
@admin_required
def create_user_api():
    try:
        payload = _request_payload()
        password = str(payload.get("password") or DEFAULT_NEW_USER_PASSWORD)
        user = _user_store().create_user(
            username=str(payload.get("username") or ""),
            password=password,
            full_name=str(payload.get("full_name") or ""),
            email=str(payload.get("email") or ""),
            role=str(payload.get("role") or "user"),
            is_active=_payload_bool(payload.get("is_active"), default=True),
        )
        return _json_response({"status": "created", "user": user, "initial_password": password}, 201)
    except UserStoreError as exc:
        return _user_error_response(exc)


@app.patch("/api/users/<int:user_id>")
@admin_required
def update_user_api(user_id: int):
    current = _current_user()
    assert current is not None
    try:
        payload = _request_payload()
        if user_id == current["id"] and payload.get("is_active") is False:
            return _json_response({"error": "不能禁用当前登录管理员"}, 400)
        updated = _user_store().update_user(
            user_id,
            full_name=payload.get("full_name") if "full_name" in payload else None,
            email=payload.get("email") if "email" in payload else None,
            role=payload.get("role") if "role" in payload else None,
            is_active=_payload_bool(payload.get("is_active")) if "is_active" in payload else None,
        )
        if user_id == current["id"]:
            session["role"] = updated["role"]
        return _json_response({"status": "updated", "user": updated})
    except UserStoreError as exc:
        return _user_error_response(exc)


@app.post("/api/users/<int:user_id>/reset-password")
@admin_required
def reset_user_password_api(user_id: int):
    try:
        payload = _request_payload()
        password = str(payload.get("password") or DEFAULT_NEW_USER_PASSWORD)
        _user_store().set_password(user_id, password)
        return _json_response({"status": "password_reset", "initial_password": password})
    except UserStoreError as exc:
        return _user_error_response(exc)


@app.delete("/api/users/<int:user_id>")
@admin_required
def delete_user_api(user_id: int):
    current = _current_user()
    assert current is not None
    if user_id == current["id"]:
        return _json_response({"error": "不能删除当前登录管理员"}, 400)
    try:
        _user_store().delete_user(user_id)
        return _json_response({"status": "deleted", "user_id": user_id})
    except UserStoreError as exc:
        return _user_error_response(exc)


@app.get("/api")
@login_required
def api_root():
    return _json_response(
        {
            "message": "Single-Cell ANN API",
            "endpoints": [
                "/api/health",
                "/api/metadata",
                "/api/search",
                "/api/auth/login",
                "/api/auth/register",
                "/api/users",
            ],
            "datasets": "/api/datasets",
            "vectordb": {
                "init": "POST /api/vectordb/init",
                "status": "GET /api/vectordb/status",
                "query": "POST /api/vectordb/query",
                "delete": "DELETE /api/vectordb/collection",
                "chroma_available": is_chroma_available(),
            },
        }
    )


@app.get("/api/health")
@login_required
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
@login_required
def metadata():
    try:
        payload = dict(request.args)
        dataset_id = _normalize_dataset_id(payload.get("dataset_id"))
        dataset_path = _resolve_dataset_path(dataset_id)
        resolved_id = _dataset_id_from_path(dataset_path)
        use_rep = payload.get("use_rep", DEFAULT_USE_REP)
        index_name = payload.get("index_name")
        index_config = _index_config_from_payload(payload)
        return _json_response(_metadata_payload(resolved_id, use_rep, index_config, index_name=index_name))
    except Exception as exc:
        return _json_response({"error": str(exc)}, 400)


@app.get("/api/datasets")
@login_required
def list_datasets():
    datasets = []
    for path in sorted(DATA_DIR.glob("*.h5ad")):
        dataset_id = _dataset_id_from_path(path)
        payload = _dataset_payload(dataset_id, path)
        payload["is_merged"] = False
        datasets.append(payload)
    for config in list_merged_configs(DATA_DIR):
        datasets.append(_merged_dataset_payload(config))
    return _json_response({"datasets": datasets})


@app.get("/api/datasets/<dataset_id>/indices")
@login_required
def dataset_indices(dataset_id: str):
    """Return all pre-built indices for a dataset so the search page can list them."""
    try:
        dataset_id = _normalize_dataset_id(dataset_id)
        _migrate_old_index(dataset_id)
        indices = _list_index_names(dataset_id)
        # Augment with any in-memory cached configs (more complete than on-disk)
        for entry in indices:
            cache_key = (dataset_id, entry["name"])
            cached = _INDEX_CACHE.get(cache_key)
            if cached is not None:
                cached_indexer, _ = cached
                entry["config"] = cached_indexer.config_summary
        return _json_response({"indices": indices, "ready": len(indices) > 0})
    except Exception as exc:
        return _json_response({"error": str(exc)}, 400)


@app.get("/api/cells")
@login_required
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
@login_required
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
        if isinstance(loader, MergedDataLoader):
            return _json_response({
                "points": [],
                "categories": [],
                "is_merged": True,
                "message": "UMAP 可视化不适用于合并数据集，各源数据集的 UMAP 坐标空间不统一。",
            })
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
@login_required
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
@admin_required
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
    current = _current_user()
    if current:
        _notify_admins("info", "新数据集上传", f"{filename} 已上传，数据集 ID: {dataset_id}")
    return _json_response({"status": "uploaded", "dataset_id": dataset_id}, 201)


@app.delete("/api/datasets/<dataset_id>")
@admin_required
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

    # Remove new subdirectory layout
    index_dir = _index_dir(dataset_id)
    if index_dir.exists():
        for f in index_dir.iterdir():
            f.unlink(missing_ok=True)
        index_dir.rmdir()

    # Clean up any leftover old-format flat files
    for index_file in INDEX_DIR.glob(f"{dataset_id}_*.index*"):
        index_file.unlink(missing_ok=True)

    return _json_response({"status": "deleted", "dataset_id": dataset_id})


@app.post("/api/merged/create")
@admin_required
def create_merged_dataset():
    try:
        payload = _request_payload()
        name = (payload.get("name") or "").strip()
        source_datasets = payload.get("source_datasets") or []
        use_rep = payload.get("use_rep") or DEFAULT_USE_REP

        if not name:
            return _json_response({"error": "name is required"}, 400)
        if not isinstance(source_datasets, list) or len(source_datasets) < 2:
            return _json_response({"error": "at least 2 source datasets are required"}, 400)

        for ds_id in source_datasets:
            try:
                _resolve_dataset_path(ds_id)
            except FileNotFoundError:
                return _json_response({"error": f"source dataset not found: {ds_id}"}, 404)

        loaders = {}
        for ds_id in source_datasets:
            loaders[ds_id] = _get_loader(ds_id)

        for ds_id, loader in loaders.items():
            if use_rep not in loader.available_reps:
                return _json_response(
                    {"error": f"source dataset '{ds_id}' does not have representation '{use_rep}', "
                              f"available: {loader.available_reps}"},
                    400,
                )

        dims = {ds_id: loader.vector_dim(use_rep) for ds_id, loader in loaders.items()}
        if len(set(dims.values())) > 1:
            return _json_response(
                {"error": f"dimension mismatch across sources for '{use_rep}'", "dimensions": dims},
                400,
            )

        merged_id = _make_merged_id(source_datasets)
        if _is_merged_dataset(merged_id):
            return _json_response({"error": "merged dataset with same sources already exists", "merged_id": merged_id}, 409)

        config = MergedDatasetConfig(
            merged_id=merged_id,
            name=name,
            source_datasets=source_datasets,
            use_rep=use_rep,
        )
        save_merged_config(DATA_DIR, config)

        merged = _get_merged_loader(merged_id)
        return _json_response(
            {
                "status": "created",
                "merged_id": merged_id,
                "config": config.to_dict(),
                "total_cells": merged.n_cells,
                "vector_dim": merged.vector_dim(use_rep),
            },
            201,
        )
    except ValueError as exc:
        return _json_response({"error": str(exc)}, 400)
    except Exception as exc:
        app.logger.exception("Failed to create merged dataset")
        return _json_response({"error": str(exc)}, 500)


@app.get("/api/merged")
@login_required
def list_merged_datasets():
    configs = list_merged_configs(DATA_DIR)
    results = [_merged_dataset_payload(c) for c in configs]
    return _json_response({"datasets": results})


@app.get("/api/merged/<merged_id>")
@login_required
def get_merged_dataset(merged_id: str):
    try:
        merged_id = _normalize_dataset_id(merged_id)
        if not _is_merged_dataset(merged_id):
            return _json_response({"error": "merged dataset not found"}, 404)
        config = load_merged_config(DATA_DIR, merged_id)
        merged = _get_merged_loader(merged_id)
        source_details = []
        for ds_id in config.source_datasets:
            loader = _get_loader(ds_id)
            source_details.append({
                "dataset_id": ds_id,
                "n_cells": loader.n_cells,
                "n_genes": loader.n_genes,
                "available_reps": loader.available_reps,
            })
        return _json_response({
            "config": config.to_dict(),
            "total_cells": merged.n_cells,
            "vector_dim": merged.vector_dim(config.use_rep),
            "available_reps": merged.available_reps,
            "obs_columns": merged.obs_columns,
            "sources": source_details,
        })
    except Exception as exc:
        return _json_response({"error": str(exc)}, 400)


@app.delete("/api/merged/<merged_id>")
@admin_required
def delete_merged_dataset(merged_id: str):
    try:
        merged_id = _normalize_dataset_id(merged_id)
        if not _is_merged_dataset(merged_id):
            return _json_response({"error": "merged dataset not found"}, 404)

        delete_merged_config(DATA_DIR, merged_id)
        _MERGED_CACHE.pop(merged_id, None)
        _DATASET_CACHE.pop(merged_id, None)
        for key in list(_INDEX_CACHE.keys()):
            if key[0] == merged_id:
                _INDEX_CACHE.pop(key, None)

        index_dir = _index_dir(merged_id)
        if index_dir.exists():
            for f in index_dir.iterdir():
                f.unlink(missing_ok=True)
            index_dir.rmdir()

        return _json_response({"status": "deleted", "merged_id": merged_id})
    except Exception as exc:
        return _json_response({"error": str(exc)}, 500)


@app.post("/api/index/build")
@admin_required
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
        # Allow user to specify a custom index name, or auto-generate one
        index_name = payload.get("index_name") or _default_index_name(
            use_rep, index_config.index_type, index_config.metric
        )
        # Ensure the subdirectory exists
        _index_dir(resolved_id).mkdir(parents=True, exist_ok=True)
        # Remove any previously cached entry for this name (force rebuild)
        _INDEX_CACHE.pop((resolved_id, index_name), None)
        indexer = _get_indexer(
            resolved_id, use_rep, index_config,
            build_if_missing=True, index_name=index_name,
        )
        current = _current_user()
        if current:
            _notify(
                current["id"], "success",
                f"索引构建完成",
                f"{resolved_id} 的 {index_name} 索引 ({indexer.backend}/{indexer.index_type}) 已构建完毕",
            )
        return _json_response(
            {
                "status": "built",
                "dataset_id": resolved_id,
                "index_name": index_name,
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


@app.delete("/api/index/<dataset_id>/<index_name>")
@admin_required
def delete_index(dataset_id: str, index_name: str):
    """Delete a single named index for a dataset."""
    try:
        dataset_id = _normalize_dataset_id(dataset_id)
        index_name = _normalize_dataset_id(index_name)
        index_path = _index_path(dataset_id, index_name)
        backup_path = _backup_index_path(index_path)

        deleted = False
        if index_path.exists():
            index_path.unlink()
            deleted = True
        if backup_path.exists():
            backup_path.unlink()
            deleted = True

        _INDEX_CACHE.pop((dataset_id, index_name), None)

        index_dir = _index_dir(dataset_id)
        if index_dir.exists() and not any(index_dir.iterdir()):
            index_dir.rmdir()

        if not deleted:
            return _json_response({"error": "index not found"}, 404)

        return _json_response({"status": "deleted", "dataset_id": dataset_id, "index_name": index_name})
    except Exception as exc:
        return _json_response({"error": str(exc)}, 400)


@app.route("/api/search", methods=["GET", "POST"])
@login_required
def search():
    try:
        total_start_time = time.perf_counter()
        payload = _request_payload()
        dataset_id = _normalize_dataset_id(payload.get("dataset_id"))
        dataset_path = _resolve_dataset_path(dataset_id)
        resolved_id = _dataset_id_from_path(dataset_path)
        use_rep = payload.get("use_rep") or DEFAULT_USE_REP
        index_name = payload.get("index_name")
        index_config = _index_config_from_payload(payload)

        # When using a pre-built index, read use_rep from the stored config
        if index_name:
            stored = _read_stored_index_config(resolved_id, index_name)
            if stored and stored.get("use_rep"):
                use_rep = stored["use_rep"]

        cell_index_value = payload.get("cell_index")
        cell_id_value = payload.get("cell_id")
        k = _parse_int(payload, "k", default=DEFAULT_TOP_K)
        include_self = _parse_bool(payload.get("include_self"), default=False)
        filter_field, filter_value = _normalize_filter(payload)

        loader = _get_loader(resolved_id, dataset_path)
        if filter_field and filter_field not in loader.obs_columns:
            return _json_response({"error": f"filter_field not found: {filter_field}"}, 400)
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

        index_name = payload.get("index_name")
        index_prepare_start_time = time.perf_counter()
        indexer = _get_indexer(
            resolved_id, use_rep, index_config,
            build_if_missing=True, index_name=index_name,
        )
        index_prepare_ms = round(
            (time.perf_counter() - index_prepare_start_time) * 1000.0, 2
        )
        query_vector = loader.get_vector(cell_index, use_rep=use_rep)
        if filter_field:
            search_k = loader.n_cells
        else:
            search_k = min(k + 1 if not include_self else k, loader.n_cells)
        start_time = time.perf_counter()
        distances, indices = indexer.search(query_vector, search_k)
        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        results = []
        scanned = 0
        metric = indexer.metric
        for idx, dist in zip(indices.tolist(), distances.tolist()):
            idx = int(idx)
            distance = float(dist)
            if not include_self and idx == cell_index:
                continue

            cell_info = loader.get_cell_info(idx)
            scanned += 1
            if not _metadata_matches_filter(cell_info, filter_field, filter_value):
                continue
            results.append(
                {
                    "rank": len(results) + 1,
                    "cell_index": idx,
                    "cell_id": cell_info.get("cell_id"),
                    "cell_type": cell_info.get("cell_type", "unknown"),
                    "source_dataset": cell_info.get("source_dataset", resolved_id),
                    "distance": round(distance, 6),
                    "similarity_score": round(
                        _similarity_from_distance(distance, metric), 6
                    ),
                    "metadata": cell_info,
                }
            )

            if len(results) == k:
                break

        query_cell_id = loader.get_cell_info(cell_index).get("cell_id")
        total_elapsed_ms = round((time.perf_counter() - total_start_time) * 1000.0, 2)
        response_payload = {
            "dataset_id": resolved_id,
            "query_cell": cell_index,
            "cell_id": query_cell_id,
            "k": k,
            "include_self": include_self,
            "filter_field": filter_field,
            "filter_value": filter_value,
            "filtered": bool(filter_field),
            "candidate_count": scanned,
            "use_rep": use_rep,
            "index_name": index_name,
            "index_backend": indexer.backend,
            "index_type": indexer.index_type,
            "index_metric": metric,
            "index_config": indexer.config_summary,
            "elapsed_ms": elapsed_ms,
            "search_elapsed_ms": elapsed_ms,
            "index_prepare_ms": index_prepare_ms,
            "total_elapsed_ms": total_elapsed_ms,
            "results": results,
        }
        current_user = _current_user()
        if current_user is not None:
            try:
                snapshot = _user_store().add_search_snapshot(
                    current_user["id"],
                    dataset_id=resolved_id,
                    cell_id=str(query_cell_id or ""),
                    k=k,
                    use_rep=str(use_rep or ""),
                    index_name=str(index_name or ""),
                    index_backend=str(indexer.backend or ""),
                    index_type=str(indexer.index_type or ""),
                    index_metric=str(metric or ""),
                    filter_field=str(filter_field or ""),
                    filter_value=str(filter_value or ""),
                    elapsed_ms=elapsed_ms,
                    total_elapsed_ms=total_elapsed_ms,
                    result_count=len(results),
                )
                response_payload["snapshot_id"] = snapshot["id"]
            except Exception:
                app.logger.exception("Failed to record search snapshot")
        return _json_response(response_payload)
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
@login_required
def benchmark_api():
    try:
        import tracemalloc

        payload = _request_payload()
        dataset_id = _normalize_dataset_id(payload.get("dataset_id"))
        dataset_path = _resolve_dataset_path(dataset_id)
        resolved_id = _dataset_id_from_path(dataset_path)
        use_rep = payload.get("use_rep") or DEFAULT_USE_REP
        metric = str(payload.get("metric") or "l2").strip().lower()
        pca_components = _parse_optional_int(payload, "pca_components") or 0

        query_count = _parse_int(payload, "query_count", default=500)
        k = _parse_int(payload, "k", default=DEFAULT_TOP_K)

        if query_count <= 0:
            return _json_response({"error": "query_count must be a positive integer"}, 400)
        if k <= 0:
            return _json_response({"error": "k must be a positive integer"}, 400)
        if k > MAX_TOP_K:
            return _json_response({"error": f"k must not exceed {MAX_TOP_K}"}, 400)
        if metric not in {"l2", "ip", "cosine", "correlation"}:
            return _json_response({"error": f"unsupported metric: {metric}"}, 400)

        loader = _get_loader(resolved_id, dataset_path)
        vectors = loader.get_vectors(use_rep)
        n_cells = loader.n_cells
        dim = vectors.shape[1]
        pq_m = _suggest_pq_m(dim)

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

        pca_info = None
        pca_vectors = None
        pca_queries = None
        if pca_components > 0:
            if pca_components >= dim:
                pca_components = max(1, dim // 2)
            reducer = PCAReducer(n_components=pca_components)
            pca_vectors = reducer.fit_transform(vectors)
            pca_queries = pca_vectors[query_indices]
            evr = reducer.explained_variance_ratio_
            pca_info = {
                "n_components": pca_components,
                "original_dim": dim,
                "reduced_dim": pca_components,
                "explained_variance_ratio": [round(float(v), 6) for v in evr] if evr is not None else None,
                "total_explained_variance": round(float(evr.sum()) * 100, 2) if evr is not None else None,
            }
            pq_m_pca = _suggest_pq_m(pca_components)
        else:
            pq_m_pca = pq_m

        standard_ks = [10, 20, 50, 100]
        ks = [value for value in standard_ks if value <= max_k]

        def _make_algorithms(target_metric, target_pq_m, label_prefix="", id_prefix="", optimized=False):
            algos = [
                {
                    "id": f"{id_prefix}brute",
                    "label": f"{label_prefix}精确搜索 (暴力)",
                    "optimized": optimized,
                    "config": IndexConfig(backend="numpy", index_type="brute", metric=target_metric),
                },
                {
                    "id": f"{id_prefix}faiss_hnsw",
                    "label": f"{label_prefix}FAISS-HNSW",
                    "optimized": optimized,
                    "config": IndexConfig(backend="faiss", index_type="hnsw", metric=target_metric, m=16, ef_construction=200, ef_search=50),
                },
                {
                    "id": f"{id_prefix}hnswlib",
                    "label": f"{label_prefix}HNSWLIB",
                    "optimized": optimized,
                    "config": IndexConfig(backend="hnswlib", index_type="hnsw", metric=target_metric, m=16, ef_construction=200, ef_search=50),
                },
                {
                    "id": f"{id_prefix}faiss_ivf",
                    "label": f"{label_prefix}FAISS-IVF",
                    "optimized": optimized,
                    "config": IndexConfig(backend="faiss", index_type="ivf_flat", metric=target_metric, nlist=100, nprobe=10),
                },
                {
                    "id": f"{id_prefix}faiss_pq",
                    "label": f"{label_prefix}FAISS-PQ",
                    "optimized": optimized,
                    "config": IndexConfig(backend="faiss", index_type="pq", metric=target_metric, pq_m=target_pq_m, pq_nbits=8),
                },
            ]
            return algos

        def _eval_algorithms(algos, target_vectors, target_queries, rep_key):
            def _build_fresh(vectors, config):
                """每次跑分都重新构建索引，绕过缓存以准确度量内存。"""
                indexer = ANNIndexer(dim=vectors.shape[1], config=config)
                indexer.build_index(vectors)
                return indexer

            brute_algo = algos[0]

            truth_indices: List[np.ndarray] = []
            tracemalloc.start()
            snap_before = tracemalloc.take_snapshot()
            brute_indexer = _build_fresh(target_vectors, brute_algo["config"])
            start_time = time.perf_counter()
            for query in target_queries:
                _, idx = brute_indexer.search(query, max_k)
                truth_indices.append(idx)
            brute_elapsed = time.perf_counter() - start_time
            snap_after = tracemalloc.take_snapshot()
            brute_mem = sum(stat.size_diff for stat in snap_after.compare_to(snap_before, "lineno") if stat.size_diff > 0) / (1024 * 1024)
            tracemalloc.stop()

            results: List[Dict[str, Any]] = []
            brute_avg_ms = round(brute_elapsed / query_count * 1000.0, 3)
            brute_recalls = [100.0 if kv <= max_k else None for kv in standard_ks]
            results.append({
                "id": brute_algo["id"],
                "label": brute_algo["label"],
                "optimized": brute_algo.get("optimized", False),
                "available": True,
                "avg_ms": brute_avg_ms,
                "qps": round(1000.0 / brute_avg_ms, 1) if brute_avg_ms > 0 else None,
                "memory_mb": round(brute_mem, 3),
                "recall_curve": brute_recalls,
            })

            for algo in algos[1:]:
                config = algo["config"]
                tracemalloc.start()
                snap_before = tracemalloc.take_snapshot()
                try:
                    indexer = _build_fresh(target_vectors, config)
                except ImportError as exc:
                    tracemalloc.stop()
                    results.append({
                        "id": algo["id"],
                        "label": algo["label"],
                        "optimized": algo.get("optimized", False),
                        "available": False,
                        "error": str(exc),
                        "avg_ms": None,
                        "qps": None,
                        "memory_mb": None,
                        "recall_curve": [None for _ in standard_ks],
                    })
                    continue
                snap_after = tracemalloc.take_snapshot()
                build_mem = sum(stat.size_diff for stat in snap_after.compare_to(snap_before, "lineno") if stat.size_diff > 0) / (1024 * 1024)
                tracemalloc.stop()

                recall_sums = {kv: 0.0 for kv in ks}
                start_time = time.perf_counter()
                for qi, query in enumerate(target_queries):
                    _, pred_indices = indexer.search(query, max_k)
                    truth = truth_indices[qi]
                    for kv in ks:
                        pred_set = set(pred_indices[:kv].tolist())
                        truth_set = set(truth[:kv].tolist())
                        recall_sums[kv] += len(pred_set.intersection(truth_set)) / kv
                elapsed = time.perf_counter() - start_time

                recall_curve: List[Optional[float]] = []
                for kv in standard_ks:
                    if kv <= max_k:
                        recall_curve.append(round(recall_sums[kv] / query_count * 100.0, 2))
                    else:
                        recall_curve.append(None)

                avg_ms = round(elapsed / query_count * 1000.0, 3)
                results.append({
                    "id": algo["id"],
                    "label": algo["label"],
                    "optimized": algo.get("optimized", False),
                    "available": True,
                    "avg_ms": avg_ms,
                    "qps": round(1000.0 / avg_ms, 1) if avg_ms > 0 else None,
                    "memory_mb": round(build_mem, 3),
                    "recall_curve": recall_curve,
                })
            return results

        base_algorithms = _make_algorithms(metric, pq_m)
        base_results = _eval_algorithms(base_algorithms, vectors, queries, use_rep)

        all_results = list(base_results)

        if pca_components > 0:
            pca_algorithms = _make_algorithms(metric, pq_m_pca, label_prefix="[PCA] ", id_prefix="pca_", optimized=True)
            pca_results = _eval_algorithms(pca_algorithms, pca_vectors, pca_queries, f"{use_rep}_pca")
            all_results.extend(pca_results)

        created_at = time.strftime("%Y-%m-%d %H:%M:%S")
        entry = {
            "created_at": created_at,
            "dataset_id": resolved_id,
            "use_rep": use_rep,
            "metric": metric,
            "query_count": query_count,
            "k": k,
            "k_values": standard_ks,
            "pca": pca_info,
            "algorithms": all_results,
        }
        history_id = _append_benchmark_history(entry)

        return _json_response({
            "dataset_id": resolved_id,
            "use_rep": use_rep,
            "metric": metric,
            "query_count": query_count,
            "k": k,
            "k_values": standard_ks,
            "pca": pca_info,
            "algorithms": all_results,
            "created_at": created_at,
            "history_id": history_id,
        })
    except ImportError as exc:
        return _json_response({"error": str(exc)}, 400)
    except (TypeError, ValueError, KeyError, IndexError) as exc:
        return _json_response({"error": str(exc)}, 400)
    except RuntimeError as exc:
        return _json_response({"error": str(exc)}, 503)


# ============================================================
# 向量数据库 (ChromaDB) API — 任务 3.1
# ============================================================

def _chroma_collection_name(dataset_id: str) -> str:
    """将 dataset_id 映射为合法的 ChromaDB Collection 名称。"""
    # ChromaDB collection 名称只允许 [a-zA-Z0-9_-]，且长度 3~63
    import re
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", str(dataset_id))
    if len(name) < 3:
        name = name + "_db"
    return name[:63]


def _extract_question_keywords(question: str) -> List[str]:
    """从用户自然语言问题中提取基因名和细胞类型关键词。

    用于 RAG 关键词检索 — 在无法直接用向量检索时（PCA 向量空间与文本
    Embedding 空间维度不匹配），通过关键词匹配细胞文档来找到相关细胞。

    提取策略：
    1. 大写基因符号（2-8 个字母，全大写，如 ALB、TP53、APOA1）
    2. 常见细胞类型后缀词（如 Hepatocyte、Kupffer cell、T cell）
    3. 中英文混合：中文问题中夹杂的英文基因名
    """
    import re as _re

    keywords: List[str] = []

    # ① 提取大写基因符号（典型模式：2-8 个大写字母或数字组合）
    gene_pattern = r'\b([A-Z][A-Z0-9]{1,7})\b'
    gene_candidates = _re.findall(gene_pattern, question)
    # 过滤常见非基因缩写
    stop_words = {
        "API", "HTTP", "URL", "JSON", "SSE", "RAG", "LLM", "VDB",
        "DNA", "RNA", "PCA", "UMAP", "ANN", "GPU", "CPU", "IO",
        "OK", "YES", "NO", "THE", "AND", "FOR", "ARE", "NOT",
        "THIS", "THAT", "WHAT", "WHEN", "WHERE", "WHICH", "THERE",
        "ABOUT", "WOULD", "COULD", "SHOULD", "FROM", "WITH", "HAVE",
        "WILL", "JUST", "LIKE", "SOME", "MANY", "MUCH", "VERY",
    }
    for g in gene_candidates:
        if g.upper() not in stop_words:
            keywords.append(g)

    # ② 提取细胞类型关键词（英文单词 + 常见后缀）
    cell_type_pattern = r'\b([A-Za-z]+(?:cyte|phil|blast|clast|cytic|thelial|endocrine|immune|kine|gen|oid))\b'
    ct_candidates = _re.findall(cell_type_pattern, question, _re.IGNORECASE)
    for ct in ct_candidates:
        if ct.lower() not in {kw.lower() for kw in keywords}:
            keywords.append(ct)

    # ③ 提取英文双词组合（如 "T cell"、"B cell"、"Kupffer cell"）
    multi_word = _re.findall(r'\b([A-Za-z]+)\s+(cell|type|lineage)\b', question, _re.IGNORECASE)
    for first, second in multi_word:
        combo = f"{first} {second}"
        if combo.lower() not in {kw.lower() for kw in keywords}:
            keywords.append(combo)
        if first.lower() not in {kw.lower() for kw in keywords}:
            keywords.append(first)

    # ④ 中文中夹杂的英文词（如「ALB 基因」→ 提取 ALB）
    cn_en_pattern = r'([A-Za-z][A-Za-z0-9]{1,15})\s*(?:基因|蛋白|细胞|表达|因子)'
    cn_matches = _re.findall(cn_en_pattern, question)
    for m in cn_matches:
        if m.upper() not in stop_words and m.lower() not in {kw.lower() for kw in keywords}:
            keywords.append(m)

    return keywords


# ---------------------------------------------------------------------------
# Markdown 强制格式化 — 确保 LLM 输出始终包含 Markdown 结构
# ---------------------------------------------------------------------------

def _enforce_markdown(text: str) -> str:
    """检测并强制 LLM 回复使用 Markdown 格式。

    如果回复缺少标题、列表、粗体等 Markdown 元素，自动添加基本结构。
    如果已有 Markdown 格式，原样返回不做修改。

    策略：
    1. 若已有 ``##`` 标题 → 说明 LLM 遵守了格式指令，原样返回
    2. 若无 Markdown 结构 → 将纯文本拆分段落，添加标题和格式化
    """
    import re as _re

    if not text or not text.strip():
        return text

    # ── 检测已有 Markdown 标记 ──
    has_headings = bool(_re.search(r'^#{1,4}\s', text, _re.MULTILINE))
    has_lists = bool(_re.search(r'^[\-\*]\s', text, _re.MULTILINE))
    has_bold = bool(_re.search(r'\*\*[^*]+\*\*', text))
    has_table = bool(_re.search(r'\|.+\|', text))
    has_code = bool(_re.search(r'```', text))

    # 若已有 2 种以上 Markdown 特征，说明格式已足够
    markdown_features = sum([has_headings, has_lists, has_bold, has_table, has_code])
    if markdown_features >= 2:
        return text

    # ── 尝试增强：为关键术语加粗 ──
    # 大写基因符号加粗
    if not has_bold:
        text = _re.sub(
            r'(?<!\*)\b([A-Z][A-Z0-9]{1,7})\b(?!\*)',
            r'**\1**',
            text,
        )

    # ── 若无标题，按段落拆分并添加标题 ──
    if not has_headings:
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if len(paragraphs) >= 2:
            # 第一个段落作为概述，后续用标题分段
            result_parts = ["## 分析结果\n\n" + paragraphs[0]]
            for i, para in enumerate(paragraphs[1:], 1):
                # 尝试从段落中提取主题作为子标题
                result_parts.append(f"### 要点 {i}\n\n{para}")
            text = "\n\n".join(result_parts)
        elif not text.startswith('#'):
            text = "## 分析结果\n\n" + text

    # ── 若无列表，检测连续逗号/分号分隔的枚举并转换 ──
    if not has_lists:
        # 检测 "A、B、C" 或 "A, B, C" 模式，超过 3 项转为列表
        enum_pattern = _re.findall(
            r'((?:\**[A-Z][a-zA-Z0-9]*\**[，,\s]+){3,}(?:\**[A-Z][a-zA-Z0-9]*\**))',
            text,
        )
        # 只在明确的长枚举处转换，避免过度修改
        # 保留原有结构，不做激进转换

    return text


def _build_dataset_info(loader: Any, dataset_id: str, use_rep: str) -> str:
    """构建注入 system prompt 的数据集背景信息字符串。

    自动从 DataLoader 中提取基础统计（细胞数、基因数）以及细胞类型分布，
    让 LLM 具备数据集全局视角，回答更准确。

    Parameters
    ----------
    loader:
        DataLoader 实例。
    dataset_id:
        数据集 ID（用于显示）。
    use_rep:
        当前使用的向量表示。

    Returns
    -------
    str
        可直接作为 ``extra_system_info`` 传给 PromptBuilder 的字符串。
    """
    lines = [
        f"数据集 ID：{dataset_id}",
        f"细胞总数：{loader.n_cells}",
        f"基因数：{loader.n_genes}",
        f"使用向量表示：{use_rep}",
    ]

    # 尝试提取细胞类型分布
    try:
        adata = loader.adata
        ct_col = None
        for candidate in ("cell_type", "celltype", "cell_ontology_class", "leiden", "louvain"):
            if candidate in adata.obs.columns:
                ct_col = candidate
                break

        if ct_col:
            counts = adata.obs[ct_col].value_counts()
            top_n = 8  # 只显示前 8 种，避免信息过长
            parts = [f"{ct}（{n}个）" for ct, n in counts.head(top_n).items()]
            if len(counts) > top_n:
                parts.append(f"...等共 {len(counts)} 种")
            lines.append(f"细胞类型分布（{ct_col}）：" + "、".join(parts))
    except Exception:
        pass  # 提取失败时静默跳过，不影响主流程

    return "，".join(lines)


def _resolve_query_vector(
    payload: Dict[str, Any],
    loader: Any,
    use_rep: str,
) -> Optional[List[float]]:
    """从请求参数中解析查询向量。

    优先级：cell_index > cell_id > None（由 RAG 引擎或调用方处理 None 情况）。

    Parameters
    ----------
    payload:
        请求参数字典（来自 JSON body 或 query string）。
    loader:
        DataLoader 实例，用于获取细胞向量。
    use_rep:
        向量表示键（如 "X_pca"）；若不可用则自动回退。

    Returns
    -------
    List[float] | None
        查询向量；未提供 cell_index/cell_id 时返回 None。
    """
    cell_index_value = payload.get("cell_index")
    cell_id_value = payload.get("cell_id")

    cell_index = None
    if cell_index_value is not None and cell_index_value != "":
        cell_index = int(cell_index_value)
    elif cell_id_value is not None and str(cell_id_value).strip():
        cell_index = loader.cell_index_from_id(str(cell_id_value))

    if cell_index is None:
        return None

    actual_rep = use_rep if use_rep in loader.available_reps else "X_pca"
    if actual_rep not in loader.available_reps:
        actual_rep = "X"
    return loader.get_vector(cell_index, use_rep=actual_rep).tolist()


@app.post("/api/vectordb/init")
@login_required
def vectordb_init():
    """初始化向量数据库：将指定数据集的细胞向量写入 ChromaDB Collection。

    Request JSON
    -----------
    dataset_id : str, optional
        目标数据集 ID，省略时使用默认数据集。
    use_rep : str, optional
        向量表示键（默认 X_pca）。
    distance_metric : str, optional
        距离度量：cosine / l2 / ip（默认 cosine）。
    force : bool, optional
        是否强制重写（默认 false，已有数据时跳过）。
    top_genes : int, optional
        元数据中记录高表达基因数（默认 20）。

    Response JSON
    ------------
    status : "initialized" | "skipped"
    collection_name : str
    count : int
    elapsed_ms : float
    """
    if not is_chroma_available():
        return _json_response(
            {"error": "chromadb 未安装，请执行: pip install chromadb"},
            501,
        )
    try:
        payload = _request_payload()
        dataset_id = _normalize_dataset_id(payload.get("dataset_id"))
        dataset_path = _resolve_dataset_path(dataset_id)
        resolved_id = _dataset_id_from_path(dataset_path)
        use_rep = str(payload.get("use_rep") or "X_pca")
        distance_metric = str(payload.get("distance_metric") or "cosine")
        force = bool(payload.get("force", False))
        top_genes = int(payload.get("top_genes") or 20)

        collection_name = _chroma_collection_name(resolved_id)
        store = get_or_create_store(
            collection_name=collection_name,
            persist_dir=CHROMA_DIR,
            distance_metric=distance_metric,
        )

        if store.is_populated() and not force:
            return _json_response(
                {
                    "status": "skipped",
                    "message": "Collection 已存在数据，传入 force=true 可强制重写",
                    "collection_name": collection_name,
                    "count": store.count(),
                }
            )

        loader = _get_loader(resolved_id, dataset_path)
        t0 = time.perf_counter()
        count = store.populate_from_loader(
            loader=loader,
            use_rep=use_rep,
            force=force,
            top_genes=top_genes,
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        current = _current_user()
        if current:
            _notify(
                current["id"], "success",
                "向量数据库初始化完成",
                f"{resolved_id} 的 ChromaDB 已写入 {count} 条细胞数据",
            )

        return _json_response(
            {
                "status": "initialized",
                "dataset_id": resolved_id,
                "collection_name": collection_name,
                "use_rep": use_rep,
                "distance_metric": distance_metric,
                "count": count,
                "elapsed_ms": elapsed_ms,
            },
            201,
        )
    except Exception as exc:
        return _json_response({"error": str(exc)}, 400)


@app.get("/api/vectordb/status")
@login_required
def vectordb_status():
    """查询向量数据库状态。

    Query Params
    -----------
    dataset_id : str, optional

    Response JSON
    ------------
    chroma_available : bool
    persist_dir : str
    collection_name : str
    count : int
    is_populated : bool
    distance_metric : str
    """
    if not is_chroma_available():
        return _json_response(
            {
                "chroma_available": False,
                "error": "chromadb 未安装",
            }
        )
    try:
        payload = dict(request.args)
        dataset_id = _normalize_dataset_id(payload.get("dataset_id"))
        dataset_path = _resolve_dataset_path(dataset_id)
        resolved_id = _dataset_id_from_path(dataset_path)
        collection_name = _chroma_collection_name(resolved_id)
        store = get_or_create_store(
            collection_name=collection_name,
            persist_dir=CHROMA_DIR,
        )
        return _json_response(store.get_collection_info())
    except Exception as exc:
        return _json_response({"error": str(exc)}, 400)


def _get_optional_vector_store(dataset_id: str) -> Optional[CellVectorStore]:
    if not is_chroma_available():
        return None
    try:
        return get_or_create_store(
            collection_name=_chroma_collection_name(dataset_id),
            persist_dir=CHROMA_DIR,
        )
    except Exception:
        logger.exception("Failed to open ChromaDB store for natural language query")
        return None


def _execute_natural_language_cell_query(
    question: str,
    loader: Any,
    dataset_id: str,
    *,
    limit: int = NLQ_DEFAULT_LIMIT,
    use_rep: str = "X_pca",
    seed_cell_id: Optional[str] = None,
) -> Dict[str, Any]:
    store = _get_optional_vector_store(dataset_id)
    plan = parse_natural_cell_query(
        question,
        loader,
        limit=limit,
        seed_cell_id=seed_cell_id,
    )
    return execute_natural_cell_query(
        plan,
        loader,
        store=store,
        use_rep=use_rep,
    )


@app.post("/api/cells/query")
@login_required
def natural_cells_query():
    """Natural language cell query: question -> structured plan -> matching cells."""
    try:
        payload = _request_payload()
        question = str(payload.get("question") or payload.get("q") or "").strip()
        if not question:
            return _json_response({"error": "question cannot be empty"}, 400)

        dataset_id = _normalize_dataset_id(payload.get("dataset_id"))
        dataset_path = _resolve_dataset_path(dataset_id)
        resolved_id = _dataset_id_from_path(dataset_path)
        use_rep = str(payload.get("use_rep") or "X_pca")
        limit = _parse_int(payload, "limit", default=NLQ_DEFAULT_LIMIT)
        seed_cell_id = str(payload.get("cell_id") or "").strip() or None

        loader = _get_loader(resolved_id, dataset_path)
        t0 = time.perf_counter()
        query_result = _execute_natural_language_cell_query(
            question,
            loader,
            resolved_id,
            limit=limit,
            use_rep=use_rep,
            seed_cell_id=seed_cell_id,
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return _json_response(
            {
                "dataset_id": resolved_id,
                "elapsed_ms": elapsed_ms,
                **query_result,
            }
        )
    except (TypeError, ValueError, KeyError, IndexError) as exc:
        return _json_response({"error": str(exc)}, 400)
    except Exception as exc:
        return _json_response({"error": str(exc)}, 500)


@app.post("/api/vectordb/query")
@login_required
def vectordb_query():
    """向量相似检索接口（直接操作 ChromaDB）。

    与 /api/search 不同，本接口使用 ChromaDB 的 HNSW 索引执行检索，
    并在结果中附带高表达基因列表和文档文本，便于后续 RAG Prompt 组装。

    Request JSON
    -----------
    dataset_id : str, optional
    cell_index : int      查询细胞的整数索引（与 cell_id 二选一）
    cell_id : str         查询细胞的字符串 ID
    n_results : int, optional  返回结果数（默认 10）
    use_rep : str, optional    向量表示（默认 X_pca）
    where : dict, optional     ChromaDB 元数据过滤条件

    Response JSON
    ------------
    results : list
        每条包含 rank / cell_id / cell_type / distance / top_genes / document
    """
    if not is_chroma_available():
        return _json_response({"error": "chromadb 未安装"}, 501)
    try:
        payload = _request_payload()
        dataset_id = _normalize_dataset_id(payload.get("dataset_id"))
        dataset_path = _resolve_dataset_path(dataset_id)
        resolved_id = _dataset_id_from_path(dataset_path)
        use_rep = str(payload.get("use_rep") or "X_pca")
        n_results = int(payload.get("n_results") or 10)
        where = payload.get("where") or None

        collection_name = _chroma_collection_name(resolved_id)
        store = get_or_create_store(
            collection_name=collection_name,
            persist_dir=CHROMA_DIR,
        )
        if not store.is_populated():
            return _json_response(
                {
                    "error": "向量数据库尚未初始化，请先调用 POST /api/vectordb/init",
                    "action": "init_vectordb",
                },
                400,
            )

        loader = _get_loader(resolved_id, dataset_path)

        # 解析查询向量（cell_index / cell_id 必须提供其一）
        query_vector = _resolve_query_vector(payload, loader, use_rep)
        if query_vector is None:
            raise ValueError("cell_index 或 cell_id 必须提供其中之一")

        t0 = time.perf_counter()
        results = store.query_similar(
            query_vector=query_vector,
            n_results=n_results,
            where=where,
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return _json_response(
            {
                "dataset_id": resolved_id,
                "n_results": n_results,
                "use_rep": use_rep,
                "elapsed_ms": elapsed_ms,
                "results": results,
            }
        )
    except (TypeError, ValueError, KeyError, IndexError) as exc:
        return _json_response({"error": str(exc)}, 400)
    except RuntimeError as exc:
        return _json_response({"error": str(exc)}, 503)
    except Exception as exc:
        return _json_response({"error": str(exc)}, 500)


@app.delete("/api/vectordb/collection")
@admin_required
def vectordb_delete_collection():
    """清空指定数据集的 ChromaDB Collection（需要管理员权限）。

    Query Params
    -----------
    dataset_id : str, optional
    """
    if not is_chroma_available():
        return _json_response({"error": "chromadb 未安装"}, 501)
    try:
        payload = dict(request.args)
        dataset_id = _normalize_dataset_id(payload.get("dataset_id"))
        dataset_path = _resolve_dataset_path(dataset_id)
        resolved_id = _dataset_id_from_path(dataset_path)
        collection_name = _chroma_collection_name(resolved_id)
        store = get_or_create_store(
            collection_name=collection_name,
            persist_dir=CHROMA_DIR,
        )
        store.delete_collection()
        return _json_response(
            {
                "status": "deleted",
                "collection_name": collection_name,
            }
        )
    except Exception as exc:
        return _json_response({"error": str(exc)}, 400)


# ============================================================
# RAG 大模型问答 API — 任务 3.3
# ============================================================

@app.post("/api/chat")
@login_required
def chat_api():
    """RAG 问答核心接口：自然语言 → 向量检索 → LLM 生成回答。

    Request JSON
    -----------
    question : str
        用户的自然语言问题（必填）。
    dataset_id : str, optional
        数据集 ID，省略时使用默认数据集。
    cell_index : int, optional
        使用指定细胞的向量作为查询向量（与 cell_id 二选一）。
        若不提供则尝试调用 Embedding API 将问题转为向量。
    cell_id : str, optional
        使用指定细胞的向量作为查询向量（与 cell_index 二选一）。
    use_rep : str, optional
        向量表示键，默认 X_pca。
    n_results : int, optional
        检索细胞数量，默认 5。
    session_id : str, optional
        会话 ID，提供时启用多轮对话（历史上下文自动注入 Prompt）。
    where : dict, optional
        ChromaDB 元数据过滤条件。

    Response JSON
    ------------
    answer : str            大模型生成的回答
    retrieved_cells : list  检索到的相似细胞列表
    context_used : str      喂给大模型的上下文（调试用）
    elapsed_ms : float      总耗时
    retrieve_ms : float     向量检索耗时
    llm_ms : float          LLM 生成耗时
    session_id : str        当前会话 ID
    model : str             使用的模型名称
    """
    if not is_chroma_available():
        return _json_response({"error": "chromadb 未安装，向量数据库不可用"}, 501)

    try:
        payload = _request_payload()
        question = str(payload.get("question") or "").strip()
        if not question:
            return _json_response({"error": "question 不能为空"}, 400)

        dataset_id = _normalize_dataset_id(payload.get("dataset_id"))
        dataset_path = _resolve_dataset_path(dataset_id)
        resolved_id = _dataset_id_from_path(dataset_path)
        use_rep = str(payload.get("use_rep") or "X_pca")
        n_results = int(payload.get("n_results") or 5)
        session_id = str(payload.get("session_id") or "").strip() or None
        where = payload.get("where") or None

        # 获取向量数据库
        collection_name = _chroma_collection_name(resolved_id)
        store = get_or_create_store(
            collection_name=collection_name,
            persist_dir=CHROMA_DIR,
        )
        if not store.is_populated():
            return _json_response(
                {
                    "error": "向量数据库尚未初始化，请先调用 POST /api/vectordb/init",
                    "action": "init_vectordb",
                },
                400,
            )

        # 构建数据集背景信息（注入 system prompt）
        loader = _get_loader(resolved_id, dataset_path)
        dataset_info = _build_dataset_info(loader, resolved_id, use_rep)

        # 获取 RAG 引擎
        engine = get_or_create_engine(
            vector_store=store,
            dataset_id=resolved_id,
            dataset_info=dataset_info,
            n_results=n_results,
        )

        # 解析查询向量（可选，不提供则用关键词检索）
        query_vector = _resolve_query_vector(payload, loader, use_rep)
        keywords: Optional[List[str]] = None
        natural_query_result: Optional[Dict[str, Any]] = None
        retrieved_cells: Optional[List[Dict[str, Any]]] = None
        if query_vector is None:
            keywords = _extract_question_keywords(question)
            logger.info("/api/chat 关键词: question='%s' → %s", question[:80], keywords)
            natural_query_result = _execute_natural_language_cell_query(
                question,
                loader,
                resolved_id,
                limit=n_results,
                use_rep=use_rep,
            )
            retrieved_cells = natural_query_result.get("results") or []

        # ── LLM Controls ──
        temperature = payload.get("temperature")
        if temperature is not None:
            temperature = float(temperature)
        max_tokens = payload.get("max_tokens")
        if max_tokens is not None:
            max_tokens = int(max_tokens)
        preset_key = str(payload.get("preset") or "").strip() or None
        system_prompt: Optional[str] = None
        if preset_key:
            from backend.prompt_builder import get_preset_prompt
            system_prompt = get_preset_prompt(preset_key)

        # 执行 RAG
        result = engine.ask(
            question=question,
            query_vector=query_vector,
            session_id=session_id,
            n_results=n_results,
            where_filter=where,
            keywords=keywords,
            retrieved_cells=retrieved_cells,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )

        if natural_query_result is not None:
            result["natural_query"] = natural_query_result.get("plan")

        # 强制 Markdown 格式化
        if result.get("answer"):
            result["answer"] = _enforce_markdown(result["answer"])

        return _json_response(result)

    except ValueError as exc:
        return _json_response({"error": str(exc)}, 400)
    except RuntimeError as exc:
        return _json_response({"error": str(exc)}, 503)
    except Exception as exc:
        return _json_response({"error": str(exc)}, 500)


@app.post("/api/chat/stream")
@login_required
def chat_stream_api():
    """RAG 问答流式接口（SSE）：逐字返回大模型回答，前端可实时显示。

    与 /api/chat 参数相同，区别在于响应格式为 text/event-stream。
    每个 SSE 事件格式：
        data: <文本片段>\\n\\n
    流结束时发送：
        data: [DONE]\\n\\n

    Request JSON
    -----------
    question : str         用户问题（必填）
    dataset_id : str       数据集 ID（可选）
    cell_index : int       查询细胞索引（可选）
    cell_id : str          查询细胞 ID（可选）
    use_rep : str          向量表示（默认 X_pca）
    n_results : int        检索细胞数（默认 5）
    session_id : str       会话 ID（可选，启用多轮对话）
    where : dict           元数据过滤条件（可选）
    """
    if not is_chroma_available():
        return _json_response({"error": "chromadb 未安装，向量数据库不可用"}, 501)

    try:
        payload = _request_payload()
        question = str(payload.get("question") or "").strip()
        if not question:
            return _json_response({"error": "question 不能为空"}, 400)

        dataset_id = _normalize_dataset_id(payload.get("dataset_id"))
        dataset_path = _resolve_dataset_path(dataset_id)
        resolved_id = _dataset_id_from_path(dataset_path)
        use_rep = str(payload.get("use_rep") or "X_pca")
        n_results = int(payload.get("n_results") or 5)
        session_id = str(payload.get("session_id") or "").strip() or None
        where = payload.get("where") or None
        _cur_user = _current_user()
        _cur_user_id = int(_cur_user["id"]) if _cur_user else None

        # ── LLM Controls（前端参数微调）──
        temperature = payload.get("temperature")
        if temperature is not None:
            temperature = float(temperature)
        max_tokens = payload.get("max_tokens")
        if max_tokens is not None:
            max_tokens = int(max_tokens)
        preset_key = str(payload.get("preset") or "").strip() or None
        system_prompt_override: Optional[str] = None
        if preset_key:
            from backend.prompt_builder import get_preset_prompt
            system_prompt_override = get_preset_prompt(preset_key)

        # 获取向量数据库
        collection_name = _chroma_collection_name(resolved_id)
        store = get_or_create_store(
            collection_name=collection_name,
            persist_dir=CHROMA_DIR,
        )
        if not store.is_populated():
            return _json_response(
                {
                    "error": "向量数据库尚未初始化，请先调用 POST /api/vectordb/init",
                    "action": "init_vectordb",
                },
                400,
            )

        # 解析查询向量
        loader = _get_loader(resolved_id, dataset_path)
        query_vector = _resolve_query_vector(payload, loader, use_rep)

        # 构建 Prompt messages（复用 RAG 引擎的检索 + 组装逻辑）
        engine = get_or_create_engine(
            vector_store=store,
            dataset_id=resolved_id,
            dataset_info=_build_dataset_info(loader, resolved_id, use_rep),
            n_results=n_results,
        )

        llm_client = get_llm_client()

        # SSE 生成器
        def generate():
            full_text_parts = []
            retrieved: List[Dict[str, Any]] = []
            natural_query_plan: Optional[Dict[str, Any]] = None
            try:
                yield "data: [PROGRESS] parse\n\n"
                if query_vector is not None:
                    yield "data: [PROGRESS] query\n\n"
                    retrieved = store.query_similar(
                        query_vector=query_vector,
                        n_results=n_results,
                        where=where,
                    )
                else:
                    keywords = _extract_question_keywords(question)
                    logger.info("RAG 关键词提取: question='%s' → keywords=%s", question[:80], keywords)
                    yield "data: [PROGRESS] query\n\n"
                    natural_query_result = _execute_natural_language_cell_query(
                        question,
                        loader,
                        resolved_id,
                        limit=n_results,
                        use_rep=use_rep,
                    )
                    natural_query_plan = natural_query_result.get("plan")
                    yield f"data: [NL_QUERY] {json.dumps(natural_query_plan, ensure_ascii=False)}\n\n"
                    retrieved = natural_query_result.get("results") or []
                    if not retrieved:
                        logger.warning("关键词无匹配，回退到全库采样 %d 条细胞", n_results)
                        retrieved = store.query_by_keywords(keywords=keywords, n_results=n_results)
                        if not retrieved:
                            retrieved = store.query_by_metadata(where={}, limit=n_results)

                yield "data: [PROGRESS] sources\n\n"
                yield f"data: [SOURCES] {json.dumps(retrieved, ensure_ascii=False)}\n\n"

                history = _CHAT_HISTORY.get(session_id) if session_id else None
                builder = engine.prompt_builder
                if history:
                    messages = builder.build_messages_with_history(
                        user_question=question,
                        retrieved_cells=retrieved,
                        history=history,
                        extra_system_info=engine.dataset_info,
                        system_prompt=system_prompt_override,
                    )
                else:
                    messages = builder.build_messages(
                        user_question=question,
                        retrieved_cells=retrieved,
                        extra_system_info=engine.dataset_info,
                        system_prompt=system_prompt_override,
                    )

                yield "data: [PROGRESS] answer\n\n"
                for chunk in llm_client.stream_chat(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ):
                    full_text_parts.append(chunk)
                    # SSE 规范：单条 data 行内不能含换行符，用 \\n 转义
                    safe_chunk = chunk.replace("\\", "\\\\").replace("\n", "\\n")
                    yield f"data: {safe_chunk}\n\n"
            except Exception as exc:
                yield f"data: [ERROR] {exc}\n\n"
                return
            # 流结束后：强制 Markdown 格式化 + 保存历史
            raw_text = "".join(full_text_parts)
            formatted_text = _enforce_markdown(raw_text)
            if session_id and formatted_text:
                _CHAT_HISTORY.append(session_id, question, formatted_text)
                # 同步持久化到 SQLite（用户可跨会话查看历史）
                if _cur_user_id is not None:
                    try:
                        store_inst = _user_store()
                        store_inst.append_chat_message(session_id, _cur_user_id, "user", question)
                        store_inst.append_chat_message(session_id, _cur_user_id, "assistant", formatted_text)
                    except Exception:
                        pass  # 持久化失败不影响正常回复
            # 发送格式化后的完整文本（换行符同样需要转义）
            safe_formatted = formatted_text.replace("\\", "\\\\").replace("\n", "\\n")
            yield f"data: [FORMATTED] {safe_formatted}\n\n"
            yield "data: [DONE]\n\n"

        return Response(
            stream_with_context(generate()),
            content_type="text/event-stream; charset=utf-8",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # 禁用 nginx 缓冲，确保实时推送
            },
        )

    except ValueError as exc:
        return _json_response({"error": str(exc)}, 400)
    except RuntimeError as exc:
        return _json_response({"error": str(exc)}, 503)
    except Exception as exc:
        return _json_response({"error": str(exc)}, 500)


@app.get("/api/chat/history")
@login_required
def chat_history_api():
    """获取指定会话的对话历史（优先从 SQLite 读取，回退到内存）。

    Query Params
    -----------
    session_id : str  （必填）
    """
    session_id = request.args.get("session_id", "").strip()
    if not session_id:
        return _json_response({"error": "session_id 不能为空"}, 400)
    user = _current_user()
    if user:
        try:
            msgs = _user_store().get_chat_messages(session_id, int(user["id"]))
            if msgs:
                return _json_response({"session_id": session_id, "history": msgs, "rounds": len(msgs) // 2})
        except Exception:
            pass
    history = _CHAT_HISTORY.get(session_id)
    return _json_response({"session_id": session_id, "history": history, "rounds": len(history) // 2})


@app.delete("/api/chat/history")
@login_required
def clear_chat_history_api():
    """清空指定会话的对话历史（内存 + SQLite）。

    Query Params
    -----------
    session_id : str  （必填）
    """
    session_id = request.args.get("session_id", "").strip()
    if not session_id:
        return _json_response({"error": "session_id 不能为空"}, 400)
    _CHAT_HISTORY.clear(session_id)
    user = _current_user()
    if user:
        try:
            _user_store().delete_chat_session(session_id, int(user["id"]))
        except Exception:
            pass
    return _json_response({"status": "cleared", "session_id": session_id})


# ── 对话历史管理 API ────────────────────────────────────────────────────────

@app.get("/api/chat/sessions")
@login_required
def list_chat_sessions_api():
    """列出当前用户的所有历史对话（SQLite 持久化 + 内存会话合并，按最近更新倒序）。"""
    user = _current_user()
    if not user:
        return _json_response({"error": "未登录"}, 401)
    try:
        limit = int(request.args.get("limit", 50))
        # 1. 从 SQLite 读取持久化会话
        try:
            db_sessions = _user_store().list_chat_sessions(int(user["id"]), limit=limit)
        except Exception:
            db_sessions = []

        # 2. 将内存中尚未持久化的会话也补入（key: session_id）
        db_ids = {s["id"] for s in db_sessions}
        mem_sessions = []
        for sid in _CHAT_HISTORY.list_sessions():
            if sid not in db_ids:
                msgs = _CHAT_HISTORY.get(sid)
                if not msgs:
                    continue
                # 取第一条 user 消息作为标题
                first_user = next((m["content"] for m in msgs if m["role"] == "user"), "新对话")
                mem_sessions.append({
                    "id": sid,
                    "title": first_user[:30].strip().replace("\n", " ") or "新对话",
                    "dataset_id": None,
                    "message_count": len(msgs),
                    "created_at": "",
                    "updated_at": "",
                })

        sessions = mem_sessions + db_sessions
        sessions = sessions[:limit]
        return _json_response({"sessions": sessions})
    except Exception as exc:
        return _json_response({"error": str(exc)}, 400)


@app.get("/api/chat/sessions/<session_id>")
@login_required
def get_chat_session_api(session_id: str):
    """获取指定对话的全部消息（优先 SQLite，回退到内存）。"""
    user = _current_user()
    if not user:
        return _json_response({"error": "未登录"}, 401)
    try:
        # 优先从 SQLite 读取
        try:
            msgs = _user_store().get_chat_messages(session_id, int(user["id"]))
        except Exception:
            msgs = []
        # 若 SQLite 无数据，从内存 ChatHistory 回退
        if not msgs:
            mem = _CHAT_HISTORY.get(session_id)
            if mem:
                msgs = [{"role": m["role"], "content": m["content"], "created_at": ""} for m in mem]
        return _json_response({"session_id": session_id, "messages": msgs})
    except Exception as exc:
        return _json_response({"error": str(exc)}, 400)


@app.delete("/api/chat/sessions/<session_id>")
@login_required
def delete_chat_session_api(session_id: str):
    """删除指定对话（消息一并删除）。"""
    user = _current_user()
    if not user:
        return _json_response({"error": "未登录"}, 401)
    try:
        _user_store().delete_chat_session(session_id, int(user["id"]))
        _CHAT_HISTORY.clear(session_id)
        return _json_response({"status": "deleted", "session_id": session_id})
    except Exception as exc:
        return _json_response({"error": str(exc)}, 400)


@app.patch("/api/chat/sessions/<session_id>")
@login_required
def rename_chat_session_api(session_id: str):
    """重命名对话标题。Request JSON: {title: str}"""
    user = _current_user()
    if not user:
        return _json_response({"error": "未登录"}, 401)
    try:
        payload = _request_payload()
        title = str(payload.get("title") or "").strip()
        if not title:
            return _json_response({"error": "title 不能为空"}, 400)
        _user_store().rename_chat_session(session_id, int(user["id"]), title)
        return _json_response({"status": "renamed", "session_id": session_id, "title": title})
    except Exception as exc:
        return _json_response({"error": str(exc)}, 400)


# ============================================================
# 系统通知中心 API
# ============================================================

def _notify(user_id: int, noti_type: str, title: str, content: str = "") -> None:
    """快捷创建通知（静默失败，不影响主流程）。"""
    try:
        _user_store().add_notification(user_id, noti_type, title, content)
    except Exception:
        app.logger.exception("Failed to create notification")


def _notify_admins(noti_type: str, title: str, content: str = "") -> None:
    """给所有管理员发送通知。"""
    try:
        admins = _user_store().list_users()
        for admin in admins:
            if admin.get("role") == "admin" and admin.get("is_active"):
                _notify(admin["id"], noti_type, title, content)
    except Exception:
        app.logger.exception("Failed to notify admins")


@app.get("/api/notifications")
@login_required
def list_notifications_api():
    user = _current_user()
    assert user is not None
    limit = _parse_int(dict(request.args), "limit", default=20)
    unread_only = request.args.get("unread_only", "").lower() in ("1", "true", "yes")
    notifications = _user_store().list_notifications(
        user["id"], limit=limit, unread_only=unread_only
    )
    unread_count = _user_store().count_unread(user["id"])
    return _json_response({
        "notifications": notifications,
        "unread_count": unread_count,
        "total": len(notifications),
    })


@app.get("/api/notifications/unread-count")
@login_required
def notification_unread_count_api():
    user = _current_user()
    assert user is not None
    count = _user_store().count_unread(user["id"])
    return _json_response({"unread_count": count})


@app.patch("/api/notifications/<int:noti_id>/read")
@login_required
def mark_notification_read_api(noti_id: int):
    user = _current_user()
    assert user is not None
    ok = _user_store().mark_read(noti_id, user["id"])
    if not ok:
        return _json_response({"error": "通知不存在"}, 404)
    return _json_response({"status": "read", "id": noti_id})


@app.post("/api/notifications/mark-all-read")
@login_required
def mark_all_notifications_read_api():
    user = _current_user()
    assert user is not None
    count = _user_store().mark_all_read(user["id"])
    return _json_response({"status": "all_read", "marked_count": count})


@app.delete("/api/notifications/<int:noti_id>")
@login_required
def delete_notification_api(noti_id: int):
    user = _current_user()
    assert user is not None
    ok = _user_store().delete_notification(noti_id, user["id"])
    if not ok:
        return _json_response({"error": "通知不存在"}, 404)
    return _json_response({"status": "deleted", "id": noti_id})


@app.get("/api/llm/info")
@login_required
def llm_info_api():
    """返回当前 LLM 客户端配置信息（不含 API Key 明文）。

    可用于前端判断 AI 功能是否已配置，以及使用的是哪个大模型。
    """
    client = get_llm_client()
    return _json_response(client.get_info())


@app.post("/api/llm/ping")
@login_required
def llm_ping_api():
    """向大模型发送测试请求，验证 API Key 和网络连通性。

    Response JSON
    ------------
    ok : bool           是否连通
    provider : str      厂商名称
    model : str         模型名称
    reply : str         模型回复（连通时）
    elapsed_ms : float  耗时
    error : str         错误信息（失败时）
    """
    client = get_llm_client()
    result = client.ping()
    status = 200 if result["ok"] else 503
    return _json_response(result, status)


@app.get("/api/benchmark/history")
@login_required
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


@app.get("/api/benchmark/export")
@login_required
def benchmark_export_csv():
    try:
        import csv
        import io

        history_id = _parse_optional_int(dict(request.args), "history_id")
        history = _load_benchmark_history()
        if not history:
            return _json_response({"error": "no benchmark history available"}, 404)

        if history_id is not None and 0 <= history_id < len(history):
            entry = history[history_id]
        else:
            entry = history[-1]

        k_values = entry.get("k_values", [10, 20, 50, 100])
        output = io.StringIO()
        writer = csv.writer(output)

        header = ["algorithm", "optimized", "available", "avg_ms", "qps", "memory_mb"]
        for kv in k_values:
            header.append(f"recall@{kv}")
        writer.writerow(header)

        for algo in entry.get("algorithms", []):
            recall_curve = algo.get("recall_curve", [])
            row = [
                algo.get("label", algo.get("id", "")),
                algo.get("optimized", False),
                algo.get("available", False),
                algo.get("avg_ms", ""),
                algo.get("qps", ""),
                algo.get("memory_mb", ""),
            ]
            for val in recall_curve:
                row.append(val if val is not None else "")
            writer.writerow(row)

        csv_content = output.getvalue()
        created_at = entry.get("created_at", "unknown").replace(" ", "_").replace(":", "")
        filename = f"benchmark_{created_at}.csv"

        return Response(
            csv_content,
            content_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
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
    if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
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
    import sys
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0").lower() in ("1", "true", "yes", "on")
    # macOS: FAISS(OpenMP) 在多线程模式下不安全，强制 threaded=False + use_reloader=False
    # Windows/Linux: 保持默认多线程，提升并发响应能力
    if sys.platform == "darwin":
        app.run(debug=debug, port=port, use_reloader=debug, threaded=False)
    else:
        app.run(debug=debug, port=port, use_reloader=debug, threaded=not debug)


if __name__ == "__main__":
    main()
