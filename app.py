from __future__ import annotations

import json
import math
import os
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

import numpy as np
from flask import Flask, Response, jsonify, redirect, render_template, request, session, stream_with_context, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename

from backend.ann_indexer import ANNIndexer, IndexConfig
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

DEFAULT_DATA_PATH = "data/liver.h5ad"
DEFAULT_USE_REP = "X_pca"
DEFAULT_TOP_K = 10
MAX_TOP_K = 100
FILTER_SEARCH_MULTIPLIER = 10
ALLOWED_EXTENSIONS = {".h5ad"}

_DATASET_CACHE: Dict[str, DataLoader] = {}
_INDEX_CACHE: Dict[Tuple[str, str], Tuple[ANNIndexer, Tuple[Any, ...]]] = {}
_BENCHMARK_INDEX_CACHE: Dict[Tuple[str, str, Tuple[Any, ...]], ANNIndexer] = {}
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

    # --- unnamed (on-the-fly) path — don't persist ---
    loader = _get_loader(dataset_id)
    indexer = ANNIndexer(dim=loader.vector_dim(use_rep), config=index_config)
    vectors = loader.get_vectors(use_rep)
    indexer.build_index(vectors)
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
        datasets.append(_dataset_payload(dataset_id, path))
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
@login_required
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

        # 解析查询向量（可选，不提供则由引擎调用 Embedding API）
        query_vector = _resolve_query_vector(payload, loader, use_rep)

        # 执行 RAG
        result = engine.ask(
            question=question,
            query_vector=query_vector,
            session_id=session_id,
            n_results=n_results,
            where_filter=where,
        )

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

        # 向量检索 + Prompt 组装

        retrieved = store.query_similar(
            query_vector=query_vector,
            n_results=n_results,
            where=where,
        )
        history = _CHAT_HISTORY.get(session_id) if session_id else None
        builder = engine._builder  # type: ignore[attr-defined]
        if history:
            messages = builder.build_messages_with_history(
                user_question=question,
                retrieved_cells=retrieved,
                history=history,
                extra_system_info=engine._dataset_info,  # type: ignore[attr-defined]
            )
        else:
            messages = builder.build_messages(
                user_question=question,
                retrieved_cells=retrieved,
                extra_system_info=engine._dataset_info,  # type: ignore[attr-defined]
            )

        llm_client = get_llm_client()

        # SSE 生成器
        def generate():
            full_text_parts = []
            try:
                for chunk in llm_client.stream_chat(messages=messages):
                    full_text_parts.append(chunk)
                    # SSE 格式：data: <内容>\n\n
                    yield f"data: {chunk}\n\n"
            except Exception as exc:
                yield f"data: [ERROR] {exc}\n\n"
                return
            # 流结束后保存历史
            if session_id and full_text_parts:
                _CHAT_HISTORY.append(session_id, question, "".join(full_text_parts))
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
    """获取指定会话的对话历史。

    Query Params
    -----------
    session_id : str  （必填）
    """
    session_id = request.args.get("session_id", "").strip()
    if not session_id:
        return _json_response({"error": "session_id 不能为空"}, 400)
    history = _CHAT_HISTORY.get(session_id)
    return _json_response({"session_id": session_id, "history": history, "rounds": len(history) // 2})


@app.delete("/api/chat/history")
@login_required
def clear_chat_history_api():
    """清空指定会话的对话历史。

    Query Params
    -----------
    session_id : str  （必填）
    """
    session_id = request.args.get("session_id", "").strip()
    if not session_id:
        return _json_response({"error": "session_id 不能为空"}, 400)
    _CHAT_HISTORY.clear(session_id)
    return _json_response({"status": "cleared", "session_id": session_id})


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
    port = int(os.getenv("PORT", "5000"))
    app.run(debug=True, port=port)


if __name__ == "__main__":
    main()
