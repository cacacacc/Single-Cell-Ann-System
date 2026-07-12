"""Approximate-nearest-neighbor indexing utilities for single-cell vectors.

The Flask app uses this module to build, search, save and reload ANN indexes
over PCA/UMAP/raw expression vectors. It hides optional backend differences
behind one API: FAISS when available, hnswlib for HNSW fallback, and NumPy
brute-force search as the always-available deterministic baseline.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    faiss = None

try:
    import hnswlib  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    hnswlib = None


_SUPPORTED_BACKENDS = {"auto", "faiss", "hnswlib", "numpy"}
_SUPPORTED_METRICS = {"l2", "ip", "cosine", "correlation"}
_SUPPORTED_INDEX_TYPES = {"flat", "ivf_flat", "hnsw", "pq", "brute"}


def _normalize_text(value: Any) -> str:
    return str(value).strip().lower()


def _ensure_int(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value is None or (isinstance(value, str) and value.strip() == ""):
        raise ValueError(f"{name} must be an integer")
    try:
        value_int = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value_int < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value_int


@dataclass(frozen=True)
class IndexConfig:
    """User-tunable ANN index parameters.

    ``backend="auto"`` lets the code pick the fastest installed dependency for
    the selected ``index_type``. IVF uses ``nlist``/``nprobe``; HNSW uses
    ``m``/``ef_*``; PQ uses ``pq_m``/``pq_nbits``.
    """

    backend: str = "auto"
    index_type: str = "flat"
    metric: str = "l2"
    nlist: int = 100
    nprobe: int = 10
    m: int = 16
    ef_construction: int = 200
    ef_search: int = 50
    pq_m: int = 8
    pq_nbits: int = 8

    def normalized(self) -> "IndexConfig":
        """Return a validated, lowercase config with numeric fields coerced."""
        backend = _normalize_text(self.backend or "auto")
        index_type = _normalize_text(self.index_type or "flat")
        metric = _normalize_text(self.metric or "l2")

        if backend not in _SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported backend: {backend}")
        if index_type not in _SUPPORTED_INDEX_TYPES:
            raise ValueError(f"Unsupported index_type: {index_type}")
        if metric not in _SUPPORTED_METRICS:
            raise ValueError(f"Unsupported metric: {metric}")

        nlist = _ensure_int(self.nlist, "nlist", minimum=1)
        nprobe = _ensure_int(self.nprobe, "nprobe", minimum=1)
        m = _ensure_int(self.m, "m", minimum=2)
        ef_construction = _ensure_int(self.ef_construction, "ef_construction", minimum=1)
        ef_search = _ensure_int(self.ef_search, "ef_search", minimum=1)
        pq_m = _ensure_int(self.pq_m, "pq_m", minimum=1)
        pq_nbits = _ensure_int(self.pq_nbits, "pq_nbits", minimum=1)

        return IndexConfig(
            backend=backend,
            index_type=index_type,
            metric=metric,
            nlist=nlist,
            nprobe=nprobe,
            m=m,
            ef_construction=ef_construction,
            ef_search=ef_search,
            pq_m=pq_m,
            pq_nbits=pq_nbits,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "index_type": self.index_type,
            "metric": self.metric,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "m": self.m,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "pq_m": self.pq_m,
            "pq_nbits": self.pq_nbits,
        }

    def update(self, **kwargs: Any) -> "IndexConfig":
        """Create a new config with non-None overrides applied."""
        data = self.to_dict()
        for key, value in kwargs.items():
            if value is None:
                continue
            data[key] = value
        return IndexConfig.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndexConfig":
        def _value(key: str, default: Any) -> Any:
            if key not in data:
                return default
            value = data.get(key, default)
            if value is None:
                return default
            if isinstance(value, str) and value.strip() == "":
                return default
            return value

        return cls(
            backend=_value("backend", cls.backend),
            index_type=_value("index_type", cls.index_type),
            metric=_value("metric", cls.metric),
            nlist=_value("nlist", cls.nlist),
            nprobe=_value("nprobe", cls.nprobe),
            m=_value("m", cls.m),
            ef_construction=_value("ef_construction", cls.ef_construction),
            ef_search=_value("ef_search", cls.ef_search),
            pq_m=_value("pq_m", cls.pq_m),
            pq_nbits=_value("pq_nbits", cls.pq_nbits),
        ).normalized()

    @classmethod
    def from_env(cls, prefix: str = "CELL_INDEX_") -> "IndexConfig":
        """Build config from environment variables such as CELL_INDEX_TYPE."""
        def _env(name: str, default: Any) -> Any:
            return os.getenv(f"{prefix}{name}", default)

        return cls.from_dict(
            {
                "backend": _env("BACKEND", cls.backend),
                "index_type": _env("TYPE", cls.index_type),
                "metric": _env("METRIC", cls.metric),
                "nlist": _env("NLIST", cls.nlist),
                "nprobe": _env("NPROBE", cls.nprobe),
                "m": _env("M", cls.m),
                "ef_construction": _env("EF_CONSTRUCTION", cls.ef_construction),
                "ef_search": _env("EF_SEARCH", cls.ef_search),
                "pq_m": _env("PQ_M", cls.pq_m),
                "pq_nbits": _env("PQ_NBITS", cls.pq_nbits),
            }
        )


class ANNIndexer:
    """ANN index wrapper with configurable backends and metrics.

    The instance keeps a copy of indexed vectors even when a native backend is
    used. That costs memory, but gives portable persistence, exact re-scoring,
    metadata compatibility checks, and a reliable NumPy fallback.
    """

    def __init__(self, dim: int, config: Optional[IndexConfig] = None):
        self.dim = self._validate_dim(dim)
        self._config = (config or IndexConfig()).normalized()
        self._backend: Optional[str] = None
        self._index_type: Optional[str] = None
        self._metric = self._config.metric
        self._index = None
        self._vectors: Optional[np.ndarray] = None
        self._count = 0

    @property
    def backend(self) -> Optional[str]:
        return self._backend

    @property
    def metric(self) -> str:
        return self._metric

    @property
    def index_type(self) -> Optional[str]:
        return self._index_type

    @property
    def config(self) -> IndexConfig:
        return self._config

    @property
    def config_summary(self) -> Dict[str, Any]:
        return {
            "backend": self._backend or self._config.backend,
            "index_type": self._index_type or self._config.index_type,
            "metric": self._metric,
            "nlist": self._config.nlist,
            "nprobe": self._config.nprobe,
            "m": self._config.m,
            "ef_construction": self._config.ef_construction,
            "ef_search": self._config.ef_search,
            "pq_m": self._config.pq_m,
            "pq_nbits": self._config.pq_nbits,
        }

    @property
    def is_built(self) -> bool:
        return self._backend is not None and self._count > 0

    def build_index(self, vectors) -> "ANNIndexer":
        """Validate vectors, choose a backend, and build the concrete index."""
        prepared = self._prepare_vectors(self._validate_vectors(vectors))
        self._vectors = prepared
        self._count = int(prepared.shape[0])

        backend = self._select_backend()
        index_type = self._resolve_index_type(backend)
        self._build_backend_index(prepared, backend, index_type)
        return self

    def search(self, query_vector, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return top-k distances and row indices for one query vector.

        Native ANN libraries may use slightly different distance conventions,
        so candidates from FAISS/hnswlib are re-scored with ``_compute_distances``
        before returning to callers.
        """
        self._ensure_index_ready()
        k = self._validate_k(k)
        if k > self._count:
            raise ValueError(
                f"k must not exceed indexed vector count ({self._count}), got {k}"
            )

        query = self._prepare_query(self._validate_query(query_vector))

        if self._backend == "faiss" and self._index is not None and faiss is not None:
            distances, indices = self._index.search(query.reshape(1, -1), k)
            indices = np.asarray(indices[0], dtype=np.int64)
            indices, distances = self._recompute_and_sort(query, indices)
            return distances, indices

        if self._backend == "hnswlib" and self._index is not None and hnswlib is not None:
            indices, _ = self._index.knn_query(query.reshape(1, -1), k=k)
            indices = np.asarray(indices[0], dtype=np.int64)
            indices, distances = self._recompute_and_sort(query, indices)
            return distances, indices

        assert self._vectors is not None
        distances = self._compute_distances(query, self._vectors)

        if k == self._count:
            topk_indices = np.arange(self._count, dtype=np.int64)
        else:
            topk_indices = np.argpartition(distances, k - 1)[:k].astype(np.int64)

        order = np.argsort(distances[topk_indices], kind="stable")
        topk_indices = topk_indices[order]
        topk_distances = distances[topk_indices].astype(np.float32, copy=False)
        return topk_distances, topk_indices

    def save_index(self, index_path, **extra_config) -> None:
        """Persist native index files plus a portable ``.npz`` vector archive."""
        self._ensure_index_ready()
        path = Path(index_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self._backend == "faiss" and faiss is not None and self._index is not None:
            faiss.write_index(self._index, str(path))
            self._save_archive(self._backup_archive_path(path), backend="faiss", **extra_config)
            return

        if self._backend == "hnswlib" and hnswlib is not None and self._index is not None:
            self._index.save_index(str(path))
            self._save_archive(self._backup_archive_path(path), backend="hnswlib", **extra_config)
            return

        self._save_archive(path, backend="numpy", **extra_config)

    def load_index(self, index_path) -> "ANNIndexer":
        """Load an index saved by ``save_index``.

        The portable archive is preferred because it stores normalized config
        and vectors. Legacy raw FAISS files are still accepted when no archive
        exists.
        """
        path = Path(index_path)
        backup_path = self._backup_archive_path(path)

        if backup_path.exists():
            return self._load_archive(backup_path)

        if path.exists():
            archive_error: Optional[Exception] = None
            try:
                return self._load_archive(path)
            except Exception as exc:
                archive_error = exc

            if faiss is not None:
                try:
                    index = faiss.read_index(str(path))
                    if index.d != self.dim:
                        raise ValueError(
                            f"Loaded index dimension {index.d} does not match ANNIndexer dim {self.dim}"
                        )
                    self._index = index
                    self._backend = "faiss"
                    self._index_type = "flat"
                    self._metric = "l2"
                    self._count = int(index.ntotal)
                    self._vectors = self._extract_vectors_from_index()
                    return self
                except Exception:
                    if archive_error is not None:
                        raise archive_error
                    raise

            if archive_error is not None:
                raise archive_error

        raise FileNotFoundError(f"Index file not found: {path}")

    def _load_archive(self, archive_path: Path) -> "ANNIndexer":
        """Restore vectors/config from the backend-independent NumPy archive."""
        with np.load(archive_path, allow_pickle=False) as data:
            vectors = np.asarray(data["vectors"], dtype=np.float32)
            dim = int(np.asarray(data["dim"]).item())
            backend = (
                str(np.asarray(data["backend"]).item())
                if "backend" in data
                else "numpy"
            )
            index_type = (
                str(np.asarray(data["index_type"]).item())
                if "index_type" in data
                else "flat"
            )
            metric = (
                str(np.asarray(data["metric"]).item()) if "metric" in data else "l2"
            )
            count = int(np.asarray(data["count"]).item())
            config_json = (
                str(np.asarray(data["config_json"]).item())
                if "config_json" in data
                else None
            )

        if dim != self.dim:
            raise ValueError(
                f"Loaded index dimension {dim} does not match ANNIndexer dim {self.dim}"
            )

        vectors = self._prepare_vectors(self._validate_vectors(vectors))
        if count != vectors.shape[0]:
            raise ValueError(
                f"Loaded index count {count} does not match vectors count {vectors.shape[0]}"
            )

        self._count = count

        if config_json:
            saved_config = IndexConfig.from_dict(json.loads(config_json))
        else:
            saved_config = IndexConfig.from_dict(
                {"backend": backend, "index_type": index_type, "metric": metric}
            )

        if not self._config_matches_saved(saved_config):
            raise ValueError(
                "Index config mismatch: please rebuild with the requested backend/index type/metric"
            )

        self._vectors = vectors
        self._metric = saved_config.metric

        resolved_backend = self._resolve_backend(saved_config.backend)
        resolved_index_type = self._resolve_index_type(resolved_backend)
        self._build_backend_index(vectors, resolved_backend, resolved_index_type)
        return self

    def _save_archive(self, archive_path: Path, backend: str, **extra_config) -> None:
        """Write the archive used for safe reloads and backend migration."""
        vectors = self._vectors
        if vectors is None:
            vectors = self._extract_vectors_from_index()
        if vectors is None:
            raise RuntimeError("Indexed vectors are unavailable for archival save")

        config_payload = self._config_for_storage(backend)
        config_payload.update(extra_config)  # e.g. use_rep

        payload = {
            "vectors": np.asarray(vectors, dtype=np.float32),
            "dim": np.array(self.dim, dtype=np.int64),
            "backend": np.array(backend),
            "index_type": np.array(self._index_type or config_payload.get("index_type")),
            "metric": np.array(self._metric),
            "count": np.array(self._count, dtype=np.int64),
            "config_json": np.array(json.dumps(config_payload, ensure_ascii=False)),
        }

        with open(archive_path, "wb") as handle:
            np.savez(handle, **payload)

    def _extract_vectors_from_index(self) -> Optional[np.ndarray]:
        if self._vectors is not None:
            return self._vectors

        if faiss is None or self._index is None:
            return None

        if not hasattr(self._index, "reconstruct"):
            return None

        vectors = []
        for i in range(self._count):
            vectors.append(np.asarray(self._index.reconstruct(i), dtype=np.float32))
        if not vectors:
            return None
        return np.ascontiguousarray(np.vstack(vectors), dtype=np.float32)

    def _ensure_index_ready(self) -> None:
        if not self.is_built:
            raise RuntimeError("ANN index is not built yet")

    def _validate_dim(self, dim: int) -> int:
        if isinstance(dim, bool) or not isinstance(dim, (int, np.integer)):
            raise TypeError("dim must be an integer")
        dim = int(dim)
        if dim <= 0:
            raise ValueError("dim must be a positive integer")
        return dim

    def _validate_vectors(self, vectors) -> np.ndarray:
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("vectors must be a 2D array")
        if arr.shape[0] == 0 or arr.shape[1] == 0:
            raise ValueError("vectors must not be empty")
        if arr.shape[1] != self.dim:
            raise ValueError(
                f"vector dimension mismatch: expected {self.dim}, got {arr.shape[1]}"
            )
        return np.ascontiguousarray(arr, dtype=np.float32)

    def _validate_query(self, query_vector) -> np.ndarray:
        arr = np.asarray(query_vector, dtype=np.float32)
        if arr.ndim == 1:
            if arr.shape[0] != self.dim:
                raise ValueError(
                    f"query_vector dimension mismatch: expected {self.dim}, got {arr.shape[0]}"
                )
            return np.ascontiguousarray(arr, dtype=np.float32)

        if arr.ndim == 2 and arr.shape[0] == 1 and arr.shape[1] == self.dim:
            return np.ascontiguousarray(arr[0], dtype=np.float32)

        raise ValueError("query_vector must be a 1D vector or a (1, dim) array")

    def _validate_k(self, k: int) -> int:
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
            raise TypeError("k must be an integer")
        k = int(k)
        if k <= 0:
            raise ValueError("k must be a positive integer")
        return k

    def _backup_archive_path(self, index_path: Path) -> Path:
        return Path(f"{index_path}.npz")

    def _config_for_storage(self, backend: str) -> Dict[str, Any]:
        payload = self._config.to_dict()
        payload["backend"] = backend
        payload["index_type"] = self._index_type or payload["index_type"]
        payload["metric"] = self._metric
        if (self._index_type or payload["index_type"]) == "ivf_flat":
            payload["nlist"] = min(payload["nlist"], self._count)
        return payload

    def _normalize_index_type_for_compare(self, index_type: str, backend: str) -> str:
        if backend == "numpy":
            return "brute"
        if index_type == "brute":
            return "flat"
        return index_type

    def _config_matches_saved(self, saved: IndexConfig) -> bool:
        """Guard against loading an index with incompatible runtime settings."""
        current = self._config.normalized()
        if current.backend != "auto" and current.backend != saved.backend:
            return False
        if current.metric != saved.metric:
            return False

        backend_for_compare = saved.backend if current.backend == "auto" else current.backend
        current_index_type = self._normalize_index_type_for_compare(
            current.index_type, backend_for_compare
        )
        saved_index_type = self._normalize_index_type_for_compare(
            saved.index_type, saved.backend
        )
        if current_index_type != saved_index_type:
            return False

        if saved_index_type == "ivf_flat":
            current_nlist = min(current.nlist, self._count)
            if current_nlist != saved.nlist:
                return False
        if saved_index_type == "hnsw":
            if current.m != saved.m or current.ef_construction != saved.ef_construction:
                return False
        if saved_index_type == "pq":
            if current.pq_m != saved.pq_m or current.pq_nbits != saved.pq_nbits:
                return False
        return True

    def _prepare_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine/correlation before indexing."""
        if self._config.metric in ("cosine", "correlation"):
            prepared = np.ascontiguousarray(vectors, dtype=np.float32)
            if self._config.metric == "correlation":
                prepared = prepared - prepared.mean(axis=1, keepdims=True)
            return self._normalize_rows(prepared)
        return np.ascontiguousarray(vectors, dtype=np.float32)

    def _prepare_query(self, query: np.ndarray) -> np.ndarray:
        if self._metric in ("cosine", "correlation"):
            prepared = np.ascontiguousarray(query, dtype=np.float32)
            if self._metric == "correlation":
                prepared = prepared - prepared.mean()
            return self._normalize_row(prepared)
        return np.ascontiguousarray(query, dtype=np.float32)

    def _normalize_rows(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return np.ascontiguousarray(vectors / norms, dtype=np.float32)

    def _normalize_row(self, vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            return np.ascontiguousarray(vector, dtype=np.float32)
        return np.ascontiguousarray(vector / norm, dtype=np.float32)

    def _select_backend(self) -> str:
        """Choose the concrete backend requested by ``IndexConfig``."""
        backend = self._config.backend
        index_type = self._config.index_type

        if backend != "auto":
            if backend == "faiss" and faiss is None:
                raise ImportError("faiss is not available")
            if backend == "hnswlib" and hnswlib is None:
                raise ImportError("hnswlib is not available")
            if backend == "numpy":
                return "numpy"
            return backend

        if index_type == "ivf_flat":
            if faiss is None:
                raise ImportError("faiss is required for ivf_flat index_type")
            return "faiss"

        if index_type == "pq":
            if faiss is None:
                raise ImportError("faiss is required for pq index_type")
            return "faiss"

        if index_type == "hnsw":
            if faiss is not None:
                return "faiss"
            if hnswlib is not None:
                return "hnswlib"
            raise ImportError("faiss or hnswlib is required for hnsw index_type")

        if faiss is not None:
            return "faiss"
        return "numpy"

    def _resolve_backend(self, saved_backend: str) -> str:
        if self._config.backend != "auto":
            return self._select_backend()
        if saved_backend == "faiss" and faiss is not None:
            return "faiss"
        if saved_backend == "hnswlib" and hnswlib is not None:
            return "hnswlib"
        return self._select_backend()

    def _resolve_index_type(self, backend: str) -> str:
        index_type = self._config.index_type
        if backend == "faiss":
            if index_type == "brute":
                return "flat"
            if index_type not in {"flat", "ivf_flat", "hnsw", "pq"}:
                raise ValueError(f"Unsupported index_type for faiss: {index_type}")
            return index_type

        if backend == "hnswlib":
            if index_type != "hnsw":
                raise ValueError("hnswlib only supports index_type=hnsw")
            return "hnsw"

        if backend == "numpy":
            if index_type not in {"flat", "brute"}:
                raise ValueError("numpy backend only supports flat/brute index_type")
            return "brute"

        raise ValueError(f"Unsupported backend: {backend}")

    def _build_backend_index(self, vectors: np.ndarray, backend: str, index_type: str) -> None:
        self._backend = backend
        self._index_type = index_type
        self._metric = self._config.metric

        if backend == "faiss":
            self._index = self._build_faiss_index(vectors, index_type)
            return

        if backend == "hnswlib":
            self._index = self._build_hnswlib_index(vectors)
            return

        self._index = None

    def _faiss_metric(self) -> int:
        if faiss is None:
            raise ImportError("faiss is not available")
        if self._metric == "l2":
            return faiss.METRIC_L2
        if self._metric in ("cosine", "correlation", "ip"):
            return faiss.METRIC_INNER_PRODUCT
        return faiss.METRIC_L2

    def _build_faiss_index(self, vectors: np.ndarray, index_type: str):
        """Create and train the requested FAISS index type."""
        if faiss is None:
            raise ImportError("faiss is not available")

        metric = self._faiss_metric()
        if index_type == "flat":
            if metric == faiss.METRIC_L2:
                index = faiss.IndexFlatL2(self.dim)
            else:
                index = faiss.IndexFlatIP(self.dim)
        elif index_type == "ivf_flat":
            nlist = min(self._config.nlist, self._count)
            if nlist <= 0:
                raise ValueError("nlist must be a positive integer")
            if metric == faiss.METRIC_L2:
                quantizer = faiss.IndexFlatL2(self.dim)
            else:
                quantizer = faiss.IndexFlatIP(self.dim)
            index = faiss.IndexIVFFlat(quantizer, self.dim, int(nlist), metric)
            index.train(vectors)
        elif index_type == "pq":
            pq_m = int(self._config.pq_m)
            pq_nbits = int(self._config.pq_nbits)
            if self.dim % pq_m != 0:
                raise ValueError(
                    f"pq_m ({pq_m}) must divide vector dimension {self.dim}"
                )
            if not 1 <= pq_nbits <= 16:
                raise ValueError("pq_nbits must be between 1 and 16")
            try:
                index = faiss.IndexPQ(self.dim, pq_m, pq_nbits, metric)
            except TypeError:
                if metric != faiss.METRIC_L2:
                    raise ValueError(
                        "FAISS IndexPQ in this build only supports L2 metric"
                    )
                index = faiss.IndexPQ(self.dim, pq_m, pq_nbits)
            index.train(vectors)
        elif index_type == "hnsw":
            index = faiss.IndexHNSWFlat(self.dim, int(self._config.m), metric)
            index.hnsw.efConstruction = int(self._config.ef_construction)
        else:
            raise ValueError(f"Unsupported index_type for faiss: {index_type}")

        index.add(vectors)

        if index_type == "ivf_flat":
            index.nprobe = int(min(self._config.nprobe, nlist))
        if index_type == "hnsw":
            index.hnsw.efSearch = int(self._config.ef_search)

        return index

    def _build_hnswlib_index(self, vectors: np.ndarray):
        if hnswlib is None:
            raise ImportError("hnswlib is not available")

        space = self._metric
        if space in ("cosine", "correlation"):
            space = "cosine"
        elif space == "ip":
            space = "ip"
        else:
            space = "l2"

        index = hnswlib.Index(space=space, dim=self.dim)
        index.init_index(
            max_elements=self._count,
            ef_construction=int(self._config.ef_construction),
            M=int(self._config.m),
        )
        index.add_items(vectors, ids=np.arange(self._count))
        index.set_ef(int(self._config.ef_search))
        return index

    def _compute_distances(self, query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        if self._metric == "l2":
            diffs = vectors - query
            return np.sum(diffs * diffs, axis=1, dtype=np.float32)
        if self._metric == "cosine":
            return 1.0 - np.sum(vectors * query, axis=1, dtype=np.float32)
        if self._metric == "correlation":
            q_centered = query - query.mean()
            q_norm = np.linalg.norm(q_centered)
            if q_norm > 0:
                q_centered = q_centered / q_norm
            v_centered = vectors - vectors.mean(axis=1, keepdims=True)
            v_norms = np.linalg.norm(v_centered, axis=1, keepdims=True)
            v_norms = np.where(v_norms == 0, 1.0, v_norms)
            v_centered = v_centered / v_norms
            return 1.0 - np.sum(v_centered * q_centered, axis=1, dtype=np.float32)
        return -np.sum(vectors * query, axis=1, dtype=np.float32)

    def _recompute_and_sort(
        self, query: np.ndarray, indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if indices.size == 0:
            return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float32)

        valid_mask = indices >= 0
        indices = indices[valid_mask]
        if indices.size == 0:
            return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float32)

        assert self._vectors is not None
        distances = self._compute_distances(query, self._vectors[indices])
        order = np.argsort(distances, kind="stable")
        indices = indices[order].astype(np.int64, copy=False)
        distances = distances[order].astype(np.float32, copy=False)
        return indices, distances


class PCAReducer:
    """基于 NumPy SVD 的 PCA 降维预处理器。

    针对单细胞高维稀疏数据，在 ANN 索引构建前对向量进行降维，
    减少计算量和内存占用，同时保留主要方差信息。

    Parameters
    ----------
    n_components : int
        降维后的目标维度数。
    """

    def __init__(self, n_components: int):
        if n_components < 1:
            raise ValueError("n_components must be >= 1")
        self.n_components = n_components
        self._mean: Optional[np.ndarray] = None
        self._components: Optional[np.ndarray] = None
        self._explained_variance_ratio: Optional[np.ndarray] = None

    @property
    def is_fitted(self) -> bool:
        return self._mean is not None and self._components is not None

    @property
    def explained_variance_ratio_(self) -> Optional[np.ndarray]:
        return self._explained_variance_ratio

    def fit(self, vectors: np.ndarray) -> "PCAReducer":
        vectors = np.asarray(vectors, dtype=np.float64)
        if vectors.ndim != 2:
            raise ValueError("vectors must be 2D")
        n_samples, n_features = vectors.shape
        n_components = min(self.n_components, n_samples, n_features)

        self._mean = vectors.mean(axis=0)
        centered = vectors - self._mean

        _, s, vt = np.linalg.svd(centered, full_matrices=False)
        self._components = vt[:n_components].astype(np.float32)

        explained_var = (s[:n_components] ** 2) / (n_samples - 1)
        total_var = (s ** 2).sum() / (n_samples - 1)
        self._explained_variance_ratio = (explained_var / total_var).astype(np.float64)

        return self

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("PCAReducer has not been fitted")
        vectors = np.asarray(vectors, dtype=np.float32)
        centered = vectors - self._mean.astype(np.float32)
        return np.ascontiguousarray(centered @ self._components.T, dtype=np.float32)

    def fit_transform(self, vectors: np.ndarray) -> np.ndarray:
        self.fit(vectors)
        return self.transform(vectors)

    def inverse_transform(self, reduced: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("PCAReducer has not been fitted")
        reduced = np.asarray(reduced, dtype=np.float32)
        return np.ascontiguousarray(
            reduced @ self._components + self._mean.astype(np.float32),
            dtype=np.float32,
        )

    def save(self, path: Union[str, Path]) -> None:
        if not self.is_fitted:
            raise RuntimeError("PCAReducer has not been fitted")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(path),
            n_components=self.n_components,
            mean=self._mean,
            components=self._components,
            explained_variance_ratio=self._explained_variance_ratio,
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PCAReducer":
        path = Path(path)
        with np.load(str(path), allow_pickle=False) as data:
            reducer = cls(n_components=int(data["n_components"]))
            reducer._mean = data["mean"]
            reducer._components = data["components"]
            reducer._explained_variance_ratio = data["explained_variance_ratio"]
        return reducer


def _demo() -> None:
    rng = np.random.default_rng(42)
    vectors = rng.random((32, 8), dtype=np.float32)

    config = IndexConfig(backend="auto", index_type="flat", metric="l2")
    indexer = ANNIndexer(dim=8, config=config)
    indexer.build_index(vectors)
    distances, indices = indexer.search(vectors[0], k=5)

    print(f"backend: {indexer.backend}")
    print(f"index_type: {indexer.index_type}")
    print(f"metric: {indexer.metric}")
    print(f"indices: {indices.tolist()}")
    print(f"distances: {distances.tolist()}")


if __name__ == "__main__":
    _demo()
