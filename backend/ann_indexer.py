from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    faiss = None


class ANNIndexer:
    """ANN index wrapper with FAISS-first, NumPy fallback behavior."""

    def __init__(self, dim: int):
        self.dim = self._validate_dim(dim)
        self._backend: Optional[str] = None
        self._index = None
        self._vectors: Optional[np.ndarray] = None
        self._count = 0

    @property
    def backend(self) -> Optional[str]:
        return self._backend

    @property
    def is_built(self) -> bool:
        return self._backend is not None and self._count > 0

    def build_index(self, vectors) -> "ANNIndexer":
        prepared = self._validate_vectors(vectors)
        self._vectors = prepared
        self._count = int(prepared.shape[0])

        if faiss is not None:
            index = faiss.IndexFlatL2(self.dim)
            index.add(prepared)
            self._index = index
            self._backend = "faiss"
        else:
            self._index = None
            self._backend = "numpy"

        return self

    def search(self, query_vector, k: int) -> Tuple[np.ndarray, np.ndarray]:
        self._ensure_index_ready()
        k = self._validate_k(k)
        if k > self._count:
            raise ValueError(
                f"k must not exceed indexed vector count ({self._count}), got {k}"
            )

        query = self._validate_query(query_vector)

        if self._backend == "faiss" and self._index is not None and faiss is not None:
            distances, indices = self._index.search(query.reshape(1, -1), k)
            return (
                np.asarray(distances[0], dtype=np.float32),
                np.asarray(indices[0], dtype=np.int64),
            )

        assert self._vectors is not None
        diffs = self._vectors - query
        distances = np.sum(diffs * diffs, axis=1, dtype=np.float32)

        if k == self._count:
            topk_indices = np.arange(self._count, dtype=np.int64)
        else:
            topk_indices = np.argpartition(distances, k - 1)[:k].astype(np.int64)

        order = np.argsort(distances[topk_indices], kind="stable")
        topk_indices = topk_indices[order]
        topk_distances = distances[topk_indices].astype(np.float32, copy=False)
        return topk_distances, topk_indices

    def save_index(self, index_path) -> None:
        self._ensure_index_ready()
        path = Path(index_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self._backend == "faiss" and faiss is not None and self._index is not None:
            faiss.write_index(self._index, str(path))
            self._save_archive(self._backup_archive_path(path), backend="faiss")
            return

        self._save_archive(path, backend="numpy")

    def load_index(self, index_path) -> "ANNIndexer":
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
        with np.load(archive_path, allow_pickle=False) as data:
            vectors = np.asarray(data["vectors"], dtype=np.float32)
            dim = int(np.asarray(data["dim"]).item())
            backend = str(np.asarray(data["backend"]).item())
            count = int(np.asarray(data["count"]).item())

        if dim != self.dim:
            raise ValueError(
                f"Loaded index dimension {dim} does not match ANNIndexer dim {self.dim}"
            )

        vectors = self._validate_vectors(vectors)
        if count != vectors.shape[0]:
            raise ValueError(
                f"Loaded index count {count} does not match vectors count {vectors.shape[0]}"
            )

        self._vectors = vectors
        self._count = count

        if backend == "faiss" and faiss is not None:
            index = faiss.IndexFlatL2(self.dim)
            index.add(self._vectors)
            self._index = index
            self._backend = "faiss"
        else:
            self._index = None
            self._backend = "numpy"

        return self

    def _save_archive(self, archive_path: Path, backend: str) -> None:
        vectors = self._vectors
        if vectors is None:
            vectors = self._extract_vectors_from_index()
        if vectors is None:
            raise RuntimeError("Indexed vectors are unavailable for archival save")

        payload = {
            "vectors": np.asarray(vectors, dtype=np.float32),
            "dim": np.array(self.dim, dtype=np.int64),
            "backend": np.array(backend),
            "count": np.array(self._count, dtype=np.int64),
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


def _demo() -> None:
    rng = np.random.default_rng(42)
    vectors = rng.random((32, 8), dtype=np.float32)

    indexer = ANNIndexer(dim=8)
    indexer.build_index(vectors)
    distances, indices = indexer.search(vectors[0], k=5)

    print(f"backend: {indexer.backend}")
    print(f"indices: {indices.tolist()}")
    print(f"distances: {distances.tolist()}")


if __name__ == "__main__":
    _demo()
