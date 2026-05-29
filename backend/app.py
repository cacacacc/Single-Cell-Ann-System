from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Flask, jsonify, request
from flask_cors import CORS

try:
    from .ann_indexer import ANNIndexer, IndexConfig
    from .data_reader import DataLoader
except ImportError:  # Allows running with: python backend/app.py
    from ann_indexer import ANNIndexer, IndexConfig
    from data_reader import DataLoader


DEFAULT_DATA_PATH = "data/liver.h5ad"
DEFAULT_INDEX_PATH = "indexes/cell_index.index"
DEFAULT_USE_REP = "X_pca"
DEFAULT_TOP_K = 10
MAX_TOP_K = 100


@dataclass
class SearchService:
    data_path: Path
    index_path: Path
    use_rep: str
    index_config: IndexConfig = field(default_factory=IndexConfig)
    loader: Optional[DataLoader] = None
    indexer: Optional[ANNIndexer] = None
    startup_error: Optional[str] = None

    @property
    def ready(self) -> bool:
        return self.loader is not None and self.indexer is not None

    def initialize(self) -> None:
        try:
            self.loader = DataLoader(self.data_path)
            dim = self.loader.vector_dim(self.use_rep)
            self.indexer = ANNIndexer(dim=dim, config=self.index_config)

            if self.index_path.exists():
                try:
                    self.indexer.load_index(self.index_path)
                except ValueError as exc:
                    if "config mismatch" in str(exc).lower():
                        vectors = self.loader.get_vectors(self.use_rep)
                        self.indexer.build_index(vectors)
                        self.indexer.save_index(self.index_path)
                    else:
                        raise
            else:
                vectors = self.loader.get_vectors(self.use_rep)
                self.indexer.build_index(vectors)
                self.indexer.save_index(self.index_path)

            self.startup_error = None
        except Exception as exc:
            self.loader = None
            self.indexer = None
            self.startup_error = str(exc)

    def status_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "ready": self.ready,
            "data_path": self.data_path.as_posix(),
            "index_path": self.index_path.as_posix(),
            "use_rep": self.use_rep,
            "index_config": self.index_config.to_dict(),
        }
        if self.startup_error:
            payload["error"] = self.startup_error
        if self.ready and self.loader is not None and self.indexer is not None:
            payload.update(
                {
                    "n_cells": self.loader.n_cells,
                    "n_genes": self.loader.n_genes,
                    "vector_dim": self.loader.vector_dim(self.use_rep),
                    "available_reps": self.loader.available_reps,
                    "obs_columns": self.loader.obs_columns,
                    "index_backend": self.indexer.backend,
                    "index_type": self.indexer.index_type,
                    "index_metric": self.indexer.metric,
                    "index_config": self.indexer.config_summary,
                }
            )
        return payload

    def search(
        self, cell_index: int, k: int, include_self: bool = False
    ) -> Dict[str, Any]:
        if not self.ready or self.loader is None or self.indexer is None:
            raise RuntimeError(self.startup_error or "Search service is not ready")

        query_vector = self.loader.get_vector(cell_index, use_rep=self.use_rep)
        search_k = min(k + 1 if not include_self else k, self.loader.n_cells)
        distances, indices = self.indexer.search(query_vector, search_k)

        results = []
        metric = self.indexer.metric
        for idx, dist in zip(indices.tolist(), distances.tolist()):
            idx = int(idx)
            distance = float(dist)
            if not include_self and idx == cell_index:
                continue

            cell_info = self.loader.get_cell_info(idx)
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

        return {
            "query_cell": cell_index,
            "k": k,
            "include_self": include_self,
            "use_rep": self.use_rep,
            "index_backend": self.indexer.backend,
            "index_type": self.indexer.index_type,
            "index_metric": metric,
            "index_config": self.indexer.config_summary,
            "results": results,
        }


def create_app(
    template_folder: Optional[Path] = None, static_folder: Optional[Path] = None
) -> Flask:
    if template_folder is not None and static_folder is not None:
        app = Flask(
            __name__,
            template_folder=str(template_folder),
            static_folder=str(static_folder),
        )
    elif template_folder is not None:
        app = Flask(__name__, template_folder=str(template_folder))
    elif static_folder is not None:
        app = Flask(__name__, static_folder=str(static_folder))
    else:
        app = Flask(__name__)
    CORS(app)

    service = SearchService(
        data_path=Path(os.getenv("CELL_DATA_PATH", DEFAULT_DATA_PATH)),
        index_path=Path(os.getenv("CELL_INDEX_PATH", DEFAULT_INDEX_PATH)),
        use_rep=os.getenv("CELL_USE_REP", DEFAULT_USE_REP),
        index_config=IndexConfig.from_env(),
    )
    service.initialize()
    app.config["search_service"] = service

    @app.get("/api")
    def api_root():
        return _json_response(
            {
                "message": "Single-Cell ANN API",
                "endpoints": ["/api/health", "/api/metadata", "/api/search"],
                "status": service.status_payload(),
            }
        )

    @app.get("/api/health")
    def health():
        status_code = 200 if service.ready else 503
        return _json_response(service.status_payload(), status_code)

    @app.get("/api/metadata")
    def metadata():
        if not service.ready:
            return _json_response({"error": service.startup_error}, 503)
        return _json_response(service.status_payload())

    @app.route("/api/search", methods=["GET", "POST"])
    def search():
        try:
            payload = _request_payload()
            cell_index = _parse_int(payload, "cell_index", required=True)
            k = _parse_int(payload, "k", default=DEFAULT_TOP_K)
            include_self = _parse_bool(payload.get("include_self"), default=False)

            if k <= 0:
                return _json_response({"error": "k must be a positive integer"}, 400)
            if k > MAX_TOP_K:
                return _json_response({"error": f"k must not exceed {MAX_TOP_K}"}, 400)

            return _json_response(service.search(cell_index, k, include_self=include_self))
        except (TypeError, ValueError, KeyError, IndexError) as exc:
            return _json_response({"error": str(exc)}, 400)
        except RuntimeError as exc:
            return _json_response({"error": str(exc)}, 503)

    return app


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


app = create_app()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
