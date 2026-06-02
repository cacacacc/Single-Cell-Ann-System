from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np

import app as app_module


class FlaskAppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.tmpdir.name)
        self.data_dir = self.base_path / "data"
        self.index_dir = self.base_path / "indexes"
        self.data_dir.mkdir()
        self.index_dir.mkdir()
        self.dataset_path = self.data_dir / "tiny.h5ad"

        adata = ad.AnnData(
            X=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            obs={"cell_type": ["query", "a", "b", "c"]},
        )
        adata.obs_names = ["cell-0", "cell-1", "cell-2", "cell-3"]
        adata.obsm["X_pca"] = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [2.0, 2.0],
            ],
            dtype=np.float32,
        )
        adata.obsm["X_umap"] = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [2.0, 2.0],
            ],
            dtype=np.float32,
        )
        adata.write_h5ad(self.dataset_path)

        self.original_data_dir = app_module.DATA_DIR
        self.original_index_dir = app_module.INDEX_DIR
        self.original_default_data_path = app_module.DEFAULT_DATA_PATH

        app_module.DATA_DIR = self.data_dir
        app_module.INDEX_DIR = self.index_dir
        app_module.DEFAULT_DATA_PATH = str(self.dataset_path)
        app_module._DATASET_CACHE.clear()
        app_module._INDEX_CACHE.clear()
        app_module._BENCHMARK_INDEX_CACHE.clear()

        self.app = app_module.create_app()
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        app_module.DATA_DIR = self.original_data_dir
        app_module.INDEX_DIR = self.original_index_dir
        app_module.DEFAULT_DATA_PATH = self.original_default_data_path
        app_module._DATASET_CACHE.clear()
        app_module._INDEX_CACHE.clear()
        app_module._BENCHMARK_INDEX_CACHE.clear()
        self.tmpdir.cleanup()

    def test_health_reports_default_dataset_metadata(self) -> None:
        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["dataset_id"], "tiny")
        self.assertEqual(payload["n_cells"], 4)
        self.assertEqual(payload["n_genes"], 3)
        self.assertEqual(payload["use_rep"], "X_pca")
        self.assertFalse(payload["ready"])

    def test_datasets_lists_available_h5ad_files(self) -> None:
        response = self.client.get("/api/datasets")

        self.assertEqual(response.status_code, 200)
        datasets = response.get_json()["datasets"]
        self.assertEqual(len(datasets), 1)
        self.assertEqual(datasets[0]["id"], "tiny")
        self.assertEqual(datasets[0]["index_status"], "missing")

    def test_index_build_and_search_return_results(self) -> None:
        build_response = self.client.post(
            "/api/index/build",
            json={
                "dataset_id": "tiny",
                "use_rep": "X_pca",
                "index_backend": "numpy",
                "index_type": "brute",
            },
        )

        self.assertEqual(build_response.status_code, 200)
        build_payload = build_response.get_json()
        self.assertEqual(build_payload["status"], "built")
        self.assertEqual(build_payload["dataset_id"], "tiny")
        self.assertEqual(build_payload["backend"], "numpy")

        search_response = self.client.post(
            "/api/search",
            json={
                "dataset_id": "tiny",
                "cell_id": "cell-0",
                "k": 2,
                "index_backend": "numpy",
                "index_type": "brute",
            },
        )

        self.assertEqual(search_response.status_code, 200)
        search_payload = search_response.get_json()
        self.assertEqual(search_payload["dataset_id"], "tiny")
        self.assertEqual(search_payload["cell_id"], "cell-0")
        self.assertEqual(search_payload["index_backend"], "numpy")
        self.assertEqual(len(search_payload["results"]), 2)
        self.assertEqual(search_payload["results"][0]["cell_id"], "cell-1")

    def test_search_requires_cell_index_or_cell_id(self) -> None:
        response = self.client.post("/api/search", json={"dataset_id": "tiny", "k": 2})

        self.assertEqual(response.status_code, 400)
        self.assertIn("cell_index or cell_id is required", response.get_json()["error"])

    def test_cells_endpoint_paginates_cell_ids(self) -> None:
        response = self.client.get("/api/cells?dataset_id=tiny&offset=1&limit=2")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["total"], 4)
        self.assertEqual(payload["next_offset"], 3)
        self.assertEqual([cell["cell_id"] for cell in payload["cells"]], ["cell-1", "cell-2"])

    def test_umap_endpoint_returns_sampled_points(self) -> None:
        response = self.client.get("/api/umap?dataset_id=tiny&limit=2&seed=1")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["dataset_id"], "tiny")
        self.assertEqual(payload["sampled"], 2)
        self.assertEqual(len(payload["points"]), 2)

    def test_json_safe_converts_non_finite_float_to_none(self) -> None:
        payload = {"value": math.nan, "nested": [{"value": math.inf}]}

        self.assertEqual(
            app_module._json_safe(payload),
            {"value": None, "nested": [{"value": None}]},
        )


if __name__ == "__main__":
    unittest.main()
