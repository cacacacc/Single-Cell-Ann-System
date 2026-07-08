from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import anndata as ad
import numpy as np

import app as app_module


class FakeChatStore:
    def is_populated(self) -> bool:
        return True

    def query_by_keywords(self, keywords, n_results=5):
        return []

    def query_by_metadata(self, where=None, limit=5):
        return [
            {
                "rank": 1,
                "cell_id": "cell-1",
                "cell_index": 1,
                "cell_type": "a",
                "metadata": {"cell_id": "cell-1", "cell_type": "a", "tissue": "liver"},
            }
        ][:limit]

    def query_similar(self, query_vector, n_results=5, where=None):
        return self.query_by_metadata(where=where, limit=n_results)

    def get_by_cell_ids(self, cell_ids):
        return {}


class FakeStreamLLM:
    def stream_chat(self, **kwargs):
        yield "测试回答"


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
            obs={
                "cell_type": ["query", "a", "b", "c"],
                "tissue": ["seed", "liver", "liver", "heart"],
            },
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
        self.original_user_db_path = app_module.USER_DB_PATH
        self.original_engine_runtime_config = {
            "index_config": dict(app_module._ENGINE_RUNTIME_CONFIG["index_config"]),
            "omp_num_threads": app_module._ENGINE_RUNTIME_CONFIG["omp_num_threads"],
            "use_gpu": app_module._ENGINE_RUNTIME_CONFIG["use_gpu"],
        }

        app_module.DATA_DIR = self.data_dir
        app_module.INDEX_DIR = self.index_dir
        app_module.USER_DB_PATH = self.base_path / "users.sqlite3"
        app_module.DEFAULT_DATA_PATH = str(self.dataset_path)
        app_module._DATASET_CACHE.clear()
        app_module._INDEX_CACHE.clear()
        app_module._BENCHMARK_INDEX_CACHE.clear()
        app_module._USER_STORE = None
        app_module._USER_STORE_PATH = None
        app_module._ENGINE_RUNTIME_CONFIG.clear()
        app_module._ENGINE_RUNTIME_CONFIG.update(
            {
                "index_config": app_module.IndexConfig(backend="auto", index_type="flat", metric="l2").to_dict(),
                "omp_num_threads": 1,
                "use_gpu": False,
            }
        )

        self.app = app_module.create_app()
        self.app.config.update(TESTING=True)
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        app_module.DATA_DIR = self.original_data_dir
        app_module.INDEX_DIR = self.original_index_dir
        app_module.DEFAULT_DATA_PATH = self.original_default_data_path
        app_module.USER_DB_PATH = self.original_user_db_path
        app_module._DATASET_CACHE.clear()
        app_module._INDEX_CACHE.clear()
        app_module._BENCHMARK_INDEX_CACHE.clear()
        app_module._USER_STORE = None
        app_module._USER_STORE_PATH = None
        app_module._ENGINE_RUNTIME_CONFIG.clear()
        app_module._ENGINE_RUNTIME_CONFIG.update(self.original_engine_runtime_config)
        self.tmpdir.cleanup()

    def login_admin(self) -> dict:
        response = self.client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "Admin@123456"},
        )
        self.assertEqual(response.status_code, 200)
        return response.get_json()["user"]

    def register_user(self, username: str = "researcher") -> dict:
        response = self.client.post(
            "/api/auth/register",
            json={
                "username": username,
                "password": "User@123456",
                "full_name": "Research User",
                "email": f"{username}@example.com",
            },
        )
        self.assertEqual(response.status_code, 201)
        return response.get_json()["user"]

    def test_protected_page_redirects_to_login(self) -> None:
        response = self.client.get("/")

        self.assertEqual(response.status_code, 302)
        self.assertIn("/login", response.headers["Location"])

    def test_admin_login_returns_current_user(self) -> None:
        user = self.login_admin()

        self.assertEqual(user["username"], "admin")
        self.assertEqual(user["role"], "admin")

        response = self.client.get("/api/auth/me")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["authenticated"])
        self.assertEqual(payload["user"]["username"], "admin")

    def test_register_login_and_update_profile(self) -> None:
        user = self.register_user("researcher_01")

        self.assertEqual(user["role"], "user")
        self.assertTrue(user["is_active"])

        update_response = self.client.patch(
            "/api/profile",
            json={"full_name": "Updated User", "email": "updated@example.com"},
        )
        self.assertEqual(update_response.status_code, 200)
        updated = update_response.get_json()["user"]
        self.assertEqual(updated["full_name"], "Updated User")
        self.assertEqual(updated["email"], "updated@example.com")

    def test_non_admin_cannot_manage_users(self) -> None:
        self.register_user("normal_user")

        response = self.client.get("/api/users")

        self.assertEqual(response.status_code, 403)
        self.assertIn("管理员权限", response.get_json()["error"])

    def test_admin_can_create_update_reset_and_delete_user(self) -> None:
        self.login_admin()

        create_response = self.client.post(
            "/api/users",
            json={
                "username": "managed_user",
                "full_name": "Managed User",
                "email": "managed@example.com",
                "role": "user",
            },
        )
        self.assertEqual(create_response.status_code, 201)
        payload = create_response.get_json()
        user_id = payload["user"]["id"]
        self.assertEqual(payload["initial_password"], "Nankai@123")

        update_response = self.client.patch(
            f"/api/users/{user_id}",
            json={"role": "admin", "is_active": False},
        )
        self.assertEqual(update_response.status_code, 200)
        updated = update_response.get_json()["user"]
        self.assertEqual(updated["role"], "admin")
        self.assertFalse(updated["is_active"])

        reset_response = self.client.post(f"/api/users/{user_id}/reset-password", json={})
        self.assertEqual(reset_response.status_code, 200)
        self.assertEqual(reset_response.get_json()["initial_password"], "Nankai@123")

        delete_response = self.client.delete(f"/api/users/{user_id}")
        self.assertEqual(delete_response.status_code, 200)

    def test_health_reports_default_dataset_metadata(self) -> None:
        self.login_admin()

        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["dataset_id"], "tiny")
        self.assertEqual(payload["n_cells"], 4)
        self.assertEqual(payload["n_genes"], 3)
        self.assertEqual(payload["use_rep"], "X_pca")
        self.assertFalse(payload["ready"])

    def test_datasets_lists_available_h5ad_files(self) -> None:
        self.login_admin()

        response = self.client.get("/api/datasets")

        self.assertEqual(response.status_code, 200)
        datasets = response.get_json()["datasets"]
        self.assertEqual(len(datasets), 1)
        self.assertEqual(datasets[0]["id"], "tiny")
        self.assertEqual(datasets[0]["index_status"], "missing")

    def test_index_build_and_search_return_results(self) -> None:
        self.login_admin()

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
        self.assertIn("snapshot_id", search_payload)

    def test_search_snapshot_is_saved_and_can_be_deleted(self) -> None:
        self.login_admin()

        search_response = self.client.post(
            "/api/search",
            json={
                "dataset_id": "tiny",
                "cell_id": "cell-0",
                "k": 2,
                "index_backend": "numpy",
                "index_type": "brute",
                "filter_field": "cell_type",
                "filter_value": "b",
            },
        )
        self.assertEqual(search_response.status_code, 200)
        snapshot_id = search_response.get_json()["snapshot_id"]

        list_response = self.client.get("/api/profile/search-snapshots")
        self.assertEqual(list_response.status_code, 200)
        snapshots = list_response.get_json()["snapshots"]
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0]["id"], snapshot_id)
        self.assertEqual(snapshots[0]["dataset_id"], "tiny")
        self.assertEqual(snapshots[0]["filter_field"], "cell_type")
        self.assertEqual(snapshots[0]["filter_value"], "b")
        self.assertEqual(snapshots[0]["rerun_payload"]["cell_id"], "cell-0")

        delete_response = self.client.delete(f"/api/profile/search-snapshots/{snapshot_id}")
        self.assertEqual(delete_response.status_code, 200)

        empty_response = self.client.get("/api/profile/search-snapshots")
        self.assertEqual(empty_response.status_code, 200)
        self.assertEqual(empty_response.get_json()["snapshots"], [])

    def test_search_requires_cell_index_or_cell_id(self) -> None:
        self.login_admin()

        response = self.client.post("/api/search", json={"dataset_id": "tiny", "k": 2})

        self.assertEqual(response.status_code, 400)
        self.assertIn("cell_index or cell_id is required", response.get_json()["error"])

    def test_admin_can_update_engine_config_defaults(self) -> None:
        self.login_admin()

        update_response = self.client.patch(
            "/api/engine/config",
            json={
                "index_backend": "numpy",
                "index_type": "brute",
                "index_metric": "cosine",
                "omp_num_threads": 2,
                "use_gpu": False,
            },
        )

        self.assertEqual(update_response.status_code, 200)
        update_payload = update_response.get_json()
        self.assertEqual(update_payload["status"], "updated")
        self.assertEqual(update_payload["index_backend"], "numpy")
        self.assertEqual(update_payload["index_type"], "brute")
        self.assertEqual(update_payload["index_metric"], "cosine")
        self.assertEqual(update_payload["omp_num_threads"], 2)

        search_response = self.client.post(
            "/api/search",
            json={
                "dataset_id": "tiny",
                "cell_id": "cell-0",
                "k": 2,
            },
        )

        self.assertEqual(search_response.status_code, 200)
        search_payload = search_response.get_json()
        self.assertEqual(search_payload["index_backend"], "numpy")
        self.assertEqual(search_payload["index_type"], "brute")
        self.assertEqual(search_payload["index_metric"], "cosine")

    def test_search_filters_results_by_metadata_field(self) -> None:
        self.login_admin()

        response = self.client.post(
            "/api/search",
            json={
                "dataset_id": "tiny",
                "cell_id": "cell-0",
                "k": 2,
                "index_backend": "numpy",
                "index_type": "brute",
                "filter_field": "cell_type",
                "filter_value": "b",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["filtered"])
        self.assertEqual(payload["filter_field"], "cell_type")
        self.assertEqual(payload["filter_value"], "b")
        self.assertEqual(len(payload["results"]), 1)
        self.assertEqual(payload["results"][0]["cell_id"], "cell-2")
        self.assertEqual(payload["results"][0]["cell_type"], "b")

    def test_search_rejects_unknown_filter_field(self) -> None:
        self.login_admin()

        response = self.client.post(
            "/api/search",
            json={
                "dataset_id": "tiny",
                "cell_id": "cell-0",
                "k": 2,
                "index_backend": "numpy",
                "index_type": "brute",
                "filter_field": "missing_field",
                "filter_value": "x",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("filter_field not found", response.get_json()["error"])

    def test_cells_endpoint_paginates_cell_ids(self) -> None:
        self.login_admin()

        response = self.client.get("/api/cells?dataset_id=tiny&offset=1&limit=2")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["total"], 4)
        self.assertEqual(payload["next_offset"], 3)
        self.assertEqual([cell["cell_id"] for cell in payload["cells"]], ["cell-1", "cell-2"])

    def test_natural_language_cell_query_filters_metadata(self) -> None:
        self.login_admin()

        response = self.client.post(
            "/api/cells/query",
            json={
                "dataset_id": "tiny",
                "question": "查询 tissue 为 liver 的前 10 个细胞",
                "limit": 10,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["dataset_id"], "tiny")
        self.assertEqual(payload["count"], 2)
        self.assertEqual([row["cell_id"] for row in payload["results"]], ["cell-1", "cell-2"])
        self.assertEqual(payload["plan"]["conditions"][0]["field"], "tissue")
        self.assertEqual(payload["plan"]["conditions"][0]["value"], "liver")

    def test_chat_stream_regular_question_does_not_emit_query_progress(self) -> None:
        self.login_admin()

        with (
            patch.object(app_module, "is_chroma_available", return_value=True),
            patch.object(app_module, "get_or_create_store", return_value=FakeChatStore()),
            patch.object(app_module, "get_llm_client", return_value=FakeStreamLLM()),
        ):
            response = self.client.post(
                "/api/chat/stream",
                json={"dataset_id": "tiny", "question": "请解释一下这个数据集适合做什么分析"},
            )

        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertNotIn("[PROGRESS]", body)
        self.assertNotIn("[NL_QUERY]", body)
        self.assertIn("测试回答", body)

    def test_chat_stream_cell_query_emits_query_progress(self) -> None:
        self.login_admin()

        with (
            patch.object(app_module, "is_chroma_available", return_value=True),
            patch.object(app_module, "get_or_create_store", return_value=FakeChatStore()),
            patch.object(app_module, "get_llm_client", return_value=FakeStreamLLM()),
        ):
            response = self.client.post(
                "/api/chat/stream",
                json={"dataset_id": "tiny", "question": "查询 tissue 为 liver 的前 2 个细胞"},
            )

        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertIn("[PROGRESS] parse", body)
        self.assertIn("[PROGRESS] query", body)
        self.assertIn("[NL_QUERY]", body)
        self.assertIn('"mode": "metadata"', body)

    def test_umap_endpoint_returns_sampled_points(self) -> None:
        self.login_admin()

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
