from __future__ import annotations

import math
import os
import unittest

os.environ.setdefault("CELL_DATA_PATH", "data/missing_for_test.h5ad")

from backend.app import _json_safe, create_app  # noqa: E402


class FlaskAppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = create_app()
        self.client = self.app.test_client()

    def test_health_reports_not_ready_without_data_file(self) -> None:
        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 503)
        payload = response.get_json()
        self.assertFalse(payload["ready"])
        self.assertIn("error", payload)
        self.assertEqual(payload["data_path"], "data/missing_for_test.h5ad")

    def test_search_requires_cell_index(self) -> None:
        response = self.client.get("/api/search?k=5")

        self.assertEqual(response.status_code, 400)
        self.assertIn("cell_index is required", response.get_json()["error"])

    def test_search_rejects_invalid_k(self) -> None:
        response = self.client.get("/api/search?cell_index=1&k=0")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "k must be a positive integer")

    def test_search_reports_unready_service(self) -> None:
        response = self.client.get("/api/search?cell_index=1&k=5")

        self.assertEqual(response.status_code, 503)
        self.assertIn("数据文件不存在", response.get_json()["error"])

    def test_json_safe_converts_non_finite_float_to_none(self) -> None:
        payload = {"value": math.nan, "nested": [{"value": math.inf}]}

        self.assertEqual(_json_safe(payload), {"value": None, "nested": [{"value": None}]})


if __name__ == "__main__":
    unittest.main()
