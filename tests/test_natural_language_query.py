from __future__ import annotations

import unittest
from types import SimpleNamespace

import pandas as pd

from backend.natural_language_query import (
    execute_natural_cell_query,
    parse_natural_cell_query,
)


class DummyLoader:
    def __init__(self) -> None:
        self._cells = [
            {"cell_id": "cell-0", "cell_type": "Hepatocyte", "tissue": "liver"},
            {"cell_id": "cell-1", "cell_type": "Kupffer cell", "tissue": "liver"},
            {"cell_id": "cell-2", "cell_type": "T cell", "tissue": "blood"},
        ]
        self.adata = SimpleNamespace(obs=pd.DataFrame(self._cells))
        self.obs_columns = ["cell_type", "tissue"]
        self.n_cells = len(self._cells)
        self.available_reps = ["X_pca"]

    def get_cell_info(self, idx: int):
        return dict(self._cells[idx])

    def cell_index_from_id(self, cell_id: str) -> int:
        for idx, cell in enumerate(self._cells):
            if cell["cell_id"] == cell_id:
                return idx
        raise KeyError(cell_id)


class DummyStore:
    def is_populated(self) -> bool:
        return True

    def query_by_keywords(self, keywords, n_results=5):
        return [
            {
                "rank": 1,
                "cell_id": "cell-0",
                "cell_index": 0,
                "cell_type": "Hepatocyte",
                "distance": 0.0,
                "top_genes": "ALB,APOA1",
                "metadata": {"cell_id": "cell-0", "cell_type": "Hepatocyte", "tissue": "liver"},
            }
        ][:n_results]


class NaturalLanguageQueryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.loader = DummyLoader()

    def test_parse_explicit_metadata_condition(self) -> None:
        plan = parse_natural_cell_query("查询 tissue 为 liver 的前 2 个细胞", self.loader)

        self.assertEqual(plan.limit, 2)
        self.assertEqual(plan.conditions[0].field, "tissue")
        self.assertEqual(plan.conditions[0].value, "liver")

    def test_execute_metadata_query(self) -> None:
        plan = parse_natural_cell_query("查询细胞类型为 Kupffer cell 的细胞", self.loader)
        result = execute_natural_cell_query(plan, self.loader)

        self.assertEqual(result["count"], 1)
        self.assertEqual(result["results"][0]["cell_id"], "cell-1")

    def test_execute_gene_keyword_query_uses_store(self) -> None:
        plan = parse_natural_cell_query("找 ALB 高表达的 Hepatocyte", self.loader)
        result = execute_natural_cell_query(plan, self.loader, store=DummyStore())

        self.assertEqual(result["count"], 1)
        self.assertEqual(result["results"][0]["cell_id"], "cell-0")
        self.assertIn("ALB", result["plan"]["gene_keywords"])


if __name__ == "__main__":
    unittest.main()
