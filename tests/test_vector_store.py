"""Tests for DataLoader gene helpers and ChromaDB vector-store enrichment."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np

from backend.data_reader import DataLoader
from backend.vector_store import CellVectorStore, is_chroma_available


class DataLoaderGeneTests(unittest.TestCase):
    def test_gene_names_prefer_feature_name_and_x_block_reads_raw_expression(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.h5ad"
            adata = ad.AnnData(
                X=np.array([[0, 3, 1], [5, 0, 2]], dtype=np.float32),
                var={"feature_name": ["GENE_A", "GENE_B", "GENE_C"]},
            )
            adata.var_names = ["ENSG_A", "ENSG_B", "ENSG_C"]
            adata.obs_names = ["cell-0", "cell-1"]
            adata.obsm["X_pca"] = np.array([[0, 0], [1, 1]], dtype=np.float32)
            adata.write_h5ad(path)

            loader = DataLoader(path)

            self.assertEqual(loader.get_gene_names(), ["GENE_A", "GENE_B", "GENE_C"])
            np.testing.assert_allclose(
                loader.get_X_block(0, 2),
                np.array([[0, 3, 1], [5, 0, 2]], dtype=np.float32),
            )


@unittest.skipUnless(is_chroma_available(), "chromadb is not installed")
class CellVectorStoreTopGenesTests(unittest.TestCase):
    def test_populate_from_loader_writes_top_gene_symbols(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            data_path = root / "tiny.h5ad"
            persist_dir = root / "chroma"
            adata = ad.AnnData(
                X=np.array([[0, 3, 1], [5, 0, 2]], dtype=np.float32),
                obs={"cell_type": ["alpha", "beta"]},
                var={"feature_name": ["GENE_A", "GENE_B", "GENE_C"]},
            )
            adata.var_names = ["ENSG_A", "ENSG_B", "ENSG_C"]
            adata.obs_names = ["cell-0", "cell-1"]
            adata.obsm["X_pca"] = np.array([[0, 0], [1, 1]], dtype=np.float32)
            adata.write_h5ad(data_path)

            loader = DataLoader(data_path)
            store = CellVectorStore("tiny_top_genes", persist_dir=persist_dir)
            try:
                store.populate_from_loader(loader, use_rep="X_pca", batch_size=2, force=True, top_genes=2)
                results = store.query_by_metadata(where={}, limit=2)
                info = store.get_collection_info()
            finally:
                try:
                    store.delete_collection()
                except Exception:
                    pass

            by_id = {row["cell_id"]: row for row in results}
            self.assertEqual(by_id["cell-0"]["top_genes"], "GENE_B,GENE_C")
            self.assertEqual(by_id["cell-1"]["top_genes"], "GENE_A,GENE_C")
            self.assertIn("Top genes: GENE_B,GENE_C", by_id["cell-0"]["document"])
            self.assertTrue(info["top_genes_available"])


if __name__ == "__main__":
    unittest.main()
