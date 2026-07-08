from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np

from backend.dataset_preprocessor import prepare_joint_dataset


class DatasetPreprocessorTests(unittest.TestCase):
    def test_prepare_joint_dataset_aligns_shared_genes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.h5ad"
            second = root / "second.h5ad"
            output = root / "joint.h5ad"
            report_path = root / "report.json"

            adata_a = ad.AnnData(
                X=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
                obs={"cell_type": ["a", "b"]},
            )
            adata_a.var_names = ["gene_a", "gene_b", "gene_c"]
            adata_a.obs_names = ["cell_1", "cell_2"]
            adata_a.write_h5ad(first)

            adata_b = ad.AnnData(
                X=np.array([[7, 8, 9]], dtype=np.float32),
                obs={"cell_type": ["c"]},
            )
            adata_b.var_names = ["gene_b", "gene_c", "gene_d"]
            adata_b.obs_names = ["cell_3"]
            adata_b.write_h5ad(second)

            report = prepare_joint_dataset(
                [first, second],
                output,
                dataset_ids=["ds_a", "ds_b"],
                report_path=report_path,
            )

            merged = ad.read_h5ad(output)
            self.assertEqual(list(merged.var_names), ["gene_b", "gene_c"])
            self.assertEqual(merged.shape, (3, 2))
            self.assertEqual(list(merged.obs["dataset_id"]), ["ds_a", "ds_a", "ds_b"])
            self.assertEqual(report.total_cells, 3)
            self.assertEqual(report.aligned_genes, 2)
            self.assertTrue(report_path.exists())

    def test_prepare_joint_dataset_outer_fills_missing_genes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.h5ad"
            second = root / "second.h5ad"
            output = root / "joint.h5ad"

            ad.AnnData(
                X=np.array([[1, 2]], dtype=np.float32),
                obs={"cell_type": ["a"]},
                var={"gene_symbol": ["gene_a", "gene_b"]},
            ).write_h5ad(first)
            a = ad.read_h5ad(first)
            a.var_names = ["gene_a", "gene_b"]
            a.write_h5ad(first)

            b = ad.AnnData(
                X=np.array([[3, 4]], dtype=np.float32),
                obs={"cell_type": ["b"]},
            )
            b.var_names = ["gene_b", "gene_c"]
            b.write_h5ad(second)

            prepare_joint_dataset([first, second], output, join="outer")

            merged = ad.read_h5ad(output)
            self.assertEqual(list(merged.var_names), ["gene_a", "gene_b", "gene_c"])
            self.assertEqual(merged.shape, (2, 3))
            np.testing.assert_allclose(merged.X[0], np.array([1, 2, 0], dtype=np.float32))
            np.testing.assert_allclose(merged.X[1], np.array([0, 3, 4], dtype=np.float32))

    def test_prepare_joint_dataset_casts_index_names_to_strings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.h5ad"
            second = root / "second.h5ad"
            output = root / "joint.h5ad"

            adata_a = ad.AnnData(X=np.array([[1, 2]], dtype=np.float32))
            adata_a.obs_names = [101]
            adata_a.var_names = [1, 2]
            adata_a.write_h5ad(first)

            adata_b = ad.AnnData(X=np.array([[3, 4]], dtype=np.float32))
            adata_b.obs_names = [202]
            adata_b.var_names = [2, 3]
            adata_b.write_h5ad(second)

            prepare_joint_dataset(
                [first, second],
                output,
                dataset_ids=["ds_a", "ds_b"],
                join="inner",
            )

            merged = ad.read_h5ad(output)
            self.assertEqual(list(merged.var_names), ["2"])
            self.assertEqual(list(merged.obs_names), ["ds_a:101", "ds_b:202"])

    def test_prepare_joint_dataset_sanitizes_object_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.h5ad"
            second = root / "second.h5ad"
            output = root / "joint.h5ad"

            adata_a = ad.AnnData(
                X=np.array([[1, 2]], dtype=np.float32),
                obs={"mixed_score": ["1.5"]},
            )
            adata_a.obs["mixed_score"] = adata_a.obs["mixed_score"].astype("category")
            adata_a.var_names = ["gene_a", "gene_b"]
            adata_a.write_h5ad(first)

            adata_b = ad.AnnData(
                X=np.array([[3, 4]], dtype=np.float32),
                obs={"mixed_score": ["2.0"]},
            )
            adata_b.obs["mixed_score"] = adata_b.obs["mixed_score"].astype("category")
            adata_b.var_names = ["gene_a", "gene_b"]
            adata_b.write_h5ad(second)

            prepare_joint_dataset(
                [first, second],
                output,
                dataset_ids=["ds_a", "ds_b"],
            )

            merged = ad.read_h5ad(output)
            self.assertEqual(list(merged.obs["mixed_score"].astype(str)), ["1.5", "2.0"])


if __name__ == "__main__":
    unittest.main()
