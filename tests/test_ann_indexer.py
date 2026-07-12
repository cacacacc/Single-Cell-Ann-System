"""Tests for ANNIndexer validation, search ordering, metrics and persistence."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from backend.ann_indexer import ANNIndexer, IndexConfig


class ANNIndexerTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(12345)
        self.vectors = rng.random((12, 6), dtype=np.float32)

    def test_build_and_search_returns_expected_shapes(self) -> None:
        indexer = ANNIndexer(dim=6)
        indexer.build_index(self.vectors)

        distances, indices = indexer.search(self.vectors[0], k=4)

        self.assertEqual(distances.shape, (4,))
        self.assertEqual(indices.shape, (4,))
        self.assertEqual(indices.dtype, np.int64)
        self.assertEqual(distances.dtype, np.float32)
        self.assertEqual(indices[0], 0)
        self.assertAlmostEqual(float(distances[0]), 0.0, places=6)
        self.assertTrue(np.all(np.diff(distances) >= 0))

    def test_query_vector_float64_and_2d_shape_are_accepted(self) -> None:
        indexer = ANNIndexer(dim=6)
        indexer.build_index(self.vectors)

        query = self.vectors[1].astype(np.float64).reshape(1, -1)
        distances, indices = indexer.search(query, k=3)

        self.assertEqual(indices[0], 1)
        self.assertEqual(distances.shape, (3,))
        self.assertEqual(indices.shape, (3,))

    def test_invalid_inputs_raise_clear_errors(self) -> None:
        indexer = ANNIndexer(dim=6)

        with self.assertRaises(RuntimeError):
            indexer.search(self.vectors[0], k=1)

        with self.assertRaises(ValueError):
            indexer.build_index(np.array([], dtype=np.float32))

        with self.assertRaises(ValueError):
            indexer.build_index(np.ones((2, 3, 4), dtype=np.float32))

        with self.assertRaises(ValueError):
            indexer.build_index(np.ones((3, 5), dtype=np.float32))

        indexer.build_index(self.vectors)

        with self.assertRaises(ValueError):
            indexer.search(self.vectors[0], k=0)

        with self.assertRaises(ValueError):
            indexer.search(self.vectors[0], k=-1)

        with self.assertRaises(TypeError):
            indexer.search(self.vectors[0], k=1.5)

        with self.assertRaises(ValueError):
            indexer.search(np.ones(5, dtype=np.float32), k=1)

        with self.assertRaises(ValueError):
            indexer.search(self.vectors[0], k=20)

    def test_save_and_load_round_trip(self) -> None:
        indexer = ANNIndexer(dim=6)
        indexer.build_index(self.vectors)

        expected_distances, expected_indices = indexer.search(self.vectors[2], k=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cell_index.index"
            indexer.save_index(path)

            loaded = ANNIndexer(dim=6)
            loaded.load_index(path)

            distances, indices = loaded.search(self.vectors[2], k=5)

            np.testing.assert_allclose(distances, expected_distances, rtol=0, atol=1e-6)
            np.testing.assert_array_equal(indices, expected_indices)

    def test_load_rejects_dimension_mismatch(self) -> None:
        indexer = ANNIndexer(dim=6)
        indexer.build_index(self.vectors)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cell_index.index"
            indexer.save_index(path)

            loaded = ANNIndexer(dim=5)
            with self.assertRaises(ValueError):
                loaded.load_index(path)

    def test_cosine_metric_numpy_backend(self) -> None:
        indexer = ANNIndexer(dim=6, config=IndexConfig(backend="numpy", metric="cosine"))
        indexer.build_index(self.vectors)

        distances, indices = indexer.search(self.vectors[0], k=3)

        self.assertEqual(indices[0], 0)
        self.assertAlmostEqual(float(distances[0]), 0.0, places=6)

    def test_load_rejects_metric_mismatch(self) -> None:
        indexer = ANNIndexer(dim=6, config=IndexConfig(metric="l2"))
        indexer.build_index(self.vectors)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cell_index.index"
            indexer.save_index(path)

            loaded = ANNIndexer(dim=6, config=IndexConfig(metric="cosine"))
            with self.assertRaises(ValueError):
                loaded.load_index(path)


if __name__ == "__main__":
    unittest.main()
