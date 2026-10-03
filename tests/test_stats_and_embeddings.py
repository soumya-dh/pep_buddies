"""
test_stats_and_embeddings.py - Comprehensive unit tests for src/stats.py and src/embeddings/cache.py.
"""

import os
import tempfile
import unittest
import numpy as np

from src.stats import (
    _spearman,
    _pearson,
    _rmse,
    _roc_auc,
    median_per_allele_spearman,
    _resample_indices,
    _percentile_ci,
    bootstrap_metric,
    paired_delta,
    METRICS,
)
from src.embeddings.cache import EmbeddingCache, model_slug


class TestStats(unittest.TestCase):
    """Tests for statistical routines and bootstrap estimation in src/stats.py."""

    def setUp(self):
        np.random.seed(42)
        self.y_true = np.array([0.5, 1.2, 2.3, 3.1, 4.0, 5.5, 6.2, 7.8, 8.1, 9.4])
        # Positively correlated predictions
        self.y_pred = self.y_true + np.random.normal(0, 0.5, size=len(self.y_true))
        self.alleles = np.array(["A*02:01"] * 5 + ["A*24:02"] * 5)

    def test_basic_metrics(self):
        rho = _spearman(self.y_true, self.y_pred)
        r = _pearson(self.y_true, self.y_pred)
        rmse = _rmse(self.y_true, self.y_pred)
        auc = _roc_auc(self.y_true, self.y_pred)

        self.assertGreater(rho, 0.7)
        self.assertGreater(r, 0.7)
        self.assertLess(rmse, 1.5)
        self.assertGreater(auc, 0.7)

    def test_constant_array_handling(self):
        constant = np.ones(10)
        self.assertTrue(np.isnan(_spearman(constant, self.y_pred)))
        self.assertTrue(np.isnan(_pearson(constant, self.y_pred)))

    def test_median_per_allele_spearman(self):
        med_rho = median_per_allele_spearman(self.alleles, self.y_true, self.y_pred, min_samples=3)
        self.assertFalse(np.isnan(med_rho))
        self.assertGreater(med_rho, 0.5)

    def test_resample_indices_row(self):
        rng = np.random.default_rng(42)
        idx = _resample_indices(10, None, "row", rng)
        self.assertEqual(len(idx), 10)
        self.assertTrue(np.all((idx >= 0) & (idx < 10)))

    def test_resample_indices_allele(self):
        rng = np.random.default_rng(42)
        idx = _resample_indices(10, self.alleles, "allele", rng)
        self.assertEqual(len(idx), 10)
        # Resampled indices should retain cluster groupings
        resampled_alleles = self.alleles[idx]
        self.assertTrue(set(resampled_alleles).issubset(set(self.alleles)))

    def test_bootstrap_metric_row(self):
        res = bootstrap_metric(
            y_true_thalf=self.y_true,
            y_pred_thalf=self.y_pred,
            metric="spearman_rho",
            unit="row",
            n_boot=50,
            seed=42,
        )
        self.assertIn("point", res)
        self.assertIn("ci_lo", res)
        self.assertIn("ci_hi", res)
        self.assertLessEqual(res["ci_lo"], res["ci_hi"])

    def test_paired_delta(self):
        y_worse = np.random.permutation(self.y_pred)
        res = paired_delta(
            y_true_thalf=self.y_true,
            y_pred_a=self.y_pred,
            y_pred_b=y_worse,
            metric="spearman_rho",
            unit="row",
            n_boot=50,
            seed=42,
        )
        self.assertIn("delta", res)
        self.assertIn("ci_lo", res)
        self.assertIn("ci_hi", res)
        self.assertIn("p_delta_gt_0", res)
        self.assertGreaterEqual(res["p_delta_gt_0"], 0.0)
        self.assertLessEqual(res["p_delta_gt_0"], 1.0)


class TestEmbeddingCache(unittest.TestCase):
    """Tests for on-disk embedding cache in src/embeddings/cache.py."""

    def test_model_slug(self):
        self.assertEqual(model_slug("facebook/esm2_t6_8M_UR50D"), "facebook__esm2_t6_8M_UR50D")
        self.assertEqual(model_slug("ankh:base"), "ankh_base")

    def test_cache_write_and_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EmbeddingCache(
                model_key="test-model",
                kind="peptides",
                root=tmpdir,
            )
            self.assertFalse(cache.exists())

            # Create synthetic per-position embeddings (3 sequences, 9 positions, 16 dims)
            seqs = ["ALAKAAAAL", "GILGFVFTL", "NLVPMVATV"]
            dim = 16
            length = 9
            data = np.random.randn(len(seqs), length, dim).astype(np.float16)

            cache.write(seqs, data, extra_meta={"test": True})
            self.assertTrue(cache.exists())

            # Read back
            cache_read = EmbeddingCache(
                model_key="test-model",
                kind="peptides",
                root=tmpdir,
            )
            self.assertEqual(len(cache_read.index), 3)

            # Test lookup by sequence
            emb1 = cache_read.get(["GILGFVFTL"])
            self.assertEqual(emb1.shape, (1, 9, 16))
            np.testing.assert_allclose(emb1[0], data[1], rtol=1e-3, atol=1e-3)

            # Test batch lookup
            batch_emb = cache_read.get(["NLVPMVATV", "ALAKAAAAL"])
            self.assertEqual(batch_emb.shape, (2, 9, 16))
            np.testing.assert_allclose(batch_emb[0], data[2], rtol=1e-3, atol=1e-3)
            np.testing.assert_allclose(batch_emb[1], data[0], rtol=1e-3, atol=1e-3)

            # Test missing sequence raises KeyError
            with self.assertRaises(KeyError):
                cache_read.get(["NONEXISTENT"])


if __name__ == "__main__":
    unittest.main()
