"""
test_phase1.py - Automated unit tests for Phase 1 components.
"""

import unittest
import numpy as np
import pandas as pd
from src.data_cleaning import normalize_allele_name, validate_peptide, stability_to_score, score_to_stability
from src.hla_database import HLADatabase, MHC_I_PSEUDO_POSITIONS_1BASED
from src.dataset_splits import make_random_split, make_unseen_peptides_split, make_unseen_alleles_split, verify_no_leakage
from src.netmhcstabpan_client import parse_netmhcstabpan_output
from scipy.stats import pearsonr, spearmanr
from src.targets import TARGET_NAME, get_target, target_to_thalf, thalf_to_target
from src.evaluate import compute_metrics, compute_per_allele_metrics
from src.head_to_head import load_head_to_head_index


class TestDataCleaning(unittest.TestCase):
    def test_normalize_allele_name(self):
        # IMGT format
        self.assertEqual(normalize_allele_name("A0201", "imgt"), "HLA-A*02:01")
        self.assertEqual(normalize_allele_name("HLA-A0201", "imgt"), "HLA-A*02:01")
        self.assertEqual(normalize_allele_name("A*02:01", "imgt"), "HLA-A*02:01")
        self.assertEqual(normalize_allele_name("HLA-B*07:02", "imgt"), "HLA-B*07:02")
        self.assertEqual(normalize_allele_name("B0702", "imgt"), "HLA-B*07:02")
        self.assertEqual(normalize_allele_name("HLA-C*07:01", "imgt"), "HLA-C*07:01")
        
        # NetMHC format
        self.assertEqual(normalize_allele_name("HLA-A*02:01", "netmhc"), "HLA-A02:01")
        self.assertEqual(normalize_allele_name("A0201", "netmhc"), "HLA-A02:01")
        
        # Short format
        self.assertEqual(normalize_allele_name("HLA-A*02:01", "short"), "A0201")
        self.assertEqual(normalize_allele_name("HLA-B*07:02", "short"), "B0702")

    def test_validate_peptide(self):
        ok, res = validate_peptide("SIINFEKL")
        self.assertTrue(ok)
        self.assertEqual(res, "SIINFEKL")
        
        # Lowercase should be uppercased
        ok, res = validate_peptide("siinfekl")
        self.assertTrue(ok)
        self.assertEqual(res, "SIINFEKL")
        
        # Non-canonical amino acid
        ok, res = validate_peptide("SIINXFKL")
        self.assertFalse(ok)
        
        # Too short
        ok, res = validate_peptide("SIINF", min_len=8)
        self.assertFalse(ok)

    def test_stability_score_conversion(self):
        # S = 1 / (1 + 5/thalf)
        # thalf = 5 -> S = 0.5
        self.assertAlmostEqual(stability_to_score(5.0, t0=5.0), 0.5, places=4)
        self.assertAlmostEqual(score_to_stability(0.5, t0=5.0), 5.0, places=4)
        
        # thalf = 0 -> S = 0.0
        self.assertEqual(stability_to_score(0.0), 0.0)


class TestHLADatabase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = HLADatabase()

    def test_contact_positions_count(self):
        self.assertEqual(len(MHC_I_PSEUDO_POSITIONS_1BASED), 34)

    def test_a0201_pseudosequence(self):
        expected = "YFAMYGEKVAHTHVDTLYVRYHYYTWAVLAYTWY"
        actual = self.db.get_pseudosequence("HLA-A*02:01")
        self.assertEqual(actual, expected)

    def test_mutated_allele(self):
        pseq_c67s = self.db.get_pseudosequence("HLA-B*14:02(C67S)")
        self.assertIsNotNone(pseq_c67s)
        self.assertEqual(len(pseq_c67s), 34)
        # Check that index 8 is S (serine)
        self.assertEqual(pseq_c67s[8], "S")


class TestDatasetSplits(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 300
        alleles = [f"HLA-A*0{i}:01" for i in range(1, 11)] + [f"HLA-B*0{i}:01" for i in range(1, 11)]
        peptides = [f"PEPTIDE{i:03d}" for i in range(50)]
        self.df = pd.DataFrame({
            "allele": np.random.choice(alleles, n),
            "peptide": np.random.choice(peptides, n),
            "thalf_hours": np.random.uniform(0.1, 20.0, n)
        })

    def test_unseen_peptides_leakage(self):
        train, val, test = make_unseen_peptides_split(self.df)
        train_peps = set(train["peptide"])
        val_peps = set(val["peptide"])
        test_peps = set(test["peptide"])
        self.assertEqual(len(train_peps.intersection(test_peps)), 0)
        self.assertEqual(len(train_peps.intersection(val_peps)), 0)
        self.assertEqual(len(val_peps.intersection(test_peps)), 0)

    def test_unseen_alleles_leakage(self):
        train, val, test = make_unseen_alleles_split(self.df, test_allele_count=4, val_allele_count=4)
        train_alleles = set(train["allele"])
        val_alleles = set(val["allele"])
        test_alleles = set(test["allele"])
        self.assertEqual(len(train_alleles.intersection(test_alleles)), 0)
        self.assertEqual(len(train_alleles.intersection(val_alleles)), 0)
        self.assertEqual(len(val_alleles.intersection(test_alleles)), 0)


class TestNetMHCParser(unittest.TestCase):
    def test_parser(self):
        sample_output = """
-----------------------------------------------------------------------------------------------------
 pos      HLA         peptide         Identity       Pred     Thalf(h) %Rank_Stab BindLevel
-----------------------------------------------------------------------------------------------------
    0  HLA-A*02:01    VTTEVAFGL         PEPLIST      0.450       0.87       5.00
    0  HLA-A*02:01    FVRQCFNPM         PEPLIST      0.093       0.29      18.00
-----------------------------------------------------------------------------------------------------
"""
        df = parse_netmhcstabpan_output(sample_output)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]["peptide"], "VTTEVAFGL")
        self.assertAlmostEqual(df.iloc[0]["netmhc_thalf_hours"], 0.87)
        self.assertAlmostEqual(df.iloc[0]["netmhc_score"], 0.450)


class TestCanonicalTarget(unittest.TestCase):
    """Fix 2: one target, one metric convention."""

    def test_target_is_log10_1p(self):
        self.assertAlmostEqual(thalf_to_target(0.0), 0.0, places=9)
        self.assertAlmostEqual(thalf_to_target(9.0), 1.0, places=9)
        self.assertAlmostEqual(thalf_to_target(1.8), np.log10(2.8), places=9)

    def test_target_handles_zero_halflife(self):
        # thalf has a true minimum of 0.0 in this dataset, so log(thalf) would be
        # -inf; the 1+ offset must keep the transform finite.
        self.assertTrue(np.all(np.isfinite(thalf_to_target([0.0, 0.0, 256.7]))))

    def test_roundtrip(self):
        thalf = np.array([0.0, 0.1, 1.8, 42.0, 256.7])
        np.testing.assert_allclose(target_to_thalf(thalf_to_target(thalf)), thalf, atol=1e-9)

    def test_get_target_rejects_inconsistent_column(self):
        df = pd.DataFrame({"thalf_hours": [1.0, 2.0], "log_thalf": [0.0, 0.0]})
        with self.assertRaises(ValueError):
            get_target(df)

    def test_get_target_uses_stored_column(self):
        thalf = np.array([0.0, 1.8, 42.0])
        df = pd.DataFrame({"thalf_hours": thalf, "log_thalf": np.log10(1 + thalf)})
        np.testing.assert_allclose(get_target(df), np.log10(1 + thalf))

    def test_metrics_all_on_one_scale(self):
        rng = np.random.default_rng(0)
        y_true = rng.uniform(0, 50, size=200)
        y_pred = np.clip(y_true + rng.normal(0, 3, size=200), 0, None)
        m = compute_metrics(y_true, y_pred)

        self.assertEqual(m["target"], TARGET_NAME)
        # Pearson must now be computed on the target, not on stability_score.
        expected_r = pearsonr(thalf_to_target(y_true), thalf_to_target(y_pred))[0]
        self.assertAlmostEqual(m["pearson_r"], round(expected_r, 4), places=4)
        # RMSE must be on the target too.
        expected_rmse = np.sqrt(np.mean(
            (thalf_to_target(y_true) - thalf_to_target(y_pred)) ** 2
        ))
        self.assertAlmostEqual(m["rmse"], round(expected_rmse, 4), places=4)

    def test_spearman_invariant_to_transform(self):
        # The transform is monotone, so rank metrics must be unchanged -- this is
        # what lets the new numbers be compared against Phase 1's Spearman.
        rng = np.random.default_rng(1)
        y_true = rng.uniform(0, 50, size=200)
        y_pred = np.clip(y_true + rng.normal(0, 5, size=200), 0, None)
        m = compute_metrics(y_true, y_pred)
        self.assertAlmostEqual(
            m["spearman_rho"], round(spearmanr(y_true, y_pred)[0], 4), places=4
        )


class TestPerAlleleMetrics(unittest.TestCase):
    """Fix 3: per-allele correlation on held-out alleles."""

    def test_pooled_inflated_by_between_allele_offsets(self):
        # Two alleles. Within each, the prediction is anti-correlated with truth,
        # but the allele-level means are ordered correctly. Pooled Spearman is
        # therefore positive while every per-allele Spearman is negative.
        alleles = np.array(["A"] * 20 + ["B"] * 20)
        y_true = np.concatenate([np.linspace(0.1, 1.0, 20), np.linspace(10.0, 40.0, 20)])
        y_pred = np.concatenate([np.linspace(1.0, 0.1, 20), np.linspace(40.0, 10.0, 20)])

        pooled = compute_metrics(y_true, y_pred)["spearman_rho"]
        per = compute_per_allele_metrics(alleles, y_true, y_pred)

        self.assertGreater(pooled, 0.5)
        self.assertLess(per["spearman_rho_across_alleles"]["median"], 0.0)
        self.assertEqual(per["n_alleles_scored"], 2)

    def test_degenerate_allele_excluded_from_median(self):
        alleles = np.array(["A"] * 20 + ["B"] * 3)
        y_true = np.concatenate([np.linspace(0.1, 5.0, 20), np.array([1.0, 2.0, 3.0])])
        y_pred = y_true.copy()
        per = compute_per_allele_metrics(alleles, y_true, y_pred, min_samples=10)

        self.assertTrue(per["per_allele"]["B"]["excluded_from_median"])
        self.assertEqual(per["n_alleles_scored"], 1)
        self.assertEqual(per["n_alleles_total"], 2)


class TestTargetScaleScoring(unittest.TestCase):
    """compute_metrics must accept predictions already on the target scale."""

    def test_requires_exactly_one_prediction_argument(self):
        y = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            compute_metrics(y)
        with self.assertRaises(ValueError):
            compute_metrics(y, y, y_pred_target=thalf_to_target(y))

    def test_hours_and_target_paths_agree_when_non_negative(self):
        rng = np.random.default_rng(3)
        y_true = rng.uniform(0, 50, size=150)
        y_pred = np.clip(y_true + rng.normal(0, 4, size=150), 0, None)
        a = compute_metrics(y_true, y_pred)
        b = compute_metrics(y_true, y_pred_target=thalf_to_target(y_pred))
        self.assertAlmostEqual(a["spearman_rho"], b["spearman_rho"], places=6)
        self.assertAlmostEqual(a["rmse"], b["rmse"], places=6)

    def test_negative_targets_are_not_clipped_into_ties(self):
        # Routing negative targets through hours clips them to 0 and manufactures
        # ties, which silently changes rank metrics.
        y_true = np.linspace(0.5, 40.0, 60)
        centred = thalf_to_target(y_true) - thalf_to_target(y_true).mean()
        self.assertLess(centred.min(), 0.0)

        via_target = compute_metrics(y_true, y_pred_target=centred)
        via_hours = compute_metrics(y_true, target_to_thalf(centred))
        self.assertAlmostEqual(via_target["spearman_rho"], 1.0, places=6)
        self.assertLess(via_hours["spearman_rho"], via_target["spearman_rho"])

    def test_per_allele_rho_invariant_to_constant_allele_shift(self):
        # Shifting an allele's predictions by a constant cannot change its
        # within-allele ranking -- the core invariant of the calibration probe.
        alleles = np.array(["A"] * 25 + ["B"] * 25)
        y_true = np.concatenate([np.linspace(0.2, 8.0, 25), np.linspace(1.0, 30.0, 25)])
        y_pred_t = thalf_to_target(y_true) + np.array([0.3] * 25 + [-0.7] * 25)

        base = compute_per_allele_metrics(alleles, y_true, y_pred_target=y_pred_t)
        shifted = y_pred_t.copy()
        shifted[alleles == "A"] -= 5.0
        moved = compute_per_allele_metrics(alleles, y_true, y_pred_target=shifted)

        self.assertEqual(
            base["spearman_rho_across_alleles"]["median"],
            moved["spearman_rho_across_alleles"]["median"],
        )


class TestHeadToHeadSubset(unittest.TestCase):
    """Fix 1: NetMHCstabpan must be compared on a common evaluation subset."""

    def test_subset_matches_netmhc_coverage(self):
        index = load_head_to_head_index()
        preds = pd.read_csv("data/netmhcstabpan_benchmark/predictions.csv")

        # Every NetMHCstabpan prediction must join to ground truth exactly once.
        self.assertEqual(len(index), len(preds))
        self.assertFalse(index.duplicated(subset=["allele_join", "peptide"]).any())

    def test_subset_is_smaller_than_full_test_set(self):
        index = load_head_to_head_index()
        full = pd.read_csv("data/splits/unseen_alleles/test.csv")
        # The whole point of the fix: these are different evaluation sets.
        self.assertLess(len(index), len(full))

    def test_subset_pairs_exist_in_test_set(self):
        index = load_head_to_head_index()
        full = pd.read_csv("data/splits/unseen_alleles/test.csv")
        full_pairs = set(zip(full["allele"], full["peptide"]))
        for allele, peptide in zip(index["allele"], index["peptide"]):
            self.assertIn((allele, peptide), full_pairs)


if __name__ == "__main__":
    unittest.main()
