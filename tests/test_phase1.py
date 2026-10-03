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


if __name__ == "__main__":
    unittest.main()
