"""
test_quantitative_metrics.py - Unit tests for quantitative metrics (AIR, MCS, HPO)
and blinded glioma prospective prediction lock.
"""

import os
import unittest
import numpy as np
import pandas as pd
import torch

from src.interpretability.quantitative_metrics import (
    compute_air,
    compute_mcs,
    compute_hpo,
    TARGET_ALLELE_SPECS,
    VALIDATED_POCKET_RESIDUES,
)
from src.prospective.glioma_lock import (
    predict_peptide,
    run_glioma_lock,
    BLINDED_CANDIDATE_PANEL,
)
from models.baseline_model import PanStabilityMLP


class TestQuantitativeMetrics(unittest.TestCase):

    def test_compute_air_uniform_baseline(self):
        """Uniform attributions should produce null chance AIR."""
        uniform_sens = np.ones(9)
        res = compute_air(uniform_sens, anchor_positions=[2, 9], peptide_len=9)
        self.assertAlmostEqual(res["air"], 2.0 / 9.0, places=3)
        self.assertAlmostEqual(res["null_air"], 2.0 / 9.0, places=3)
        self.assertEqual(res["fold_over_null"], 1.0)

    def test_compute_air_concentrated(self):
        """Attributions concentrated at anchors should pass."""
        sens = np.array([0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0])
        res = compute_air(sens, anchor_positions=[2, 9], peptide_len=9)
        self.assertTrue(res["air"] > 0.60)
        self.assertTrue(res["passed"])

    def test_compute_mcs_concordant(self):
        """Test MCS matches when top amino acids overlap preferred motifs."""
        mock_pred = np.zeros((9, 20))
        # For HLA-A*02:01, pos 2 (index 1) prefers L, pos 9 (index 8) prefers V
        from models.baseline_model import AMINO_ACIDS
        l_idx = AMINO_ACIDS.index("L")
        v_idx = AMINO_ACIDS.index("V")
        mock_pred[1, l_idx] = 1.5
        mock_pred[8, v_idx] = 2.0

        spec = TARGET_ALLELE_SPECS["HLA-A*02:01"]["preferred_motifs"]
        res = compute_mcs(mock_pred, spec, top_k=2)
        self.assertTrue(res["allele_concordant"])
        self.assertEqual(res["concordance_fraction"], 1.0)

    def test_compute_hpo_overlap(self):
        """Test HLA Pocket Overlap calculation with known pocket positions."""
        dummy_model = PanStabilityMLP(input_dim=860, hidden_dim=64, dropout=0.0)
        peps = ["GILGFVFTL"]
        pseudos = ["YFAMYGEKVAHTHVDTLYVRYHYYTWAVLAYTWY"]
        res = compute_hpo(dummy_model, peps, pseudos, validated_pocket_residues=VALIDATED_POCKET_RESIDUES, top_n=20)
        self.assertEqual(res["top_n"], 20)
        self.assertIn("hpo", res)
        self.assertIn("overlapping_residues", res)

    def test_glioma_panel_antigens_defined(self):
        """Verify all 11 blinded panel candidates are present."""
        self.assertEqual(len(BLINDED_CANDIDATE_PANEL), 11)
        ids = [entry["Antigen_ID"] for entry in BLINDED_CANDIDATE_PANEL]
        self.assertIn("GLIOMA-01", ids)
        self.assertIn("GLIOMA-01-WT", ids)
        self.assertIn("GLIOMA-02", ids)
        self.assertIn("GLIOMA-02-WT", ids)
        self.assertIn("GLIOMA-08", ids)

    def test_glioma_lock_execution_and_hash(self):
        """Verify execution of glioma prospective lock and hash verification."""
        lock_res = run_glioma_lock(output_csv="glioma_prospective_predictions.csv")
        self.assertTrue(os.path.exists("glioma_prospective_predictions.csv"))
        self.assertTrue(os.path.exists("glioma_prospective_predictions.csv.sha256"))
        self.assertEqual(len(lock_res["df"]), 11)
        self.assertEqual(len(lock_res["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
