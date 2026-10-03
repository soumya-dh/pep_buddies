"""
test_phase3_phase4.py - Unit and integration tests for Interpretability (Phase 3)
and Brain Cancer Prospective Run (Phase 4).
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
import unittest
import numpy as np
import pandas as pd
import torch

from models.baseline_model import PanStabilityMLP
from src.hla_database import HLADatabase
from src.interpretability.mutation_scan import (
    create_saturation_mutants,
    run_mutation_scan_single_peptide,
    run_mutation_scan_allele,
)
from src.interpretability.gradients import (
    compute_peptide_position_attributions,
    cross_check_gradients_vs_mutations,
)
from src.interpretability.hla_masking import run_hla_residue_masking, get_pocket_annotations
from src.interpretability.faithfulness import (
    evaluate_anchor_vs_nonanchor_sensitivity,
    compute_random_weights_sanity_check,
)
from src.prospective.antigens import generate_brain_cancer_antigen_library
from src.prospective.prospective_runner import (
    freeze_model,
    predict_peptide_hla_pair,
    analyze_k27m_neoantigen,
    analyze_model_failures,
)


class TestPhase3Phase4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hla_db = HLADatabase()
        cls.pseudo_a0201 = cls.hla_db.get_pseudosequence("HLA-A*02:01")
        cls.model = PanStabilityMLP(input_dim=860)
        cls.checkpoint_path = "models/checkpoints/random/mlp_baseline.pt"
        if os.path.exists(cls.checkpoint_path):
            cls.model.load_state_dict(torch.load(cls.checkpoint_path, map_location="cpu"))
        cls.model.eval()

    def test_01_create_saturation_mutants(self):
        pep = "NLVPMVATV"
        X, records = create_saturation_mutants(pep, self.pseudo_a0201)
        # 9 positions * 20 amino acids = 180 mutants
        self.assertEqual(len(records), 180)
        self.assertEqual(X.shape, (180, 860))
        wt_count = sum(1 for r in records if r["is_wildtype"])
        self.assertEqual(wt_count, 9)

    def test_02_mutation_scan_single_peptide(self):
        pep = "NLVPMVATV"
        df = run_mutation_scan_single_peptide(self.model, pep, self.pseudo_a0201)
        self.assertEqual(len(df), 180)
        self.assertIn("delta_target", df.columns)
        self.assertIn("delta_thalf", df.columns)
        # For wildtype substitutions, delta should be exactly 0
        wt_deltas = df[df["is_wildtype"]]["delta_target"].values
        np.testing.assert_allclose(wt_deltas, 0.0, atol=1e-5)

    def test_03_mutation_scan_allele_aggregation(self):
        peps = ["NLVPMVATV", "GILGFVFTL", "CLGGLLTMV"]
        res = run_mutation_scan_allele(
            self.model, "HLA-A*02:01", peps, self.pseudo_a0201, max_peptides=3
        )
        self.assertEqual(res["mean_delta_matrix"].shape, (9, 20))
        self.assertEqual(len(res["position_sensitivity"]), 9)
        self.assertGreater(res["anchor_importance_fraction"], 0.0)
        self.assertLessEqual(res["anchor_importance_fraction"], 1.0)

    def test_04_captum_integrated_gradients(self):
        peps = ["NLVPMVATV", "GILGFVFTL"]
        signed, mag = compute_peptide_position_attributions(
            self.model, peps, self.pseudo_a0201
        )
        self.assertEqual(signed.shape, (2, 9))
        self.assertEqual(mag.shape, (2, 9))
        self.assertTrue(np.all(mag >= 0))

    def test_05_gradient_mutation_cross_check(self):
        mut_sens = np.array([0.1, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4])
        grad_imp = np.array([0.15, 0.48, 0.12, 0.09, 0.11, 0.08, 0.10, 0.11, 0.42])
        cc = cross_check_gradients_vs_mutations(mut_sens, grad_imp)
        self.assertGreater(cc["pearson_r"], 0.90)
        self.assertIn(2, cc["top3_mutation_positions"])
        self.assertIn(9, cc["top3_mutation_positions"])

    def test_06_hla_pocket_masking(self):
        peps = ["NLVPMVATV"]
        pseudos = [self.pseudo_a0201]
        res = run_hla_residue_masking(self.model, peps, pseudos, mode="zero")
        self.assertEqual(len(res["mean_importance"]), 34)
        self.assertIn("pocket_b_mean_importance", res)
        self.assertIn("pocket_f_mean_importance", res)
        self.assertGreater(res["anchor_pocket_ratio"], 0.0)

    def test_07_faithfulness_anchor_sensitivity(self):
        # Create synthetic records with high anchor sensitivity
        records = []
        for pos in range(9):
            for aa in "ACDEFGHIKLMNPQRSTVWY":
                is_wt = (aa == "A")
                delta = (np.random.normal(0.8, 0.1) if pos in [1, 8] else np.random.normal(0.2, 0.1))
                records.append({
                    "pos": pos,
                    "mut_aa": aa,
                    "is_wildtype": is_wt,
                    "delta_target": 0.0 if is_wt else delta,
                })
        df_records = pd.DataFrame(records)
        faith = evaluate_anchor_vs_nonanchor_sensitivity(df_records)
        self.assertTrue(faith["is_statistically_significant"])
        self.assertGreater(faith["anchor_to_nonanchor_ratio"], 1.5)

    def test_08_adebayo_random_weights_sanity_check(self):
        peps = ["NLVPMVATV", "GILGFVFTL"]
        sanity = compute_random_weights_sanity_check(
            self.model, "HLA-A*02:01", peps, self.pseudo_a0201
        )
        self.assertIn("pearson_r_trained_vs_random", sanity)
        # Random weights should yield near-zero correlation with trained model
        self.assertLess(abs(sanity["pearson_r_trained_vs_random"]), 0.5)

    def test_09_brain_cancer_antigen_library(self):
        df_antigens = generate_brain_cancer_antigen_library()
        self.assertGreater(len(df_antigens), 20)
        genes = set(df_antigens["gene"])
        self.assertTrue(any("H3F3A" in g for g in genes))
        self.assertIn("IDH1", genes)
        self.assertIn("EGFR", genes)
        self.assertIn("BRAF", genes)
        self.assertIn("TP53", genes)
        lengths = set(df_antigens["length"])
        self.assertTrue({8, 9, 10, 11}.issubset(lengths))

    def test_10_freeze_model_and_hash(self):
        if os.path.exists(self.checkpoint_path):
            manifest = freeze_model(
                source_checkpoint_path=self.checkpoint_path,
                frozen_dir="models/frozen"
            )
            self.assertTrue(os.path.exists("models/frozen/pan_stability_mlp_frozen.pt"))
            self.assertTrue(os.path.exists("models/frozen/model_manifest.json"))
            self.assertEqual(len(manifest["sha256_hash"]), 64)

    def test_11_k27m_neoantigen_mechanistic_analysis(self):
        res = analyze_k27m_neoantigen(self.model, self.hla_db)
        self.assertEqual(res["peptide_wt_9"], "RKSAPSTGG")
        self.assertEqual(res["peptide_mut_9"], "RMSAPSTGG")
        self.assertIn("wt_predicted_thalf_hours", res)
        self.assertIn("mut_predicted_thalf_hours", res)
        self.assertIn("biological_mechanism", res)


if __name__ == "__main__":
    unittest.main()
