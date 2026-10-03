"""
run_phase4.py - Main execution script for Phase 4: Brain Cancer Prospective Run & Diagnostics.

Pipeline:
1. Freezes the final model weights and generates SHA-256 hash manifest (Step 14).
2. Generates sliding windows (8 to 11-mers, mutant and normal) for brain cancer mutations (Step 15).
3. Computes prospective stability predictions across patient HLA alleles and saves
   timestamped, cryptographically locked prediction and checksum files (Step 16).
4. Executes mutation scan on prospective neoantigens, specifically quantifying the
   K -> M anchor rescue effect for Histone H3.3 K27M in HLA-A*02:01 (Step 17).
5. Conducts diagnostic failure case analysis on test errors and inspects attributions (Step 18).
6. Produces publication-ready figures and saves comprehensive structured reports.
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
import json
import logging
import pandas as pd
import torch

from models.baseline_model import PanStabilityMLP
from src.hla_database import HLADatabase
from src.prospective.prospective_runner import (
    freeze_model,
    run_prospective_predictions,
    analyze_k27m_neoantigen,
    analyze_model_failures,
)
from src.visualization.plot_interpretability import (
    plot_prospective_k27m_figure,
    plot_failure_explanations,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Starting Phase 4 Prospective Brain Cancer Pipeline on device: {device}")

    # 1. Step 14: Freeze the model and save SHA-256 hash manifest
    manifest = freeze_model(
        source_checkpoint_path="models/checkpoints/random/mlp_baseline.pt",
        frozen_dir="models/frozen"
    )
    logger.info(f"Model locked and frozen. Hash: {manifest['sha256_hash']}")

    # Load the frozen model
    frozen_weights = manifest["frozen_weights_path"]
    model = PanStabilityMLP(input_dim=860).to(device)
    model.load_state_dict(torch.load(frozen_weights, map_location=device))
    model.eval()

    hla_db = HLADatabase()
    os.makedirs("predictions", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    # 2. Steps 15 & 16: Prospective brain cancer antigen predictions & timestamped lock file
    logger.info("Running prospective predictions on brain cancer neoantigen candidates...")
    pred_df = run_prospective_predictions(
        model=model,
        hla_db=hla_db,
        output_dir="predictions",
        device=device
    )

    # 3. Step 17: In-depth H3.3 K27M Neoantigen Mutation Scan & Mechanistic Evaluation
    logger.info("Analyzing Histone H3.3 K27M mutation mechanism for HLA-A*02:01...")
    k27m_results = analyze_k27m_neoantigen(model, hla_db, device=device)
    logger.info(
        f"K27M Result: WT thalf={k27m_results['wt_predicted_thalf_hours']}h vs "
        f"Mutant thalf={k27m_results['mut_predicted_thalf_hours']}h "
        f"({k27m_results['fold_change_thalf']}x increase)"
    )

    plot_prospective_k27m_figure(pred_df, k27m_results, "figures/brain_cancer_prospective_k27m.png")

    # 4. Step 18: Failure case analysis on test set
    logger.info("Performing failure mode attribution diagnostics on test split...")
    test_df = pd.read_csv("data/splits/random/test.csv")
    failure_results = analyze_model_failures(model, test_df, device=device)
    logger.info(
        f"Failure Analysis: True Positives Anchor Weight={failure_results['mean_anchor_ratio_true_positives']*100:.1f}% vs "
        f"False Positives Anchor Weight={failure_results['mean_anchor_ratio_false_positives']*100:.1f}%"
    )

    plot_failure_explanations(failure_results, "figures/failure_case_explanations.png")

    # 5. Export structured reports
    prospective_report = {
        "frozen_model_manifest": manifest,
        "prospective_predictions_summary": {
            "n_evaluated_candidates": len(pred_df),
            "n_stable_binders": int(pred_df["is_stable_binder"].sum()),
            "stable_binder_rate_pct": round(float(pred_df["is_stable_binder"].mean() * 100), 2),
            "locked_predictions_file": "predictions/prospective_brain_cancer_predictions.csv",
            "sha256_checksum_file": "predictions/prospective_brain_cancer_predictions.sha256",
        },
        "h3_k27m_mechanistic_analysis": {
            k: v for k, v in k27m_results.items() if k != "p2_substitutions_top5"
        },
        "model_failure_analysis": {
            "mean_anchor_ratio_true_positives": failure_results["mean_anchor_ratio_true_positives"],
            "mean_anchor_ratio_false_positives": failure_results["mean_anchor_ratio_false_positives"],
            "mean_anchor_ratio_false_negatives": failure_results["mean_anchor_ratio_false_negatives"],
            "conclusion": failure_results["failure_analysis_conclusion"],
            "n_false_positives_analyzed": len(failure_results["false_positive_cases"]),
            "n_false_negatives_analyzed": len(failure_results["false_negative_cases"]),
        }
    }

    report_path = "reports/prospective_brain_cancer_report.json"
    with open(report_path, "w") as f:
        json.dump(prospective_report, f, indent=2)

    logger.info("✓ Phase 4 Brain Cancer Prospective Run completed successfully!")
    logger.info(f"  Report written to: {report_path}")
    logger.info("  Locked files: predictions/prospective_brain_cancer_predictions.csv, predictions/prospective_brain_cancer_predictions.sha256")
    logger.info("  Figures written to: figures/brain_cancer_prospective_k27m.png, figures/failure_case_explanations.png")


if __name__ == "__main__":
    main()
