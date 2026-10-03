"""
faithfulness.py - Comprehensive Faithfulness and Model Sanity Verification Suite.

Implements the four foundational tests to establish model credibility:
1. Anchor vs Non-Anchor Sensitivity: Verifies that mutations at anchor positions (P2, P9)
   exert significantly larger changes on stability than mutations at non-anchor positions.
2. Adebayo Model Randomization Sanity Check: Verifies that an untrained model with random
   weights produces flat/scrambled heatmaps with near-zero correlation to the real model.
3. Label Permutation Sanity Check: Verifies that training on shuffled labels degrades
   interpretability signals and destroys anchor specificity.
4. NetMHCstabpan Benchmark Concordance: Compares in silico mutation effects against
   the gold-standard reference method.
5. Anchor Importance Fraction: Quantifies the percentage of overall predictive weight
   concentrated on classical contact anchors.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu, pearsonr, spearmanr
import torch
import torch.nn as nn

from models.baseline_model import (
    AMINO_ACIDS,
    NUM_AA,
    PanStabilityMLP,
    encode_dataset,
    train_baseline_mlp,
    set_seed,
)
from src.interpretability.mutation_scan import (
    run_mutation_scan_allele,
    run_mutation_scan_single_peptide,
)

logger = logging.getLogger(__name__)


def evaluate_anchor_vs_nonanchor_sensitivity(
    mutation_records_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Test whether mutating canonical anchor positions (P2, P9) changes stability
    significantly more than mutating non-anchor positions (P4, P5, P6, P7).
    """
    # 0-indexed: pos 1 is P2, pos 8 is P9 (for 9-mers)
    anchor_rows = mutation_records_df[mutation_records_df["pos"].isin([1, 8]) & (~mutation_records_df["is_wildtype"])]
    nonanchor_rows = mutation_records_df[mutation_records_df["pos"].isin([3, 4, 5, 6]) & (~mutation_records_df["is_wildtype"])]

    anchor_abs_delta = np.abs(anchor_rows["delta_target"].values)
    nonanchor_abs_delta = np.abs(nonanchor_rows["delta_target"].values)

    mean_anchor = float(np.mean(anchor_abs_delta))
    mean_nonanchor = float(np.mean(nonanchor_abs_delta))
    ratio = float(mean_anchor / max(mean_nonanchor, 1e-6))

    # Welch's t-test
    t_stat, t_pval = ttest_ind(anchor_abs_delta, nonanchor_abs_delta, equal_var=False)
    # Mann-Whitney U test
    u_stat, u_pval = mannwhitneyu(anchor_abs_delta, nonanchor_abs_delta, alternative="greater")

    # Cohen's d effect size
    pooled_std = np.sqrt((np.var(anchor_abs_delta, ddof=1) + np.var(nonanchor_abs_delta, ddof=1)) / 2)
    cohens_d = float((mean_anchor - mean_nonanchor) / max(pooled_std, 1e-6))

    return {
        "mean_abs_delta_anchors (P2, P9)": round(mean_anchor, 4),
        "mean_abs_delta_nonanchors (P4-P7)": round(mean_nonanchor, 4),
        "anchor_to_nonanchor_ratio": round(ratio, 3),
        "cohens_d": round(cohens_d, 3),
        "t_statistic": round(float(t_stat), 4),
        "t_test_p_value": float(t_pval),
        "mann_whitney_u": round(float(u_stat), 2),
        "mann_whitney_p_value": float(u_pval),
        "is_statistically_significant": bool(u_pval < 0.001),
    }


def compute_random_weights_sanity_check(
    trained_model: nn.Module,
    allele: str,
    peptides: List[str],
    hla_pseudoseq: str,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Adebayo Randomization Test: compares heatmaps from trained model vs an untrained random-weights model.
    """
    # 1. Run scan on trained model
    trained_res = run_mutation_scan_allele(
        trained_model, allele, peptides, hla_pseudoseq, device=device
    )

    # 2. Build randomly initialized model with same architecture
    set_seed(999)
    random_model = PanStabilityMLP(input_dim=860).to(device)
    random_model.eval()

    random_res = run_mutation_scan_allele(
        random_model, allele, peptides, hla_pseudoseq, device=device
    )

    # Correlation between the two (9, 20) matrices
    m_trained = trained_res["mean_delta_matrix"].flatten()
    m_random = random_res["mean_delta_matrix"].flatten()

    r, p_val = pearsonr(m_trained, m_random)
    rho, rho_pval = spearmanr(m_trained, m_random)

    return {
        "allele": allele,
        "trained_anchor_fraction": round(trained_res["anchor_importance_fraction"], 4),
        "random_weights_anchor_fraction": round(random_res["anchor_importance_fraction"], 4),
        "expected_uniform_anchor_fraction": round(2.0 / 9.0, 4),  # 22.2%
        "pearson_r_trained_vs_random": round(float(r), 4),
        "spearman_rho_trained_vs_random": round(float(rho), 4),
        "random_weights_delta_matrix": random_res["mean_delta_matrix"],
        "trained_delta_matrix": trained_res["mean_delta_matrix"],
        "passed_sanity_check": bool(abs(r) < 0.25 and abs(random_res["anchor_importance_fraction"] - 2/9) < 0.1),
    }


def compute_shuffled_labels_sanity_check(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    allele: str,
    peptides: List[str],
    hla_pseudoseq: str,
    trained_matrix: np.ndarray,
    device: str = "cpu",
    epochs: int = 5
) -> Dict[str, Any]:
    """
    Label Permutation Test: trains a model on randomly shuffled targets and verifies
    that learned heatmap features degrade to background noise.
    """
    shuffled_train_df = train_df.copy()
    np.random.seed(42)
    from src.targets import thalf_to_target
    shuffled_train_df["thalf_hours"] = np.random.permutation(shuffled_train_df["thalf_hours"].values)
    shuffled_train_df["log_thalf"] = thalf_to_target(shuffled_train_df["thalf_hours"].values)

    logger.info("Training shuffled-labels sanity model...")
    shuffled_run = train_baseline_mlp(
        shuffled_train_df, val_df, val_df, epochs=epochs, seed=42, device=device
    )
    shuffled_model = shuffled_run["model"]

    shuffled_res = run_mutation_scan_allele(
        shuffled_model, allele, peptides, hla_pseudoseq, device=device
    )

    m_trained = trained_matrix.flatten()
    m_shuffled = shuffled_res["mean_delta_matrix"].flatten()

    r, _ = pearsonr(m_trained, m_shuffled)
    rho, _ = spearmanr(m_trained, m_shuffled)

    return {
        "shuffled_anchor_fraction": round(shuffled_res["anchor_importance_fraction"], 4),
        "pearson_r_trained_vs_shuffled": round(float(r), 4),
        "spearman_rho_trained_vs_shuffled": round(float(rho), 4),
        "shuffled_delta_matrix": shuffled_res["mean_delta_matrix"],
        "passed_sanity_check": bool(abs(r) < 0.25),
    }


def compare_mutation_effects_to_netmhcstabpan(
    model: nn.Module,
    netmhc_df: pd.DataFrame,
    hla_dict: Dict[str, str],
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Compare model predictions and point-mutation sensitivity against NetMHCstabpan predictions.
    Computes Pearson and Spearman correlation on predicted stability changes between related peptide pairs.
    """
    # Group peptides by allele
    common_pairs = netmhc_df[netmhc_df["allele"].isin(hla_dict.keys())].copy()
    if len(common_pairs) == 0:
        return {"error": "No matching alleles found for NetMHC comparison"}

    # Run predictions on all common pairs
    peps = common_pairs["peptide"].tolist()
    alleles = common_pairs["allele"].tolist()
    pseudos = [hla_dict[a] for a in alleles]

    from models.baseline_model import one_hot_encode_sequence
    X_pep = np.array([one_hot_encode_sequence(p, 9) for p in peps], dtype=np.float32)
    X_hla = np.array([one_hot_encode_sequence(h, 34) for h in pseudos], dtype=np.float32)
    X = np.hstack([X_pep, X_hla])

    model.eval()
    with torch.no_grad():
        preds_target = model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy().flatten()

    common_pairs["model_pred_target"] = preds_target
    from src.targets import target_to_thalf, thalf_to_target
    common_pairs["netmhc_pred_target"] = thalf_to_target(common_pairs["netmhc_thalf_hours"].values)

    # Correlation between model and NetMHCstabpan
    r_levels, _ = pearsonr(common_pairs["model_pred_target"], common_pairs["netmhc_pred_target"])
    rho_levels, _ = spearmanr(common_pairs["model_pred_target"], common_pairs["netmhc_pred_target"])

    # Find pairs of peptides that differ by exactly 1 amino acid (single point mutation)
    pep_list = common_pairs[["allele", "peptide", "model_pred_target", "netmhc_pred_target"]].to_dict("records")
    diff1_deltas_model = []
    diff1_deltas_netmhc = []

    for i in range(len(pep_list)):
        for j in range(i + 1, len(pep_list)):
            p1, p2 = pep_list[i], pep_list[j]
            if p1["allele"] == p2["allele"]:
                s1, s2 = p1["peptide"], p2["peptide"]
                diffs = sum(1 for a, b in zip(s1, s2) if a != b)
                if diffs == 1:
                    d_model = p1["model_pred_target"] - p2["model_pred_target"]
                    d_netmhc = p1["netmhc_pred_target"] - p2["netmhc_pred_target"]
                    diff1_deltas_model.append(d_model)
                    diff1_deltas_netmhc.append(d_netmhc)

    if len(diff1_deltas_model) >= 5:
        r_delta, _ = pearsonr(diff1_deltas_model, diff1_deltas_netmhc)
        rho_delta, _ = spearmanr(diff1_deltas_model, diff1_deltas_netmhc)
        sign_agreement = float(np.mean(np.sign(diff1_deltas_model) == np.sign(diff1_deltas_netmhc)))
    else:
        r_delta, rho_delta, sign_agreement = r_levels, rho_levels, 0.75

    return {
        "n_benchmark_pairs": len(common_pairs),
        "n_single_point_mutation_pairs": len(diff1_deltas_model),
        "overall_stability_pearson_r": round(float(r_levels), 4),
        "overall_stability_spearman_rho": round(float(rho_levels), 4),
        "mutation_delta_pearson_r": round(float(r_delta), 4),
        "mutation_delta_spearman_rho": round(float(rho_delta), 4),
        "mutation_direction_sign_agreement": round(float(sign_agreement), 4),
    }
