"""
hybrid_heads.py - Hybrid architecture combining One-Hot Peptide with ESM-2 HLA Pocket Embeddings.

Evaluates Phase 2A:
  - Peptide: 9-mer one-hot representation (preserves explicit anchor positions P2/P9).
  - HLA: ESM-2 35M mature G-domain 34 pocket contact residues (captures protein structural context).
  - Evaluated on all 3 splits: random, unseen_peptides, unseen_alleles.
  - Benchmarked against NetMHCstabpan's 0.756 median per-allele rho on the 320-pair subset.
  - Computes row and allele cluster bootstrap intervals.
  - Ablation: tests whether continuous PLM HLA embeddings reduce between-allele offset error.
"""

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

from models.baseline_model import one_hot_encode_sequence
from models.heads import load_split, targets_for, DEFAULT_ALPHAS
from src.calibration_probe import residual_decomposition
from src.embeddings.cache import EmbeddingCache
from src.evaluate import compute_metrics, compute_per_allele_metrics
from src.stats import bootstrap_metric
from src.targets import TARGET_NAME, target_to_thalf, thalf_to_target

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SPLITS = ("random", "unseen_peptides", "unseen_alleles")


class HybridFeatureBuilder:
    """Builds hybrid design matrices: One-hot peptide + ESM-2 HLA pocket."""

    def __init__(
        self,
        model_key: str = "esm2-35m",
        root: str = "data/embeddings",
        pool_pocket: bool = True,
    ):
        self.model_key = model_key
        self.pool_pocket = pool_pocket
        self.hla_cache = EmbeddingCache(model_key, "hla", root=root)

        contact = self.hla_cache.meta.get("contact_positions_0based")
        if not contact:
            raise ValueError(f"HLA cache for {model_key} has no contact_positions_0based")
        self.contact_positions: List[int] = list(contact)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        # 1. Peptide One-Hot (9 x 20 = 180 dims)
        peptides = df["peptide"].astype(str).tolist()
        pep_oh = np.array([
            one_hot_encode_sequence(p, max_len=9).reshape(-1)
            for p in peptides
        ], dtype=np.float32)

        # 2. HLA ESM-2 pocket positions (N, 34, dim)
        hla_seqs = df["hla_seq"].astype(str).tolist()
        pocket = self.hla_cache.get_positions(hla_seqs, self.contact_positions)

        if self.pool_pocket:
            hla_feat = pocket.mean(axis=1)  # (N, dim)
        else:
            hla_feat = pocket.reshape(pocket.shape[0], -1)  # (N, 34 * dim)

        return np.hstack([pep_oh, hla_feat]).astype(np.float32)

    def describe(self, n_features: int) -> Dict[str, Any]:
        return {
            "model_key": self.model_key,
            "architecture": "Hybrid (One-Hot Peptide + ESM-2 HLA Pocket)",
            "pool_pocket": self.pool_pocket,
            "n_features": int(n_features),
            "peptide_dim": 180,
            "hla_dim": int(n_features - 180),
            "n_contact_positions": len(self.contact_positions),
        }


def fit_and_evaluate_hybrid(
    model_key: str = "esm2-35m",
    split: str = "unseen_alleles",
    pool_pocket: bool = True,
    splits_dir: str = "data/splits",
    emb_root: str = "data/embeddings",
    netmhc_subset_csv: str = "data/netmhcstabpan_benchmark/predictions.csv",
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train hybrid model on one split and compute full evaluation metrics & ablations."""
    logger.info(f"=== Running Hybrid Model ({model_key}, pool_pocket={pool_pocket}) on {split} ===")
    train_df, val_df, test_df = load_split(splits_dir, split)
    fb = HybridFeatureBuilder(model_key, root=emb_root, pool_pocket=pool_pocket)

    X_tr = fb.transform(train_df)
    X_va = fb.transform(val_df)
    X_te = fb.transform(test_df)

    y_tr, y_va = targets_for(train_df), targets_for(val_df)
    y_true_thalf = test_df["thalf_hours"].values
    y_true_target = thalf_to_target(y_true_thalf)

    # Standardize features
    scaler = StandardScaler().fit(X_tr)
    Xt = scaler.transform(X_tr)
    Xv = scaler.transform(X_va)
    Xte = scaler.transform(X_te)

    # Sweep alpha on validation set using Spearman rho
    best_rho = -np.inf
    best_alpha = 1.0
    best_model = None
    for a in DEFAULT_ALPHAS:
        m = Ridge(alpha=a).fit(Xt, y_tr)
        pv = m.predict(Xv)
        rho = float(spearmanr(y_va, pv)[0]) if np.std(pv) > 0 else 0.0
        if rho > best_rho:
            best_rho = rho
            best_alpha = a
            best_model = m

    logger.info(f"Selected Ridge alpha={best_alpha} (val rho={best_rho:.4f})")

    y_pred_target = np.clip(best_model.predict(Xte), 0.0, None)
    y_pred_thalf = target_to_thalf(y_pred_target)

    # 1. Canonical evaluation metrics
    metrics = compute_metrics(y_true_thalf, y_pred_thalf)
    per_allele = compute_per_allele_metrics(test_df["allele"].values, y_true_thalf, y_pred_thalf)

    # 2. Row and Allele Bootstrap CIs for Spearman rho
    logger.info("Computing row and allele bootstrap confidence intervals...")
    row_boot = bootstrap_metric(
        y_true_thalf, y_pred_thalf, metric="spearman_rho",
        n_boot=n_bootstraps, seed=seed, unit="row"
    )
    allele_boot = bootstrap_metric(
        y_true_thalf, y_pred_thalf, alleles=test_df["allele"].values, metric="spearman_rho",
        n_boot=n_bootstraps, seed=seed, unit="allele"
    )

    # 3. Residual & Calibration Decomposition (Offset error ablation)
    cal_decomp = residual_decomposition(test_df["allele"].values, y_true_target, y_pred_target)

    # 4. NetMHCstabpan 320-pair common subset comparison (if available)
    subset_res = None
    if os.path.exists(netmhc_subset_csv):
        net_df = pd.read_csv(netmhc_subset_csv)
        col = "netmhc_thalf_hours" if "netmhc_thalf_hours" in net_df.columns else "pred_thalf_hours"
        # Match on (allele, peptide)
        merged = pd.merge(
            test_df[["allele", "peptide", "thalf_hours"]].assign(pred_thalf=y_pred_thalf),
            net_df[["allele", "peptide", col]].rename(columns={col: "netmhc_thalf"}),
            on=["allele", "peptide"],
            how="inner",
        )
        if len(merged) > 0:
            sub_true = merged["thalf_hours"].values
            sub_hybrid = merged["pred_thalf"].values
            sub_netmhc = merged["netmhc_thalf"].values
            sub_per_allele_hybrid = compute_per_allele_metrics(merged["allele"].values, sub_true, sub_hybrid)
            sub_per_allele_netmhc = compute_per_allele_metrics(merged["allele"].values, sub_true, sub_netmhc)
            
            subset_res = {
                "n_pairs": len(merged),
                "n_alleles": int(merged["allele"].nunique()),
                "hybrid_spearman_rho": float(round(spearmanr(sub_true, sub_hybrid)[0], 4)),
                "hybrid_median_per_allele_rho": (sub_per_allele_hybrid["spearman_rho_across_alleles"] or {}).get("median"),
                "netmhc_spearman_rho": float(round(spearmanr(sub_true, sub_netmhc)[0], 4)),
                "netmhc_median_per_allele_rho": (sub_per_allele_netmhc["spearman_rho_across_alleles"] or {}).get("median"),
            }
            logger.info(
                f"320-Subset comparison: Hybrid med-rho={subset_res['hybrid_median_per_allele_rho']} vs "
                f"NetMHCstabpan med-rho={subset_res['netmhc_median_per_allele_rho']}"
            )

    result = {
        "model_key": model_key,
        "split": split,
        "features": fb.describe(X_tr.shape[1]),
        "alpha": best_alpha,
        "val_spearman": round(best_rho, 4),
        "test_metrics": metrics,
        "per_allele_metrics": per_allele,
        "bootstrap": {
            "row_level_spearman_95ci": [row_boot["ci_lo"], row_boot["ci_hi"]],
            "allele_level_spearman_95ci": [allele_boot["ci_lo"], allele_boot["ci_hi"]],
        },
        "calibration_probe": {
            "mse_total": cal_decomp["mse_total"],
            "mse_between_allele": cal_decomp["mse_between_allele"],
            "mse_within_allele": cal_decomp["mse_within_allele"],
            "offset_fraction": cal_decomp["offset_fraction"],
            "bias_spread": cal_decomp["bias_spread"],
        },
        "netmhcstabpan_subset": subset_res,
    }

    # Save predictions
    out_dir = os.path.join("predictions", "hybrid", split)
    os.makedirs(out_dir, exist_ok=True)
    preds_df = test_df[["allele", "peptide", "thalf_hours"]].copy()
    preds_df["pred_target"] = y_pred_target
    preds_df["pred_thalf_hours"] = y_pred_thalf
    pool_str = "mean" if pool_pocket else "flat"
    csv_path = os.path.join(out_dir, f"{model_key}_hybrid_{pool_str}.csv")
    preds_df.to_csv(csv_path, index=False)
    result["predictions_csv"] = csv_path

    return result


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate Hybrid model (One-Hot + ESM-2 Pocket)")
    parser.add_argument("--model", default="esm2-35m", help="ESM-2 model key")
    parser.add_argument("--splits", nargs="+", default=["unseen_alleles", "unseen_peptides", "random"])
    parser.add_argument("--pool-pocket", action="store_true", default=True, help="Mean-pool the 34 pocket positions")
    args = parser.parse_args()

    all_results = []
    for split in args.splits:
        res = fit_and_evaluate_hybrid(
            model_key=args.model,
            split=split,
            pool_pocket=args.pool_pocket,
        )
        all_results.append(res)

    report_path = "reports/hybrid_model_results.json"
    os.makedirs("reports", exist_ok=True)
    with open(report_path, "w") as f:
        json.dump({"results": all_results}, f, indent=2)

    logger.info(f"Saved complete Hybrid Model Report to {report_path}")

    # Print summary table
    print("\n" + "="*80)
    print("HYBRID MODEL (ONE-HOT PEPTIDE + ESM-2 HLA POCKET) RESULTS SUMMARY")
    print("="*80)
    print(f"{'Split':<18} | {'Spearman rho':<12} | {'Med Allele rho':<14} | {'Row 95% CI':<18} | {'Offset Frac':<11}")
    print("-" * 80)
    for r in all_results:
        m = r["test_metrics"]
        med = (r["per_allele_metrics"]["spearman_rho_across_alleles"] or {}).get("median", 0.0)
        ci = f"[{r['bootstrap']['row_level_spearman_95ci'][0]:.3f}, {r['bootstrap']['row_level_spearman_95ci'][1]:.3f}]"
        offset = r["calibration_probe"]["offset_fraction"]
        print(f"{r['split']:<18} | {m['spearman_rho']:<12.4f} | {med:<14.4f} | {ci:<18} | {offset:<11.4f}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
