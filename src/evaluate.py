"""
evaluate.py - Evaluation metrics calculation, benchmark aggregation, and publication-ready visualization.
"""

import os
import glob
import json
import logging
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import (
    roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score,
    mean_squared_error, mean_absolute_error, r2_score
)

from src.targets import (
    STABILITY_THRESHOLD_HOURS, TARGET_LABEL, TARGET_NAME, thalf_to_target
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Configure publication-quality plot style
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})


def compute_metrics(
    y_true_thalf: np.ndarray,
    y_pred_thalf: Optional[np.ndarray] = None,
    stability_threshold_hours: float = STABILITY_THRESHOLD_HOURS,
    y_pred_target: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Canonical metrics for peptide-HLA stability prediction.

    ``y_true_thalf`` is always half-lives in hours. Predictions are supplied either
    as hours (``y_pred_thalf``) or already on the canonical target scale
    (``y_pred_target``); every correlation and error metric is computed on the
    single canonical target ``log10(1 + thalf)`` (see ``src/targets.py``).
    Previously Spearman was computed on raw hours, Pearson on ``stability_score``
    and RMSE on ``stability_score``, so the three numbers in one results row were
    not on a common scale.

    Pass ``y_pred_target`` when predictions may fall below zero on the target
    scale. Routing those through hours would clip them at 0 (a negative target
    implies a negative half-life), collapsing distinct values into ties and
    silently changing rank metrics. Real models should use ``y_pred_thalf``, where
    that clipping is the physically correct behaviour.

    Spearman and the AUCs are unchanged by the transform (it is monotone); Pearson,
    RMSE, MAE and R^2 are now all on the target scale and therefore comparable
    across models.
    """
    if (y_pred_thalf is None) == (y_pred_target is None):
        raise ValueError("pass exactly one of y_pred_thalf or y_pred_target")

    y_true_thalf = np.asarray(y_true_thalf, dtype=float)
    y_true = thalf_to_target(y_true_thalf)
    y_pred = (
        np.asarray(y_pred_target, dtype=float)
        if y_pred_target is not None
        else thalf_to_target(np.asarray(y_pred_thalf, dtype=float))
    )

    y_true_binary = (y_true_thalf >= stability_threshold_hours).astype(int)

    # Correlations -- both on the canonical target
    sp_rho, sp_p = spearmanr(y_true, y_pred)
    pe_r, pe_p = pearsonr(y_true, y_pred)

    # Errors -- all on the canonical target
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Binary classification; the predicted target is a monotone score, so the AUCs
    # are invariant to the transform.
    has_pos_neg = (len(np.unique(y_true_binary)) > 1)
    if has_pos_neg:
        roc_auc = roc_auc_score(y_true_binary, y_pred)
        pr_auc = average_precision_score(y_true_binary, y_pred)
    else:
        roc_auc = 0.5
        pr_auc = 0.0

    return {
        "target": TARGET_NAME,
        "n_samples": int(len(y_true_thalf)),
        "spearman_rho": float(round(sp_rho, 4)),
        "spearman_pvalue": float(sp_p),
        "pearson_r": float(round(pe_r, 4)),
        "pearson_pvalue": float(pe_p),
        "rmse": float(round(rmse, 4)),
        "mae": float(round(mae, 4)),
        "r2_score": float(round(r2, 4)),
        "roc_auc": float(round(roc_auc, 4)),
        "pr_auc": float(round(pr_auc, 4)),
        "stability_threshold_hours": stability_threshold_hours
    }


def _spread_summary(vals: List[float]) -> Optional[Dict[str, float]]:
    """Median / mean / IQR summary of a list of per-allele statistics."""
    if not vals:
        return None
    arr = np.asarray(vals, dtype=float)
    return {
        "median": float(round(float(np.median(arr)), 4)),
        "mean": float(round(float(arr.mean()), 4)),
        "std": float(round(float(arr.std(ddof=1)), 4)) if len(arr) > 1 else 0.0,
        "iqr_low": float(round(float(np.percentile(arr, 25)), 4)),
        "iqr_high": float(round(float(np.percentile(arr, 75)), 4)),
        "min": float(round(float(arr.min()), 4)),
        "max": float(round(float(arr.max()), 4)),
    }


def compute_per_allele_metrics(
    alleles: np.ndarray,
    y_true_thalf: np.ndarray,
    y_pred_thalf: Optional[np.ndarray] = None,
    min_samples: int = 10,
    stability_threshold_hours: float = STABILITY_THRESHOLD_HOURS,
    y_pred_target: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Per-allele correlations, plus their median across alleles.

    Pooled Spearman on a held-out-allele test set is inflated by between-allele
    differences in mean stability: a model that learns only "allele X is generally
    unstable" scores respectably without having learned any peptide preference.
    Taking the median of the per-allele correlations removes that between-allele
    variance, and is the honest measure of pan-specific performance.

    Alleles with fewer than ``min_samples`` rows, or with no variance in either the
    ground truth or the prediction, are still listed but excluded from the median.
    """
    if (y_pred_thalf is None) == (y_pred_target is None):
        raise ValueError("pass exactly one of y_pred_thalf or y_pred_target")

    alleles = np.asarray(alleles)
    y_true_thalf = np.asarray(y_true_thalf, dtype=float)
    # See compute_metrics: y_pred_target avoids clipping negative targets to ties.
    y_pred_all = (
        np.asarray(y_pred_target, dtype=float)
        if y_pred_target is not None
        else thalf_to_target(np.asarray(y_pred_thalf, dtype=float))
    )

    per_allele: Dict[str, Any] = {}
    usable_rho: List[float] = []
    usable_r: List[float] = []

    for allele in sorted(set(alleles.tolist())):
        mask = (alleles == allele)
        n = int(mask.sum())
        yt = thalf_to_target(y_true_thalf[mask])
        yp = y_pred_all[mask]

        if (n < min_samples) or (np.std(yt) == 0) or (np.std(yp) == 0):
            per_allele[str(allele)] = {
                "n_samples": n,
                "spearman_rho": None,
                "pearson_r": None,
                "rmse": float(round(float(np.sqrt(mean_squared_error(yt, yp))), 4)) if n else None,
                "excluded_from_median": True,
            }
            continue

        rho, _ = spearmanr(yt, yp)
        r, _ = pearsonr(yt, yp)
        yb = (y_true_thalf[mask] >= stability_threshold_hours).astype(int)
        auc_val = (
            float(round(float(roc_auc_score(yb, yp)), 4))
            if len(np.unique(yb)) > 1 else None
        )

        per_allele[str(allele)] = {
            "n_samples": n,
            "spearman_rho": float(round(float(rho), 4)),
            "pearson_r": float(round(float(r), 4)),
            "rmse": float(round(float(np.sqrt(mean_squared_error(yt, yp))), 4)),
            "roc_auc": auc_val,
            "excluded_from_median": False,
        }
        usable_rho.append(float(rho))
        usable_r.append(float(r))

    return {
        "target": TARGET_NAME,
        "n_alleles_total": int(len(per_allele)),
        "n_alleles_scored": int(len(usable_rho)),
        "spearman_rho_across_alleles": _spread_summary(usable_rho),
        "pearson_r_across_alleles": _spread_summary(usable_r),
        "per_allele": per_allele,
    }


def plot_split_distributions(
    splits_summary_path: str = "data/splits/splits_summary.json",
    output_png: str = "figures/splits_distribution.png"
):
    """
    Plot sample count and label distribution across the 3 splits.
    """
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    with open(splits_summary_path, "r") as f:
        summary = json.load(f)
        
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    split_types = ["random", "unseen_peptides", "unseen_alleles"]
    split_titles = ["Random Split\n(Easy)", "Unseen Peptides\n(Harder)", "Unseen HLA Alleles\n(Hardest / Zero-Shot)"]
    
    palette = {"train": "#4C72B0", "val": "#55A868", "test": "#C44E52"}
    
    for idx, (st, title) in enumerate(zip(split_types, split_titles)):
        ax = axes[idx]
        data = summary[st]
        subsets = ["train", "val", "test"]
        samples = [data[s]["num_samples"] for s in subsets]
        stable_pcts = [data[s]["stable_pct"] for s in subsets]
        
        bars = ax.bar(subsets, samples, color=[palette[s] for s in subsets], alpha=0.85, edgecolor="black", width=0.55)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Number of Samples" if idx == 0 else "")
        ax.set_ylim(0, max(samples) * 1.18)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        
        # Add data labels
        for bar, count, pct in zip(bars, samples, stable_pcts):
            h = bar.get_height()
            ax.annotate(
                f"{count:,}\n({pct:.1f}% stab)",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=9, fontweight="semibold"
            )
            
    plt.suptitle("Dataset Partitioning & Stability Balance Across the Three Splits", fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()
    logger.info(f"Saved split distribution figure to: {output_png}")


def plot_allele_representation(
    cleaned_csv_path: str = "data/processed/cleaned_stability_data.csv",
    output_png: str = "figures/allele_representation.png",
    top_n: int = 25
):
    """
    Plot representation and stable binder fractions for top HLA alleles.
    """
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    df = pd.read_csv(cleaned_csv_path)
    
    allele_stats = df.groupby("allele").agg(
        total_count=("peptide", "count"),
        stable_count=("is_stable", "sum"),
        mean_thalf=("thalf_hours", "mean")
    ).reset_index()
    
    allele_stats["unstable_count"] = allele_stats["total_count"] - allele_stats["stable_count"]
    allele_stats = allele_stats.sort_values("total_count", ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    p1 = ax.bar(range(len(allele_stats)), allele_stats["stable_count"], label="Stable ($T_{1/2} \\geq 2h$)", color="#2ca02c", alpha=0.85, edgecolor="black")
    p2 = ax.bar(range(len(allele_stats)), allele_stats["unstable_count"], bottom=allele_stats["stable_count"], label="Unstable ($T_{1/2} < 2h$)", color="#d62728", alpha=0.75, edgecolor="black")
    
    ax.set_ylabel("Number of Peptide Measurements", fontweight="bold")
    ax.set_title(f"Allele Representation and Stability Distribution (Top {top_n} Alleles)", fontweight="bold", fontsize=14)
    ax.set_xticks(range(len(allele_stats)))
    ax.set_xticklabels(allele_stats["allele"], rotation=45, ha="right", fontsize=9)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()
    logger.info(f"Saved allele representation figure to: {output_png}")


def plot_benchmark_results(
    merged_df: pd.DataFrame,
    metrics: Dict[str, Any],
    output_png: str = "figures/benchmark_netmhcstabpan.png"
):
    """
    Generate publication-ready correlation and ROC/PRC figures for NetMHCstabpan benchmark.
    """
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    
    # Left: Correlation Scatter Plot
    ax1 = axes[0]
    sns.scatterplot(
        data=merged_df,
        x="thalf_hours",
        y="netmhc_thalf_hours",
        hue="allele" if "allele" in merged_df.columns else None,
        alpha=0.7,
        s=45,
        ax=ax1,
        palette="tab10"
    )
    
    # Diagonal reference line
    max_val = max(merged_df["thalf_hours"].max(), merged_df["netmhc_thalf_hours"].max())
    ax1.plot([0, max_val], [0, max_val], "k--", alpha=0.6, label="Ideal (x=y)")
    
    ax1.set_xlabel("Experimental Half-Life $T_{1/2}$ (hours)", fontweight="bold")
    ax1.set_ylabel("NetMHCstabpan Predicted $T_{1/2}$ (hours)", fontweight="bold")
    ax1.set_title("Experimental vs Predicted Stability", fontweight="bold")
    
    # Annotation box
    stat_text = (
        f"Spearman $\\rho$: {metrics.get('spearman_rho', 0):.3f}\n"
        f"Pearson $r$: {metrics.get('pearson_r', 0):.3f}\n"
        f"RMSE ({TARGET_LABEL}): {metrics.get('rmse', 0):.3f}\n"
        f"ROC-AUC: {metrics.get('roc_auc', 0):.3f}\n"
        f"N = {len(merged_df):,}"
    )
    ax1.text(
        0.05, 0.95, stat_text,
        transform=ax1.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="#999"),
        fontsize=10,
        fontweight="semibold"
    )
    ax1.grid(True, linestyle="--", alpha=0.4)
    if ax1.get_legend():
        ax1.legend(loc="lower right", fontsize=8)
        
    # Right: ROC Curve
    ax2 = axes[1]
    y_true_binary = (merged_df["thalf_hours"] >= metrics.get("stability_threshold_hours", 2.0)).astype(int)
    y_scores = merged_df["netmhc_score"] if "netmhc_score" in merged_df.columns else merged_df["netmhc_thalf_hours"]
    
    if len(np.unique(y_true_binary)) > 1:
        fpr, tpr, _ = roc_curve(y_true_binary, y_scores)
        roc_val = auc(fpr, tpr)
        ax2.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"NetMHCstabpan (AUC = {roc_val:.3f})")
    ax2.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.50)")
    
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    ax2.set_ylabel("True Positive Rate (Sensitivity)", fontweight="bold")
    ax2.set_title("ROC Curve: Stable Binder Identification ($T_{1/2} \\geq 2h$)", fontweight="bold")
    ax2.legend(loc="lower right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.4)
    
    plt.suptitle("Benchmark Reproduction: NetMHCstabpan-1.0 Performance", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()
    logger.info(f"Saved benchmark results figure to: {output_png}")


def aggregate_manual_web_outputs(
    raw_outputs_dir: str = "data/netmhcstabpan_benchmark/raw_outputs",
    test_csv_path: str = "data/splits/unseen_alleles/test.csv"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Search raw outputs directory for text/html files saved by teammate,
    parse them all, join with ground truth, and evaluate benchmark.
    """
    from src.netmhcstabpan_client import parse_netmhcstabpan_output, evaluate_benchmark
    
    test_df = pd.read_csv(test_csv_path)
    output_files = glob.glob(os.path.join(raw_outputs_dir, "*.*"))
    logger.info(f"Found {len(output_files)} manual output file(s) in {raw_outputs_dir}")
    
    dfs = []
    for f in output_files:
        try:
            with open(f, "r", encoding="utf-8", errors="ignore") as fp:
                content = fp.read()
            df_parsed = parse_netmhcstabpan_output(content)
            if not df_parsed.empty:
                dfs.append(df_parsed)
        except Exception as e:
            logger.warning(f"Could not parse {f}: {e}")
            
    if not dfs:
        logger.warning(f"No valid parsed NetMHCstabpan files found in {raw_outputs_dir}.")
        return pd.DataFrame(), {}
        
    all_preds = pd.concat(dfs, ignore_index=True)
    metrics = evaluate_benchmark(test_df, all_preds)
    return metrics.get("merged_dataframe", pd.DataFrame()), metrics


if __name__ == "__main__":
    plot_split_distributions()
    plot_allele_representation()
