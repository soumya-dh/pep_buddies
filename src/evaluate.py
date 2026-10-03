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
    y_pred_thalf: np.ndarray,
    y_true_score: Optional[np.ndarray] = None,
    y_pred_score: Optional[np.ndarray] = None,
    stability_threshold_hours: float = 2.0
) -> Dict[str, Any]:
    """
    Compute comprehensive immuno-oncology metrics.
    """
    y_true_thalf = np.asarray(y_true_thalf, dtype=float)
    y_pred_thalf = np.asarray(y_pred_thalf, dtype=float)
    
    if y_true_score is None:
        y_true_score = 1.0 / (1.0 + 5.0 / np.maximum(y_true_thalf, 1e-4))
    if y_pred_score is None:
        y_pred_score = 1.0 / (1.0 + 5.0 / np.maximum(y_pred_thalf, 1e-4))
        
    y_true_binary = (y_true_thalf >= stability_threshold_hours).astype(int)
    
    # Correlations
    sp_rho, sp_p = spearmanr(y_true_thalf, y_pred_thalf)
    pe_r, pe_p = pearsonr(y_true_score, y_pred_score)
    
    # Errors on score
    rmse = np.sqrt(mean_squared_error(y_true_score, y_pred_score))
    mae = mean_absolute_error(y_true_score, y_pred_score)
    r2 = r2_score(y_true_score, y_pred_score)
    
    # Binary classification
    has_pos_neg = (len(np.unique(y_true_binary)) > 1)
    if has_pos_neg:
        roc_auc = roc_auc_score_val = roc_auc_score(y_true_binary, y_pred_score)
        pr_auc = average_precision_score(y_true_binary, y_pred_score)
    else:
        roc_auc = 0.5
        pr_auc = 0.0
        
    return {
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
        f"RMSE (score): {metrics.get('rmse', metrics.get('rmse_stability_score', 0)):.3f}\n"
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
