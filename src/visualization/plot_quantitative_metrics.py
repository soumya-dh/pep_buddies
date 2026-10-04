"""
plot_quantitative_metrics.py - Generate publication-quality figures for the 3 quantitative metrics:
  Metric 1: Anchor Importance Ratio (AIR) across 6 target alleles
  Metric 2: Motif Concordance Score (MCS) biological validation
  Metric 3: HLA Pocket Overlap (HPO) with 19 crystallographic pocket residues
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
})


def plot_quantitative_metrics(
    report_path: str = "reports/quantitative_metrics_report.json",
    output_path: str = "figures/quantitative_metrics_validation.png"
):
    """
    Renders 3-panel figure validating the Quantitative Metric Specifications.
    """
    if not os.path.exists(report_path):
        raise FileNotFoundError(f"Report not found at {report_path}")

    with open(report_path, "r") as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(
        "Formal Quantitative Interpretability Metrics: AIR, MCS, and HPO",
        fontsize=14,
        fontweight="bold",
        y=0.98
    )

    # -------------------------------------------------------------
    # Panel A: Anchor Importance Ratio (AIR) across 6 alleles
    # -------------------------------------------------------------
    ax_air = axes[0]
    air_data = data["metric_1_air"]
    alleles = list(air_data.keys())
    model_airs = [air_data[a]["air_pct"] for a in alleles]
    null_airs = [air_data[a]["null_air_pct"] for a in alleles]

    x = np.arange(len(alleles))
    width = 0.35

    rects1 = ax_air.bar(x - width/2, model_airs, width, label="Model AIR", color="#1f77b4", edgecolor="black", alpha=0.85)
    rects2 = ax_air.bar(x + width/2, null_airs, width, label="Uniform Chance Null", color="#aec7e8", edgecolor="black", alpha=0.7)

    ax_air.axhline(40.0, color="#d62728", linestyle="--", linewidth=1.5, label="Target Threshold (40%)")
    ax_air.set_title("Metric 1: Anchor Importance Ratio (AIR)\nConcentration at Canonical Anchors", fontweight="bold")
    ax_air.set_ylabel("Anchor Share of Importance (%)")
    ax_air.set_xticks(x)
    ax_air.set_xticklabels([a.replace("HLA-", "") for a in alleles], rotation=25, ha="right")
    ax_air.set_ylim(0, 65)
    ax_air.legend(loc="upper left")

    for rect in rects1:
        h = rect.get_height()
        ax_air.text(rect.get_x() + rect.get_width()/2., h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # -------------------------------------------------------------
    # Panel B: Motif Concordance Score (MCS)
    # -------------------------------------------------------------
    ax_mcs = axes[1]
    mcs_data = data["metric_2_mcs"]

    # Gather rows for table-like heatmap visualization
    table_rows = []
    cell_text = []
    col_labels = ["Target Allele", "Anchor", "Model Top-2", "Known Motif", "Concordant?"]
    row_colors = []

    for a in alleles:
        evals = mcs_data[a]["anchor_evaluations"]
        a_clean = a.replace("HLA-", "")
        for ev in evals:
            top_str = "/".join(ev["top_model_aas"])
            known_str = "/".join(ev["target_preferred"] + ev["target_tolerated"])
            status = "MATCH" if ev["is_concordant"] else "MISS"
            cell_text.append([a_clean, ev["position"], top_str, known_str, status])
            row_colors.append("#d4edda" if ev["is_concordant"] else "#f8d7da")

    ax_mcs.axis("off")
    table = ax_mcs.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)

    # Style table header and rows
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#343a40")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor(row_colors[row - 1])
            if col == 4:
                cell.set_text_props(weight="bold", color="#155724" if "MATCH" in cell_text[row-1][4] else "#721c24")

    mcs_rate = data["summary"]["mcs_concordance_rate_pct"]
    ax_mcs.set_title(f"Metric 2: Motif Concordance Score (MCS)\nAllele Top-2 Concordance: {mcs_rate:.1f}% (Target ≥ 80%)", fontweight="bold")

    # -------------------------------------------------------------
    # Panel C: HLA Pocket Overlap (HPO) on HLA-A*02:01
    # -------------------------------------------------------------
    ax_hpo = axes[2]
    hpo_data = data["metric_3_hpo"]
    top20 = hpo_data["top_model_residues"]
    val_set = set(hpo_data["validated_pocket_residues"])

    overlaps = [pos in val_set for pos in top20]
    bar_colors = ["#2ca02c" if ol else "#7f7f7f" for ol in overlaps]

    y_pos = np.arange(len(top20))
    ax_hpo.barh(y_pos, [1] * len(top20), color=bar_colors, edgecolor="black", height=0.7)
    ax_hpo.set_yticks(y_pos)
    ax_hpo.set_yticklabels([f"Pos {p} {'(Pocket B/F)' if p in val_set else ''}" for p in top20], fontsize=7.5)
    ax_hpo.invert_yaxis()
    ax_hpo.set_xticks([])
    ax_hpo.set_xlim(0, 1.2)

    hpo_score = hpo_data["hpo_pct"]
    overlap_cnt = hpo_data["overlap_count"]
    ax_hpo.set_title(
        f"Metric 3: HLA Pocket Overlap (HPO)\n{overlap_cnt}/20 Overlap = {hpo_score:.1f}% (34-Input Null = 50.0%, 1.30x)",
        fontweight="bold"
    )

    # Custom legend for Panel C
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ca02c", edgecolor="black", label=f"Pocket B/F Match ({overlap_cnt}/20)"),
        Patch(facecolor="#7f7f7f", edgecolor="black", label="Other Groove Position"),
    ]
    ax_hpo.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved Quantitative Metrics figure to {output_path}")


if __name__ == "__main__":
    plot_quantitative_metrics()
