"""
plot_model_comparison.py - Generate comparison bar chart of Ridge, MLP, and NetMHCstabpan.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

splits = ["Random Split\n(Easy)", "Unseen Peptides\n(Harder)", "Unseen HLA Alleles\n(Hardest / Zero-Shot)"]
x = np.arange(len(splits))
width = 0.25

# Performance metrics from baseline_results.json and NetMHCstabpan benchmark
ridge_rho = [0.591, 0.584, 0.094]
mlp_rho = [0.788, 0.754, 0.438]
netmhc_rho = [0.885, 0.860, 0.854]

ridge_auc = [0.806, 0.799, 0.591]
mlp_auc = [0.902, 0.882, 0.772]
netmhc_auc = [0.945, 0.938, 0.932]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Plot Spearman Rho
b1 = ax1.bar(x - width, ridge_rho, width, label="Ridge Baseline", color="#7f7f7f", alpha=0.85, edgecolor="black")
b2 = ax1.bar(x, mlp_rho, width, label="Pan-Specific MLP", color="#1f77b4", alpha=0.85, edgecolor="black")
b3 = ax1.bar(x + width, netmhc_rho, width, label="NetMHCstabpan (Benchmark)", color="#2ca02c", alpha=0.85, edgecolor="black")

ax1.set_ylabel(r"Spearman Rank Correlation ($\rho$)", fontweight="bold", fontsize=11)
ax1.set_title(r"Generalization Across Splits: Spearman $\rho$", fontweight="bold", fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(splits, fontweight="semibold", fontsize=10)
ax1.set_ylim(0, 1.05)
ax1.grid(axis="y", linestyle="--", alpha=0.4)
ax1.legend(loc="upper right", frameon=True)

for bars in [b1, b2, b3]:
    for b in bars:
        h = b.get_height()
        ax1.annotate(f"{h:.3f}", xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

# Plot ROC-AUC
c1 = ax2.bar(x - width, ridge_auc, width, label="Ridge Baseline", color="#7f7f7f", alpha=0.85, edgecolor="black")
c2 = ax2.bar(x, mlp_auc, width, label="Pan-Specific MLP", color="#1f77b4", alpha=0.85, edgecolor="black")
c3 = ax2.bar(x + width, netmhc_auc, width, label="NetMHCstabpan (Benchmark)", color="#2ca02c", alpha=0.85, edgecolor="black")

ax2.set_ylabel(r"ROC-AUC ($T_{1/2} \geq 2h$)", fontweight="bold", fontsize=11)
ax2.set_title(r"Binary Classification: ROC-AUC ($T_{1/2} \geq 2h$)", fontweight="bold", fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(splits, fontweight="semibold", fontsize=10)
ax2.set_ylim(0.45, 1.05)
ax2.grid(axis="y", linestyle="--", alpha=0.4)
ax2.legend(loc="lower left", frameon=True)

for bars in [c1, c2, c3]:
    for b in bars:
        h = b.get_height()
        ax2.annotate(f"{h:.3f}", xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

plt.suptitle("Model Benchmark vs Baselines Across Split Difficulty Regimes", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/model_comparison.png", dpi=300, bbox_inches="tight")
print("Saved figures/model_comparison.png successfully!")
