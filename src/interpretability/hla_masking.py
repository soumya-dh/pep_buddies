"""
hla_masking.py - HLA contact residue masking and in silico alanine mutagenesis.

Quantifies the functional dependence of stability predictions on specific HLA binding
groove residues by systematically masking or mutating each of the 34 Nielsen pocket
contact positions one at a time.

Maps contact positions to classical crystallographic pockets (Pocket B for P2 anchor,
Pocket F for P9 anchor) and benchmarks pocket-specific sensitivity.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from models.baseline_model import (
    AMINO_ACIDS,
    NUM_AA,
    one_hot_encode_sequence,
)
from src.hla_database import MHC_I_PSEUDO_POSITIONS_1BASED

logger = logging.getLogger(__name__)

# Key crystallographic binding pockets for HLA Class I heavy chain residues:
# Pocket B forms the primary P2 anchor specificity pocket.
POCKET_B_RESIDUES = {9, 45, 63, 66, 67, 70, 99}
# Pocket F forms the primary C-terminal (P9/P-omega) anchor specificity pocket.
POCKET_F_RESIDUES = {77, 80, 81, 84, 116, 118, 143, 147}


def get_pocket_annotations() -> List[Dict[str, Any]]:
    """Annotate the 34 Nielsen contact positions with functional pocket designations."""
    annotations = []
    for idx, pos in enumerate(MHC_I_PSEUDO_POSITIONS_1BASED):
        pocket = "Other"
        if pos in POCKET_B_RESIDUES:
            pocket = "Pocket B (P2 anchor)"
        elif pos in POCKET_F_RESIDUES:
            pocket = "Pocket F (P9 anchor)"

        annotations.append({
            "pseudo_idx": idx,
            "chain_pos": pos,
            "label": f"Pos{pos}",
            "pocket": pocket,
            "is_pocket_b": pos in POCKET_B_RESIDUES,
            "is_pocket_f": pos in POCKET_F_RESIDUES,
        })
    return annotations


def run_hla_residue_masking(
    model: nn.Module,
    peptides: List[str],
    hla_pseudoseqs: List[str],
    mode: str = "zero",  # "zero" (hide residue) or "alanine" (mutate to A)
    max_pep_len: int = 9,
    pseudo_len: int = 34,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Evaluate prediction sensitivity when masking each of the 34 HLA pocket residues.

    Args:
        model: Trained PanStabilityMLP model.
        peptides: List of peptide sequences.
        hla_pseudoseqs: List of corresponding 34-mer HLA pseudo-sequences.
        mode: "zero" masks residue to all zeros; "alanine" replaces residue with Alanine.

    Returns:
        Dict with per-residue importance array (34,), pocket aggregations, and detailed records.
    """
    model.eval()
    model.to(device)

    n_samples = len(peptides)
    if n_samples == 0:
        raise ValueError("peptides list cannot be empty")

    # Encode wildtype baseline inputs
    X_pep = np.array([one_hot_encode_sequence(p, max_pep_len) for p in peptides], dtype=np.float32)
    X_hla_orig = np.array([one_hot_encode_sequence(h, pseudo_len) for h in hla_pseudoseqs], dtype=np.float32)
    X_wt = np.hstack([X_pep, X_hla_orig])

    with torch.no_grad():
        wt_preds = model(torch.tensor(X_wt, dtype=torch.float32).to(device)).cpu().numpy().flatten()

    # Matrix of absolute prediction changes: (n_samples, 34)
    delta_matrix = np.zeros((n_samples, pseudo_len), dtype=np.float32)

    for j in range(pseudo_len):
        X_hla_perturbed = X_hla_orig.copy()
        start_idx = j * NUM_AA
        end_idx = (j + 1) * NUM_AA

        if mode == "zero":
            # Mask out the 20 one-hot channels for contact position j
            X_hla_perturbed[:, start_idx:end_idx] = 0.0
        elif mode == "alanine":
            # Replace with Alanine one-hot
            ala_idx = AMINO_ACIDS.index("A")
            X_hla_perturbed[:, start_idx:end_idx] = 0.0
            X_hla_perturbed[:, start_idx + ala_idx] = 1.0
        else:
            raise ValueError(f"Unknown masking mode: {mode}")

        X_perturbed = np.hstack([X_pep, X_hla_perturbed])
        with torch.no_grad():
            pert_preds = model(torch.tensor(X_perturbed, dtype=torch.float32).to(device)).cpu().numpy().flatten()

        delta_matrix[:, j] = np.abs(pert_preds - wt_preds)

    mean_importance_per_pos = delta_matrix.mean(axis=0)  # (34,)
    annotations = get_pocket_annotations()

    # Aggregate by pocket category
    b_indices = [a["pseudo_idx"] for a in annotations if a["is_pocket_b"]]
    f_indices = [a["pseudo_idx"] for a in annotations if a["is_pocket_f"]]
    other_indices = [a["pseudo_idx"] for a in annotations if not a["is_pocket_b"] and not a["is_pocket_f"]]

    pocket_b_mean = float(np.mean(mean_importance_per_pos[b_indices]))
    pocket_f_mean = float(np.mean(mean_importance_per_pos[f_indices]))
    pocket_other_mean = float(np.mean(mean_importance_per_pos[other_indices]))

    results_table = []
    for ann in annotations:
        idx = ann["pseudo_idx"]
        results_table.append({
            "pseudo_idx": idx,
            "chain_pos": ann["chain_pos"],
            "label": ann["label"],
            "pocket": ann["pocket"],
            "mean_delta": float(round(mean_importance_per_pos[idx], 4)),
        })

    return {
        "mode": mode,
        "n_samples": n_samples,
        "mean_importance": [float(round(v, 4)) for v in mean_importance_per_pos],
        "pocket_b_mean_importance": round(pocket_b_mean, 4),
        "pocket_f_mean_importance": round(pocket_f_mean, 4),
        "other_pocket_mean_importance": round(pocket_other_mean, 4),
        "anchor_pocket_ratio": round((pocket_b_mean + pocket_f_mean) / (2 * max(pocket_other_mean, 1e-6)), 3),
        "annotations": annotations,
        "per_position_table": results_table,
    }
