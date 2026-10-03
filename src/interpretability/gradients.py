"""
gradients.py - Integrated Gradients feature attribution using Captum.

Cross-checks empirical mutation scan sensitivities with gradient-based attributions
computed along the straight-line path from a zero-baseline to the input:
    IG_i(x) = (x_i - x'_i) * integral_0^1 (dF(x' + alpha*(x - x')) / dx_i) d alpha

Aggregates attributions across one-hot amino acid dimensions to yield per-position
peptide attributions (P1..P9) and correlates them with empirical in silico
mutation sensitivities.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr, pearsonr

try:
    from captum.attr import IntegratedGradients
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False

from models.baseline_model import (
    AMINO_ACIDS,
    NUM_AA,
    one_hot_encode_sequence,
)

logger = logging.getLogger(__name__)


def compute_integrated_gradients_captum(
    model: nn.Module,
    X: np.ndarray,
    baseline: Optional[np.ndarray] = None,
    n_steps: int = 50,
    device: str = "cpu"
) -> np.ndarray:
    """
    Compute Integrated Gradients using Captum.
    Returns attributions array with shape matching X.
    """
    model.eval()
    model.to(device)

    tensor_X = torch.tensor(X, dtype=torch.float32, requires_grad=True).to(device)
    if baseline is None:
        tensor_baseline = torch.zeros_like(tensor_X).to(device)
    else:
        tensor_baseline = torch.tensor(baseline, dtype=torch.float32).to(device)

    if CAPTUM_AVAILABLE:
        ig = IntegratedGradients(model)
        attributions = ig.attribute(
            tensor_X,
            baselines=tensor_baseline,
            n_steps=n_steps,
            method="gausslegendre"
        )
        return attributions.detach().cpu().numpy()
    else:
        # Exact numerical Integrated Gradients fallback (Riemann sum approximation)
        logger.info("Captum not detected; running exact Riemann-sum Integrated Gradients...")
        alphas = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
        total_grads = torch.zeros_like(tensor_X)

        diff = tensor_X - tensor_baseline
        for alpha in alphas[1:]:
            interpolated = (tensor_baseline + alpha * diff).detach().requires_grad_(True)
            output = model(interpolated).sum()
            grad = torch.autograd.grad(output, interpolated)[0]
            total_grads += grad

        avg_grads = total_grads / n_steps
        attributions = diff * avg_grads
        return attributions.detach().cpu().numpy()


def compute_peptide_position_attributions(
    model: nn.Module,
    peptides: List[str],
    hla_pseudoseq: str,
    max_pep_len: int = 9,
    pseudo_len: int = 34,
    device: str = "cpu"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-position attribution vectors for a list of peptides against an HLA pseudo-sequence.

    Returns:
        pos_attributions: (N, max_pep_len) array of signed position attributions
        pos_importance: (N, max_pep_len) array of L1/magnitude position importance
    """
    X_pep = np.array([one_hot_encode_sequence(p, max_pep_len) for p in peptides], dtype=np.float32)
    hla_encoded = one_hot_encode_sequence(hla_pseudoseq, pseudo_len)
    X_hla = np.tile(hla_encoded, (len(peptides), 1)).astype(np.float32)
    X = np.hstack([X_pep, X_hla])

    # Attributions on full input (N, 860)
    raw_attr = compute_integrated_gradients_captum(model, X, device=device)

    # Peptide segment is the first (max_pep_len * NUM_AA) features
    pep_attr = raw_attr[:, :max_pep_len * NUM_AA].reshape(len(peptides), max_pep_len, NUM_AA)

    # Position attribution: sum over actual active one-hot features
    # For one-hot sequences, IG_i * x_i is the attribution of the residue present.
    pos_signed = pep_attr.sum(axis=2)
    pos_magnitude = np.abs(pep_attr).sum(axis=2)

    return pos_signed, pos_magnitude


def cross_check_gradients_vs_mutations(
    mutation_position_sensitivity: np.ndarray,
    gradient_position_importance: np.ndarray
) -> Dict[str, Any]:
    """
    Quantitatively cross-check Integrated Gradients attribution vs Mutation Scan sensitivity.

    Args:
        mutation_position_sensitivity: (9,) vector of mean empirical mutation sensitivities
        gradient_position_importance: (9,) vector of mean Integrated Gradients importance

    Returns:
        Dictionary with Pearson r, Spearman rho, and concordance statistics.
    """
    mut_sens = np.asarray(mutation_position_sensitivity, dtype=float)
    grad_imp = np.asarray(gradient_position_importance, dtype=float)

    pr, p_val_p = pearsonr(mut_sens, grad_imp)
    sr, p_val_s = spearmanr(mut_sens, grad_imp)

    # Top positions identified by both methods
    top_mut_pos = np.argsort(mut_sens)[::-1][:3]
    top_grad_pos = np.argsort(grad_imp)[::-1][:3]
    overlap_count = len(set(top_mut_pos).intersection(set(top_grad_pos)))

    return {
        "pearson_r": float(round(pr, 4)),
        "pearson_p_value": float(round(p_val_p, 6)),
        "spearman_rho": float(round(sr, 4)),
        "spearman_p_value": float(round(p_val_s, 6)),
        "top3_mutation_positions": [int(p + 1) for p in top_mut_pos],
        "top3_gradient_positions": [int(p + 1) for p in top_grad_pos],
        "top3_overlap": int(overlap_count),
        "mutation_sensitivity": [float(round(v, 4)) for v in mut_sens],
        "gradient_importance": [float(round(v, 4)) for v in grad_imp],
    }
