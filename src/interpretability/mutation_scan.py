"""
mutation_scan.py - In silico Deep Mutational Scanning (saturation mutagenesis).

For each peptide bound to an HLA allele, substitutes each position with each of
the 20 canonical amino acids and records the resulting change in predicted stability
(delta stability = S_mutant - S_wildtype).

Averaged across peptides for an allele, this generates an empirical Position x Amino Acid
sensitivity matrix and reveals the allele-specific binding motif (e.g. HLA-A*02:01 preference
for L/M at P2 and V/L at P9).
"""

import os
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from models.baseline_model import (
    AMINO_ACIDS,
    NUM_AA,
    AA_TO_IDX,
    one_hot_encode_sequence,
    encode_dataset,
)
from src.targets import target_to_thalf

logger = logging.getLogger(__name__)


def create_saturation_mutants(
    peptide: str,
    hla_pseudoseq: str,
    max_pep_len: int = 9,
    pseudo_len: int = 34
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Generate all single-point mutants for a peptide (L x 20 substitutions).

    Returns:
        X: Feature matrix of shape (L * 20, max_pep_len * 20 + pseudo_len * 20)
        mutant_records: Metadata for each substitution (pos, orig_aa, mut_aa, mutant_seq)
    """
    pep_len = min(len(peptide), max_pep_len)
    mutant_records = []
    mutant_peptides = []

    for pos in range(pep_len):
        orig_aa = peptide[pos]
        for mut_aa in AMINO_ACIDS:
            mut_pep = list(peptide)
            mut_pep[pos] = mut_aa
            mut_seq = "".join(mut_pep)
            mutant_peptides.append(mut_seq)
            mutant_records.append({
                "pos": pos,
                "orig_aa": orig_aa,
                "mut_aa": mut_aa,
                "is_wildtype": (orig_aa == mut_aa),
                "mutant_seq": mut_seq,
            })

    # Encode all mutants in bulk
    X_pep = np.array([one_hot_encode_sequence(p, max_pep_len) for p in mutant_peptides], dtype=np.float32)
    hla_encoded = one_hot_encode_sequence(hla_pseudoseq, pseudo_len)
    X_hla = np.tile(hla_encoded, (len(mutant_peptides), 1)).astype(np.float32)
    X = np.hstack([X_pep, X_hla])

    return X, mutant_records


def run_mutation_scan_single_peptide(
    model: nn.Module,
    peptide: str,
    hla_pseudoseq: str,
    device: str = "cpu"
) -> pd.DataFrame:
    """
    Run in silico saturation mutagenesis for a single peptide.
    Returns DataFrame with columns: [pos, orig_aa, mut_aa, is_wildtype, wt_pred, mut_pred, delta_target, delta_thalf]
    """
    model.eval()
    X, records = create_saturation_mutants(peptide, hla_pseudoseq)

    with torch.no_grad():
        inputs = torch.tensor(X, dtype=torch.float32).to(device)
        preds_target = model(inputs).cpu().numpy().flatten()

    df_mut = pd.DataFrame(records)
    df_mut["mut_pred_target"] = preds_target
    df_mut["mut_pred_thalf"] = target_to_thalf(preds_target)

    # Get wildtype prediction
    wt_rows = df_mut[df_mut["is_wildtype"]]
    wt_target = wt_rows["mut_pred_target"].iloc[0] if len(wt_rows) > 0 else df_mut["mut_pred_target"].iloc[0]
    wt_thalf = wt_rows["mut_pred_thalf"].iloc[0] if len(wt_rows) > 0 else df_mut["mut_pred_thalf"].iloc[0]

    df_mut["wt_pred_target"] = wt_target
    df_mut["wt_pred_thalf"] = wt_thalf
    df_mut["delta_target"] = df_mut["mut_pred_target"] - df_mut["wt_pred_target"]
    df_mut["delta_thalf"] = df_mut["mut_pred_thalf"] - df_mut["wt_pred_thalf"]

    return df_mut


def run_mutation_scan_allele(
    model: nn.Module,
    allele: str,
    peptides: List[str],
    hla_pseudoseq: str,
    max_peptides: int = 100,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Run mutation scan across multiple peptides for a given HLA allele.
    Aggregates the position x amino acid sensitivity matrix.

    Returns:
        Dict containing:
          - allele: HLA allele name
          - n_peptides: number of peptides evaluated
          - mean_delta_matrix: (9, 20) array of mean delta target stability
          - mean_pred_matrix: (9, 20) array of mean predicted target stability
          - position_sensitivity: (9,) array of mean absolute effect per position
          - anchor_importance_fraction: fraction of total sensitivity at P2 and P9
          - matrix_df: DataFrame version of the (9, 20) matrix
    """
    peps_to_eval = peptides[:max_peptides]
    if not peps_to_eval:
        raise ValueError(f"No peptides supplied for allele {allele}")

    logger.info(f"Running mutation scan for {allele} over {len(peps_to_eval)} peptides...")

    all_dfs = []
    for pep in peps_to_eval:
        df_p = run_mutation_scan_single_peptide(model, pep, hla_pseudoseq, device=device)
        df_p["peptide"] = pep
        all_dfs.append(df_p)

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Build (9, 20) matrix: rows are positions (0..8, P1..P9), cols are amino acids
    pivot_delta = combined_df.pivot_table(
        index="pos",
        columns="mut_aa",
        values="delta_target",
        aggfunc="mean"
    ).reindex(columns=list(AMINO_ACIDS), fill_value=0.0)

    pivot_pred = combined_df.pivot_table(
        index="pos",
        columns="mut_aa",
        values="mut_pred_target",
        aggfunc="mean"
    ).reindex(columns=list(AMINO_ACIDS), fill_value=0.0)

    # Position sensitivity: average absolute delta per position
    pos_sens = combined_df.groupby("pos")["delta_target"].apply(lambda s: np.mean(np.abs(s))).values

    # Anchor importance fraction (P2 is index 1, P9 is index 8)
    p2_idx, p9_idx = 1, min(8, len(pos_sens) - 1)
    total_sens = np.sum(pos_sens)
    anchor_sens = (pos_sens[p2_idx] + pos_sens[p9_idx]) if total_sens > 0 else 0.0
    anchor_fraction = float(anchor_sens / total_sens) if total_sens > 0 else 0.0

    return {
        "allele": allele,
        "n_peptides": len(peps_to_eval),
        "mean_delta_matrix": pivot_delta.values,
        "mean_pred_matrix": pivot_pred.values,
        "position_sensitivity": pos_sens.tolist(),
        "anchor_importance_fraction": anchor_fraction,
        "positions": [f"P{i+1}" for i in range(len(pos_sens))],
        "amino_acids": list(AMINO_ACIDS),
        "delta_df": pivot_delta,
        "pred_df": pivot_pred,
        "raw_records": combined_df,
    }
