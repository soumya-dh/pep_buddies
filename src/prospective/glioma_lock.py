"""
glioma_lock.py - Blinded prospective brain cancer prediction and cryptographic locking.

Runs frozen model inference on the blinded candidate antigen panel:
  - H3.3 K27M (10-mer and 9-mer, mutant vs wild-type)
  - IDH1 R132H (9-mer and 10-mer, mutant vs wild-type)
  - EGFRvIII novel junction (9-mer)
  - IL13Ralpha2 (9-mer)
  - Survivin BIRC5 (9-mer)
  - Poly-Aspartate negative control (9-mer)

Generates:
  - glioma_prospective_predictions.csv
  - glioma_prospective_predictions.csv.sha256
"""

import os
import hashlib
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from models.baseline_model import (
    PanStabilityMLP,
    one_hot_encode_sequence,
)
from src.hla_database import HLADatabase
from src.targets import target_to_thalf

logger = logging.getLogger(__name__)

BLINDED_CANDIDATE_PANEL = [
    {
        "Antigen_ID": "GLIOMA-01",
        "Gene_Mutation": "H3.3 K27M",
        "Peptide_Type": "Mutant 10-mer",
        "Sequence": "RMSAPATGGV",
        "Length": 10,
        "Target_HLA": "HLA-A*02:01",
    },
    {
        "Antigen_ID": "GLIOMA-01-WT",
        "Gene_Mutation": "H3.3 Wild-Type",
        "Peptide_Type": "Wild-Type 10-mer",
        "Sequence": "RKSAPATGGV",
        "Length": 10,
        "Target_HLA": "HLA-A*02:01",
    },
    {
        "Antigen_ID": "GLIOMA-02",
        "Gene_Mutation": "H3.3 K27M",
        "Peptide_Type": "Mutant 9-mer",
        "Sequence": "RMSAPATGG",
        "Length": 9,
        "Target_HLA": "HLA-A*02:01",
    },
    {
        "Antigen_ID": "GLIOMA-02-WT",
        "Gene_Mutation": "H3.3 Wild-Type",
        "Peptide_Type": "Wild-Type 9-mer",
        "Sequence": "RKSAPATGG",
        "Length": 9,
        "Target_HLA": "HLA-A*02:01",
    },
    {
        "Antigen_ID": "GLIOMA-03",
        "Gene_Mutation": "IDH1 R132H",
        "Peptide_Type": "Mutant 9-mer",
        "Sequence": "HAYGDQYRA",
        "Length": 9,
        "Target_HLA": "HLA-A*02:01",
    },
    {
        "Antigen_ID": "GLIOMA-03-WT",
        "Gene_Mutation": "IDH1 Wild-Type",
        "Peptide_Type": "Wild-Type 9-mer",
        "Sequence": "RAYGDQYRA",
        "Length": 9,
        "Target_HLA": "HLA-A*02:01",
    },
    {
        "Antigen_ID": "GLIOMA-04",
        "Gene_Mutation": "IDH1 R132H",
        "Peptide_Type": "Mutant 10-mer",
        "Sequence": "HHAYGDQYRA",
        "Length": 10,
        "Target_HLA": "HLA-A*02:01",
    },
    {
        "Antigen_ID": "GLIOMA-05",
        "Gene_Mutation": "EGFRvIII",
        "Peptide_Type": "Novel Junction 9-mer",
        "Sequence": "LEEKKGNYV",
        "Length": 9,
        "Target_HLA": "HLA-A*02:01",
    },
    {
        "Antigen_ID": "GLIOMA-06",
        "Gene_Mutation": "IL13Rα2",
        "Peptide_Type": "Overexpressed 9-mer",
        "Sequence": "WLPFGFILI",
        "Length": 9,
        "Target_HLA": "HLA-A*02:01",
    },
    {
        "Antigen_ID": "GLIOMA-07",
        "Gene_Mutation": "Survivin (BIRC5)",
        "Peptide_Type": "Overexpressed 9-mer",
        "Sequence": "LTLGEFLKL",
        "Length": 9,
        "Target_HLA": "HLA-A*02:01",
    },
    {
        "Antigen_ID": "GLIOMA-08",
        "Gene_Mutation": "Negative Control",
        "Peptide_Type": "Poly-Aspartate 9-mer",
        "Sequence": "DDDDDDDDD",
        "Length": 9,
        "Target_HLA": "HLA-A*02:01",
    },
]


def predict_peptide(
    model: nn.Module,
    peptide: str,
    hla_pseudoseq: str,
    device: str = "cpu"
) -> Tuple[float, float, str]:
    """
    Predict stability target and half-life for a peptide sequence.
    Handles length 9 directly. For length 10, scans structural bulge deletion
    alignments (positions 4 to 7) to identify the optimal 9-mer binding core.
    """
    model.eval()
    x_hla = one_hot_encode_sequence(hla_pseudoseq, max_len=34)

    if len(peptide) == 9:
        x_pep = one_hot_encode_sequence(peptide, max_len=9)
        feat = np.hstack([x_pep, x_hla]).reshape(1, -1)
        with torch.no_grad():
            t_pred = float(model(torch.tensor(feat, dtype=torch.float32).to(device)).item())
        t_pred = max(0.0, t_pred)
        return t_pred, float(target_to_thalf(t_pred)), peptide

    best_core, best_target, _ = find_best_core_for_10mer(model, peptide, hla_pseudoseq, device)
    return best_target, float(target_to_thalf(best_target)), best_core


def find_best_core_for_10mer(
    model: nn.Module,
    peptide: str,
    hla_pseudoseq: str,
    device: str = "cpu"
) -> Tuple[str, float, int]:
    """
    Find optimal 9-mer core from a 10-mer by scanning internal bulge positions (indices 3 to 7).
    Returns (best_core, best_target, best_del_idx).
    """
    model.eval()
    x_hla = one_hot_encode_sequence(hla_pseudoseq, max_len=34)
    best_target = -1.0
    best_core = peptide[:9]
    best_del_pos = 3

    for del_idx in range(3, 8):
        core = peptide[:del_idx] + peptide[del_idx + 1:]
        x_pep = one_hot_encode_sequence(core, max_len=9)
        feat = np.hstack([x_pep, x_hla]).reshape(1, -1)
        with torch.no_grad():
            t_pred = float(model(torch.tensor(feat, dtype=torch.float32).to(device)).item())
        t_pred = max(0.0, t_pred)
        if t_pred > best_target:
            best_target = t_pred
            best_core = core
            best_del_pos = del_idx

    return best_core, best_target, best_del_pos


def run_glioma_lock(
    frozen_model_path: str = "models/frozen/pan_stability_mlp_frozen.pt",
    output_csv: str = "glioma_prospective_predictions.csv",
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Execute blinded predictions and produce locked, cryptographically hashed CSV.
    """
    hla_db = HLADatabase()
    checkpoint = torch.load(frozen_model_path, map_location=device)
    model = PanStabilityMLP(input_dim=860, hidden_dim=256, dropout=0.2).to(device)
    model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
    model.eval()

    hla_pseudoseq = hla_db.get_pseudosequence("HLA-A*02:01")

    records = []
    for entry in BLINDED_CANDIDATE_PANEL:
        seq = entry["Sequence"]
        t_pred, thalf, core = predict_peptide(model, seq, hla_pseudoseq, device=device)

        # Truncation baseline (seq[:9])
        x_pep_trunc = one_hot_encode_sequence(seq[:9], max_len=9)
        x_hla = one_hot_encode_sequence(hla_pseudoseq, max_len=34)
        feat_trunc = np.hstack([x_pep_trunc, x_hla]).reshape(1, -1)
        with torch.no_grad():
            trunc_target = float(model(torch.tensor(feat_trunc, dtype=torch.float32).to(device)).item())
        trunc_target = max(0.0, trunc_target)
        trunc_thalf = float(target_to_thalf(trunc_target))

        rec = dict(entry)
        rec["Core_Aligned_9mer"] = core
        rec["Predicted_Target_log10"] = round(t_pred, 4)
        rec["Predicted_HalfLife_Hours"] = round(thalf, 3)
        rec["Truncation_Target_log10"] = round(trunc_target, 4)
        rec["Truncation_HalfLife_Hours"] = round(trunc_thalf, 3)
        records.append(rec)

    df_out = pd.DataFrame(records)
    # Sort by predicted half-life descending
    df_out = df_out.sort_values("Predicted_HalfLife_Hours", ascending=False).reset_index(drop=True)
    df_out["Rank"] = range(1, len(df_out) + 1)

    # Reorder columns
    cols = [
        "Rank", "Antigen_ID", "Gene_Mutation", "Peptide_Type", "Sequence",
        "Length", "Target_HLA", "Core_Aligned_9mer", "Predicted_Target_log10",
        "Predicted_HalfLife_Hours", "Truncation_Target_log10", "Truncation_HalfLife_Hours"
    ]
    df_out = df_out[cols]

    # Save to root and to predictions/ directory
    df_out.to_csv(output_csv, index=False)
    preds_dir_copy = os.path.join("predictions", os.path.basename(output_csv))
    df_out.to_csv(preds_dir_copy, index=False)

    # Compute SHA-256 hash
    hasher = hashlib.sha256()
    with open(output_csv, "rb") as f:
        hasher.update(f.read())
    file_hash = hasher.hexdigest()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    hash_file_content = f"# Timestamp: {timestamp}\n{file_hash}  {os.path.basename(output_csv)}\n"

    sha_file = f"{output_csv}.sha256"
    with open(sha_file, "w") as f:
        f.write(hash_file_content)

    with open(f"predictions/{os.path.basename(sha_file)}", "w") as f:
        f.write(hash_file_content)

    logger.info(f"Locked prospective predictions written to {output_csv}")
    logger.info(f"SHA-256: {file_hash}")

    return {
        "output_csv": output_csv,
        "sha256": file_hash,
        "timestamp": timestamp,
        "df": df_out,
    }
