"""
run_air_auroc_experiment.py - Evaluate per-peptide AIR vs MC-dropout sigma for error prediction.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, pearsonr

from models.baseline_model import PanStabilityMLP, one_hot_encode_sequence, AMINO_ACIDS
from src.hla_database import HLADatabase
from src.targets import get_target, target_to_thalf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def evaluate_air_auroc(n_samples: int = 250, seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 1. Load test data
    test_path = "data/splits/unseen_peptides/test.csv"
    if not os.path.exists(test_path):
        test_path = "data/splits/random/test.csv"
    df = pd.read_csv(test_path)
    
    # Filter 9-mers with valid target
    df = df[df["peptide"].str.len() == 9].copy()
    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=seed).reset_index(drop=True)
    
    logger.info(f"Loaded {len(df)} test samples from {test_path}")
    
    # 2. Load frozen model
    hla_db = HLADatabase()
    model = PanStabilityMLP(input_dim=9*20 + 34*20, hidden_dim=256, dropout=0.20)
    ckpt = torch.load("models/frozen/pan_stability_mlp_frozen.pt", map_location="cpu")
    model.load_state_dict(ckpt)
    
    # Feature construction
    peptides = df["peptide"].tolist()
    alleles = df["allele"].tolist()
    y_true_target = get_target(df)
    
    # Pre-build features
    X_list = []
    for pep, al in zip(peptides, alleles):
        pseudo = hla_db.get_pseudosequence(al)
        pep_oh = one_hot_encode_sequence(pep, max_len=9).reshape(-1)
        hla_oh = one_hot_encode_sequence(pseudo, max_len=34).reshape(-1)
        X_list.append(np.concatenate([pep_oh, hla_oh]))
    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    
    # 3. MC Dropout Inference (30 passes)
    model.train() # enable dropout
    n_passes = 30
    mc_preds = []
    with torch.no_grad():
        for _ in range(n_passes):
            mc_preds.append(model(X).squeeze(-1).numpy())
    mc_preds = np.array(mc_preds) # (30, N)
    
    pred_mean = mc_preds.mean(axis=0)
    pred_sigma = mc_preds.std(axis=0)
    abs_error = np.abs(y_true_target - pred_mean)
    
    # 4. In Silico Saturation Mutagenesis per peptide to compute AIR
    model.eval()
    airs = []
    logger.info("Computing per-peptide in silico saturation mutagenesis & AIR...")
    
    for idx, (pep, al) in enumerate(zip(peptides, alleles)):
        pseudo = hla_db.get_pseudosequence(al)
        hla_oh = one_hot_encode_sequence(pseudo, max_len=34).reshape(-1)
        
        # Build 9 x 20 single point mutant features
        mutant_feats = []
        for pos in range(9):
            for aa in AMINO_ACIDS:
                mut_pep = list(pep)
                mut_pep[pos] = aa
                mut_oh = one_hot_encode_sequence("".join(mut_pep), max_len=9).reshape(-1)
                mutant_feats.append(np.concatenate([mut_oh, hla_oh]))
        
        mut_tensor = torch.tensor(np.array(mutant_feats), dtype=torch.float32)
        with torch.no_grad():
            mut_preds = model(mut_tensor).squeeze(-1).numpy().reshape(9, 20)
        
        # Base prediction for this sequence
        base_pred = pred_mean[idx]
        # Position sensitivity = mean absolute delta across substitutions
        pos_sens = np.mean(np.abs(mut_preds - base_pred), axis=1) # (9,)
        
        total_sens = float(np.sum(pos_sens))
        if total_sens > 1e-8:
            # P2 and P9 (indices 1 and 8)
            anchor_sens = float(pos_sens[1] + pos_sens[8])
            air = anchor_sens / total_sens
        else:
            air = 2.0 / 9.0
        airs.append(air)
    
    airs = np.array(airs)
    
    # 5. Correlation & AUROC analysis
    # Does MC dropout sigma correlate with absolute error?
    rho_sigma, p_sigma = spearmanr(pred_sigma, abs_error)
    # Does AIR correlate with absolute error? (hypothesis: higher AIR -> more focused on anchors -> lower error)
    rho_air, p_air = spearmanr(airs, abs_error)
    
    # Define high error label (top 25% error)
    err_threshold = np.percentile(abs_error, 75)
    y_high_error = (abs_error >= err_threshold).astype(int)
    
    # AUROC of sigma predicting high error (higher sigma -> higher error)
    auroc_sigma = roc_auc_score(y_high_error, pred_sigma)
    # AUROC of inverted AIR predicting high error (lower AIR -> higher error)
    auroc_inv_air = roc_auc_score(y_high_error, -airs)
    # Combined predictor (z-scored)
    z_sigma = (pred_sigma - np.mean(pred_sigma)) / (np.std(pred_sigma) + 1e-6)
    z_inv_air = (-airs - np.mean(-airs)) / (np.std(-airs) + 1e-6)
    auroc_combined = roc_auc_score(y_high_error, z_sigma + z_inv_air)
    
    results = {
        "n_samples": len(df),
        "error_threshold_75th": float(err_threshold),
        "spearman_sigma_vs_abs_error": float(round(rho_sigma, 4)),
        "spearman_air_vs_abs_error": float(round(rho_air, 4)),
        "auroc_mc_dropout_sigma": float(round(auroc_sigma, 4)),
        "auroc_inverted_air": float(round(auroc_inv_air, 4)),
        "auroc_combined_sigma_and_air": float(round(auroc_combined, 4)),
    }
    
    os.makedirs("reports", exist_ok=True)
    with open("reports/air_auroc_experiment.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "="*60)
    print("AIR-AUROC & UNCERTAINTY ERROR DIAGNOSTIC REPORT")
    print("="*60)
    print(f"Evaluated Test Samples: {results['n_samples']}")
    print(f"MC-Dropout Sigma vs Abs Error Spearman Rho: {results['spearman_sigma_vs_abs_error']} (p = {p_sigma:.3e})")
    print(f"AIR vs Abs Error Spearman Rho: {results['spearman_air_vs_abs_error']} (p = {p_air:.3e})")
    print(f"AUROC (MC-Dropout Sigma predicting High Error): {results['auroc_mc_dropout_sigma']:.4f}")
    print(f"AUROC (Inverted AIR predicting High Error):      {results['auroc_inverted_air']:.4f}")
    print(f"AUROC (Combined Sigma + Inverted AIR):          {results['auroc_combined_sigma_and_air']:.4f}")
    print("="*60 + "\n")
    return results

if __name__ == "__main__":
    evaluate_air_auroc()
