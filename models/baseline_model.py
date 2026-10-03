"""
baseline_model.py - Baseline Pan-Specific Peptide-HLA stability prediction models:
  1. Ridge Regression Baseline (Fast Linear Baseline)
  2. Pan-Specific Neural Network (PyTorch MLP/CNN taking peptide + 34-mer HLA pseudo-sequence)
"""

import os
import json
import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
NUM_AA = len(AMINO_ACIDS)


# Standard BLOSUM62 matrix entries for the 20 canonical amino acids
BLOSUM62_DIAG = {
    'A': 4, 'C': 9, 'D': 6, 'E': 5, 'F': 6, 'G': 6, 'H': 8, 'I': 4,
    'K': 5, 'L': 4, 'M': 5, 'N': 6, 'P': 7, 'Q': 5, 'R': 5, 'S': 4,
    'T': 5, 'V': 4, 'W': 11, 'Y': 7
}


def one_hot_encode_sequence(seq: str, max_len: int) -> np.ndarray:
    """One-hot encode an amino acid sequence with zero-padding."""
    mat = np.zeros((max_len, NUM_AA), dtype=np.float32)
    for i, aa in enumerate(seq[:max_len]):
        if aa in AA_TO_IDX:
            mat[i, AA_TO_IDX[aa]] = 1.0
    return mat.flatten()


def encode_dataset(
    df: pd.DataFrame,
    max_pep_len: int = 9,
    pseudo_len: int = 34
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode dataframe peptides and HLA pseudo-sequences into feature matrices."""
    X_pep = np.array([one_hot_encode_sequence(p, max_pep_len) for p in df["peptide"]])
    X_hla = np.array([one_hot_encode_sequence(p, pseudo_len) for p in df["hla_pseudoseq"]])
    X = np.hstack([X_pep, X_hla])
    y_score = df["stability_score"].values.astype(np.float32)
    y_thalf = df["thalf_hours"].values.astype(np.float32)
    return X, y_score, y_thalf


class PeptideHLADataset(Dataset):
    """PyTorch Dataset for (Peptide, HLA Pseudo-sequence) stability pairs."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class PanStabilityMLP(nn.Module):
    """
    Pan-specific MLP model combining peptide features and HLA pocket pseudo-sequence.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # outputs stability score in [0, 1]
        )

    def forward(self, x):
        return self.net(x)


def train_baseline_ridge(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
    """Train Ridge regression baseline."""
    X_train, y_train, _ = encode_dataset(train_df)
    X_test, y_test, y_test_thalf = encode_dataset(test_df)
    
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    y_pred_score = np.clip(model.predict(X_test), 0.0, 1.0)
    # inverting score S = 1/(1 + 5/thalf) -> thalf = 5 * S / (1 - S)
    y_pred_thalf = 5.0 * y_pred_score / np.maximum(1.0 - y_pred_score, 1e-4)
    
    sp_rho, _ = spearmanr(y_test_thalf, y_pred_thalf)
    pe_r, _ = pearsonr(y_test, y_pred_score)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_score))
    
    y_binary = (y_test_thalf >= 2.0).astype(int)
    auc_val = roc_auc_score(y_binary, y_pred_score) if len(np.unique(y_binary)) > 1 else 0.5
    
    return {
        "model_type": "Ridge Regression",
        "spearman_rho": float(round(sp_rho, 4)),
        "pearson_r": float(round(pe_r, 4)),
        "rmse": float(round(rmse, 4)),
        "roc_auc": float(round(auc_val, 4)),
        "y_pred_score": y_pred_score,
        "y_pred_thalf": y_pred_thalf
    }


def train_baseline_mlp(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Train PyTorch Pan-Specific MLP baseline."""
    X_train, y_train, _ = encode_dataset(train_df)
    X_val, y_val, _ = encode_dataset(val_df)
    X_test, y_test, y_test_thalf = encode_dataset(test_df)
    
    train_ds = PeptideHLADataset(X_train, y_train)
    val_ds = PeptideHLADataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    model = PanStabilityMLP(input_dim=X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_val_loss = float("inf")
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(bx)
            
        train_loss /= len(train_ds)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                val_loss += criterion(out, by).item() * len(bx)
        val_loss /= len(val_ds)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            
    if best_state is not None:
        model.load_state_dict(best_state)
        
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_inputs = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_pred_score = model(test_inputs).cpu().numpy().flatten()
        
    y_pred_thalf = 5.0 * y_pred_score / np.maximum(1.0 - y_pred_score, 1e-4)
    
    sp_rho, _ = spearmanr(y_test_thalf, y_pred_thalf)
    pe_r, _ = pearsonr(y_test, y_pred_score)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_score))
    
    y_binary = (y_test_thalf >= 2.0).astype(int)
    auc_val = roc_auc_score(y_binary, y_pred_score) if len(np.unique(y_binary)) > 1 else 0.5
    
    return {
        "model_type": "Pan-Specific PyTorch MLP",
        "spearman_rho": float(round(sp_rho, 4)),
        "pearson_r": float(round(pe_r, 4)),
        "rmse": float(round(rmse, 4)),
        "roc_auc": float(round(auc_val, 4)),
        "best_val_loss": float(round(best_val_loss, 4)),
        "y_pred_score": y_pred_score,
        "y_pred_thalf": y_pred_thalf,
        "model": model
    }


def evaluate_baselines_on_all_splits(splits_dir: str = "data/splits") -> Dict[str, Any]:
    """Train and evaluate baseline models across all three splits."""
    results = {}
    split_names = ["random", "unseen_peptides", "unseen_alleles"]
    
    for split in split_names:
        logger.info(f"=== Evaluating baselines on {split.upper()} split ===")
        train_df = pd.read_csv(os.path.join(splits_dir, split, "train.csv"))
        val_df = pd.read_csv(os.path.join(splits_dir, split, "val.csv"))
        test_df = pd.read_csv(os.path.join(splits_dir, split, "test.csv"))
        
        ridge_res = train_baseline_ridge(train_df, test_df)
        mlp_res = train_baseline_mlp(train_df, val_df, test_df, epochs=10)
        
        # Save model checkpoint
        os.makedirs(f"models/checkpoints/{split}", exist_ok=True)
        torch.save(mlp_res["model"].state_dict(), f"models/checkpoints/{split}/mlp_baseline.pt")
        
        results[split] = {
            "Ridge": {k: v for k, v in ridge_res.items() if not isinstance(v, (np.ndarray, nn.Module))},
            "MLP": {k: v for k, v in mlp_res.items() if not isinstance(v, (np.ndarray, nn.Module))}
        }
        
    summary_path = "models/baseline_results.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
        
    logger.info(f"Saved baseline results summary to {summary_path}")
    return results


if __name__ == "__main__":
    res = evaluate_baselines_on_all_splits()
    print(json.dumps(res, indent=2))
