"""
baseline_model.py - Baseline Pan-Specific Peptide-HLA stability prediction models:
  1. Ridge Regression Baseline (Fast Linear Baseline)
  2. Pan-Specific Neural Network (PyTorch MLP taking peptide + 34-mer HLA pseudo-sequence)

Both models regress the canonical target ``log10(1 + thalf_hours)`` defined in
``src/targets.py``, and are scored by the single ``src.evaluate.compute_metrics``
convention, so their numbers are mutually comparable and comparable to
NetMHCstabpan.
"""

import os
import json
import logging
import random
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import Ridge

from src.evaluate import compute_metrics, compute_per_allele_metrics
from src.targets import TARGET_NAME, get_target, target_to_thalf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
NUM_AA = len(AMINO_ACIDS)

#: Default seeds for the multi-seed protocol (Phase 2 requirement 8).
DEFAULT_SEEDS = (0, 1, 2, 3, 4)


# Standard BLOSUM62 matrix entries for the 20 canonical amino acids
BLOSUM62_DIAG = {
    'A': 4, 'C': 9, 'D': 6, 'E': 5, 'F': 6, 'G': 6, 'H': 8, 'I': 4,
    'K': 5, 'L': 4, 'M': 5, 'N': 6, 'P': 7, 'Q': 5, 'R': 5, 'S': 4,
    'T': 5, 'V': 4, 'W': 11, 'Y': 7
}


def set_seed(seed: int) -> None:
    """Seed every RNG that affects training (init, shuffling, dropout)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    """
    Encode peptides and HLA pseudo-sequences into feature matrices.

    Returns ``(X, y_target, y_thalf)`` where ``y_target`` is the canonical
    ``log10(1 + thalf)`` target and ``y_thalf`` is the raw half-life in hours
    (kept for the binary stability metrics).
    """
    X_pep = np.array([one_hot_encode_sequence(p, max_pep_len) for p in df["peptide"]])
    X_hla = np.array([one_hot_encode_sequence(p, pseudo_len) for p in df["hla_pseudoseq"]])
    X = np.hstack([X_pep, X_hla])
    y_target = get_target(df).astype(np.float32)
    y_thalf = df["thalf_hours"].values.astype(np.float32)
    return X, y_target, y_thalf


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
    Pan-specific MLP combining peptide features and HLA pocket pseudo-sequence.

    The output head is linear: the canonical target ``log10(1 + thalf)`` spans
    roughly [0, 2.41] on this dataset, so the previous sigmoid head (which bounded
    predictions to [0, 1]) could not reach the stable end of the range at all.
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
            nn.Linear(32, 1)  # linear head: regresses log10(1 + thalf) directly
        )

    def forward(self, x):
        return self.net(x)


def _score(
    test_df: pd.DataFrame,
    y_test_thalf: np.ndarray,
    y_pred_target: np.ndarray,
    model_type: str,
    extra: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Score predictions on the canonical convention, with a per-allele breakdown."""
    y_pred_thalf = target_to_thalf(y_pred_target)

    res: Dict[str, Any] = {"model_type": model_type}
    res.update(compute_metrics(y_test_thalf, y_pred_thalf))
    res["per_allele"] = compute_per_allele_metrics(
        test_df["allele"].values, y_test_thalf, y_pred_thalf
    )
    if extra:
        res.update(extra)
    res["y_pred_target"] = np.asarray(y_pred_target, dtype=float)
    res["y_pred_thalf"] = np.asarray(y_pred_thalf, dtype=float)
    return res


def train_baseline_ridge(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    alpha: float = 1.0
) -> Dict[str, Any]:
    """
    Train Ridge regression baseline on the canonical target.

    Ridge is deterministic given the data, so it carries no seed spread -- the
    multi-seed protocol applies only to the stochastic models.
    """
    X_train, y_train, _ = encode_dataset(train_df)
    X_test, _, y_test_thalf = encode_dataset(test_df)

    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    # Predict the target directly. The old code predicted the saturating
    # stability_score and inverted it with thalf = 5S/(1-S), which blew up to
    # ~50,000 h whenever S clipped to 1.0.
    y_pred_target = np.clip(model.predict(X_test), 0.0, None)

    return _score(
        test_df, y_test_thalf, y_pred_target,
        "Ridge Regression",
        extra={"deterministic": True, "alpha": alpha}
    )


def train_baseline_mlp(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 0
) -> Dict[str, Any]:
    """Train the PyTorch pan-specific MLP baseline on the canonical target."""
    set_seed(seed)

    X_train, y_train, _ = encode_dataset(train_df)
    X_val, y_val, _ = encode_dataset(val_df)
    X_test, _, y_test_thalf = encode_dataset(test_df)

    train_ds = PeptideHLADataset(X_train, y_train)
    val_ds = PeptideHLADataset(X_val, y_val)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)
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
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_inputs = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_pred_target = model(test_inputs).cpu().numpy().flatten()

    res = _score(
        test_df, y_test_thalf, y_pred_target,
        "Pan-Specific PyTorch MLP",
        extra={
            "best_val_loss": float(round(best_val_loss, 4)),
            "seed": seed,
            "epochs": epochs,
        }
    )
    res["model"] = model
    return res


def aggregate_over_seeds(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collapse several single-seed runs into mean +/- std per metric.

    Reported as ``{metric: {"mean":..., "std":..., "values":[...]}}`` so the spread
    is visible rather than implied.
    """
    metric_keys = [
        "spearman_rho", "pearson_r", "rmse", "mae", "r2_score", "roc_auc", "pr_auc"
    ]
    out: Dict[str, Any] = {
        "model_type": runs[0].get("model_type"),
        "target": TARGET_NAME,
        "n_seeds": len(runs),
        "seeds": [r.get("seed") for r in runs],
        "n_samples": runs[0].get("n_samples"),
    }
    for key in metric_keys:
        vals = [float(r[key]) for r in runs if r.get(key) is not None]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        out[key] = {
            "mean": float(round(float(arr.mean()), 4)),
            "std": float(round(float(arr.std(ddof=1)), 4)) if len(arr) > 1 else 0.0,
            "values": [float(round(v, 4)) for v in vals],
        }

    # Median per-allele Spearman is the honest pan-specific number; aggregate it too.
    med = [
        r["per_allele"]["spearman_rho_across_alleles"]["median"]
        for r in runs
        if r.get("per_allele", {}).get("spearman_rho_across_alleles")
    ]
    if med:
        arr = np.asarray(med, dtype=float)
        out["per_allele_spearman_median"] = {
            "mean": float(round(float(arr.mean()), 4)),
            "std": float(round(float(arr.std(ddof=1)), 4)) if len(arr) > 1 else 0.0,
            "values": [float(round(v, 4)) for v in med],
        }
    return out


def _strip_arrays(d: Dict[str, Any]) -> Dict[str, Any]:
    """Drop array/model values so a result dict is JSON-serialisable."""
    return {
        k: v for k, v in d.items()
        if not isinstance(v, (np.ndarray, nn.Module))
    }


def evaluate_baselines_on_all_splits(
    splits_dir: str = "data/splits",
    seeds: Tuple[int, ...] = DEFAULT_SEEDS,
    epochs: int = 10,
    predictions_dir: str = "models/predictions"
) -> Dict[str, Any]:
    """
    Train and evaluate baselines across all three splits, over several seeds.

    Per-row test predictions are written to
    ``{predictions_dir}/{split}/{model}.csv`` so downstream analyses -- in
    particular the NetMHCstabpan head-to-head on its 320-pair subset -- can
    re-score the same models on any row subset without retraining.
    """
    results: Dict[str, Any] = {}
    split_names = ["random", "unseen_peptides", "unseen_alleles"]

    for split in split_names:
        logger.info(f"=== Evaluating baselines on {split.upper()} split ===")
        train_df = pd.read_csv(os.path.join(splits_dir, split, "train.csv"))
        val_df = pd.read_csv(os.path.join(splits_dir, split, "val.csv"))
        test_df = pd.read_csv(os.path.join(splits_dir, split, "test.csv"))

        pred_dir = os.path.join(predictions_dir, split)
        os.makedirs(pred_dir, exist_ok=True)
        key_cols = ["allele", "peptide", "thalf_hours"]

        # --- Ridge (deterministic) ---
        ridge_res = train_baseline_ridge(train_df, test_df)
        ridge_preds = test_df[key_cols].copy()
        ridge_preds["pred_target"] = ridge_res["y_pred_target"]
        ridge_preds["pred_thalf_hours"] = ridge_res["y_pred_thalf"]
        ridge_preds.to_csv(os.path.join(pred_dir, "ridge.csv"), index=False)

        # --- MLP (one run per seed) ---
        mlp_runs = []
        for seed in seeds:
            logger.info(f"  MLP seed {seed} ...")
            run = train_baseline_mlp(train_df, val_df, test_df, epochs=epochs, seed=seed)
            mlp_runs.append(run)

            preds = test_df[key_cols].copy()
            preds["pred_target"] = run["y_pred_target"]
            preds["pred_thalf_hours"] = run["y_pred_thalf"]
            preds.to_csv(os.path.join(pred_dir, f"mlp_seed{seed}.csv"), index=False)

        # Checkpoint the first seed's model for reference
        os.makedirs(f"models/checkpoints/{split}", exist_ok=True)
        torch.save(
            mlp_runs[0]["model"].state_dict(),
            f"models/checkpoints/{split}/mlp_baseline.pt"
        )

        results[split] = {
            "Ridge": _strip_arrays(ridge_res),
            "MLP": {
                "aggregate": aggregate_over_seeds(mlp_runs),
                "per_seed": [_strip_arrays(r) for r in mlp_runs],
            },
        }

    summary_path = "models/baseline_results.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved baseline results summary to {summary_path}")
    logger.info(f"Saved per-row test predictions under {predictions_dir}/")
    return results


if __name__ == "__main__":
    res = evaluate_baselines_on_all_splits()
    print(json.dumps(res, indent=2))
