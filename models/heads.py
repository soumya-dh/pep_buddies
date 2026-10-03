"""
heads.py - Featurisations and prediction heads on top of frozen embeddings.

Phase 2 requirement 6 says start simple and escalate only while the gains clear
the error bars. The featurisations below are ordered by that principle:

``mean``
    ``[mean(peptide) | mean(hla_pocket)]``. The weakest option, and the "does any
    signal survive pooling" probe. Mean-pooling a 9-mer averages away the anchor
    positions (P2/P9) that dominate stability.

``perpos``
    ``[flatten(peptide 9 x d) | mean(hla_pocket)]``. Keeps peptide position
    identity. If this clearly beats ``mean``, that alone justifies caching
    per-position embeddings.

``perpos_pocket``
    ``[flatten(peptide 9 x d) | flatten(hla_pocket 34 x d)]``. Everything, and
    the widest design matrix -- at 650M this is 9*1280 + 34*1280 = 55,040
    features against ~22k training rows, so ridge regularisation is doing real
    work here and the alpha sweep matters.

The HLA side is always the 34 pocket positions sliced out of the full 182-aa
G-domain embedding, never an embedding of the 34-mer pseudosequence string.

Heads: ``RidgeHead`` (with a validation-chosen alpha) is the honest first rung.
``MLPHead`` projects each position to a small dimension before mixing, because
55k raw features into a dense layer would overfit 22k rows immediately.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.embeddings.cache import EmbeddingCache
from src.targets import get_target

logger = logging.getLogger(__name__)

FEATURISATIONS = ("mean", "perpos", "perpos_pocket")

#: Alphas swept on the validation split. Embedding features are dense and
#: correlated, so the useful range sits far higher than Ridge's default of 1.0.
DEFAULT_ALPHAS = (1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0)


class FeatureBuilder:
    """Builds design matrices for a dataframe from the embedding caches."""

    def __init__(
        self,
        model_key: str,
        root: str = "data/embeddings",
        featurisation: str = "perpos",
    ):
        if featurisation not in FEATURISATIONS:
            raise ValueError(
                f"featurisation must be one of {FEATURISATIONS}, got {featurisation!r}"
            )
        self.model_key = model_key
        self.featurisation = featurisation
        self.pep = EmbeddingCache(model_key, "peptides", root=root)
        self.hla = EmbeddingCache(model_key, "hla", root=root)

        contact = self.hla.meta.get("contact_positions_0based")
        if not contact:
            raise ValueError(
                f"HLA cache for {model_key} has no contact_positions_0based; "
                "rebuild it with src/embeddings/build.py"
            )
        self.contact_positions: List[int] = list(contact)  # type: ignore[arg-type]

    @property
    def uses_random_weights(self) -> bool:
        """True if either cache was built with --random-init (results invalid)."""
        return bool(
            self.pep.meta.get("random_init") or self.hla.meta.get("random_init")
        )

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Design matrix for ``df``, one row per dataframe row."""
        peptides = df["peptide"].astype(str).tolist()
        hla_seqs = df["hla_seq"].astype(str).tolist()

        pocket = self.hla.get_positions(hla_seqs, self.contact_positions)  # (n,34,d)

        if self.featurisation == "mean":
            left = self.pep.get_mean(peptides)                 # (n, d)
            right = pocket.mean(axis=1)                        # (n, d)
        elif self.featurisation == "perpos":
            left = self.pep.get_flat(peptides)                 # (n, 9d)
            right = pocket.mean(axis=1)                        # (n, d)
        else:  # perpos_pocket
            left = self.pep.get_flat(peptides)                 # (n, 9d)
            right = pocket.reshape(pocket.shape[0], -1)        # (n, 34d)

        return np.hstack([left, right]).astype(np.float32)

    def describe(self, n_features: int) -> Dict[str, Any]:
        return {
            "model_key": self.model_key,
            "featurisation": self.featurisation,
            "n_features": int(n_features),
            "embedding_dim": int(self.pep.meta["dim"]),
            "peptide_length": int(self.pep.meta["length"]),
            "n_contact_positions": len(self.contact_positions),
            "random_init": self.uses_random_weights,
        }


class RidgeHead:
    """
    Standardised ridge regression on embedding features.

    Alpha is chosen on the validation split by Spearman, not by MSE: the headline
    metric is rank correlation, and on the held-out-allele split the two can
    disagree because a model may rank well while being poorly calibrated.

    Standardisation matters here -- embedding dimensions have very different
    scales, and an unscaled ridge penalty would effectively regularise them
    unequally.
    """

    def __init__(self, alphas: Sequence[float] = DEFAULT_ALPHAS):
        self.alphas = list(alphas)
        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[Ridge] = None
        self.alpha_: Optional[float] = None
        self.val_curve_: List[Dict[str, float]] = []

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "RidgeHead":
        from scipy.stats import spearmanr

        self.scaler = StandardScaler().fit(X_train)
        Xt = self.scaler.transform(X_train)
        Xv = self.scaler.transform(X_val)

        best = (-np.inf, None, None)
        for a in self.alphas:
            m = Ridge(alpha=a).fit(Xt, y_train)
            pv = m.predict(Xv)
            rho = float(spearmanr(y_val, pv)[0]) if np.std(pv) > 0 else 0.0
            self.val_curve_.append({"alpha": a, "val_spearman": round(rho, 4)})
            if rho > best[0]:
                best = (rho, a, m)

        _, self.alpha_, self.model = best
        logger.info(
            "  ridge alpha=%g (val rho=%.4f) from %s",
            self.alpha_, best[0], [c["alpha"] for c in self.val_curve_],
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None or self.scaler is None:
            raise RuntimeError("fit() first")
        # Predictions are on the canonical target; clip at 0 since a negative
        # log10(1+thalf) implies a negative half-life.
        return np.clip(self.model.predict(self.scaler.transform(X)), 0.0, None)

    @property
    def params(self) -> Dict[str, Any]:
        return {
            "head": "ridge",
            "alpha": self.alpha_,
            "deterministic": True,
            "val_curve": self.val_curve_,
        }


def load_split(splits_dir: str, split: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val/test frames for one split."""
    import os
    return tuple(  # type: ignore[return-value]
        pd.read_csv(os.path.join(splits_dir, split, f"{part}.csv"))
        for part in ("train", "val", "test")
    )


def targets_for(df: pd.DataFrame) -> np.ndarray:
    """Canonical regression target for a dataframe."""
    return get_target(df).astype(np.float64)
