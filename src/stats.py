"""
stats.py - Bootstrap confidence intervals and paired model comparisons.

Phase 2 requirement 8 asks for correlation and error on every split with several
seeds, reported as mean +/- spread. Seed spread alone is not enough to claim a
win: it measures training variance, not sampling variance of the *test set*. Two
models can differ by more than their seed spread and still be indistinguishable
given only 3,078 test rows across 8 alleles.

So a win is claimed only when a **paired bootstrap** interval for the difference
excludes zero.

Two resampling units are provided, and they answer different questions:

``unit="row"``
    Resample test rows. Answers "would this ranking hold on another sample of
    peptide-HLA pairs from these same alleles?"

``unit="allele"``
    Resample whole alleles (a cluster bootstrap). Answers "would this ranking
    hold on another sample of *alleles*?" -- which is the actual claim a
    pan-specific model makes. With only 8 held-out alleles these intervals are
    wide, and that width is the honest result, not a defect to hide.
"""

import logging
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, roc_auc_score

from src.targets import STABILITY_THRESHOLD_HOURS, thalf_to_target

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Metrics, all on the canonical target
# --------------------------------------------------------------------------- #

def _spearman(yt: np.ndarray, yp: np.ndarray) -> float:
    if np.std(yt) == 0 or np.std(yp) == 0:
        return np.nan
    return float(spearmanr(yt, yp)[0])


def _pearson(yt: np.ndarray, yp: np.ndarray) -> float:
    if np.std(yt) == 0 or np.std(yp) == 0:
        return np.nan
    return float(pearsonr(yt, yp)[0])


def _rmse(yt: np.ndarray, yp: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(yt, yp)))


def _roc_auc(yt_thalf: np.ndarray, yp: np.ndarray) -> float:
    yb = (yt_thalf >= STABILITY_THRESHOLD_HOURS).astype(int)
    if len(np.unique(yb)) < 2:
        return np.nan
    return float(roc_auc_score(yb, yp))


#: name -> fn(y_true_target, y_pred_target, y_true_thalf) -> float
METRICS: Dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray], float]] = {
    "spearman_rho": lambda yt, yp, th: _spearman(yt, yp),
    "pearson_r": lambda yt, yp, th: _pearson(yt, yp),
    "rmse": lambda yt, yp, th: _rmse(yt, yp),
    "roc_auc": lambda yt, yp, th: _roc_auc(th, yp),
}


def median_per_allele_spearman(
    alleles: np.ndarray,
    yt: np.ndarray,
    yp: np.ndarray,
    min_samples: int = 10,
) -> float:
    """Median of the within-allele Spearman values; nan if nothing is scorable."""
    vals = []
    for a in np.unique(alleles):
        m = alleles == a
        if m.sum() < min_samples:
            continue
        v = _spearman(yt[m], yp[m])
        if not np.isnan(v):
            vals.append(v)
    return float(np.median(vals)) if vals else np.nan


# --------------------------------------------------------------------------- #
# Resampling
# --------------------------------------------------------------------------- #

def _resample_indices(
    n: int,
    alleles: Optional[np.ndarray],
    unit: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Indices for one bootstrap replicate."""
    if unit == "row":
        return rng.integers(0, n, size=n)
    if unit == "allele":
        if alleles is None:
            raise ValueError("unit='allele' needs the alleles array")
        groups = np.unique(alleles)
        picked = rng.choice(groups, size=len(groups), replace=True)
        # Concatenating whole alleles keeps within-allele structure intact.
        return np.concatenate([np.flatnonzero(alleles == g) for g in picked])
    raise ValueError(f"unit must be 'row' or 'allele', got {unit!r}")


def _percentile_ci(vals: np.ndarray, alpha: float) -> Dict[str, float]:
    good = vals[~np.isnan(vals)]
    if good.size == 0:
        return {"lo": float("nan"), "hi": float("nan"), "n_valid": 0}
    return {
        "lo": float(round(float(np.percentile(good, 100 * alpha / 2)), 4)),
        "hi": float(round(float(np.percentile(good, 100 * (1 - alpha / 2))), 4)),
        "n_valid": int(good.size),
    }


def bootstrap_metric(
    y_true_thalf: np.ndarray,
    y_pred_thalf: np.ndarray,
    metric: str = "spearman_rho",
    alleles: Optional[np.ndarray] = None,
    unit: str = "row",
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, Any]:
    """Point estimate plus a percentile bootstrap CI for one model."""
    yt_thalf = np.asarray(y_true_thalf, dtype=float)
    yt = thalf_to_target(yt_thalf)
    yp = thalf_to_target(np.asarray(y_pred_thalf, dtype=float))
    fn = METRICS[metric]

    point = fn(yt, yp, yt_thalf)
    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = _resample_indices(len(yt), alleles, unit, rng)
        reps[b] = fn(yt[idx], yp[idx], yt_thalf[idx])

    ci = _percentile_ci(reps, alpha)
    return {
        "metric": metric,
        "unit": unit,
        "point": float(round(point, 4)) if not np.isnan(point) else None,
        "ci_lo": ci["lo"],
        "ci_hi": ci["hi"],
        "n_boot": n_boot,
        "n_valid_replicates": ci["n_valid"],
    }


def paired_delta(
    y_true_thalf: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    metric: str = "spearman_rho",
    alleles: Optional[np.ndarray] = None,
    unit: str = "row",
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    label_a: str = "A",
    label_b: str = "B",
) -> Dict[str, Any]:
    """
    Paired bootstrap for ``metric(A) - metric(B)``.

    Both models are evaluated on the *same* resampled rows each replicate, which
    cancels the shared difficulty of those rows and gives a far tighter interval
    than comparing two independent CIs.

    ``significant`` is True when the interval excludes zero. For RMSE, where
    lower is better, ``favours`` accounts for the sign.
    """
    yt_thalf = np.asarray(y_true_thalf, dtype=float)
    yt = thalf_to_target(yt_thalf)
    ypa = thalf_to_target(np.asarray(y_pred_a, dtype=float))
    ypb = thalf_to_target(np.asarray(y_pred_b, dtype=float))
    fn = METRICS[metric]

    point = fn(yt, ypa, yt_thalf) - fn(yt, ypb, yt_thalf)
    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = _resample_indices(len(yt), alleles, unit, rng)
        reps[b] = fn(yt[idx], ypa[idx], yt_thalf[idx]) - fn(yt[idx], ypb[idx], yt_thalf[idx])

    ci = _percentile_ci(reps, alpha)
    significant = bool(
        not np.isnan(ci["lo"]) and not np.isnan(ci["hi"])
        and (ci["lo"] > 0 or ci["hi"] < 0)
    )
    lower_is_better = metric == "rmse"
    if not significant:
        favours = None
    elif (point > 0) != lower_is_better:
        favours = label_a
    else:
        favours = label_b

    good = reps[~np.isnan(reps)]
    return {
        "metric": metric,
        "unit": unit,
        "comparison": f"{label_a} - {label_b}",
        "delta": float(round(point, 4)),
        "ci_lo": ci["lo"],
        "ci_hi": ci["hi"],
        "significant": significant,
        "favours": favours,
        "p_delta_gt_0": float(round(float((good > 0).mean()), 4)) if good.size else None,
        "n_boot": n_boot,
        "lower_is_better": lower_is_better,
    }
