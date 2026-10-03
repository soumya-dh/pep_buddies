"""
targets.py - Single source of truth for the regression target and evaluation convention.

Phase 1 reported Spearman on raw ``thalf_hours``, Pearson on ``stability_score`` and
RMSE on ``stability_score`` -- three different scales inside one results row, which
makes the numbers non-comparable across models.

Everything downstream now regresses and scores a single canonical target:

    y = log10(1 + thalf_hours)

Why this one:
  * ``thalf_hours`` has a true minimum of 0.0 in this dataset, so ``log(thalf)`` is
    undefined. The ``1 +`` offset keeps the transform total over the observed range.
  * It is already materialised as the ``log_thalf`` column of the cleaned dataset.
  * It is monotone in ``thalf_hours``, so Spearman and ROC-AUC are identical whether
    computed on the target or on hours -- only Pearson/RMSE/MAE change, and those are
    exactly the metrics that need a sane scale.
  * The legacy ``stability_score = thalf / (thalf + 5)`` saturates hard: everything
    above ~20 h is compressed into a sliver, so RMSE on it is dominated by the
    unstable end of the range and understates error where it matters.
"""

from typing import Union

import numpy as np
import pandas as pd

ArrayLike = Union[np.ndarray, pd.Series, list]

#: Canonical target name, recorded in every metrics payload so old and new
#: result files can never be silently mixed.
TARGET_NAME = "log10_1p_thalf"

#: Human-readable axis label for figures.
TARGET_LABEL = r"$\log_{10}(1 + T_{1/2}\;[\mathrm{h}])$"

#: Stability cut-off for the binary (stable binder) metrics, in hours.
STABILITY_THRESHOLD_HOURS = 2.0

#: Half-saturation constant of the legacy ``stability_score`` transform.
LEGACY_SCORE_HALF_SATURATION = 5.0


def thalf_to_target(thalf_hours: ArrayLike) -> np.ndarray:
    """Map half-life in hours onto the canonical regression target."""
    thalf = np.asarray(thalf_hours, dtype=float)
    return np.log10(1.0 + np.clip(thalf, 0.0, None))


def target_to_thalf(y: ArrayLike) -> np.ndarray:
    """Invert :func:`thalf_to_target` back to half-life in hours."""
    y_arr = np.asarray(y, dtype=float)
    return np.clip(np.power(10.0, y_arr) - 1.0, 0.0, None)


def target_threshold() -> float:
    """The stability cut-off expressed on the canonical target scale."""
    return float(thalf_to_target(STABILITY_THRESHOLD_HOURS))


def thalf_to_legacy_score(thalf_hours: ArrayLike) -> np.ndarray:
    """
    Legacy ``stability_score`` transform, kept only so Phase 1 numbers remain
    reproducible. Do not use for new models.
    """
    thalf = np.asarray(thalf_hours, dtype=float)
    return thalf / (thalf + LEGACY_SCORE_HALF_SATURATION)


def get_target(df: pd.DataFrame) -> np.ndarray:
    """
    Pull the canonical target out of a dataframe.

    Uses the precomputed ``log_thalf`` column when present and consistent with
    ``thalf_hours``, otherwise recomputes it from hours.
    """
    if "thalf_hours" not in df.columns:
        if "log_thalf" in df.columns:
            return np.asarray(df["log_thalf"].values, dtype=float)
        raise KeyError("dataframe needs either 'thalf_hours' or 'log_thalf'")

    recomputed = thalf_to_target(df["thalf_hours"].values)
    if "log_thalf" in df.columns:
        stored = np.asarray(df["log_thalf"].values, dtype=float)
        if not np.allclose(stored, recomputed, atol=1e-6):
            raise ValueError(
                "'log_thalf' column disagrees with log10(1 + thalf_hours); "
                "regenerate the cleaned dataset before training"
            )
        return stored
    return recomputed


def get_binary_labels(df_or_thalf: Union[pd.DataFrame, ArrayLike]) -> np.ndarray:
    """Binary stable-binder labels at :data:`STABILITY_THRESHOLD_HOURS`."""
    if isinstance(df_or_thalf, pd.DataFrame):
        thalf = df_or_thalf["thalf_hours"].values
    else:
        thalf = df_or_thalf
    return (np.asarray(thalf, dtype=float) >= STABILITY_THRESHOLD_HOURS).astype(int)
