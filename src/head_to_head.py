"""
head_to_head.py - Fair NetMHCstabpan comparison on a common evaluation subset.

Phase 1 put these two numbers in the same results table:

  * NetMHCstabpan: Spearman 0.8537 over **320** rows (``benchmark_metrics.json``),
    because only 320 (allele, peptide) pairs were ever submitted to the DTU server.
  * Our MLP: Spearman 0.4382 over **3,078** rows -- the whole ``unseen_alleles``
    test set.

Those are different evaluation sets, so the 0.41 gap between them is not a
measured quantity. This module re-scores every model on exactly the pairs
NetMHCstabpan returned, and reports the full-test-set number beside it so the
coverage difference is explicit rather than hidden.

Our models are read from the per-row prediction CSVs written by
``models.baseline_model.evaluate_baselines_on_all_splits``, so no retraining is
needed to add a model to the comparison.
"""

import glob
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.evaluate import compute_metrics, compute_per_allele_metrics
from src.targets import TARGET_NAME

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

JOIN_KEYS = ["allele_join", "peptide"]


def _allele_join_key(s: pd.Series) -> pd.Series:
    """Normalise allele spellings (``HLA-A*24:02`` / ``HLA-A2402`` / ``A2402``) for joining."""
    return (
        s.astype(str)
        .str.upper()
        .str.replace("HLA-", "", regex=False)
        .str.replace("*", "", regex=False)
        .str.replace(":", "", regex=False)
        .str.replace("_", "", regex=False)
    )


def load_head_to_head_index(
    test_csv: str = "data/splits/unseen_alleles/test.csv",
    netmhc_csv: str = "data/netmhcstabpan_benchmark/predictions.csv",
) -> pd.DataFrame:
    """
    Build the common evaluation subset: ground truth joined to NetMHCstabpan output.

    Returns one row per (allele, peptide) pair that NetMHCstabpan actually scored,
    carrying the experimental ``thalf_hours`` and NetMHCstabpan's predicted
    ``netmhc_thalf_hours``.
    """
    test = pd.read_csv(test_csv)
    pred = pd.read_csv(netmhc_csv)

    test["allele_join"] = _allele_join_key(test["allele"])
    pred["allele_join"] = _allele_join_key(pred["allele"])

    merged = pd.merge(
        test[["allele", "peptide", "thalf_hours", "allele_join"]],
        pred[["peptide", "allele_join", "netmhc_score", "netmhc_thalf_hours"]],
        on=JOIN_KEYS,
        how="inner",
    )

    n_requested = len(pred)
    if len(merged) != n_requested:
        logger.warning(
            "Matched %d of %d NetMHCstabpan predictions to ground truth; "
            "%d unmatched rows are excluded from the head-to-head.",
            len(merged), n_requested, n_requested - len(merged),
        )
    if merged.duplicated(subset=JOIN_KEYS).any():
        n_dupes = int(merged.duplicated(subset=JOIN_KEYS).sum())
        raise ValueError(
            f"{n_dupes} duplicate (allele, peptide) pairs in the head-to-head join; "
            "the subset must be one row per pair or metrics will be weighted unevenly"
        )

    logger.info(
        "Head-to-head subset: %d pairs across %d alleles.",
        len(merged), merged["allele"].nunique(),
    )
    return merged


def _discover_prediction_files(predictions_dir: str) -> Dict[str, List[str]]:
    """
    Group per-row prediction CSVs by model name.

    ``mlp_seed0.csv`` ... ``mlp_seed4.csv`` collapse into one ``mlp`` entry so the
    head-to-head can report mean +/- std across seeds.
    """
    groups: Dict[str, List[str]] = {}
    for path in sorted(glob.glob(os.path.join(predictions_dir, "*.csv"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        model = re.sub(r"_seed\d+$", "", stem)
        groups.setdefault(model, []).append(path)
    return groups


def _score_subset(
    subset: pd.DataFrame,
    y_pred_thalf: np.ndarray,
    label: str
) -> Dict[str, Any]:
    """Canonical metrics plus a per-allele breakdown for one model on one subset."""
    res: Dict[str, Any] = {"model": label}
    res.update(compute_metrics(subset["thalf_hours"].values, y_pred_thalf))
    res["per_allele"] = compute_per_allele_metrics(
        subset["allele"].values, subset["thalf_hours"].values, y_pred_thalf
    )
    return res


def _aggregate_seed_scores(scores: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    """Collapse per-seed subset scores into mean +/- std."""
    keys = ["spearman_rho", "pearson_r", "rmse", "mae", "r2_score", "roc_auc", "pr_auc"]
    out: Dict[str, Any] = {
        "model": label,
        "target": TARGET_NAME,
        "n_seeds": len(scores),
        "n_samples": scores[0]["n_samples"],
    }
    for key in keys:
        vals = np.asarray([float(s[key]) for s in scores], dtype=float)
        out[key] = {
            "mean": float(round(float(vals.mean()), 4)),
            "std": float(round(float(vals.std(ddof=1)), 4)) if len(vals) > 1 else 0.0,
            "values": [float(round(v, 4)) for v in vals],
        }
    med = [
        s["per_allele"]["spearman_rho_across_alleles"]["median"]
        for s in scores
        if s["per_allele"]["spearman_rho_across_alleles"]
    ]
    if med:
        arr = np.asarray(med, dtype=float)
        out["per_allele_spearman_median"] = {
            "mean": float(round(float(arr.mean()), 4)),
            "std": float(round(float(arr.std(ddof=1)), 4)) if len(arr) > 1 else 0.0,
            "values": [float(round(v, 4)) for v in med],
        }
    return out


def run_head_to_head(
    test_csv: str = "data/splits/unseen_alleles/test.csv",
    netmhc_csv: str = "data/netmhcstabpan_benchmark/predictions.csv",
    predictions_dir: str = "models/predictions/unseen_alleles",
    output_json: str = "reports/head_to_head.json",
    output_md: Optional[str] = "reports/head_to_head.md",
) -> Dict[str, Any]:
    """
    Score NetMHCstabpan and every available model on the common subset.

    For each of our models two numbers are produced:
      * ``subset`` -- the 320 pairs NetMHCstabpan also scored (the only fair
        comparison against it);
      * ``full_test`` -- the whole held-out-allele test set (the honest measure of
        the model itself, on far more data).
    """
    index = load_head_to_head_index(test_csv, netmhc_csv)

    report: Dict[str, Any] = {
        "target": TARGET_NAME,
        "subset": {
            "n_pairs": int(len(index)),
            "n_alleles": int(index["allele"].nunique()),
            "alleles": sorted(index["allele"].unique().tolist()),
            "source": os.path.basename(netmhc_csv),
            "pairs_per_allele": {
                str(k): int(v) for k, v in index.groupby("allele").size().items()
            },
        },
        "models": {},
    }

    # --- NetMHCstabpan on the subset (it has no predictions outside it) ---
    report["models"]["netmhcstabpan"] = {
        "subset": _score_subset(
            index, index["netmhc_thalf_hours"].values, "NetMHCstabpan-1.0"
        ),
        "full_test": None,
        "full_test_note": (
            "NetMHCstabpan was only queried for the subset pairs, so it has no "
            "full-test-set number. Any comparison against a full-test number for "
            "another model is invalid."
        ),
    }

    # --- Our models, from the saved per-row predictions ---
    if not os.path.isdir(predictions_dir):
        logger.warning(
            "No prediction directory at %s -- run "
            "models.baseline_model.evaluate_baselines_on_all_splits first.",
            predictions_dir,
        )
    else:
        groups = _discover_prediction_files(predictions_dir)
        if not groups:
            logger.warning("No prediction CSVs found in %s", predictions_dir)

        for model, paths in groups.items():
            subset_scores, full_scores = [], []

            for path in paths:
                preds = pd.read_csv(path)
                preds["allele_join"] = _allele_join_key(preds["allele"])

                # Full test set
                full_scores.append(
                    _score_subset(preds, preds["pred_thalf_hours"].values, model)
                )

                # Restricted to the head-to-head pairs
                joined = pd.merge(
                    index[JOIN_KEYS + ["allele", "thalf_hours"]],
                    preds[JOIN_KEYS + ["pred_thalf_hours"]],
                    on=JOIN_KEYS,
                    how="left",
                )
                missing = int(joined["pred_thalf_hours"].isna().sum())
                if missing:
                    raise ValueError(
                        f"{model} ({os.path.basename(path)}) is missing predictions for "
                        f"{missing} of {len(index)} head-to-head pairs; the subset "
                        "comparison would not be like-for-like"
                    )
                subset_scores.append(
                    _score_subset(joined, joined["pred_thalf_hours"].values, model)
                )

            multi_seed = len(paths) > 1
            report["models"][model] = {
                "n_runs": len(paths),
                "subset": (
                    _aggregate_seed_scores(subset_scores, model)
                    if multi_seed else subset_scores[0]
                ),
                "full_test": (
                    _aggregate_seed_scores(full_scores, model)
                    if multi_seed else full_scores[0]
                ),
            }

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote head-to-head report to %s", output_json)

    if output_md:
        md = render_markdown(report)
        with open(output_md, "w") as f:
            f.write(md)
        logger.info("Wrote head-to-head table to %s", output_md)

    return report


def _fmt(entry: Any, key: str) -> str:
    """Format a metric that may be a scalar or a {mean, std} dict."""
    if entry is None:
        return "n/a"
    val = entry.get(key)
    if val is None:
        return "n/a"
    if isinstance(val, dict):
        return f"{val['mean']:.4f} ± {val['std']:.4f}"
    return f"{float(val):.4f}"


def _fmt_per_allele(entry: Any) -> str:
    """Format the median per-allele Spearman, scalar or aggregated."""
    if entry is None:
        return "n/a"
    if "per_allele_spearman_median" in entry:
        d = entry["per_allele_spearman_median"]
        return f"{d['mean']:.4f} ± {d['std']:.4f}"
    summary = entry.get("per_allele", {}).get("spearman_rho_across_alleles")
    if not summary:
        return "n/a"
    return f"{summary['median']:.4f}"


def render_markdown(report: Dict[str, Any]) -> str:
    """Render the head-to-head report as a Markdown section."""
    sub = report["subset"]
    lines = [
        "# NetMHCstabpan Head-to-Head",
        "",
        f"Target: `{report['target']}` — all correlations and errors on this scale.",
        "",
        f"Common evaluation subset: **{sub['n_pairs']} (allele, peptide) pairs** "
        f"across **{sub['n_alleles']} held-out alleles**. These are the only pairs "
        "NetMHCstabpan was queried for, so they are the only fair basis for "
        "comparing against it.",
        "",
        "## On the common subset (n = %d)" % sub["n_pairs"],
        "",
        "| Model | Spearman ρ | Pearson r | RMSE | ROC-AUC | Median per-allele ρ |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for name, payload in report["models"].items():
        s = payload.get("subset")
        lines.append(
            f"| {name} | {_fmt(s, 'spearman_rho')} | {_fmt(s, 'pearson_r')} | "
            f"{_fmt(s, 'rmse')} | {_fmt(s, 'roc_auc')} | {_fmt_per_allele(s)} |"
        )

    lines += [
        "",
        "## On the full held-out-allele test set",
        "",
        "Reported separately because it is a *different and larger* evaluation set. "
        "Do not compare these numbers against the NetMHCstabpan row above.",
        "",
        "| Model | n | Spearman ρ | Pearson r | RMSE | ROC-AUC | Median per-allele ρ |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for name, payload in report["models"].items():
        f_ = payload.get("full_test")
        if f_ is None:
            lines.append(f"| {name} | — | n/a | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {name} | {f_.get('n_samples', '—')} | {_fmt(f_, 'spearman_rho')} | "
            f"{_fmt(f_, 'pearson_r')} | {_fmt(f_, 'rmse')} | {_fmt(f_, 'roc_auc')} | "
            f"{_fmt_per_allele(f_)} |"
        )

    lines += [
        "",
        "Pooled ρ on held-out alleles is inflated by between-allele differences in "
        "mean stability; the median per-allele ρ is the honest pan-specific number.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    rep = run_head_to_head()
    print(render_markdown(rep))
