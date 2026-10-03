"""
calibration_probe.py - How much of the unseen-allele gap is allele-level miscalibration?

Motivation
----------
On the held-out-allele split the one-hot MLP scores a *lower* pooled Spearman
(0.4378 on the NetMHCstabpan subset) than its median per-allele Spearman
(0.5562). That ordering is diagnostic: the model ranks peptides sensibly *within*
an allele but places the alleles at the wrong absolute levels. Pooling then mixes
good within-allele ranking with bad between-allele offsets and the pooled number
comes out worse than either.

If that is the dominant error, a calibration head is a far cheaper fix than
better peptide representations -- so it is worth measuring before committing
compute to foundation-model embeddings.

What this reports
-----------------
A ladder of four scorings of the *same* predictions, differing only in what is
done to each allele's offset on the canonical target scale:

  1. ``as_is``        -- raw predictions, no adjustment. The status quo.
  2. ``centred``      -- subtract each allele's own *predicted* mean. Adds no
                         information; it only removes the model's offsets, so it
                         isolates pooled within-allele ranking quality.
  3. ``knn_offset``   -- replace each allele's predicted mean with one estimated
                         from pseudosequence-similar *training* alleles. Uses no
                         test labels, so this rung is actually deployable
                         zero-shot and is the realistic win.
  4. ``oracle_offset``-- replace it with the allele's *true* test mean. Uses test
                         labels, so it is an UPPER BOUND, not an achievable
                         result. It bounds how much calibration can ever buy.

Plus a residual-variance decomposition (what fraction of squared error is a pure
per-allele offset) and the kNN estimator's own accuracy at predicting allele
means.

Run: ``python -m src.calibration_probe``
"""

import glob
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.evaluate import compute_metrics, compute_per_allele_metrics
from src.targets import TARGET_NAME, thalf_to_target

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

#: Rungs that use test-set labels and are therefore upper bounds, not results.
ORACLE_RUNGS = ("oracle_offset",)


# --------------------------------------------------------------------------- #
# Allele pseudosequence similarity
# --------------------------------------------------------------------------- #

def allele_pseudoseq_map(df: pd.DataFrame) -> Dict[str, str]:
    """Map allele -> 34-mer pseudosequence, asserting one sequence per allele."""
    out: Dict[str, str] = {}
    for allele, grp in df.groupby("allele"):
        seqs = grp["hla_pseudoseq"].dropna().unique()
        if len(seqs) != 1:
            raise ValueError(f"{allele} has {len(seqs)} distinct pseudosequences")
        out[str(allele)] = str(seqs[0])
    return out


def pseudoseq_identity(a: str, b: str) -> float:
    """Fraction of identical positions between two equal-length pseudosequences."""
    if len(a) != len(b):
        raise ValueError(f"pseudosequence length mismatch: {len(a)} vs {len(b)}")
    return float(np.mean([x == y for x, y in zip(a, b)]))


def estimate_allele_mean_knn(
    query_pseudoseq: str,
    train_means: Dict[str, float],
    train_pseudoseqs: Dict[str, str],
    k: int = 5,
    power: float = 4.0,
) -> Tuple[float, List[Tuple[str, float]]]:
    """
    Estimate an unseen allele's mean target from its nearest training alleles.

    Weights are ``identity ** power`` over the ``k`` most similar training
    alleles, which sharpens the average toward the closest pockets. Returns the
    estimate and the neighbours used, so the choice is auditable.
    """
    sims = sorted(
        ((a, pseudoseq_identity(query_pseudoseq, s)) for a, s in train_pseudoseqs.items()),
        key=lambda kv: kv[1],
        reverse=True,
    )[:k]

    weights = np.array([s ** power for _, s in sims], dtype=float)
    if weights.sum() <= 0:
        weights = np.ones(len(sims), dtype=float)
    values = np.array([train_means[a] for a, _ in sims], dtype=float)
    return float(np.average(values, weights=weights)), sims


# --------------------------------------------------------------------------- #
# Error decomposition
# --------------------------------------------------------------------------- #

def residual_decomposition(
    alleles: np.ndarray,
    y_true_target: np.ndarray,
    y_pred_target: np.ndarray,
) -> Dict[str, Any]:
    """
    Split squared error into a between-allele (offset) part and a within-allele part.

    ``offset_fraction`` is the share of mean squared error that a perfect
    per-allele intercept would remove. High values mean the model's problem is
    calibration, not ranking.
    """
    resid = y_pred_target - y_true_target
    mse_total = float(np.mean(resid ** 2))

    per_allele_bias = {}
    within = np.empty_like(resid)
    for allele in np.unique(alleles):
        mask = (alleles == allele)
        bias = float(resid[mask].mean())
        per_allele_bias[str(allele)] = round(bias, 4)
        within[mask] = resid[mask] - bias

    mse_within = float(np.mean(within ** 2))
    mse_between = mse_total - mse_within

    return {
        "mse_total": round(mse_total, 5),
        "mse_between_allele": round(mse_between, 5),
        "mse_within_allele": round(mse_within, 5),
        "offset_fraction": round(mse_between / mse_total, 4) if mse_total > 0 else None,
        "per_allele_bias": per_allele_bias,
        "bias_spread": round(float(np.std(list(per_allele_bias.values()), ddof=1)), 4),
    }


# --------------------------------------------------------------------------- #
# The ladder
# --------------------------------------------------------------------------- #

def _recalibrate(
    alleles: np.ndarray,
    y_pred_target: np.ndarray,
    new_means: Optional[Dict[str, float]],
) -> np.ndarray:
    """
    Shift each allele's predictions to sit at ``new_means[allele]``.

    With ``new_means=None`` the predictions are merely centred per allele (mean
    zero), which adds no information.
    """
    out = y_pred_target.astype(float).copy()
    for allele in np.unique(alleles):
        mask = (alleles == allele)
        out[mask] -= out[mask].mean()
        if new_means is not None:
            out[mask] += new_means[str(allele)]
    return out


def build_ladder(
    preds: pd.DataFrame,
    train_df: pd.DataFrame,
    k: int = 5,
) -> Dict[str, Any]:
    """Score one model's predictions at each rung of the calibration ladder."""
    alleles = preds["allele"].values
    y_true_thalf = preds["thalf_hours"].values
    y_true_t = thalf_to_target(y_true_thalf)
    y_pred_t = thalf_to_target(preds["pred_thalf_hours"].values)

    # Reference quantities from the TRAINING alleles only.
    train_means = {
        str(a): float(thalf_to_target(g["thalf_hours"].values).mean())
        for a, g in train_df.groupby("allele")
    }
    train_pseudo = allele_pseudoseq_map(train_df)
    test_pseudo = allele_pseudoseq_map(preds) if "hla_pseudoseq" in preds.columns else None

    # True test means -- oracle only.
    oracle_means = {
        str(a): float(y_true_t[alleles == a].mean()) for a in np.unique(alleles)
    }

    # kNN-estimated means, from pseudosequence similarity to training alleles.
    knn_means: Dict[str, float] = {}
    knn_detail: Dict[str, Any] = {}
    if test_pseudo:
        for allele, seq in test_pseudo.items():
            est, neigh = estimate_allele_mean_knn(seq, train_means, train_pseudo, k=k)
            knn_means[allele] = est
            knn_detail[allele] = {
                "estimated_mean": round(est, 4),
                "true_mean": round(oracle_means[allele], 4),
                "error": round(est - oracle_means[allele], 4),
                "neighbours": [
                    {"allele": a, "identity": round(s, 4), "train_mean": round(train_means[a], 4)}
                    for a, s in neigh
                ],
            }

    rungs: Dict[str, np.ndarray] = {
        "as_is": y_pred_t,
        "centred": _recalibrate(alleles, y_pred_t, None),
        "oracle_offset": _recalibrate(alleles, y_pred_t, oracle_means),
    }
    if knn_means:
        rungs["knn_offset"] = _recalibrate(alleles, y_pred_t, knn_means)

    order = ["as_is", "centred", "knn_offset", "oracle_offset"]
    scored: Dict[str, Any] = {}
    for name in order:
        if name not in rungs:
            continue
        # Score on the target scale. Re-centring drives many predicted targets
        # negative, and routing those back through hours would clip them at 0 and
        # manufacture ties, which would corrupt the rank metrics.
        m = compute_metrics(y_true_thalf, y_pred_target=rungs[name])
        pa = compute_per_allele_metrics(
            alleles, y_true_thalf, y_pred_target=rungs[name]
        )
        scored[name] = {
            "is_upper_bound": name in ORACLE_RUNGS,
            "spearman_rho": m["spearman_rho"],
            "pearson_r": m["pearson_r"],
            "rmse": m["rmse"],
            "roc_auc": m["roc_auc"],
            "median_per_allele_rho": (
                pa["spearman_rho_across_alleles"]["median"]
                if pa["spearman_rho_across_alleles"] else None
            ),
        }

    return {
        "n_samples": int(len(preds)),
        "n_alleles": int(len(np.unique(alleles))),
        "ladder": scored,
        "decomposition": residual_decomposition(alleles, y_true_t, y_pred_t),
        "knn": {
            "k": k,
            "mae_of_allele_mean_estimate": (
                round(float(np.mean([abs(v["error"]) for v in knn_detail.values()])), 4)
                if knn_detail else None
            ),
            "per_allele": knn_detail,
        },
    }


def _discover(predictions_dir: str) -> Dict[str, List[str]]:
    """Group per-row prediction CSVs by model, collapsing ``_seedN`` suffixes."""
    groups: Dict[str, List[str]] = {}
    for path in sorted(glob.glob(os.path.join(predictions_dir, "*.csv"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        groups.setdefault(re.sub(r"_seed\d+$", "", stem), []).append(path)
    return groups


def _mean_std(vals: List[float]) -> Dict[str, float]:
    arr = np.asarray(vals, dtype=float)
    return {
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std(ddof=1)), 4) if len(arr) > 1 else 0.0,
    }


def run_probe(
    train_csv: str = "data/splits/unseen_alleles/train.csv",
    test_csv: str = "data/splits/unseen_alleles/test.csv",
    predictions_dir: str = "models/predictions/unseen_alleles",
    netmhc_csv: str = "data/netmhcstabpan_benchmark/predictions.csv",
    output_json: str = "reports/calibration_probe.json",
    output_md: Optional[str] = "reports/calibration_probe.md",
    k: int = 5,
) -> Dict[str, Any]:
    """Run the calibration ladder for every saved model, plus NetMHCstabpan."""
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # Pseudosequences for the held-out alleles, needed by the kNN rung.
    pseudo_by_allele = allele_pseudoseq_map(test_df)

    report: Dict[str, Any] = {"target": TARGET_NAME, "k": k, "models": {}}

    for model, paths in _discover(predictions_dir).items():
        runs = []
        for path in paths:
            preds = pd.read_csv(path)
            preds["hla_pseudoseq"] = preds["allele"].map(pseudo_by_allele)
            runs.append(build_ladder(preds, train_df, k=k))

        if len(runs) == 1:
            report["models"][model] = runs[0]
        else:
            # Aggregate the ladder across seeds; the decomposition and kNN detail
            # are reported from the first seed (they barely vary).
            agg: Dict[str, Any] = {
                "n_seeds": len(runs),
                "n_samples": runs[0]["n_samples"],
                "n_alleles": runs[0]["n_alleles"],
                "ladder": {},
                "decomposition": {
                    "offset_fraction": _mean_std(
                        [r["decomposition"]["offset_fraction"] for r in runs]
                    ),
                    "bias_spread": _mean_std(
                        [r["decomposition"]["bias_spread"] for r in runs]
                    ),
                },
                "knn": runs[0]["knn"],
            }
            for rung in runs[0]["ladder"]:
                agg["ladder"][rung] = {
                    "is_upper_bound": rung in ORACLE_RUNGS,
                    **{
                        metric: _mean_std([r["ladder"][rung][metric] for r in runs])
                        for metric in
                        ["spearman_rho", "pearson_r", "rmse", "roc_auc", "median_per_allele_rho"]
                    },
                }
            report["models"][model] = agg

    # NetMHCstabpan as a control: a well-calibrated model should gain little.
    if os.path.exists(netmhc_csv):
        from src.head_to_head import load_head_to_head_index

        idx = load_head_to_head_index(test_csv, netmhc_csv)
        net = idx.rename(columns={"netmhc_thalf_hours": "pred_thalf_hours"})
        net["hla_pseudoseq"] = net["allele"].map(pseudo_by_allele)
        report["models"]["netmhcstabpan"] = build_ladder(net, train_df, k=k)
        report["netmhcstabpan_note"] = (
            "Scored on its 320-pair subset only; other models here cover the full "
            "test set, so compare rung-to-rung gains rather than absolute values."
        )

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote calibration probe to %s", output_json)

    if output_md:
        with open(output_md, "w") as f:
            f.write(render_markdown(report))
        logger.info("Wrote calibration summary to %s", output_md)

    return report


def _cell(entry: Any, key: str) -> str:
    v = entry.get(key)
    if v is None:
        return "n/a"
    if isinstance(v, dict):
        return f"{v['mean']:.4f} ± {v['std']:.4f}"
    return f"{float(v):.4f}"


def render_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Calibration Probe: Unseen-Allele Split",
        "",
        f"Target: `{report['target']}`. kNN k = {report['k']}.",
        "",
        "Each rung re-scores the **same predictions**, changing only each allele's "
        "offset on the target scale.",
        "",
        "| Rung | What it does | Uses test labels? |",
        "| :--- | :--- | :---: |",
        "| `as_is` | raw predictions | no |",
        "| `centred` | removes the model's own allele offsets (adds no information) | no |",
        "| `knn_offset` | allele mean estimated from pseudosequence-similar *training* alleles | **no — deployable** |",
        "| `oracle_offset` | allele mean set to its true test mean | **yes — upper bound only** |",
        "",
    ]

    for model, payload in report["models"].items():
        lad = payload["ladder"]
        lines += [
            f"## {model} (n = {payload['n_samples']}, {payload['n_alleles']} alleles)",
            "",
            "| Rung | Spearman ρ | Pearson r | RMSE | ROC-AUC | Median per-allele ρ |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
        for rung, vals in lad.items():
            tag = " *(bound)*" if vals.get("is_upper_bound") else ""
            lines.append(
                f"| `{rung}`{tag} | {_cell(vals, 'spearman_rho')} | {_cell(vals, 'pearson_r')} | "
                f"{_cell(vals, 'rmse')} | {_cell(vals, 'roc_auc')} | "
                f"{_cell(vals, 'median_per_allele_rho')} |"
            )

        dec = payload["decomposition"]
        off = dec.get("offset_fraction")
        off_s = _cell(dec, "offset_fraction") if isinstance(off, dict) else f"{off:.4f}"
        lines += [
            "",
            f"Share of MSE removable by a perfect per-allele intercept: **{off_s}**. "
            f"Spread of per-allele bias: {_cell(dec, 'bias_spread')}.",
        ]
        knn_mae = payload.get("knn", {}).get("mae_of_allele_mean_estimate")
        if knn_mae is not None:
            lines.append(
                f"kNN error in estimating an allele's mean target: MAE {knn_mae:.4f}."
            )
        lines.append("")

    if "netmhcstabpan_note" in report:
        lines += [report["netmhcstabpan_note"], ""]
    return "\n".join(lines)


if __name__ == "__main__":
    rep = run_probe()
    print(render_markdown(rep))
