"""
train_heads.py - Train embedding heads on every split and score them canonically.

Produces, per (model, featurisation, split):
  * the canonical metrics from ``src.evaluate.compute_metrics``;
  * the per-allele breakdown (the honest pan-specific number);
  * per-row test predictions, written next to the baselines' own predictions so
    the NetMHCstabpan head-to-head and the paired bootstrap can pick them up
    without retraining anything.

Examples::

    python -m models.train_heads --model esm2-35m --featurisation mean perpos
    python -m models.train_heads --model esm2-35m --split unseen_alleles
"""

import argparse
import json
import logging
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from models.heads import FEATURISATIONS, FeatureBuilder, RidgeHead, load_split, targets_for
from src.evaluate import compute_metrics, compute_per_allele_metrics
from src.targets import TARGET_NAME, target_to_thalf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SPLITS = ("random", "unseen_peptides", "unseen_alleles")


def run_one(
    model_key: str,
    featurisation: str,
    split: str,
    splits_dir: str = "data/splits",
    emb_root: str = "data/embeddings",
    predictions_dir: str = "models/predictions",
) -> Dict[str, Any]:
    """Train and score one (model, featurisation, split) combination."""
    train_df, val_df, test_df = load_split(splits_dir, split)
    fb = FeatureBuilder(model_key, root=emb_root, featurisation=featurisation)

    X_tr, X_va, X_te = (fb.transform(d) for d in (train_df, val_df, test_df))
    y_tr, y_va = targets_for(train_df), targets_for(val_df)

    logger.info(
        "%s | %s | %s: X_train=%s", model_key, featurisation, split, X_tr.shape
    )

    head = RidgeHead().fit(X_tr, y_tr, X_va, y_va)
    y_pred_target = head.predict(X_te)
    y_pred_thalf = target_to_thalf(y_pred_target)
    y_true_thalf = test_df["thalf_hours"].values

    res: Dict[str, Any] = {
        "model_key": model_key,
        "featurisation": featurisation,
        "split": split,
        "features": fb.describe(X_tr.shape[1]),
        "head": head.params,
    }
    res.update(compute_metrics(y_true_thalf, y_pred_thalf))
    res["per_allele"] = compute_per_allele_metrics(
        test_df["allele"].values, y_true_thalf, y_pred_thalf
    )

    # Save per-row predictions in the same schema the baselines use.
    out_dir = os.path.join(predictions_dir, split)
    os.makedirs(out_dir, exist_ok=True)
    name = f"{model_key}_{featurisation}_ridge".replace("/", "__")
    preds = test_df[["allele", "peptide", "thalf_hours"]].copy()
    preds["pred_target"] = y_pred_target
    preds["pred_thalf_hours"] = y_pred_thalf
    preds.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
    res["predictions_csv"] = os.path.join(out_dir, f"{name}.csv")

    if fb.uses_random_weights:
        res["WARNING"] = (
            "Embeddings were built with --random-init; this row validates the "
            "pipeline and is NOT a result."
        )

    logger.info(
        "  -> rho=%.4f  median per-allele rho=%s  rmse=%.4f",
        res["spearman_rho"],
        (res["per_allele"]["spearman_rho_across_alleles"] or {}).get("median"),
        res["rmse"],
    )
    return res


def parse_args():
    p = argparse.ArgumentParser(description="Train ridge heads on frozen embeddings")
    p.add_argument("--model", required=True)
    p.add_argument("--featurisation", nargs="+", default=["mean", "perpos"],
                   choices=list(FEATURISATIONS))
    p.add_argument("--split", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    p.add_argument("--splits-dir", default="data/splits")
    p.add_argument("--emb-root", default="data/embeddings")
    p.add_argument("--out", default="reports/phase2_heads.json")
    return p.parse_args()


def main():
    args = parse_args()
    rows: List[Dict[str, Any]] = []
    for feat in args.featurisation:
        for split in args.split:
            rows.append(
                run_one(
                    args.model, feat, split,
                    splits_dir=args.splits_dir, emb_root=args.emb_root,
                )
            )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    payload = {"target": TARGET_NAME, "results": rows}
    existing: Dict[str, Any] = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            existing = json.load(f)
    merged = {
        (r["model_key"], r["featurisation"], r["split"]): r
        for r in existing.get("results", []) + rows
    }
    payload["results"] = list(merged.values())
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote %d result rows to %s", len(payload["results"]), args.out)

    print(f"\n{'model':<12}{'featurisation':<16}{'split':<18}{'rho':>8}{'medPA':>9}{'rmse':>8}")
    for r in rows:
        med = (r["per_allele"]["spearman_rho_across_alleles"] or {}).get("median")
        print(
            f"{r['model_key']:<12}{r['featurisation']:<16}{r['split']:<18}"
            f"{r['spearman_rho']:>8.4f}{(med if med is not None else float('nan')):>9.4f}"
            f"{r['rmse']:>8.4f}"
        )


if __name__ == "__main__":
    main()
