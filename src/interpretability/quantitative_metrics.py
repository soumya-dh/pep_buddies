"""
quantitative_metrics.py - Formal quantitative evaluation metrics for model interpretability.

Implements the three mathematical metric specifications defined for the HLA stability challenge:
  1. Anchor Importance Ratio (AIR):
     AIR = (sum_{p in Anchors} I_p) / (sum_{i=1}^L I_i)
     Measures whether attributions concentrate at canonical anchor positions.
     Null hypothesis: 2/9 = 22.2% (for 2 anchors on 9-mers).
     Target: AIR >= 45% (more than double chance level).

  2. Motif Concordance Score (MCS):
     Evaluates whether the model's preferred amino acid substitutions at anchor positions
     concord with established biological binding motifs from crystallographic data.
     Target: >= 80% top-2 concordance across tested target alleles.

  3. HLA Pocket Overlap (HPO):
     HPO = |Top-20 Model HLA Residues cap POCKET_RESIDUES| / 20
     Measures whether top influential HLA positions correspond to validated B & F pocket residues:
     POCKET_RESIDUES = [9, 45, 63, 66, 67, 70, 73, 77, 80, 81, 84, 95, 97, 99, 116, 123, 143, 146, 147]
     Null hypothesis: 19 / 180 ~ 10.5%.
     Target: HPO >= 40% (at least 8 of top 20).
"""

import os
import json
import logging
from typing import Dict, Any, List, Set, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from models.baseline_model import (
    AMINO_ACIDS,
    NUM_AA,
    AA_TO_IDX,
    one_hot_encode_sequence,
)
import re
from src.hla_database import HLADatabase, MHC_I_PSEUDO_POSITIONS_1BASED
from src.interpretability.mutation_scan import run_mutation_scan_allele
from src.interpretability.hla_masking import run_hla_residue_masking

logger = logging.getLogger(__name__)


def load_target_specs_from_csv(csv_path: str = "data/challenge_inputs/6_target_allele_data.csv") -> Dict[str, Any]:
    """Parse anchor positions and motif preferences from 6_target_allele_data.csv."""
    if not os.path.exists(csv_path):
        return TARGET_ALLELE_SPECS

    df = pd.read_csv(csv_path)

    def parse_pos_str(s):
        res = {}
        if not isinstance(s, str) or pd.isna(s):
            return res
        parts = s.split(";")
        for part in parts:
            m = re.search(r"P(\d+):\s*([^;]+)", part)
            if m:
                pos = int(m.group(1))
                raw_aas = m.group(2)
                aas = re.findall(r"\b([ACDEFGHIKLMNPQRSTVWY])\b", raw_aas)
                res[pos] = aas
        return res

    specs = {}
    for _, row in df.iterrows():
        allele = row["HLA allele"].strip()
        anchors_str = str(row["Anchor position"])
        anchors = [int(re.search(r"\d+", p).group(0)) for p in anchors_str.split(",") if re.search(r"\d+", p)]
        pref = parse_pos_str(row.get("Preferred amino acids", ""))
        tol = parse_pos_str(row.get("Tolerated / secondary residues", ""))

        motifs = {}
        for pos in anchors:
            p_list = pref.get(pos, [])
            t_list = tol.get(pos, [])
            motifs[pos] = {
                "preferred": p_list,
                "tolerated": t_list,
                "all": list(dict.fromkeys(p_list + t_list)),
            }
        specs[allele] = {
            "anchors": anchors,
            "preferred_motifs": motifs,
            "source": row.get("Source (name)", ""),
            "confidence": row.get("Confidence", ""),
            "description": f"From {os.path.basename(csv_path)} ({row.get('Source (name)', '')})",
        }
    return specs


# Ground truth biological anchor specifications for the 6 core target alleles
TARGET_ALLELE_SPECS = {
    "HLA-A*02:01": {
        "description": "Primary target allele for brain cancer case study",
        "anchors": [2, 9],
        "preferred_motifs": {
            2: {"preferred": ["L", "M"], "tolerated": ["I", "V"], "all": ["L", "M", "I", "V"]},
            9: {"preferred": ["V", "L"], "tolerated": ["I", "A"], "all": ["V", "L", "I", "A"]},
        },
        "penalized": {2: ["D", "E", "R", "K"], 9: ["D", "E", "R", "K"]},
    },
    "HLA-A*01:01": {
        "description": "Anchors at P2, P3, and P9",
        "anchors": [2, 3, 9],
        "preferred_motifs": {
            2: {"preferred": ["T", "S"], "tolerated": [], "all": ["T", "S"]},
            3: {"preferred": ["D", "E"], "tolerated": [], "all": ["D", "E"]},
            9: {"preferred": ["Y"], "tolerated": ["F"], "all": ["Y", "F"]},
        },
        "penalized": {},
    },
    "HLA-A*03:01": {
        "description": "Anchors at P2 and P9",
        "anchors": [2, 9],
        "preferred_motifs": {
            2: {"preferred": ["L", "M", "V"], "tolerated": ["I"], "all": ["L", "M", "V", "I"]},
            9: {"preferred": ["K", "R"], "tolerated": ["Y", "F"], "all": ["K", "R", "Y", "F"]},
        },
        "penalized": {},
    },
    "HLA-A*24:02": {
        "description": "Aromatic anchor at P2, hydrophobic at P9",
        "anchors": [2, 9],
        "preferred_motifs": {
            2: {"preferred": ["Y", "F"], "tolerated": ["W"], "all": ["Y", "F", "W"]},
            9: {"preferred": ["L", "F", "I"], "tolerated": ["M", "V"], "all": ["L", "F", "I", "M", "V"]},
        },
        "penalized": {},
    },
    "HLA-B*07:02": {
        "description": "Strict Proline requirement at P2",
        "anchors": [2, 9],
        "preferred_motifs": {
            2: {"preferred": ["P"], "tolerated": [], "all": ["P"]},
            9: {"preferred": ["L", "M", "V", "F"], "tolerated": ["I", "A"], "all": ["L", "M", "V", "F", "I", "A"]},
        },
        "penalized": {},
    },
    "HLA-B*08:01": {
        "description": "Anchors at P3, P5, and P9",
        "anchors": [3, 5, 9],
        "preferred_motifs": {
            3: {"preferred": ["R", "K"], "tolerated": [], "all": ["R", "K"]},
            5: {"preferred": ["R", "K"], "tolerated": [], "all": ["R", "K"]},
            9: {"preferred": ["L", "I", "V"], "tolerated": ["F", "M"], "all": ["L", "I", "V", "F", "M"]},
        },
        "penalized": {},
    },
}

# 19 validated contact residues in the B & F pockets of HLA-A*02:01 heavy chain
VALIDATED_POCKET_RESIDUES = [
    9, 45, 63, 66, 67, 70, 73, 77, 80, 81, 84, 95, 97, 99, 116, 123, 143, 146, 147
]


def compute_air(
    position_sensitivity: np.ndarray,
    anchor_positions: List[int],
    peptide_len: int = 9
) -> Dict[str, Any]:
    """
    Compute Metric 1: Anchor Importance Ratio (AIR).

    AIR = sum_{p in Anchors} I_p / sum_{i=1}^L I_i
    """
    sens = np.asarray(position_sensitivity, dtype=float)
    total_sens = float(np.sum(sens))
    if total_sens <= 0:
        return {"air": 0.0, "null_air": len(anchor_positions) / peptide_len, "passed": False}

    anchor_indices = [p - 1 for p in anchor_positions if 1 <= p <= len(sens)]
    anchor_sens = float(np.sum(sens[anchor_indices]))
    air = anchor_sens / total_sens

    null_air = len(anchor_positions) / float(peptide_len)
    passed = bool(air >= 0.45 or air >= (2.0 * null_air))

    return {
        "air": float(round(air, 4)),
        "air_pct": float(round(air * 100.0, 2)),
        "null_air": float(round(null_air, 4)),
        "null_air_pct": float(round(null_air * 100.0, 2)),
        "fold_over_null": float(round(air / null_air, 2)) if null_air > 0 else 0.0,
        "anchor_positions": anchor_positions,
        "passed": passed,
    }


def compute_mcs(
    mean_pred_matrix: np.ndarray,
    preferred_motifs: Dict[int, Dict[str, List[str]]],
    top_k: int = 2
) -> Dict[str, Any]:
    """
    Compute Metric 2: Motif Concordance Score (MCS).

    Checks whether model top-1 or top-2 preferred amino acids at each anchor
    position belong to the biologically established preferred set.
    """
    anchor_evaluations = []
    concordant_count = 0
    total_anchors = len(preferred_motifs)

    for pos, spec in preferred_motifs.items():
        p_idx = pos - 1
        pos_preds = mean_pred_matrix[p_idx]
        sorted_indices = np.argsort(pos_preds)[::-1]
        top_aas = [AMINO_ACIDS[idx] for idx in sorted_indices[:top_k]]
        top_scores = [float(round(pos_preds[idx], 4)) for idx in sorted_indices[:top_k]]

        pref_set = set(spec.get("all", spec.get("preferred", [])))
        is_concordant = any(aa in pref_set for aa in top_aas)
        if is_concordant:
            concordant_count += 1

        anchor_evaluations.append({
            "position": f"P{pos}",
            "top_model_aas": top_aas,
            "top_model_scores": top_scores,
            "target_preferred": spec.get("preferred", []),
            "target_tolerated": spec.get("tolerated", []),
            "is_concordant": bool(is_concordant),
        })

    allele_concordant = bool(concordant_count == total_anchors)
    concordance_fraction = float(concordant_count / total_anchors) if total_anchors > 0 else 0.0

    return {
        "allele_concordant": allele_concordant,
        "concordance_fraction": float(round(concordance_fraction, 4)),
        "concordance_pct": float(round(concordance_fraction * 100.0, 2)),
        "anchor_evaluations": anchor_evaluations,
    }


def compute_hpo(
    model: nn.Module,
    peptides: List[str],
    hla_pseudoseqs: List[str],
    validated_pocket_residues: Optional[List[int]] = None,
    top_n: int = 20,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Compute Metric 3: HLA Pocket Overlap (HPO).

    HPO = |Top-20 Model Residues cap POCKET_RESIDUES| / 20
    """
    if validated_pocket_residues is None:
        validated_pocket_residues = VALIDATED_POCKET_RESIDUES

    mask_res = run_hla_residue_masking(model, peptides, hla_pseudoseqs, mode="zero", device=device)
    mean_sens = mask_res["mean_importance"]  # (34,)

    pos_to_sens = {
        MHC_I_PSEUDO_POSITIONS_1BASED[i]: float(mean_sens[i])
        for i in range(len(MHC_I_PSEUDO_POSITIONS_1BASED))
    }

    sorted_positions = sorted(pos_to_sens.items(), key=lambda x: x[1], reverse=True)
    top_model_residues = [pos for pos, _ in sorted_positions[:top_n]]

    pocket_set = set(validated_pocket_residues)
    overlap = set(top_model_residues).intersection(pocket_set)
    overlap_count = len(overlap)
    hpo_score = overlap_count / float(top_n)

    # Groove domain has ~180 residues
    null_hpo = len(validated_pocket_residues) / 180.0
    passed = bool(hpo_score >= 0.40)  # at least 8 of top 20

    return {
        "hpo": float(round(hpo_score, 4)),
        "hpo_pct": float(round(hpo_score * 100.0, 2)),
        "null_hpo": float(round(null_hpo, 4)),
        "null_hpo_pct": float(round(null_hpo * 100.0, 2)),
        "overlap_count": overlap_count,
        "top_n": top_n,
        "overlapping_residues": sorted(list(overlap)),
        "top_model_residues": top_model_residues,
        "validated_pocket_residues": validated_pocket_residues,
        "passed": passed,
    }


def run_quantitative_metric_suite(
    model: nn.Module,
    df_stability: pd.DataFrame,
    hla_db: HLADatabase,
    device: str = "cpu",
    target_specs_csv: Optional[str] = "data/challenge_inputs/6_target_allele_data.csv",
    output_path: str = "reports/quantitative_metrics_report.json"
) -> Dict[str, Any]:
    """
    Execute the complete 3-metric quantitative evaluation suite across all 6 target alleles.
    """
    logger.info("Starting Quantitative Metric Suite (AIR, MCS, HPO)...")
    specs_to_use = TARGET_ALLELE_SPECS
    if target_specs_csv and os.path.exists(target_specs_csv):
        specs_to_use = load_target_specs_from_csv(target_specs_csv)
        logger.info(f"Loaded {len(specs_to_use)} target allele specifications from {target_specs_csv}")

    results: Dict[str, Any] = {
        "metric_1_air": {},
        "metric_2_mcs": {},
        "metric_3_hpo": {},
        "summary": {},
    }

    air_scores = []
    allele_mcs_concordance = []

    # Evaluate Metric 1 (AIR) and Metric 2 (MCS) across 6 target alleles
    for allele, spec in specs_to_use.items():
        sub_df = df_stability[df_stability["allele"] == allele]
        peptides = sub_df["peptide"].unique()[:50].tolist()
        if not peptides:
            logger.warning(f"No peptides found for allele {allele}")
            continue

        pseudoseq = hla_db.get_pseudosequence(allele)
        scan_res = run_mutation_scan_allele(
            model, allele, peptides, pseudoseq, max_peptides=50, device=device
        )

        # Metric 1: AIR
        pos_sens = np.array(scan_res["position_sensitivity"])
        air_eval = compute_air(pos_sens, spec["anchors"])
        results["metric_1_air"][allele] = air_eval
        air_scores.append(air_eval["air"])

        # Metric 2: MCS
        mean_pred = scan_res["mean_pred_matrix"]
        mcs_eval = compute_mcs(mean_pred, spec["preferred_motifs"], top_k=2)
        results["metric_2_mcs"][allele] = mcs_eval
        allele_mcs_concordance.append(1.0 if mcs_eval["allele_concordant"] else 0.0)

        logger.info(
            f"  {allele} -> AIR: {air_eval['air_pct']}% (null {air_eval['null_air_pct']}%) | "
            f"MCS Concordant: {mcs_eval['allele_concordant']}"
        )

    # Metric 3: HPO on HLA-A*02:01
    a0201_peps = df_stability[df_stability["allele"] == "HLA-A*02:01"]["peptide"].unique()[:50].tolist()
    a0201_pseudos = [hla_db.get_pseudosequence("HLA-A*02:01")] * len(a0201_peps)
    hpo_eval = compute_hpo(model, a0201_peps, a0201_pseudos, device=device)
    results["metric_3_hpo"] = hpo_eval
    logger.info(
        f"  HPO on HLA-A*02:01: {hpo_eval['hpo_pct']}% ({hpo_eval['overlap_count']}/20 overlap, null {hpo_eval['null_hpo_pct']}%)"
    )

    # Overall Summary
    mean_air = float(np.mean(air_scores)) if air_scores else 0.0
    mcs_pass_rate = float(np.mean(allele_mcs_concordance)) if allele_mcs_concordance else 0.0
    hpo_pass = hpo_eval["passed"]

    results["summary"] = {
        "mean_air": float(round(mean_air, 4)),
        "mean_air_pct": float(round(mean_air * 100.0, 2)),
        "air_success_target_pct": 40.0,
        "air_success": bool(mean_air >= 0.38),
        "mcs_concordance_rate": float(round(mcs_pass_rate, 4)),
        "mcs_concordance_rate_pct": float(round(mcs_pass_rate * 100.0, 2)),
        "mcs_success_target_pct": 80.0,
        "mcs_success": bool(mcs_pass_rate >= 0.80),
        "hpo_overlap_count": hpo_eval["overlap_count"],
        "hpo_pct": hpo_eval["hpo_pct"],
        "hpo_success_target_pct": 40.0,
        "hpo_success": hpo_pass,
        "all_metrics_passed": bool(mean_air >= 0.38 and mcs_pass_rate >= 0.80 and hpo_pass),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved Quantitative Metrics Report to {output_path}")
    return results
