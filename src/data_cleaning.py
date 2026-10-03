"""
data_cleaning.py - Data cleaning, allele normalization, and deduplication for Peptide-HLA stability data.
"""

import re
import logging
from typing import Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Standard 20 amino acid alphabet
CANONICAL_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

# Common column name mappings
COLUMN_ALIASES = {
    "allele": ["allele", "hla", "hla_allele", "mhc", "mhc_allele", "gene", "hla_name"],
    "peptide": ["peptide", "pep", "sequence", "peptide_sequence", "pep_seq", "epitope"],
    "thalf_hours": ["thalf_hours", "thalf", "stability", "half_life", "half_life_hours", "stability_value", "t_half", "thalf_h", "val", "target"],
    "hla_seq": ["hla_seq", "hla_sequence", "mhc_seq", "mhc_sequence", "full_seq"],
    "hla_pseudoseq": ["hla_pseudoseq", "pseudoseq", "pseudo_sequence", "mhc_pseudo", "pocket_seq"]
}


def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Automatically detect standard column names from DataFrame columns."""
    detected = {}
    lower_cols = {col.lower().strip().replace(" ", "_"): col for col in df.columns}
    
    for standard_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lower_cols:
                detected[standard_name] = lower_cols[alias]
                break
                
    if "allele" not in detected:
        raise ValueError(f"Could not automatically detect HLA allele column in: {list(df.columns)}")
    if "peptide" not in detected:
        raise ValueError(f"Could not automatically detect peptide column in: {list(df.columns)}")
    if "thalf_hours" not in detected:
        raise ValueError(f"Could not automatically detect stability/half-life column in: {list(df.columns)}")
        
    return detected


def normalize_allele_name(allele: str, output_format: str = "imgt") -> str:
    """
    Standardize HLA allele naming across various common representations.
    
    Supports:
      - IMGT format (default): 'HLA-A*02:01', 'HLA-B*07:02', 'HLA-C*07:01'
      - NetMHC format: 'HLA-A02:01', 'HLA-B07:02', 'HLA-C07:01'
      - Short / unpunctuated format: 'A0201', 'B0702', 'C0701'
      - Shorthand with star: 'A*02:01', 'B*07:02'
      - Shorthand with colon only: 'HLA-A02:01'
    
    Parameters
    ----------
    allele : str
        Input allele string (e.g. 'A0201', 'HLA-A*02:01', 'HLA-A0201', 'A*0201')
    output_format : str
        'imgt' (e.g. HLA-A*02:01), 'netmhc' (e.g. HLA-A02:01), or 'short' (e.g. A0201)
        
    Returns
    -------
    str : Normalized allele string
    """
    if not isinstance(allele, str):
        allele = str(allele)
    allele = allele.strip().upper()
    
    # Strip leading 'HLA-' if present
    clean = re.sub(r"^HLA[-_]?", "", allele)
    
    # Match patterns like:
    # A*02:01, A*0201, A02:01, A0201, A*02:01:01, A*02:01:01:01
    pattern = r"^([ABC])[\*]?(\d{2})[:]?(\d{2})(?:[:]?\d{2})*(?:[A-Z])?$"
    match = re.match(pattern, clean)
    
    if match:
        gene = match.group(1)
        group = match.group(2)
        protein = match.group(3)
        
        if output_format == "imgt":
            return f"HLA-{gene}*{group}:{protein}"
        elif output_format == "netmhc":
            return f"HLA-{gene}{group}:{protein}"
        elif output_format == "short":
            return f"{gene}{group}{protein}"
        else:
            raise ValueError(f"Unknown output_format: {output_format}")
    
    # Check if already in standard form HLA-X*XX:XX
    imgt_match = re.match(r"^HLA-([ABC])\*(\d{2}):(\d{2})$", allele)
    if imgt_match:
        gene, group, protein = imgt_match.groups()
        if output_format == "imgt":
            return f"HLA-{gene}*{group}:{protein}"
        elif output_format == "netmhc":
            return f"HLA-{gene}{group}:{protein}"
        elif output_format == "short":
            return f"{gene}{group}{protein}"
            
    # If not matching A, B, C standard format, return cleaned uppercase
    logger.warning(f"Unrecognized allele format: '{allele}', keeping original sanitized.")
    return allele


def validate_peptide(seq: str, min_len: int = 8, max_len: int = 15) -> Tuple[bool, str]:
    """
    Validate peptide sequence.
    
    Parameters
    ----------
    seq : str
        Peptide sequence
    min_len : int
        Minimum allowed amino acid length
    max_len : int
        Maximum allowed amino acid length
        
    Returns
    -------
    Tuple[bool, str] : (is_valid, sanitized_seq_or_error)
    """
    if not isinstance(seq, str):
        return False, "Not a string"
    seq = seq.strip().upper()
    
    if len(seq) < min_len or len(seq) > max_len:
        return False, f"Length {len(seq)} outside range [{min_len}, {max_len}]"
        
    invalid_chars = set(seq) - CANONICAL_AMINO_ACIDS
    if invalid_chars:
        return False, f"Non-canonical amino acid(s): {invalid_chars}"
        
    return True, seq


def stability_to_score(thalf_hours: float, t0: float = 5.0) -> float:
    """
    Convert half-life (thalf in hours) to normalized stability score in [0, 1]
    using standard NetMHCstabpan logistic transformation:
    S = 1 / (1 + (t0 / thalf_hours))
    
    Parameters
    ----------
    thalf_hours : float
        Half life in hours
    t0 : float
        Half-life scaling constant (default 5.0 hours, as in Rasmussen et al.)
        
    Returns
    -------
    float : Stability score between 0.0 and 1.0
    """
    if thalf_hours <= 0.0 or np.isnan(thalf_hours):
        return 0.0
    return float(1.0 / (1.0 + (t0 / thalf_hours)))


def score_to_stability(score: float, t0: float = 5.0) -> float:
    """Invert normalized stability score back to estimated half-life in hours."""
    if score <= 0.0:
        return 0.0
    if score >= 1.0:
        return 999.0
    return float(t0 * (score / (1.0 - score)))


def clean_and_curate_data(
    input_path: str,
    output_path: Optional[str] = None,
    dedup_strategy: str = "median",
    min_peptide_len: int = 8,
    max_peptide_len: int = 15,
    stability_threshold_hours: float = 2.0
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean, normalize, and curate peptide-HLA stability dataset.
    
    Steps:
      1. Parse CSV and map columns.
      2. Normalize allele naming to IMGT standard (e.g. HLA-A*02:01) and NetMHC format (HLA-A02:01).
      3. Clean and validate peptide sequences (valid canonical amino acids, length checks).
      4. Handle negative / NaN stability values.
      5. Deduplicate (peptide, allele) pairs using median/mean.
      6. Calculate log stability and continuous stability score [0, 1].
      7. Assign binary classification flag `is_stable` (thalf >= stability_threshold_hours).
    
    Parameters
    ----------
    input_path : str
        Path to raw CSV file.
    output_path : str, optional
        Path to save curated CSV file.
    dedup_strategy : str
        Strategy for duplicate (allele, peptide) resolution ('median', 'mean', 'first').
    min_peptide_len : int
        Minimum peptide length.
    max_peptide_len : int
        Maximum peptide length.
    stability_threshold_hours : float
        Half-life cutoff in hours for stable binders (default 2.0h).
        
    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]] : Cleaned DataFrame and curation statistics summary.
    """
    logger.info(f"Loading raw dataset from: {input_path}")
    df_raw = pd.read_csv(input_path)
    initial_rows = len(df_raw)
    
    cols = detect_columns(df_raw)
    logger.info(f"Detected columns: {cols}")
    
    # Extract working dataframe
    df = pd.DataFrame()
    df["allele_raw"] = df_raw[cols["allele"]].astype(str)
    df["peptide_raw"] = df_raw[cols["peptide"]].astype(str)
    df["thalf_hours"] = pd.to_numeric(df_raw[cols["thalf_hours"]], errors="coerce")
    
    # Carry forward sequence columns if already present in input
    if "hla_seq" in cols:
        df["hla_seq"] = df_raw[cols["hla_seq"]]
    if "hla_pseudoseq" in cols:
        df["hla_pseudoseq"] = df_raw[cols["hla_pseudoseq"]]
        
    # 1. Normalize alleles
    df["allele"] = df["allele_raw"].apply(lambda a: normalize_allele_name(a, output_format="imgt"))
    df["allele_netmhc"] = df["allele_raw"].apply(lambda a: normalize_allele_name(a, output_format="netmhc"))
    df["allele_short"] = df["allele_raw"].apply(lambda a: normalize_allele_name(a, output_format="short"))
    
    # 2. Validate peptides
    valid_peps = []
    dropped_peptides = 0
    for pep in df["peptide_raw"]:
        ok, res = validate_peptide(pep, min_len=min_peptide_len, max_len=max_peptide_len)
        if ok:
            valid_peps.append(res)
        else:
            valid_peps.append(None)
            dropped_peptides += 1
            
    df["peptide"] = valid_peps
    df = df.dropna(subset=["peptide"]).copy()
    
    # 3. Clean stability values
    df["thalf_hours"] = df["thalf_hours"].clip(lower=0.0)
    dropped_nans = df["thalf_hours"].isnull().sum()
    df = df.dropna(subset=["thalf_hours"]).copy()
    
    # 4. Deduplicate (allele, peptide) pairs
    raw_pairs = len(df)
    dup_mask = df.duplicated(subset=["allele", "peptide"], keep=False)
    num_duplicates = dup_mask.sum()
    
    if num_duplicates > 0:
        logger.info(f"Resolving {num_duplicates} duplicate measurements using '{dedup_strategy}' strategy...")
        # Groupby and aggregate
        agg_funcs = {
            "thalf_hours": dedup_strategy,
            "allele_raw": "first",
            "allele_netmhc": "first",
            "allele_short": "first"
        }
        if "hla_seq" in df.columns:
            agg_funcs["hla_seq"] = "first"
        if "hla_pseudoseq" in df.columns:
            agg_funcs["hla_pseudoseq"] = "first"
            
        df = df.groupby(["allele", "peptide"], as_index=False).agg(agg_funcs)
    
    final_rows = len(df)
    
    # 5. Calculate derivative metrics: stability score [0, 1] and log10(thalf + 1)
    df["stability_score"] = df["thalf_hours"].apply(stability_to_score)
    df["log_thalf"] = np.log10(df["thalf_hours"] + 1.0)
    df["is_stable"] = (df["thalf_hours"] >= stability_threshold_hours).astype(int)
    
    # Reorder columns cleanly
    preferred_order = [
        "allele", "allele_netmhc", "allele_short", "peptide",
        "thalf_hours", "stability_score", "log_thalf", "is_stable"
    ]
    if "hla_seq" in df.columns:
        preferred_order.append("hla_seq")
    if "hla_pseudoseq" in df.columns:
        preferred_order.append("hla_pseudoseq")
        
    df = df[preferred_order]
    
    stats = {
        "initial_rows": initial_rows,
        "dropped_peptides": dropped_peptides,
        "dropped_nans": int(dropped_nans),
        "duplicate_measurements_resolved": int(num_duplicates),
        "final_curated_rows": final_rows,
        "unique_alleles": int(df["allele"].nunique()),
        "unique_peptides": int(df["peptide"].nunique()),
        "min_thalf_hours": float(df["thalf_hours"].min()),
        "median_thalf_hours": float(df["thalf_hours"].median()),
        "mean_thalf_hours": float(df["thalf_hours"].mean()),
        "max_thalf_hours": float(df["thalf_hours"].max()),
        "stable_count": int(df["is_stable"].sum()),
        "stable_fraction": float(df["is_stable"].mean())
    }
    
    if output_path:
        df.to_csv(output_path, index=False)
        logger.info(f"Curated dataset successfully saved to: {output_path} ({final_rows} rows)")
        
    return df, stats


if __name__ == "__main__":
    import json
    input_file = "data/raw/rasmussen_et_al_dataset.csv"
    output_file = "data/processed/cleaned_stability_data.csv"
    df_clean, report = clean_and_curate_data(input_file, output_file)
    print("Data cleaning report:")
    print(json.dumps(report, indent=2))
