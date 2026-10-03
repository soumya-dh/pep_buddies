"""
dataset_splits.py - Generate the three canonical evaluation splits:
  1. Random split (easy - standard i.i.d.)
  2. Unseen peptides split (harder - neoantigen generalization)
  3. Unseen HLA alleles split (hardest - pan-specific zero-shot generalization)
"""

import os
import json
import logging
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def verify_no_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, split_type: str):
    """
    Rigorously assert zero data leakage between train, val, and test splits.
    """
    if split_type == "unseen_peptides":
        train_peps = set(train_df["peptide"])
        val_peps = set(val_df["peptide"])
        test_peps = set(test_df["peptide"])
        
        leak_train_val = train_peps.intersection(val_peps)
        leak_train_test = train_peps.intersection(test_peps)
        leak_val_test = val_peps.intersection(test_peps)
        
        assert len(leak_train_val) == 0, f"Leakage detected between Train and Val peptides: {len(leak_train_val)}"
        assert len(leak_train_test) == 0, f"Leakage detected between Train and Test peptides: {len(leak_train_test)}"
        assert len(leak_val_test) == 0, f"Leakage detected between Val and Test peptides: {len(leak_val_test)}"
        logger.info("✓ Zero leakage verified for UNSEEN PEPTIDES split.")

    elif split_type == "unseen_alleles":
        train_alleles = set(train_df["allele"])
        val_alleles = set(val_df["allele"])
        test_alleles = set(test_df["allele"])
        
        leak_train_val = train_alleles.intersection(val_alleles)
        leak_train_test = train_alleles.intersection(test_alleles)
        leak_val_test = val_alleles.intersection(test_alleles)
        
        assert len(leak_train_val) == 0, f"Leakage detected between Train and Val alleles: {len(leak_train_val)}"
        assert len(leak_train_test) == 0, f"Leakage detected between Train and Test alleles: {len(leak_train_test)}"
        assert len(leak_val_test) == 0, f"Leakage detected between Val and Test alleles: {len(leak_val_test)}"
        logger.info("✓ Zero leakage verified for UNSEEN HLA ALLELES split.")


def get_split_stats(name: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate summary statistics for a dataset split."""
    return {
        "split_name": name,
        "num_samples": int(len(df)),
        "num_alleles": int(df["allele"].nunique()),
        "num_peptides": int(df["peptide"].nunique()),
        "median_thalf_h": float(round(df["thalf_hours"].median(), 2)),
        "mean_thalf_h": float(round(df["thalf_hours"].mean(), 2)),
        "stable_count": int(df["is_stable"].sum()) if "is_stable" in df.columns else 0,
        "stable_pct": float(round(df["is_stable"].mean() * 100, 2)) if "is_stable" in df.columns else 0.0
    }


def make_random_split(
    df: pd.DataFrame,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Standard random split (easy)."""
    assert abs((train_frac + val_frac + test_frac) - 1.0) < 1e-5
    
    # Stratify by binned stability to ensure balanced stability distributions
    bins = [0, 0.5, 2.0, 10.0, float("inf")]
    df_temp = df.copy()
    df_temp["stab_bin"] = pd.cut(df_temp["thalf_hours"], bins=bins, labels=False, include_lowest=True)
    
    train_df, rest_df = train_test_split(
        df_temp,
        test_size=(val_frac + test_frac),
        random_state=seed,
        stratify=df_temp["stab_bin"]
    )
    
    rel_test_size = test_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        rest_df,
        test_size=rel_test_size,
        random_state=seed,
        stratify=rest_df["stab_bin"]
    )
    
    train_df = train_df.drop(columns=["stab_bin"]).copy()
    val_df = val_df.drop(columns=["stab_bin"]).copy()
    test_df = test_df.drop(columns=["stab_bin"]).copy()
    
    return train_df, val_df, test_df


def make_unseen_peptides_split(
    df: pd.DataFrame,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by peptide group so that test and validation peptides are completely unseen during training.
    """
    unique_peps = np.array(df["peptide"].unique())
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_peps)
    
    n_total = len(unique_peps)
    n_train = int(round(train_frac * n_total))
    n_val = int(round(val_frac * n_total))
    
    train_peps = set(unique_peps[:n_train])
    val_peps = set(unique_peps[n_train:n_train + n_val])
    test_peps = set(unique_peps[n_train + n_val:])
    
    train_df = df[df["peptide"].isin(train_peps)].copy()
    val_df = df[df["peptide"].isin(val_peps)].copy()
    test_df = df[df["peptide"].isin(test_peps)].copy()
    
    verify_no_leakage(train_df, val_df, test_df, "unseen_peptides")
    return train_df, val_df, test_df


def make_unseen_alleles_split(
    df: pd.DataFrame,
    test_allele_count: int = 8,
    val_allele_count: int = 7,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by HLA allele group so that test and validation alleles are completely unseen during training.
    This simulates zero-shot pan-specific generalization to novel patient HLA types.
    """
    # Group alleles and sample counts
    allele_counts = df["allele"].value_counts().to_dict()
    all_alleles = sorted(list(allele_counts.keys()))
    
    rng = np.random.RandomState(seed)
    
    # Partition alleles ensuring both HLA-A and HLA-B are represented in val/test
    a_alleles = [a for a in all_alleles if "HLA-A" in a or a.startswith("A")]
    b_alleles = [a for a in all_alleles if "HLA-B" in a or a.startswith("B")]
    c_alleles = [a for a in all_alleles if "HLA-C" in a or a.startswith("C")]
    
    rng.shuffle(a_alleles)
    rng.shuffle(b_alleles)
    rng.shuffle(c_alleles)
    
    # Distribute alleles across Test and Val
    # For test (8 alleles): 4 from A, 4 from B
    test_alleles = set(a_alleles[:4] + b_alleles[:4])
    rem_a = a_alleles[4:]
    rem_b = b_alleles[4:]
    
    # For val (7 alleles): 3 from A, 4 from B
    val_alleles = set(rem_a[:3] + rem_b[:4])
    
    # Remaining alleles for training
    train_alleles = set(rem_a[3:] + rem_b[4:] + c_alleles)
    
    logger.info(f"Unseen Alleles Split - Test Alleles ({len(test_alleles)}): {sorted(list(test_alleles))}")
    logger.info(f"Unseen Alleles Split - Val Alleles ({len(val_alleles)}): {sorted(list(val_alleles))}")
    logger.info(f"Unseen Alleles Split - Train Alleles: {len(train_alleles)} alleles")
    
    train_df = df[df["allele"].isin(train_alleles)].copy()
    val_df = df[df["allele"].isin(val_alleles)].copy()
    test_df = df[df["allele"].isin(test_alleles)].copy()
    
    verify_no_leakage(train_df, val_df, test_df, "unseen_alleles")
    return train_df, val_df, test_df


def generate_all_splits(
    curated_csv_path: str,
    output_dir: str = "data/splits",
    seed: int = 42
) -> Dict[str, Any]:
    """
    Generate all 3 splits and export train.csv, val.csv, test.csv for each.
    """
    df = pd.read_csv(curated_csv_path)
    logger.info(f"Loaded {len(df)} curated records for split generation.")
    
    splits_config = {
        "random": make_random_split(df, seed=seed),
        "unseen_peptides": make_unseen_peptides_split(df, seed=seed),
        "unseen_alleles": make_unseen_alleles_split(df, seed=seed)
    }
    
    full_report = {}
    
    for split_type, (train_df, val_df, test_df) in splits_config.items():
        split_path = os.path.join(output_dir, split_type)
        os.makedirs(split_path, exist_ok=True)
        
        train_file = os.path.join(split_path, "train.csv")
        val_file = os.path.join(split_path, "val.csv")
        test_file = os.path.join(split_path, "test.csv")
        
        train_df.to_csv(train_file, index=False)
        val_df.to_csv(val_file, index=False)
        test_df.to_csv(test_file, index=False)
        
        stats = {
            "train": get_split_stats("train", train_df),
            "val": get_split_stats("val", val_df),
            "test": get_split_stats("test", test_df),
            "total_samples": len(df)
        }
        
        if split_type == "unseen_alleles":
            stats["test_alleles"] = sorted(list(test_df["allele"].unique()))
            stats["val_alleles"] = sorted(list(val_df["allele"].unique()))
            
        full_report[split_type] = stats
        logger.info(f"Saved {split_type} splits to {split_path}/ (Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)})")
        
    report_file = os.path.join(output_dir, "splits_summary.json")
    with open(report_file, "w") as f:
        json.dump(full_report, f, indent=2)
        
    logger.info(f"Split generation completed! Summary written to {report_file}")
    return full_report


if __name__ == "__main__":
    summary = generate_all_splits("data/processed/cleaned_stability_data.csv")
    print(json.dumps(summary, indent=2))
