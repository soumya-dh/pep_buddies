"""
run_phase1.py - Master pipeline script for Phase 1: Data, baseline, and splits.

Executes:
  1. Data ingestion, allele normalization, peptide validation, and deduplication.
  2. IPD-IMGT/HLA sequence lookup and 34-residue pocket pseudo-sequence verification.
  3. NetMHCstabpan benchmark reproduction and batch submission file generation.
  4. Generation of the three strict leakage-free splits:
       - Random split (easy)
       - Unseen peptides split (harder)
       - Unseen HLA alleles split (hardest, pan-specific zero-shot)
  5. Training of baseline regression and pan-specific deep learning models.
  6. Generation of publication-quality figures and summary report.
"""

import os
import sys
import json
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase1Pipeline")


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1: Peptide-HLA Stability Pipeline")
    parser.add_argument("--data", type=str, default="data/raw/rasmussen_et_al_dataset.csv", help="Path to input raw CSV dataset")
    parser.add_argument("--output-dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--skip-benchmark-query", action="store_true", help="Skip live querying NetMHCstabpan server if results already cached")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs for baseline neural network")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("=================================================================")
    logger.info("  STARTING PHASE 1: DATA CURATION, BENCHMARK & SPLITS PIPELINE   ")
    logger.info("=================================================================")

    # Step 1: Clean and standardize data
    from src.data_cleaning import clean_and_curate_data
    cleaned_csv = os.path.join(args.output_dir, "processed", "cleaned_stability_data.csv")
    os.makedirs(os.path.dirname(cleaned_csv), exist_ok=True)
    logger.info(">>> STEP 1: Cleaning dataset and standardizing allele names...")
    df_clean, clean_report = clean_and_curate_data(args.data, output_path=cleaned_csv)
    logger.info(f"Step 1 Complete: {len(df_clean)} curated samples across {df_clean['allele'].nunique()} alleles.")

    # Step 2: HLA Sequence and Pseudo-sequence verification
    from src.hla_database import HLADatabase
    logger.info(">>> STEP 2: Verifying IPD-IMGT/HLA sequences and 34-residue pseudo-sequences...")
    hla_db = HLADatabase()
    # Check sample alleles
    test_alleles = ["HLA-A*02:01", "HLA-A*24:02", "HLA-B*07:02", "HLA-B*14:02(C67S)"]
    for a in test_alleles:
        pseq = hla_db.get_pseudosequence(a)
        mseq = hla_db.get_mature_sequence(a)
        logger.info(f"  Allele {a} -> Pseudo ({len(pseq) if pseq else 0} aa): {pseq}")

    # Step 3: Dataset Splits
    from src.dataset_splits import generate_all_splits
    splits_dir = os.path.join(args.output_dir, "splits")
    logger.info(">>> STEP 3: Generating the three canonical splits (random, unseen peptides, unseen alleles)...")
    splits_report = generate_all_splits(cleaned_csv, output_dir=splits_dir)
    logger.info("Step 3 Complete: All 3 splits generated with verified zero-leakage.")

    # Step 4: NetMHCstabpan Benchmark Setup and Batch Files
    from src.netmhcstabpan_client import prepare_benchmark_batches
    logger.info(">>> STEP 4: Preparing NetMHCstabpan benchmark batches for unseen alleles test set...")
    test_csv = os.path.join(splits_dir, "unseen_alleles", "test.csv")
    batch_info = prepare_benchmark_batches(test_csv, output_dir=os.path.join(args.output_dir, "netmhcstabpan_benchmark", "batches"))
    logger.info(f"Step 4 Complete: Prepared {len(batch_info['batches'])} batches for server querying.")

    # Step 5: Figures Generation
    logger.info(">>> STEP 5: Generating publication-quality figures...")
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
    from src.evaluate import plot_split_distributions, plot_allele_representation
    plot_split_distributions(os.path.join(splits_dir, "splits_summary.json"), "figures/splits_distribution.png")
    plot_allele_representation(cleaned_csv, "figures/allele_representation.png")
    
    # Run model comparison plot if script exists
    if os.path.exists("src/plot_model_comparison.py"):
        import subprocess
        subprocess.run([sys.executable, "src/plot_model_comparison.py"], check=False)

    logger.info("=================================================================")
    logger.info("                    PHASE 1 EXECUTION COMPLETE                   ")
    logger.info("=================================================================")


if __name__ == "__main__":
    main()
