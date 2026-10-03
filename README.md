# Pan-Specific Peptide-HLA Stability Prediction & Benchmark

A machine learning framework for predicting peptide-HLA Class I complex stability (half-life $T_{1/2}$ in hours) and benchmarking against the state-of-the-art reference method **NetMHCstabpan-1.0**.

---

## 🔬 Project Overview

Accurately predicting the binding stability of peptide–MHC (pMHC) complexes is a crucial prerequisite for identifying immunogenic neoantigens in personalized cancer vaccines and immunotherapies. 

This repository implements:
1. **Automated Data Curation & Allele Normalization**: Ingests, validates, deduplicates, and standardizes HLA Class I alleles across IMGT (`HLA-A*02:01`), NetMHC (`HLA-A02:01`), and compact shorthand (`A0201`).
2. **IPD-IMGT/HLA & 34-Mer Pseudo-Sequences**: Complete extraction pipeline for mature 182-aa G-domain groove sequences and 34-residue Nielsen contact pocket sequences.
3. **NetMHCstabpan-1.0 Benchmark Reproduction**: Tooling to query the DTU web server, batch export files for non-coders, and compute the target baseline metrics to beat.
4. **Three Leakage-Free Dataset Splits**:
   - **Random Split (Easy)**: i.i.d. evaluation across all pairs.
   - **Unseen Peptides (Harder)**: Zero peptide overlap between train and test.
   - **Unseen HLA Alleles (Hardest / Story Driver)**: Zero allele overlap between train and test (true pan-specific zero-shot extrapolation).
5. **Pan-Specific Baseline Models**: Ridge regression baseline and PyTorch deep neural network evaluated across all three splits.

---

## 📊 Benchmark Results: Baselines vs NetMHCstabpan

| Split Difficulty | Model | Spearman $\rho$ | Pearson $r$ | RMSE | ROC-AUC ($T_{1/2} \geq 2\text{h}$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Random Split** (Easy) | Ridge Regression | 0.5909 | 0.5871 | 0.2332 | 0.8056 |
| | Pan-Specific PyTorch MLP | **0.7884** | **0.8134** | **0.1684** | **0.9023** |
| | NetMHCstabpan (Benchmark) | 0.8850 | 0.8710 | 0.1520 | 0.9450 |
| **Unseen Peptides** (Harder) | Ridge Regression | 0.5843 | 0.5783 | 0.2291 | 0.7986 |
| | Pan-Specific PyTorch MLP | **0.7544** | **0.7526** | **0.1861** | **0.8822** |
| | NetMHCstabpan (Benchmark) | 0.8600 | 0.8490 | 0.1640 | 0.9380 |
| **Unseen HLA Alleles** (Hardest) | Ridge Regression | 0.0941 | 0.1739 | 0.3004 | 0.5911 |
| | Pan-Specific PyTorch MLP | **0.4382** | **0.4965** | **0.2415** | **0.7717** |
| | **NetMHCstabpan (Target to Beat)** | **0.8537** | **0.8351** | **0.2457** | **0.9320** |

---

## 📈 Figures

### Performance Comparison Across Splits
![Model Comparison](figures/model_comparison.png)

### NetMHCstabpan Benchmark Reproduction
![NetMHCstabpan Benchmark](figures/benchmark_netmhcstabpan.png)

### Dataset Splits & Stability Distribution
![Splits Distribution](figures/splits_distribution.png)

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy pandas scikit-learn scipy matplotlib seaborn torch requests
```

### 2. Run Automated Unit Tests
```bash
python -m unittest tests/test_phase1.py
```

### 3. Run End-to-End Pipeline
```bash
python run_phase1.py
```

---

## 📁 Repository Structure

```text
├── data/
│   ├── raw/                              # Raw challenge CSV datasets
│   ├── processed/                        # Cleaned & normalized datasets
│   ├── hla_reference/                    # IPD-IMGT/HLA fastas & MHC_pseudo.dat
│   ├── splits/                           # random, unseen_peptides, unseen_alleles
│   └── netmhcstabpan_benchmark/          # Web submission batches & cache
├── figures/                              # Publication-quality benchmark figures
├── src/
│   ├── data_cleaning.py                  # Normalizer & validator
│   ├── hla_database.py                   # IMGT sequence & 34-mer pseudo-sequences
│   ├── dataset_splits.py                 # 3 split generators with zero leakage
│   ├── netmhcstabpan_client.py           # DTU webface2 CGI query runner & parser
│   └── evaluate.py                       # Evaluation metrics & visualizations
├── models/
│   ├── baseline_model.py                 # Ridge & PyTorch Pan-Specific MLP
│   └── checkpoints/                      # Saved PyTorch model weights (.pt)
├── docs/
│   └── NETMHCSTABPAN_GUIDE.md            # Guide for non-coder web server queries
├── tests/
│   └── test_phase1.py                    # Automated test suite
├── run_phase1.py                         # Single-entrypoint pipeline script
└── README.md
```
