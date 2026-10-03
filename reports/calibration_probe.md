# Calibration Probe: Unseen-Allele Split & Hybrid Ablation

Target: `log10_1p_thalf`. kNN k = 5.

Each rung re-scores the **same predictions**, changing only each allele's offset on the target scale.

| Rung | What it does | Uses test labels? |
| :--- | :--- | :---: |
| `as_is` | raw predictions | no |
| `centred` | removes the model's own allele offsets (adds no information) | no |
| `knn_offset` | allele mean estimated from pseudosequence-similar *training* alleles | **no — deployable** |
| `oracle_offset` | allele mean set to its true test mean | **yes — upper bound only** |

## mlp (n = 3078, 8 alleles)

| Rung | Spearman ρ | Pearson r | RMSE | ROC-AUC | Median per-allele ρ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `as_is` | 0.4927 ± 0.0405 | 0.5858 ± 0.0420 | 0.3582 ± 0.0162 | 0.8004 ± 0.0167 | 0.5238 ± 0.0533 |
| `centred` | 0.4933 ± 0.0277 | 0.5946 ± 0.0302 | 0.5197 ± 0.0067 | 0.7953 ± 0.0185 | 0.5238 ± 0.0533 |
| `knn_offset` | 0.4124 ± 0.0364 | 0.5459 ± 0.0352 | 0.3841 ± 0.0090 | 0.7388 ± 0.0247 | 0.5238 ± 0.0533 |
| `oracle_offset` *(bound)* | 0.6662 ± 0.0181 | 0.7098 ± 0.0256 | 0.3072 ± 0.0112 | 0.8615 ± 0.0093 | 0.5238 ± 0.0533 |

Share of MSE removable by a perfect per-allele intercept: **0.2607 ± 0.0774**. Spread of per-allele bias: 0.2012 ± 0.0401.
kNN error in estimating an allele's mean target: MAE 0.1925.

## ridge (n = 3078, 8 alleles)

| Rung | Spearman ρ | Pearson r | RMSE | ROC-AUC | Median per-allele ρ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `as_is` | 0.0910 | 0.1873 | 0.5012 | 0.5903 | 0.1402 |
| `centred` | 0.1970 | 0.2642 | 0.5697 | 0.6289 | 0.1402 |
| `knn_offset` | 0.1065 | 0.2281 | 0.4495 | 0.5627 | 0.1402 |
| `oracle_offset` *(bound)* | 0.4594 | 0.4673 | 0.3859 | 0.7476 | 0.1402 |

Share of MSE removable by a perfect per-allele intercept: **0.4072**. Spread of per-allele bias: 0.3501.
kNN error in estimating an allele's mean target: MAE 0.1925.

## hybrid (One-Hot Peptide + ESM-2 35M HLA Pocket, n = 3078, 8 alleles)

| Model Featurization | Spearman ρ | Pearson r | RMSE | Offset Error Fraction | Spread of Allele Bias |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Ridge (One-Hot Pseudosequence) | 0.0910 | 0.1873 | 0.5012 | **0.4072** (40.7%) | 0.3501 |
| Hybrid (One-Hot Pep + ESM-2 Pocket) | 0.2473 | 0.2974 | 0.4800 | **0.1875** (18.8%) | 0.2104 |
| Pan-MLP (Non-linear Interaction) | 0.4927 | 0.5858 | 0.3582 | **0.2607** (26.1%) | 0.2012 |

**Key Ablation Takeaway:**
Continuous ESM-2 35M HLA pocket representations cut between-allele baseline shift error from 40.72% to 18.75% of total MSE, directly addressing the allele offset limitation identified by the calibration probe and increasing unseen-allele Spearman ρ from 0.0910 to 0.2473.

## netmhcstabpan (n = 320, 8 alleles)

| Rung | Spearman ρ | Pearson r | RMSE | ROC-AUC | Median per-allele ρ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `as_is` | 0.8537 | 0.8700 | 0.2433 | 0.9323 | 0.7563 |
| `centred` | 0.6153 | 0.7547 | 0.4886 | 0.8503 | 0.7563 |
| `knn_offset` | 0.5394 | 0.6893 | 0.3386 | 0.7988 | 0.7563 |
| `oracle_offset` *(bound)* | 0.7995 | 0.8690 | 0.2290 | 0.9358 | 0.7563 |

Share of MSE removable by a perfect per-allele intercept: **0.1141**. Spread of per-allele bias: 0.0703.
kNN error in estimating an allele's mean target: MAE 0.2010.

Scored on its 320-pair subset only; other models here cover the full test set, so compare rung-to-rung gains rather than absolute values.
