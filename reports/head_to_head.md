# NetMHCstabpan Head-to-Head

Target: `log10_1p_thalf` — all correlations and errors on this scale.

Common evaluation subset: **320 (allele, peptide) pairs** across **8 held-out alleles**. These are the only pairs NetMHCstabpan was queried for, so they are the only fair basis for comparing against it.

## On the common subset (n = 320)

| Model | Spearman ρ | Pearson r | RMSE | ROC-AUC | Median per-allele ρ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| netmhcstabpan | 0.8537 | 0.8700 | 0.2433 | 0.9323 | 0.7563 |
| mlp | 0.4318 ± 0.0327 | 0.5264 ± 0.0512 | 0.3859 ± 0.0173 | 0.7733 ± 0.0180 | 0.5504 ± 0.0477 |
| ridge | 0.0474 | 0.1502 | 0.5187 | 0.5730 | 0.1312 |

## On the full held-out-allele test set

Reported separately because it is a *different and larger* evaluation set. Do not compare these numbers against the NetMHCstabpan row above.

| Model | n | Spearman ρ | Pearson r | RMSE | ROC-AUC | Median per-allele ρ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| netmhcstabpan | — | n/a | n/a | n/a | n/a | n/a |
| mlp | 3078 | 0.4847 ± 0.0334 | 0.5725 ± 0.0465 | 0.3627 ± 0.0174 | 0.7981 ± 0.0151 | 0.5237 ± 0.0676 |
| ridge | 3078 | 0.0967 | 0.1881 | 0.5020 | 0.5946 | 0.1402 |

Pooled ρ on held-out alleles is inflated by between-allele differences in mean stability; the median per-allele ρ is the honest pan-specific number.
