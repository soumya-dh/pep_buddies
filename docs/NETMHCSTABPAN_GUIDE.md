# NetMHCstabpan-1.0 Benchmark Reproduction Guide

This guide is designed for any team member (including non-coding teammates) to reproduce the **NetMHCstabpan-1.0** baseline benchmark on the evaluation test sets.

---

## 1. Overview

**NetMHCstabpan** is the state-of-the-art reference method developed by DTU Health Tech (Rasmussen et al., *Bioinformatics*, 2016) for predicting peptide-MHC Class I stability (half-life $T_{1/2}$ in hours).

In this challenge, NetMHCstabpan serves as the **gold-standard benchmark number to match or beat**, especially on the **Unseen HLA Alleles** test split.

---

## 2. Option A: Manual Web Submission (Recommended for Non-Coders)

The automated pre-processor has already prepared batch submission files grouped by allele in:
`data/netmhcstabpan_benchmark/batches/`

Each file contains peptide sequences (one per line) formatted specifically for the DTU server:
- `HLA_A2402_batch1.pep`
- `HLA_A3207_batch1.pep`
- `HLA_A6801_batch1.pep`
- `HLA_A8001_batch1.pep`
- `HLA_B3503_batch1.pep`
- `HLA_B4001_batch1.pep`
- `HLA_B4501_batch1.pep`
- `HLA_B5401_batch1.pep`

### Step-by-Step Instructions:

1. **Open the Web Server**:
   Navigate to [https://services.healthtech.dtu.dk/services/NetMHCstabpan-1.0/](https://services.healthtech.dtu.dk/services/NetMHCstabpan-1.0/) in your web browser.

2. **Select Input Type**:
   - Under **"Type of input"**, choose: **PEPTIDE** (radio button or dropdown).

3. **Upload or Paste Peptides**:
   - Either click **"Choose File"** next to **"Paste or upload peptide file"** and select one of the `.pep` files (e.g. `HLA_A2402_batch1.pep`), **OR** open the `.pep` file in any text editor, copy all text, and paste into the **PEPTIDE** box.

4. **Select or Enter the Allele**:
   - In the **"Allele"** section:
     - Type or select the matching allele name shown in the file name (e.g. `HLA-A24:02`, `HLA-B35:03`, `HLA-B40:01`).
     - Note: NetMHC naming omits the asterisk `*` (e.g., use `HLA-A24:02` instead of `HLA-A*24:02`).
   - Leave peptide length set to `9` (or default).

5. **Submit**:
   - Click the green **"Submit"** button at the bottom of the page.
   - Wait ~15–45 seconds while the calculation runs.

6. **Save the Results**:
   - When the results page finishes loading, press **Ctrl+S** (or **Cmd+S** on Mac) to save the web page as text or HTML, **OR** simply select all text on the output screen (Ctrl+A / Cmd+A), copy it, and save it as a text file named:
     `data/netmhcstabpan_benchmark/raw_outputs/<allele>_output.txt`
     (e.g., `data/netmhcstabpan_benchmark/raw_outputs/HLA-A2402_output.txt`).

7. **Run Automated Benchmark Aggregator**:
   Once you've placed any or all output files into `data/netmhcstabpan_benchmark/raw_outputs/`, run:
   ```bash
   python src/evaluate.py --benchmark
   ```
   The script will automatically parse your files, compute Spearman $\rho$, Pearson $r$, RMSE, and ROC-AUC, and save the publication-quality comparison figure to `figures/benchmark_netmhcstabpan.png`!

---

## 3. Option B: Automated Command-Line Client

If you prefer running server queries programmatically via Python, our built-in client handles queuing, polling, and local caching automatically:

```bash
# Query a specific allele batch automatically:
python src/netmhcstabpan_client.py --query --allele HLA-A24:02 --batch data/netmhcstabpan_benchmark/batches/HLA_A2402_batch1.pep

# Or run automated benchmark on all test batches:
python src/netmhcstabpan_client.py --run-all --split unseen_alleles
```

All responses are cached under `data/netmhcstabpan_benchmark/cache/` so you never have to re-query the same peptide set twice.

---

## 4. Key Metrics to Record

| Metric | Target to Beat / Match | Description |
| :--- | :--- | :--- |
| **Spearman $\rho$** | $\ge 0.45 - 0.65$ | Rank correlation on predicted vs experimental half-life |
| **Pearson $r$** | $\ge 0.40 - 0.60$ | Linear correlation on logistic stability score |
| **RMSE** | $\le 0.25 - 0.35$ | Root Mean Squared Error on normalized stability score $[0, 1]$ |
| **ROC-AUC** | $\ge 0.75 - 0.85$ | Binary discrimination of stable binders ($T_{1/2} \ge 2\text{ hours}$) |
| **PR-AUC** | $\ge 0.60 - 0.75$ | Precision-Recall AUC for identifying true stable neoantigens |
