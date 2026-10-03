"""
netmhcstabpan_client.py - Tooling to reproduce the NetMHCstabpan benchmark:
  1. Batch input preparation for manual web submission (non-coder workflow).
  2. Automated programmatic query runner via DTU webface2 CGI with local caching.
  3. Raw output parser (HTML / Text / TSV).
  4. Benchmark metric evaluator (Spearman rho, Pearson r, RMSE, AUC).
"""

import os
import re
import time
import json
import logging
import urllib.request
import urllib.parse
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NETMHCSTABPAN_URL = "https://services.healthtech.dtu.dk/cgi-bin/webface2.cgi"
CONFIG_FILE = "/var/www/services/services/NetMHCstabpan-1.0/webface.cf"


def prepare_benchmark_batches(
    test_csv_path: str,
    output_dir: str = "data/netmhcstabpan_benchmark/batches",
    max_batch_size: int = 1500
) -> Dict[str, Any]:
    """
    Format test set peptides grouped by allele for web server submission.
    Creates .pep files and a submission manifest for the non-coder teammate.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(test_csv_path)
    logger.info(f"Loaded test set with {len(df)} samples across {df['allele'].nunique()} alleles.")
    
    manifest = []
    
    for allele, group in df.groupby("allele"):
        # Format allele for NetMHC web server: HLA-A*02:01 -> HLA-A02:01
        clean_allele = allele.replace("*", "").split("(")[0]
        sanitized_name = clean_allele.replace(":", "").replace("-", "_")
        
        peptides = group["peptide"].unique().tolist()
        num_peptides = len(peptides)
        
        # Split into batches if needed
        for batch_idx in range(0, num_peptides, max_batch_size):
            batch_peps = peptides[batch_idx:batch_idx + max_batch_size]
            batch_num = (batch_idx // max_batch_size) + 1
            file_name = f"{sanitized_name}_batch{batch_num}.pep"
            file_path = os.path.join(output_dir, file_name)
            
            with open(file_path, "w") as f:
                f.write("\n".join(batch_peps) + "\n")
                
            manifest.append({
                "allele_raw": allele,
                "allele_netmhc": clean_allele,
                "batch_file": file_name,
                "batch_path": file_path,
                "peptide_count": len(batch_peps),
                "peptide_length": len(batch_peps[0]) if batch_peps else 9
            })
            
    manifest_path = os.path.join(output_dir, "submission_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        
    logger.info(f"Generated {len(manifest)} batch submission files in {output_dir}")
    return {"manifest_path": manifest_path, "batches": manifest}


def parse_netmhcstabpan_output(text: str) -> pd.DataFrame:
    """
    Parse NetMHCstabpan server output (raw text or HTML) into a structured DataFrame.
    
    Target columns in output:
      pos, HLA, peptide, Identity, Pred, Thalf(h), %Rank_Stab, BindLevel
    """
    records = []
    # Match data rows:
    # 0  HLA-A*02:01    VTTEVAFGL    PEPLIST    0.450    0.87    5.00  (optional SB/WB)
    row_pattern = re.compile(
        r"^\s*(\d+)\s+([A-Za-z0-9\*\:\-]+)\s+([A-Z]+)\s+([A-Za-z0-9_-]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)(?:\s+([A-Z]+))?",
        re.MULTILINE
    )
    
    for match in row_pattern.finditer(text):
        pos, hla, peptide, identity, pred, thalf, rank, bindlevel = match.groups()
        records.append({
            "pos": int(pos),
            "allele": hla,
            "peptide": peptide,
            "netmhc_identity": identity,
            "netmhc_score": float(pred),
            "netmhc_thalf_hours": float(thalf),
            "netmhc_rank": float(rank),
            "netmhc_bind_level": bindlevel if bindlevel else "NB"
        })
        
    df_res = pd.DataFrame(records)
    logger.info(f"Parsed {len(df_res)} prediction rows from NetMHCstabpan output.")
    return df_res


def query_netmhcstabpan_server(
    allele: str,
    peptides: List[str],
    cache_dir: str = "data/netmhcstabpan_benchmark/cache",
    max_wait_seconds: int = 120
) -> pd.DataFrame:
    """
    Programmatically submit a query to the NetMHCstabpan-1.0 web server,
    poll until job completion, and return parsed DataFrame with local caching.
    """
    os.makedirs(cache_dir, exist_ok=True)
    clean_allele = allele.replace("*", "").split("(")[0]
    cache_key = f"{clean_allele.replace(':', '').replace('-', '_')}_{len(peptides)}_{hash(tuple(peptides[:5]))}"
    cache_file = os.path.join(cache_dir, f"{cache_key}.csv")
    
    if os.path.exists(cache_file):
        logger.info(f"Loading cached NetMHCstabpan predictions from: {cache_file}")
        return pd.read_csv(cache_file)
        
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = []
    
    def add_field(name, value):
        body.append(f"--{boundary}")
        body.append(f'Content-Disposition: form-data; name="{name}"')
        body.append("")
        body.append(str(value))
        
    pep_text = "\n".join(peptides) + "\n"
    pep_len = str(len(peptides[0])) if peptides else "9"
    
    add_field("configfile", CONFIG_FILE)
    add_field("inp", "1")  # 1 = peptide list
    add_field("PEPPASTE", pep_text)
    add_field("allele", clean_allele)
    add_field("len", pep_len)
    body.append(f"--{boundary}--")
    body.append("")
    
    payload = "\r\n".join(body).encode("utf-8")
    req = urllib.request.Request(
        NETMHCSTABPAN_URL,
        data=payload,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "Mozilla/5.0 (Research-Benchmark-Client)"
        }
    )
    
    logger.info(f"Submitting {len(peptides)} peptides for {clean_allele} to NetMHCstabpan server...")
    with urllib.request.urlopen(req, timeout=30) as resp:
        submit_html = resp.read().decode("utf-8", errors="ignore")
        
    # Extract job ID
    job_match = re.search(r"Job status of ([A-Za-z0-9]+)", submit_html) or re.search(r"launchcheck\(['\"]queued['\"],\s*['\"]([A-Za-z0-9]+)['\"]", submit_html)
    if not job_match:
        logger.error(f"Could not extract Job ID from response: {submit_html[:500]}")
        raise RuntimeError("Failed to extract Job ID from NetMHCstabpan submission response.")
        
    job_id = job_match.group(1)
    logger.info(f"Job successfully queued! Job ID: {job_id}. Polling for completion...")
    
    # Poll status
    check_url = f"{NETMHCSTABPAN_URL}?jobid={job_id}&wait=20"
    start_time = time.time()
    
    while time.time() - start_time < max_wait_seconds:
        time.sleep(3)
        poll_req = urllib.request.Request(check_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(poll_req, timeout=30) as r:
            poll_resp = r.read().decode("utf-8", errors="ignore")
            
        if "NetMHCstabpan version 1.0" in poll_resp and "Thalf(h)" in poll_resp:
            logger.info(f"Job {job_id} finished successfully!")
            df_parsed = parse_netmhcstabpan_output(poll_resp)
            if not df_parsed.empty:
                df_parsed.to_csv(cache_file, index=False)
            return df_parsed
            
        if "Fatal Error" in poll_resp:
            raise RuntimeError(f"NetMHCstabpan returned Fatal Error: {poll_resp[:300]}")
            
    raise TimeoutError(f"Job {job_id} timed out after {max_wait_seconds} seconds.")


def evaluate_benchmark(
    test_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    stability_threshold_hours: float = 2.0
) -> Dict[str, Any]:
    """
    Compute rigorous immuno-oncology benchmark metrics comparing
    NetMHCstabpan predictions against experimental ground truth.
    """
    from scipy.stats import spearmanr, pearsonr
    from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error, mean_absolute_error
    
    # Standardize allele names for joining
    test = test_df.copy()
    pred = pred_df.copy()
    
    test["allele_join"] = test["allele"].astype(str).str.replace("*", "").str.replace("HLA-", "")
    pred["allele_join"] = pred["allele"].astype(str).str.replace("*", "").str.replace("HLA-", "")
    
    # Join on allele_join and peptide
    merged = pd.merge(test, pred, on=["allele_join", "peptide"], suffixes=("_exp", "_pred"))
    
    if merged.empty:
        # Fallback join on peptide only if single allele
        merged = pd.merge(test, pred, on=["peptide"], suffixes=("_exp", "_pred"))
        
    logger.info(f"Evaluating benchmark on {len(merged)} matched predictions.")
    
    y_true_thalf = merged["thalf_hours"].values
    y_true_score = merged["stability_score"].values if "stability_score" in merged.columns else (1.0 / (1.0 + 5.0 / y_true_thalf))
    y_true_binary = (y_true_thalf >= stability_threshold_hours).astype(int)
    
    y_pred_thalf = merged["netmhc_thalf_hours"].values
    y_pred_score = merged["netmhc_score"].values
    
    # 1. Rank & Correlation metrics
    spearman_thalf, spearman_p = spearmanr(y_true_thalf, y_pred_thalf)
    pearson_score, pearson_p = pearsonr(y_true_score, y_pred_score)
    
    # 2. Regression error
    rmse_score = float(np.sqrt(mean_squared_error(y_true_score, y_pred_score)))
    mae_score = float(mean_absolute_error(y_true_score, y_pred_score))
    
    # 3. Binary classification metrics (stability >= threshold)
    has_pos_and_neg = (len(np.unique(y_true_binary)) > 1)
    if has_pos_and_neg:
        roc_auc = float(roc_auc_score(y_true_binary, y_pred_score))
        pr_auc = float(average_precision_score(y_true_binary, y_pred_score))
    else:
        roc_auc, pr_auc = 0.5, 0.0
        
    metrics = {
        "n_samples": int(len(merged)),
        "spearman_rho": float(round(spearman_thalf, 4)),
        "spearman_pvalue": float(spearman_p),
        "pearson_r": float(round(pearson_score, 4)),
        "pearson_pvalue": float(pearson_p),
        "rmse_stability_score": float(round(rmse_score, 4)),
        "mae_stability_score": float(round(mae_score, 4)),
        "roc_auc": float(round(roc_auc, 4)),
        "pr_auc": float(round(pr_auc, 4)),
        "stability_threshold_hours": stability_threshold_hours,
        "merged_dataframe": merged
    }
    
    return metrics


if __name__ == "__main__":
    # Test batch file preparation on unseen_alleles test set
    prep = prepare_benchmark_batches("data/splits/unseen_alleles/test.csv")
    print(f"Prepared {len(prep['batches'])} batches.")
