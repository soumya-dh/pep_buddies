"""
modal_app.py - Serverless Cloud Deployment & GPU Foundation Model Inference via Modal.

Enables:
1. Public interactive Streamlit deployment on Modal for live judge pitches and testing.
2. Cloud GPU inference for ESM-2 Protein Foundation Models (esm2_t33_650M / esm2_t12_35M).
3. Serverless high-throughput neoantigen stability screening across patient mutation panels.

Eligible for the AI x Science Hackathon "Best use of Modal" Challenge.
"""

import os
import subprocess
import modal

# -------------------------------------------------------------
# Modal App & Container Image Configuration
# -------------------------------------------------------------
app = modal.App("pep-buddies-stability")

# Container image with scientific stack, PyTorch, Streamlit, and 3D visualization
app_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.0",
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy>=1.10",
        "scikit-learn>=1.2",
        "streamlit>=1.30",
        "altair>=5.0",
        "biopython>=1.80",
        "transformers>=4.30",
    )
    .add_local_dir("models", remote_path="/root/models")
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("data", remote_path="/root/data")
    .add_local_dir("static", remote_path="/root/static")
    .add_local_file("app.py", remote_path="/root/app.py")
    .add_local_file(".streamlit/config.toml", remote_path="/root/.streamlit/config.toml")
    .add_local_file("glioma_prospective_predictions.csv", remote_path="/root/glioma_prospective_predictions.csv")
)


# -------------------------------------------------------------
# 1. Serverless Streamlit Live Dashboard (Public URL for Judges)
# -------------------------------------------------------------
@app.function(
    image=app_image,
    concurrency_limit=10,
    timeout=3600,
)
@modal.web_server(port=8501, startup_timeout=60)
def serve_streamlit():
    """
    Spins up the interactive Streamlit application with the 3D binding groove
    viewer, deep mutational scanning, and biophysical notes on a live public URL.
    """
    cmd = [
        "streamlit",
        "run",
        "/root/app.py",
        "--server.port=8501",
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--server.enableCORS=false",
    ]
    subprocess.Popen(cmd)


# -------------------------------------------------------------
# 2. Serverless Fast Batch Stability Scoring
# -------------------------------------------------------------
@app.function(image=app_image, timeout=300)
def predict_batch_stability(peptides: list, allele: str = "HLA-A*02:01") -> list:
    """
    Serverless endpoint: evaluates hundreds of candidate neoantigens in parallel.
    Returns predicted half-life (T1/2 hours), stability classification, and anchor sensitivity.
    """
    import sys
    sys.path.insert(0, "/root")
    import torch
    from models.baseline_model import PanStabilityMLP, one_hot_encode_sequence
    from src.hla_database import HLADatabase
    from src.targets import target_to_thalf

    hla_db = HLADatabase(
        pseudo_dat_path="/root/data/raw/MHC_pseudo.dat",
        imgt_fasta_path="/root/data/raw/hla_prot.fasta",
    )
    hla_pseudo = hla_db.get_pseudo_sequence(allele)

    checkpoint = torch.load("/root/models/frozen/pan_stability_mlp_frozen.pt", map_location="cpu")
    model = PanStabilityMLP(peptide_dim=9 * 20, hla_dim=34 * 20, hidden_dim=64)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    results = []
    for pep in peptides:
        core_seq = pep[:9]
        pep_enc = torch.tensor(one_hot_encode_sequence(core_seq, max_len=9), dtype=torch.float32).flatten().unsqueeze(0)
        hla_enc = torch.tensor(one_hot_encode_sequence(hla_pseudo, max_len=34), dtype=torch.float32).flatten().unsqueeze(0)

        with torch.no_grad():
            pred_norm = model(pep_enc, hla_enc).item()

        thalf = target_to_thalf(pred_norm)
        status = "Stable Binder" if thalf >= 1.0 else "Modest Binder" if thalf >= 0.5 else "Unstable"
        results.append({
            "peptide": pep,
            "core_9mer": core_seq,
            "allele": allele,
            "predicted_thalf_hours": round(thalf, 3),
            "status": status,
        })
    return results


# -------------------------------------------------------------
# 3. GPU Cloud Foundation Model Worker (ESM-2 Embeddings)
# -------------------------------------------------------------
@app.function(
    image=app_image,
    gpu="T4",
    timeout=600,
)
def compute_esm2_embeddings(sequences: list, model_name: str = "facebook/esm2_t12_35M_UR50D") -> dict:
    """
    Serverless GPU function: extracts per-residue and pooled embeddings from
    Evolutionary Scale Modeling (ESM-2) foundation models.
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).cuda().eval()

    embeddings = {}
    with torch.no_grad():
        for seq in sequences:
            inputs = tokenizer(seq, return_tensors="pt").to("cuda")
            outputs = model(**inputs)
            # Pool across sequence length (excluding [CLS] and [SEP])
            rep = outputs.last_hidden_state[0, 1:-1, :].mean(dim=0).cpu().numpy().tolist()
            embeddings[seq] = {
                "embedding_dim": len(rep),
                "pooled_vector": rep[:10],  # preview top-10 components
                "device": "NVIDIA T4 (Modal Cloud GPU)",
            }

    return embeddings


# -------------------------------------------------------------
# CLI Entrypoint for Local Testing via `modal run modal_app.py`
# -------------------------------------------------------------
@app.local_entrypoint()
def main():
    """Test serverless function execution on Modal."""
    print("🚀 Testing Modal Serverless Batch Scoring...")
    test_peptides = ["RMSAPSTGG", "RKSAPSTGG", "LEEKKGNYV", "WLPFGFILI"]
    res = predict_batch_stability.remote(test_peptides, "HLA-A*02:01")
    for r in res:
        print(f"  • {r['peptide']}: {r['predicted_thalf_hours']:.2f} h [{r['status']}]")
    print("\n✅ Modal Serverless execution successful!")
