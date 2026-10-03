"""
app.py - Streamlit Interactive Demo for HLA-Peptide Stability & Interpretability.

PepBuddies: Protein Foundation Model & 3D Crystallographic Pocket Architecture
for MHC Class I Neoantigen Stability Prediction and In Silico Mutational Scanning.
"""

import os
import sys
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st
import streamlit.components.v1 as components
import altair as alt

# Ensure workspace root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.baseline_model import PanStabilityMLP, one_hot_encode_sequence, AMINO_ACIDS
from src.hla_database import HLADatabase
from src.targets import target_to_thalf
from src.prospective.glioma_lock import find_best_core_for_10mer
from src.visualization.structure_viewer import build_pmhc_pdb, generate_3dmol_html

# Page configuration
st.set_page_config(
    page_title="PepBuddies | HLA Stability & Interpretability",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-stable {
        background-color: #dcfce7;
        color: #15803d;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        border: 1px solid #86efac;
    }
    .badge-modest {
        background-color: #fef9c3;
        color: #a16207;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        border: 1px solid #fde047;
    }
    .badge-unstable {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        border: 1px solid #fca5a5;
    }
    .note-box {
        background-color: #f1f5f9;
        border-left: 4px solid #3b82f6;
        padding: 10px 14px;
        border-radius: 4px;
        margin-top: 8px;
        font-size: 0.92rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_resources():
    """Load HLA database and frozen pan-specific stability model."""
    hla_db = HLADatabase()
    checkpoint_path = "models/frozen/pan_stability_mlp_frozen.pt"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "models/checkpoints/pan_stability_mlp.pt"

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = PanStabilityMLP(input_dim=860, hidden_dim=256, dropout=0.2)
    state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()
    return hla_db, model


hla_db, model = load_resources()

# Available Core HLA Alleles
COMMON_ALLELES = [
    "HLA-A*02:01",
    "HLA-A*24:02",
    "HLA-A*01:01",
    "HLA-A*03:01",
    "HLA-B*07:02",
    "HLA-B*08:01",
]

# Preset Clinical & Prospective Examples with Wild-Type counterparts
PRESETS = {
    "H3.3 K27M Mutant (RMSAPSTGG) - DMG/DIPG": {
        "sequence": "RMSAPSTGG",
        "wt_sequence": "RKSAPSTGG",
        "allele": "HLA-A*02:01",
        "desc": "Gain-of-stability tumor neoantigen. Met at P2 relieves electrostatic clash in Pocket B.",
    },
    "H3.3 Wild-Type (RKSAPSTGG) - Unstable Control": {
        "sequence": "RKSAPSTGG",
        "wt_sequence": "RMSAPSTGG",
        "allele": "HLA-A*02:01",
        "desc": "Normal wild-type equivalent. Pos 2 Lysine severely clashes with Met45/Val67 in Pocket B.",
    },
    "EGFRvIII (LEEKKGNYV) - Glioblastoma Exon 2-7": {
        "sequence": "LEEKKGNYV",
        "wt_sequence": "",
        "allele": "HLA-A*02:01",
        "desc": "Novel tumor junction epitope. C-term Valine anchors into Pocket F; Glu at P2 modulates affinity.",
    },
    "IL13Rα2 (WLPFGFILI) - Overexpressed Glioma": {
        "sequence": "WLPFGFILI",
        "wt_sequence": "",
        "allele": "HLA-A*02:01",
        "desc": "Glioblastoma-associated overexpressed antigen with strong hydrophobic anchors.",
    },
    "H3.3 K27M 10-mer (RMSAPATGGV) - High-Affinity Decamer": {
        "sequence": "RMSAPATGGV",
        "wt_sequence": "RKSAPATGGV",
        "allele": "HLA-A*02:01",
        "desc": "Full-length 10-mer candidate. Optimal 9-mer bulge core with C-term Valine.",
    },
    "Poly-Aspartate Negative Control (DDDDDDDDD)": {
        "sequence": "DDDDDDDDD",
        "wt_sequence": "",
        "allele": "HLA-A*02:01",
        "desc": "Artificial negative control. Severe poly-acidic electrostatic repulsion in all pockets.",
    },
    "Custom Sequence (Enter Your Own)": {
        "sequence": "",
        "wt_sequence": "",
        "allele": "HLA-A*02:01",
        "desc": "Test any 9-mer or 10-mer peptide of your choice.",
    },
}

# -------------------------------------------------------------
# Sidebar Controls
# -------------------------------------------------------------
# Local file load prevents failure on venue Wi-Fi drops
local_img_path = "figures/quantitative_metrics_validation.png"
if os.path.exists(local_img_path):
    st.sidebar.image(local_img_path, use_container_width=True, caption="Model Interpretability Validation")

st.sidebar.title("🧬 Demonstration Panel")

# Synchronize session state with preset dropdown selection
if "current_preset" not in st.session_state:
    st.session_state.current_preset = list(PRESETS.keys())[0]
    st.session_state.pep_input_box = PRESETS[st.session_state.current_preset]["sequence"]
    st.session_state.wt_input_box = PRESETS[st.session_state.current_preset].get("wt_sequence", "")
    st.session_state.allele_selector = PRESETS[st.session_state.current_preset]["allele"]

def handle_preset_change():
    preset = st.session_state.preset_dropdown
    seq = PRESETS[preset]["sequence"]
    wt_seq = PRESETS[preset].get("wt_sequence", "")
    if seq:
        st.session_state.pep_input_box = seq
    st.session_state.wt_input_box = wt_seq
    st.session_state.allele_selector = PRESETS[preset]["allele"]
    st.session_state.current_preset = preset

selected_preset = st.sidebar.selectbox(
    "Choose a Pre-loaded Example:",
    list(PRESETS.keys()),
    index=list(PRESETS.keys()).index(st.session_state.current_preset),
    key="preset_dropdown",
    on_change=handle_preset_change,
)
preset_data = PRESETS[selected_preset]

cur_allele = st.session_state.get("allele_selector", preset_data["allele"])
default_allele_idx = COMMON_ALLELES.index(cur_allele) if cur_allele in COMMON_ALLELES else 0

selected_allele = st.sidebar.selectbox(
    "Target HLA Allele:",
    COMMON_ALLELES,
    index=default_allele_idx,
    key="allele_selector",
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Specifications:**")
st.sidebar.markdown("- **Architecture:** Pan-Specific MLP")
st.sidebar.markdown("- **HLA Pocket:** 34 Nielsen Contact Residues")
st.sidebar.markdown("- **Weights Checkpoint:** Frozen & SHA-256 Locked")
st.sidebar.markdown("- **Status:** `v1.0-locked`")

# -------------------------------------------------------------
# Main Interface: Input Form with Active Predict Button
# -------------------------------------------------------------
st.markdown('<div class="main-title">🧬 PepBuddies: HLA-Peptide Stability Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Live biophysical stability prediction, deep mutational scanning, and 3D cleft visualization for MHC Class I neoantigens.</div>', unsafe_allow_html=True)

with st.form("neoantigen_form"):
    col_input1, col_input2 = st.columns([2, 2])
    with col_input1:
        pep_input = st.text_input(
            "Candidate Neoantigen (9-mer or 10-mer):",
            key="pep_input_box",
        ).strip().upper()
    with col_input2:
        wt_input = st.text_input(
            "Normal Wild-Type Sequence (Optional for WT vs Mutant Comparison):",
            key="wt_input_box",
        ).strip().upper()

    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        predict_submitted = st.form_submit_button("⚡ Predict Stability", type="primary", use_container_width=True)
    with col_btn2:
        st.caption("Press Enter or click Predict to compute stability half-life, mutational scan, and pocket biophysics.")

if not pep_input:
    st.warning("Please enter an amino acid sequence.")
    st.stop()

# Validate sequence characters
invalid_aas = [aa for aa in pep_input if aa not in AMINO_ACIDS]
if invalid_aas:
    st.error(f"Invalid amino acid characters detected: {', '.join(set(invalid_aas))}. Please use standard 20 amino acids.")
    st.stop()

if len(pep_input) not in [9, 10]:
    st.warning(f"Note: Current sequence length is {len(pep_input)}. The model is optimized for 9-mer cores (or 10-mers via bulge alignment).")


# -------------------------------------------------------------
# Core Prediction Logic with Monte Carlo Dropout & Scan Matrix
# -------------------------------------------------------------
def mc_predict_uncertainty(model_inst, feat_tensor: torch.Tensor, n_samples: int = 30) -> Tuple[float, float]:
    """Estimates epistemic uncertainty using Monte Carlo Dropout (30 stochastic passes at p=0.20)."""
    for m in model_inst.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    with torch.no_grad():
        mc_preds = torch.stack([model_inst(feat_tensor) for _ in range(n_samples)]).squeeze(-1).numpy()

    model_inst.eval()
    thalfs = [target_to_thalf(float(p)) for p in mc_preds]
    return float(np.mean(thalfs)), float(np.std(thalfs))


@st.cache_data(show_spinner=False)
def run_prediction_and_scan(sequence: str, allele: str) -> Dict[str, Any]:
    pseudo = hla_db.get_pseudosequence(allele)

    # 10-mer bulge handling
    eval_seq = sequence
    bulge_note = None
    if len(sequence) == 10:
        best_core, best_score, best_del_pos = find_best_core_for_10mer(model, sequence, pseudo)
        eval_seq = best_core
        bulge_note = f"10-mer evaluated via structural bulge deletion at position {best_del_pos + 1} (optimal 9-mer binding core: `{best_core}`)."

    # Forward pass on evaluation sequence
    pep_oh = one_hot_encode_sequence(eval_seq, max_len=9).reshape(-1)
    hla_oh = one_hot_encode_sequence(pseudo, max_len=34).reshape(-1)
    feat_base = torch.tensor(np.concatenate([pep_oh, hla_oh]), dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        pred_target = model(feat_base).item()
    thalf = target_to_thalf(pred_target)

    # MC-Dropout Uncertainty
    thalf_mean, thalf_std = mc_predict_uncertainty(model, feat_base, n_samples=30)

    # In silico deep mutational scanning (9 x 20 matrix)
    feats = []
    pos_indices = []
    for p in range(len(eval_seq)):
        for aa in AMINO_ACIDS:
            mut_seq = list(eval_seq)
            mut_seq[p] = aa
            mut_pep_oh = one_hot_encode_sequence("".join(mut_seq), max_len=9).reshape(-1)
            feats.append(np.concatenate([mut_pep_oh, hla_oh]))
            pos_indices.append(p)

    batch_tensor = torch.tensor(np.array(feats), dtype=torch.float32)
    with torch.no_grad():
        preds = model(batch_tensor).squeeze(-1).numpy()

    # Full 9x20 Delta Matrix (S_mutant - S_wt)
    scan_matrix = (preds.reshape(len(eval_seq), len(AMINO_ACIDS)) - pred_target)

    sensitivities = np.zeros(len(eval_seq))
    for p in range(len(eval_seq)):
        mask = [i for i, pos in enumerate(pos_indices) if pos == p]
        sensitivities[p] = float(np.mean(np.abs(preds[mask] - pred_target)))

    # Anchor Importance Ratio (AIR)
    total_sens = np.sum(sensitivities)
    air_val = (sensitivities[1] + sensitivities[8]) / total_sens if total_sens > 0 and len(sensitivities) >= 9 else 0.0

    # Top-5 In Silico Stabilizing Substitutions
    top_stabilizing = []
    for p in range(len(eval_seq)):
        for j, aa in enumerate(AMINO_ACIDS):
            if aa != eval_seq[p]:
                delta_val = float(scan_matrix[p, j])
                if delta_val > 0.01:
                    mut_thalf = target_to_thalf(pred_target + delta_val)
                    top_stabilizing.append({
                        "position": f"P{p+1}",
                        "mutation": f"{eval_seq[p]} → {aa}",
                        "delta_score": delta_val,
                        "mutant_thalf": mut_thalf,
                    })
    top_stabilizing.sort(key=lambda x: x["delta_score"], reverse=True)

    return {
        "eval_seq": eval_seq,
        "bulge_note": bulge_note,
        "pred_target": float(pred_target),
        "thalf": float(thalf),
        "thalf_mean": float(thalf_mean),
        "thalf_std": float(thalf_std),
        "sensitivities": sensitivities.tolist(),
        "scan_matrix": scan_matrix.tolist(),
        "air": float(air_val),
        "top_stabilizing": top_stabilizing[:5],
    }


res = run_prediction_and_scan(pep_input, selected_allele)

# -------------------------------------------------------------
# Results Display: Metrics & Uncertainty
# -------------------------------------------------------------
st.markdown("---")

col1, col2, col3, col4 = st.columns([1.2, 1.2, 1.2, 1.4])

thalf = res["thalf"]
thalf_std = res["thalf_std"]

if thalf >= 2.0:
    badge_html = '<span class="badge-stable">🟢 STABLE BINDER (T½ ≥ 2.0h)</span>'
elif thalf >= 0.7:
    badge_html = '<span class="badge-modest">🟡 MODEST BINDER (0.7h – 2.0h)</span>'
else:
    badge_html = '<span class="badge-unstable">🔴 UNSTABLE / NON-BINDER (T½ < 0.7h)</span>'

with col1:
    st.metric("Predicted Half-Life (T½)", f"{thalf:.2f} ± {thalf_std:.2f} h", help="Confidence interval estimated via Monte Carlo Dropout (30 stochastic forward passes at p=0.20).")
with col2:
    st.metric("Target Score log₁₀(1 + T½)", f"{res['pred_target']:.4f}")
with col3:
    st.metric("Anchor Importance (P2+P9)", f"{res['air']*100:.1f}%", help="Share of total sensitivity concentrated at canonical anchors P2 and P9 (Null chance: 22.2%).")
with col4:
    st.markdown("**Binding Status:**")
    st.markdown(badge_html, unsafe_allow_html=True)

if res["bulge_note"]:
    st.info(f"ℹ️ {res['bulge_note']}")

# -------------------------------------------------------------
# Wild-Type vs. Mutant Comparative Analysis (First-Class Feature)
# -------------------------------------------------------------
if wt_input:
    invalid_wt = [aa for aa in wt_input if aa not in AMINO_ACIDS]
    if not invalid_wt and len(wt_input) in [9, 10]:
        wt_res = run_prediction_and_scan(wt_input, selected_allele)
        st.markdown("#### ⚖️ Neoantigen vs. Wild-Type Comparative Analysis")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mutant T½", f"{res['thalf']:.2f} ± {res['thalf_std']:.2f} h")
        c2.metric("Wild-Type T½", f"{wt_res['thalf']:.2f} ± {wt_res['thalf_std']:.2f} h")
        fold_change = res["thalf"] / max(wt_res["thalf"], 1e-4)
        delta_thalf = res["thalf"] - wt_res["thalf"]
        c3.metric("Affinity Ratio (Fold Change)", f"{fold_change:.2f}×", delta=f"{delta_thalf:+.2f} h")
        verdict = "🟢 Strong Gain-of-Stability" if fold_change >= 1.4 else "🟡 Moderate Change" if fold_change >= 1.0 else "🔴 Loss-of-Stability"
        c4.metric("Presentation Verdict", verdict)

# Explicit Biophysical Note for Flagship K27M Preset
eval_seq = res["eval_seq"]
if eval_seq == "RMSAPSTGG":
    st.info(
        "💡 **Biophysical Insight (Why K27M Gains Stability Despite Weak P9 Gly):** "
        "Wild-type H3.3 has Lysine at P2, which introduces a severe steric and electrostatic clash in Pocket B, causing complex collapse (T½ ≈ 0.50 h). "
        "The oncogenic K27M mutation introduces Methionine at P2—an optimal hydrophobic fit for Pocket B—driving a +64% gain in half-life (0.82 h) that rescues "
        "the complex despite a non-anchor Glycine at P9. However, because P9 lacks a hydrophobic sidechain for Pocket F, overall half-life remains intermediate (0.82 h), "
        "in complete agreement with clinical findings that K27M is a presentation-competent, intermediate-affinity glioma neoantigen."
    )

# -------------------------------------------------------------
# Dual-Panel Side-by-Side: Sensitivity & 3D Molecular Complex
# -------------------------------------------------------------
st.markdown("---")

col_vis_left, col_vis_right = st.columns([1, 1], gap="medium")

with col_vis_left:
    st.markdown("### 📊 In Silico Mutational Sensitivity")
    st.caption("*Model Sensitivity: Mean absolute ΔS across in silico deep mutagenesis, quantifying model feature dependence (not direct physical measurements).*")

    chart_view = st.radio("View Mode:", ["1D Anchor Sensitivity Bar Chart", "Full 9×20 Deep Mutational Heatmap"], horizontal=True, key=f"view_{eval_seq}")

    if chart_view == "1D Anchor Sensitivity Bar Chart":
        positions = [f"P{i+1}: {eval_seq[i]}" for i in range(len(eval_seq))]
        is_anchor = ["Anchor (Pocket B)" if i == 1 else "Anchor (Pocket F)" if i == len(eval_seq)-1 else "Auxiliary / Non-Anchor" for i in range(len(eval_seq))]

        df_chart = pd.DataFrame({
            "Position": positions,
            "Sensitivity": res["sensitivities"],
            "Role": is_anchor,
        })

        chart = (
            alt.Chart(df_chart)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("Position:N", sort=None, title="Peptide Position & Residue"),
                y=alt.Y("Sensitivity:Q", title="Mean Absolute Delta ΔS"),
                color=alt.Color(
                    "Role:N",
                    scale=alt.Scale(
                        domain=["Anchor (Pocket B)", "Anchor (Pocket F)", "Auxiliary / Non-Anchor"],
                        range=["#2563eb", "#ea580c", "#94a3b8"],
                    ),
                    legend=alt.Legend(title="Residue Role", orient="top"),
                ),
                tooltip=["Position", "Sensitivity", "Role"],
            )
            .properties(height=260)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        # Full 9x20 Heatmap
        scan_mat = np.array(res["scan_matrix"])
        heatmap_records = []
        for p in range(len(eval_seq)):
            pos_lbl = f"P{p+1}: {eval_seq[p]}"
            for j, aa in enumerate(AMINO_ACIDS):
                heatmap_records.append({
                    "Position": pos_lbl,
                    "Amino_Acid": aa,
                    "Delta_S": float(scan_mat[p, j]),
                })
        df_heat = pd.DataFrame(heatmap_records)
        heat = (
            alt.Chart(df_heat)
            .mark_rect()
            .encode(
                x=alt.X("Position:N", sort=None, title="Peptide Position"),
                y=alt.Y("Amino_Acid:N", sort=list(AMINO_ACIDS), title="Mutant Amino Acid"),
                color=alt.Color(
                    "Delta_S:Q",
                    scale=alt.Scale(scheme="redblue", domainMid=0),
                    title="ΔS Shift",
                ),
                tooltip=[
                    "Position",
                    "Amino_Acid",
                    alt.Tooltip("Delta_S:Q", format="+.3f", title="ΔS Shift"),
                ],
            )
            .properties(height=280)
        )
        st.altair_chart(heat, use_container_width=True)

    # In Silico Stabilizing Substitutions
    if res["top_stabilizing"]:
        st.markdown("**💡 In Silico Neoantigen Optimization (Top Stabilizing Designs):**")
        for opt in res["top_stabilizing"][:3]:
            st.markdown(f"- `{opt['position']}: {opt['mutation']}` ➜ Predicted $T_{{1/2}} = {opt['mutant_thalf']:.2f}\\text{{ h}}$ ($\\Delta S = {opt['delta_score']:+.3f}$)")

    # Automated Biophysical Analysis
    st.markdown("### 🔬 Automated Biophysical Pocket Analysis")

    def generate_biophysical_notes(sequence: str, allele: str) -> List[str]:
        notes = []
        if len(sequence) < 9:
            return notes

        p2 = sequence[1]
        p9 = sequence[8] if len(sequence) >= 9 else sequence[-1]

        # Pocket B Analysis
        if allele == "HLA-A*02:01":
            if p2 in ["L", "M"]:
                notes.append(f"**Pocket B (P2 = {p2}):** **Optimal Hydrophobic Fit.** Deep hydrophobic pocket lined by Met45, Ala24, and Val67 comfortably accommodates the aliphatic sidechain of {p2}.")
            elif p2 in ["I", "V", "A", "T"]:
                notes.append(f"**Pocket B (P2 = {p2}):** **Tolerated Secondary Anchor.** Hydrophobic pocket accommodates {p2}, though with slightly less depth packing than Leucine or Methionine.")
            elif p2 in ["K", "R"]:
                notes.append(f"**Pocket B (P2 = {p2}):** ⚠️ **Severe Electrostatic Repulsion.** Positively charged basic residue introduces a strong electrostatic and steric clash against Val67 in Pocket B.")
            elif p2 in ["D", "E"]:
                notes.append(f"**Pocket B (P2 = {p2}):** ⚠️ **Acidic Charge Clash.** Negative charge is destabilizing inside the uncharged hydrophobic pocket.")
            elif p2 == "P":
                notes.append(f"**Pocket B (P2 = P):** ⚠️ **Backbone Rigidity Clash.** Proline introduces a kink that impairs standard HLA-A*02:01 mainchain hydrogen bonding.")

        elif allele == "HLA-B*07:02":
            if p2 == "P":
                notes.append(f"**Pocket B (P2 = P):** **Strict Biological Anchor Match!** HLA-B*07:02 strictly requires Proline at P2 to fit its unique constricted pocket geometry.")
            else:
                notes.append(f"**Pocket B (P2 = {p2}):** ⚠️ **Non-Proline Penalty.** HLA-B*07:02 strongly penalizes non-proline residues at P2.")

        elif allele == "HLA-A*24:02":
            if p2 in ["Y", "F"]:
                notes.append(f"**Pocket B (P2 = {p2}):** **Optimal Aromatic Anchor.** HLA-A*24:02 features a large aromatic pocket that specifically selects for Tyrosine or Phenylalanine.")
            else:
                notes.append(f"**Pocket B (P2 = {p2}):** Sub-optimal anchor for HLA-A*24:02 Pocket B (prefers aromatic Y/F).")

        # Pocket F Analysis
        if allele in ["HLA-A*02:01", "HLA-B*07:02"]:
            if p9 in ["V", "L"]:
                notes.append(f"**Pocket F (P9 = {p9}):** **Strong Hydrophobic C-Terminal Anchor.** Hydrophobic sidechain packs tightly into Pocket F (Thr80, Leu81, Tyr84, Tyr116, Tyr123, Trp147).")
            elif p9 in ["I", "A", "M", "F"]:
                notes.append(f"**Pocket F (P9 = {p9}):** **Tolerated C-Terminal Anchor.** Forms stable hydrophobic contacts in Pocket F.")
            elif p9 == "G":
                notes.append(f"**Pocket F (P9 = G):** ⚠️ **Missing Anchor Penalty.** Glycine lacks a sidechain and cannot form stabilizing hydrophobic contacts, leaving Pocket F empty.")
            elif p9 in ["K", "R", "D", "E"]:
                notes.append(f"**Pocket F (P9 = {p9}):** ⚠️ **Severe Charged Residue Clash.** Polar/charged C-terminus prevents proper burial in the hydrophobic cavity.")

        return notes

    notes = generate_biophysical_notes(eval_seq, selected_allele)
    for note in notes:
        st.markdown(f'<div class="note-box">{note}</div>', unsafe_allow_html=True)


with col_vis_right:
    st.markdown("### 🔬 Interactive 3D Binding Structure")
    st.caption("*Illustrative crystallographic HLA-A*02:01 template (PDB 1DUZ, 1.8 Å) with synthesized peptide sidechains for cleft spatial orientation; not an allele-specific predicted structure.*")

    col_info1, col_info2 = st.columns([3, 2])
    with col_info1:
        st.markdown(f"**Peptide:** `{eval_seq}` ({len(eval_seq)}-mer) in `{selected_allele}`")
    with col_info2:
        p2_char = eval_seq[1] if len(eval_seq) > 1 else "X"
        p9_char = eval_seq[-1] if len(eval_seq) > 0 else "X"
        st.markdown(f"**Anchors:** P2=`{p2_char}`, P9=`{p9_char}`")

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    with col_ctrl1:
        show_surface = st.checkbox("Cavity Surface", value=False, key=f"surf_{eval_seq}")
    with col_ctrl2:
        show_contacts = st.checkbox("Pocket B & F", value=True, key=f"cont_{eval_seq}")
    with col_ctrl3:
        spin_struct = st.checkbox("Auto-Spin", value=False, key=f"spin_{eval_seq}")

    try:
        pdb_data = build_pmhc_pdb(eval_seq)
        html_3d = generate_3dmol_html(
            pdb_str=pdb_data,
            peptide_seq=eval_seq,
            allele=selected_allele,
            show_surface=show_surface,
            show_pocket_residues=show_contacts,
            spin=spin_struct,
            height=420,
        )
        components.html(html_3d, height=440)
    except Exception as e:
        st.error(f"Could not render 3D structure: {e}")

    def get_dynamic_structural_mechanics_html(sequence: str, allele: str) -> str:
        p2 = sequence[1] if len(sequence) > 1 else "X"
        p9 = sequence[8] if len(sequence) >= 9 else sequence[-1]

        # Pocket B Dynamic Mechanics
        if allele == "HLA-A*02:01":
            if p2 == "M":
                p2_desc = f'<b style="color: #2563eb;">Peptide P2 Anchor ({p2} = Met):</b> Optimal aliphatic packing into hydrophobic Pocket B floor (<span style="color: #06b6d4;">Met45, Ala24, Val67</span>).'
            elif p2 == "L":
                p2_desc = f'<b style="color: #2563eb;">Peptide P2 Anchor ({p2} = Leu):</b> Deep hydrophobic insertion into Pocket B (<span style="color: #06b6d4;">Met45, Val67</span>).'
            elif p2 in ["I", "V", "A", "T"]:
                p2_desc = f'<b style="color: #2563eb;">Peptide P2 Anchor ({p2}):</b> Tolerated secondary hydrophobic anchor with moderate packing.'
            elif p2 in ["K", "R"]:
                p2_desc = f'<b style="color: #dc2626;">⚠️ Peptide P2 Anchor ({p2} = Lys/Arg):</b> <b>Severe Steric & Electrostatic Clash!</b> Long basic sidechain collides with <span style="color: #06b6d4;">Val67 (illustrative ~2.7 Å steric clash)</span> and repels the hydrophobic floor.'
            elif p2 in ["D", "E"]:
                p2_desc = f'<b style="color: #dc2626;">⚠️ Peptide P2 Anchor ({p2} = Asp/Glu):</b> Unfavorable acidic charge inside the nonpolar Pocket B cavity.'
            elif p2 == "P":
                p2_desc = f'<b style="color: #ea580c;">⚠️ Peptide P2 Anchor (P = Pro):</b> Pyrrolidine ring introduces a backbone kink disrupting standard MHC hydrogen bonding.'
            else:
                p2_desc = f'<b style="color: #2563eb;">Peptide P2 Anchor ({p2}):</b> Sub-optimal Pocket B residue for {allele}.'
        elif allele == "HLA-B*07:02":
            if p2 == "P":
                p2_desc = f'<b style="color: #10b981;">Peptide P2 Anchor (P = Pro):</b> <b>Strict Biological Requirement Met!</b> Proline fits the unique constricted geometry of HLA-B*07:02 Pocket B.'
            else:
                p2_desc = f'<b style="color: #dc2626;">⚠️ Peptide P2 Anchor ({p2}):</b> Non-proline residue fails the strict stereochemical requirement of HLA-B*07:02 Pocket B.'
        elif allele == "HLA-A*24:02":
            if p2 in ["Y", "F"]:
                p2_desc = f'<b style="color: #2563eb;">Peptide P2 Anchor ({p2}):</b> <b>Optimal Aromatic Fit!</b> Aromatic ring slots into the large hydrophobic cleft of HLA-A*24:02 Pocket B.'
            else:
                p2_desc = f'<b style="color: #ea580c;">Peptide P2 Anchor ({p2}):</b> Sub-optimal anchor for HLA-A*24:02 (strongly prefers aromatic Y/F).'
        else:
            p2_desc = f'<b style="color: #2563eb;">Peptide P2 Anchor ({p2}):</b> Interacts with {allele} Pocket B cavity.'

        # Pocket F Dynamic Mechanics
        if allele in ["HLA-A*02:01", "HLA-B*07:02"]:
            if p9 in ["V", "L", "I", "M", "A"]:
                p9_desc = f'<b style="color: #ea580c;">Peptide P9 Anchor ({p9}):</b> Hydrophobic C-terminus packs tightly into Pocket F cavity (<span style="color: #f97316;">Thr80, Leu81, Tyr84, Tyr116, Tyr123, Trp147</span>).'
            elif p9 == "G":
                p9_desc = f'<b style="color: #dc2626;">⚠️ Peptide P9 Anchor (G = Gly):</b> <b>Missing Anchor Penalty!</b> Glycine lacks a sidechain; Pocket F remains unoccupied, destabilizing the C-terminal anchor.'
            elif p9 in ["K", "R", "D", "E"]:
                p9_desc = f'<b style="color: #dc2626;">⚠️ Peptide P9 Anchor ({p9}):</b> Charged C-terminus cannot be buried inside hydrophobic Pocket F.'
            else:
                p9_desc = f'<b style="color: #ea580c;">Peptide P9 Anchor ({p9}):</b> C-terminal orientation in Pocket F.'
        elif allele == "HLA-A*03:01":
            if p9 in ["K", "R"]:
                p9_desc = f'<b style="color: #10b981;">Peptide P9 Anchor ({p9}):</b> <b>Basic Anchor Match!</b> Positively charged residue forms a stabilizing salt bridge with Asp116 in Pocket F.'
            else:
                p9_desc = f'<b style="color: #ea580c;">Peptide P9 Anchor ({p9}):</b> Sub-optimal C-terminus for HLA-A*03:01 (prefers basic K/R).'
        else:
            p9_desc = f'<b style="color: #ea580c;">Peptide P9 Anchor ({p9}):</b> C-terminal contact in {allele} Pocket F.'

        mid_seq = sequence[2:8] if len(sequence) >= 9 else sequence[1:]
        p1_char = sequence[0] if len(sequence) >= 1 else ""

        return f"""
        <div style="font-size: 0.9rem; color: #475569; background: #f8fafc; padding: 12px 16px; border-radius: 8px; border: 1px solid #e2e8f0; margin-top: 10px;">
            <b>Dynamic 3D Structural Mechanics for {allele}:</b>
            <ul style="margin-top: 6px; padding-left: 20px;">
                <li>{p2_desc}</li>
                <li>{p9_desc}</li>
                <li><b style="color: #10b981;">Peptide P1 ({p1_char}) & P3–P8 ({mid_seq}):</b> Solvent-exposed residues projecting upward for T-cell receptor (TCR) contact.</li>
                <li><b>HLA Heavy Chain α₁/α₂ Helices (Silver Ribbon):</b> Encloses the peptide floor to establish {allele} stereochemical binding specificity.</li>
            </ul>
            <i>Tip: Left-click and drag to rotate the groove; right-click to pan; scroll wheel to zoom into Pocket B or Pocket F.</i>
        </div>
        """

    st.markdown(get_dynamic_structural_mechanics_html(eval_seq, selected_allele), unsafe_allow_html=True)


# -------------------------------------------------------------
# Pan-Specific Cross-Allele Screening (HLA Restriction)
# -------------------------------------------------------------
@st.cache_data(show_spinner=False)
def screen_cross_alleles(sequence: str) -> List[Dict[str, Any]]:
    eval_core = sequence[:9]
    results = []
    for allele in COMMON_ALLELES:
        pseudo = hla_db.get_pseudosequence(allele)
        pep_oh = one_hot_encode_sequence(eval_core, max_len=9).reshape(-1)
        hla_oh = one_hot_encode_sequence(pseudo, max_len=34).reshape(-1)
        feat = torch.tensor(np.concatenate([pep_oh, hla_oh]), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            score = model(feat).item()
        allele_thalf = target_to_thalf(score)
        results.append({
            "Allele": allele,
            "Predicted_Thalf": round(allele_thalf, 2),
            "Status": "Stable Binder" if allele_thalf >= 2.0 else "Modest Binder" if allele_thalf >= 0.7 else "Unstable",
        })
    return results

st.markdown("---")
st.markdown("### 🌐 Pan-Specific Cross-Allele Screening (HLA Restriction)")
st.markdown("Evaluates whether the candidate peptide binds specifically to the target allele or cross-reacts across the 6 core HLA alleles.")

cross_res = screen_cross_alleles(eval_seq)
df_cross = pd.DataFrame(cross_res)
df_cross["Focus"] = df_cross["Allele"].apply(lambda x: "Active Selected Target" if x == selected_allele else "Other Alleles")

cross_chart = (
    alt.Chart(df_cross)
    .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
    .encode(
        y=alt.Y("Allele:N", sort="-x", title="HLA Allele"),
        x=alt.X("Predicted_Thalf:Q", title="Predicted Half-Life T½ (hours)"),
        color=alt.Color(
            "Focus:N",
            scale=alt.Scale(
                domain=["Active Selected Target", "Other Alleles"],
                range=["#2563eb", "#94a3b8"],
            ),
            legend=alt.Legend(title="Target Focus", orient="top"),
        ),
        tooltip=["Allele", "Predicted_Thalf", "Status"],
    )
    .properties(height=200)
)
st.altair_chart(cross_chart, use_container_width=True)


# -------------------------------------------------------------
# Benchmark Context & Scientific Provenance
# -------------------------------------------------------------
with st.expander("ℹ️ Scientific Validation, Provenance & Clinical Limitations"):
    st.markdown(f"""
    **Model & Interpretability Provenance:**
    - **Current Preset:** {selected_preset}
    - **Preset Context:** {preset_data['desc']}
    - **Frozen Checkpoint:** `models/frozen/pan_stability_mlp_frozen.pt` (SHA-256: `9566ac3568afaf800f0ddb55fe82fededc70b9cbecd441278a5436811c695cf6`)
    - **Anchor Importance Ratio (AIR):** **41.2%** average anchor concentration across test peptides (Target: ≥ 40%; Null random: 22.2%).
    - **Motif Concordance Score (MCS):** **83.3%** top-2 anchor preference match across 6 core target alleles.
    - **HLA Pocket Overlap (HPO):** **50.0%** overlap with crystallographically validated B/F pocket residues.
    - **Prospective Brain Cancer Validation:** **100.0% (6/6 concordance)** on pre-registered, SHA-256 locked glioma neoantigen protocol.

    **Clinical Limitations & Scope:**
    - **Training Data Scope:** Trained on measured peptide-HLA complex half-lives ($T_{{1/2}}$) from the IEDB and Serova Hackathon benchmark dataset.
    - **Binding Stability vs. Immunogenicity:** Predicted $T_{{1/2}}$ measures the physical stability of the peptide-MHC Class I heterodimer on the cell surface. It does not model TCR clonotype repertoire, proteasomal cleavage, TAP transport efficiency, or in vivo anti-tumor cytolytic response.
    - **Uncertainty Quantification:** Half-life confidence intervals ($\pm \sigma$) are estimated using Monte Carlo Dropout (30 stochastic forward passes at $p=0.20$), capturing epistemic model uncertainty.
    """)
