"""
antigens.py - Neuro-oncology prospective neoantigen peptide library generator.

Generates all sliding 8-mer, 9-mer, 10-mer, and 11-mer windows spanning hallmark
pediatric and adult brain cancer mutations, for both mutant and wild-type (normal) alleles:
1. Histone H3.3 K27M (H3F3A) - Hallmark driver of pediatric diffuse midline glioma (DMG / DIPG).
2. IDH1 R132H - Definitive diagnostic driver in lower-grade glioma and secondary glioblastoma.
3. EGFRvIII - Extracellular domain junction neoepitope in primary glioblastoma.
4. BRAF V600E - Oncogenic driver in pleomorphic xanthoastrocytoma and pediatric ganglioglioma.
5. TP53 R273H - High-frequency contact-site mutation in astrocytomas and glioblastoma.
"""

from typing import List, Dict, Any
import pandas as pd

# Canonical protein sequences curated from UniProt
# 1. Human Histone H3.3 (H3F3A, UniProt P84243)
# Mature chain (after cleavage of initiator Met) positions 1-45:
# ARTKQTARKSTGGKAPRKQLATKAARKSAPSTGGVKKPHRYRPGTVAL
# Position 27 (mature chain) is Lysine (K).
H3F3A_FLANKING_WT = "KQLATKAARKSAPSTGGVKKPHRY"
H3F3A_MUTATION_POS = 10  # 0-indexed position of K27 within this flanking region (K)

# 2. Human IDH1 (UniProt O75874) around R132:
# ...VSGWVKPIIIGRHAYGDQYRATDFV...
IDH1_FLANKING_WT = "VSGWVKPIIIGRHAYGDQYRATDFV"
IDH1_MUTATION_POS = 11  # R132

# 3. EGFRvIII deletion novel junction (exons 2-7 skipped):
# WT EGFR: ...LEEKK... (exon 1-2) / ...VCQGT... (exon 8)
# Novel EGFRvIII junction sequence: LEEKKGNYVVTDH
EGFRVIII_FLANKING_MUT = "ALEEKKGNYVVTDHGSCVR"
EGFRVIII_FLANKING_WT  = "ALEEKKVCQGTSNKLTQLG"
EGFRVIII_MUTATION_POS = 6  # Glycine novel junction

# 4. Human BRAF (UniProt P15056) around V600:
# ...LTVKIGDFGLATVKSRWSGSPPQG...
BRAF_FLANKING_WT = "LTVKIGDFGLATVKSRWSGSPPQG"
BRAF_MUTATION_POS = 12  # V600

# 5. Human TP53 (UniProt P04637) around R273:
# ...CNSSCMGGMNRRPILTIITLE...
TP53_FLANKING_WT = "CNSSCMGGMNRRPILTIITLE"
TP53_MUTATION_POS = 10  # R273


def extract_mutation_windows(
    wt_sequence: str,
    mut_pos_in_seq: int,
    mut_aa: str,
    gene: str,
    mutation_name: str,
    disease_context: str,
    window_lengths: List[int] = [8, 9, 10, 11]
) -> List[Dict[str, Any]]:
    """
    Extract all sliding windows of given lengths that cover the mutated residue.
    """
    records = []
    orig_aa = wt_sequence[mut_pos_in_seq]

    # Create mutant sequence
    mut_sequence = wt_sequence[:mut_pos_in_seq] + mut_aa + wt_sequence[mut_pos_in_seq + 1:]

    seq_len = len(wt_sequence)

    for k in window_lengths:
        # A window of length k covering mut_pos_in_seq can start at any index start_idx
        # such that start_idx <= mut_pos_in_seq < start_idx + k
        min_start = max(0, mut_pos_in_seq - k + 1)
        max_start = min(seq_len - k, mut_pos_in_seq)

        for start_idx in range(min_start, max_start + 1):
            end_idx = start_idx + k
            pep_wt = wt_sequence[start_idx:end_idx]
            pep_mut = mut_sequence[start_idx:end_idx]
            pos_in_pep = mut_pos_in_seq - start_idx  # 0-indexed position within peptide

            records.append({
                "gene": gene,
                "mutation": mutation_name,
                "disease_context": disease_context,
                "length": k,
                "start_idx": start_idx,
                "pos_in_pep": pos_in_pep,
                "pos_in_pep_1based": pos_in_pep + 1,
                "wt_aa": orig_aa,
                "mut_aa": mut_aa,
                "peptide_wt": pep_wt,
                "peptide_mut": pep_mut,
                "is_p2_anchor": (pos_in_pep == 1),
                "is_c_terminal_anchor": (pos_in_pep == k - 1),
            })

    return records


def generate_brain_cancer_antigen_library() -> pd.DataFrame:
    """
    Build the complete brain cancer prospective peptide candidate library.
    """
    all_records = []

    # 1. Histone H3.3 K27M
    all_records.extend(extract_mutation_windows(
        wt_sequence=H3F3A_FLANKING_WT,
        mut_pos_in_seq=H3F3A_MUTATION_POS,
        mut_aa="M",
        gene="H3F3A (Histone H3.3)",
        mutation_name="K27M",
        disease_context="Pediatric Diffuse Midline Glioma (DIPG/DMG)",
        window_lengths=[8, 9, 10, 11]
    ))

    # 2. IDH1 R132H
    all_records.extend(extract_mutation_windows(
        wt_sequence=IDH1_FLANKING_WT,
        mut_pos_in_seq=IDH1_MUTATION_POS,
        mut_aa="H",
        gene="IDH1",
        mutation_name="R132H",
        disease_context="Lower-Grade Glioma & Secondary Glioblastoma",
        window_lengths=[8, 9, 10, 11]
    ))

    # 3. EGFRvIII
    # For EGFRvIII, mutant sequence has GNYV insertion; extract direct sliding windows
    for k in [8, 9, 10, 11]:
        for start_idx in range(len(EGFRVIII_FLANKING_MUT) - k + 1):
            pep_mut = EGFRVIII_FLANKING_MUT[start_idx:start_idx + k]
            pep_wt = EGFRVIII_FLANKING_WT[start_idx:start_idx + k] if start_idx + k <= len(EGFRVIII_FLANKING_WT) else ""
            all_records.append({
                "gene": "EGFR",
                "mutation": "EGFRvIII (delta 2-7)",
                "disease_context": "Primary Glioblastoma (GBM)",
                "length": k,
                "start_idx": start_idx,
                "pos_in_pep": 6 - start_idx if 0 <= (6 - start_idx) < k else -1,
                "pos_in_pep_1based": (6 - start_idx + 1) if 0 <= (6 - start_idx) < k else -1,
                "wt_aa": "Exon2-7",
                "mut_aa": "Junction",
                "peptide_wt": pep_wt,
                "peptide_mut": pep_mut,
                "is_p2_anchor": False,
                "is_c_terminal_anchor": False,
            })

    # 4. BRAF V600E
    all_records.extend(extract_mutation_windows(
        wt_sequence=BRAF_FLANKING_WT,
        mut_pos_in_seq=BRAF_MUTATION_POS,
        mut_aa="E",
        gene="BRAF",
        mutation_name="V600E",
        disease_context="Pleomorphic Xanthoastrocytoma & Ganglioglioma",
        window_lengths=[8, 9, 10, 11]
    ))

    # 5. TP53 R273H
    all_records.extend(extract_mutation_windows(
        wt_sequence=TP53_FLANKING_WT,
        mut_pos_in_seq=TP53_MUTATION_POS,
        mut_aa="H",
        gene="TP53",
        mutation_name="R273H",
        disease_context="High-Grade Astrocytoma & Glioblastoma",
        window_lengths=[8, 9, 10, 11]
    ))

    df = pd.DataFrame(all_records)
    return df
