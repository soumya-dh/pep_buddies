"""
hla_database.py - IPD-IMGT/HLA sequence lookup and 34-residue pocket pseudo-sequence generator.
"""

import os
import re
import logging
from typing import Dict, Optional, Tuple, List
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Nielsen et al. (NetMHCpan / NetMHCstabpan) 34 key contact residues
# Positions are 1-indexed relative to the mature HLA-A/B/C heavy chain (after 24-aa signal peptide)
MHC_I_PSEUDO_POSITIONS_1BASED = [
    7, 9, 24, 45, 59, 62, 63, 66, 67, 69, 70, 73, 74, 76, 77, 80, 81, 84,
    95, 97, 99, 114, 116, 118, 143, 147, 150, 152, 156, 158, 159, 163, 167, 171
]

DEFAULT_REF_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "hla_reference")


class HLADatabase:
    """
    HLA database for mapping allele names to full precursor sequences,
    mature G-domain sequences (alpha 1 and alpha 2 binding groove),
    and the 34-residue Nielsen pocket pseudo-sequence.
    """

    def __init__(self, ref_dir: Optional[str] = None):
        self.ref_dir = os.path.abspath(ref_dir or DEFAULT_REF_DIR)
        self.pseudo_dict: Dict[str, str] = {}
        self.imgt_full_seqs: Dict[str, str] = {}
        self.mature_gdomain_seqs: Dict[str, str] = {}
        
        self._load_pseudosequence_db()
        self._load_imgt_fastas()

    def _load_pseudosequence_db(self):
        """Load canonical NetMHC/NetMHCstabpan pseudo-sequences from MHC_pseudo.dat."""
        pseudo_file = os.path.join(self.ref_dir, "MHC_pseudo.dat")
        if not os.path.exists(pseudo_file):
            logger.warning(f"MHC_pseudo.dat not found at {pseudo_file}. Pseudo-sequences will be computed dynamically.")
            return

        with open(pseudo_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    allele_netmhc, pseq = parts[0], parts[1]
                    self.pseudo_dict[allele_netmhc] = pseq
                    
                    # Also index standard IMGT representation
                    # E.g. HLA-A02:01 -> HLA-A*02:01
                    match = re.match(r"^HLA-([ABC])(\d{2}):(\d{2})$", allele_netmhc)
                    if match:
                        imgt_key = f"HLA-{match.group(1)}*{match.group(2)}:{match.group(3)}"
                        self.pseudo_dict[imgt_key] = pseq
                        self.pseudo_dict[f"{match.group(1)}*{match.group(2)}:{match.group(3)}"] = pseq
                        self.pseudo_dict[f"{match.group(1)}{match.group(2)}{match.group(3)}"] = pseq

        logger.info(f"Loaded {len(self.pseudo_dict)} pseudo-sequence entries from MHC_pseudo.dat.")

    def _load_imgt_fastas(self):
        """Load IPD-IMGT/HLA FASTA files for HLA-A, B, and C."""
        for gene in ["A", "B", "C"]:
            fasta_path = os.path.join(self.ref_dir, f"{gene}_prot.fasta")
            if not os.path.exists(fasta_path):
                continue
                
            cur_id = None
            cur_seq = []
            with open(fasta_path, "r") as f:
                for line in f:
                    if line.startswith(">HLA:"):
                        if cur_id and cur_seq:
                            seq = "".join(cur_seq)
                            self._index_imgt_sequence(cur_id, seq)
                        parts = line.split()
                        cur_id = parts[1]  # e.g. A*02:01:01:01
                        cur_seq = []
                    else:
                        cur_seq.append(line.strip())
                        
            if cur_id and cur_seq:
                self._index_imgt_sequence(cur_id, "".join(cur_seq))
                
        logger.info(f"Indexed {len(self.imgt_full_seqs)} IMGT allele sequences.")

    def _index_imgt_sequence(self, allele_id: str, seq: str):
        """Index IMGT sequence under full and 4-digit keys."""
        self.imgt_full_seqs[allele_id] = seq
        self.imgt_full_seqs[f"HLA-{allele_id}"] = seq
        
        # 4-digit key (e.g. A*02:01 from A*02:01:01:01)
        parts = allele_id.split(":")
        if len(parts) >= 2:
            four_digit = f"{parts[0]}:{parts[1]}"
            if four_digit not in self.imgt_full_seqs:
                self.imgt_full_seqs[four_digit] = seq
                self.imgt_full_seqs[f"HLA-{four_digit}"] = seq

            # Also compute mature G-domain (182 aa) if length >= 206 (24 aa leader + 182 aa groove)
            if len(seq) >= 206 and four_digit not in self.mature_gdomain_seqs:
                mature = seq[24:]
                gdomain = mature[:182]
                self.mature_gdomain_seqs[four_digit] = gdomain
                self.mature_gdomain_seqs[f"HLA-{four_digit}"] = gdomain

    def extract_pseudosequence_from_mature(self, mature_seq: str) -> str:
        """
        Extract the 34-residue pocket pseudo-sequence directly from a mature HLA sequence.
        
        Parameters
        ----------
        mature_seq : str
            Mature HLA protein sequence starting at residue 1 (GSHSMRY...)
            
        Returns
        -------
        str : 34-character pocket pseudo-sequence
        """
        chars = []
        for pos in MHC_I_PSEUDO_POSITIONS_1BASED:
            idx = pos - 1
            if idx < len(mature_seq):
                chars.append(mature_seq[idx])
            else:
                chars.append("-")
        return "".join(chars)

    def get_pseudosequence(self, allele: str) -> Optional[str]:
        """
        Get the 34-residue pseudo-sequence for a given allele.
        Handles mutations such as (C67S).
        """
        if not allele:
            return None
        allele_clean = allele.strip().upper()
        
        # Check for mutation annotation like (C67S)
        has_c67s = "C67S" in allele_clean or "(C67S)" in allele_clean
        base_allele = re.sub(r"\(.*?\)", "", allele_clean).strip()
        
        # 1. Direct dictionary lookup
        pseq = self.pseudo_dict.get(base_allele)
        if not pseq:
            # Try netmhc format HLA-A02:01
            match = re.match(r"^HLA-([ABC])\*?(\d{2}):?(\d{2})", base_allele)
            if match:
                g, grp, prot = match.groups()
                pseq = self.pseudo_dict.get(f"HLA-{g}{grp}:{prot}") or self.pseudo_dict.get(f"HLA-{g}*{grp}:{prot}")
        
        # 2. Dynamic extraction from IMGT sequence if not in precomputed dictionary
        if not pseq and base_allele in self.mature_gdomain_seqs:
            pseq = self.extract_pseudosequence_from_mature(self.mature_gdomain_seqs[base_allele])
            
        if pseq and has_c67s:
            # In the 34-mer, position 67 corresponds to index 8 (the 9th residue)
            # Contact positions: 7(0), 9(1), 24(2), 45(3), 59(4), 62(5), 63(6), 66(7), 67(8)
            pseq = pseq[:8] + "S" + pseq[9:]
            
        return pseq

    def get_mature_sequence(self, allele: str) -> Optional[str]:
        """Get mature G-domain (182 aa) sequence for an allele."""
        base_allele = re.sub(r"\(.*?\)", "", allele).strip().upper()
        has_c67s = "C67S" in allele.upper()
        
        seq = self.mature_gdomain_seqs.get(base_allele)
        if not seq:
            match = re.match(r"^HLA-([ABC])\*?(\d{2}):?(\d{2})", base_allele)
            if match:
                g, grp, prot = match.groups()
                seq = self.mature_gdomain_seqs.get(f"HLA-{g}*{grp}:{prot}") or self.mature_gdomain_seqs.get(f"{g}*{grp}:{prot}")
                
        if seq and has_c67s and len(seq) >= 67:
            # mature position 67 (0-indexed: 66)
            seq = seq[:66] + "S" + seq[67:]
            
        return seq

    def enrich_dataframe(self, df: pd.DataFrame, allele_col: str = "allele") -> pd.DataFrame:
        """
        Enrich a dataframe with full mature HLA sequence and 34-residue pseudo-sequence.
        """
        df_out = df.copy()
        
        pseudoseqs = []
        mature_seqs = []
        
        for a in df_out[allele_col]:
            pseq = self.get_pseudosequence(a)
            mseq = self.get_mature_sequence(a)
            pseudoseqs.append(pseq)
            mature_seqs.append(mseq)
            
        df_out["hla_pseudoseq_db"] = pseudoseqs
        df_out["hla_mature_seq_db"] = mature_seqs
        
        return df_out


if __name__ == "__main__":
    db = HLADatabase()
    print("Testing HLA-A*02:01:")
    print("  Pseudo-sequence:", db.get_pseudosequence("HLA-A*02:01"))
    print("  Mature G-domain (first 40 aa):", db.get_mature_sequence("HLA-A*02:01")[:40])
    
    print("\nTesting HLA-B*14:02(C67S):")
    print("  Pseudo-sequence:", db.get_pseudosequence("HLA-B*14:02(C67S)"))
    print("  Wild-type Pseudo:", db.get_pseudosequence("HLA-B*14:02"))
