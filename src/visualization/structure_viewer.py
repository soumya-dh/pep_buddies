"""
structure_viewer.py - 3D Molecular Structure generator for HLA-peptide complexes.

Transforms crystallographic templates (e.g. HLA-A*02:01 1DUZ) into sequence-matched
complexes and produces self-contained 3Dmol.js HTML widgets for Streamlit.
"""

import os
from typing import Optional, List, Dict

AA_3LETTER: Dict[str, str] = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
    "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
    "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
    "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
}

# Validated Contact Residues in HLA-A*02:01
POCKET_B_RESIDUES = [9, 45, 63, 66, 67, 70]
POCKET_F_RESIDUES = [77, 80, 81, 116, 123, 143, 146, 147]


def build_pmhc_pdb(
    peptide_seq: str,
    template_pdb_path: str = "data/structures/hla_a0201_groove.pdb"
) -> str:
    """
    Mutate Chain C residues of template PDB to match the user's input peptide sequence.
    Preserves crystallographic backbone and coordinates while updating residue identity.
    """
    if not os.path.exists(template_pdb_path):
        raise FileNotFoundError(f"Template PDB not found at {template_pdb_path}")

    with open(template_pdb_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    modified_lines: List[str] = []
    pep_len = len(peptide_seq)

    for line in lines:
        if line.startswith("ATOM") and len(line) >= 26 and line[21] == "C":
            try:
                res_num = int(line[22:26].strip())
            except ValueError:
                res_num = 1

            if 1 <= res_num <= pep_len:
                one_letter = peptide_seq[res_num - 1]
                three_letter = AA_3LETTER.get(one_letter, "ALA")
                # PDB ATOM format: col 18-20 (0-indexed 17:20) is 3-letter res name
                line = line[:17] + f"{three_letter:>3}" + line[20:]
                modified_lines.append(line)
        elif line.startswith("TER") and len(line) >= 22 and line[21] == "C":
            modified_lines.append(line)
        else:
            # HLA heavy chain (Chain A) or other structural lines
            modified_lines.append(line)

    return "".join(modified_lines)


def generate_3dmol_html(
    pdb_str: str,
    peptide_seq: str,
    allele: str = "HLA-A*02:01",
    show_surface: bool = False,
    show_pocket_residues: bool = True,
    spin: bool = False,
    height: int = 480
) -> str:
    """
    Generate an offline-capable, interactive 3Dmol.js HTML snippet.
    Embeds local static/3Dmol-min.js if available, falling back to CDN.
    """
    # Try embedding local offline JS
    local_js_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "static", "3Dmol-min.js")
    js_tag = ""
    if os.path.exists(local_js_path):
        with open(local_js_path, "r", encoding="utf-8") as f:
            js_tag = f"<script>{f.read()}</script>"
    else:
        js_tag = '<script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>'

    # Escape newlines and backslashes for JS template literal
    escaped_pdb = pdb_str.replace("\\", "\\\\").replace("`", "\\`")

    p2_res = peptide_seq[1] if len(peptide_seq) >= 2 else "X"
    p9_res = peptide_seq[8] if len(peptide_seq) >= 9 else peptide_seq[-1]

    pocket_b_str = str(POCKET_B_RESIDUES)
    pocket_f_str = str(POCKET_F_RESIDUES)

    surface_code = ""
    if show_surface:
        surface_code = """
        viewer.addSurface($3Dmol.SurfaceType.VDW, {
            opacity: 0.35,
            color: '#64748b'
        }, {chain: 'A'});
        """

    pocket_residues_code = ""
    if show_pocket_residues:
        pocket_residues_code = f"""
        // Pocket B contact residues (cyan)
        viewer.addStyle({{chain: 'A', resi: {pocket_b_str}}}, {{
            stick: {{color: '#06b6d4', radius: 0.16, opacity: 0.95}}
        }});
        // Pocket F contact residues (orange)
        viewer.addStyle({{chain: 'A', resi: {pocket_f_str}}}, {{
            stick: {{color: '#f97316', radius: 0.16, opacity: 0.95}}
        }});
        """

    spin_code = "viewer.spin(true);" if spin else ""

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                margin: 0;
                padding: 0;
                background-color: #0b1329;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                overflow: hidden;
            }}
            #viewport-container {{
                width: 100%;
                height: {height}px;
                position: relative;
                border-radius: 8px;
                border: 1px solid #1e293b;
            }}
            .legend-overlay {{
                position: absolute;
                top: 12px;
                left: 14px;
                background: rgba(15, 23, 42, 0.88);
                color: #e2e8f0;
                padding: 10px 14px;
                border-radius: 8px;
                font-size: 12px;
                line-height: 1.5;
                backdrop-filter: blur(4px);
                border: 1px solid #334155;
                z-index: 100;
                pointer-events: none;
            }}
            .legend-item {{
                display: flex;
                align-items: center;
                margin-bottom: 4px;
            }}
            .legend-dot {{
                width: 10px;
                height: 10px;
                border-radius: 50%;
                margin-right: 8px;
                display: inline-block;
            }}
            .controls-overlay {{
                position: absolute;
                bottom: 12px;
                right: 14px;
                background: rgba(15, 23, 42, 0.85);
                color: #94a3b8;
                padding: 6px 12px;
                border-radius: 6px;
                font-size: 11px;
                border: 1px solid #334155;
                z-index: 100;
                pointer-events: none;
            }}
        </style>
        {js_tag}
    </head>
    <body>
        <div id="viewport-container">
            <div class="legend-overlay">
                <div style="font-weight: 700; color: #f8fafc; margin-bottom: 6px;">🧬 {allele} Binding Groove</div>
                <div class="legend-item"><span class="legend-dot" style="background: #3b82f6;"></span><b>Peptide P2:</b> {p2_res} (Pocket B Anchor)</div>
                <div class="legend-item"><span class="legend-dot" style="background: #ea580c;"></span><b>Peptide P9:</b> {p9_res} (Pocket F Anchor)</div>
                <div class="legend-item"><span class="legend-dot" style="background: #10b981;"></span><b>Peptide Non-Anchors:</b> P1, P3–P8</div>
                <div class="legend-item"><span class="legend-dot" style="background: #06b6d4;"></span><b>Pocket B Residues:</b> 9, 45, 63, 66, 67, 70</div>
                <div class="legend-item"><span class="legend-dot" style="background: #f97316;"></span><b>Pocket F Residues:</b> 77, 80, 81, 116, 123...</div>
            </div>
            <div class="controls-overlay">
                🖱️ Left-Click: Rotate | Right-Click: Pan | Scroll: Zoom
            </div>
        </div>

        <script>
            (function() {{
                const element = document.getElementById('viewport-container');
                const config = {{ backgroundColor: '#0b1329' }};
                const viewer = $3Dmol.createViewer(element, config);

                const pdbData = `{escaped_pdb}`;
                viewer.addModel(pdbData, "pdb");

                // HLA Heavy Chain (Chain A): Cartoon ribbon in subtle silver-slate
                viewer.setStyle({{chain: 'A'}}, {{
                    cartoon: {{color: '#94a3b8', opacity: 0.72}}
                }});

                {pocket_residues_code}
                {surface_code}

                // Peptide (Chain C): Distinctive colored sticks & spheres
                viewer.setStyle({{chain: 'C'}}, {{
                    stick: {{color: '#10b981', radius: 0.24}}
                }});

                // Highlight P2 Anchor (Pocket B) in Royal Blue
                viewer.addStyle({{chain: 'C', resi: 2}}, {{
                    stick: {{color: '#2563eb', radius: 0.38}},
                    sphere: {{color: '#3b82f6', radius: 0.45}}
                }});

                // Highlight P9 Anchor (Pocket F) in Fiery Orange
                viewer.addStyle({{chain: 'C', resi: 9}}, {{
                    stick: {{color: '#c2410c', radius: 0.38}},
                    sphere: {{color: '#ea580c', radius: 0.45}}
                }});

                // 3D Floating Labels
                viewer.addLabel("P2: {p2_res}", {{
                    fontSize: 12,
                    fontColor: "#ffffff",
                    backgroundColor: "#1d4ed8",
                    backgroundOpacity: 0.9,
                    borderColor: "#60a5fa",
                    borderWidth: 1.0,
                    borderRadius: 4
                }}, {{chain: 'C', resi: 2}});

                viewer.addLabel("P9: {p9_res}", {{
                    fontSize: 12,
                    fontColor: "#ffffff",
                    backgroundColor: "#c2410c",
                    backgroundOpacity: 0.9,
                    borderColor: "#fb923c",
                    borderWidth: 1.0,
                    borderRadius: 4
                }}, {{chain: 'C', resi: 9}});

                // Focus camera directly onto the peptide binding groove
                viewer.zoomTo({{chain: 'C'}}, 800);
                {spin_code}
                viewer.render();
            }})();
        </script>
    </body>
    </html>
    """
    return html
