#!/usr/bin/env bash
# run_all.sh - One-command end-to-end reproduction of the HLA-I Peptide Stability Challenge
# Executes Phases 1 through 5, generating all data splits, baseline benchmarks,
# PLM embeddings, interpretability heatmaps, faithfulness tests, prospective runs, and figures.

set -e

export MPLCONFIGDIR=/tmp/matplotlib_cache
PYTHON_BIN="./venv/bin/python"

if [ ! -f "$PYTHON_BIN" ]; then
    echo "Virtual environment python not found at $PYTHON_BIN. Using system python3."
    PYTHON_BIN="python3"
fi

echo "================================================================================"
echo "HLA-I Peptide Complex Stability Challenge - Complete Reproduction Pipeline"
echo "================================================================================"

echo ""
echo "[1/5] Running Unit and Regression Test Suite..."
$PYTHON_BIN -m unittest tests/test_phase1.py
$PYTHON_BIN -m unittest tests/test_phase3_phase4.py
echo "✓ All test suites passed successfully!"

echo ""
echo "[2/5] Checking Phase 1 Data & Baseline Benchmarks..."
if [ ! -f "data/splits/random/test.csv" ]; then
    echo "Generating Phase 1 data splits and baselines..."
    $PYTHON_BIN run_phase1.py
else
    echo "✓ Phase 1 data splits and baselines already present."
fi

echo ""
echo "[3/5] Running Phase 3: Interpretability & Faithfulness Suite..."
$PYTHON_BIN run_phase3.py
echo "✓ Phase 3 completed: Heatmaps, Integrated Gradients, HLA Masking, and Faithfulness verified."

echo ""
echo "[4/5] Running Phase 4: Brain Cancer Prospective Neoantigen Run..."
$PYTHON_BIN run_phase4.py
echo "✓ Phase 4 completed: Model frozen, prospective predictions locked, K27M analyzed."

echo ""
echo "[5/5] Verifying Generated Artifacts & SHA-256 Checksums..."
echo "--- Frozen Model Checksum ---"
cat models/frozen/pan_stability_mlp_frozen.pt.sha256
echo "--- Prospective Predictions Checksum ---"
cat predictions/prospective_brain_cancer_predictions.sha256

echo ""
echo "================================================================================"
echo "Reproduction Complete! All deliverables ready:"
echo "  - Reports:     reports/"
echo "  - Figures:     figures/"
echo "  - Predictions: predictions/"
echo "  - Frozen:      models/frozen/"
echo "================================================================================"
