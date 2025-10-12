#!/bin/bash
# Training script for filtered dataset (ДМФА only, 340 samples)

echo "=========================================="
echo "Training Inverse Design Pipeline"
echo "Dataset: ДМФА only, no Y metal (338/380)"
echo "Pipeline: 8 stages (solvent removed)"
echo "Metals: 6 classes (Cu, Al, Fe, Zr, Zn, La)"
echo "=========================================="
echo ""

cd /home/ruslan_safaev/adsorb_synth/adsorb_synthesis

# Clean Python cache
echo "[1/3] Cleaning Python cache..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# Run training
echo "[2/3] Running training (this may take 5-15 minutes)..."
.venv/bin/python scripts/train_inverse_design.py \
    --data data/SEC_SYN_with_features_DMFA_only_no_Y.csv \
    --output artifacts

# Check results
echo ""
echo "[3/3] Training completed!"
echo ""
echo "Expected improvements:"
echo "  - Metal: ~0.51-0.52 balanced_acc (stable)"
echo "  - Ligand: ~0.55-0.57 balanced_acc (stable)"
echo "  - Salt_mass: MAE ~0.35-0.47 (improved)"
echo "  - Solvent_volume: R² ~0.77-0.83 (stable)"
echo "  - Temperature stages: ~0.70-0.80 balanced_acc (stable)"
echo ""
echo "Solvent stage removed (was 0.286 balanced_acc)"
echo "=========================================="
