#!/bin/bash
#
# Script: run_biomarker_ksweep.sh
# Purpose: Run k-value sweep to find optimal biomarker count
# Author: Phuong
# Created: 2024-11-15
#
# This script runs the k-sweep analysis to determine the optimal number
# of genes (k) to extract from transfer learning models that yields
# ~20-30 consensus biomarkers with maximum stability.

# Set Python path
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Configuration
MODELS_DIR="results/transfer_learning"
OUTPUT_DIR="results/biomarker_ksweep_transfer"
SEEDS="42 123 456 789 1011"
K_VALUES="60 70 80 90 95 100 110 120 130 140 150"

echo "=================================================="
echo "BIOMARKER K-SWEEP ANALYSIS"
echo "=================================================="
echo ""
echo "Configuration:"
echo "  Models directory: ${MODELS_DIR}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Seeds: ${SEEDS}"
echo "  K values: ${K_VALUES}"
echo ""
echo "Starting k-sweep..."
echo ""

# Run k-sweep
python scripts/biomarker_ksweep.py \
    --models_dir ${MODELS_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --k_values ${K_VALUES} \
    --seeds ${SEEDS} \
    --min_appearances 3

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✅ K-SWEEP COMPLETED SUCCESSFULLY"
    echo "=================================================="
    echo ""
    echo "Results saved in: ${OUTPUT_DIR}/"
    echo ""
    echo "Next steps:"
    echo "  1. Review: ${OUTPUT_DIR}/ksweep_summary_table.csv"
    echo "  2. Check visualization: ${OUTPUT_DIR}/ksweep_analysis.png"
    echo "  3. Look at recommendations in the output above"
    echo "  4. Select optimal k value (probably k=95)"
    echo ""
else
    echo ""
    echo "❌ K-sweep failed. Check error messages above."
    exit 1
fi
