#!/bin/bash
#
# Script: run_consensus_ksweep.sh
# Purpose: Train and evaluate transfer learning with consensus genes
# Author: Phuong
# Created: 2024-11-15
#
# This script tests multiple k values to find optimal gene count
# based on C-index performance (not just stability metrics).

# Set Python path
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Configuration
GENE_LISTS_DIR="results/biomarker_ksweep_transfer/gene_lists"
OUTPUT_DIR="results/consensus_ksweep_evaluation"
SEED=42

# K values to test (representative subset - recommended)
# These cover the range from 32 to 87 consensus genes
K_VALUES="90 95 100 120 140 150"

# Alternative: Test all k values (uncomment if you want comprehensive analysis)
# K_VALUES="60 70 80 90 95 100 110 120 130 140 150"

echo "=================================================="
echo "CONSENSUS K-SWEEP: PERFORMANCE-BASED SELECTION"
echo "=================================================="
echo ""
echo "Configuration:"
echo "  Gene lists dir: ${GENE_LISTS_DIR}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  K values: ${K_VALUES}"
echo "  Random seed: ${SEED}"
echo ""
echo "This will:"
echo "  1. Load consensus genes for each k value"
echo "  2. Train transfer learning models (TCGA↔ORIEN)"
echo "  3. Evaluate C-index for each k"
echo "  4. Recommend optimal k based on performance"
echo ""
echo "Estimated runtime: ~30-40 min per k value"
echo "Total: ~2-4 hours for selected k values"
echo ""
echo "Starting evaluation..."
echo ""

# Run evaluation
python scripts/train_consensus_ksweep.py \
    --k_values ${K_VALUES} \
    --gene_lists_dir ${GENE_LISTS_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --seed ${SEED}

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✅ EVALUATION COMPLETED SUCCESSFULLY"
    echo "=================================================="
    echo ""
    echo "Results saved in: ${OUTPUT_DIR}/"
    echo ""
    echo "Key files:"
    echo "  - consensus_ksweep_summary.csv    (comparison table)"
    echo "  - RECOMMENDATION.txt              (optimal k selection)"
    echo "  - consensus_ksweep_full_results.json"
    echo ""
    echo "Next steps:"
    echo "  1. Review: ${OUTPUT_DIR}/consensus_ksweep_summary.csv"
    echo "  2. Check: ${OUTPUT_DIR}/RECOMMENDATION.txt"
    echo "  3. Select optimal k based on C-index"
    echo "  4. Proceed with final biomarker analysis"
    echo ""
else
    echo ""
    echo "❌ Evaluation failed. Check error messages above."
    exit 1
fi
