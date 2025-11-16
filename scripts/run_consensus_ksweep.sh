#!/bin/bash
#
# Script: run_consensus_ksweep.sh
# Purpose: Evaluate consensus genes using existing transfer_learning_trainer.py
# Author: Phuong
# Created: 2024-11-15
#
# Strategy: Wrapper approach - leverage existing working code!

# Set Python path
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Configuration
GENE_LISTS_DIR="results/biomarker_ksweep_transfer/gene_lists"
OUTPUT_DIR="results/consensus_ksweep_evaluation"
SEED=42

# K values to test (representative subset)
K_VALUES="90 95 100 120 140 150"

# Hyperparameter files (update paths if needed)
SOURCE_PARAMS="results/hyperparam_FIXED_tcga_20251109_194909/best_params.json"
TARGET_PARAMS="results/hyperparam_FIXED_orien_20251109_195430/best_params.json"

echo "=================================================="
echo "CONSENSUS K-SWEEP: WRAPPER APPROACH"
echo "=================================================="
echo ""
echo "Strategy: Use existing transfer_learning_trainer.py"
echo "  ✓ Proven code that already works"
echo "  ✓ No compatibility issues"
echo "  ✓ Faster and more reliable"
echo ""
echo "Configuration:"
echo "  Gene lists: ${GENE_LISTS_DIR}"
echo "  Output: ${OUTPUT_DIR}"
echo "  K values: ${K_VALUES}"
echo "  Seed: ${SEED}"
echo "  Source params: ${SOURCE_PARAMS}"
echo "  Target params: ${TARGET_PARAMS}"
echo ""
echo "Estimated runtime: ~40-60 min per k value"
echo "Total: ~4-6 hours for 6 k values"
echo ""
echo "Starting evaluation..."
echo ""

# Run wrapper
python scripts/train_consensus_ksweep.py \
    --k_values ${K_VALUES} \
    --gene_lists_dir ${GENE_LISTS_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --seed ${SEED} \
    --source_params ${SOURCE_PARAMS} \
    --target_params ${TARGET_PARAMS}

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
    echo "  - RECOMMENDATION.txt              (optimal k)"
    echo "  - consensus_ksweep_full_results.json"
    echo "  - k*/                             (models for each k)"
    echo ""
    echo "Next steps:"
    echo "  1. Review summary: cat ${OUTPUT_DIR}/consensus_ksweep_summary.csv"
    echo "  2. Check recommendation: cat ${OUTPUT_DIR}/RECOMMENDATION.txt"
    echo "  3. Proceed with optimal k for patient stratification"
    echo ""
else
    echo ""
    echo "❌ Evaluation failed. Check error messages above."
    exit 1
fi