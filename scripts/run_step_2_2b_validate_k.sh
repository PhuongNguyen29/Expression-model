#!/bin/bash
#
# Script: run_step_2_2b_validate_k.sh
# Purpose: Validate k values by training models with consensus genes
# Author: Claude (for Phuong's dissertation)
# Created: 2024-11-17
#
# This script trains models for each k value to validate that the consensus
# genes are sufficient for good predictive performance

export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Configuration
GENE_LISTS_DIR="results_v2/02_biomarker_discovery/ksweep_analysis/gene_lists"
TCGA_PARAMS="results_v2/01_hyperparameter_tuning/tcga_308genes/best_params.json"
ORIEN_PARAMS="results_v2/01_hyperparameter_tuning/orien_308genes/best_params.json"
OUTPUT_DIR="results_v2/02_biomarker_discovery/k_validation"

# K values to validate (your selection: 80-150)
K_VALUES="80 90 100 110 120 130 140 150"

# Multi-seed validation
SEEDS="42 123 456 789 1011"

# Training epochs
MAX_EPOCHS=100

echo "=================================================="
echo "STEP 2.2B: K-VALUE VALIDATION (MULTI-SEED)"
echo "=================================================="
echo ""
echo "Strategy: Train models with consensus genes"
echo "  ✓ Full dataset training"
echo "  ✓ Multi-seed (5 seeds) for robustness"
echo "  ✓ Evaluate C-index for each k"
echo "  ✓ Find where performance plateaus"
echo ""
echo "Configuration:"
echo "  Gene lists: ${GENE_LISTS_DIR}"
echo "  TCGA params: ${TCGA_PARAMS}"
echo "  ORIEN params: ${ORIEN_PARAMS}"
echo "  Output: ${OUTPUT_DIR}"
echo "  K values: ${K_VALUES}"
echo "  Seeds: ${SEEDS}"
echo "  Max epochs: ${MAX_EPOCHS}"
echo ""
echo "Estimated runtime: ~2.5-3.5 hours per k value"
echo "Total: ~20-28 hours for 8 k values"
echo ""
echo "Starting validation..."
echo ""

# Run validation
python scripts/step2_2b_validate_k_values.py \
    --gene_lists_dir ${GENE_LISTS_DIR} \
    --tcga_params ${TCGA_PARAMS} \
    --orien_params ${ORIEN_PARAMS} \
    --output_dir ${OUTPUT_DIR} \
    --k_values ${K_VALUES} \
    --seeds ${SEEDS} \
    --max_epochs ${MAX_EPOCHS}

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✅ STEP 2.2B COMPLETED SUCCESSFULLY"
    echo "=================================================="
    echo ""
    echo "Results saved in: ${OUTPUT_DIR}/"
    echo ""
    echo "Key files:"
    echo "  - k_validation_summary.csv           (performance table)"
    echo "  - k_validation_performance.png       (visualizations)"
    echo "  - FINAL_RECOMMENDATION.json          (optimal k)"
    echo "  - k{080..150}/seed_*/                (trained models)"
    echo ""
    echo "Next steps:"
    echo "  1. Review: cat ${OUTPUT_DIR}/FINAL_RECOMMENDATION.json"
    echo "  2. Check visualization: ${OUTPUT_DIR}/k_validation_performance.png"
    echo "  3. Proceed to Step 3 with optimal k value"
    echo ""
else
    echo ""
    echo "❌ Step 2.2B failed. Check error messages above."
    exit 1
fi
