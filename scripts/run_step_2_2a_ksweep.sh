#!/bin/bash
#
# Script: run_step_2_2a_ksweep.sh
# Purpose: Run k-sweep analysis from Step 2.1 aggregated importance scores
# Author: Claude (for Phuong's dissertation)
# Created: 2024-11-17
#
# This script identifies optimal k values by analyzing importance score rankings
# without retraining models (fast - completes in ~5 minutes)

export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Configuration
IMPORTANCE_FILE="results_v2/02_biomarker_discovery/aggregated_gene_importances.csv"
OUTPUT_DIR="results_v2/02_biomarker_discovery/ksweep_analysis"
COX_GENES="data/raw/cox_consensus_genes_20.txt"

# K values to test (comprehensive range)
K_VALUES="50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200"

echo "=================================================="
echo "STEP 2.2A: K-SWEEP ANALYSIS (FAST)"
echo "=================================================="
echo ""
echo "Strategy: Analyze importance score rankings"
echo "  ✓ No model retraining required"
echo "  ✓ Fast execution (~5 minutes)"
echo "  ✓ Identifies candidate k values for validation"
echo ""
echo "Configuration:"
echo "  Input: ${IMPORTANCE_FILE}"
echo "  Output: ${OUTPUT_DIR}"
echo "  K values: ${K_VALUES}"
echo ""
echo "Starting analysis..."
echo ""

# Run k-sweep
python scripts/step2_2a_ksweep_from_aggregated.py \
    --importance_file ${IMPORTANCE_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --k_values ${K_VALUES} \
    --cox_genes ${COX_GENES}

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✅ STEP 2.2A COMPLETED SUCCESSFULLY"
    echo "=================================================="
    echo ""
    echo "Results saved in: ${OUTPUT_DIR}/"
    echo ""
    echo "Key files:"
    echo "  - ksweep_summary.csv           (comparison table)"
    echo "  - ksweep_analysis.png          (visualizations)"
    echo "  - RECOMMENDATIONS.json         (optimal k values)"
    echo "  - gene_lists/k*.txt            (gene lists for each k)"
    echo ""
    echo "Next steps:"
    echo "  1. Review: cat ${OUTPUT_DIR}/RECOMMENDATIONS.json"
    echo "  2. Check visualization: ${OUTPUT_DIR}/ksweep_analysis.png"
    echo "  3. Proceed to Step 2.2B with recommended k values"
    echo ""
else
    echo ""
    echo "❌ Step 2.2A failed. Check error messages above."
    exit 1
fi
