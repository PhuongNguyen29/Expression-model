#!/bin/bash


# Set Python path
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Configuration
TCGA_MODEL="results/biomarker_IMPROVED_20251111_011201/tcga_model.pth"
ORIEN_MODEL="results/biomarker_IMPROVED_20251111_011201/orien_model.pth"
TCGA_PARAMS="results/hyperparam_FIXED_tcga_20251109_194909/best_params.json"
ORIEN_PARAMS="results/hyperparam_FIXED_orien_20251109_195430/best_params.json"
COX_GENES="data/raw/cox_consensus_genes_20.txt"

# K values (no commas, space-separated)
K_VALUES="70 75 80 85 90 95 100 105 110 115 120 125 130 135 140 145 150"

# Random seeds (following machine learning best practices)
SEEDS=(42 123 456 789 1011)

# Base output directory
BASE_OUTPUT_DIR="results/multiseed_validation_$(date +"%Y%m%d_%H%M%S")"
mkdir -p "$BASE_OUTPUT_DIR"

echo ""
echo "Configuration:"
echo "  TCGA Model: $TCGA_MODEL"
echo "  ORIEN Model: $ORIEN_MODEL"
echo "  Seeds: ${SEEDS[@]}"
echo "  Number of seeds: ${#SEEDS[@]}"
echo "  K Values: $K_VALUES"
echo "  Base output: $BASE_OUTPUT_DIR"
echo ""

# Verify required files exist
echo "Verifying input files..."
for FILE in "$TCGA_MODEL" "$ORIEN_MODEL" "$TCGA_PARAMS" "$ORIEN_PARAMS" "$COX_GENES"; do
    if [ ! -f "$FILE" ]; then
        echo "ERROR: Required file not found: $FILE"
        exit 1
    fi
    echo "  ✓ $FILE"
done
echo ""

# Track success/failure
TOTAL_SEEDS=${#SEEDS[@]}
SUCCESSFUL=0
FAILED=0

# Run for each seed
for SEED in "${SEEDS[@]}"; do
    echo "================================================================"
    echo "Running with SEED = $SEED (${SUCCESSFUL}/${TOTAL_SEEDS} completed)"
    echo "================================================================"
    echo "Started: $(date)"
    
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/seed${SEED}"
    LOG_FILE="${BASE_OUTPUT_DIR}/seed${SEED}_run.log"
    
    python scripts/cross_cohort_analysis.py \
        --tcga_model "$TCGA_MODEL" \
        --orien_model "$ORIEN_MODEL" \
        --tcga_params "$TCGA_PARAMS" \
        --orien_params "$ORIEN_PARAMS" \
        --k_values $K_VALUES \
        --cox_genes "$COX_GENES" \
        --seed $SEED \
        --output_dir "$OUTPUT_DIR" \
        2>&1 | tee "$LOG_FILE"
    
    # Check exit status
    EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Seed $SEED completed successfully"
        ((SUCCESSFUL++))
    else
        echo "✗ Seed $SEED failed with exit code: $EXIT_CODE"
        ((FAILED++))
    fi
    
    echo "Finished: $(date)"
    echo ""
done

# Final summary
echo "================================================================"
echo "MULTI-SEED ANALYSIS COMPLETE"
echo "================================================================"
echo "Total seeds: $TOTAL_SEEDS"
echo "Successful: $SUCCESSFUL"
echo "Failed: $FAILED"
echo ""
echo "Results directory: $BASE_OUTPUT_DIR"
echo "  Seed directories: ${BASE_OUTPUT_DIR}/seed*"
echo "  Individual logs: ${BASE_OUTPUT_DIR}/seed*_run.log"
echo ""

# List all result directories
echo "Generated results:"
ls -lh "$BASE_OUTPUT_DIR"
echo ""

# Create summary of all seeds
echo "Creating aggregate summary..."
SUMMARY_FILE="${BASE_OUTPUT_DIR}/AGGREGATE_SUMMARY.txt"
{
    echo "Multi-Seed Cross-Cohort Validation Summary"
    echo "=========================================="
    echo "Run date: $(date)"
    echo "Total seeds tested: $TOTAL_SEEDS"
    echo "Seeds: ${SEEDS[@]}"
    echo ""
    echo "Results by seed:"
    echo "----------------"
    
    for SEED in "${SEEDS[@]}"; do
        RESULT_FILE="${BASE_OUTPUT_DIR}/seed${SEED}/consensus_validation_results.csv"
        if [ -f "$RESULT_FILE" ]; then
            echo ""
            echo "Seed $SEED:"
            # Show optimal k result
            python -c "
import pandas as pd
df = pd.read_csv('$RESULT_FILE')
best_idx = df['avg_cindex'].idxmax()
best = df.iloc[best_idx]
print(f\"  Optimal k: {int(best['k'])}\")
print(f\"  Consensus genes: {int(best['n_consensus'])}\")
print(f\"  Avg C-index: {best['avg_cindex']:.4f}\")
print(f\"  TCGA→ORIEN: {best['tcga_on_orien_cindex']:.4f}\")
print(f\"  ORIEN→TCGA: {best['orien_on_tcga_cindex']:.4f}\")
" 2>/dev/null || echo "  Could not parse results"
        else
            echo ""
            echo "Seed $SEED: Results file not found"
        fi
    done
} > "$SUMMARY_FILE"

cat "$SUMMARY_FILE"
echo ""
echo "Summary saved to: $SUMMARY_FILE"
echo ""
echo "================================================================"
echo "Job ended: $(date)"
echo "================================================================"

# Exit with error if any seed failed
if [ $FAILED -gt 0 ]; then
    exit 1
else
    exit 0
fi