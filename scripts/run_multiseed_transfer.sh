#!/bin/bash
# Multi-Seed Transfer Learning Pipeline for Chapter 4
# Runs transfer learning with 5 seeds for statistical validation

set -e  # Exit on error

# Configuration
SEEDS=(42 123 456 789 1011)
SOURCE_PARAMS_ORIEN="results/hyperparam_FIXED_orien_20251109_195430/best_params.json"
SOURCE_PARAMS_TCGA="results/hyperparam_FIXED_tcga_20251109_194909/best_params.json"
DEVICE="cuda"

# Create output directories
RESULTS_DIR="results/transfer_learning_multiseed_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR/logs"

# Set Python path
export PYTHONPATH="${PWD}:${PYTHONPATH}"

echo "=========================================="
echo "MULTI-SEED TRANSFER LEARNING PIPELINE"
echo "=========================================="
echo "Seeds: ${SEEDS[@]}"
echo "Results directory: $RESULTS_DIR"
echo "=========================================="
echo ""

# Loop through each seed
for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Processing Seed: $SEED"
    echo "=========================================="
    
    # ========================================
    # Direction 1: ORIEN → TCGA
    # ========================================
    echo ""
    echo "[1/3] Training ORIEN → TCGA transfer (seed=$SEED)..."
    
    python scripts/transfer_learning_trainer.py \
        --source_cohort orien \
        --target_cohort tcga \
        --source_params "$SOURCE_PARAMS_ORIEN" \
        --target_params "$SOURCE_PARAMS_TCGA" \
        --seed "$SEED" \
        --device "$DEVICE" \
        2>&1 | tee "$RESULTS_DIR/logs/orien_to_tcga_seed${SEED}.log"
    
    # Store the output directory path
    ORIEN_TO_TCGA_DIR=$(ls -td results/transfer_learning/orien_to_tcga_seed${SEED}_* | head -1)
    echo "✓ Saved to: $ORIEN_TO_TCGA_DIR"
    
    # ========================================
    # Direction 2: TCGA → ORIEN
    # ========================================
    echo ""
    echo "[2/3] Training TCGA → ORIEN transfer (seed=$SEED)..."
    
    python scripts/transfer_learning_trainer.py \
        --source_cohort tcga \
        --target_cohort orien \
        --source_params "$SOURCE_PARAMS_TCGA" \
        --target_params "$SOURCE_PARAMS_ORIEN" \
        --seed "$SEED" \
        --device "$DEVICE" \
        2>&1 | tee "$RESULTS_DIR/logs/tcga_to_orien_seed${SEED}.log"
    
    # Store the output directory path
    TCGA_TO_ORIEN_DIR=$(ls -td results/transfer_learning/tcga_to_orien_seed${SEED}_* | head -1)
    echo "✓ Saved to: $TCGA_TO_ORIEN_DIR"
    
    # ========================================
    # Direction 3: Evaluate Both Directions
    # ========================================
    echo ""
    echo "[3/3] Evaluating transfer learning (seed=$SEED)..."
    
    # Capture output to get the results file path
    EVAL_OUTPUT=$(python scripts/evaluate_transfer_learning.py \
        --transfer_dir "$ORIEN_TO_TCGA_DIR" \
        --reverse_transfer_dir "$TCGA_TO_ORIEN_DIR" \
        --seed "$SEED" \
        --device "$DEVICE" \
        2>&1 | tee "$RESULTS_DIR/logs/evaluation_seed${SEED}.log")
    
    # Extract the results file path from output
    RESULTS_FILE=$(echo "$EVAL_OUTPUT" | grep "Results saved to:" | sed 's/.*Results saved to: //')
    
    if [ -f "$RESULTS_FILE" ]; then
        # Copy evaluation results to our multi-seed directory
        cp "$RESULTS_FILE" "$RESULTS_DIR/seed${SEED}_results.json"
        echo "✓ Copied results to: $RESULTS_DIR/seed${SEED}_results.json"
    else
        echo "⚠️  Warning: Could not find results file at: $RESULTS_FILE"
        # Fallback to old method
        EVAL_DIR=$(ls -td results/transfer_learning_evaluation_* | head -1)
        cp "$EVAL_DIR/evaluation_results.json" "$RESULTS_DIR/seed${SEED}_results.json"
    fi
    
    echo ""
    echo "✓ Completed seed $SEED"
    echo "=========================================="
done

echo ""
echo "=========================================="
echo "ALL SEEDS COMPLETED!"
echo "=========================================="
echo ""
echo "Results saved in: $RESULTS_DIR"
echo ""
echo "Individual seed results:"
ls -1 "$RESULTS_DIR"/seed*_results.json
echo ""
echo "Next step: Run aggregate analysis"
echo "  python scripts/aggregate_multiseed_results.py --results_dir $RESULTS_DIR"
echo ""
