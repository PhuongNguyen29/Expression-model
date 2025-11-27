#!/bin/bash
#
# Run Step 2: K-Selection with Per-K Hyperparameter Tuning
#
# This script performs hyperparameter tuning for each k-value,
# then evaluates cross-cohort performance to select optimal k.
#
# Usage:
#   bash run_step2_tune_and_validate_k.sh [options]
#
# Options:
#   --k_values: Space-separated k-values (default: 80 90 100 110 120 130 140 150)
#   --n_trials: Optuna trials per cohort (default: 50)
#   --n_jobs: Parallel jobs (default: 8)
#   --device: cuda or cpu (default: cuda if available)

set -e  # Exit on error

# Default parameters
K_VALUES="80 85 90 95"
N_TRIALS=50

OUTPUT_DIR="results_v2/02_biomarker_discovery/k_selection_with_tuning"
STEP1_DIR="results_v2/01_hyperparameter_tuning"
DATA_DIR="data"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --k_values)
            K_VALUES="$2"
            shift 2
            ;;
        --n_trials)
            N_TRIALS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: bash run_step2_tune_and_validate_k.sh [options]"
            echo ""
            echo "Options:"
            echo "  --k_values    Space-separated k-values (default: 80 90 100 110 120 130 140 150)"
            echo "  --n_trials    Optuna trials per cohort (default: 50)"
            echo "  --n_jobs      Parallel jobs (default: 8)"
            echo "  --device      cuda or cpu (default: cuda)"
            echo "  --help        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print configuration
echo "========================================"
echo "Step 2: K-Selection with Hyperparameter Tuning"
echo "========================================"
echo ""
echo "Configuration:"
echo "  K-values: $K_VALUES"
echo "  Optuna trials per cohort: $N_TRIALS"
echo "  Parallel jobs: $N_JOBS"
echo "  Device: $DEVICE"
echo "  Output: $OUTPUT_DIR"
echo ""
echo "Estimated time per k-value:"
echo "  With n_jobs=$N_JOBS: ~2-3 hours"
echo "  Total for ${#K_VALUES[@]} k-values: ~16-24 hours"
echo ""


# Create timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "Started at: $TIMESTAMP"

# Run the script
python scripts/step2_tune_and_validate_k.py \
    --k_values $K_VALUES \
    --n_trials $N_TRIALS \
    --output_dir "$OUTPUT_DIR" \
    --step1_dir "$STEP1_DIR" \
    --data_dir "$DATA_DIR"

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Step 2 COMPLETE!"
    echo "========================================"
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Summary files:"
    echo "  - k_selection_summary.csv"
    echo "  - optimal_k_recommendation.json"
    echo "  - k_selection_analysis.png"
    echo ""
    echo "Next step: Review optimal k recommendation and proceed to Step 3"
else
    echo ""
    echo "ERROR: Step 2 failed!"
    echo "Check logs in: $OUTPUT_DIR/logs/"
    exit 1
fi
