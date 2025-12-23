#!/bin/bash
#$ -N step2_signfilter_job1
#$ -cwd
#$ -V
#$ -q UI
#$ -pe smp 24
#$ -l mem_free=32G
#$ -o $HOME/Expression-model/logs/step2_signfilter_job1_$JOB_ID.out
#$ -e $HOME/Expression-model/logs/step2_signfilter_job1_$JOB_ID.err
#$ -m bea
#$ -M thingnguyen@uiowa.edu

# =============================================================================
# Job 1: k=40, 50 (Sign-Consistent Gene Pool)
# =============================================================================
# This script runs k-selection with hyperparameter tuning using:
# - 141 sign-consistent genes (filtered from 308)
# - Integrated Gradients importance ranking
# =============================================================================

K_VALUES="40 50"
N_TRIALS=50

PROJECT_DIR="$HOME/Expression-model"
SCRIPT_DIR="${PROJECT_DIR}/scripts/retune_sign_consensus_genes"
OUTPUT_DIR="${PROJECT_DIR}/results_v2/02c_biomarker_discovery_ig_signfilter/k_selection_with_tuning"
IG_RANKING_DIR="${PROJECT_DIR}/results_v2/06_importance_methods/aggregated"
SIGN_GENES="${PROJECT_DIR}/data/processed/sign_consistent_genes_141.txt"
DATA_DIR="${PROJECT_DIR}/data"

echo "========================================"
echo "Step 2 (IG + Sign Filter) - Job 1"
echo "========================================"
echo "Job ID: $JOB_ID | Host: $(hostname) | Start: $(date)"
echo "K-values: $K_VALUES"
echo "Sign-consistent genes: $SIGN_GENES"
echo "Output: $OUTPUT_DIR"
echo "========================================"
echo ""

# Create directories
mkdir -p $HOME/Expression-model/logs
mkdir -p $OUTPUT_DIR

# Navigate to project directory
cd $PROJECT_DIR || exit 1

# Activate environment
source $HOME/Expression-model-env-py38/bin/activate

# Verify sign-consistent genes file exists
if [ ! -f "$SIGN_GENES" ]; then
    echo "ERROR: Sign-consistent genes file not found: $SIGN_GENES"
    exit 1
fi

echo "Starting k-selection with sign-consistent genes..."
echo ""

python ${SCRIPT_DIR}/step2_tune_and_validate_k_ig_signfilter.py \
    --k_values $K_VALUES \
    --n_trials $N_TRIALS \
    --output_dir "$OUTPUT_DIR" \
    --ig_ranking_dir "$IG_RANKING_DIR" \
    --sign_genes "$SIGN_GENES" \
    --data_dir "$DATA_DIR"

EXIT_STATUS=$?
echo ""
echo "========================================"
echo "Job 1 finished with status: $EXIT_STATUS"
echo "End: $(date)"
echo "========================================"
exit $EXIT_STATUS
