#!/bin/bash
#
# SGE Job Script for Step 2 (IG): K-Selection using Integrated Gradients
# Submit with: qsub run_step2_tuning_ig_argon.sh
#
# This script runs the IG-based k-sweep analysis with CV-derived epochs
# for fair comparison with Cox elastic net (Option 2).
#
# ============================================================================
# SGE DIRECTIVES
# ============================================================================

#$ -N step2_ksweep_ig
#$ -cwd
#$ -V
#$ -q UI

# Request 16 CPU slots (for parallel data loading and Optuna)
#$ -pe smp 16

# Memory request 
#$ -l mem_free=32G

# Output and error logs
#$ -o $HOME/Expression-model/logs/step2_ksweep_ig_$JOB_ID.out
#$ -e $HOME/Expression-model/logs/step2_ksweep_ig_$JOB_ID.err

#$ -m bea
#$ -M thingnguyen@uiowa.edu

# ============================================================================
# CONFIGURATION
# ============================================================================

# K-values to test (11 values based on IG consensus analysis)
# Range selected based on: k=80 minimum for m≥23, k≥140 for ≥15/20 Cox genes
K_VALUES="80 90 100 110 120 130 140 150 160 170 180"

# Optuna trials per cohort per k-value
N_TRIALS=50

# Directories
PROJECT_DIR="$HOME/Expression-model"
OUTPUT_DIR="${PROJECT_DIR}/results_v2/02b_biomarker_discovery_ig/k_selection_with_tuning"
IG_RANKING_DIR="${PROJECT_DIR}/results_v2/06_importance_methods/aggregated"
DATA_DIR="${PROJECT_DIR}/data"

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

echo "========================================"
echo "Step 2 (IG): K-Selection with Integrated Gradients"
echo "========================================"
echo ""
echo "Job Information:"
echo "  Job ID: $JOB_ID"
echo "  Job Name: $JOB_NAME"
echo "  Host: $(hostname)"
echo "  Queue: $QUEUE"
echo "  Start Time: $(date)"
echo ""
echo "Method: Integrated Gradients (IG) importance"
echo "Stopping: CV-derived epochs (Option 2 for fair Cox comparison)"
echo ""

# Create directories
mkdir -p $HOME/Expression-model/logs
mkdir -p $OUTPUT_DIR

# Navigate to project directory
cd $PROJECT_DIR || { echo "ERROR: Cannot cd to $PROJECT_DIR"; exit 1; }

# Load conda environment
source $HOME/Expression-model-env-py38/bin/activate 

# Verify environment
echo "Environment Check:"
echo "  Python: $(which python)"
echo "  Python version: $(python --version)"
python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    python -c "import torch; print(f'  CUDA device: {torch.cuda.get_device_name(0)}')"
else
    echo "  CUDA device: N/A (using CPU)"
fi
echo ""

# Verify IG ranking files exist
echo "Checking IG ranking files:"
if [ -f "${IG_RANKING_DIR}/tcga_ig_aggregated.csv" ]; then
    echo "  ✓ tcga_ig_aggregated.csv found"
else
    echo "  ✗ ERROR: tcga_ig_aggregated.csv not found!"
    exit 1
fi

if [ -f "${IG_RANKING_DIR}/orien_ig_aggregated.csv" ]; then
    echo "  ✓ orien_ig_aggregated.csv found"
else
    echo "  ✗ ERROR: orien_ig_aggregated.csv not found!"
    exit 1
fi
echo ""

# ============================================================================
# ESTIMATED RUNTIME
# ============================================================================

echo "Estimated Runtime:"
echo "  K-values: 11"
echo "  Trials per cohort: $N_TRIALS"
echo "  Cohorts: 2 (TCGA, ORIEN)"
echo "  Est. time per trial: ~3-5 min"
echo "  Est. total: 11 × 2 × 50 × 4 min = ~73 hours"
echo ""
echo "  Note: Actual time depends on early stopping and pruning"
echo ""

# ============================================================================
# RUN STEP 2 (IG)
# ============================================================================

echo "Configuration:"
echo "  K-values: $K_VALUES"
echo "  Optuna trials: $N_TRIALS"
echo "  IG rankings: $IG_RANKING_DIR"
echo "  Output: $OUTPUT_DIR"
echo ""
echo "Starting Step 2 (IG-based k-sweep)..."
echo ""

# Run the Python script
python scripts/step2_tune_and_validate_k_ig.py \
    --k_values $K_VALUES \
    --n_trials $N_TRIALS \
    --output_dir "$OUTPUT_DIR" \
    --ig_ranking_dir "$IG_RANKING_DIR" \
    --data_dir "$DATA_DIR"

# Capture exit status
EXIT_STATUS=$?

# ============================================================================
# COMPLETION
# ============================================================================

echo ""
echo "========================================"
if [ $EXIT_STATUS -eq 0 ]; then
    echo "Step 2 (IG) COMPLETED SUCCESSFULLY"
    echo "========================================"
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Output structure:"
    echo "  k{XXX}/consensus_genes/     - Consensus gene lists"
    echo "  k{XXX}/hyperparameter_tuning/ - Per-cohort tuning results"
    echo "  k{XXX}/cross_cohort_validation/ - Bidirectional validation"
    echo "  summary/                    - Aggregated results"
    echo ""
    echo "Key files:"
    ls -la $OUTPUT_DIR/summary/ 2>/dev/null || echo "  (summary directory pending)"
    echo ""
    echo "Next steps:"
    echo "  1. Review k_selection_summary.csv"
    echo "  2. Compare with L2-based results"
    echo "  3. Select optimal k for Step 3 (transfer learning)"
else
    echo "Step 2 (IG) FAILED with exit code: $EXIT_STATUS"
    echo "========================================"
    echo ""
    echo "Check error log: $HOME/Expression-model/logs/step2_ksweep_ig_${JOB_ID}.err"
    echo ""
    echo "Common issues:"
    echo "  - Missing IG ranking files"
    echo "  - Memory exceeded (try reducing batch size)"
    echo "  - CUDA out of memory (reduce architecture size)"
fi

echo ""
echo "End Time: $(date)"
echo "========================================"

exit $EXIT_STATUS
