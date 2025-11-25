#!/bin/bash
#
# SGE Job Script for Step 2: K-Selection with Per-K Hyperparameter Tuning
# Submit with: qsub run_step2_argon.sh
#
# ============================================================================
# SGE DIRECTIVES
# ============================================================================

#$ -N step2_ksweep
#$ -cwd
#$ -V

# Queue: UI-GPU for GPU-accelerated training
#$ -q UI


# Request 8 CPU slots (for data loading, etc.)
# Using 2x slots to avoid HT core sharing for CPU-bound operations
#$ -pe smp 16

# Memory request (adjust based on your data size)
#$ -l mem_free=32G

# Output and error logs
#$ -o $HOME/Expression-model/logs/step2_ksweep_$JOB_ID.out
#$ -e $HOME/Expression-model/logs/step2_ksweep_$JOB_ID.err

# Email notifications (begin, end, abort)
#$ -m bea
#$ -M thingnguyen@uiowa.edu

# ============================================================================
# CONFIGURATION
# ============================================================================

# K-values to test
K_VALUES="110 120 125 130 135 140 145 150 160"

# Optuna trials per cohort per k-value
N_TRIALS=50

# Directories (adjust paths as needed)
PROJECT_DIR="$HOME/Expression-model"
OUTPUT_DIR="${PROJECT_DIR}/results_v2/02_biomarker_discovery/k_selection_with_tuning"
STEP1_DIR="${PROJECT_DIR}/results_v2/01_hyperparameter_tuning"
DATA_DIR="${PROJECT_DIR}/data"

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

echo "========================================"
echo "Step 2: K-Selection with Hyperparameter Tuning"
echo "========================================"
echo ""
echo "Job Information:"
echo "  Job ID: $JOB_ID"
echo "  Job Name: $JOB_NAME"
echo "  Host: $(hostname)"
echo "  Queue: $QUEUE"
echo "  Start Time: $(date)"
echo ""

# Create log directory if it doesn't exist
mkdir -p $HOME/Expression-model/logs

# Navigate to project directory
cd $PROJECT_DIR || { echo "ERROR: Cannot cd to $PROJECT_DIR"; exit 1; }

# Load conda environment (adjust module/conda path as needed for your setup)
# Option 1: If using module system
# module load miniconda3
# conda activate your_env_name

# Option 2: If using direct conda
source $HOME/Expression-model-env-py38/bin/activate 

# Verify CUDA is available
echo "Environment Check:"
echo "  Python: $(which python)"
echo "  Python version: $(python --version)"
python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'  CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# ============================================================================
# RUN STEP 2
# ============================================================================

echo "Configuration:"
echo "  K-values: $K_VALUES"
echo "  Optuna trials: $N_TRIALS"
echo "  Output: $OUTPUT_DIR"
echo ""
echo "Starting Step 2..."
echo ""

# Run the Python script
python scripts/step2_tune_and_validate_k.py \
    --k_values $K_VALUES \
    --n_trials $N_TRIALS \
    --output_dir "$OUTPUT_DIR" \
    --step1_dir "$STEP1_DIR" \
    --data_dir "$DATA_DIR"

# Capture exit status
EXIT_STATUS=$?

# ============================================================================
# COMPLETION
# ============================================================================

echo ""
echo "========================================"
if [ $EXIT_STATUS -eq 0 ]; then
    echo "Step 2 COMPLETED SUCCESSFULLY"
    echo "========================================"
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Output files:"
    ls -la $OUTPUT_DIR/summary/ 2>/dev/null || echo "  (summary directory not found)"
else
    echo "Step 2 FAILED with exit code: $EXIT_STATUS"
    echo "========================================"
    echo ""
    echo "Check error log: $HOME/Expression-model/logs/step2_ksweep_${JOB_ID}.err"
fi

echo ""
echo "End Time: $(date)"
echo "========================================"

exit $EXIT_STATUS
