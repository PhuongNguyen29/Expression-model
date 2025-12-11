#!/bin/bash
#$ -N step2_ig_job1
#$ -cwd
#$ -V
#$ -q UI
#$ -pe smp 16
#$ -l mem_free=32G
#$ -o $HOME/Expression-model/logs/step2_ig_job1_$JOB_ID.out
#$ -e $HOME/Expression-model/logs/step2_ig_job1_$JOB_ID.err
#$ -m bea
#$ -M thingnguyen@uiowa.edu

# Job 1: k=80, 90
K_VALUES="80 90"
N_TRIALS=50

PROJECT_DIR="$HOME/Expression-model"
OUTPUT_DIR="${PROJECT_DIR}/results_v2/02b_biomarker_discovery_ig/k_selection_with_tuning"
IG_RANKING_DIR="${PROJECT_DIR}/results_v2/06_importance_methods/aggregated"
DATA_DIR="${PROJECT_DIR}/data"

echo "========================================"
echo "Step 2 (IG) - Job 1: k=80, 90"
echo "========================================"
echo "Job ID: $JOB_ID | Host: $(hostname) | Start: $(date)"
echo ""

mkdir -p $HOME/Expression-model/logs
mkdir -p $OUTPUT_DIR
cd $PROJECT_DIR || exit 1

source $HOME/Expression-model-env-py38/bin/activate

echo "K-values: $K_VALUES"
echo "Output: $OUTPUT_DIR"
echo ""

python scripts/step2_tune_and_validate_k_ig.py \
    --k_values $K_VALUES \
    --n_trials $N_TRIALS \
    --output_dir "$OUTPUT_DIR" \
    --ig_ranking_dir "$IG_RANKING_DIR" \
    --data_dir "$DATA_DIR"

EXIT_STATUS=$?
echo ""
echo "Job 1 finished with status: $EXIT_STATUS | End: $(date)"
exit $EXIT_STATUS
