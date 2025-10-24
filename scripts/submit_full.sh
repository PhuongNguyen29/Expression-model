#!/bin/bash
#$ -cwd
#$ -V
#$ -pe smp 16
#$ -q UI
#$ -N deepsurv_full
#$ -o $HOME/Expression-model/logs/full_genes_$JOB_ID.out
#$ -e $HOME/Expression-model/logs/full_genes_$JOB_ID.err
#$ -m abe
#$ -M phuong-nguyen@uiowa.edu

# Load modules
module purge
module load stack/2021.1
module load python/3.8.8_gcc-9.3.0

# Activate environment
source $HOME/Expression-model-env-py38/bin/activate

# Move to project root
cd $HOME/Expression-model

echo "=========================================="
echo "DeepSurv - Full Gene Set Experiment"
echo "=========================================="
echo "Job ID: $JOB_ID"
echo "Hostname: $(hostname)"
echo "Start time: $(date)"
echo "=========================================="
echo ""

# Run experiment
python scripts/run_experiment.py --config config/experiments/deepsurv_full.yaml

echo ""
echo "=========================================="
echo "Experiment completed"
echo "End time: $(date)"
echo "=========================================="
