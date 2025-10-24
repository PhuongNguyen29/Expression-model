#!/bin/bash
#$ -cwd
#$ -V
#$ -pe smp 8
#$ -q UI
#$ -N tunning_deepsurv
#$ -o $HOME/Expression-model/logs/tunning_orien_$JOB_ID.out
#$ -e $HOME/Expression-model/logs/tunning_orien_$JOB_ID.err
#$ -m abe
#$ -M your_email@uiowa.edu

module purge
module load stack/2021.1
module load python/3.8.8_gcc-9.3.0

source $HOME/Expression-model-env-py38/bin/activate


# Move to project root
cd $HOME/Expression-model

set -e

echo "=========================================="
echo "BATCH EXPERIMENT RUNNER"
echo "=========================================="
echo ""

# Define experiments to run
experiments=(
    "config/experiments/deepsurv_iqr.yaml"
    "config/experiments/deepsurv_biomarker.yaml"
)

# Run each experiment
for config in "${experiments[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running: $config"
    echo "=========================================="
    echo ""
    
    # Check if config file exists
    if [ ! -f "$config" ]; then
        echo "ERROR: Config file not found: $config"
        echo "Skipping..."
        continue
    fi
    
    # Run experiment
    python scripts/run_experiment.py --config "$config"
    
    # Check if successful
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Experiment completed successfully: $config"
    else
        echo ""
        echo "✗ Experiment failed: $config"
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETED!"
echo "=========================================="
echo ""
echo "Check results/ directory for outputs"
