#!/bin/bash
# Batch Experiment Runner
# Runs multiple experiments sequentially with different configurations

# Exit on error
set -e

echo "=========================================="
echo "BATCH EXPERIMENT RUNNER"
echo "=========================================="
echo ""

# Define experiments to run
experiments=(
    "config/experiments/deepsurv_full.yaml"
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
