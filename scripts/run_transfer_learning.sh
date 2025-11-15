#!/bin/bash

# Set Python path
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Run transfer learning: ORIEN → TCGA
python scripts/transfer_learning_trainer.py \
    --source_cohort orien \
    --target_cohort tcga \
    --source_params results/hyperparam_FIXED_orien_20251109_195430/best_params.json \
    --target_params results/hyperparam_FIXED_tcga_20251109_194909/best_params.json \
    --seed 42 \
    --device cuda \
    2>&1 | tee logs/transfer_learning_seed42.log