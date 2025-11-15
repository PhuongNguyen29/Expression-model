#!/bin/bash

# Set Python path
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# python scripts/evaluate_transfer_learning.py \
#     --transfer_dir results/transfer_learning/orien_to_tcga_seed42_20251114_235110 \
#     --reverse_transfer_dir results/transfer_learning/tcga_to_orien_seed42_20251115_001458 \
#     --seed 42

for seed in 42 123 456 789 1011; do
    echo "Evaluating seed $seed..."
    python scripts/evaluate_transfer_learning.py \
        --transfer_dir $(ls -d results/transfer_learning/orien_to_tcga_seed${seed}_* | head -1) \
        --reverse_transfer_dir $(ls -d results/transfer_learning/tcga_to_orien_seed${seed}_* | head -1) \
        --seed $seed \
        --device cuda
    
    # Copy to multiseed directory
    EVAL_DIR=$(ls -td results/transfer_learning_evaluation_* | head -1)
    cp "$EVAL_DIR/evaluation_results.json" "results/transfer_learning_multiseed_20251115_002608/seed${seed}_results.json"
done