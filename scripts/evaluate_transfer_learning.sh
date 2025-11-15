#!/bin/bash

# Set Python path
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# python scripts/evaluate_transfer_learning.py \
#     --transfer_dir results/transfer_learning/orien_to_tcga_seed42_20251114_235110 \
#     --reverse_transfer_dir results/transfer_learning/tcga_to_orien_seed42_20251115_001458 \
#     --seed 42

python scripts/evaluate_transfer_learning.py \
    --transfer_dir results/transfer_learning/orien_to_tcga_seed42_20251114_235110 \
    --reverse_transfer_dir results/transfer_learning/tcga_to_orien_seed42_20251115_001458 \
    --seed 42 \
    --device cuda > /tmp/seed42_eval.log 2>&1

# Copy the result
EVAL_DIR=$(ls -td results/transfer_learning_evaluation_* | head -1)
cp "$EVAL_DIR/evaluation_results.json" "results/transfer_learning_multiseed_20251115_002608/seed42_results.json"


# for seed in 42 123 456 789 1011; do
#     echo "=========================================="
#     echo "Evaluating seed $seed..."
#     echo "=========================================="
    
#     python scripts/evaluate_transfer_learning.py \
#         --transfer_dir $(ls -d results/transfer_learning/orien_to_tcga_seed${seed}_* | head -1) \
#         --reverse_transfer_dir $(ls -d results/transfer_learning/tcga_to_orien_seed${seed}_* | head -1) \
#         --seed $seed \
#         --device cuda 2>&1 | grep -E "(RESULTS SUMMARY|Direction|Baseline|Transfer|Improvement|Bidirectional|DEBUG)"
    
#     # Copy to multiseed directory
#     EVAL_DIR=$(ls -td results/transfer_learning_evaluation_* | head -1)
#     cp "$EVAL_DIR/evaluation_results.json" "results/transfer_learning_multiseed_20251115_002608/seed${seed}_results.json"
#     echo "✓ Copied to seed${seed}_results.json"
#     echo ""
# done