#!/bin/bash

# Set Python path
export PYTHONPATH="${PWD}:${PYTHONPATH}"

python scripts/evaluate_transfer_learning.py \
    --transfer_dir results/transfer_learning/orien_to_tcga_seed42_20251114_235110 \
    --reverse_transfer_dir results/transfer_learning/tcga_to_orien_seed42_20251115_001458 \
    --seed 42