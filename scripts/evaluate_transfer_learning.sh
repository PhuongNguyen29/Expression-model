#!/bin/bash

# Set Python path
export PYTHONPATH="${PWD}:${PYTHONPATH}"

python scripts/evaluate_transfer_learning.py \
    --transfer_dir results/transfer_learning/orien_to_tcga_seed42_20251114_235110 \
    --seed 42