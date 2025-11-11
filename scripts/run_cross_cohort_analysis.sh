#!/bin/bash

export PYTHONPATH="${PWD}:${PYTHONPATH}"

python scripts/consensus_validation.py \
  --tcga_model results/biomarker_IMPROVED_20251111_011201/tcga_model.pth \
  --orien_model results/biomarker_IMPROVED_20251111_011201/orien_model.pth \
  --tcga_params results/hyperparam_FIXED_tcga_20251109_194909/best_params.json \
  --orien_params results/hyperparam_FIXED_orien_20251109_195430/best_params.json \
  --k_values 40 45 50 55 60 65 70 75 80 85 90 95 100 110 120 130 140 150 \
  --cox_genes data/raw/cox_consensus_genes_20.txt \
  2>&1 | tee consensus_validation.log