#!/bin/bash

# Cross-Cohort Validation Analysis
# Uses existing trained models - NO RETRAINING

export PYTHONPATH="${PWD}:${PYTHONPATH}"

python scripts/cross_cohort_analysis.py \
  --tcga_model results/biomarker_IMPROVED_20251111_011201/tcga_model.pth \
  --orien_model results/biomarker_IMPROVED_20251111_011201/orien_model.pth \
  --tcga_params results/hyperparam_FIXED_tcga_20251109_194909/best_params.json \
  --orien_params results/hyperparam_FIXED_orien_20251109_195430/best_params.json \
  --cox_genes data/raw/cox_consensus_genes_20.txt \
  2>&1 | tee cross_cohort_analysis.log