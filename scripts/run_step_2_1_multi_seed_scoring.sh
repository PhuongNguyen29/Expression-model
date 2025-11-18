#!/bin/bash
export PYTHONPATH="${PWD}:${PYTHONPATH}"

python scripts/extract_biomarkers_from_best_params.py \
  --tcga_params results_v2/01_hyperparameter_tuning/tcga_308genes/best_params.json \
  --orien_params results_v2/01_hyperparameter_tuning/orien_308genes/best_params.json \
  --consensus_genes data/raw/consensus_genes_308.txt \
  --output_dir results_v2/02_biomarker_discovery \
  --max_epochs 150 \
  --seeds 42 123 456 789 1011