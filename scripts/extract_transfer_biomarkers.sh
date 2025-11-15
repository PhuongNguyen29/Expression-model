
#!/bin/bash

# Set Python path
export PYTHONPATH="${PWD}:${PYTHONPATH}"

python scripts/extract_transfer_biomarkers.py \
    --models_dir results/transfer_learning_multiseed_20251115_002608 \
    --output_dir results/biomarker_analysis_transfer \
    --top_k 50