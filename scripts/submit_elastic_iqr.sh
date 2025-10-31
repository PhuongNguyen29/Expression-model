#!/bin/bash
#$ -N elastic_deepsurv_iqr
#$ -q UI
#$ -pe smp 4
#$ -l h_vmem=16G
#$ -l gpu=1
#$ -l h_rt=12:00:00
#$ -cwd
#$ -j y
#$ -o logs/elastic_deepsurv_iqr_$JOB_ID.log

# ElasticDeepSurv training on IQR-filtered genes
# This script trains DeepSurv with Elastic Net regularization
# Expected runtime: 2-4 hours on single GPU

echo "================================================================"
echo "ElasticDeepSurv Training - IQR Filtered Genes"
echo "================================================================"
echo "Job ID: $JOB_ID"
echo "Host: $HOSTNAME"
echo "Started: $(date)"
echo "Working directory: $(pwd)"
echo ""

# Load conda environment
echo "Loading conda environment..."
source ~/miniforge3/etc/profile.d/conda.sh
conda activate Expression-model  # Change to your env name if different

# Verify environment
echo ""
echo "Python version:"
python --version
echo ""
echo "PyTorch version:"
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
fi
echo ""

# Create output directories
mkdir -p results/elastic_deepsurv_iqr
mkdir -p logs/elastic_deepsurv

echo "================================================================"
echo "Starting Training"
echo "================================================================"
echo "Model: ElasticDeepSurv"
echo "Dataset: IQR filtered genes"
echo "Config: config/experiments/elastic_deepsurv_iqr.yaml"
echo "Regularization: L1 ratio=0.7, Alpha=0.01"
echo ""

# Run experiment
python scripts/run_experiment.py \
    --config config/experiments/elastic_deepsurv_iqr.yaml \
    --verbose

EXIT_CODE=$?

echo ""
echo "================================================================"
echo "Job Complete"
echo "================================================================"
echo "Exit code: $EXIT_CODE"
echo "Finished: $(date)"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully!"
    echo ""
    echo "Results saved to: results/elastic_deepsurv_iqr_*/"
    echo "To view results:"
    echo "  ls -lht results/ | head -5"
    echo ""
    echo "To check feature importance:"
    echo "  cat results/elastic_deepsurv_iqr_*/feature_importance.csv | head -20"
else
    echo "✗ Training failed with exit code $EXIT_CODE"
    echo "Check logs for errors: logs/elastic_deepsurv_iqr_$JOB_ID.log"
fi

echo "================================================================"