#!/bin/bash
#$ -cwd
#$ -V
#$ -pe smp 8
#$ -q UI
#$ -N elastic_deepsurv_iqr
#$ -o $HOME/Expression-model/logs/elastic_deepsurv_iqr_$JOB_ID.out
#$ -e $HOME/Expression-model/logs/elastic_deepsurv_iqr_$JOB_ID.err
#$ -m abe
#$ -M your_email@uiowa.edu

# ElasticDeepSurv Training on IQR-Filtered Genes (CPU version)
# Adapted to UI HPC environment
# Expected runtime: 4-8 hours on CPU (8 cores)

echo "================================================================"
echo "ElasticDeepSurv Training - IQR Filtered Genes"
echo "================================================================"
echo "Job ID: $JOB_ID"
echo "Host: $HOSTNAME"
echo "Started: $(date)"
echo "Working directory: $(pwd)"
echo "CPUs allocated: $NSLOTS"
echo ""

# Load modules
module purge
module load stack/2021.1
module load python/3.8.8_gcc-9.3.0

# Activate environment
source $HOME/Expression-model-env-py38/bin/activate

# Verify environment
echo ""
echo "Python version:"
python --version
echo ""
echo "PyTorch version:"
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import torch; print(f'Number of CPU threads: {torch.get_num_threads()}')"
echo ""

# Set PyTorch to use available CPUs efficiently
export OMP_NUM_THREADS=$NSLOTS
export MKL_NUM_THREADS=$NSLOTS

# Move to project root
cd $HOME/Expression-model

# Create output directories
mkdir -p results/elastic_deepsurv_iqr
mkdir -p logs/elastic_deepsurv

echo "================================================================"
echo "Starting Training"
echo "================================================================"
echo "Model: ElasticDeepSurv"
echo "Dataset: IQR filtered genes"
echo "Config: config/experiments/elastic_deepsurv_iqr.yaml"
echo "Device: CPU (${NSLOTS} cores)"
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
    echo ""
    echo "To view latest results:"
    echo "  cd ~/Expression-model"
    echo "  ls -lht results/ | head -5"
    echo ""
    echo "To check C-index:"
    echo "  cat results/elastic_deepsurv_iqr_*/tcga_to_orien_results.json | grep c_index"
    echo ""
    echo "To view top genes:"
    echo "  cat results/elastic_deepsurv_iqr_*/feature_importance.csv | head -21"
else
    echo "✗ Training failed with exit code $EXIT_CODE"
    echo "Check error log: logs/elastic_deepsurv_iqr_${JOB_ID}.err"
fi

echo "================================================================"