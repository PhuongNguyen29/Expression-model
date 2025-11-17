#!/bin/bash
################################################################################
# STEP 1: HYPERPARAMETER TUNING & BASELINE ESTABLISHMENT
################################################################################
#
# Purpose: Determine optimal model architectures for TCGA and ORIEN cohorts
#          using 308 consensus genes with 5-fold stratified cross-validation
#
# Evidence Base:
#   - Optuna framework: Akiba et al., 2019, KDD
#   - Stratified CV for survival: Simon et al., 2011, JCO
#   - Elastic Net regularization: Zou & Hastie, 2005, JRSS-B
#
# Output Structure:
#   results_v2/01_hyperparameter_tuning/
#   ├── orien_308genes/
#   │   ├── best_params.json
#   │   ├── trials.csv
#   │   ├── cv_performance.json
#   │   └── study.pkl
#   ├── tcga_308genes/
#   │   ├── best_params.json
#   │   ├── trials.csv
#   │   ├── cv_performance.json
#   │   └── study.pkl
#   └── summary_comparison.txt
#
# Author: Phuong
# Date: 2024-11-17
# Status: ACTIVE - Step 1 of 5-step transfer learning pipeline
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

################################################################################
# CONFIGURATION
################################################################################

# Directory structure (NEW for v2)
BASE_DIR="results_v2"
STEP1_DIR="${BASE_DIR}/01_hyperparameter_tuning"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Hyperparameter tuning settings
N_TRIALS=50  # Number of Optuna trials (50-100 recommended)
N_FOLDS=5    # Cross-validation folds
SEED=42      # Random seed for reproducibility

# Python environment
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

################################################################################
# PRE-FLIGHT CHECKS
################################################################################

log "=========================================="
log "STEP 1: HYPERPARAMETER TUNING"
log "=========================================="
echo ""

log "Pre-flight checks..."

# Check Python script exists
if [ ! -f "scripts/hyperparam_tuning_elastic_FIXED.py" ]; then
    error "Required script not found: scripts/hyperparam_tuning_elastic_FIXED.py"
    exit 1
fi
success "✓ Training script found"

# Check configuration file exists
if [ ! -f "config/default_config.yaml" ]; then
    error "Required config not found: config/default_config.yaml"
    exit 1
fi
success "✓ Configuration file found"

# Check data files exist
DATA_FILES=(
    "data/raw/tcga_batch_corrected_2sv.csv"
    "data/raw/orien_batch_corrected.csv"
    "data/raw/surv_tcga.csv"
    "data/raw/surv_orien_update.csv"
    "data/raw/consensus_genes_308.txt"
)

for file in "${DATA_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        error "Required data file not found: $file"
        exit 1
    fi
done
success "✓ All required data files found"

# Create output directory structure
mkdir -p "${STEP1_DIR}"
mkdir -p "${STEP1_DIR}/logs"
success "✓ Output directories created: ${STEP1_DIR}"

echo ""
log "Configuration:"
log "  Base directory: ${BASE_DIR}"
log "  Output directory: ${STEP1_DIR}"
log "  Number of trials: ${N_TRIALS}"
log "  CV folds: ${N_FOLDS}"
log "  Random seed: ${SEED}"
echo ""

################################################################################
# STEP 1.1: HYPERPARAMETER TUNING ON ORIEN (SOURCE COHORT)
################################################################################

log "=========================================="
log "STEP 1.1: TUNING ORIEN (Source Cohort)"
log "=========================================="
echo ""

ORIEN_OUTPUT="${STEP1_DIR}/orien_308genes"
ORIEN_LOG="${STEP1_DIR}/logs/orien_tuning_${TIMESTAMP}.log"

log "Configuration:"
log "  Cohort: ORIEN (n=1,112 samples, 450 events)"
log "  Features: 308 consensus genes"
log "  Output: ${ORIEN_OUTPUT}"
log "  Log file: ${ORIEN_LOG}"
echo ""

log "Starting hyperparameter optimization..."
log "This may take 1-3 hours depending on hardware..."
echo ""

if python scripts/hyperparam_tuning_elastic_FIXED.py \
    --cohort orien \
    --n_trials ${N_TRIALS} \
    --output_dir "${ORIEN_OUTPUT}" \
    2>&1 | tee "${ORIEN_LOG}"; then
    
    success "✓ ORIEN hyperparameter tuning completed"
    
    # Check if output files were created
    if [ -f "${ORIEN_OUTPUT}/best_params.json" ]; then
        success "  ✓ best_params.json created"
        log "  Best parameters:"
        python -c "import json; print(json.dumps(json.load(open('${ORIEN_OUTPUT}/best_params.json')), indent=2))" | head -20
    else
        error "  ✗ best_params.json not found"
        exit 1
    fi
    
else
    error "✗ ORIEN hyperparameter tuning failed"
    error "Check log file: ${ORIEN_LOG}"
    exit 1
fi

echo ""
log "ORIEN tuning checkpoint passed ✓"
echo ""

################################################################################
# STEP 1.2: HYPERPARAMETER TUNING ON TCGA (TARGET COHORT)
################################################################################

log "=========================================="
log "STEP 1.2: TUNING TCGA (Target Cohort)"
log "=========================================="
echo ""

TCGA_OUTPUT="${STEP1_DIR}/tcga_308genes"
TCGA_LOG="${STEP1_DIR}/logs/tcga_tuning_${TIMESTAMP}.log"

log "Configuration:"
log "  Cohort: TCGA (n=339 samples, 153 events)"
log "  Features: 308 consensus genes"
log "  Output: ${TCGA_OUTPUT}"
log "  Log file: ${TCGA_LOG}"
echo ""

log "Starting hyperparameter optimization..."
log "This may take 1-2 hours depending on hardware..."
echo ""

if python scripts/hyperparam_tuning_elastic_FIXED.py \
    --cohort tcga \
    --n_trials ${N_TRIALS} \
    --output_dir "${TCGA_OUTPUT}" \
    2>&1 | tee "${TCGA_LOG}"; then
    
    success "✓ TCGA hyperparameter tuning completed"
    
    # Check if output files were created
    if [ -f "${TCGA_OUTPUT}/best_params.json" ]; then
        success "  ✓ best_params.json created"
        log "  Best parameters:"
        python -c "import json; print(json.dumps(json.load(open('${TCGA_OUTPUT}/best_params.json')), indent=2))" | head -20
    else
        error "  ✗ best_params.json not found"
        exit 1
    fi
    
else
    error "✗ TCGA hyperparameter tuning failed"
    error "Check log file: ${TCGA_LOG}"
    exit 1
fi

echo ""
log "TCGA tuning checkpoint passed ✓"
echo ""

################################################################################
# STEP 1.3: GENERATE COMPARISON SUMMARY
################################################################################

log "=========================================="
log "STEP 1.3: GENERATING COMPARISON SUMMARY"
log "=========================================="
echo ""

SUMMARY_FILE="${STEP1_DIR}/summary_comparison.txt"

cat > "${SUMMARY_FILE}" << EOF
================================================================================
STEP 1: HYPERPARAMETER TUNING SUMMARY
================================================================================
Date: $(date)
Timestamp: ${TIMESTAMP}

CONFIGURATION
-------------
- Number of trials: ${N_TRIALS}
- CV folds: ${N_FOLDS}
- Random seed: ${SEED}

COHORT COMPARISON
-----------------
EOF

# Extract best C-index for ORIEN
if [ -f "${ORIEN_OUTPUT}/cv_performance.json" ]; then
    ORIEN_CINDEX=$(python -c "import json; data=json.load(open('${ORIEN_OUTPUT}/cv_performance.json')); print(f\"{data.get('mean_c_index', 'N/A'):.4f}\")")
    ORIEN_STD=$(python -c "import json; data=json.load(open('${ORIEN_OUTPUT}/cv_performance.json')); print(f\"{data.get('std_c_index', 'N/A'):.4f}\")")
    
    cat >> "${SUMMARY_FILE}" << EOF

ORIEN (Source Cohort):
  - Samples: 1,112 (450 events)
  - Best CV C-index: ${ORIEN_CINDEX} ± ${ORIEN_STD}
  - Hyperparameters: ${ORIEN_OUTPUT}/best_params.json

EOF
fi

# Extract best C-index for TCGA
if [ -f "${TCGA_OUTPUT}/cv_performance.json" ]; then
    TCGA_CINDEX=$(python -c "import json; data=json.load(open('${TCGA_OUTPUT}/cv_performance.json')); print(f\"{data.get('mean_c_index', 'N/A'):.4f}\")")
    TCGA_STD=$(python -c "import json; data=json.load(open('${TCGA_OUTPUT}/cv_performance.json')); print(f\"{data.get('std_c_index', 'N/A'):.4f}\")")
    
    cat >> "${SUMMARY_FILE}" << EOF
TCGA (Target Cohort):
  - Samples: 339 (153 events)
  - Best CV C-index: ${TCGA_CINDEX} ± ${TCGA_STD}
  - Hyperparameters: ${TCGA_OUTPUT}/best_params.json

EOF
fi

cat >> "${SUMMARY_FILE}" << EOF

VERIFICATION CHECKLIST
----------------------
EOF

# Verification checks
ALL_PASSED=true

# Check 1: C-index threshold
if [ ! -z "${ORIEN_CINDEX:-}" ] && [ ! -z "${TCGA_CINDEX:-}" ]; then
    if (( $(echo "$ORIEN_CINDEX >= 0.65" | bc -l) )) && (( $(echo "$TCGA_CINDEX >= 0.65" | bc -l) )); then
        echo "✓ Both cohorts achieve C-index ≥ 0.65" >> "${SUMMARY_FILE}"
    else
        echo "✗ WARNING: Some cohorts below C-index 0.65 threshold" >> "${SUMMARY_FILE}"
        ALL_PASSED=false
    fi
else
    echo "✗ Unable to verify C-index values" >> "${SUMMARY_FILE}"
    ALL_PASSED=false
fi

# Check 2: Files created
if [ -f "${ORIEN_OUTPUT}/best_params.json" ] && [ -f "${TCGA_OUTPUT}/best_params.json" ]; then
    echo "✓ Best hyperparameters saved for both cohorts" >> "${SUMMARY_FILE}"
else
    echo "✗ Missing hyperparameter files" >> "${SUMMARY_FILE}"
    ALL_PASSED=false
fi

# Check 3: CV variance
if [ ! -z "${ORIEN_STD:-}" ] && [ ! -z "${TCGA_STD:-}" ]; then
    if (( $(echo "$ORIEN_STD < 0.05" | bc -l) )) && (( $(echo "$TCGA_STD < 0.05" | bc -l) )); then
        echo "✓ CV variance < 0.05 for both cohorts (stable)" >> "${SUMMARY_FILE}"
    else
        echo "⚠ WARNING: High CV variance detected (>0.05)" >> "${SUMMARY_FILE}"
    fi
fi

cat >> "${SUMMARY_FILE}" << EOF

NEXT STEPS
----------
If all checks pass:
  → Proceed to Step 2: Biomarker Discovery (K-Sweep Analysis)
  → Run: bash run_step2_biomarker_discovery.sh

If checks fail:
  → Review log files in ${STEP1_DIR}/logs/
  → Check for gradient explosion warnings
  → Consider adjusting hyperparameter search space

FILES GENERATED
---------------
${ORIEN_OUTPUT}/
  - best_params.json (optimal hyperparameters)
  - trials.csv (all trial results)
  - cv_performance.json (cross-validation metrics)
  - study.pkl (Optuna study object)

${TCGA_OUTPUT}/
  - best_params.json
  - trials.csv
  - cv_performance.json
  - study.pkl

LOGS
----
- ORIEN: ${ORIEN_LOG}
- TCGA: ${TCGA_LOG}

================================================================================
EOF

log "Summary saved to: ${SUMMARY_FILE}"
echo ""
cat "${SUMMARY_FILE}"

################################################################################
# FINAL STATUS
################################################################################

echo ""
log "=========================================="
if [ "$ALL_PASSED" = true ]; then
    success "STEP 1 COMPLETED SUCCESSFULLY ✓"
    log "=========================================="
    echo ""
    success "All verification checks passed!"
    log "Ready to proceed to Step 2: Biomarker Discovery"
    echo ""
    log "Next command:"
    echo "  bash run_step2_biomarker_discovery.sh"
else
    warning "STEP 1 COMPLETED WITH WARNINGS ⚠"
    log "=========================================="
    echo ""
    warning "Some verification checks failed. Review the summary above."
    log "Check log files for details before proceeding."
fi

echo ""
log "Results saved to: ${STEP1_DIR}"
log "Summary: ${SUMMARY_FILE}"
echo ""

exit 0
