#!/bin/bash

# ============================================================
# Step 4: Final Model Evaluation
# ============================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print banner
print_banner() {
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║ $1${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# ============================================================
# Parse Arguments
# ============================================================

STEP="${1:-all}"

print_banner "Step 4: Final Model Evaluation"
echo ""
print_info "Mode: $STEP"
echo ""

# ============================================================
# Check Prerequisites
# ============================================================

print_banner "Checking Prerequisites"

# Check if Step 3 results exist
if [ ! -d "results_v2/03_transfer_learning" ]; then
    print_error "Step 3 results not found. Please run Step 3 first."
    exit 1
fi
print_success "Step 3 results found"

# Check if consensus genes exist
if [ ! -f "results_v2/02_biomarker_discovery/ksweep_analysis/consensus_genes_k120.txt" ]; then
    print_error "Consensus genes file not found"
    exit 1
fi
print_success "Consensus genes file found"

# Check Python environment
if ! command -v python &> /dev/null; then
    print_error "Python not found"
    exit 1
fi
print_success "Python environment ready"

# Check if GPU is available
if command -v nvidia-smi &> /dev/null; then
    print_success "GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    print_info "No GPU detected, using CPU"
fi

echo ""

# ============================================================
# Run Steps
# ============================================================

run_step_4_1() {
    print_banner "Step 4.1: Train Final Models on Full Data"
    print_info "Training Cox, Target-only, and Transfer learning models..."
    print_info "Calculating bootstrap-corrected C-index..."
    print_info "This will take approximately 2-3 hours..."
    echo ""
    
    python scripts/step4_1_train_final_full_data.py
    
    if [ $? -eq 0 ]; then
        print_success "Step 4.1 completed successfully"
        print_info "Results saved to: results_v2/04_final_models/"
        echo ""
    else
        print_error "Step 4.1 failed"
        exit 1
    fi
}

run_step_4_2() {
    print_banner "Step 4.2: Survival Analysis"
    print_info "Analyzing survival curves for all models..."
    print_info "Calculating log-rank tests and hazard ratios..."
    echo ""
    
    # Check if Step 4.1 results exist
    if [ ! -d "results_v2/04_final_models" ]; then
        print_error "Step 4.1 results not found. Run Step 4.1 first."
        exit 1
    fi
    
    python scripts/step4_2_survival_analysis.py
    
    if [ $? -eq 0 ]; then
        print_success "Step 4.2 completed successfully"
        print_info "Results saved to: results_v2/04_final_models/survival_analysis/"
        echo ""
        print_info "Generated files:"
        print_info "  - survival_analysis_results.csv"
        print_info "  - comprehensive_comparison.csv"
        print_info "  - Kaplan-Meier plots (*.png)"
        echo ""
    else
        print_error "Step 4.2 failed"
        exit 1
    fi
}

# ============================================================
# Execute Based on Argument
# ============================================================

case "$STEP" in
    4.1)
        run_step_4_1
        ;;
    4.2)
        run_step_4_2
        ;;
    all)
        run_step_4_1
        run_step_4_2
        
        print_banner "Step 4 Complete!"
        echo ""
        print_success "All final evaluations completed"
        print_info "Results saved to: results_v2/04_final_models/"
        echo ""
        print_info "Next: Review results and prepare dissertation figures"
        ;;
    *)
        print_error "Unknown step: $STEP"
        echo ""
        echo "Usage: bash run_step4_final_evaluation.sh [4.1|4.2|all]"
        echo ""
        echo "Steps:"
        echo "  4.1  - Train final models on full data with bootstrap C-index"
        echo "  4.2  - Survival analysis and Kaplan-Meier curves"
        echo "  all  - Run all steps (default)"
        exit 1
        ;;
esac

echo ""