#!/bin/bash
################################################################################
# Master Script for Step 3: Transfer Learning Evaluation
#
# This script runs all substeps of Step 3 in sequence:
#   3.1: Baseline target-only training
#   3.2: Pre-training on source cohorts
#   3.3: Fine-tuning on target cohorts
#   3.4: Statistical analysis and comparison
#
# Usage:
#   bash run_step3_transfer_learning.sh [substep]
#
# Arguments:
#   substep: Optional. Run specific substep (3.1, 3.2, 3.3, 3.4, or 'all')
#            If not provided, runs all substeps in sequence.
#
# Requirements:
#   - Completed Step 1 (hyperparameter tuning)
#   - Completed Step 2.2B (k-value validation with k=120)
#   - CUDA-capable GPU recommended (but not required)
#
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║ $1${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Check Step 1 results
    if [ ! -f "results_v2/01_hyperparameter_tuning/tcga_308genes/best_params.json" ]; then
        print_error "Step 1 TCGA results not found"
        print_info "Please run Step 1 hyperparameter tuning first"
        exit 1
    fi
    
    if [ ! -f "results_v2/01_hyperparameter_tuning/orien_308genes/best_params.json" ]; then
        print_error "Step 1 ORIEN results not found"
        print_info "Please run Step 1 hyperparameter tuning first"
        exit 1
    fi
    
    print_success "Step 1 results found"
    
    # Check Step 2.2B results
    if [ ! -f "results_v2/02_biomarker_discovery/ksweep_analysis/gene_lists/k120_consensus.txt" ]; then
        print_error "Step 2.2B consensus genes (k=120) not found"
        print_info "Please run Step 2.2B k-value validation first"
        exit 1
    fi
    
    print_success "Step 2.2B results found"
    
    # Check Python environment
    if ! command -v python &> /dev/null; then
        print_error "Python not found"
        exit 1
    fi
    
    print_success "Python environment ready"
    
    # Check GPU availability
    if command -v nvidia-smi &> /dev/null; then
        print_success "GPU detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
    else
        print_info "No GPU detected - will use CPU (slower)"
    fi
    
    echo ""
}

# Run Step 3.1
run_step_3_1() {
    print_header "Step 3.1: Baseline Target-only Training"
    print_info "Training models from scratch on target cohorts..."
    print_info "This creates train/test splits for Step 3.3 comparison"
    echo ""
    
    python scripts/step3_1_baseline_target_only.py
    
    if [ $? -eq 0 ]; then
        print_success "Step 3.1 completed successfully"
        print_info "Results saved to: results_v2/03_transfer_learning/baseline2_target_only/"
        print_info "Split indices saved to: results_v2/03_transfer_learning/baseline2_target_only/splits/"
    else
        print_error "Step 3.1 failed"
        exit 1
    fi
    echo ""
}

# Run Step 3.2
run_step_3_2() {
    print_header "Step 3.2: Pre-training Phase"
    print_info "Pre-training models on source cohorts..."
    print_info "This may take 10-15 hours depending on hardware"
    echo ""
    
    python scripts/step3_2_pretrain.py
    
    if [ $? -eq 0 ]; then
        print_success "Step 3.2 completed successfully"
        print_info "Results saved to: results_v2/03_transfer_learning/pretraining/"
    else
        print_error "Step 3.2 failed"
        exit 1
    fi
    echo ""
}

# Run Step 3.3
run_step_3_3() {
    print_header "Step 3.3: Fine-tuning Phase"
    print_info "Fine-tuning pre-trained models on target cohorts..."
    print_info "Using same train/test splits as Step 3.1 for fair comparison"
    echo ""
    
    # Check if Step 3.1 splits exist
    if [ ! -d "results_v2/03_transfer_learning/baseline2_target_only/splits" ]; then
        print_error "Step 3.1 splits not found"
        print_info "Please run Step 3.1 first"
        exit 1
    fi
    
    # Check if Step 3.2 pre-trained models exist
    if [ ! -d "results_v2/03_transfer_learning/pretraining" ]; then
        print_error "Step 3.2 pre-trained models not found"
        print_info "Please run Step 3.2 first"
        exit 1
    fi
    
    python scripts/step3_3_finetune.py
    
    if [ $? -eq 0 ]; then
        print_success "Step 3.3 completed successfully"
        print_info "Results saved to: results_v2/03_transfer_learning/finetuning/"
    else
        print_error "Step 3.3 failed"
        exit 1
    fi
    echo ""
}

# Run Step 3.4
run_step_3_4() {
    print_header "Step 3.4: Statistical Analysis"
    print_info "Comparing all methods and calculating statistics..."
    echo ""
    
    # Check if previous steps completed
    if [ ! -d "results_v2/03_transfer_learning/baseline2_target_only" ]; then
        print_error "Step 3.1 results not found"
        print_info "Please run Step 3.1 first"
        exit 1
    fi
    
    if [ ! -d "results_v2/03_transfer_learning/finetuning" ]; then
        print_error "Step 3.3 results not found"
        print_info "Please run Step 3.3 first"
        exit 1
    fi
    
    python step3_4_statistical_analysis.py
    
    if [ $? -eq 0 ]; then
        print_success "Step 3.4 completed successfully"
        print_info "Results saved to: results_v2/03_transfer_learning/analysis/"
        echo ""
        print_info "Generated files:"
        print_info "  - comparison_table.csv"
        print_info "  - statistical_tests.txt"
        print_info "  - effect_sizes.csv"
        print_info "  - improvement_analysis.csv"
        print_info "  - performance_comparison_plot.png"
    else
        print_error "Step 3.4 failed"
        exit 1
    fi
    echo ""
}

# Main execution
main() {
    SUBSTEP=${1:-all}
    
    print_header "Step 3: Transfer Learning Evaluation"
    echo ""
    print_info "Mode: $SUBSTEP"
    echo ""
    
    check_prerequisites
    
    case $SUBSTEP in
        3.1)
            run_step_3_1
            ;;
        3.2)
            run_step_3_2
            ;;
        3.3)
            run_step_3_3
            ;;
        3.4)
            run_step_3_4
            ;;
        all)
            run_step_3_1
            run_step_3_2
            run_step_3_3
            run_step_3_4
            ;;
        *)
            print_error "Invalid substep: $SUBSTEP"
            echo ""
            echo "Usage: bash run_step3_transfer_learning.sh [substep]"
            echo ""
            echo "Available substeps:"
            echo "  3.1  - Baseline target-only training"
            echo "  3.2  - Pre-training on source cohorts"
            echo "  3.3  - Fine-tuning on target cohorts"
            echo "  3.4  - Statistical analysis"
            echo "  all  - Run all substeps (default)"
            exit 1
            ;;
    esac
    
    if [ "$SUBSTEP" == "all" ]; then
        print_header "Step 3 Complete!"
        print_success "All substeps completed successfully"
        echo ""
        print_info "Final results in: results_v2/03_transfer_learning/"
        echo ""
        print_info "Next steps:"
        print_info "  1. Review results in results_v2/03_transfer_learning/analysis/"
        print_info "  2. Check performance_comparison_plot.png"
        print_info "  3. Read statistical_tests.txt for detailed analysis"
        print_info "  4. Update zero-shot baseline in step3_4_statistical_analysis.py with Step 2.2B results"
    fi
}

# Run main function
main "$@"
