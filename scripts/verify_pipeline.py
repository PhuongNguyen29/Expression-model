"""
Verification script to check if all dependencies are in place
before running biomarker extraction.

Run this on your HPC to ensure smooth execution.
"""

import sys
import os
from pathlib import Path

def check_imports():
    """Check if all required modules can be imported."""
    print("="*60)
    print("CHECKING PYTHON IMPORTS")
    print("="*60)
    
    required_modules = [
        ('torch', 'PyTorch'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('yaml', 'PyYAML'),
        ('lifelines', 'Lifelines'),
    ]
    
    all_good = True
    for module, name in required_modules:
        try:
            __import__(module)
            print(f"✓ {name:20s} - OK")
        except ImportError as e:
            print(f"✗ {name:20s} - MISSING: {e}")
            all_good = False
    
    return all_good


def check_project_structure():
    """Check if required project files exist."""
    print("\n" + "="*60)
    print("CHECKING PROJECT STRUCTURE")
    print("="*60)
    
    required_files = [
        'config/default_config.yaml',
        'src/data/preprocessor.py',
        'src/data/dataset.py',
        'src/models/elastic_deepsurv.py',
        'src/utils/batch_samplers.py',
        'src/utils/feature_selection.py',
        'data/raw/tcga_batch_corrected_2sv.csv',
        'data/raw/orien_batch_corrected.csv',
        'data/processed/surv_tcga_harmonized.csv',
        'data/processed/surv_orien_harmonized.csv',
    ]
    
    all_good = True
    for filepath in required_files:
        if Path(filepath).exists():
            print(f"✓ {filepath}")
        else:
            print(f"✗ {filepath} - MISSING")
            all_good = False
    
    return all_good


def check_hyperparameter_results():
    """Check if hyperparameter tuning results exist."""
    print("\n" + "="*60)
    print("CHECKING HYPERPARAMETER TUNING RESULTS")
    print("="*60)
    
    tcga_path = Path('results/20251104/hyperparam_tcga_20251104_034717/best_params.json')
    orien_path = Path('results/20251104/hyperparam_orien_20251104_032058/best_params.json')
    
    all_good = True
    
    if tcga_path.exists():
        print(f"✓ TCGA best params found: {tcga_path}")
        import json
        with open(tcga_path) as f:
            params = json.load(f)
        print(f"  Architecture: {params.get('architecture_2layer', params.get('architecture_1layer', 'unknown'))}")
        print(f"  Dropout: {params.get('dropout')}")
        print(f"  Learning rate: {params.get('learning_rate')}")
    else:
        print(f"✗ TCGA best params not found: {tcga_path}")
        all_good = False
    
    if orien_path.exists():
        print(f"✓ ORIEN best params found: {orien_path}")
        import json
        with open(orien_path) as f:
            params = json.load(f)
        print(f"  Architecture: {params.get('architecture_2layer', params.get('architecture_1layer', 'unknown'))}")
        print(f"  Dropout: {params.get('dropout')}")
        print(f"  Learning rate: {params.get('learning_rate')}")
    else:
        print(f"✗ ORIEN best params not found: {orien_path}")
        all_good = False
    
    return all_good


def check_feature_selection_functions():
    """Check if feature_selection module has required functions."""
    print("\n" + "="*60)
    print("CHECKING FEATURE SELECTION MODULE")
    print("="*60)
    
    try:
        sys.path.insert(0, '.')
        from src.utils import feature_selection
        
        required_functions = [
            'compute_gene_importance_l2',
            'select_features_percentile',
            'get_selected_gene_names',
            'compute_bidirectional_consensus',
            'compare_with_chapter2_biomarkers'
        ]
        
        all_good = True
        for func_name in required_functions:
            if hasattr(feature_selection, func_name):
                print(f"✓ {func_name}")
            else:
                print(f"✗ {func_name} - MISSING")
                all_good = False
        
        return all_good
        
    except ImportError as e:
        print(f"✗ Cannot import feature_selection module: {e}")
        return False


def check_consensus_genes():
    """Check if Chapter 2 consensus genes file exists."""
    print("\n" + "="*60)
    print("CHECKING CHAPTER 2 CONSENSUS GENES")
    print("="*60)
    
    consensus_path = Path('data/raw/consensus_genes_308.txt')
    
    if consensus_path.exists():
        print(f"✓ Chapter 2 consensus genes found: {consensus_path}")
        with open(consensus_path) as f:
            genes = [line.strip() for line in f if line.strip()]
        print(f"  Total genes: {len(genes)}")
        print(f"  First 5: {genes[:5]}")
        return True
    else:
        print(f"⚠ Chapter 2 consensus genes not found: {consensus_path}")
        print("  This is optional - extraction will still work")
        return True  # Not critical


def main():
    """Run all verification checks."""
    print("\n" + "="*60)
    print("BIOMARKER EXTRACTION - DEPENDENCY VERIFICATION")
    print("="*60 + "\n")
    
    checks = [
        ("Python Imports", check_imports),
        ("Project Structure", check_project_structure),
        ("Hyperparameter Results", check_hyperparameter_results),
        ("Feature Selection Module", check_feature_selection_functions),
        ("Consensus Genes", check_consensus_genes),
    ]
    
    results = {}
    for name, check_func in checks:
        results[name] = check_func()
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    all_passed = all(results.values())
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:10s} {name}")
    
    print("="*60)
    
    if all_passed:
        print("\n✅ ALL CHECKS PASSED!")
        print("\nYou can proceed with biomarker extraction:")
        print("\npython scripts/extract_biomarkers_from_best_params.py \\")
        print("    --tcga_params results/20251104/hyperparam_tcga_20251104_034717/best_params.json \\")
        print("    --orien_params results/20251104/hyperparam_orien_20251104_032058/best_params.json \\")
        print("    --output_dir results/biomarker_extraction/ \\")
        print("    --selection_method percentile \\")
        print("    --percentile 95.0 \\")
        print("    --n_epochs 150")
        return 0
    else:
        print("\n⚠ SOME CHECKS FAILED")
        print("\nPlease fix the issues above before proceeding.")
        print("If you need help, share the output of this script.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
