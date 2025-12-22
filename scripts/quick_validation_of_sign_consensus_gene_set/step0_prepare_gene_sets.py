#!/usr/bin/env python3
"""
Step 0: Prepare Gene Sets for Sign-Filter Validation

Purpose: Create two gene sets for comparison
- Set A: 68 consensus genes (magnitude-only, from k=140)
- Set B: 26 sign-consistent genes (magnitude + sign filter)

Output:
    results_v2/07_sign_filter_validation/inputs/
    ├── gene_set_A_68_magnitude_only.txt
    ├── gene_set_B_26_sign_consistent.txt
    └── validation_config.json

Author: [Your Name]
Date: 2024-12
"""

import os
import json
import pandas as pd
from datetime import datetime

# ============================================================================
# CONFIGURATION - MODIFY THESE PATHS FOR YOUR SYSTEM
# ============================================================================
CONFIG = {
    # Input files
    'consensus_genes_file': 'results_v2/02b_biomarker_discovery_ig/k_selection_with_tuning/k140/consensus_genes/consensus_genes.txt',
    'sign_analysis_file': 'results_v2/06_importance_methods/sign_consistency_analysis/tables/sign_analysis_full_table.csv',
    
    # Hyperparameter files
    'best_params_tcga': 'results_v2/02b_biomarker_discovery_ig/k_selection_with_tuning/k140/hyperparameter_tuning/tcga/best_params.json',
    'best_params_orien': 'results_v2/02b_biomarker_discovery_ig/k_selection_with_tuning/k140/hyperparameter_tuning/orien/best_params.json',
    
    # Data files
    'expr_tcga': 'data/raw/tcga_batch_corrected_2sv.csv',
    'expr_orien': 'data/raw/orien_batch_corrected.csv',
    'surv_tcga': 'data/processed/surv_tcga_harmonized.csv',
    'surv_orien': 'data/processed/surv_orien_harmonized.csv',
    
    # Output directory
    'output_dir': 'results_v2/07_sign_filter_validation',
    
    # Validation settings
    'seeds': [42, 123, 456, 789, 1011],
    'max_epochs': 200,
}


def main():
    print("=" * 70)
    print("Step 0: Prepare Gene Sets for Sign-Filter Validation")
    print("=" * 70)
    
    # Create output directories
    output_dir = CONFIG['output_dir']
    inputs_dir = os.path.join(output_dir, 'inputs')
    os.makedirs(inputs_dir, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Load 68 consensus genes
    # -------------------------------------------------------------------------
    print("\n[1] Loading 68 consensus genes from k=140...")
    with open(CONFIG['consensus_genes_file'], 'r') as f:
        consensus_68 = [line.strip() for line in f if line.strip()]
    print(f"    Loaded {len(consensus_68)} genes")
    
    # -------------------------------------------------------------------------
    # Load sign analysis and identify sign-consistent genes
    # -------------------------------------------------------------------------
    print("\n[2] Loading sign analysis table...")
    sign_df = pd.read_csv(CONFIG['sign_analysis_file'])
    print(f"    Total genes in sign analysis: {len(sign_df)}")
    
    # Filter for cross-cohort sign match
    sign_consistent_all = sign_df[sign_df['cross_cohort_match'] == True]['gene'].tolist()
    print(f"    Sign-consistent genes (cross_cohort_match=True): {len(sign_consistent_all)}")
    
    # Intersection with 68 consensus genes
    sign_consistent_consensus = [g for g in consensus_68 if g in sign_consistent_all]
    print(f"    Intersection (Set B): {len(sign_consistent_consensus)} genes")
    
    # -------------------------------------------------------------------------
    # Get sign direction for Set B genes
    # -------------------------------------------------------------------------
    print("\n[3] Extracting sign direction for Set B genes...")
    set_b_info = sign_df[sign_df['gene'].isin(sign_consistent_consensus)][
        ['gene', 'tcga_consensus_sign', 'orien_consensus_sign', 'both_positive', 'both_negative']
    ].copy()
    
    n_positive = set_b_info['both_positive'].sum()
    n_negative = set_b_info['both_negative'].sum()
    print(f"    Risk genes (positive IG): {n_positive}")
    print(f"    Protective genes (negative IG): {n_negative}")
    
    # -------------------------------------------------------------------------
    # Save gene sets
    # -------------------------------------------------------------------------
    print("\n[4] Saving gene sets...")
    
    # Set A: 68 genes (magnitude-only)
    set_a_file = os.path.join(inputs_dir, 'gene_set_A_68_magnitude_only.txt')
    with open(set_a_file, 'w') as f:
        for gene in consensus_68:
            f.write(f"{gene}\n")
    print(f"    Set A: {set_a_file}")
    
    # Set B: 26 genes (sign-consistent)
    set_b_file = os.path.join(inputs_dir, 'gene_set_B_26_sign_consistent.txt')
    with open(set_b_file, 'w') as f:
        for gene in sign_consistent_consensus:
            f.write(f"{gene}\n")
    print(f"    Set B: {set_b_file}")
    
    # Set B with annotations
    set_b_annotated_file = os.path.join(inputs_dir, 'gene_set_B_26_annotated.csv')
    set_b_info.to_csv(set_b_annotated_file, index=False)
    print(f"    Set B annotated: {set_b_annotated_file}")
    
    # -------------------------------------------------------------------------
    # Create validation config
    # -------------------------------------------------------------------------
    print("\n[5] Creating validation config...")
    
    validation_config = {
        'created_at': datetime.now().isoformat(),
        'description': 'Quick validation: Compare 68 magnitude-only vs 26 sign-consistent genes',
        
        'gene_sets': {
            'set_A': {
                'name': '68_magnitude_only',
                'n_genes': len(consensus_68),
                'file': 'gene_set_A_68_magnitude_only.txt',
                'description': 'Consensus genes from k=140 based on IG magnitude only'
            },
            'set_B': {
                'name': '26_sign_consistent',
                'n_genes': len(sign_consistent_consensus),
                'file': 'gene_set_B_26_sign_consistent.txt',
                'description': 'Subset of Set A with cross-cohort sign consistency',
                'n_risk_genes': int(n_positive),
                'n_protective_genes': int(n_negative)
            }
        },
        
        'hyperparameters': {
            'tcga': CONFIG['best_params_tcga'],
            'orien': CONFIG['best_params_orien']
        },
        
        'data_paths': {
            'expr_tcga': CONFIG['expr_tcga'],
            'expr_orien': CONFIG['expr_orien'],
            'surv_tcga': CONFIG['surv_tcga'],
            'surv_orien': CONFIG['surv_orien']
        },
        
        'validation_protocol': {
            'directions': ['tcga_to_orien', 'orien_to_tcga'],
            'seeds': CONFIG['seeds'],
            'max_epochs': CONFIG['max_epochs'],
            'metric': 'c_index'
        }
    }
    
    config_file = os.path.join(inputs_dir, 'validation_config.json')
    with open(config_file, 'w') as f:
        json.dump(validation_config, f, indent=2)
    print(f"    Config: {config_file}")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Gene Set Comparison:
--------------------
Set A (magnitude-only):    {len(consensus_68):3d} genes
Set B (sign-consistent):   {len(sign_consistent_consensus):3d} genes
Reduction:                 {len(consensus_68) - len(sign_consistent_consensus):3d} genes ({100*(1 - len(sign_consistent_consensus)/len(consensus_68)):.1f}%)

Set B Breakdown:
- Risk genes (positive IG in both cohorts):      {n_positive}
- Protective genes (negative IG in both cohorts): {n_negative}

Output files created in: {inputs_dir}/
- gene_set_A_68_magnitude_only.txt
- gene_set_B_26_sign_consistent.txt
- gene_set_B_26_annotated.csv
- validation_config.json

Next step: Run step1_quick_validation.py
""")
    
    return validation_config


if __name__ == '__main__':
    main()
