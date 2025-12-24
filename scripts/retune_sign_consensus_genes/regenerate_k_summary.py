#!/usr/bin/env python3
"""
Regenerate k_selection_summary.csv from individual k-folder results.

This script reads the best_params.json files from each k-folder and
creates a consolidated summary CSV.

Usage:
    python regenerate_k_summary.py --input_dir results_v2/02c_biomarker_discovery_ig_signfilter/k_selection_with_tuning
"""

import json
import argparse
from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='Regenerate k_selection_summary.csv')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing k-value subdirectories')
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    
    # Find all k-directories
    k_dirs = sorted([d for d in input_dir.iterdir() 
                     if d.is_dir() and d.name.startswith('k') and d.name[1:].isdigit()])
    
    print(f"Found {len(k_dirs)} k-directories: {[d.name for d in k_dirs]}")
    
    all_results = []
    
    for k_dir in k_dirs:
        k = int(k_dir.name[1:])  # Extract k value from folder name (e.g., k040 -> 40)
        
        # Load TCGA best params
        tcga_file = k_dir / "hyperparameter_tuning" / "tcga" / "best_params.json"
        orien_file = k_dir / "hyperparameter_tuning" / "orien" / "best_params.json"
        gene_info_file = k_dir / "consensus_genes" / "gene_info.json"
        
        if not tcga_file.exists() or not orien_file.exists():
            print(f"  Skipping k={k}: missing best_params.json")
            continue
        
        with open(tcga_file, 'r') as f:
            tcga_results = json.load(f)
        
        with open(orien_file, 'r') as f:
            orien_results = json.load(f)
        
        # Load gene info if available
        if gene_info_file.exists():
            with open(gene_info_file, 'r') as f:
                gene_info = json.load(f)
            m = gene_info.get('m', tcga_results.get('m', 0))
            overlap_pct = gene_info.get('overlap_pct', 0)
            enrichment = gene_info.get('enrichment', 0)
        else:
            m = tcga_results.get('m', 0)
            overlap_pct = 0
            enrichment = 0
        
        k_results = {
            'k': k,
            'm': m,
            'overlap_pct': overlap_pct,
            'enrichment': enrichment,
            'tcga_cv_cindex': tcga_results.get('best_cv_cindex', 0),
            'orien_cv_cindex': orien_results.get('best_cv_cindex', 0),
            'importance_method': tcga_results.get('importance_method', 'integrated_gradients'),
            'gene_pool': tcga_results.get('gene_pool', 'sign_consistent_141')
        }
        
        all_results.append(k_results)
        print(f"  k={k}: m={m}, TCGA CV={k_results['tcga_cv_cindex']:.4f}, ORIEN CV={k_results['orien_cv_cindex']:.4f}")
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(all_results)
    summary_df = summary_df.sort_values('k').reset_index(drop=True)
    
    # Save summary
    summary_dir = input_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = summary_dir / 'k_selection_summary.csv'
    summary_df.to_csv(output_file, index=False)
    
    print(f"\n{'='*60}")
    print("REGENERATED K-SELECTION SUMMARY")
    print('='*60)
    print(summary_df.to_string(index=False))
    print(f"\nSaved to: {output_file}")
    print('='*60)
    
    return summary_df

if __name__ == '__main__':
    main()
