#!/usr/bin/env python3
"""
Quick script to examine consensus genes at different k values.
Helps decide k-range for k-sweep validation.

Usage:
    python scripts/examine_consensus_at_k.py \
        --input_dir results_v2/06_importance_methods/aggregated
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, 
                        default='results_v2/06_importance_methods/aggregated')
    parser.add_argument('--cox_genes', type=str,
                        default='data/raw/cox_consensus_genes_20.txt')
    parser.add_argument('--k_values', type=int, nargs='+',
                        default=[20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 175, 200])
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    
    # Load aggregated rankings
    tcga_file = input_dir / 'tcga_ig_aggregated.csv'
    orien_file = input_dir / 'orien_ig_aggregated.csv'
    
    if not tcga_file.exists() or not orien_file.exists():
        print(f"ERROR: Aggregated files not found in {input_dir}")
        print("Run aggregate_ig_score_ranking.py first")
        return
    
    tcga_df = pd.read_csv(tcga_file)
    orien_df = pd.read_csv(orien_file)
    
    # Get ranked gene lists
    tcga_ranked = tcga_df.sort_values('ig_mean', ascending=False)['gene'].tolist()
    orien_ranked = orien_df.sort_values('ig_mean', ascending=False)['gene'].tolist()
    
    total_genes = len(tcga_ranked)
    
    # Load Cox genes if available
    cox_genes = []
    if Path(args.cox_genes).exists():
        with open(args.cox_genes, 'r') as f:
            cox_genes = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(cox_genes)} Cox consensus genes\n")
    
    # Print header
    print("="*80)
    print("CONSENSUS GENE ANALYSIS AT DIFFERENT K VALUES")
    print("="*80)
    print(f"\nTotal genes: {total_genes}")
    print(f"\n{'k':>6} | {'Consensus (m)':>14} | {'Overlap %':>10} | {'Expected':>10} | {'Enrichment':>10} | {'Cox Overlap':>12}")
    print("-"*80)
    
    results = []
    
    for k in args.k_values:
        if k > total_genes:
            continue
            
        tcga_top_k = set(tcga_ranked[:k])
        orien_top_k = set(orien_ranked[:k])
        
        consensus = tcga_top_k & orien_top_k
        n_consensus = len(consensus)
        overlap_pct = n_consensus / k * 100
        
        # Expected by random chance
        expected = (k * k) / total_genes
        enrichment = n_consensus / expected if expected > 0 else 0
        
        # Cox overlap
        if cox_genes:
            cox_in_consensus = len(consensus & set(cox_genes))
            cox_str = f"{cox_in_consensus}/20"
        else:
            cox_in_consensus = None
            cox_str = "N/A"
        
        print(f"{k:>6} | {n_consensus:>14} | {overlap_pct:>9.1f}% | {expected:>9.1f} | {enrichment:>9.1f}x | {cox_str:>12}")
        
        results.append({
            'k': k,
            'm': n_consensus,
            'overlap_pct': overlap_pct,
            'expected': expected,
            'enrichment': enrichment,
            'cox_overlap': cox_in_consensus
        })
    
    print("-"*80)
    
    # Find interesting k values
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR K-SWEEP RANGE")
    print("="*80)
    
    # Find k where enrichment starts to plateau
    results_df = pd.DataFrame(results)
    
    # Best enrichment
    best_enrichment_idx = results_df['enrichment'].idxmax()
    best_k_enrichment = results_df.loc[best_enrichment_idx, 'k']
    
    # Best Cox overlap
    if cox_genes:
        best_cox_idx = results_df['cox_overlap'].idxmax()
        best_k_cox = results_df.loc[best_cox_idx, 'k']
        print(f"\nBest Cox overlap: k={best_k_cox} ({results_df.loc[best_cox_idx, 'cox_overlap']}/20 Cox genes)")
    
    print(f"Best enrichment: k={best_k_enrichment} ({results_df.loc[best_enrichment_idx, 'enrichment']:.1f}x)")
    
    # Suggest k range where m >= 20 (minimum for meaningful biomarker set)
    valid_ks = results_df[results_df['m'] >= 20]['k'].tolist()
    if valid_ks:
        print(f"\nK values with m >= 20 consensus genes: {valid_ks}")
        print(f"Suggested k-sweep range: [{min(valid_ks)}, {max(valid_ks)}]")
    
    # Show top consensus genes at a few k values
    print("\n" + "="*80)
    print("TOP CONSENSUS GENES (for verification)")
    print("="*80)
    
    for k in [50, 100, 150]:
        if k > total_genes:
            continue
        tcga_top_k = set(tcga_ranked[:k])
        orien_top_k = set(orien_ranked[:k])
        consensus = sorted(list(tcga_top_k & orien_top_k))
        
        # Check Cox overlap
        if cox_genes:
            cox_in = sorted([g for g in consensus if g in cox_genes])
            print(f"\nk={k}: {len(consensus)} consensus genes, {len(cox_in)}/20 Cox genes")
            if cox_in:
                print(f"  Cox genes in consensus: {cox_in[:10]}{'...' if len(cox_in) > 10 else ''}")
        else:
            print(f"\nk={k}: {len(consensus)} consensus genes")
            print(f"  First 10: {consensus[:10]}")


if __name__ == "__main__":
    main()
