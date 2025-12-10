#!/usr/bin/env python3
"""
Script: aggregate_ig_score_ranking.py
Purpose: Aggregate Integrated Gradients (IG) importance scores across multiple seeds
         and generate consensus gene rankings for biomarker selection.

Method:
    1. Load IG importance from all seeds for TCGA and ORIEN
    2. Compute mean/std/CV across seeds for each gene
    3. Rank genes by mean IG magnitude (separately per cohort)
    4. Find consensus genes (TCGA top-k ∩ ORIEN top-k)
    5. Evaluate overlap with Cox consensus genes

Reference:
    - Sundararajan et al. (2017) "Axiomatic Attribution for Deep Networks" - ICML
    - Multi-seed aggregation for robust feature importance (Picard et al., 2021)

Usage:
    python scripts/aggregate_ig_score_ranking.py \
        --input_dir results_v2/06_importance_methods \
        --output_dir results_v2/06_importance_methods/aggregated \
        --seeds 42 123 456 789 1011 \
        --k_values 20 50 75 100 150

Author: Phuong
Created: 2024
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Set

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Default configuration
DEFAULT_SEEDS = [42, 123, 456, 789, 1011]
DEFAULT_K_VALUES = [20, 30, 50, 75, 100, 150]
COX_GENES_FILE = "data/raw/cox_consensus_genes_20.txt"


def load_cox_consensus_genes(filepath: str) -> List[str]:
    """Load Cox consensus genes from file."""
    if not Path(filepath).exists():
        logger.warning(f"Cox genes file not found: {filepath}")
        return []
    
    with open(filepath, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Loaded {len(genes)} Cox consensus genes")
    return genes


def load_seed_importance(
    input_dir: Path,
    seed: int,
    cohort: str
) -> pd.DataFrame:
    """
    Load IG importance for a single seed and cohort.
    
    Args:
        input_dir: Base directory containing seed folders
        seed: Seed number
        cohort: 'tcga' or 'orien'
    
    Returns:
        DataFrame with columns: gene, importance_magnitude, importance_signed, importance_std
    """
    seed_dir = input_dir / f"seed_{seed}"
    filepath = seed_dir / f"{cohort}_ig_importance.csv"
    
    if not filepath.exists():
        raise FileNotFoundError(f"IG importance file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Validate expected columns
    required_cols = ['gene', 'importance_magnitude']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {filepath}")
    
    return df


def aggregate_cohort_importance(
    input_dir: Path,
    seeds: List[int],
    cohort: str
) -> pd.DataFrame:
    """
    Aggregate IG importance across all seeds for one cohort.
    
    Args:
        input_dir: Base directory containing seed folders
        seeds: List of seed numbers
        cohort: 'tcga' or 'orien'
    
    Returns:
        DataFrame with aggregated statistics per gene
    """
    logger.info(f"\nAggregating {cohort.upper()} IG importance across {len(seeds)} seeds...")
    
    # Load all seeds
    seed_dfs = {}
    for seed in seeds:
        try:
            df = load_seed_importance(input_dir, seed, cohort)
            seed_dfs[seed] = df
            logger.info(f"  Seed {seed}: {len(df)} genes loaded")
        except FileNotFoundError as e:
            logger.error(f"  Seed {seed}: {e}")
            continue
    
    if len(seed_dfs) == 0:
        raise RuntimeError(f"No valid seed data found for {cohort}")
    
    if len(seed_dfs) < len(seeds):
        logger.warning(f"  Only {len(seed_dfs)}/{len(seeds)} seeds loaded successfully")
    
    # Get gene list from first seed (should be consistent)
    first_seed = list(seed_dfs.keys())[0]
    genes = seed_dfs[first_seed]['gene'].tolist()
    
    # Verify all seeds have same genes
    for seed, df in seed_dfs.items():
        if set(df['gene']) != set(genes):
            logger.warning(f"  Seed {seed} has different gene set!")
    
    # Create merged DataFrame
    merged = pd.DataFrame({'gene': genes})
    
    # Add magnitude from each seed
    for seed, df in seed_dfs.items():
        df_indexed = df.set_index('gene')
        merged[f'seed_{seed}_mag'] = merged['gene'].map(
            df_indexed['importance_magnitude']
        )
        
        # Also store signed importance if available
        if 'importance_signed' in df.columns:
            merged[f'seed_{seed}_signed'] = merged['gene'].map(
                df_indexed['importance_signed']
            )
    
    # Compute statistics across seeds (magnitude)
    mag_cols = [f'seed_{s}_mag' for s in seed_dfs.keys()]
    merged['ig_mean'] = merged[mag_cols].mean(axis=1)
    merged['ig_std'] = merged[mag_cols].std(axis=1, ddof=1)
    merged['ig_cv'] = merged['ig_std'] / (merged['ig_mean'] + 1e-10)
    merged['ig_min'] = merged[mag_cols].min(axis=1)
    merged['ig_max'] = merged[mag_cols].max(axis=1)
    
    # Compute statistics for signed importance if available
    signed_cols = [f'seed_{s}_signed' for s in seed_dfs.keys() 
                   if f'seed_{s}_signed' in merged.columns]
    if signed_cols:
        merged['ig_signed_mean'] = merged[signed_cols].mean(axis=1)
        merged['ig_signed_std'] = merged[signed_cols].std(axis=1, ddof=1)
    
    # Rank by mean magnitude (descending)
    merged = merged.sort_values('ig_mean', ascending=False).reset_index(drop=True)
    merged['rank'] = range(1, len(merged) + 1)
    
    # Log statistics
    logger.info(f"\n  Aggregation Statistics for {cohort.upper()}:")
    logger.info(f"    Total genes: {len(merged)}")
    logger.info(f"    IG mean range: [{merged['ig_mean'].min():.6f}, {merged['ig_mean'].max():.6f}]")
    logger.info(f"    IG mean CV (across genes): {merged['ig_mean'].std() / merged['ig_mean'].mean():.4f}")
    logger.info(f"    Average cross-seed CV: {merged['ig_cv'].mean():.4f}")
    
    return merged


def compute_cox_overlap(
    ranked_genes: List[str],
    cox_genes: List[str],
    k_values: List[int]
) -> Dict[int, Dict]:
    """
    Compute overlap between top-k ranked genes and Cox consensus genes.
    
    Args:
        ranked_genes: List of genes sorted by importance (descending)
        cox_genes: List of Cox consensus genes
        k_values: List of k values to evaluate
    
    Returns:
        Dictionary with overlap statistics for each k
    """
    cox_set = set(cox_genes)
    results = {}
    
    for k in k_values:
        top_k = set(ranked_genes[:k])
        overlap = top_k & cox_set
        
        results[k] = {
            'n_overlap': len(overlap),
            'n_cox': len(cox_genes),
            'overlap_pct': len(overlap) / len(cox_genes) * 100 if cox_genes else 0,
            'genes': sorted(list(overlap))
        }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate IG importance scores across seeds and generate consensus rankings"
    )
    parser.add_argument('--input_dir', type=str, 
                        default='results_v2/06_importance_methods',
                        help='Directory containing seed_* folders with IG results')
    parser.add_argument('--output_dir', type=str,
                        default=None,
                        help='Output directory (default: input_dir/aggregated)')
    parser.add_argument('--seeds', type=int, nargs='+',
                        default=DEFAULT_SEEDS,
                        help='List of seeds to aggregate')
    parser.add_argument('--k_values', type=int, nargs='+',
                        default=DEFAULT_K_VALUES,
                        help='K values for consensus gene selection')
    parser.add_argument('--cox_genes', type=str,
                        default=COX_GENES_FILE,
                        help='Path to Cox consensus genes file')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    
    if args.output_dir is None:
        output_dir = input_dir / 'aggregated'
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("AGGREGATE IG IMPORTANCE SCORES ACROSS SEEDS")
    logger.info("="*70)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"K values: {args.k_values}")
    logger.info("="*70)
    
    # Load Cox consensus genes
    cox_genes = load_cox_consensus_genes(args.cox_genes)
    
    # ================================================================
    # STEP 1: Aggregate TCGA importance
    # ================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 1: Aggregate TCGA IG Importance")
    logger.info("="*70)
    
    tcga_agg = aggregate_cohort_importance(input_dir, args.seeds, 'tcga')
    tcga_ranked = tcga_agg['gene'].tolist()
    
    # Save TCGA aggregated results
    tcga_output = output_dir / 'tcga_ig_aggregated.csv'
    tcga_agg.to_csv(tcga_output, index=False)
    logger.info(f"\nSaved: {tcga_output}")
    
    # ================================================================
    # STEP 2: Aggregate ORIEN importance
    # ================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 2: Aggregate ORIEN IG Importance")
    logger.info("="*70)
    
    orien_agg = aggregate_cohort_importance(input_dir, args.seeds, 'orien')
    orien_ranked = orien_agg['gene'].tolist()
    
    # Save ORIEN aggregated results
    orien_output = output_dir / 'orien_ig_aggregated.csv'
    orien_agg.to_csv(orien_output, index=False)
    logger.info(f"\nSaved: {orien_output}")
    
    # ================================================================
    # STEP 3: Evaluate Cox gene overlap
    # ================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 3: Evaluate Cox Gene Overlap")
    logger.info("="*70)
    
    if cox_genes:
        tcga_cox_overlap = compute_cox_overlap(tcga_ranked, cox_genes, args.k_values)
        orien_cox_overlap = compute_cox_overlap(orien_ranked, cox_genes, args.k_values)
        
        logger.info("\nCox Gene Overlap (Aggregated IG Rankings):")
        logger.info("-" * 50)
        logger.info(f"{'k':>6} | {'TCGA':>10} | {'ORIEN':>10}")
        logger.info("-" * 50)
        
        for k in args.k_values:
            tcga_n = tcga_cox_overlap[k]['n_overlap']
            orien_n = orien_cox_overlap[k]['n_overlap']
            logger.info(f"{k:>6} | {tcga_n:>6}/20   | {orien_n:>6}/20")
    else:
        tcga_cox_overlap = {}
        orien_cox_overlap = {}
        logger.warning("Cox genes not available - skipping overlap analysis")
    
    # ================================================================
    # STEP 4: Compare with L2 method (if available)
    # ================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 4: Method Comparison Summary")
    logger.info("="*70)
    
    # Check if L2 aggregated results exist for comparison
    l2_comparison = {}
    l2_file = input_dir.parent / '02_biomarker_discovery' / 'aggregated_gene_importances.csv'
    
    if l2_file.exists():
        logger.info(f"\nFound L2 aggregated results: {l2_file}")
        l2_df = pd.read_csv(l2_file)
        
        if 'tcga_importance_mean' in l2_df.columns and 'orien_importance_mean' in l2_df.columns:
            # Get L2 rankings
            l2_tcga_ranked = l2_df.sort_values('tcga_importance_mean', ascending=False)['gene_name'].tolist()
            l2_orien_ranked = l2_df.sort_values('orien_importance_mean', ascending=False)['gene_name'].tolist()
            
            logger.info("\nCox Overlap Comparison (IG vs L2):")
            logger.info("-" * 60)
            logger.info(f"{'k':>6} | {'TCGA IG':>10} | {'TCGA L2':>10} | {'ORIEN IG':>10} | {'ORIEN L2':>10}")
            logger.info("-" * 60)
            
            for k in args.k_values:
                if cox_genes:
                    l2_tcga_overlap = len(set(l2_tcga_ranked[:k]) & set(cox_genes))
                    l2_orien_overlap = len(set(l2_orien_ranked[:k]) & set(cox_genes))
                    
                    l2_comparison[k] = {
                        'tcga_l2': l2_tcga_overlap,
                        'orien_l2': l2_orien_overlap
                    }
                    
                    logger.info(f"{k:>6} | {tcga_cox_overlap[k]['n_overlap']:>6}/20   | {l2_tcga_overlap:>6}/20   | {orien_cox_overlap[k]['n_overlap']:>6}/20   | {l2_orien_overlap:>6}/20")
    else:
        logger.info(f"\nL2 aggregated results not found at: {l2_file}")
        logger.info("Skipping method comparison")
    
    # ================================================================
    # STEP 5: Save summary
    # ================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 5: Save Summary")
    logger.info("="*70)
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'input_dir': str(input_dir),
        'seeds': args.seeds,
        'n_seeds': len(args.seeds),
        'k_values': args.k_values,
        'n_genes': len(tcga_ranked),
        'tcga': {
            'ig_mean_range': [float(tcga_agg['ig_mean'].min()), float(tcga_agg['ig_mean'].max())],
            'avg_cross_seed_cv': float(tcga_agg['ig_cv'].mean()),
            'cox_overlap': {str(k): v for k, v in tcga_cox_overlap.items()} if cox_genes else {}
        },
        'orien': {
            'ig_mean_range': [float(orien_agg['ig_mean'].min()), float(orien_agg['ig_mean'].max())],
            'avg_cross_seed_cv': float(orien_agg['ig_cv'].mean()),
            'cox_overlap': {str(k): v for k, v in orien_cox_overlap.items()} if cox_genes else {}
        },
        'l2_comparison': {str(k): v for k, v in l2_comparison.items()} if l2_comparison else None
    }
    
    summary_file = output_dir / 'aggregation_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nSaved summary: {summary_file}")
    
    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    logger.info("\n" + "="*70)
    logger.info("AGGREGATION COMPLETE")
    logger.info("="*70)
    
    logger.info(f"\nOutput files:")
    logger.info(f"  - {tcga_output}")
    logger.info(f"  - {orien_output}")
    logger.info(f"  - {summary_file}")
    
    # Report best Cox overlap for verification
    if cox_genes:
        best_k_tcga = max(args.k_values, key=lambda k: tcga_cox_overlap[k]['n_overlap'])
        best_k_orien = max(args.k_values, key=lambda k: orien_cox_overlap[k]['n_overlap'])
        
        logger.info(f"\n*** IG METHOD VERIFICATION ***")
        logger.info(f"Best Cox overlap:")
        logger.info(f"  TCGA:  {tcga_cox_overlap[best_k_tcga]['n_overlap']}/20 at k={best_k_tcga}")
        logger.info(f"  ORIEN: {orien_cox_overlap[best_k_orien]['n_overlap']}/20 at k={best_k_orien}")
        
        if l2_comparison:
            logger.info(f"\nComparison with L2 (at k=100):")
            logger.info(f"  TCGA:  IG {tcga_cox_overlap[100]['n_overlap']}/20 vs L2 {l2_comparison.get('100', {}).get('tcga_l2', 'N/A')}/20")
            logger.info(f"  ORIEN: IG {orien_cox_overlap[100]['n_overlap']}/20 vs L2 {l2_comparison.get('100', {}).get('orien_l2', 'N/A')}/20")
    
    logger.info(f"\n*** NEXT STEP ***")
    logger.info(f"Use aggregated rankings for k-sweep validation:")
    logger.info(f"  - {tcga_output}")
    logger.info(f"  - {orien_output}")
    
    logger.info("\n" + "="*70)
    
    return summary


if __name__ == "__main__":
    summary = main()
