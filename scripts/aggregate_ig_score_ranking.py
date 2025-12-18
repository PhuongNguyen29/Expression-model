#!/usr/bin/env python3
"""
Script: aggregate_ig_score_ranking.py
Purpose: Aggregate Integrated Gradients (IG) importance scores across multiple seeds
         and generate consensus gene rankings for biomarker selection.

Method:
    1. Load IG importance from all seeds for TCGA and ORIEN
    2. Compute mean/std/CV across seeds for each gene
    3. Rank genes by mean IG magnitude (separately per cohort)
    4. Find consensus genes (TCGA top-k ∩ ORIEN top-k) for each k
    5. Evaluate overlap with Cox consensus genes
    6. Compute cross-cohort ranking correlation

Reference:
    - Sundararajan et al. (2017) "Axiomatic Attribution for Deep Networks" - ICML
    - Multi-seed aggregation for robust feature importance (Picard et al., 2021)

Usage:
    python scripts/aggregate_ig_score_ranking.py \
        --input_dir results_v2/06_importance_methods \
        --output_dir results_v2/06_importance_methods/aggregated \
        --seeds 42 123 456 789 1011 \
        --k_values 

Author: Phuong
Created: 2025
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple
from scipy.stats import spearmanr

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Default configuration
DEFAULT_SEEDS = [42, 123, 456, 789, 1011]
DEFAULT_K_VALUES = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]
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


def compute_consensus_genes(
    tcga_ranked: List[str],
    orien_ranked: List[str],
    k_values: List[int]
) -> Dict[int, Dict]:
    """
    Compute consensus genes (TCGA top-k ∩ ORIEN top-k) for each k.
    
    Args:
        tcga_ranked: List of TCGA genes sorted by importance (descending)
        orien_ranked: List of ORIEN genes sorted by importance (descending)
        k_values: List of k values to evaluate
    
    Returns:
        Dictionary with consensus gene statistics for each k
    """
    results = {}
    
    for k in k_values:
        tcga_top_k = set(tcga_ranked[:k])
        orien_top_k = set(orien_ranked[:k])
        consensus = tcga_top_k & orien_top_k
        
        # Expected overlap by chance
        n_total = len(tcga_ranked)
        expected_overlap = (k * k) / n_total
        
        results[k] = {
            'n_consensus': len(consensus),
            'k': k,
            'overlap_pct': len(consensus) / k * 100,
            'expected_overlap': expected_overlap,
            'enrichment': len(consensus) / expected_overlap if expected_overlap > 0 else 0,
            'genes': sorted(list(consensus))
        }
    
    return results


def compute_cross_cohort_correlation(
    tcga_agg: pd.DataFrame,
    orien_agg: pd.DataFrame
) -> Dict:
    """
    Compute Spearman correlation between TCGA and ORIEN gene rankings.
    
    This assesses whether both cohorts identify similar important genes,
    which is important for biomarker generalizability.
    
    Args:
        tcga_agg: TCGA aggregated DataFrame with 'gene' and 'ig_mean' columns
        orien_agg: ORIEN aggregated DataFrame with 'gene' and 'ig_mean' columns
    
    Returns:
        Dictionary with correlation statistics
    """
    # Merge on gene name to align rankings
    merged = tcga_agg[['gene', 'ig_mean', 'rank']].merge(
        orien_agg[['gene', 'ig_mean', 'rank']],
        on='gene',
        suffixes=('_tcga', '_orien')
    )
    
    # Compute Spearman correlation on importance scores
    rho_importance, p_importance = spearmanr(
        merged['ig_mean_tcga'],
        merged['ig_mean_orien']
    )
    
    # Compute Spearman correlation on ranks
    rho_rank, p_rank = spearmanr(
        merged['rank_tcga'],
        merged['rank_orien']
    )
    
    return {
        'spearman_rho_importance': float(rho_importance),
        'spearman_p_importance': float(p_importance),
        'spearman_rho_rank': float(rho_rank),
        'spearman_p_rank': float(p_rank),
        'n_genes': len(merged)
    }


def save_consensus_gene_lists(
    consensus_results: Dict[int, Dict],
    output_dir: Path,
    cox_genes: List[str] = None
):
    """
    Save consensus gene lists for each k value.
    
    Args:
        consensus_results: Dictionary with consensus gene info for each k
        output_dir: Output directory
        cox_genes: Optional list of Cox consensus genes for annotation
    """
    consensus_dir = output_dir / 'consensus_genes'
    consensus_dir.mkdir(parents=True, exist_ok=True)
    
    cox_set = set(cox_genes) if cox_genes else set()
    
    for k, info in consensus_results.items():
        genes = info['genes']
        
        # Create DataFrame with Cox overlap annotation
        df = pd.DataFrame({'gene': genes})
        if cox_genes:
            df['in_cox_consensus'] = df['gene'].isin(cox_set)
        
        # Save to file
        filepath = consensus_dir / f'ig_consensus_k{k}.txt'
        with open(filepath, 'w') as f:
            f.write('\n'.join(genes))
        
        # Also save as CSV with annotations
        csv_filepath = consensus_dir / f'ig_consensus_k{k}.csv'
        df.to_csv(csv_filepath, index=False)
    
    logger.info(f"  Saved consensus gene lists to: {consensus_dir}")


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
    # STEP 3: Compute cross-cohort correlation
    # ================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 3: Cross-Cohort Ranking Correlation")
    logger.info("="*70)
    
    correlation = compute_cross_cohort_correlation(tcga_agg, orien_agg)
    
    logger.info(f"\n  Cross-cohort Spearman correlation:")
    logger.info(f"    IG importance: rho = {correlation['spearman_rho_importance']:.4f} (p = {correlation['spearman_p_importance']:.2e})")
    logger.info(f"    Rank:          rho = {correlation['spearman_rho_rank']:.4f} (p = {correlation['spearman_p_rank']:.2e})")
    
    if correlation['spearman_rho_importance'] > 0.3:
        logger.info("    → Moderate-to-strong cross-cohort agreement")
    elif correlation['spearman_rho_importance'] > 0.1:
        logger.info("    → Weak cross-cohort agreement")
    else:
        logger.info("    → Poor cross-cohort agreement - consensus genes may be limited")
    
    # ================================================================
    # STEP 4: Compute consensus genes (TCGA ∩ ORIEN)
    # ================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 4: Compute Consensus Genes (TCGA top-k ∩ ORIEN top-k)")
    logger.info("="*70)
    
    consensus_results = compute_consensus_genes(tcga_ranked, orien_ranked, args.k_values)
    
    logger.info("\n  Consensus Gene Statistics:")
    logger.info("-" * 60)
    logger.info(f"{'k':>6} | {'Consensus':>10} | {'Overlap %':>10} | {'Expected':>10} | {'Enrichment':>10}")
    logger.info("-" * 60)
    
    for k in args.k_values:
        info = consensus_results[k]
        logger.info(f"{k:>6} | {info['n_consensus']:>10} | {info['overlap_pct']:>9.1f}% | {info['expected_overlap']:>10.1f} | {info['enrichment']:>10.2f}x")
    
    # Save consensus gene lists
    save_consensus_gene_lists(consensus_results, output_dir, cox_genes)
    
    # ================================================================
    # STEP 5: Evaluate Cox gene overlap
    # ================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 5: Evaluate Cox Gene Overlap")
    logger.info("="*70)
    
    if cox_genes:
        tcga_cox_overlap = compute_cox_overlap(tcga_ranked, cox_genes, args.k_values)
        orien_cox_overlap = compute_cox_overlap(orien_ranked, cox_genes, args.k_values)
        
        logger.info("\n  Cox Gene Overlap (Aggregated IG Rankings):")
        logger.info("-" * 50)
        logger.info(f"{'k':>6} | {'TCGA':>10} | {'ORIEN':>10}")
        logger.info("-" * 50)
        
        for k in args.k_values:
            tcga_n = tcga_cox_overlap[k]['n_overlap']
            orien_n = orien_cox_overlap[k]['n_overlap']
            logger.info(f"{k:>6} | {tcga_n:>6}/20   | {orien_n:>6}/20")
        
        # Also check Cox overlap within consensus genes
        logger.info("\n  Cox Genes in Consensus (TCGA ∩ ORIEN ∩ Cox):")
        logger.info("-" * 50)
        cox_set = set(cox_genes)
        for k in args.k_values:
            consensus_set = set(consensus_results[k]['genes'])
            triple_overlap = consensus_set & cox_set
            logger.info(f"    k={k}: {len(triple_overlap)}/20 Cox genes in {len(consensus_set)} consensus genes")
    else:
        tcga_cox_overlap = {}
        orien_cox_overlap = {}
        logger.warning("Cox genes not available - skipping overlap analysis")
    
    # ================================================================
    # STEP 6: Compare with L2 method (if available)
    # ================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 6: Method Comparison (IG vs L2)")
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
            
            # Compute L2 consensus genes
            l2_consensus = compute_consensus_genes(l2_tcga_ranked, l2_orien_ranked, args.k_values)
            
            logger.info("\n  Cox Overlap Comparison (IG vs L2):")
            logger.info("-" * 70)
            logger.info(f"{'k':>6} | {'TCGA IG':>10} | {'TCGA L2':>10} | {'ORIEN IG':>10} | {'ORIEN L2':>10}")
            logger.info("-" * 70)
            
            for k in args.k_values:
                if cox_genes:
                    l2_tcga_overlap = len(set(l2_tcga_ranked[:k]) & set(cox_genes))
                    l2_orien_overlap = len(set(l2_orien_ranked[:k]) & set(cox_genes))
                    
                    l2_comparison[k] = {
                        'tcga_l2': l2_tcga_overlap,
                        'orien_l2': l2_orien_overlap,
                        'l2_consensus_n': l2_consensus[k]['n_consensus']
                    }
                    
                    logger.info(f"{k:>6} | {tcga_cox_overlap[k]['n_overlap']:>6}/20   | {l2_tcga_overlap:>6}/20   | {orien_cox_overlap[k]['n_overlap']:>6}/20   | {l2_orien_overlap:>6}/20")
            
            # Compare consensus gene counts
            logger.info("\n  Consensus Gene Count Comparison (IG vs L2):")
            logger.info("-" * 50)
            logger.info(f"{'k':>6} | {'IG Consensus':>15} | {'L2 Consensus':>15}")
            logger.info("-" * 50)
            for k in args.k_values:
                logger.info(f"{k:>6} | {consensus_results[k]['n_consensus']:>15} | {l2_consensus[k]['n_consensus']:>15}")
    else:
        logger.info(f"\nL2 aggregated results not found at: {l2_file}")
        logger.info("Skipping method comparison")
    
    # ================================================================
    # STEP 7: Save summary
    # ================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 7: Save Summary")
    logger.info("="*70)
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'input_dir': str(input_dir),
        'seeds': args.seeds,
        'n_seeds': len(args.seeds),
        'k_values': args.k_values,
        'n_genes': len(tcga_ranked),
        'cross_cohort_correlation': correlation,
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
        'consensus_genes': {
            str(k): {
                'n_consensus': v['n_consensus'],
                'overlap_pct': v['overlap_pct'],
                'enrichment': v['enrichment'],
                'genes': v['genes']
            } for k, v in consensus_results.items()
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
    logger.info(f"  - {output_dir / 'consensus_genes/'} (consensus gene lists for each k)")
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
            k_compare = 100 if 100 in args.k_values else args.k_values[-1]
            logger.info(f"\nComparison with L2 (at k={k_compare}):")
            logger.info(f"  TCGA:  IG {tcga_cox_overlap[k_compare]['n_overlap']}/20 vs L2 {l2_comparison.get(str(k_compare), {}).get('tcga_l2', 'N/A')}/20")
            logger.info(f"  ORIEN: IG {orien_cox_overlap[k_compare]['n_overlap']}/20 vs L2 {l2_comparison.get(str(k_compare), {}).get('orien_l2', 'N/A')}/20")
    
    # Cross-cohort summary
    logger.info(f"\n*** CROSS-COHORT AGREEMENT ***")
    logger.info(f"  Spearman rho (importance): {correlation['spearman_rho_importance']:.4f}")
    
    # Consensus gene summary
    logger.info(f"\n*** CONSENSUS GENES (for k-selection) ***")
    for k in args.k_values:
        info = consensus_results[k]
        logger.info(f"  k={k}: {info['n_consensus']} consensus genes ({info['enrichment']:.1f}x enrichment)")
    
    logger.info(f"\n*** NEXT STEP ***")
    logger.info(f"Use consensus gene lists for k-sweep cross-cohort validation:")
    logger.info(f"  - {output_dir / 'consensus_genes/'}")
    
    logger.info("\n" + "="*70)
    
    return summary


if __name__ == "__main__":
    summary = main()
