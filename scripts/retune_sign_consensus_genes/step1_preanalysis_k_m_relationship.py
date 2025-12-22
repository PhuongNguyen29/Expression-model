#!/usr/bin/env python3
"""
Step 1 Pre-Analysis: K vs M Relationship for Sign-Consistent Genes
======================================================================

Purpose:
    Compute consensus gene counts (m) for each k value BEFORE running 
    expensive hyperparameter tuning. This helps select the optimal k range
    for tuning.

Input:
    - sign_consistent_genes_141.txt: 141 genes with consistent sign direction
    - tcga_ig_aggregated.csv: TCGA IG importance rankings
    - orien_ig_aggregated.csv: ORIEN IG importance rankings

Output:
    - k_m_relationship_table.csv: Table of k, m, overlap%, enrichment
    - k_m_relationship_figure.png: Visualization
    - k_range_recommendation.json: Suggested k range for tuning

Method:
    For each k in [20, 30, 40, ..., 141]:
    1. Get top-k genes from TCGA (within 141 sign-consistent pool)
    2. Get top-k genes from ORIEN (within 141 sign-consistent pool)
    3. Compute intersection (m = consensus count)
    4. Calculate overlap% = m/k * 100
    5. Calculate enrichment = observed_overlap / expected_overlap
       where expected_overlap = k²/141 (random expectation)

References:
    - Bernau et al. (2014) Bioinformatics - Cross-study validation
    - Waldron et al. (2014) BMC Genomics - Biomarker specificity

Author: Generated for Phuong's dissertation
Date: 2024
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    # Input files
    'sign_consistent_genes': 'data/processed/sign_consistent_genes_141.txt',
    'tcga_ig_aggregated': 'results_v2/06_importance_methods/aggregated/tcga_ig_aggregated.csv',
    'orien_ig_aggregated': 'results_v2/06_importance_methods/aggregated/orien_ig_aggregated.csv',
    
    # Output directory
    'output_dir': 'results_v2/07_sign_filter_validation/preanalysis_k_m',
    
    # K values to analyze
    'k_values': [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 141],
    
    # Minimum consensus genes for practical biomarker panel
    'min_consensus_genes': 10,
    
    # Target consensus gene range (for recommendation)
    'target_m_min': 15,
    'target_m_max': 50,
}


def load_sign_consistent_genes(filepath: str) -> list:
    """Load the 141 sign-consistent genes."""
    with open(filepath, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]
    logger.info(f"Loaded {len(genes)} sign-consistent genes from {filepath}")
    return genes


def load_ig_rankings(tcga_file: str, orien_file: str, gene_pool: list) -> dict:
    """
    Load IG aggregated rankings and filter to sign-consistent gene pool.
    
    Returns:
        dict with 'tcga_ranked' and 'orien_ranked' lists (genes sorted by IG importance)
    """
    # Load TCGA
    tcga_df = pd.read_csv(tcga_file)
    tcga_df = tcga_df.set_index('gene')
    
    # Load ORIEN
    orien_df = pd.read_csv(orien_file)
    orien_df = orien_df.set_index('gene')
    
    # Filter to sign-consistent gene pool
    tcga_filtered = tcga_df.loc[tcga_df.index.isin(gene_pool)]
    orien_filtered = orien_df.loc[orien_df.index.isin(gene_pool)]
    
    # Sort by IG importance (descending)
    tcga_ranked = tcga_filtered.sort_values('ig_mean', ascending=False).index.tolist()
    orien_ranked = orien_filtered.sort_values('ig_mean', ascending=False).index.tolist()
    
    logger.info(f"TCGA: {len(tcga_ranked)} genes ranked by IG importance")
    logger.info(f"ORIEN: {len(orien_ranked)} genes ranked by IG importance")
    
    # Verify all genes are present
    missing_tcga = set(gene_pool) - set(tcga_ranked)
    missing_orien = set(gene_pool) - set(orien_ranked)
    
    if missing_tcga:
        logger.warning(f"Missing {len(missing_tcga)} genes from TCGA rankings")
    if missing_orien:
        logger.warning(f"Missing {len(missing_orien)} genes from ORIEN rankings")
    
    return {
        'tcga_ranked': tcga_ranked,
        'orien_ranked': orien_ranked,
        'tcga_df': tcga_filtered,
        'orien_df': orien_filtered
    }


def compute_k_m_relationship(tcga_ranked: list, orien_ranked: list, 
                              k_values: list, n_total: int) -> pd.DataFrame:
    """
    Compute consensus gene count (m) for each k value.
    
    Args:
        tcga_ranked: TCGA genes sorted by IG importance (descending)
        orien_ranked: ORIEN genes sorted by IG importance (descending)
        k_values: List of k values to analyze
        n_total: Total genes in pool (for enrichment calculation)
    
    Returns:
        DataFrame with columns: k, m, overlap_pct, expected_overlap, enrichment
    """
    results = []
    
    for k in k_values:
        # Handle k > available genes
        k_actual = min(k, len(tcga_ranked), len(orien_ranked))
        
        # Get top-k from each cohort
        tcga_top_k = set(tcga_ranked[:k_actual])
        orien_top_k = set(orien_ranked[:k_actual])
        
        # Compute intersection
        consensus = tcga_top_k & orien_top_k
        m = len(consensus)
        
        # Compute statistics
        overlap_pct = (m / k_actual) * 100 if k_actual > 0 else 0
        expected_overlap = (k_actual * k_actual) / n_total
        enrichment = m / expected_overlap if expected_overlap > 0 else 0
        
        results.append({
            'k': k,
            'k_actual': k_actual,
            'm': m,
            'overlap_pct': overlap_pct,
            'expected_overlap': expected_overlap,
            'enrichment': enrichment,
            'consensus_genes': sorted(list(consensus))
        })
        
        logger.info(f"k={k:3d}: m={m:3d} ({overlap_pct:5.1f}% overlap, {enrichment:.2f}x enrichment)")
    
    return pd.DataFrame(results)


def create_visualization(results_df: pd.DataFrame, output_dir: str):
    """Create visualization of k vs m relationship."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel A: k vs m (consensus gene count)
    ax1 = axes[0, 0]
    ax1.plot(results_df['k'], results_df['m'], 'o-', color='#2ca02c', 
             linewidth=2, markersize=8, label='Observed')
    ax1.plot(results_df['k'], results_df['expected_overlap'], '--', 
             color='gray', linewidth=1.5, label='Random expectation')
    ax1.fill_between(results_df['k'], 0, results_df['m'], alpha=0.2, color='#2ca02c')
    ax1.set_xlabel('k (top genes per cohort)', fontsize=12)
    ax1.set_ylabel('m (consensus genes)', fontsize=12)
    ax1.set_title('A. Consensus Gene Count vs k', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Add annotations for key points
    for _, row in results_df.iterrows():
        if row['k'] in [40, 70, 100, 141]:
            ax1.annotate(f"m={int(row['m'])}", 
                        (row['k'], row['m']),
                        textcoords="offset points",
                        xytext=(5, 5),
                        fontsize=9)
    
    # Panel B: Overlap percentage
    ax2 = axes[0, 1]
    ax2.bar(results_df['k'], results_df['overlap_pct'], color='#1f77b4', alpha=0.7, width=8)
    ax2.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% overlap')
    ax2.set_xlabel('k (top genes per cohort)', fontsize=12)
    ax2.set_ylabel('Overlap Percentage (%)', fontsize=12)
    ax2.set_title('B. Cross-Cohort Overlap Percentage', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Panel C: Enrichment over random
    ax3 = axes[1, 0]
    colors = ['#d62728' if e < 1.5 else '#ff7f0e' if e < 2.0 else '#2ca02c' 
              for e in results_df['enrichment']]
    ax3.bar(results_df['k'], results_df['enrichment'], color=colors, alpha=0.7, width=8)
    ax3.axhline(y=1.0, color='gray', linestyle='-', alpha=0.5, label='Random (1.0x)')
    ax3.axhline(y=2.0, color='green', linestyle='--', alpha=0.5, label='2x enrichment')
    ax3.set_xlabel('k (top genes per cohort)', fontsize=12)
    ax3.set_ylabel('Enrichment (vs random)', fontsize=12)
    ax3.set_title('C. Enrichment Over Random Expectation', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Panel D: Trade-off visualization (m vs overlap%)
    ax4 = axes[1, 1]
    scatter = ax4.scatter(results_df['m'], results_df['overlap_pct'], 
                          c=results_df['k'], cmap='viridis', s=100, alpha=0.8)
    ax4.set_xlabel('m (consensus genes)', fontsize=12)
    ax4.set_ylabel('Overlap Percentage (%)', fontsize=12)
    ax4.set_title('D. Trade-off: Gene Count vs Overlap Quality', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('k value', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # Add k labels to points
    for _, row in results_df.iterrows():
        ax4.annotate(f"k={int(row['k'])}", 
                    (row['m'], row['overlap_pct']),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(output_dir, 'k_m_relationship_figure.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Figure saved: {fig_path}")


def generate_recommendation(results_df: pd.DataFrame, config: dict) -> dict:
    """
    Generate k range recommendation based on analysis.
    
    Criteria:
    1. m >= min_consensus_genes (practical minimum)
    2. Enrichment > 1.5 (better than random)
    3. Target m in [target_m_min, target_m_max]
    """
    
    # Filter by minimum consensus genes
    valid_k = results_df[results_df['m'] >= config['min_consensus_genes']]
    
    if len(valid_k) == 0:
        logger.warning("No k values meet minimum consensus gene requirement!")
        return {'error': 'No valid k values found'}
    
    # Find k values in target m range
    target_k = valid_k[
        (valid_k['m'] >= config['target_m_min']) & 
        (valid_k['m'] <= config['target_m_max'])
    ]
    
    # Find optimal k (highest enrichment in target range)
    if len(target_k) > 0:
        optimal_idx = target_k['enrichment'].idxmax()
        optimal_k = target_k.loc[optimal_idx, 'k']
        optimal_m = target_k.loc[optimal_idx, 'm']
    else:
        # Fallback: use k with highest enrichment
        optimal_idx = valid_k['enrichment'].idxmax()
        optimal_k = valid_k.loc[optimal_idx, 'k']
        optimal_m = valid_k.loc[optimal_idx, 'm']
    
    # Determine recommended k range for tuning
    # Include k values where m >= 10 and enrichment >= 1.5
    tuning_candidates = valid_k[valid_k['enrichment'] >= 1.5]
    
    if len(tuning_candidates) > 0:
        k_min = int(tuning_candidates['k'].min())
        k_max = int(tuning_candidates['k'].max())
    else:
        k_min = int(valid_k['k'].min())
        k_max = int(valid_k['k'].max())
    
    recommendation = {
        'optimal_k_estimate': int(optimal_k),
        'optimal_m_estimate': int(optimal_m),
        'recommended_k_range': {
            'min': k_min,
            'max': k_max,
            'suggested_values': list(range(k_min, k_max + 1, 10))
        },
        'criteria': {
            'min_consensus_genes': config['min_consensus_genes'],
            'target_m_range': [config['target_m_min'], config['target_m_max']],
            'min_enrichment': 1.5
        },
        'rationale': (
            f"Based on pre-analysis of 141 sign-consistent genes:\n"
            f"- Optimal k estimate: {optimal_k} (yields m={optimal_m} consensus genes)\n"
            f"- Recommended tuning range: k={k_min} to k={k_max}\n"
            f"- All recommended k values have enrichment >= 1.5x over random"
        ),
        'timestamp': datetime.now().isoformat()
    }
    
    return recommendation


def main():
    parser = argparse.ArgumentParser(
        description='Pre-analysis: K vs M relationship for sign-consistent genes'
    )
    parser.add_argument('--sign_genes', type=str, 
                        default=CONFIG['sign_consistent_genes'],
                        help='Path to sign-consistent genes file')
    parser.add_argument('--tcga_ig', type=str,
                        default=CONFIG['tcga_ig_aggregated'],
                        help='Path to TCGA IG aggregated CSV')
    parser.add_argument('--orien_ig', type=str,
                        default=CONFIG['orien_ig_aggregated'],
                        help='Path to ORIEN IG aggregated CSV')
    parser.add_argument('--output_dir', type=str,
                        default=CONFIG['output_dir'],
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("PRE-ANALYSIS: K vs M RELATIONSHIP FOR SIGN-CONSISTENT GENES")
    logger.info("=" * 70)
    logger.info(f"Sign-consistent genes: {args.sign_genes}")
    logger.info(f"TCGA IG rankings: {args.tcga_ig}")
    logger.info(f"ORIEN IG rankings: {args.orien_ig}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 70)
    
    # ========================================================================
    # STEP 1: Load sign-consistent genes
    # ========================================================================
    logger.info("\n[STEP 1] Loading sign-consistent genes...")
    sign_genes = load_sign_consistent_genes(args.sign_genes)
    n_total = len(sign_genes)
    
    # ========================================================================
    # STEP 2: Load and filter IG rankings
    # ========================================================================
    logger.info("\n[STEP 2] Loading IG rankings and filtering to sign-consistent pool...")
    rankings = load_ig_rankings(args.tcga_ig, args.orien_ig, sign_genes)
    
    # ========================================================================
    # STEP 3: Compute k vs m relationship
    # ========================================================================
    logger.info("\n[STEP 3] Computing k vs m relationship...")
    logger.info("-" * 50)
    
    results_df = compute_k_m_relationship(
        tcga_ranked=rankings['tcga_ranked'],
        orien_ranked=rankings['orien_ranked'],
        k_values=CONFIG['k_values'],
        n_total=n_total
    )
    
    # ========================================================================
    # STEP 4: Save results table
    # ========================================================================
    logger.info("\n[STEP 4] Saving results...")
    
    # Save main table (without gene lists for readability)
    table_df = results_df[['k', 'k_actual', 'm', 'overlap_pct', 'expected_overlap', 'enrichment']]
    table_path = os.path.join(args.output_dir, 'k_m_relationship_table.csv')
    table_df.to_csv(table_path, index=False, float_format='%.4f')
    logger.info(f"Table saved: {table_path}")
    
    # Save full results with gene lists
    full_path = os.path.join(args.output_dir, 'k_m_relationship_full.json')
    results_dict = results_df.to_dict(orient='records')
    with open(full_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    logger.info(f"Full results saved: {full_path}")
    
    # ========================================================================
    # STEP 5: Create visualization
    # ========================================================================
    logger.info("\n[STEP 5] Creating visualization...")
    create_visualization(results_df, args.output_dir)
    
    # ========================================================================
    # STEP 6: Generate recommendation
    # ========================================================================
    logger.info("\n[STEP 6] Generating k range recommendation...")
    recommendation = generate_recommendation(results_df, CONFIG)
    
    rec_path = os.path.join(args.output_dir, 'k_range_recommendation.json')
    with open(rec_path, 'w') as f:
        json.dump(recommendation, f, indent=2)
    logger.info(f"Recommendation saved: {rec_path}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    
    logger.info("\nK vs M Relationship Table:")
    logger.info("-" * 60)
    logger.info(f"{'k':>5} | {'m':>5} | {'Overlap%':>10} | {'Expected':>10} | {'Enrichment':>10}")
    logger.info("-" * 60)
    for _, row in results_df.iterrows():
        logger.info(f"{int(row['k']):>5} | {int(row['m']):>5} | {row['overlap_pct']:>9.1f}% | "
                   f"{row['expected_overlap']:>10.1f} | {row['enrichment']:>9.2f}x")
    logger.info("-" * 60)
    
    logger.info("\n" + "=" * 70)
    logger.info("RECOMMENDATION")
    logger.info("=" * 70)
    logger.info(recommendation['rationale'])
    logger.info(f"\nSuggested k values for tuning: {recommendation['recommended_k_range']['suggested_values']}")
    logger.info("=" * 70)
    
    logger.info(f"\nOutput files saved to: {args.output_dir}")
    logger.info("\nNEXT STEP: Use recommended k range for hyperparameter tuning")
    
    return results_df, recommendation


if __name__ == "__main__":
    results_df, recommendation = main()
