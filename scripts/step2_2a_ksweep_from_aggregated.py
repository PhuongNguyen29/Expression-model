#!/usr/bin/env python3
"""
Script: step2_2a_ksweep_from_aggregated.py
Purpose: K-sweep analysis using aggregated importance scores from Step 2.1
Status: ACTIVE (Step 2.2A - Quick k-value identification)
Author: Claude (for Phuong's dissertation)
Created: 2024-11-17

This script:
1. Reads aggregated_gene_importances.csv from Step 2.1
2. For each k value, extracts top-k genes from TCGA and ORIEN
3. Computes consensus genes (intersection of top-k lists)
4. Generates summary table and visualization
5. Recommends optimal k values for Step 2.2B validation

Methodology:
- Consensus = Intersection of TCGA top-k and ORIEN top-k genes
- Target: ~20-30 consensus genes (matching Chapter 2 scale)
- Stability metric: Jaccard index between TCGA and ORIEN rankings

Reference:
- Ein-Dor et al. (2005) Nature Genetics: Consensus gene selection
- Your Chapter 3 methodology: k-sweep for stable gene count

Usage:
    python step2_2a_ksweep_from_aggregated.py \
        --importance_file results_v2/02_biomarker_discovery/aggregated_gene_importances.csv \
        --output_dir results_v2/02_biomarker_discovery/ksweep_analysis \
        --k_values 20 30 40 50 60 70 80 90 100 120 150 200
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Set, Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


def load_importance_scores(filepath: Path) -> pd.DataFrame:
    """
    Load aggregated importance scores from Step 2.1.
    
    Expected columns:
    - gene_name
    - tcga_importance_mean
    - orien_importance_mean
    - overall_mean
    """
    df = pd.read_csv(filepath)
    
    required_cols = ['gene_name', 'tcga_importance_mean', 'orien_importance_mean']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    print(f"✓ Loaded {len(df)} genes from {filepath}")
    return df


def get_top_k_genes(df: pd.DataFrame, cohort: str, k: int) -> List[str]:
    """
    Get top k genes for a specific cohort.
    
    Args:
        df: DataFrame with importance scores
        cohort: 'tcga' or 'orien'
        k: Number of top genes to extract
        
    Returns:
        List of top k gene names
    """
    col = f'{cohort}_importance_mean'
    top_k = df.nlargest(k, col)['gene_name'].tolist()
    return top_k


def compute_consensus_metrics(
    tcga_genes: List[str],
    orien_genes: List[str]
) -> Dict:
    """
    Compute consensus and overlap metrics between TCGA and ORIEN gene lists.
    
    Returns:
        Dictionary with consensus genes and metrics
    """
    tcga_set = set(tcga_genes)
    orien_set = set(orien_genes)
    
    # Consensus = intersection
    consensus = tcga_set & orien_set
    
    # Union for Jaccard
    union = tcga_set | orien_set
    
    # Metrics
    jaccard = len(consensus) / len(union) if union else 0
    overlap_pct = 100 * len(consensus) / min(len(tcga_set), len(orien_set)) if tcga_set and orien_set else 0
    
    return {
        'consensus_genes': sorted(list(consensus)),
        'n_consensus': len(consensus),
        'n_tcga': len(tcga_set),
        'n_orien': len(orien_set),
        'jaccard_index': jaccard,
        'overlap_pct': overlap_pct
    }


def run_ksweep(
    importance_df: pd.DataFrame,
    k_values: List[int],
    output_dir: Path,
    cox_genes_file: str = 'data/raw/cox_consensus_genes_20.txt'
) -> pd.DataFrame:
    """
    Run k-sweep analysis across multiple k values.
    
    Args:
        importance_df: DataFrame with aggregated importance scores
        k_values: List of k values to test
        output_dir: Output directory for results
        cox_genes_file: Optional Chapter 2 Cox genes for comparison
        
    Returns:
        Summary DataFrame with results for each k
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Cox genes if available
    cox_genes = []
    try:
        with open(cox_genes_file, 'r') as f:
            cox_genes = [line.strip() for line in f if line.strip()]
        print(f"✓ Loaded {len(cox_genes)} Cox genes for comparison")
    except FileNotFoundError:
        print(f"  ⚠️  Cox genes not found: {cox_genes_file}")
        print(f"     Chapter 2 comparison will be skipped")
    
    print(f"\n{'='*80}")
    print(f"RUNNING K-SWEEP ANALYSIS")
    print(f"{'='*80}\n")
    print(f"K values to test: {k_values}")
    print(f"Total genes available: {len(importance_df)}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Store results
    results = []
    gene_lists_dir = output_dir / 'gene_lists'
    gene_lists_dir.mkdir(exist_ok=True)
    
    # Run k-sweep
    for k in k_values:
        print(f"Processing k={k}...")
        
        # Extract top k genes for each cohort
        tcga_top_k = get_top_k_genes(importance_df, 'tcga', k)
        orien_top_k = get_top_k_genes(importance_df, 'orien', k)
        
        # Compute consensus
        metrics = compute_consensus_metrics(tcga_top_k, orien_top_k)
        
        # Cox overlap if available
        cox_overlap = 0
        cox_overlap_pct = 0.0
        if cox_genes:
            consensus_set = set(metrics['consensus_genes'])
            cox_set = set(cox_genes)
            cox_overlap = len(consensus_set & cox_set)
            cox_overlap_pct = 100 * cox_overlap / len(consensus_set) if consensus_set else 0
        
        # Store results
        result = {
            'k': k,
            'n_tcga': len(tcga_top_k),
            'n_orien': len(orien_top_k),
            'n_consensus': metrics['n_consensus'],
            'overlap_pct': metrics['overlap_pct'],
            'jaccard_index': metrics['jaccard_index'],
            'cox_overlap': cox_overlap,
            'cox_overlap_pct': cox_overlap_pct
        }
        results.append(result)
        
        # Save gene lists
        with open(gene_lists_dir / f'k{k:03d}_tcga_top.txt', 'w') as f:
            f.write('\n'.join(tcga_top_k))
        
        with open(gene_lists_dir / f'k{k:03d}_orien_top.txt', 'w') as f:
            f.write('\n'.join(orien_top_k))
        
        with open(gene_lists_dir / f'k{k:03d}_consensus.txt', 'w') as f:
            f.write('\n'.join(metrics['consensus_genes']))
        
        print(f"  ✓ k={k}: {metrics['n_consensus']} consensus genes, {metrics['overlap_pct']:.1f}% overlap")
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(output_dir / 'ksweep_summary.csv', index=False)
    print(f"\n✓ Summary saved: ksweep_summary.csv")
    
    # Save full results as JSON
    full_results = {
        'timestamp': datetime.now().isoformat(),
        'method': 'k-sweep from aggregated importance scores',
        'n_total_genes': len(importance_df),
        'k_values': k_values,
        'results': results
    }
    
    with open(output_dir / 'ksweep_full_results.json', 'w') as f:
        json.dump(full_results, f, indent=2)
    
    print(f"✓ Full results: ksweep_full_results.json")
    
    # Generate visualization
    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*80}\n")
    
    generate_visualizations(summary_df, output_dir, bool(cox_genes))
    
    # Generate recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}\n")
    
    generate_recommendations(summary_df, output_dir)
    
    return summary_df


def generate_visualizations(summary_df: pd.DataFrame, output_dir: Path, has_cox: bool):
    """Generate visualization plots for k-sweep results."""
    
    # Determine layout
    if has_cox:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
    
    k_vals = summary_df['k'].values
    
    # Plot 1: Number of consensus genes vs k
    ax1 = axes[0]
    ax1.plot(k_vals, summary_df['n_consensus'], 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax1.axhline(y=20, color='gray', linestyle='--', alpha=0.5, label='Chapter 2 (n=20)')
    ax1.axhline(y=30, color='gray', linestyle=':', alpha=0.5, label='Target max (n=30)')
    ax1.set_xlabel('Top k genes extracted', fontsize=11)
    ax1.set_ylabel('Number of consensus genes', fontsize=11)
    ax1.set_title('Consensus Gene Count vs k', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Overlap percentage vs k
    ax2 = axes[1]
    ax2.plot(k_vals, summary_df['overlap_pct'], 'o-', linewidth=2, markersize=8, color='#A23B72')
    ax2.set_xlabel('Top k genes extracted', fontsize=11)
    ax2.set_ylabel('Overlap percentage (%)', fontsize=11)
    ax2.set_title('TCGA-ORIEN Overlap vs k', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Jaccard index vs k
    ax3 = axes[2] if has_cox else axes[2]
    ax3.plot(k_vals, summary_df['jaccard_index'], 'o-', linewidth=2, markersize=8, color='#F18F01')
    ax3.set_xlabel('Top k genes extracted', fontsize=11)
    ax3.set_ylabel('Jaccard index', fontsize=11)
    ax3.set_title('Set Similarity: Jaccard Index vs k', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Comparison bar chart
    ax4 = axes[3]
    x = np.arange(len(k_vals))
    width = 0.35
    
    ax4.bar(x - width/2, summary_df['n_tcga'], width, label='TCGA top-k', alpha=0.8, color='#6A994E')
    ax4.bar(x + width/2, summary_df['n_orien'], width, label='ORIEN top-k', alpha=0.8, color='#BC4749')
    ax4.plot(x, summary_df['n_consensus'], 'ko-', linewidth=2, markersize=6, label='Consensus', zorder=5)
    
    ax4.set_xlabel('k value', fontsize=11)
    ax4.set_ylabel('Number of genes', fontsize=11)
    ax4.set_title('Cohort-Specific vs Consensus Genes', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(k_vals, rotation=45)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5 & 6: Cox overlap (if available)
    if has_cox:
        ax5 = axes[4]
        ax5.plot(k_vals, summary_df['cox_overlap'], 'o-', linewidth=2, markersize=8, color='#C73E1D')
        ax5.axhline(y=20, color='gray', linestyle='--', alpha=0.5, label='Chapter 2 total (n=20)')
        ax5.set_xlabel('Top k genes extracted', fontsize=11)
        ax5.set_ylabel('Number of shared genes', fontsize=11)
        ax5.set_title('Chapter 2 Cox Overlap: Count', fontsize=12, fontweight='bold')
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3)
        
        ax6 = axes[5]
        ax6.plot(k_vals, summary_df['cox_overlap_pct'], 'o-', linewidth=2, markersize=8, color='#540B0E')
        ax6.set_xlabel('Top k genes extracted', fontsize=11)
        ax6.set_ylabel('Cox overlap (%)', fontsize=11)
        ax6.set_title('Chapter 2 Cox Overlap: Percentage', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ksweep_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved: ksweep_analysis.png")
    print(f"  {'(6 panels with Cox comparison)' if has_cox else '(4 panels)'}")


def generate_recommendations(summary_df: pd.DataFrame, output_dir: Path):
    """Generate k-value recommendations based on analysis."""
    
    target_range = (20, 30)
    
    print(f"Goal 1: Find k values that yield ~{target_range[0]}-{target_range[1]} consensus genes")
    print(f"Goal 2: Find k where performance would plateau (high stability)\n")
    
    # Find candidates in target range
    candidates = summary_df[
        (summary_df['n_consensus'] >= target_range[0]) & 
        (summary_df['n_consensus'] <= target_range[1])
    ].copy()
    
    if len(candidates) > 0:
        print(f"✓ Found {len(candidates)} k values in target range:\n")
        
        for _, row in candidates.iterrows():
            print(f"  k={int(row['k']):3d}: {int(row['n_consensus']):2d} consensus genes, "
                  f"{row['overlap_pct']:5.1f}% overlap, Jaccard={row['jaccard_index']:.3f}")
        
        # Recommend highest stability
        best = candidates.loc[candidates['overlap_pct'].idxmax()]
        
        print(f"\n🎯 PRIMARY RECOMMENDATION: k={int(best['k'])}")
        print(f"   - Consensus genes: {int(best['n_consensus'])}")
        print(f"   - Overlap: {best['overlap_pct']:.1f}%")
        print(f"   - Jaccard index: {best['jaccard_index']:.3f}")
        print(f"   - Rationale: Highest stability within target range")
        
        # Secondary recommendations
        print(f"\n📋 ADDITIONAL CANDIDATES for Step 2.2B validation:")
        
        # Get top 3 by different criteria
        top_by_jaccard = candidates.nlargest(3, 'jaccard_index')['k'].tolist()
        top_by_overlap = candidates.nlargest(3, 'overlap_pct')['k'].tolist()
        
        recommended_k = sorted(list(set(top_by_jaccard + top_by_overlap)))[:4]
        
        print(f"   Recommended k values for validation: {recommended_k}")
        print(f"   (These will be tested in Step 2.2B with actual model training)")
        
        # Save recommendations
        recommendations = {
            'timestamp': datetime.now().isoformat(),
            'primary_recommendation': {
                'k': int(best['k']),
                'n_consensus': int(best['n_consensus']),
                'overlap_pct': float(best['overlap_pct']),
                'jaccard_index': float(best['jaccard_index'])
            },
            'validation_candidates': recommended_k,
            'rationale': 'Selected based on target consensus count (20-30) and maximum stability'
        }
        
        with open(output_dir / 'RECOMMENDATIONS.json', 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        print(f"\n✓ Recommendations saved: RECOMMENDATIONS.json")
        
    else:
        print(f"⚠️  No k value yielded {target_range[0]}-{target_range[1]} consensus genes")
        print(f"   Consider adjusting k range or target")
        
        # Show closest
        closest = summary_df.iloc[(summary_df['n_consensus'] - 25).abs().argsort()[:3]]
        print(f"\n   Closest options:")
        for _, row in closest.iterrows():
            print(f"     k={int(row['k']):3d}: {int(row['n_consensus']):2d} consensus genes")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="K-sweep analysis from aggregated importance scores (Step 2.2A)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python step2_2a_ksweep_from_aggregated.py \\
      --importance_file results_v2/02_biomarker_discovery/aggregated_gene_importances.csv \\
      --output_dir results_v2/02_biomarker_discovery/ksweep_analysis \\
      --k_values 20 30 40 50 60 70 80 90 100 120 150 200
        """
    )
    
    parser.add_argument('--importance_file', type=str, required=True,
                       help='Path to aggregated_gene_importances.csv from Step 2.1')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for k-sweep results')
    parser.add_argument('--k_values', type=int, nargs='+',
                       default=[20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200],
                       help='K values to test (default: 20 to 200)')
    parser.add_argument('--cox_genes', type=str,
                       default='data/raw/cox_consensus_genes_20.txt',
                       help='Optional Cox genes file for Chapter 2 comparison')
    
    args = parser.parse_args()
    
    # Load importance scores
    importance_df = load_importance_scores(args.importance_file)
    
    # Run k-sweep
    summary_df = run_ksweep(
        importance_df=importance_df,
        k_values=args.k_values,
        output_dir=args.output_dir,
        cox_genes_file=args.cox_genes
    )
    
    print(f"\n{'='*80}")
    print("✅ STEP 2.2A COMPLETE!")
    print(f"{'='*80}")
    print(f"\n📁 Results saved in: {args.output_dir}/")
    print(f"\n📋 Next step: Review RECOMMENDATIONS.json and proceed to Step 2.2B")
    print(f"   Step 2.2B will validate the recommended k values by retraining models")
    print(f"{'='*80}\n")
