#!/usr/bin/env python3
"""
Step 2: Compare Results and Generate Figures

Purpose: Create comparison table and visualization for gene set validation

Output:
    results_v2/07_sign_filter_validation/comparison/
    ├── performance_comparison.csv
    ├── comparison_figure.png
    └── VALIDATION_SUMMARY.json

Author: [Your Name]
Date: 2024-12
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    'output_dir': 'results_v2/07_sign_filter_validation',
}


def load_summaries():
    """Load summary.json from both gene sets."""
    
    summaries = {}
    
    set_a_file = os.path.join(CONFIG['output_dir'], 'set_A_68genes', 'summary.json')
    set_b_file = os.path.join(CONFIG['output_dir'], 'set_B_26genes', 'summary.json')
    
    if os.path.exists(set_a_file):
        with open(set_a_file, 'r') as f:
            summaries['set_A'] = json.load(f)
    else:
        raise FileNotFoundError(f"Set A summary not found: {set_a_file}")
    
    if os.path.exists(set_b_file):
        with open(set_b_file, 'r') as f:
            summaries['set_B'] = json.load(f)
    else:
        raise FileNotFoundError(f"Set B summary not found: {set_b_file}")
    
    return summaries


def compute_statistics(set_a, set_b):
    """Compute statistical comparison between gene sets."""
    
    results = {}
    
    for direction in ['tcga_to_orien', 'orien_to_tcga']:
        a_vals = set_a[direction]['all_cindices']
        b_vals = set_b[direction]['all_cindices']
        
        # Paired t-test (same seeds used)
        t_stat, p_value = stats.ttest_rel(a_vals, b_vals)
        
        # Effect size (Cohen's d for paired samples)
        diff = np.array(a_vals) - np.array(b_vals)
        cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff) > 0 else 0
        
        results[direction] = {
            'set_a_mean': np.mean(a_vals),
            'set_a_std': np.std(a_vals),
            'set_b_mean': np.mean(b_vals),
            'set_b_std': np.std(b_vals),
            'difference': np.mean(b_vals) - np.mean(a_vals),
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant': p_value < 0.05
        }
    
    # Overall comparison
    a_all = set_a['tcga_to_orien']['all_cindices'] + set_a['orien_to_tcga']['all_cindices']
    b_all = set_b['tcga_to_orien']['all_cindices'] + set_b['orien_to_tcga']['all_cindices']
    
    t_stat, p_value = stats.ttest_rel(a_all, b_all)
    diff = np.array(a_all) - np.array(b_all)
    cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff) > 0 else 0
    
    results['overall'] = {
        'set_a_mean': np.mean(a_all),
        'set_a_std': np.std(a_all),
        'set_b_mean': np.mean(b_all),
        'set_b_std': np.std(b_all),
        'difference': np.mean(b_all) - np.mean(a_all),
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant': p_value < 0.05
    }
    
    return results


def create_comparison_table(summaries, stats_results):
    """Create comparison table as DataFrame."""
    
    rows = []
    
    for direction, label in [
        ('tcga_to_orien', 'TCGA → ORIEN'),
        ('orien_to_tcga', 'ORIEN → TCGA'),
        ('overall', 'Overall')
    ]:
        s = stats_results[direction]
        rows.append({
            'Direction': label,
            'Set A (68 genes) Mean': f"{s['set_a_mean']:.4f}",
            'Set A (68 genes) Std': f"{s['set_a_std']:.4f}",
            'Set B (26 genes) Mean': f"{s['set_b_mean']:.4f}",
            'Set B (26 genes) Std': f"{s['set_b_std']:.4f}",
            'Difference (B - A)': f"{s['difference']:+.4f}",
            'p-value': f"{s['p_value']:.4f}" if s['p_value'] >= 0.001 else "<0.001",
            "Cohen's d": f"{s['cohens_d']:.3f}",
            'Significant': 'Yes' if s['significant'] else 'No'
        })
    
    return pd.DataFrame(rows)


def create_figure(summaries, stats_results, output_file):
    """Create comparison visualization."""
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    # Colors
    color_a = '#3498db'  # Blue
    color_b = '#e74c3c'  # Red
    
    # Subplot 1: Bar chart comparison
    ax1 = axes[0]
    directions = ['TCGA→ORIEN', 'ORIEN→TCGA', 'Overall']
    x = np.arange(len(directions))
    width = 0.35
    
    means_a = [stats_results['tcga_to_orien']['set_a_mean'],
               stats_results['orien_to_tcga']['set_a_mean'],
               stats_results['overall']['set_a_mean']]
    stds_a = [stats_results['tcga_to_orien']['set_a_std'],
              stats_results['orien_to_tcga']['set_a_std'],
              stats_results['overall']['set_a_std']]
    
    means_b = [stats_results['tcga_to_orien']['set_b_mean'],
               stats_results['orien_to_tcga']['set_b_mean'],
               stats_results['overall']['set_b_mean']]
    stds_b = [stats_results['tcga_to_orien']['set_b_std'],
              stats_results['orien_to_tcga']['set_b_std'],
              stats_results['overall']['set_b_std']]
    
    bars_a = ax1.bar(x - width/2, means_a, width, yerr=stds_a, 
                     label='Set A: 68 genes', color=color_a, capsize=5, alpha=0.8)
    bars_b = ax1.bar(x + width/2, means_b, width, yerr=stds_b,
                     label='Set B: 26 genes', color=color_b, capsize=5, alpha=0.8)
    
    ax1.set_ylabel('C-index', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(directions)
    ax1.legend(loc='lower right')
    ax1.set_ylim(0.5, 0.75)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax1.set_title('Cross-Cohort C-index Comparison', fontsize=12, fontweight='bold')
    
    # Add significance markers
    for i, direction in enumerate(['tcga_to_orien', 'orien_to_tcga', 'overall']):
        if stats_results[direction]['significant']:
            y_max = max(means_a[i] + stds_a[i], means_b[i] + stds_b[i])
            ax1.annotate('*', xy=(i, y_max + 0.01), ha='center', fontsize=14)
    
    # Subplot 2: Individual seed results
    ax2 = axes[1]
    
    seeds = [42, 123, 456, 789, 1011]
    
    a_tcga_orien = summaries['set_A']['tcga_to_orien']['all_cindices']
    b_tcga_orien = summaries['set_B']['tcga_to_orien']['all_cindices']
    
    ax2.plot(seeds, a_tcga_orien, 'o-', color=color_a, label='Set A', markersize=8)
    ax2.plot(seeds, b_tcga_orien, 's-', color=color_b, label='Set B', markersize=8)
    
    ax2.set_xlabel('Random Seed', fontsize=12)
    ax2.set_ylabel('C-index', fontsize=12)
    ax2.set_title('TCGA→ORIEN by Seed', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.set_ylim(0.5, 0.75)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Subplot 3: Difference plot
    ax3 = axes[2]
    
    diff_tcga_orien = np.array(b_tcga_orien) - np.array(a_tcga_orien)
    a_orien_tcga = summaries['set_A']['orien_to_tcga']['all_cindices']
    b_orien_tcga = summaries['set_B']['orien_to_tcga']['all_cindices']
    diff_orien_tcga = np.array(b_orien_tcga) - np.array(a_orien_tcga)
    
    ax3.bar(np.arange(5) - 0.2, diff_tcga_orien, 0.4, label='TCGA→ORIEN', color=color_a, alpha=0.8)
    ax3.bar(np.arange(5) + 0.2, diff_orien_tcga, 0.4, label='ORIEN→TCGA', color=color_b, alpha=0.8)
    
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_xlabel('Seed Index', fontsize=12)
    ax3.set_ylabel('Difference (Set B - Set A)', fontsize=12)
    ax3.set_title('Performance Difference by Seed', fontsize=12, fontweight='bold')
    ax3.set_xticks(np.arange(5))
    ax3.set_xticklabels(seeds)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Figure saved to: {output_file}")


def main():
    print("=" * 70)
    print("Step 2: Compare Results and Generate Figures")
    print("=" * 70)
    
    # Create output directory
    comparison_dir = os.path.join(CONFIG['output_dir'], 'comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Load summaries
    print("\n[1] Loading validation summaries...")
    summaries = load_summaries()
    print(f"    Set A: {summaries['set_A']['n_genes']} genes")
    print(f"    Set B: {summaries['set_B']['n_genes']} genes")
    
    # Compute statistics
    print("\n[2] Computing statistical comparison...")
    stats_results = compute_statistics(summaries['set_A'], summaries['set_B'])
    
    # Create comparison table
    print("\n[3] Creating comparison table...")
    table_df = create_comparison_table(summaries, stats_results)
    table_file = os.path.join(comparison_dir, 'performance_comparison.csv')
    table_df.to_csv(table_file, index=False)
    print(f"    Saved: {table_file}")
    
    print("\n" + table_df.to_string(index=False))
    
    # Create figure
    print("\n[4] Creating comparison figure...")
    figure_file = os.path.join(comparison_dir, 'comparison_figure.png')
    create_figure(summaries, stats_results, figure_file)
    
    # Create final summary
    print("\n[5] Creating validation summary...")
    
    final_summary = {
        'created_at': datetime.now().isoformat(),
        'comparison': {
            'set_A': {
                'name': 'magnitude_only',
                'n_genes': summaries['set_A']['n_genes'],
                'overall_cindex': stats_results['overall']['set_a_mean'],
                'overall_std': stats_results['overall']['set_a_std']
            },
            'set_B': {
                'name': 'sign_consistent',
                'n_genes': summaries['set_B']['n_genes'],
                'overall_cindex': stats_results['overall']['set_b_mean'],
                'overall_std': stats_results['overall']['set_b_std']
            }
        },
        'statistics': stats_results,
        'recommendation': None
    }
    
    # Generate recommendation
    diff = stats_results['overall']['difference']
    p_val = stats_results['overall']['p_value']
    
    if p_val >= 0.05:
        if abs(diff) < 0.01:
            recommendation = "USE SET B: Performance equivalent, but Set B has better interpretability (all genes have consistent effect direction across cohorts)."
        elif diff > 0:
            recommendation = "USE SET B: Slightly better performance (not significant) with much better interpretability."
        else:
            recommendation = "CONSIDER SET B: Slightly lower performance (not significant) but much better interpretability. Trade-off is worthwhile."
    else:  # Significant difference
        if diff > 0:
            recommendation = "USE SET B: Significantly better performance AND better interpretability."
        else:
            recommendation = f"TRADE-OFF REQUIRED: Set A significantly better by {-diff:.4f}. Consider reporting both in dissertation."
    
    final_summary['recommendation'] = recommendation
    
    summary_file = os.path.join(comparison_dir, 'VALIDATION_SUMMARY.json')
    with open(summary_file, 'w') as f:
        json.dump(final_summary, f, indent=2)
    print(f"    Saved: {summary_file}")
    
    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"""
Gene Set Comparison Results:
----------------------------
Set A (68 genes, magnitude-only):    {stats_results['overall']['set_a_mean']:.4f} ± {stats_results['overall']['set_a_std']:.4f}
Set B (26 genes, sign-consistent):   {stats_results['overall']['set_b_mean']:.4f} ± {stats_results['overall']['set_b_std']:.4f}

Difference (B - A): {diff:+.4f}
p-value: {p_val:.4f}
Cohen's d: {stats_results['overall']['cohens_d']:.3f}
Significant: {'Yes' if stats_results['overall']['significant'] else 'No'}

RECOMMENDATION:
{recommendation}

Output files:
- {table_file}
- {figure_file}
- {summary_file}
""")


if __name__ == '__main__':
    main()
