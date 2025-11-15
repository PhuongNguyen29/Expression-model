#!/usr/bin/env python3
"""
Script: aggregate_multiseed_results.py
Purpose: Aggregate and analyze transfer learning results across multiple seeds
Author: Phuong
Created: 2024-11-15

This script:
1. Loads results from all seeds
2. Computes mean ± std for all metrics
3. Performs statistical tests
4. Generates publication-ready tables
5. Creates visualization plots

Usage:
    python scripts/aggregate_multiseed_results.py \
        --results_dir results/transfer_learning_multiseed_20241115_001933
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_seed_results(results_dir: Path) -> List[Dict]:
    """Load results from all seeds."""
    results = []
    
    seed_files = sorted(results_dir.glob('seed*_results.json'))
    
    if not seed_files:
        raise FileNotFoundError(f"No seed result files found in {results_dir}")
    
    print(f"Found {len(seed_files)} seed results:")
    for seed_file in seed_files:
        with open(seed_file, 'r') as f:
            result = json.load(f)
            results.append(result)
            print(f"  - {seed_file.name}: seed={result['seed']}")
    
    return results


def compute_statistics(results: List[Dict]) -> Dict:
    """Compute mean, std, and statistical tests across seeds."""
    
    # Extract metrics
    seeds = [r['seed'] for r in results]
    n_seeds = len(results)
    
    # Baseline metrics
    baseline_tcga_on_orien = [r['baseline']['tcga_on_orien'] for r in results]
    baseline_orien_on_tcga = [r['baseline']['orien_on_tcga'] for r in results]
    baseline_avg = [r['baseline']['average'] for r in results]
    
    # Transfer metrics
    transfer_tcga_on_orien = [r['transfer']['tcga_on_orien'] for r in results]
    transfer_orien_on_tcga = [r['transfer']['orien_on_tcga'] for r in results]
    transfer_avg = [r['transfer']['average'] for r in results]
    
    # Compute improvements
    improvement_tcga = np.array(transfer_tcga_on_orien) - np.array(baseline_tcga_on_orien)
    improvement_orien = np.array(transfer_orien_on_tcga) - np.array(baseline_orien_on_tcga)
    improvement_avg = np.array(transfer_avg) - np.array(baseline_avg)
    
    # Paired t-tests
    t_tcga, p_tcga = stats.ttest_rel(transfer_tcga_on_orien, baseline_tcga_on_orien)
    t_orien, p_orien = stats.ttest_rel(transfer_orien_on_tcga, baseline_orien_on_tcga)
    t_avg, p_avg = stats.ttest_rel(transfer_avg, baseline_avg)
    
    # Cohen's d effect size
    def cohens_d(x1, x2):
        return (np.mean(x1) - np.mean(x2)) / np.sqrt((np.std(x1)**2 + np.std(x2)**2) / 2)
    
    d_tcga = cohens_d(transfer_tcga_on_orien, baseline_tcga_on_orien)
    d_orien = cohens_d(transfer_orien_on_tcga, baseline_orien_on_tcga)
    d_avg = cohens_d(transfer_avg, baseline_avg)
    
    stats_summary = {
        'n_seeds': n_seeds,
        'seeds': seeds,
        'baseline': {
            'tcga_on_orien': {
                'mean': np.mean(baseline_tcga_on_orien),
                'std': np.std(baseline_tcga_on_orien, ddof=1),
                'values': baseline_tcga_on_orien
            },
            'orien_on_tcga': {
                'mean': np.mean(baseline_orien_on_tcga),
                'std': np.std(baseline_orien_on_tcga, ddof=1),
                'values': baseline_orien_on_tcga
            },
            'average': {
                'mean': np.mean(baseline_avg),
                'std': np.std(baseline_avg, ddof=1),
                'values': baseline_avg
            }
        },
        'transfer': {
            'tcga_on_orien': {
                'mean': np.mean(transfer_tcga_on_orien),
                'std': np.std(transfer_tcga_on_orien, ddof=1),
                'values': transfer_tcga_on_orien
            },
            'orien_on_tcga': {
                'mean': np.mean(transfer_orien_on_tcga),
                'std': np.std(transfer_orien_on_tcga, ddof=1),
                'values': transfer_orien_on_tcga
            },
            'average': {
                'mean': np.mean(transfer_avg),
                'std': np.std(transfer_avg, ddof=1),
                'values': transfer_avg
            }
        },
        'improvement': {
            'tcga_on_orien': {
                'mean': np.mean(improvement_tcga),
                'std': np.std(improvement_tcga, ddof=1),
                'percent': (np.mean(improvement_tcga) / np.mean(baseline_tcga_on_orien)) * 100
            },
            'orien_on_tcga': {
                'mean': np.mean(improvement_orien),
                'std': np.std(improvement_orien, ddof=1),
                'percent': (np.mean(improvement_orien) / np.mean(baseline_orien_on_tcga)) * 100
            },
            'average': {
                'mean': np.mean(improvement_avg),
                'std': np.std(improvement_avg, ddof=1),
                'percent': (np.mean(improvement_avg) / np.mean(baseline_avg)) * 100
            }
        },
        'statistical_tests': {
            'tcga_on_orien': {
                't_statistic': t_tcga,
                'p_value': p_tcga,
                'cohens_d': d_tcga,
                'significant': p_tcga < 0.05
            },
            'orien_on_tcga': {
                't_statistic': t_orien,
                'p_value': p_orien,
                'cohens_d': d_orien,
                'significant': p_orien < 0.05
            },
            'average': {
                't_statistic': t_avg,
                'p_value': p_avg,
                'cohens_d': d_avg,
                'significant': p_avg < 0.05
            }
        }
    }
    
    return stats_summary


def create_summary_table(stats: Dict) -> pd.DataFrame:
    """Create publication-ready summary table."""
    
    def format_metric(mean, std):
        return f"{mean:.4f} ± {std:.4f}"
    
    def sig_stars(p):
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return "ns"
    
    table_data = {
        'Direction': [
            'TCGA→ORIEN',
            'ORIEN→TCGA',
            'Bidirectional Average'
        ],
        'Baseline (from scratch)': [
            format_metric(stats['baseline']['tcga_on_orien']['mean'], 
                         stats['baseline']['tcga_on_orien']['std']),
            format_metric(stats['baseline']['orien_on_tcga']['mean'], 
                         stats['baseline']['orien_on_tcga']['std']),
            format_metric(stats['baseline']['average']['mean'], 
                         stats['baseline']['average']['std'])
        ],
        'Transfer Learning': [
            format_metric(stats['transfer']['tcga_on_orien']['mean'], 
                         stats['transfer']['tcga_on_orien']['std']),
            format_metric(stats['transfer']['orien_on_tcga']['mean'], 
                         stats['transfer']['orien_on_tcga']['std']),
            format_metric(stats['transfer']['average']['mean'], 
                         stats['transfer']['average']['std'])
        ],
        'Improvement': [
            f"+{stats['improvement']['tcga_on_orien']['mean']:.4f} (+{stats['improvement']['tcga_on_orien']['percent']:.1f}%)",
            f"+{stats['improvement']['orien_on_tcga']['mean']:.4f} (+{stats['improvement']['orien_on_tcga']['percent']:.1f}%)",
            f"+{stats['improvement']['average']['mean']:.4f} (+{stats['improvement']['average']['percent']:.1f}%)"
        ],
        'p-value': [
            f"{stats['statistical_tests']['tcga_on_orien']['p_value']:.4f}{sig_stars(stats['statistical_tests']['tcga_on_orien']['p_value'])}",
            f"{stats['statistical_tests']['orien_on_tcga']['p_value']:.4f}{sig_stars(stats['statistical_tests']['orien_on_tcga']['p_value'])}",
            f"{stats['statistical_tests']['average']['p_value']:.4f}{sig_stars(stats['statistical_tests']['average']['p_value'])}"
        ],
        "Cohen's d": [
            f"{stats['statistical_tests']['tcga_on_orien']['cohens_d']:.2f}",
            f"{stats['statistical_tests']['orien_on_tcga']['cohens_d']:.2f}",
            f"{stats['statistical_tests']['average']['cohens_d']:.2f}"
        ]
    }
    
    df = pd.DataFrame(table_data)
    return df


def print_results(stats: Dict, table: pd.DataFrame):
    """Print formatted results to console."""
    
    print(f"\n{'='*80}")
    print("MULTI-SEED TRANSFER LEARNING RESULTS")
    print(f"{'='*80}\n")
    
    print(f"Number of seeds: {stats['n_seeds']}")
    print(f"Seeds: {stats['seeds']}\n")
    
    print("Table: Transfer Learning Performance (Mean ± SD across seeds)\n")
    print(table.to_string(index=False))
    
    print(f"\n{'='*80}")
    print("STATISTICAL SIGNIFICANCE")
    print(f"{'='*80}\n")
    
    for direction in ['tcga_on_orien', 'orien_on_tcga', 'average']:
        test = stats['statistical_tests'][direction]
        print(f"{direction.replace('_', ' ').upper()}:")
        print(f"  t-statistic: {test['t_statistic']:.4f}")
        print(f"  p-value: {test['p_value']:.6f} {'(significant)' if test['significant'] else '(not significant)'}")
        print(f"  Cohen's d: {test['cohens_d']:.2f} ({'large' if abs(test['cohens_d']) > 0.8 else 'medium' if abs(test['cohens_d']) > 0.5 else 'small'} effect)")
        print()
    
    print(f"{'='*80}")
    print("INTERPRETATION")
    print(f"{'='*80}\n")
    
    avg_improvement = stats['improvement']['average']['percent']
    p_value = stats['statistical_tests']['average']['p_value']
    
    if p_value < 0.05 and avg_improvement > 0:
        print(f"✓ Transfer learning shows STATISTICALLY SIGNIFICANT improvement")
        print(f"  Average improvement: +{avg_improvement:.1f}% (p={p_value:.4f})")
        print(f"  This is a {'large' if avg_improvement > 25 else 'moderate' if avg_improvement > 10 else 'small'} practical improvement.")
    else:
        print(f"⚠️  Results are not statistically significant or show no improvement")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate multi-seed transfer learning results"
    )
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing seed*_results.json files')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    # Load results
    print(f"\nLoading results from: {results_dir}")
    results = load_seed_results(results_dir)
    
    # Compute statistics
    print(f"\nComputing statistics across {len(results)} seeds...")
    stats = compute_statistics(results)
    
    # Create summary table
    table = create_summary_table(stats)
    
    # Print results
    print_results(stats, table)
    
    # Save results
    output_file = results_dir / 'multiseed_summary.json'
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    
    table_file = results_dir / 'summary_table.csv'
    table.to_csv(table_file, index=False)
    
    print(f"\nResults saved:")
    print(f"  Summary statistics: {output_file}")
    print(f"  Summary table: {table_file}")
    print()


if __name__ == "__main__":
    main()
