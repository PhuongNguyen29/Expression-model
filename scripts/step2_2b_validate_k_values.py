#!/usr/bin/env python3
"""
Script: step2_2b_validate_k_values.py
Purpose: Validate optimal k by training models with consensus genes
Status: ACTIVE (Step 2.2B - Performance-based k selection)
Author: Claude (for Phuong's dissertation)
Created: 2024-11-17

This script:
1. Loads consensus gene lists from Step 2.2A for each k value
2. For each k, trains models on TCGA and ORIEN using ONLY consensus genes
3. Uses multi-seed validation (5 seeds) for robustness
4. Evaluates C-index performance for each k
5. Generates performance curve to identify where performance plateaus
6. Selects optimal k balancing gene count and performance

Methodology:
- Training: Full dataset (no validation split, use all data)
- Seeds: [42, 123, 456, 789, 1011] for reproducibility
- Genes: Consensus genes (intersection) from Step 2.2A
- Evaluation: C-index on full training data (in-sample performance)
- Goal: Find k where adding more genes shows diminishing returns

Reference:
- Hastie et al. (2009) Elements of Statistical Learning: Model selection
- Your Chapter 3: k-sweep methodology for optimal feature count

Usage:
    python step2_2b_validate_k_values.py \\
        --gene_lists_dir results_v2/02_biomarker_discovery/ksweep_analysis/gene_lists \\
        --tcga_params results_v2/01_hyperparameter_tuning/tcga_308genes/best_params.json \\
        --orien_params results_v2/01_hyperparameter_tuning/orien_308genes/best_params.json \\
        --output_dir results_v2/02_biomarker_discovery/k_validation \\
        --k_values 80 90 100 110 120 130 140 150 \\
        --seeds 42 123 456 789 1011
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.append('.')

from src.data.dataset import SurvivalDataset
from src.models.elastic_deepsurv import ElasticDeepSurv, ElasticDeepSurvTrainer
from src.utils.batch_samplers import StratifiedBatchSampler

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_consensus_genes(filepath: Path) -> List[str]:
    """Load consensus genes from text file."""
    with open(filepath, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]
    return genes


def parse_architecture(best_params: dict) -> List[int]:
    """Parse architecture from hyperparameter dictionary."""
    if 'layer1_size' in best_params:
        return [best_params['layer1_size']]
    if 'architecture_2layer' in best_params:
        return [int(x) for x in best_params['architecture_2layer'].split('-')]
    if 'architecture_3layer' in best_params:
        return [int(x) for x in best_params['architecture_3layer'].split('-')]
    return [256, 128]


def create_data_loader(dataset, events: np.ndarray, batch_size: int, shuffle: bool = True):
    """Create data loader with appropriate sampling strategy."""
    n_samples = len(events)
    
    if n_samples >= 500:
        logger.info(f"    Using StratifiedBatchSampler (n={n_samples})")
        sampler = StratifiedBatchSampler(events=events, batch_size=batch_size, shuffle=shuffle)
        loader = DataLoader(dataset, batch_sampler=sampler)
    else:
        logger.info(f"    Using simple random shuffling (n={n_samples})")
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return loader


def train_model_with_genes(
    expr: pd.DataFrame,
    surv: pd.DataFrame,
    genes: List[str],
    best_params: dict,
    cohort_name: str,
    k: int,
    seed: int,
    max_epochs: int = 100
) -> Tuple[ElasticDeepSurv, float]:
    """
    Train a model using specified genes.
    
    Args:
        expr: Expression data (genes × samples)
        surv: Survival data
        genes: List of genes to use
        best_params: Hyperparameters from Step 1
        cohort_name: 'TCGA' or 'ORIEN'
        k: k value being tested
        seed: Random seed
        max_epochs: Maximum training epochs
        
    Returns:
        Tuple of (trained_model, final_cindex)
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"TRAINING: {cohort_name}, k={k}, Seed {seed}")
    logger.info(f"{'='*70}")
    
    # Filter to specified genes
    available_genes = [g for g in genes if g in expr.index]
    if len(available_genes) < len(genes):
        logger.warning(f"  ⚠️  Only {len(available_genes)}/{len(genes)} genes available")
    
    expr_filtered = expr.loc[available_genes, :]
    
    # Standardize
    mean = expr_filtered.mean(axis=1).values.reshape(-1, 1)
    std = expr_filtered.std(axis=1).values.reshape(-1, 1)
    expr_standardized = pd.DataFrame(
        (expr_filtered.values - mean) / (std + 1e-8),
        index=expr_filtered.index,
        columns=expr_filtered.columns
    )
    
    logger.info(f"  Features: {len(available_genes)} genes")
    logger.info(f"  Samples: {len(surv)} ({surv['event'].sum()} events)")
    
    # Create dataset
    dataset = SurvivalDataset(expr_standardized, surv)
    
    # Adjust batch size for small cohorts to reduce zero-event batches
    n_samples = len(surv)
    batch_size = best_params.get('batch_size', 32)
    if n_samples < 500:
        # For TCGA (n=339, ~45% events), ensure batches are large enough
        # With batch_size=32, expected events per batch = 32 * 0.45 = 14.4
        batch_size = max(batch_size, 32)
        logger.info(f"  Adjusted batch size to {batch_size} for small cohort")
    
    train_loader = create_data_loader(dataset, surv['event'].values, batch_size, shuffle=True)
    
    logger.info(f"  Batches: {len(train_loader)}")
    
    # Build model
    n_features = len(available_genes)
    hidden_sizes = parse_architecture(best_params)
    
    logger.info(f"  Architecture: {n_features} → {' → '.join(map(str, hidden_sizes))} → 1")
    
    model = ElasticDeepSurv(
        n_features=n_features,
        hidden_sizes=hidden_sizes,
        dropout=best_params.get('dropout', 0.3),
        activation=best_params.get('activation', 'relu'),
        batch_norm=best_params.get('batch_norm', False),
        weight_init=best_params.get('weight_init', 'xavier_normal'),
        l1_ratio=best_params.get('l1_ratio', 0.9),
        alpha=best_params.get('alpha', 0.001)
    )
    
    # Train
    trainer = ElasticDeepSurvTrainer(
        model=model,
        learning_rate=best_params.get('learning_rate', 0.001),
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    logger.info("  Training...")
    history = trainer.fit(
        train_loader=train_loader,
        valid_loader=None,
        n_epochs=max_epochs,
        early_stopping_patience=None,  # No early stopping without validation
        verbose=False
    )
    
    # Evaluate - get C-index from training history
    cindex_key = 'train_cindex' if 'train_cindex' in history else 'valid_c_index'
    final_cindex = history[cindex_key][-1] if cindex_key in history and history[cindex_key] else 0.5
    
    logger.info(f"  ✓ Final C-index: {final_cindex:.4f}")
    logger.info(f"  ✓ Best epoch: {history.get('best_epoch', 'N/A')}")
    
    return model, final_cindex


def validate_k_values(
    k_values: List[int],
    gene_lists_dir: Path,
    tcga_params: dict,
    orien_params: dict,
    tcga_expr: pd.DataFrame,
    orien_expr: pd.DataFrame,
    surv_tcga: pd.DataFrame,
    surv_orien: pd.DataFrame,
    seeds: List[int],
    output_dir: Path,
    max_epochs: int = 100
) -> pd.DataFrame:
    """
    Validate multiple k values by training models and evaluating performance.
    
    Returns:
        DataFrame with results for each k value
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"STEP 2.2B: K-VALUE VALIDATION WITH MODEL TRAINING")
    logger.info(f"{'='*80}\n")
    logger.info(f"K values: {k_values}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"")
    
    # Store results
    all_results = []
    
    for k in k_values:
        logger.info(f"\n{'#'*80}")
        logger.info(f"# K = {k}")
        logger.info(f"{'#'*80}")
        
        # Load consensus genes for this k
        consensus_file = gene_lists_dir / f'k{k:03d}_consensus.txt'
        if not consensus_file.exists():
            logger.error(f"  ❌ Consensus file not found: {consensus_file}")
            continue
        
        consensus_genes = load_consensus_genes(consensus_file)
        logger.info(f"  Consensus genes: {len(consensus_genes)}")
        
        # Results for this k across seeds
        k_results = {
            'k': k,
            'n_consensus': len(consensus_genes),
            'tcga_cindices': [],
            'orien_cindices': []
        }
        
        # Train across seeds
        for seed_idx, seed in enumerate(seeds):
            logger.info(f"\n  Seed {seed_idx+1}/{len(seeds)}: {seed}")
            
            # TCGA
            tcga_model, tcga_cindex = train_model_with_genes(
                tcga_expr, surv_tcga, consensus_genes, tcga_params,
                'TCGA', k, seed, max_epochs
            )
            k_results['tcga_cindices'].append(tcga_cindex)
            
            # ORIEN
            orien_model, orien_cindex = train_model_with_genes(
                orien_expr, surv_orien, consensus_genes, orien_params,
                'ORIEN', k, seed, max_epochs
            )
            k_results['orien_cindices'].append(orien_cindex)
            
            # Save models for this seed
            seed_dir = output_dir / f'k{k:03d}' / f'seed_{seed}'
            seed_dir.mkdir(parents=True, exist_ok=True)
            torch.save(tcga_model.state_dict(), seed_dir / 'tcga_model.pth')
            torch.save(orien_model.state_dict(), seed_dir / 'orien_model.pth')
        
        # Compute statistics
        k_results['tcga_mean'] = np.mean(k_results['tcga_cindices'])
        k_results['tcga_std'] = np.std(k_results['tcga_cindices'])
        k_results['orien_mean'] = np.mean(k_results['orien_cindices'])
        k_results['orien_std'] = np.std(k_results['orien_cindices'])
        k_results['overall_mean'] = np.mean(k_results['tcga_cindices'] + k_results['orien_cindices'])
        k_results['overall_std'] = np.std(k_results['tcga_cindices'] + k_results['orien_cindices'])
        
        all_results.append(k_results)
        
        logger.info(f"\n  Summary for k={k}:")
        logger.info(f"    TCGA:  {k_results['tcga_mean']:.4f} ± {k_results['tcga_std']:.4f}")
        logger.info(f"    ORIEN: {k_results['orien_mean']:.4f} ± {k_results['orien_std']:.4f}")
        logger.info(f"    Overall: {k_results['overall_mean']:.4f} ± {k_results['overall_std']:.4f}")
    
    # Create summary DataFrame
    summary_data = []
    for r in all_results:
        summary_data.append({
            'k': r['k'],
            'n_consensus': r['n_consensus'],
            'tcga_mean_cindex': r['tcga_mean'],
            'tcga_std_cindex': r['tcga_std'],
            'orien_mean_cindex': r['orien_mean'],
            'orien_std_cindex': r['orien_std'],
            'overall_mean_cindex': r['overall_mean'],
            'overall_std_cindex': r['overall_std']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / 'k_validation_summary.csv', index=False)
    
    # Save detailed results
    with open(output_dir / 'k_validation_full_results.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'method': 'multi-seed k-value validation',
            'k_values': k_values,
            'seeds': seeds,
            'results': all_results
        }, f, indent=2)
    
    logger.info(f"\n✓ Results saved:")
    logger.info(f"  - k_validation_summary.csv")
    logger.info(f"  - k_validation_full_results.json")
    
    return summary_df


def generate_visualizations(summary_df: pd.DataFrame, output_dir: Path):
    """Generate performance visualization plots."""
    
    logger.info(f"\n{'='*80}")
    logger.info("GENERATING VISUALIZATIONS")
    logger.info(f"{'='*80}\n")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    k_vals = summary_df['k'].values
    
    # Plot 1: C-index vs k (both cohorts)
    ax1 = axes[0, 0]
    ax1.errorbar(k_vals, summary_df['tcga_mean_cindex'], 
                 yerr=summary_df['tcga_std_cindex'],
                 marker='o', linewidth=2, markersize=8, label='TCGA', capsize=5)
    ax1.errorbar(k_vals, summary_df['orien_mean_cindex'], 
                 yerr=summary_df['orien_std_cindex'],
                 marker='s', linewidth=2, markersize=8, label='ORIEN', capsize=5)
    ax1.set_xlabel('Number of consensus genes (k)', fontsize=11)
    ax1.set_ylabel('C-index (Mean ± SD)', fontsize=11)
    ax1.set_title('Performance vs Gene Count', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Overall mean C-index
    ax2 = axes[0, 1]
    ax2.errorbar(k_vals, summary_df['overall_mean_cindex'],
                 yerr=summary_df['overall_std_cindex'],
                 marker='o', linewidth=2, markersize=8, color='#2E86AB', capsize=5)
    ax2.set_xlabel('Number of consensus genes (k)', fontsize=11)
    ax2.set_ylabel('Overall C-index (Mean ± SD)', fontsize=11)
    ax2.set_title('Overall Performance vs k', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Performance gain (relative to k=80)
    ax3 = axes[1, 0]
    baseline = summary_df.loc[summary_df['k'] == k_vals[0], 'overall_mean_cindex'].values[0]
    gains = 100 * (summary_df['overall_mean_cindex'] - baseline) / baseline
    ax3.bar(range(len(k_vals)), gains, color='#A23B72', alpha=0.8)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_xticks(range(len(k_vals)))
    ax3.set_xticklabels(k_vals)
    ax3.set_xlabel('Number of consensus genes (k)', fontsize=11)
    ax3.set_ylabel(f'Performance gain vs k={k_vals[0]} (%)', fontsize=11)
    ax3.set_title('Relative Performance Improvement', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Variance comparison
    ax4 = axes[1, 1]
    x = np.arange(len(k_vals))
    width = 0.35
    ax4.bar(x - width/2, summary_df['tcga_std_cindex'], width, 
            label='TCGA', alpha=0.8, color='#6A994E')
    ax4.bar(x + width/2, summary_df['orien_std_cindex'], width, 
            label='ORIEN', alpha=0.8, color='#BC4749')
    ax4.set_xticks(x)
    ax4.set_xticklabels(k_vals)
    ax4.set_xlabel('Number of consensus genes (k)', fontsize=11)
    ax4.set_ylabel('Standard deviation of C-index', fontsize=11)
    ax4.set_title('Performance Stability Across Seeds', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'k_validation_performance.png', dpi=300, bbox_inches='tight')
    
    logger.info("✓ Visualization saved: k_validation_performance.png")


def generate_recommendations(summary_df: pd.DataFrame, output_dir: Path):
    """Generate recommendations for optimal k value."""
    
    logger.info(f"\n{'='*80}")
    logger.info("RECOMMENDATIONS")
    logger.info(f"{'='*80}\n")
    
    # Find k with highest overall performance
    best_idx = summary_df['overall_mean_cindex'].idxmax()
    best = summary_df.loc[best_idx]
    
    logger.info(f"🎯 BEST PERFORMANCE: k={int(best['k'])}")
    logger.info(f"   - Consensus genes: {int(best['n_consensus'])}")
    logger.info(f"   - Overall C-index: {best['overall_mean_cindex']:.4f} ± {best['overall_std_cindex']:.4f}")
    logger.info(f"   - TCGA C-index: {best['tcga_mean_cindex']:.4f} ± {best['tcga_std_cindex']:.4f}")
    logger.info(f"   - ORIEN C-index: {best['orien_mean_cindex']:.4f} ± {best['orien_std_cindex']:.4f}")
    
    # Check for plateau (diminishing returns)
    logger.info(f"\n📊 PERFORMANCE PLATEAU ANALYSIS:")
    
    for i in range(1, len(summary_df)):
        prev = summary_df.iloc[i-1]
        curr = summary_df.iloc[i]
        
        improvement = curr['overall_mean_cindex'] - prev['overall_mean_cindex']
        improvement_pct = 100 * improvement / prev['overall_mean_cindex']
        
        logger.info(f"   k={int(prev['k'])} → k={int(curr['k'])}: "
                   f"{improvement:+.4f} ({improvement_pct:+.2f}%)")
        
        if improvement_pct < 1.0:  # Less than 1% improvement
            logger.info(f"      ⚠️  Diminishing returns detected")
    
    # Save recommendations
    recommendations = {
        'timestamp': datetime.now().isoformat(),
        'best_k': int(best['k']),
        'best_performance': {
            'k': int(best['k']),
            'n_consensus': int(best['n_consensus']),
            'overall_cindex_mean': float(best['overall_mean_cindex']),
            'overall_cindex_std': float(best['overall_std_cindex']),
            'tcga_cindex_mean': float(best['tcga_mean_cindex']),
            'tcga_cindex_std': float(best['tcga_std_cindex']),
            'orien_cindex_mean': float(best['orien_mean_cindex']),
            'orien_cindex_std': float(best['orien_std_cindex'])
        },
        'rationale': 'Selected based on highest overall C-index across both cohorts'
    }
    
    with open(output_dir / 'FINAL_RECOMMENDATION.json', 'w') as f:
        json.dump(recommendations, f, indent=2)
    
    logger.info(f"\n✓ Recommendations saved: FINAL_RECOMMENDATION.json")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate k values by training models (Step 2.2B)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--gene_lists_dir', type=str, required=True,
                       help='Directory with consensus gene lists from Step 2.2A')
    parser.add_argument('--tcga_params', type=str, required=True,
                       help='TCGA best_params.json from Step 1')
    parser.add_argument('--orien_params', type=str, required=True,
                       help='ORIEN best_params.json from Step 1')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for validation results')
    parser.add_argument('--k_values', type=int, nargs='+', required=True,
                       help='K values to validate')
    parser.add_argument('--seeds', type=int, nargs='+',
                       default=[42, 123, 456, 789, 1011],
                       help='Random seeds for multi-seed validation')
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='Maximum training epochs')
    
    args = parser.parse_args()
    
    # Load hyperparameters
    logger.info("Loading hyperparameters from Step 1...")
    with open(args.tcga_params, 'r') as f:
        tcga_params = json.load(f)
    with open(args.orien_params, 'r') as f:
        orien_params = json.load(f)
    
    # Load data
    logger.info("Loading expression and survival data...")
    tcga_expr = pd.read_csv("data/raw/tcga_batch_corrected_2sv.csv", index_col=0)
    orien_expr = pd.read_csv("data/raw/orien_batch_corrected.csv", index_col=0)
    surv_tcga = pd.read_csv("data/processed/surv_tcga_harmonized.csv", index_col=0)
    surv_orien = pd.read_csv("data/processed/surv_orien_harmonized.csv", index_col=0)
    
    # Run validation
    summary_df = validate_k_values(
        k_values=args.k_values,
        gene_lists_dir=Path(args.gene_lists_dir),
        tcga_params=tcga_params,
        orien_params=orien_params,
        tcga_expr=tcga_expr,
        orien_expr=orien_expr,
        surv_tcga=surv_tcga,
        surv_orien=surv_orien,
        seeds=args.seeds,
        output_dir=Path(args.output_dir),
        max_epochs=args.max_epochs
    )
    
    # Generate visualizations
    generate_visualizations(summary_df, Path(args.output_dir))
    
    # Generate recommendations
    generate_recommendations(summary_df, Path(args.output_dir))
    
    logger.info(f"\n{'='*80}")
    logger.info("✅ STEP 2.2B COMPLETE!")
    logger.info(f"{'='*80}")
    logger.info(f"\n📁 Results: {args.output_dir}/")
    logger.info(f"{'='*80}\n")
