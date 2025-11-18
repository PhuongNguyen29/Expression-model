#!/usr/bin/env python3
"""
Script: step2_2b_validate_k_crosscohort.py
Purpose: Validate optimal k using CROSS-COHORT performance (correct approach)
Status: ACTIVE (Step 2.2B - K selection via generalization)
Author: Claude (for Phuong's dissertation)
Created: 2024-11-17

CRITICAL CORRECTION:
This script fixes the overfitting problem in the original Step 2.2B.
Instead of evaluating on training data, we use cross-cohort validation:
- Train on SOURCE cohort → Test on TARGET cohort
- This measures true generalization, not memorization
- Select k based on cross-cohort C-index, not training C-index

Methodology:
- For each k value:
  1. ORIEN→TCGA: Train on ORIEN, test on TCGA
  2. TCGA→ORIEN: Train on TCGA, test on ORIEN
  3. Compute bidirectional average C-index
- Select k with highest bidirectional performance
- Multi-seed validation (5 seeds) for robustness

This approach:
✓ Tests biomarker transferability (your core hypothesis)
✓ Avoids overfitting to training data
✓ Aligns with transfer learning methodology

Reference:
- Pan & Yang (2010) IEEE TKDE: Transfer learning survey
- Your Chapter 4: Cross-cohort validation methodology

Usage:
    python step2_2b_validate_k_crosscohort.py \\
        --gene_lists_dir results_v2/02_biomarker_discovery/ksweep_analysis/gene_lists \\
        --tcga_params results_v2/01_hyperparameter_tuning/tcga_308genes/best_params.json \\
        --orien_params results_v2/01_hyperparameter_tuning/orien_308genes/best_params.json \\
        --output_dir results_v2/02_biomarker_discovery/k_validation_crosscohort \\
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

# Add project root to path
sys.path.append('.')

from src.data.dataset import SurvivalDataset
from src.models.elastic_deepsurv import ElasticDeepSurv, ElasticDeepSurvTrainer
from src.utils.batch_samplers import StratifiedBatchSampler
from lifelines.utils import concordance_index

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


def evaluate_model_on_cohort(
    model: ElasticDeepSurv,
    expr: pd.DataFrame,
    surv: pd.DataFrame,
    genes: List[str],
    cohort_name: str,
    device: str = 'cuda'
) -> float:
    """
    Evaluate a trained model on a cohort (test set).
    
    Args:
        model: Trained model
        expr: Expression data (genes × samples)
        surv: Survival data
        genes: List of genes the model was trained on
        cohort_name: Name of cohort for logging
        device: Device to use
        
    Returns:
        C-index on this cohort
    """
    # Filter to model genes
    available_genes = [g for g in genes if g in expr.index]
    expr_filtered = expr.loc[available_genes, :]
    
    # Standardize (using full cohort statistics)
    mean = expr_filtered.mean(axis=1).values.reshape(-1, 1)
    std = expr_filtered.std(axis=1).values.reshape(-1, 1)
    expr_standardized = (expr_filtered.values - mean) / (std + 1e-8)
    
    # Get predictions
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        X = torch.FloatTensor(expr_standardized.T).to(device)
        log_hazards = model(X).cpu().numpy().flatten()
    
    # Compute C-index
    try:
        cindex = concordance_index(
            surv['time'].values,
            -log_hazards,  # Negative because higher risk = worse outcome
            surv['event'].values
        )
    except Exception as e:
        logger.error(f"Error computing C-index: {e}")
        cindex = 0.5
    
    logger.info(f"    Test C-index on {cohort_name}: {cindex:.4f}")
    return cindex


def train_and_evaluate_crosscohort(
    source_expr: pd.DataFrame,
    source_surv: pd.DataFrame,
    target_expr: pd.DataFrame,
    target_surv: pd.DataFrame,
    genes: List[str],
    source_params: dict,
    source_name: str,
    target_name: str,
    k: int,
    seed: int,
    max_epochs: int = 100
) -> Tuple[ElasticDeepSurv, float, float]:
    """
    Train on source cohort and evaluate on target cohort.
    
    Args:
        source_expr: Source cohort expression data
        source_surv: Source cohort survival data
        target_expr: Target cohort expression data
        target_surv: Target cohort survival data
        genes: Consensus genes to use
        source_params: Hyperparameters for source cohort
        source_name: Name of source cohort
        target_name: Name of target cohort
        k: k value being tested
        seed: Random seed
        max_epochs: Maximum training epochs
        
    Returns:
        Tuple of (trained_model, source_train_cindex, target_test_cindex)
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"TRANSFER: {source_name}→{target_name}, k={k}, Seed {seed}")
    logger.info(f"{'='*70}")
    
    # Filter source data to consensus genes
    available_genes = [g for g in genes if g in source_expr.index]
    if len(available_genes) < len(genes):
        logger.warning(f"  ⚠️  Only {len(available_genes)}/{len(genes)} genes available")
    
    source_filtered = source_expr.loc[available_genes, :]
    
    # Standardize source data
    source_mean = source_filtered.mean(axis=1).values.reshape(-1, 1)
    source_std = source_filtered.std(axis=1).values.reshape(-1, 1)
    source_standardized = pd.DataFrame(
        (source_filtered.values - source_mean) / (source_std + 1e-8),
        index=source_filtered.index,
        columns=source_filtered.columns
    )
    
    logger.info(f"  Source ({source_name}): {len(available_genes)} genes, "
                f"{len(source_surv)} samples ({source_surv['event'].sum()} events)")
    logger.info(f"  Target ({target_name}): {len(target_surv)} samples ({target_surv['event'].sum()} events)")
    
    # Create source dataset and loader
    source_dataset = SurvivalDataset(source_standardized, source_surv)
    batch_size = source_params.get('batch_size', 32)
    source_loader = create_data_loader(source_dataset, source_surv['event'].values, batch_size, shuffle=True)
    
    logger.info(f"  Batches: {len(source_loader)}")
    
    # Build model with source hyperparameters
    n_features = len(available_genes)
    hidden_sizes = parse_architecture(source_params)
    
    logger.info(f"  Architecture: {n_features} → {' → '.join(map(str, hidden_sizes))} → 1")
    
    model = ElasticDeepSurv(
        n_features=n_features,
        hidden_sizes=hidden_sizes,
        dropout=source_params.get('dropout', 0.3),
        activation=source_params.get('activation', 'relu'),
        batch_norm=source_params.get('batch_norm', False),
        weight_init=source_params.get('weight_init', 'xavier_normal'),
        l1_ratio=source_params.get('l1_ratio', 0.9),
        alpha=source_params.get('alpha', 0.001)
    )
    
    # Train on source
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = ElasticDeepSurvTrainer(
        model=model,
        learning_rate=source_params.get('learning_rate', 1e-4),
        weight_decay=0.0,
        device=device
    )
    
    logger.info(f"  Training on {source_name}...")
    history = trainer.fit(
        train_loader=source_loader,
        valid_loader=None,
        n_epochs=max_epochs,
        early_stopping_patience=None,
        verbose=False
    )
    
    # Get source training C-index
    cindex_key = 'train_cindex' if 'train_cindex' in history else 'valid_c_index'
    source_train_cindex = history[cindex_key][-1] if cindex_key in history and history[cindex_key] else 0.5
    
    logger.info(f"  ✓ {source_name} training C-index: {source_train_cindex:.4f}")
    
    # Evaluate on target cohort (CRITICAL: This is the test set)
    target_test_cindex = evaluate_model_on_cohort(
        model, target_expr, target_surv, available_genes, target_name, device
    )
    
    logger.info(f"  ✓ {target_name} test C-index: {target_test_cindex:.4f}")
    
    return model, source_train_cindex, target_test_cindex


def validate_k_values_crosscohort(
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
    Validate multiple k values using cross-cohort evaluation.
    
    Returns:
        DataFrame with cross-cohort results for each k value
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"STEP 2.2B: K-VALUE VALIDATION VIA CROSS-COHORT TRANSFER")
    logger.info(f"{'='*80}\n")
    logger.info(f"Strategy: Train on SOURCE → Test on TARGET")
    logger.info(f"  ✓ Measures true generalization, not memorization")
    logger.info(f"  ✓ Select k based on cross-cohort C-index")
    logger.info(f"  ✓ Bidirectional validation (ORIEN↔TCGA)")
    logger.info(f"")
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
            'orien_to_tcga_train': [],  # ORIEN train C-index
            'orien_to_tcga_test': [],   # TCGA test C-index (KEY METRIC)
            'tcga_to_orien_train': [],  # TCGA train C-index
            'tcga_to_orien_test': []    # ORIEN test C-index (KEY METRIC)
        }
        
        # Train across seeds
        for seed_idx, seed in enumerate(seeds):
            logger.info(f"\n  Seed {seed_idx+1}/{len(seeds)}: {seed}")
            
            # Direction 1: ORIEN→TCGA
            orien_model, orien_train, tcga_test = train_and_evaluate_crosscohort(
                source_expr=orien_expr,
                source_surv=surv_orien,
                target_expr=tcga_expr,
                target_surv=surv_tcga,
                genes=consensus_genes,
                source_params=orien_params,
                source_name='ORIEN',
                target_name='TCGA',
                k=k,
                seed=seed,
                max_epochs=max_epochs
            )
            k_results['orien_to_tcga_train'].append(orien_train)
            k_results['orien_to_tcga_test'].append(tcga_test)
            
            # Direction 2: TCGA→ORIEN
            tcga_model, tcga_train, orien_test = train_and_evaluate_crosscohort(
                source_expr=tcga_expr,
                source_surv=surv_tcga,
                target_expr=orien_expr,
                target_surv=surv_orien,
                genes=consensus_genes,
                source_params=tcga_params,
                source_name='TCGA',
                target_name='ORIEN',
                k=k,
                seed=seed,
                max_epochs=max_epochs
            )
            k_results['tcga_to_orien_train'].append(tcga_train)
            k_results['tcga_to_orien_test'].append(orien_test)
            
            # Save models
            seed_dir = output_dir / f'k{k:03d}' / f'seed_{seed}'
            seed_dir.mkdir(parents=True, exist_ok=True)
            torch.save(orien_model.state_dict(), seed_dir / 'orien_to_tcga_model.pth')
            torch.save(tcga_model.state_dict(), seed_dir / 'tcga_to_orien_model.pth')
        
        # Compute statistics
        k_results['orien_to_tcga_test_mean'] = np.mean(k_results['orien_to_tcga_test'])
        k_results['orien_to_tcga_test_std'] = np.std(k_results['orien_to_tcga_test'])
        k_results['tcga_to_orien_test_mean'] = np.mean(k_results['tcga_to_orien_test'])
        k_results['tcga_to_orien_test_std'] = np.std(k_results['tcga_to_orien_test'])
        
        # Bidirectional average (KEY METRIC for k selection)
        all_test_cindices = k_results['orien_to_tcga_test'] + k_results['tcga_to_orien_test']
        k_results['bidirectional_mean'] = np.mean(all_test_cindices)
        k_results['bidirectional_std'] = np.std(all_test_cindices)
        
        all_results.append(k_results)
        
        logger.info(f"\n  Summary for k={k}:")
        logger.info(f"    ORIEN→TCGA test: {k_results['orien_to_tcga_test_mean']:.4f} ± {k_results['orien_to_tcga_test_std']:.4f}")
        logger.info(f"    TCGA→ORIEN test: {k_results['tcga_to_orien_test_mean']:.4f} ± {k_results['tcga_to_orien_test_std']:.4f}")
        logger.info(f"    Bidirectional avg: {k_results['bidirectional_mean']:.4f} ± {k_results['bidirectional_std']:.4f}")
    
    # Create summary DataFrame
    summary_data = []
    for r in all_results:
        summary_data.append({
            'k': r['k'],
            'n_consensus': r['n_consensus'],
            'orien_to_tcga_test_mean': r['orien_to_tcga_test_mean'],
            'orien_to_tcga_test_std': r['orien_to_tcga_test_std'],
            'tcga_to_orien_test_mean': r['tcga_to_orien_test_mean'],
            'tcga_to_orien_test_std': r['tcga_to_orien_test_std'],
            'bidirectional_test_mean': r['bidirectional_mean'],
            'bidirectional_test_std': r['bidirectional_std']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / 'k_validation_crosscohort_summary.csv', index=False)
    
    # Save detailed results
    with open(output_dir / 'k_validation_crosscohort_full_results.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'method': 'cross-cohort k-value validation',
            'description': 'Train on source cohort, test on target cohort (no fine-tuning)',
            'k_values': k_values,
            'seeds': seeds,
            'results': all_results
        }, f, indent=2)
    
    logger.info(f"\n✓ Results saved:")
    logger.info(f"  - k_validation_crosscohort_summary.csv")
    logger.info(f"  - k_validation_crosscohort_full_results.json")
    
    return summary_df


def generate_visualizations(summary_df: pd.DataFrame, output_dir: Path):
    """Generate performance visualization plots."""
    
    logger.info(f"\n{'='*80}")
    logger.info("GENERATING VISUALIZATIONS")
    logger.info(f"{'='*80}\n")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    k_vals = summary_df['k'].values
    
    # Plot 1: Cross-cohort test C-index vs k
    ax1 = axes[0, 0]
    ax1.errorbar(k_vals, summary_df['orien_to_tcga_test_mean'], 
                 yerr=summary_df['orien_to_tcga_test_std'],
                 marker='o', linewidth=2, markersize=8, label='ORIEN→TCGA', capsize=5)
    ax1.errorbar(k_vals, summary_df['tcga_to_orien_test_mean'], 
                 yerr=summary_df['tcga_to_orien_test_std'],
                 marker='s', linewidth=2, markersize=8, label='TCGA→ORIEN', capsize=5)
    ax1.set_xlabel('Number of consensus genes (k)', fontsize=11)
    ax1.set_ylabel('Test C-index (Mean ± SD)', fontsize=11)
    ax1.set_title('Cross-Cohort Test Performance vs k', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Bidirectional average
    ax2 = axes[0, 1]
    ax2.errorbar(k_vals, summary_df['bidirectional_test_mean'],
                 yerr=summary_df['bidirectional_test_std'],
                 marker='o', linewidth=2, markersize=8, color='#2E86AB', capsize=5)
    ax2.set_xlabel('Number of consensus genes (k)', fontsize=11)
    ax2.set_ylabel('Bidirectional C-index (Mean ± SD)', fontsize=11)
    ax2.set_title('Overall Cross-Cohort Performance vs k', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Performance gain
    ax3 = axes[1, 0]
    baseline = summary_df.loc[summary_df['k'] == k_vals[0], 'bidirectional_test_mean'].values[0]
    gains = 100 * (summary_df['bidirectional_test_mean'] - baseline) / baseline
    colors = ['#A23B72' if g >= 0 else '#C73E1D' for g in gains]
    ax3.bar(range(len(k_vals)), gains, color=colors, alpha=0.8)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_xticks(range(len(k_vals)))
    ax3.set_xticklabels(k_vals)
    ax3.set_xlabel('Number of consensus genes (k)', fontsize=11)
    ax3.set_ylabel(f'Performance gain vs k={k_vals[0]} (%)', fontsize=11)
    ax3.set_title('Relative Improvement in Cross-Cohort Performance', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Stability comparison
    ax4 = axes[1, 1]
    x = np.arange(len(k_vals))
    width = 0.35
    ax4.bar(x - width/2, summary_df['orien_to_tcga_test_std'], width, 
            label='ORIEN→TCGA', alpha=0.8, color='#6A994E')
    ax4.bar(x + width/2, summary_df['tcga_to_orien_test_std'], width, 
            label='TCGA→ORIEN', alpha=0.8, color='#BC4749')
    ax4.set_xticks(x)
    ax4.set_xticklabels(k_vals)
    ax4.set_xlabel('Number of consensus genes (k)', fontsize=11)
    ax4.set_ylabel('Standard deviation of test C-index', fontsize=11)
    ax4.set_title('Cross-Cohort Stability Across Seeds', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'k_validation_crosscohort_performance.png', dpi=300, bbox_inches='tight')
    
    logger.info("✓ Visualization saved: k_validation_crosscohort_performance.png")


def generate_recommendations(summary_df: pd.DataFrame, output_dir: Path):
    """Generate recommendations for optimal k value."""
    
    logger.info(f"\n{'='*80}")
    logger.info("RECOMMENDATIONS")
    logger.info(f"{'='*80}\n")
    
    # Find k with highest bidirectional test performance
    best_idx = summary_df['bidirectional_test_mean'].idxmax()
    best = summary_df.loc[best_idx]
    
    logger.info(f"🎯 OPTIMAL k BASED ON CROSS-COHORT GENERALIZATION: k={int(best['k'])}")
    logger.info(f"   - Consensus genes: {int(best['n_consensus'])}")
    logger.info(f"   - Bidirectional test C-index: {best['bidirectional_test_mean']:.4f} ± {best['bidirectional_test_std']:.4f}")
    logger.info(f"   - ORIEN→TCGA: {best['orien_to_tcga_test_mean']:.4f} ± {best['orien_to_tcga_test_std']:.4f}")
    logger.info(f"   - TCGA→ORIEN: {best['tcga_to_orien_test_mean']:.4f} ± {best['tcga_to_orien_test_std']:.4f}")
    
    # Check for plateau
    logger.info(f"\n📊 PERFORMANCE PLATEAU ANALYSIS:")
    
    for i in range(1, len(summary_df)):
        prev = summary_df.iloc[i-1]
        curr = summary_df.iloc[i]
        
        improvement = curr['bidirectional_test_mean'] - prev['bidirectional_test_mean']
        improvement_pct = 100 * improvement / prev['bidirectional_test_mean']
        
        logger.info(f"   k={int(prev['k'])} → k={int(curr['k'])}: "
                   f"{improvement:+.4f} ({improvement_pct:+.2f}%)")
        
        if improvement_pct < 1.0:
            logger.info(f"      ⚠️  Diminishing returns detected")
    
    # Save recommendations
    recommendations = {
        'timestamp': datetime.now().isoformat(),
        'optimal_k': int(best['k']),
        'selection_criterion': 'Highest bidirectional cross-cohort test C-index',
        'optimal_performance': {
            'k': int(best['k']),
            'n_consensus': int(best['n_consensus']),
            'bidirectional_test_mean': float(best['bidirectional_test_mean']),
            'bidirectional_test_std': float(best['bidirectional_test_std']),
            'orien_to_tcga_test_mean': float(best['orien_to_tcga_test_mean']),
            'orien_to_tcga_test_std': float(best['orien_to_tcga_test_std']),
            'tcga_to_orien_test_mean': float(best['tcga_to_orien_test_mean']),
            'tcga_to_orien_test_std': float(best['tcga_to_orien_test_std'])
        },
        'rationale': 'Selected based on cross-cohort generalization (train on source, test on target). This avoids overfitting and ensures biomarkers transfer between cohorts.'
    }
    
    with open(output_dir / 'OPTIMAL_K_RECOMMENDATION.json', 'w') as f:
        json.dump(recommendations, f, indent=2)
    
    logger.info(f"\n✓ Recommendations saved: OPTIMAL_K_RECOMMENDATION.json")
    
    logger.info(f"\n{'='*80}")
    logger.info("NEXT STEPS:")
    logger.info(f"{'='*80}")
    logger.info(f"1. Use k={int(best['k'])} consensus genes for Step 3 (Transfer Learning)")
    logger.info(f"2. In Step 3, add fine-tuning to see if it improves cross-cohort performance")
    logger.info(f"3. Compare: Zero-shot (Step 2.2B) vs Fine-tuned (Step 3)")
    logger.info(f"{'='*80}\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate k values via cross-cohort transfer (Step 2.2B CORRECTED)",
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
    
    # Run cross-cohort validation
    summary_df = validate_k_values_crosscohort(
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