"""
Diagnostic Experiment: Test if ORIEN with TCGA-like Architecture Fixes Importance Compression

Hypothesis: ORIEN's 3-layer network (256-128-32) distributes weights across many neurons,
causing L2 importance compression. Using TCGA's 1-layer [64] architecture should produce
more discriminative importance scores.

Experiment:
1. Train ORIEN with original 3-layer architecture (baseline)
2. Train ORIEN with TCGA-like 1-layer [64] architecture
3. Compare L2 importance distributions
4. Compare cross-cohort validation C-index

Usage:
    python scripts/diagnostic_architecture_experiment.py \
        --consensus_genes data/raw/consensus_genes_308.txt \
        --output_dir results_v2/diagnostics/architecture_comparison

Author: [Your name]
Date: 2024-12
"""

import sys
sys.path.append('.')

import torch
import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
import torch.nn as nn
import matplotlib.pyplot as plt

from src.data.dataset import SurvivalDataset
from torch.utils.data import DataLoader
from src.models.elastic_deepsurv import ElasticDeepSurv, ElasticDeepSurvTrainer
from src.utils.batch_samplers import StratifiedBatchSampler
from lifelines.utils import concordance_index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

SEEDS = [42, 123, 456, 789, 1011]

# Original ORIEN hyperparameters (from your tuning)
ORIEN_ORIGINAL_PARAMS = {
    'n_layers': 3,
    'hidden_sizes': [256, 128, 32],
    'dropout': 0.3,
    'batch_size': 32,
    'alpha': 0.00008100923610860498,
    'activation': 'relu',
    'batch_norm': True,
    'l1_ratio': 0.5,
    'learning_rate': 0.0006197015748809143,
    'weight_init': 'kaiming_normal'
}

# TCGA-like hyperparameters (to test on ORIEN)
TCGA_LIKE_PARAMS = {
    'n_layers': 1,
    'hidden_sizes': [64],
    'dropout': 0.3,
    'batch_size': 32,  # Keep same batch size for fair comparison
    'alpha': 0.000283352705217738,  # TCGA's stronger regularization
    'activation': 'relu',
    'batch_norm': False,
    'l1_ratio': 0.3,  # TCGA's L1 ratio
    'learning_rate': 0.0009944355825542129,
    'weight_init': 'xavier_normal'
}

# Original TCGA hyperparameters (for reference/comparison)
TCGA_ORIGINAL_PARAMS = {
    'n_layers': 1,
    'hidden_sizes': [64],
    'dropout': 0.3,
    'batch_size': 24,
    'alpha': 0.000283352705217738,
    'activation': 'relu',
    'batch_norm': False,
    'l1_ratio': 0.3,
    'learning_rate': 0.0009944355825542129,
    'weight_init': 'xavier_normal'
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_consensus_genes(consensus_file: str) -> List[str]:
    """Load consensus genes from file."""
    with open(consensus_file, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]
    logger.info(f"Loaded {len(genes)} consensus genes")
    return genes


def load_data(consensus_genes: List[str]) -> Dict:
    """Load and preprocess expression and survival data."""
    logger.info("Loading data...")
    
    # Load expression data
    tcga_expr = pd.read_csv("data/raw/tcga_batch_corrected_2sv.csv", index_col=0)
    orien_expr = pd.read_csv("data/raw/orien_batch_corrected.csv", index_col=0)
    
    # Load survival data
    surv_tcga = pd.read_csv("data/processed/surv_tcga_harmonized.csv", index_col=0)
    surv_orien = pd.read_csv("data/processed/surv_orien_harmonized.csv", index_col=0)
    
    # Filter to consensus genes
    available_genes = [g for g in consensus_genes if g in tcga_expr.index and g in orien_expr.index]
    logger.info(f"Using {len(available_genes)} genes available in both cohorts")
    
    tcga_expr = tcga_expr.loc[available_genes]
    orien_expr = orien_expr.loc[available_genes]
    
    # Standardize (z-score per gene)
    def standardize(df):
        mean = df.mean(axis=1)
        std = df.std(axis=1)
        std = std.replace(0, 1)  # Avoid division by zero
        return df.subtract(mean, axis=0).divide(std, axis=0)
    
    tcga_expr = standardize(tcga_expr)
    orien_expr = standardize(orien_expr)
    
    logger.info(f"TCGA: {tcga_expr.shape[0]} genes x {tcga_expr.shape[1]} samples")
    logger.info(f"ORIEN: {orien_expr.shape[0]} genes x {orien_expr.shape[1]} samples")
    
    return {
        'tcga_expr': tcga_expr,
        'orien_expr': orien_expr,
        'surv_tcga': surv_tcga,
        'surv_orien': surv_orien,
        'gene_names': available_genes
    }


def compute_l2_importance(model: ElasticDeepSurv) -> np.ndarray:
    """Compute L2 norm of first layer weights as feature importance."""
    first_layer = model.network[0]
    if not isinstance(first_layer, nn.Linear):
        raise TypeError(f"First layer is {type(first_layer)}, not nn.Linear")
    
    weights = first_layer.weight.data.cpu().numpy()
    importance = np.linalg.norm(weights, axis=0)
    return importance


def train_model(
    expr_df: pd.DataFrame,
    surv_df: pd.DataFrame,
    params: Dict,
    seed: int,
    cohort_name: str,
    max_epochs: int = 150,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[ElasticDeepSurv, Dict, float]:
    """
    Train a model with given parameters.
    
    Uses 80/20 split to find optimal epochs, then trains on 100%.
    
    Returns:
        Tuple of (trained_model, training_history, final_train_cindex)
    """
    set_seed(seed)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {cohort_name} - Seed {seed}")
    logger.info(f"Architecture: {params['hidden_sizes']}")
    logger.info(f"{'='*60}")
    
    n_features = expr_df.shape[0]
    n_samples = expr_df.shape[1]
    
    # Step 1: Use 80/20 split to find optimal epochs
    logger.info("Step 1: Finding optimal epochs with 80/20 split...")
    
    train_idx, val_idx = train_test_split(
        np.arange(n_samples),
        test_size=0.2,
        stratify=surv_df['event'].values,
        random_state=seed
    )
    
    # Create train/val datasets
    train_expr = expr_df.iloc[:, train_idx]
    val_expr = expr_df.iloc[:, val_idx]
    train_surv = surv_df.iloc[train_idx]
    val_surv = surv_df.iloc[val_idx]
    
    train_dataset = SurvivalDataset(train_expr, train_surv)
    val_dataset = SurvivalDataset(val_expr, val_surv)
    
    batch_size = params['batch_size']
    
    # Create dataloaders
    if n_samples >= 500:
        train_sampler = StratifiedBatchSampler(
            events=train_surv['event'].values,
            batch_size=batch_size,
            shuffle=True
        )
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = ElasticDeepSurv(
        n_features=n_features,
        hidden_sizes=params['hidden_sizes'],
        dropout=params['dropout'],
        activation=params['activation'],
        batch_norm=params['batch_norm'],
        weight_init=params['weight_init'],
        l1_ratio=params['l1_ratio'],
        alpha=params['alpha']
    )
    
    trainer = ElasticDeepSurvTrainer(
        model=model,
        learning_rate=params['learning_rate'],
        device=device
    )
    
    # Train with early stopping
    history = trainer.fit(
        train_loader=train_loader,
        valid_loader=val_loader,
        n_epochs=max_epochs,
        early_stopping_patience=20,
        verbose=False
    )
    
    best_epoch = history.get('best_epoch', len(history['train_loss']))
    best_val_cindex = max([c for c in history['valid_c_index'] if c is not None])
    
    logger.info(f"  Best epoch: {best_epoch}, Best val C-index: {best_val_cindex:.4f}")
    
    # Step 2: Train on 100% of data for scaled epochs
    logger.info("Step 2: Training on 100% of data...")
    
    full_dataset = SurvivalDataset(expr_df, surv_df)
    
    if n_samples >= 500:
        full_sampler = StratifiedBatchSampler(
            events=surv_df['event'].values,
            batch_size=batch_size,
            shuffle=True
        )
        full_loader = DataLoader(full_dataset, batch_sampler=full_sampler)
    else:
        full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
    
    # Scale epochs based on data size difference
    train_batches = len(train_loader)
    full_batches = len(full_loader)
    scaled_epochs = max(1, int(best_epoch * train_batches / full_batches))
    
    logger.info(f"  Scaled epochs: {scaled_epochs} (from {best_epoch})")
    
    # Reset model
    model = ElasticDeepSurv(
        n_features=n_features,
        hidden_sizes=params['hidden_sizes'],
        dropout=params['dropout'],
        activation=params['activation'],
        batch_norm=params['batch_norm'],
        weight_init=params['weight_init'],
        l1_ratio=params['l1_ratio'],
        alpha=params['alpha']
    )
    
    trainer = ElasticDeepSurvTrainer(
        model=model,
        learning_rate=params['learning_rate'],
        device=device
    )
    
    # Train on full data (no validation, no early stopping)
    final_history = trainer.fit(
        train_loader=full_loader,
        valid_loader=None,
        n_epochs=scaled_epochs,
        early_stopping_patience=None,
        verbose=False
    )
    
    # Evaluate on training data
    _, _, _, train_cindex = trainer.evaluate(full_loader)
    
    logger.info(f"  Final train C-index: {train_cindex:.4f}")
    
    return model, final_history, train_cindex


def evaluate_cross_cohort(
    model: ElasticDeepSurv,
    test_expr: pd.DataFrame,
    test_surv: pd.DataFrame,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> float:
    """Evaluate model on test cohort."""
    model.to(device)
    model.eval()
    
    test_dataset = SurvivalDataset(test_expr, test_surv)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    all_risks = []
    all_times = []
    all_events = []
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            risks = model(features).cpu().numpy().flatten()
            all_risks.extend(risks)
            all_times.extend(batch['time'].numpy())
            all_events.extend(batch['event'].numpy())
    
    c_index = concordance_index(
        np.array(all_times),
        -np.array(all_risks),
        np.array(all_events)
    )
    
    return c_index


def analyze_importance_distribution(
    importance: np.ndarray,
    label: str
) -> Dict:
    """Analyze importance score distribution."""
    stats = {
        'label': label,
        'min': float(importance.min()),
        'max': float(importance.max()),
        'mean': float(importance.mean()),
        'std': float(importance.std()),
        'range': float(importance.max() - importance.min()),
        'cv': float(importance.std() / importance.mean())  # Coefficient of variation
    }
    
    logger.info(f"\n{label} Importance Distribution:")
    logger.info(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    logger.info(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
    logger.info(f"  Spread (max-min): {stats['range']:.4f}")
    logger.info(f"  CV (std/mean): {stats['cv']:.4f}")
    
    return stats


def plot_importance_comparison(
    original_importance: np.ndarray,
    tcga_like_importance: np.ndarray,
    gene_names: List[str],
    output_dir: Path
):
    """Create comparison plots for importance distributions."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Histogram comparison
    ax1 = axes[0, 0]
    ax1.hist(original_importance, bins=50, alpha=0.7, label='Original (3-layer)', color='blue')
    ax1.hist(tcga_like_importance, bins=50, alpha=0.7, label='TCGA-like (1-layer)', color='orange')
    ax1.set_xlabel('L2 Importance Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Importance Scores')
    ax1.legend()
    
    # Plot 2: Rank comparison
    ax2 = axes[0, 1]
    original_ranks = np.argsort(np.argsort(-original_importance))
    tcga_like_ranks = np.argsort(np.argsort(-tcga_like_importance))
    ax2.scatter(original_ranks, tcga_like_ranks, alpha=0.5, s=10)
    ax2.plot([0, len(gene_names)], [0, len(gene_names)], 'r--', label='Perfect agreement')
    ax2.set_xlabel('Rank (Original 3-layer)')
    ax2.set_ylabel('Rank (TCGA-like 1-layer)')
    ax2.set_title('Rank Comparison')
    ax2.legend()
    
    # Plot 3: Sorted importance curves
    ax3 = axes[1, 0]
    ax3.plot(np.sort(original_importance)[::-1], label='Original (3-layer)', linewidth=2)
    ax3.plot(np.sort(tcga_like_importance)[::-1], label='TCGA-like (1-layer)', linewidth=2)
    ax3.set_xlabel('Gene Rank')
    ax3.set_ylabel('L2 Importance Score')
    ax3.set_title('Sorted Importance Curves')
    ax3.legend()
    
    # Plot 4: Top-k overlap analysis
    ax4 = axes[1, 1]
    k_values = [10, 20, 30, 50, 75, 100, 125, 150]
    overlaps = []
    
    for k in k_values:
        top_k_orig = set(np.argsort(-original_importance)[:k])
        top_k_tcga = set(np.argsort(-tcga_like_importance)[:k])
        overlap = len(top_k_orig & top_k_tcga) / k * 100
        overlaps.append(overlap)
    
    ax4.plot(k_values, overlaps, 'o-', linewidth=2, markersize=8)
    ax4.set_xlabel('k (Top genes)')
    ax4.set_ylabel('Overlap (%)')
    ax4.set_title('Top-k Overlap Between Architectures')
    ax4.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% overlap')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'importance_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved comparison plot to {output_dir / 'importance_comparison.png'}")


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_experiment(
    consensus_genes_file: str,
    output_dir: str,
    seeds: List[int] = SEEDS
):
    """Run the full diagnostic experiment."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("DIAGNOSTIC EXPERIMENT: Architecture Effect on Importance Scores")
    logger.info("="*70)
    logger.info(f"Output: {output_dir}")
    logger.info(f"Seeds: {seeds}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    
    # Load data
    consensus_genes = load_consensus_genes(consensus_genes_file)
    data = load_data(consensus_genes)
    
    # Storage for results
    all_results = {
        'orien_original': {'importance': [], 'train_cindex': [], 'test_cindex': []},
        'orien_tcga_like': {'importance': [], 'train_cindex': [], 'test_cindex': []},
        'tcga_original': {'importance': [], 'train_cindex': [], 'test_cindex': []}
    }
    
    for seed in seeds:
        logger.info(f"\n{'#'*70}")
        logger.info(f"# SEED: {seed}")
        logger.info(f"{'#'*70}")
        
        # ============================================================
        # 1. ORIEN with Original 3-layer architecture
        # ============================================================
        logger.info("\n>>> ORIEN with Original 3-layer architecture <<<")
        
        orien_orig_model, _, orien_orig_train_c = train_model(
            expr_df=data['orien_expr'],
            surv_df=data['surv_orien'],
            params=ORIEN_ORIGINAL_PARAMS,
            seed=seed,
            cohort_name='ORIEN_original'
        )
        
        orien_orig_importance = compute_l2_importance(orien_orig_model)
        orien_orig_test_c = evaluate_cross_cohort(
            orien_orig_model,
            data['tcga_expr'],
            data['surv_tcga'],
            device
        )
        
        logger.info(f"  ORIEN→TCGA C-index: {orien_orig_test_c:.4f}")
        
        all_results['orien_original']['importance'].append(orien_orig_importance)
        all_results['orien_original']['train_cindex'].append(orien_orig_train_c)
        all_results['orien_original']['test_cindex'].append(orien_orig_test_c)
        
        # ============================================================
        # 2. ORIEN with TCGA-like 1-layer architecture
        # ============================================================
        logger.info("\n>>> ORIEN with TCGA-like 1-layer architecture <<<")
        
        orien_tcga_model, _, orien_tcga_train_c = train_model(
            expr_df=data['orien_expr'],
            surv_df=data['surv_orien'],
            params=TCGA_LIKE_PARAMS,
            seed=seed,
            cohort_name='ORIEN_tcga_like'
        )
        
        orien_tcga_importance = compute_l2_importance(orien_tcga_model)
        orien_tcga_test_c = evaluate_cross_cohort(
            orien_tcga_model,
            data['tcga_expr'],
            data['surv_tcga'],
            device
        )
        
        logger.info(f"  ORIEN→TCGA C-index: {orien_tcga_test_c:.4f}")
        
        all_results['orien_tcga_like']['importance'].append(orien_tcga_importance)
        all_results['orien_tcga_like']['train_cindex'].append(orien_tcga_train_c)
        all_results['orien_tcga_like']['test_cindex'].append(orien_tcga_test_c)
        
        # ============================================================
        # 3. TCGA with Original architecture (for reference)
        # ============================================================
        logger.info("\n>>> TCGA with Original 1-layer architecture <<<")
        
        tcga_model, _, tcga_train_c = train_model(
            expr_df=data['tcga_expr'],
            surv_df=data['surv_tcga'],
            params=TCGA_ORIGINAL_PARAMS,
            seed=seed,
            cohort_name='TCGA_original'
        )
        
        tcga_importance = compute_l2_importance(tcga_model)
        tcga_test_c = evaluate_cross_cohort(
            tcga_model,
            data['orien_expr'],
            data['surv_orien'],
            device
        )
        
        logger.info(f"  TCGA→ORIEN C-index: {tcga_test_c:.4f}")
        
        all_results['tcga_original']['importance'].append(tcga_importance)
        all_results['tcga_original']['train_cindex'].append(tcga_train_c)
        all_results['tcga_original']['test_cindex'].append(tcga_test_c)
    
    # ============================================================
    # AGGREGATE RESULTS
    # ============================================================
    
    logger.info(f"\n{'='*70}")
    logger.info("AGGREGATED RESULTS")
    logger.info(f"{'='*70}")
    
    # Average importance across seeds
    orien_orig_avg_importance = np.mean(all_results['orien_original']['importance'], axis=0)
    orien_tcga_avg_importance = np.mean(all_results['orien_tcga_like']['importance'], axis=0)
    tcga_avg_importance = np.mean(all_results['tcga_original']['importance'], axis=0)
    
    # Analyze distributions
    stats_orig = analyze_importance_distribution(orien_orig_avg_importance, "ORIEN Original (3-layer)")
    stats_tcga_like = analyze_importance_distribution(orien_tcga_avg_importance, "ORIEN TCGA-like (1-layer)")
    stats_tcga = analyze_importance_distribution(tcga_avg_importance, "TCGA Original (1-layer)")
    
    # C-index summary
    logger.info("\n=== Cross-Cohort C-Index Summary ===")
    
    for config_name in ['orien_original', 'orien_tcga_like', 'tcga_original']:
        test_cindices = all_results[config_name]['test_cindex']
        mean_c = np.mean(test_cindices)
        std_c = np.std(test_cindices)
        
        if 'orien' in config_name.lower() and 'tcga_like' not in config_name:
            direction = "ORIEN→TCGA"
        elif 'orien_tcga_like' in config_name:
            direction = "ORIEN(1-layer)→TCGA"
        else:
            direction = "TCGA→ORIEN"
        
        logger.info(f"  {config_name}: {mean_c:.4f} ± {std_c:.4f} ({direction})")
    
    # Create comparison plots
    plot_importance_comparison(
        orien_orig_avg_importance,
        orien_tcga_avg_importance,
        data['gene_names'],
        output_dir
    )
    
    # Save detailed results
    results_summary = {
        'experiment': 'architecture_comparison',
        'timestamp': datetime.now().isoformat(),
        'seeds': seeds,
        'n_genes': len(data['gene_names']),
        'orien_original': {
            'architecture': ORIEN_ORIGINAL_PARAMS['hidden_sizes'],
            'importance_stats': stats_orig,
            'train_cindex': {
                'mean': float(np.mean(all_results['orien_original']['train_cindex'])),
                'std': float(np.std(all_results['orien_original']['train_cindex']))
            },
            'test_cindex': {
                'mean': float(np.mean(all_results['orien_original']['test_cindex'])),
                'std': float(np.std(all_results['orien_original']['test_cindex']))
            }
        },
        'orien_tcga_like': {
            'architecture': TCGA_LIKE_PARAMS['hidden_sizes'],
            'importance_stats': stats_tcga_like,
            'train_cindex': {
                'mean': float(np.mean(all_results['orien_tcga_like']['train_cindex'])),
                'std': float(np.std(all_results['orien_tcga_like']['train_cindex']))
            },
            'test_cindex': {
                'mean': float(np.mean(all_results['orien_tcga_like']['test_cindex'])),
                'std': float(np.std(all_results['orien_tcga_like']['test_cindex']))
            }
        },
        'tcga_original': {
            'architecture': TCGA_ORIGINAL_PARAMS['hidden_sizes'],
            'importance_stats': stats_tcga,
            'train_cindex': {
                'mean': float(np.mean(all_results['tcga_original']['train_cindex'])),
                'std': float(np.std(all_results['tcga_original']['train_cindex']))
            },
            'test_cindex': {
                'mean': float(np.mean(all_results['tcga_original']['test_cindex'])),
                'std': float(np.std(all_results['tcga_original']['test_cindex']))
            }
        }
    }
    
    with open(output_dir / 'results_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Save importance scores
    importance_df = pd.DataFrame({
        'gene': data['gene_names'],
        'orien_original_importance': orien_orig_avg_importance,
        'orien_original_std': np.std(all_results['orien_original']['importance'], axis=0),
        'orien_tcga_like_importance': orien_tcga_avg_importance,
        'orien_tcga_like_std': np.std(all_results['orien_tcga_like']['importance'], axis=0),
        'tcga_importance': tcga_avg_importance,
        'tcga_std': np.std(all_results['tcga_original']['importance'], axis=0)
    })
    
    # Add ranks
    importance_df['orien_original_rank'] = importance_df['orien_original_importance'].rank(ascending=False)
    importance_df['orien_tcga_like_rank'] = importance_df['orien_tcga_like_importance'].rank(ascending=False)
    importance_df['tcga_rank'] = importance_df['tcga_importance'].rank(ascending=False)
    
    importance_df.to_csv(output_dir / 'importance_scores.csv', index=False)
    
    logger.info(f"\n{'='*70}")
    logger.info("EXPERIMENT COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"  - results_summary.json: Summary statistics")
    logger.info(f"  - importance_scores.csv: Gene-level importance scores")
    logger.info(f"  - importance_comparison.png: Visualization")
    
    # Print key comparison
    logger.info(f"\n{'='*70}")
    logger.info("KEY COMPARISON: Importance Score Range")
    logger.info(f"{'='*70}")
    logger.info(f"  ORIEN Original (3-layer): {stats_orig['range']:.4f}")
    logger.info(f"  ORIEN TCGA-like (1-layer): {stats_tcga_like['range']:.4f}")
    logger.info(f"  TCGA Original (1-layer): {stats_tcga['range']:.4f}")
    
    improvement = stats_tcga_like['range'] / stats_orig['range']
    logger.info(f"\n  Improvement factor: {improvement:.2f}x")
    
    if improvement > 1.5:
        logger.info("  ✓ TCGA-like architecture significantly improves importance discrimination!")
    elif improvement > 1.0:
        logger.info("  ~ TCGA-like architecture slightly improves importance discrimination")
    else:
        logger.info("  ✗ TCGA-like architecture does not improve importance discrimination")
    
    return results_summary


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Diagnostic experiment: Architecture effect on importance scores'
    )
    parser.add_argument('--consensus_genes', type=str, 
                        default='data/raw/consensus_genes_308.txt',
                        help='Path to consensus genes file')
    parser.add_argument('--output_dir', type=str,
                        default='results_v2/diagnostics/architecture_comparison',
                        help='Output directory')
    parser.add_argument('--seeds', type=int, nargs='+',
                        default=SEEDS,
                        help='Random seeds')
    
    args = parser.parse_args()
    
    results = run_experiment(
        consensus_genes_file=args.consensus_genes,
        output_dir=args.output_dir,
        seeds=args.seeds
    )
