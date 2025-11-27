"""
Step 2.2: Cross-Cohort Validation and K-Selection Analysis

This script uses the hyperparameter tuning results from Step 2 to:
1. Run cross-cohort validation (TCGA → ORIEN, ORIEN → TCGA) for each k
2. Generate summary statistics and select optimal k
3. Create visualization figures

Usage:
    python scripts/step2_2_cross_cohort_validation.py \
        --input_dir results_v2/02_biomarker_discovery/k_selection_with_tuning \
        --data_dir data

Based on: Bernau et al. (2014, Bioinformatics) - Cross-study validation
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import json
import yaml
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

from src.data.preprocessor import GeneExpressionPreprocessor
from src.data.dataset import SurvivalDataset
from torch.utils.data import DataLoader
from src.models.elastic_deepsurv import ElasticDeepSurv, ElasticDeepSurvTrainer
from src.utils.batch_samplers import StratifiedBatchSampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_consensus_genes(k_dir: Path) -> List[str]:
    """Load consensus genes for a k-value."""
    gene_file = k_dir / "consensus_genes" / "consensus_genes.txt"
    if not gene_file.exists():
        raise FileNotFoundError(f"Consensus gene file not found: {gene_file}")
    
    with open(gene_file, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]
    
    return genes


def load_best_params(k_dir: Path, cohort: str) -> Dict:
    """Load best hyperparameters for a cohort."""
    params_file = k_dir / "hyperparameter_tuning" / cohort / "best_params.json"
    if not params_file.exists():
        raise FileNotFoundError(f"Best params file not found: {params_file}")
    
    with open(params_file, 'r') as f:
        return json.load(f)


def train_and_test_direction(
    source_cohort: str,
    target_cohort: str,
    source_params: Dict,
    consensus_genes: List[str],
    data_dir: Path,
    config: dict,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict:
    """
    Train on source (full cohort) → Test on target (full cohort).
    
    Args:
        source_cohort: 'tcga' or 'orien'
        target_cohort: 'tcga' or 'orien'
        source_params: Optimal hyperparameters from tuning
        consensus_genes: List of consensus genes
        data_dir: Data directory path
        config: Configuration dict
        device: 'cuda' or 'cpu'
        
    Returns:
        Dict with train and test C-indices
    """
    logger.info(f"\n--- Training: {source_cohort.upper()} → Testing: {target_cohort.upper()} ---")
    
    # Parse hyperparameters from source tuning
    best_params = source_params['best_params']
    
    # Parse architecture
    if 'architecture_2layer' in best_params:
        hidden_sizes = [int(x) for x in best_params['architecture_2layer'].split('-')]
    elif 'architecture_3layer' in best_params:
        hidden_sizes = [int(x) for x in best_params['architecture_3layer'].split('-')]
    elif 'layer1_size' in best_params:
        hidden_sizes = [best_params['layer1_size']]
    else:
        raise ValueError(f"Cannot parse architecture from {best_params}")
    
    dropout = best_params['dropout']
    learning_rate = best_params['learning_rate']
    alpha = best_params['alpha']
    l1_ratio = best_params['l1_ratio']
    batch_size = best_params['batch_size']
    activation = best_params['activation']
    batch_norm = best_params['batch_norm']
    weight_init = best_params.get('weight_init', 'kaiming_normal')
    
    logger.info(f"Using architecture: {hidden_sizes}")
    logger.info(f"Using hyperparameters from {source_cohort} tuning")
    
    # Load RAW data for both cohorts
    tcga_expr_raw = pd.read_csv(data_dir / "raw" / "tcga_batch_corrected_2sv.csv", index_col=0)
    orien_expr_raw = pd.read_csv(data_dir / "raw" / "orien_batch_corrected.csv", index_col=0)
    tcga_surv = pd.read_csv(data_dir / "processed" / "surv_tcga_harmonized.csv", index_col=0)
    orien_surv = pd.read_csv(data_dir / "processed" / "surv_orien_harmonized.csv", index_col=0)
    
    # Filter to consensus genes
    tcga_expr_raw = tcga_expr_raw.loc[consensus_genes]
    orien_expr_raw = orien_expr_raw.loc[consensus_genes]
    
    logger.info(f"TCGA: {tcga_expr_raw.shape[0]} genes × {tcga_expr_raw.shape[1]} samples")
    logger.info(f"ORIEN: {orien_expr_raw.shape[0]} genes × {orien_expr_raw.shape[1]} samples")
    
    # Preprocess BOTH cohorts
    preprocessor = GeneExpressionPreprocessor(config)
    
    if source_cohort.lower() == 'tcga':
        source_processed = preprocessor.fit_transform_single_cohort(
            tcga_expr_raw,
            cohort_name='TCGA'
        )
        target_processed = preprocessor.transform_single_cohort(orien_expr_raw)
        source_surv = tcga_surv
        target_surv = orien_surv
    else:  # source is ORIEN
        source_processed = preprocessor.fit_transform_single_cohort(
            orien_expr_raw,
            cohort_name='ORIEN'
        )
        target_processed = preprocessor.transform_single_cohort(tcga_expr_raw)
        source_surv = orien_surv
        target_surv = tcga_surv
    
    logger.info(f"After preprocessing:")
    logger.info(f"  Source: {source_processed.shape[0]} genes × {source_processed.shape[1]} samples")
    logger.info(f"  Target: {target_processed.shape[0]} genes × {target_processed.shape[1]} samples")
    
    # Create datasets
    source_dataset = SurvivalDataset(source_processed, source_surv)
    target_dataset = SurvivalDataset(target_processed, target_surv)
    
    # Create dataloaders - use simple shuffle for small datasets
    n_source_samples = len(source_dataset)
    
    if n_source_samples < 500:  # TCGA - use simple shuffle
        source_loader = DataLoader(
            source_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
    else:  # ORIEN - use stratified sampling
        source_events = source_surv['event'].values
        source_sampler = StratifiedBatchSampler(
            events=source_events,
            batch_size=batch_size,
            min_events_per_batch=2,
            shuffle=True
        )
        source_loader = DataLoader(
            source_dataset,
            batch_sampler=source_sampler,
            num_workers=0
        )
    
    target_loader = DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    n_features = source_processed.shape[0]
    model = ElasticDeepSurv(
        n_features=n_features,
        hidden_sizes=hidden_sizes,
        dropout=dropout,
        activation=activation,
        batch_norm=batch_norm,
        weight_init=weight_init,
        l1_ratio=l1_ratio,
        alpha=alpha
    )
    
    logger.info(f"Model created: {n_features} → {hidden_sizes} → 1")
    
    # Create trainer
    trainer = ElasticDeepSurvTrainer(
        model=model,
        learning_rate=learning_rate,
        device=device
    )
    
    # Train on source (full cohort)
    logger.info(f"Training on {source_cohort.upper()} (full cohort: {len(source_dataset)} samples)...")
    
    history = trainer.fit(
        train_loader=source_loader,
        valid_loader=None,  # No validation set
        n_epochs=100,
        early_stopping_patience=None,  # No early stopping
        verbose=False
    )
    
    # Get training performance
    _, _, _, train_cindex = trainer.evaluate(source_loader)
    logger.info(f"  Training C-index on {source_cohort.upper()}: {train_cindex:.4f}")
    
    # Test on target
    logger.info(f"Testing on {target_cohort.upper()} (full cohort: {len(target_dataset)} samples)...")
    _, _, _, test_cindex = trainer.evaluate(target_loader)
    logger.info(f"  Test C-index on {target_cohort.upper()}: {test_cindex:.4f}")
    
    return {
        'source': source_cohort,
        'target': target_cohort,
        'train_cindex': train_cindex,
        'test_cindex': test_cindex,
        'architecture': hidden_sizes,
        'n_source_samples': len(source_dataset),
        'n_target_samples': len(target_dataset)
    }


def cross_cohort_validation(
    consensus_genes: List[str],
    tcga_params: Dict,
    orien_params: Dict,
    k: int,
    output_dir: Path,
    data_dir: Path
) -> Dict:
    """
    Cross-cohort validation using optimal hyperparameters.
    
    Performs bidirectional validation:
    1. Train ORIEN → Test TCGA
    2. Train TCGA → Test ORIEN
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Cross-Cohort Validation (k={k}, m={len(consensus_genes)})")
    logger.info(f"{'='*60}")
    
    # Create output directory
    validation_dir = output_dir / f"k{k:03d}" / "cross_cohort_validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    with open('config/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config - disable variance filter for pre-selected genes
    config['data']['use_consensus_genes'] = False
    config['data']['min_variance_percentile'] = 0
    config['data']['standardize'] = True
    
    # Direction 1: ORIEN → TCGA
    logger.info("\n" + "="*60)
    logger.info("Direction 1: ORIEN → TCGA")
    logger.info("="*60)
    
    o2t_results = train_and_test_direction(
        source_cohort='orien',
        target_cohort='tcga',
        source_params=orien_params,
        consensus_genes=consensus_genes,
        data_dir=data_dir,
        config=config
    )
    
    # Direction 2: TCGA → ORIEN
    logger.info("\n" + "="*60)
    logger.info("Direction 2: TCGA → ORIEN")
    logger.info("="*60)
    
    t2o_results = train_and_test_direction(
        source_cohort='tcga',
        target_cohort='orien',
        source_params=tcga_params,
        consensus_genes=consensus_genes,
        data_dir=data_dir,
        config=config
    )
    
    # Calculate bidirectional statistics
    o2t_cindex = o2t_results['test_cindex']
    t2o_cindex = t2o_results['test_cindex']
    mean_cindex = (o2t_cindex + t2o_cindex) / 2
    
    # Compile results
    summary = {
        'k': k,
        'm': len(consensus_genes),
        'orien_to_tcga': {
            'train_cindex': o2t_results['train_cindex'],
            'test_cindex': o2t_results['test_cindex'],
            'architecture': o2t_results['architecture']
        },
        'tcga_to_orien': {
            'train_cindex': t2o_results['train_cindex'],
            'test_cindex': t2o_results['test_cindex'],
            'architecture': t2o_results['architecture']
        },
        'orien_to_tcga_cindex': o2t_cindex,
        'tcga_to_orien_cindex': t2o_cindex,
        'mean_bidirectional_cindex': mean_cindex,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save results
    with open(validation_dir / 'results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info("Cross-Cohort Validation Results:")
    logger.info(f"{'='*60}")
    logger.info(f"  ORIEN → TCGA:")
    logger.info(f"    Train C-index: {o2t_results['train_cindex']:.4f}")
    logger.info(f"    Test C-index:  {o2t_cindex:.4f}")
    logger.info(f"  TCGA → ORIEN:")
    logger.info(f"    Train C-index: {t2o_results['train_cindex']:.4f}")
    logger.info(f"    Test C-index:  {t2o_cindex:.4f}")
    logger.info(f"  Mean Bidirectional: {mean_cindex:.4f}")
    logger.info(f"{'='*60}\n")
    logger.info(f"Results saved to: {validation_dir}")
    
    return summary


def create_summary_figures(summary_df: pd.DataFrame, output_dir: Path):
    """Create visualization figures for k-selection analysis."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Figure 1: C-index vs k-value
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: CV C-index by cohort
    ax1 = axes[0]
    ax1.plot(summary_df['k'], summary_df['tcga_cv_cindex'], 
             'o-', label='TCGA CV', color='#1f77b4', linewidth=2, markersize=8)
    ax1.plot(summary_df['k'], summary_df['orien_cv_cindex'], 
             's-', label='ORIEN CV', color='#ff7f0e', linewidth=2, markersize=8)
    ax1.set_xlabel('k (top genes per cohort)', fontsize=12)
    ax1.set_ylabel('CV C-index', fontsize=12)
    ax1.set_title('Cross-Validation Performance', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    
    # Right: Cross-cohort transfer
    ax2 = axes[1]
    ax2.plot(summary_df['k'], summary_df['orien_to_tcga_cindex'], 
             'o-', label='ORIEN → TCGA', color='#2ca02c', linewidth=2, markersize=8)
    ax2.plot(summary_df['k'], summary_df['tcga_to_orien_cindex'], 
             's-', label='TCGA → ORIEN', color='#d62728', linewidth=2, markersize=8)
    ax2.plot(summary_df['k'], summary_df['mean_bidirectional_cindex'], 
             '^-', label='Mean Bidirectional', color='#9467bd', linewidth=2, markersize=8)
    ax2.set_xlabel('k (top genes per cohort)', fontsize=12)
    ax2.set_ylabel('Test C-index', fontsize=12)
    ax2.set_title('Cross-Cohort Transfer Performance', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'k_selection_performance.png', dpi=150, bbox_inches='tight')
    plt.savefig(fig_dir / 'k_selection_performance.pdf', bbox_inches='tight')
    plt.close()
    
    # Figure 2: Consensus genes (m) vs k
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(summary_df['k'], summary_df['m'], 'o-', color='#17becf', linewidth=2, markersize=8)
    ax.set_xlabel('k (top genes per cohort)', fontsize=12)
    ax.set_ylabel('m (consensus genes)', fontsize=12)
    ax.set_title('Consensus Gene Count vs k', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add annotations
    for _, row in summary_df.iterrows():
        ax.annotate(f"{int(row['m'])}", 
                    (row['k'], row['m']), 
                    textcoords="offset points", 
                    xytext=(0, 10), 
                    ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'consensus_genes_vs_k.png', dpi=150, bbox_inches='tight')
    plt.savefig(fig_dir / 'consensus_genes_vs_k.pdf', bbox_inches='tight')
    plt.close()
    
    # Figure 3: Comprehensive summary
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top-left: Mean CV C-index
    ax = axes[0, 0]
    mean_cv = (summary_df['tcga_cv_cindex'] + summary_df['orien_cv_cindex']) / 2
    ax.plot(summary_df['k'], mean_cv, 'o-', color='#1f77b4', linewidth=2, markersize=8)
    ax.fill_between(summary_df['k'], 
                    summary_df[['tcga_cv_cindex', 'orien_cv_cindex']].min(axis=1),
                    summary_df[['tcga_cv_cindex', 'orien_cv_cindex']].max(axis=1),
                    alpha=0.3)
    ax.set_xlabel('k', fontsize=11)
    ax.set_ylabel('Mean CV C-index', fontsize=11)
    ax.set_title('A. Within-Cohort CV Performance', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Top-right: Mean bidirectional transfer
    ax = axes[0, 1]
    ax.plot(summary_df['k'], summary_df['mean_bidirectional_cindex'], 
            'o-', color='#9467bd', linewidth=2, markersize=8)
    ax.fill_between(summary_df['k'],
                    summary_df[['orien_to_tcga_cindex', 'tcga_to_orien_cindex']].min(axis=1),
                    summary_df[['orien_to_tcga_cindex', 'tcga_to_orien_cindex']].max(axis=1),
                    alpha=0.3, color='#9467bd')
    ax.set_xlabel('k', fontsize=11)
    ax.set_ylabel('Mean Bidirectional C-index', fontsize=11)
    ax.set_title('B. Cross-Cohort Transfer Performance', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Bottom-left: Cohort stability (difference between directions)
    ax = axes[1, 0]
    transfer_diff = abs(summary_df['orien_to_tcga_cindex'] - summary_df['tcga_to_orien_cindex'])
    ax.bar(summary_df['k'], transfer_diff, color='#ff7f0e', alpha=0.7)
    ax.set_xlabel('k', fontsize=11)
    ax.set_ylabel('|ORIEN→TCGA - TCGA→ORIEN|', fontsize=11)
    ax.set_title('C. Transfer Direction Stability (lower = more stable)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Bottom-right: Combined score
    ax = axes[1, 1]
    # Normalize metrics to 0-1 scale
    cv_norm = (mean_cv - mean_cv.min()) / (mean_cv.max() - mean_cv.min() + 1e-8)
    transfer_norm = (summary_df['mean_bidirectional_cindex'] - summary_df['mean_bidirectional_cindex'].min()) / \
                    (summary_df['mean_bidirectional_cindex'].max() - summary_df['mean_bidirectional_cindex'].min() + 1e-8)
    stability_norm = 1 - (transfer_diff - transfer_diff.min()) / (transfer_diff.max() - transfer_diff.min() + 1e-8)
    
    # Combined score (equal weights)
    combined_score = (cv_norm + transfer_norm + stability_norm) / 3
    
    ax.plot(summary_df['k'], combined_score, 'o-', color='#2ca02c', linewidth=2, markersize=8)
    best_idx = combined_score.idxmax()
    best_k = summary_df.loc[best_idx, 'k']
    ax.axvline(x=best_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal k={int(best_k)}')
    ax.set_xlabel('k', fontsize=11)
    ax.set_ylabel('Combined Score (normalized)', fontsize=11)
    ax.set_title('D. Combined Performance Score', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'k_selection_comprehensive.png', dpi=150, bbox_inches='tight')
    plt.savefig(fig_dir / 'k_selection_comprehensive.pdf', bbox_inches='tight')
    plt.close()
    
    logger.info(f"Figures saved to: {fig_dir}")


def select_optimal_k(summary_df: pd.DataFrame) -> Dict:
    """
    Select optimal k based on multiple criteria.
    
    Criteria:
    1. Highest mean CV C-index
    2. Highest mean bidirectional transfer C-index
    3. Best stability (smallest difference between transfer directions)
    4. Combined score
    """
    results = {}
    
    # Criterion 1: Best CV performance
    mean_cv = (summary_df['tcga_cv_cindex'] + summary_df['orien_cv_cindex']) / 2
    best_cv_idx = mean_cv.idxmax()
    results['best_cv'] = {
        'k': int(summary_df.loc[best_cv_idx, 'k']),
        'm': int(summary_df.loc[best_cv_idx, 'm']),
        'mean_cv_cindex': float(mean_cv.loc[best_cv_idx]),
        'criterion': 'Highest mean CV C-index'
    }
    
    # Criterion 2: Best transfer performance
    best_transfer_idx = summary_df['mean_bidirectional_cindex'].idxmax()
    results['best_transfer'] = {
        'k': int(summary_df.loc[best_transfer_idx, 'k']),
        'm': int(summary_df.loc[best_transfer_idx, 'm']),
        'mean_bidirectional_cindex': float(summary_df.loc[best_transfer_idx, 'mean_bidirectional_cindex']),
        'criterion': 'Highest mean bidirectional transfer C-index'
    }
    
    # Criterion 3: Best stability
    transfer_diff = abs(summary_df['orien_to_tcga_cindex'] - summary_df['tcga_to_orien_cindex'])
    best_stability_idx = transfer_diff.idxmin()
    results['best_stability'] = {
        'k': int(summary_df.loc[best_stability_idx, 'k']),
        'm': int(summary_df.loc[best_stability_idx, 'm']),
        'transfer_difference': float(transfer_diff.loc[best_stability_idx]),
        'criterion': 'Most stable transfer (smallest direction difference)'
    }
    
    # Criterion 4: Combined score
    cv_norm = (mean_cv - mean_cv.min()) / (mean_cv.max() - mean_cv.min() + 1e-8)
    transfer_norm = (summary_df['mean_bidirectional_cindex'] - summary_df['mean_bidirectional_cindex'].min()) / \
                    (summary_df['mean_bidirectional_cindex'].max() - summary_df['mean_bidirectional_cindex'].min() + 1e-8)
    stability_norm = 1 - (transfer_diff - transfer_diff.min()) / (transfer_diff.max() - transfer_diff.min() + 1e-8)
    
    combined_score = (cv_norm + transfer_norm + stability_norm) / 3
    best_combined_idx = combined_score.idxmax()
    
    results['best_combined'] = {
        'k': int(summary_df.loc[best_combined_idx, 'k']),
        'm': int(summary_df.loc[best_combined_idx, 'm']),
        'combined_score': float(combined_score.loc[best_combined_idx]),
        'mean_cv_cindex': float(mean_cv.loc[best_combined_idx]),
        'mean_bidirectional_cindex': float(summary_df.loc[best_combined_idx, 'mean_bidirectional_cindex']),
        'criterion': 'Combined score (CV + Transfer + Stability)'
    }
    
    # Primary recommendation
    results['recommended'] = results['best_combined'].copy()
    results['recommended']['rationale'] = (
        f"k={results['best_combined']['k']} selected based on combined score "
        f"balancing CV performance, cross-cohort transfer, and stability."
    )
    
    return results


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Cross-cohort validation and k-selection analysis')
    parser.add_argument('--input_dir', type=str, 
                        default='results_v2/02_biomarker_discovery/k_selection_with_tuning',
                        help='Directory containing Step 2 results')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Data directory')
    parser.add_argument('--skip_validation', action='store_true',
                        help='Skip cross-cohort validation (use existing results)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    data_dir = Path(args.data_dir)
    
    logger.info("="*80)
    logger.info("STEP 2.2: CROSS-COHORT VALIDATION AND K-SELECTION ANALYSIS")
    logger.info("="*80)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Data directory: {data_dir}")
    
    # Find all k-value directories
    k_dirs = sorted([d for d in input_dir.iterdir() 
                     if d.is_dir() and d.name.startswith('k') and d.name[1:].isdigit()])
    
    if not k_dirs:
        # Try with 3-digit format (k080, k090, etc.)
        k_dirs = sorted([d for d in input_dir.iterdir() 
                         if d.is_dir() and d.name.startswith('k')])
    
    logger.info(f"Found {len(k_dirs)} k-value directories: {[d.name for d in k_dirs]}")
    
    all_results = []
    
    for k_dir in k_dirs:
        # Parse k value from directory name
        k_str = k_dir.name[1:]  # Remove 'k' prefix
        k = int(k_str)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing k={k}")
        logger.info(f"{'='*60}")
        
        try:
            # Load consensus genes
            consensus_genes = load_consensus_genes(k_dir)
            logger.info(f"Loaded {len(consensus_genes)} consensus genes")
            
            # Load best parameters
            tcga_params = load_best_params(k_dir, 'tcga')
            orien_params = load_best_params(k_dir, 'orien')
            logger.info(f"TCGA best CV C-index: {tcga_params['best_cv_cindex']:.4f}")
            logger.info(f"ORIEN best CV C-index: {orien_params['best_cv_cindex']:.4f}")
            
            # Check if cross-cohort validation already exists
            validation_file = k_dir / "cross_cohort_validation" / "results.json"
            
            if args.skip_validation and validation_file.exists():
                logger.info("Loading existing cross-cohort validation results...")
                with open(validation_file, 'r') as f:
                    validation_results = json.load(f)
            else:
                # Run cross-cohort validation
                validation_results = cross_cohort_validation(
                    consensus_genes=consensus_genes,
                    tcga_params=tcga_params,
                    orien_params=orien_params,
                    k=k,
                    output_dir=input_dir,
                    data_dir=data_dir
                )
            
            # Compile results
            k_results = {
                'k': k,
                'm': len(consensus_genes),
                'tcga_cv_cindex': tcga_params['best_cv_cindex'],
                'orien_cv_cindex': orien_params['best_cv_cindex'],
                'orien_to_tcga_cindex': validation_results['orien_to_tcga_cindex'],
                'tcga_to_orien_cindex': validation_results['tcga_to_orien_cindex'],
                'mean_bidirectional_cindex': validation_results['mean_bidirectional_cindex']
            }
            
            all_results.append(k_results)
            
        except Exception as e:
            logger.error(f"Error processing k={k}: {e}", exc_info=True)
            continue
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(all_results)
    summary_df = summary_df.sort_values('k').reset_index(drop=True)
    
    # Save summary
    summary_dir = input_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    summary_df.to_csv(summary_dir / 'k_selection_summary.csv', index=False)
    
    logger.info("\n" + "="*80)
    logger.info("K-SELECTION SUMMARY")
    logger.info("="*80)
    logger.info("\n" + summary_df.to_string(index=False))
    
    # Select optimal k
    optimal_k_results = select_optimal_k(summary_df)
    
    with open(summary_dir / 'optimal_k_recommendation.json', 'w') as f:
        json.dump(optimal_k_results, f, indent=2)
    
    logger.info("\n" + "="*80)
    logger.info("OPTIMAL K RECOMMENDATION")
    logger.info("="*80)
    logger.info(f"\nBest by CV performance: k={optimal_k_results['best_cv']['k']} "
                f"(mean CV C-index: {optimal_k_results['best_cv']['mean_cv_cindex']:.4f})")
    logger.info(f"Best by transfer performance: k={optimal_k_results['best_transfer']['k']} "
                f"(mean bidirectional: {optimal_k_results['best_transfer']['mean_bidirectional_cindex']:.4f})")
    logger.info(f"Best by stability: k={optimal_k_results['best_stability']['k']} "
                f"(transfer diff: {optimal_k_results['best_stability']['transfer_difference']:.4f})")
    logger.info(f"\n*** RECOMMENDED: k={optimal_k_results['recommended']['k']} ***")
    logger.info(f"    Consensus genes (m): {optimal_k_results['recommended']['m']}")
    logger.info(f"    {optimal_k_results['recommended']['rationale']}")
    
    # Create figures
    if len(summary_df) > 1:
        create_summary_figures(summary_df, input_dir)
    
    logger.info(f"\nResults saved to: {summary_dir}")
    logger.info("="*80)


if __name__ == '__main__':
    main()
