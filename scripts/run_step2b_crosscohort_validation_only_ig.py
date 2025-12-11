"""
Cross-Cohort Validation Only (for IG-based k-sweep)

This script:
1. Loads existing hyperparameter tuning results
2. Runs cross-cohort validation for all k-values
3. Generates summary CSV and figures

Use this when hyperparameter tuning is complete but cross-cohort validation
was interrupted or needs to be re-run.

Usage:
    python run_crosscohort_validation_only.py \
        --input_dir results_v2/02b_biomarker_discovery_ig/k_selection_with_tuning \
        --data_dir data
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
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
from sklearn.model_selection import train_test_split

from src.data.preprocessor import GeneExpressionPreprocessor
from src.data.dataset import SurvivalDataset
from torch.utils.data import DataLoader
from src.models.elastic_deepsurv import ElasticDeepSurv, ElasticDeepSurvTrainer
from src.utils.batch_samplers import StratifiedBatchSampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Seeds for multi-seed validation
SEEDS = [42, 123, 456, 789, 1011]


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def train_and_test_direction_single_seed(
    source_cohort: str,
    target_cohort: str,
    source_params: Dict,
    consensus_genes: List[str],
    data_dir: Path,
    config: dict,
    seed: int,
    device: str = None
) -> Dict:
    """
    Train on source (with 80/20 split for early stopping) → Test on target.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    set_seed(seed)
    
    logger.info(f"\n--- Seed {seed}: Training {source_cohort.upper()} → Testing {target_cohort.upper()} ---")
    logger.info(f"Using device: {device}")
    
    # Parse hyperparameters
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
    
    # Get CV-derived epochs
    cv_epochs_info = source_params.get('cv_epochs_info', {})
    cv_derived_epochs = int(cv_epochs_info.get('mean_best_epoch', 100))
    logger.info(f"Using CV-derived epochs: {cv_derived_epochs}")
    
    # Load RAW data
    tcga_expr_raw = pd.read_csv(data_dir / "raw" / "tcga_batch_corrected_2sv.csv", index_col=0)
    orien_expr_raw = pd.read_csv(data_dir / "raw" / "orien_batch_corrected.csv", index_col=0)
    tcga_surv = pd.read_csv(data_dir / "processed" / "surv_tcga_harmonized.csv", index_col=0)
    orien_surv = pd.read_csv(data_dir / "processed" / "surv_orien_harmonized.csv", index_col=0)
    
    # Filter to consensus genes
    tcga_expr_raw = tcga_expr_raw.loc[consensus_genes]
    orien_expr_raw = orien_expr_raw.loc[consensus_genes]
    
    # Determine source and target data
    if source_cohort.lower() == 'tcga':
        source_expr_raw = tcga_expr_raw
        source_surv = tcga_surv
        target_expr_raw = orien_expr_raw
        target_surv = orien_surv
    else:
        source_expr_raw = orien_expr_raw
        source_surv = orien_surv
        target_expr_raw = tcga_expr_raw
        target_surv = tcga_surv
    
    # Split source into 80% train, 20% validation for early stopping
    source_samples = source_expr_raw.columns.tolist()
    source_events = source_surv.loc[source_samples, 'event'].values
    
    train_samples, val_samples = train_test_split(
        source_samples,
        test_size=0.2,
        random_state=seed,
        stratify=source_events
    )
    
    logger.info(f"Source split: {len(train_samples)} train, {len(val_samples)} validation")
    
    # Extract train and validation data
    train_expr_raw = source_expr_raw[train_samples]
    val_expr_raw = source_expr_raw[val_samples]
    train_surv = source_surv.loc[train_samples]
    val_surv = source_surv.loc[val_samples]
    
    # Preprocess: fit on train, transform val and target
    preprocessor = GeneExpressionPreprocessor(config)
    
    train_processed = preprocessor.fit_transform_single_cohort(
        train_expr_raw,
        cohort_name=f'{source_cohort}_train'
    )
    val_processed = preprocessor.transform_single_cohort(val_expr_raw)
    target_processed = preprocessor.transform_single_cohort(target_expr_raw)
    
    # Create datasets
    train_dataset = SurvivalDataset(train_processed, train_surv)
    val_dataset = SurvivalDataset(val_processed, val_surv)
    target_dataset = SurvivalDataset(target_processed, target_surv)
    
    # Create dataloaders
    train_events = train_surv['event'].values
    n_train_samples = len(train_dataset)
    
    if n_train_samples < 400:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
    else:
        train_sampler = StratifiedBatchSampler(
            events=train_events,
            batch_size=batch_size,
            min_events_per_batch=2,
            shuffle=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=0
        )
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Create model
    n_features = train_processed.shape[0]
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
    
    # Create trainer
    trainer = ElasticDeepSurvTrainer(
        model=model,
        learning_rate=learning_rate,
        device=device
    )
    
    # Train with early stopping
    history = trainer.fit(
        train_loader=train_loader,
        valid_loader=val_loader,
        n_epochs=cv_derived_epochs,
        early_stopping_patience=20,
        verbose=False
    )
    
    # Evaluate
    _, _, _, train_cindex = trainer.evaluate(train_loader)
    _, _, _, val_cindex = trainer.evaluate(val_loader)
    _, _, _, test_cindex = trainer.evaluate(target_loader)
    
    logger.info(f"  Train C-index: {train_cindex:.4f}")
    logger.info(f"  Val C-index: {val_cindex:.4f}")
    logger.info(f"  Test C-index (target): {test_cindex:.4f}")
    
    return {
        'seed': seed,
        'source': source_cohort,
        'target': target_cohort,
        'train_cindex': train_cindex,
        'val_cindex': val_cindex,
        'test_cindex': test_cindex,
        'architecture': hidden_sizes,
        'cv_derived_epochs': cv_derived_epochs
    }


def train_and_test_direction_multi_seed(
    source_cohort: str,
    target_cohort: str,
    source_params: Dict,
    consensus_genes: List[str],
    data_dir: Path,
    config: dict,
    seeds: List[int] = SEEDS,
    device: str = None
) -> Dict:
    """Run training/testing across multiple seeds and aggregate results."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Direction: {source_cohort.upper()} → {target_cohort.upper()}")
    logger.info(f"Running {len(seeds)} seeds: {seeds}")
    logger.info(f"{'='*60}")
    
    all_results = []
    
    for seed in seeds:
        try:
            result = train_and_test_direction_single_seed(
                source_cohort=source_cohort,
                target_cohort=target_cohort,
                source_params=source_params,
                consensus_genes=consensus_genes,
                data_dir=data_dir,
                config=config,
                seed=seed,
                device=device
            )
            all_results.append(result)
        except Exception as e:
            logger.error(f"Seed {seed} failed: {e}")
            continue
    
    if not all_results:
        raise RuntimeError(f"All seeds failed for {source_cohort} → {target_cohort}")
    
    # Aggregate
    test_cindices = [r['test_cindex'] for r in all_results]
    
    aggregated = {
        'source': source_cohort,
        'target': target_cohort,
        'n_seeds': len(all_results),
        'test_cindex_mean': float(np.mean(test_cindices)),
        'test_cindex_std': float(np.std(test_cindices)),
        'test_cindices_all': test_cindices,
        'architecture': all_results[0]['architecture'],
        'per_seed_results': all_results
    }
    
    logger.info(f"\nAggregated: {aggregated['test_cindex_mean']:.4f} ± {aggregated['test_cindex_std']:.4f}")
    
    return aggregated


def cross_cohort_validation(
    consensus_genes: List[str],
    tcga_params: Dict,
    orien_params: Dict,
    k: int,
    output_dir: Path,
    data_dir: Path,
    seeds: List[int] = SEEDS
) -> Dict:
    """Cross-cohort validation using optimal hyperparameters with multiple seeds."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Cross-Cohort Validation (k={k}, m={len(consensus_genes)})")
    logger.info(f"{'='*60}")
    
    validation_dir = output_dir / f"k{k:03d}" / "cross_cohort_validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    with open('config/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    config['data']['use_consensus_genes'] = False
    config['data']['min_variance_percentile'] = 0
    config['data']['standardize'] = True
    
    # Direction 1: ORIEN → TCGA
    o2t_results = train_and_test_direction_multi_seed(
        source_cohort='orien',
        target_cohort='tcga',
        source_params=orien_params,
        consensus_genes=consensus_genes,
        data_dir=data_dir,
        config=config,
        seeds=seeds
    )
    
    # Direction 2: TCGA → ORIEN
    t2o_results = train_and_test_direction_multi_seed(
        source_cohort='tcga',
        target_cohort='orien',
        source_params=tcga_params,
        consensus_genes=consensus_genes,
        data_dir=data_dir,
        config=config,
        seeds=seeds
    )
    
    # Calculate statistics
    o2t_mean = o2t_results['test_cindex_mean']
    o2t_std = o2t_results['test_cindex_std']
    t2o_mean = t2o_results['test_cindex_mean']
    t2o_std = t2o_results['test_cindex_std']
    
    mean_bidirectional = (o2t_mean + t2o_mean) / 2
    mean_bidirectional_std = np.sqrt((o2t_std**2 + t2o_std**2) / 4)
    
    summary = {
        'k': k,
        'm': len(consensus_genes),
        'n_seeds': len(seeds),
        'orien_to_tcga': {
            'test_cindex_mean': o2t_mean,
            'test_cindex_std': o2t_std,
            'all_test_cindices': o2t_results['test_cindices_all']
        },
        'tcga_to_orien': {
            'test_cindex_mean': t2o_mean,
            'test_cindex_std': t2o_std,
            'all_test_cindices': t2o_results['test_cindices_all']
        },
        'mean_bidirectional_cindex': mean_bidirectional,
        'mean_bidirectional_std': mean_bidirectional_std,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save results
    with open(validation_dir / 'results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nResults:")
    logger.info(f"  ORIEN → TCGA: {o2t_mean:.4f} ± {o2t_std:.4f}")
    logger.info(f"  TCGA → ORIEN: {t2o_mean:.4f} ± {t2o_std:.4f}")
    logger.info(f"  Mean Bidirectional: {mean_bidirectional:.4f} ± {mean_bidirectional_std:.4f}")
    
    return summary


def create_summary_figures(summary_df: pd.DataFrame, output_dir: Path):
    """Create visualization figures matching the style you showed."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Figure 1: Consensus Gene Count vs k
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(summary_df['k'], summary_df['m'], 'o-', color='#17becf', linewidth=2, markersize=10)
    ax.set_xlabel('k (top genes per cohort)', fontsize=14)
    ax.set_ylabel('m (consensus genes)', fontsize=14)
    ax.set_title('Consensus Gene Count vs k', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add annotations
    for _, row in summary_df.iterrows():
        ax.annotate(f"{int(row['m'])}", 
                    (row['k'], row['m']), 
                    textcoords="offset points", 
                    xytext=(0, 10), 
                    ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'consensus_genes_vs_k.png', dpi=150, bbox_inches='tight')
    plt.savefig(fig_dir / 'consensus_genes_vs_k.pdf', bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {fig_dir / 'consensus_genes_vs_k.png'}")
    
    # Figure 2: Two-panel performance figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left panel: CV C-index by cohort
    ax1 = axes[0]
    ax1.plot(summary_df['k'], summary_df['tcga_cv_cindex'], 
             'o-', label='TCGA CV', color='#1f77b4', linewidth=2, markersize=8)
    ax1.plot(summary_df['k'], summary_df['orien_cv_cindex'], 
             'o-', label='ORIEN CV', color='#ff7f0e', linewidth=2, markersize=8)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('k (top genes per cohort)', fontsize=14)
    ax1.set_ylabel('CV C-index', fontsize=14)
    ax1.set_title('Cross-Validation Performance', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.48, 0.70])
    
    # Right panel: Cross-cohort transfer with error bars
    ax2 = axes[1]
    ax2.errorbar(summary_df['k'], summary_df['orien_to_tcga_mean'], 
                 yerr=summary_df['orien_to_tcga_std'],
                 fmt='o-', label='ORIEN → TCGA', color='#2ca02c', 
                 linewidth=2, markersize=8, capsize=4)
    ax2.errorbar(summary_df['k'], summary_df['tcga_to_orien_mean'], 
                 yerr=summary_df['tcga_to_orien_std'],
                 fmt='s-', label='TCGA → ORIEN', color='#d62728', 
                 linewidth=2, markersize=8, capsize=4)
    ax2.errorbar(summary_df['k'], summary_df['mean_bidirectional_cindex'], 
                 yerr=summary_df['mean_bidirectional_std'],
                 fmt='^-', label='Mean Bidirectional', color='#9467bd', 
                 linewidth=2, markersize=8, capsize=4)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('k (top genes per cohort)', fontsize=14)
    ax2.set_ylabel('Test C-index', fontsize=14)
    ax2.set_title('Cross-Cohort Transfer Performance (mean ± std)', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=11, loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.42, 0.68])
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'k_selection_performance.png', dpi=150, bbox_inches='tight')
    plt.savefig(fig_dir / 'k_selection_performance.pdf', bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {fig_dir / 'k_selection_performance.png'}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run cross-cohort validation only')
    parser.add_argument('--input_dir', type=str, 
                        default='results_v2/02b_biomarker_discovery_ig/k_selection_with_tuning',
                        help='Directory containing tuning results')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Data directory')
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS,
                        help='Random seeds for multi-seed validation')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    data_dir = Path(args.data_dir)
    seeds = args.seeds
    
    logger.info("="*80)
    logger.info("CROSS-COHORT VALIDATION (IG-based k-sweep)")
    logger.info("="*80)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Seeds: {seeds}")
    
    # Find all k-value directories
    k_dirs = sorted([d for d in input_dir.iterdir() 
                     if d.is_dir() and d.name.startswith('k') and d.name[1:].isdigit()])
    
    logger.info(f"Found {len(k_dirs)} k-value directories: {[d.name for d in k_dirs]}")
    
    all_results = []
    
    for k_dir in k_dirs:
        k_str = k_dir.name[1:]
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
            
            tcga_cv = tcga_params.get('best_cv_cindex', 0.0)
            orien_cv = orien_params.get('best_cv_cindex', 0.0)
            logger.info(f"TCGA best CV C-index: {tcga_cv:.4f}")
            logger.info(f"ORIEN best CV C-index: {orien_cv:.4f}")
            
            # Run cross-cohort validation
            validation_results = cross_cohort_validation(
                consensus_genes=consensus_genes,
                tcga_params=tcga_params,
                orien_params=orien_params,
                k=k,
                output_dir=input_dir,
                data_dir=data_dir,
                seeds=seeds
            )
            
            # Compile results
            k_results = {
                'k': k,
                'm': len(consensus_genes),
                'tcga_cv_cindex': tcga_cv,
                'orien_cv_cindex': orien_cv,
                'orien_to_tcga_mean': validation_results['orien_to_tcga']['test_cindex_mean'],
                'orien_to_tcga_std': validation_results['orien_to_tcga']['test_cindex_std'],
                'tcga_to_orien_mean': validation_results['tcga_to_orien']['test_cindex_mean'],
                'tcga_to_orien_std': validation_results['tcga_to_orien']['test_cindex_std'],
                'mean_bidirectional_cindex': validation_results['mean_bidirectional_cindex'],
                'mean_bidirectional_std': validation_results['mean_bidirectional_std']
            }
            
            all_results.append(k_results)
            
        except Exception as e:
            logger.error(f"Error processing k={k}: {e}", exc_info=True)
            continue
    
    # Create summary
    summary_df = pd.DataFrame(all_results)
    summary_df = summary_df.sort_values('k').reset_index(drop=True)
    
    summary_dir = input_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    summary_df.to_csv(summary_dir / 'k_selection_summary.csv', index=False)
    
    logger.info("\n" + "="*80)
    logger.info("K-SELECTION SUMMARY")
    logger.info("="*80)
    logger.info("\n" + summary_df.to_string(index=False))
    
    # Find optimal k
    best_idx = summary_df['mean_bidirectional_cindex'].idxmax()
    best_k = summary_df.loc[best_idx, 'k']
    best_cindex = summary_df.loc[best_idx, 'mean_bidirectional_cindex']
    best_m = summary_df.loc[best_idx, 'm']
    
    logger.info(f"\n*** OPTIMAL: k={best_k} (m={best_m} genes) ***")
    logger.info(f"    Mean bidirectional C-index: {best_cindex:.4f}")
    
    # Save optimal recommendation
    optimal_info = {
        'optimal_k': int(best_k),
        'optimal_m': int(best_m),
        'mean_bidirectional_cindex': float(best_cindex),
        'mean_bidirectional_std': float(summary_df.loc[best_idx, 'mean_bidirectional_std']),
        'orien_to_tcga': float(summary_df.loc[best_idx, 'orien_to_tcga_mean']),
        'tcga_to_orien': float(summary_df.loc[best_idx, 'tcga_to_orien_mean'])
    }
    
    with open(summary_dir / 'optimal_k_recommendation.json', 'w') as f:
        json.dump(optimal_info, f, indent=2)
    
    # Create figures
    if len(summary_df) > 1:
        create_summary_figures(summary_df, input_dir)
    
    logger.info(f"\nResults saved to: {summary_dir}")
    logger.info("="*80)


if __name__ == '__main__':
    main()
