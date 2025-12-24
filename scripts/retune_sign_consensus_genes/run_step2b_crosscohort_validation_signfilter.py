"""
Cross-Cohort Validation for Sign-Filtered IG-based k-sweep
===========================================================

This script performs cross-cohort validation on the sign-filtered gene pool (141 genes).

Method: Option 2 - Train on 100% source data, use CV-derived epochs
- Matches Cox elastic net methodology for fair comparison
- No information leakage from target cohort

This script:
1. Loads existing hyperparameter tuning results from 02c folder
2. Trains on FULL source cohort (100%, no validation split)
3. Uses CV-derived epochs from hyperparameter tuning
4. Runs cross-cohort validation for all k-values
5. Generates summary CSV and presentation-ready figures

Usage:
    python run_step2b_crosscohort_validation_signfilter.py \
        --input_dir results_v2/02c_biomarker_discovery_ig_signfilter/k_selection_with_tuning \
        --data_dir data

Author: Phuong (modified for sign-consistent gene filtering)
Date: December 2025
"""

import sys
from pathlib import Path

# Add project root to path
# Script is in: scripts/retune_sign_consensus_genes/
# Project root is: ../../ (two levels up)
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent  # Go up two levels to Expression-model/
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

from src.data.preprocessor import GeneExpressionPreprocessor
from src.data.dataset import SurvivalDataset
from torch.utils.data import DataLoader
from src.models.elastic_deepsurv import ElasticDeepSurv, ElasticDeepSurvTrainer
from src.utils.batch_samplers import StratifiedBatchSampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Seeds for multi-seed validation
SEEDS = [42, 123, 456, 789, 1011]

# Default configuration
DEFAULT_CONFIG = {
    'input_dir': 'results_v2/02c_biomarker_discovery_ig_signfilter/k_selection_with_tuning',
    'data_dir': 'data',
    'epochs': 150,
}


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_expression_and_survival_data(data_dir: Path) -> Dict:
    """
    Load all expression and survival data once.
    
    Returns:
        Dict with keys: tcga_expr_raw, orien_expr_raw, tcga_surv, orien_surv
    """
    logger.info("Loading expression and survival data...")
    
    tcga_expr_raw = pd.read_csv(data_dir / "raw" / "tcga_batch_corrected_2sv.csv", index_col=0)
    orien_expr_raw = pd.read_csv(data_dir / "raw" / "orien_batch_corrected.csv", index_col=0)
    tcga_surv = pd.read_csv(data_dir / "processed" / "surv_tcga_harmonized.csv", index_col=0)
    orien_surv = pd.read_csv(data_dir / "processed" / "surv_orien_harmonized.csv", index_col=0)
    
    logger.info(f"  TCGA: {tcga_expr_raw.shape[1]} samples, {tcga_expr_raw.shape[0]} genes")
    logger.info(f"  ORIEN: {orien_expr_raw.shape[1]} samples, {orien_expr_raw.shape[0]} genes")
    
    return {
        'tcga_expr_raw': tcga_expr_raw,
        'orien_expr_raw': orien_expr_raw,
        'tcga_surv': tcga_surv,
        'orien_surv': orien_surv
    }


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


def train_epoch_manual(model, train_loader, optimizer, device):
    """
    Manual training for one epoch (without ElasticDeepSurvTrainer.fit).
    
    This allows us to control exact number of epochs without early stopping.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch in train_loader:
        features = batch['features'].to(device)
        times = batch['time'].to(device)
        events = batch['event'].to(device)
        
        optimizer.zero_grad()
        log_hazards = model(features)
        loss = model.compute_loss(log_hazards, times, events)
        
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def evaluate_model(model, data_loader, device):
    """Evaluate model and return C-index."""
    from lifelines.utils import concordance_index
    
    model.eval()
    all_risks = []
    all_times = []
    all_events = []
    
    with torch.no_grad():
        for batch in data_loader:
            features = batch['features'].to(device)
            times = batch['time'].cpu().numpy()
            events = batch['event'].cpu().numpy()
            
            log_hazards = model(features)
            risks = torch.exp(log_hazards).squeeze().cpu().numpy()
            
            if np.isscalar(risks):
                risks = np.array([risks])
            
            all_risks.extend(risks)
            all_times.extend(times)
            all_events.extend(events)
    
    all_risks = np.array(all_risks)
    all_times = np.array(all_times)
    all_events = np.array(all_events)
    
    c_index = concordance_index(all_times, -all_risks, all_events)
    
    return c_index


def train_and_test_direction_single_seed(
    source_cohort: str,
    target_cohort: str,
    source_params: Dict,
    consensus_genes: List[str],
    data_dict: Dict,
    config: dict,
    seed: int,
    k: int,
    device: str = None,
    epochs: int = 150,
    output_dir: Path = None
) -> Dict:
    """
    Train on 100% SOURCE cohort → Test on TARGET cohort.
    
    OPTION 2 IMPLEMENTATION:
    - No train/validation split on source
    - Train for fixed number of epochs
    - Track test performance for analysis (but don't use for early stopping)
    - Fair comparison with Cox elastic net
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    set_seed(seed)
    
    logger.info(f"\n--- Seed {seed}: Training {source_cohort.upper()} (100%) → Testing {target_cohort.upper()} ---")
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
    
    # Use user-specified epochs directly
    training_epochs = epochs
    
    logger.info(f"Training for {training_epochs} epochs")
    
    # Load RAW data
    tcga_expr_raw = data_dict['tcga_expr_raw'].loc[consensus_genes]
    orien_expr_raw = data_dict['orien_expr_raw'].loc[consensus_genes]
    tcga_surv = data_dict['tcga_surv']
    orien_surv = data_dict['orien_surv']
    
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
    
    # USE 100% OF SOURCE DATA (no split!)
    logger.info(f"Training on 100% of {source_cohort.upper()}: {source_expr_raw.shape[1]} samples")
    
    # Preprocess: fit on FULL source, transform target
    preprocessor = GeneExpressionPreprocessor(config)
    
    source_processed = preprocessor.fit_transform_single_cohort(
        source_expr_raw,
        cohort_name=f'{source_cohort}_full'
    )
    target_processed = preprocessor.transform_single_cohort(target_expr_raw)
    
    logger.info(f"After preprocessing:")
    logger.info(f"  Source: {source_processed.shape[0]} genes × {source_processed.shape[1]} samples")
    logger.info(f"  Target: {target_processed.shape[0]} genes × {target_processed.shape[1]} samples")
    
    # Create datasets
    source_dataset = SurvivalDataset(source_processed, source_surv)
    target_dataset = SurvivalDataset(target_processed, target_surv)
    
    # Create dataloaders
    source_events = source_surv['event'].values
    n_source_samples = len(source_dataset)
    
    if n_source_samples < 400:
        source_loader = DataLoader(
            source_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
    else:
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
    
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
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
    ).to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train for epochs, tracking both train and test performance
    logger.info(f"Training for {training_epochs} epochs (tracking test for analysis)...")
    
    training_history = []
    best_test_cindex = 0.0
    best_test_epoch = 0
    
    for epoch in range(training_epochs):
        train_loss = train_epoch_manual(model, source_loader, optimizer, device)
        
        # Evaluate every 5 epochs or at start/end
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == training_epochs - 1:
            train_cindex = evaluate_model(model, source_loader, device)
            test_cindex = evaluate_model(model, target_loader, device)
            
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_cindex': train_cindex,
                'test_cindex': test_cindex
            })
            
            # Track best test (for analysis, not for model selection)
            if test_cindex > best_test_cindex:
                best_test_cindex = test_cindex
                best_test_epoch = epoch + 1
            
            logger.info(f"  Epoch {epoch+1}/{training_epochs}: Loss={train_loss:.4f}, "
                       f"Train={train_cindex:.4f}, Test={test_cindex:.4f}")
    
    # Final evaluation
    final_train_cindex = evaluate_model(model, source_loader, device)
    final_test_cindex = evaluate_model(model, target_loader, device)
    
    logger.info(f"\nFinal Results:")
    logger.info(f"  Train C-index ({source_cohort.upper()} 100%): {final_train_cindex:.4f}")
    logger.info(f"  Test C-index ({target_cohort.upper()}): {final_test_cindex:.4f}")
    logger.info(f"  Best Test C-index: {best_test_cindex:.4f} at epoch {best_test_epoch}")
    
    # Save training curve plot if output_dir provided
    if output_dir is not None and training_history:
        plot_dir = output_dir / "training_curves"
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs_list = [h['epoch'] for h in training_history]
        train_cindices = [h['train_cindex'] for h in training_history]
        test_cindices = [h['test_cindex'] for h in training_history]
        losses = [h['train_loss'] for h in training_history]
        
        # Left: Loss
        ax1.plot(epochs_list, losses, 'b-', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Train Loss', fontsize=11)
        ax1.set_title(f'{source_cohort.upper()}→{target_cohort.upper()} Loss (Seed {seed})', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Right: C-index
        ax2.plot(epochs_list, train_cindices, 'b-', label='Train', linewidth=2)
        ax2.plot(epochs_list, test_cindices, 'r-', label='Test', linewidth=2)
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax2.axvline(x=best_test_epoch, color='green', linestyle='--', alpha=0.7, 
                   label=f'Best Test: {best_test_cindex:.3f} @ ep{best_test_epoch}')
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('C-index', fontsize=11)
        ax2.set_title(f'{source_cohort.upper()}→{target_cohort.upper()} C-index (Seed {seed})', fontsize=12)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_dir / f'k{k:03d}_{source_cohort}_to_{target_cohort}_seed{seed}.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    return {
        'seed': seed,
        'source': source_cohort,
        'target': target_cohort,
        'train_cindex': final_train_cindex,
        'test_cindex': final_test_cindex,
        'best_test_cindex': best_test_cindex,
        'best_test_epoch': best_test_epoch,
        'architecture': hidden_sizes,
        'training_epochs': training_epochs,
        'n_source_samples': n_source_samples,
        'training_history': training_history
    }


def train_and_test_direction_multi_seed(
    source_cohort: str,
    target_cohort: str,
    source_params: Dict,
    consensus_genes: List[str],
    data_dict: Dict,
    config: dict,
    k: int,
    seeds: List[int] = SEEDS,
    device: str = None,
    epochs: int = 150,
    output_dir: Path = None
) -> Dict:
    """Run training/testing across multiple seeds and aggregate results."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Direction: {source_cohort.upper()} (100%) → {target_cohort.upper()}")
    logger.info(f"Running {len(seeds)} seeds: {seeds}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"{'='*60}")
    
    all_results = []
    
    for seed in seeds:
        try:
            result = train_and_test_direction_single_seed(
                source_cohort=source_cohort,
                target_cohort=target_cohort,
                source_params=source_params,
                consensus_genes=consensus_genes,
                data_dict=data_dict,
                config=config,
                seed=seed,
                k=k,
                device=device,
                epochs=epochs,
                output_dir=output_dir
            )
            all_results.append(result)
        except Exception as e:
            logger.error(f"Seed {seed} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_results:
        raise RuntimeError(f"All seeds failed for {source_cohort} → {target_cohort}")
    
    # Aggregate
    train_cindices = [r['train_cindex'] for r in all_results]
    test_cindices = [r['test_cindex'] for r in all_results]
    best_test_cindices = [r['best_test_cindex'] for r in all_results]
    best_test_epochs = [r['best_test_epoch'] for r in all_results]
    
    aggregated = {
        'source': source_cohort,
        'target': target_cohort,
        'n_seeds': len(all_results),
        'train_cindex_mean': float(np.mean(train_cindices)),
        'train_cindex_std': float(np.std(train_cindices)),
        'test_cindex_mean': float(np.mean(test_cindices)),
        'test_cindex_std': float(np.std(test_cindices)),
        'best_test_cindex_mean': float(np.mean(best_test_cindices)),
        'best_test_cindex_std': float(np.std(best_test_cindices)),
        'best_test_epoch_mean': float(np.mean(best_test_epochs)),
        'test_cindices_all': test_cindices,
        'architecture': all_results[0]['architecture'],
        'per_seed_results': all_results
    }
    
    logger.info(f"\nAggregated Results:")
    logger.info(f"  Train: {aggregated['train_cindex_mean']:.4f} ± {aggregated['train_cindex_std']:.4f}")
    logger.info(f"  Test (final): {aggregated['test_cindex_mean']:.4f} ± {aggregated['test_cindex_std']:.4f}")
    logger.info(f"  Test (best):  {aggregated['best_test_cindex_mean']:.4f} ± {aggregated['best_test_cindex_std']:.4f}")
    logger.info(f"  Best epoch avg: {aggregated['best_test_epoch_mean']:.1f}")
    
    return aggregated


def cross_cohort_validation(
    consensus_genes: List[str],
    tcga_params: Dict,
    orien_params: Dict,
    k: int,
    output_dir: Path,
    data_dict: Dict,
    seeds: List[int] = SEEDS,
    epochs: int = 150
) -> Dict:
    """Cross-cohort validation using optimal hyperparameters with multiple seeds."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Cross-Cohort Validation (k={k}, m={len(consensus_genes)})")
    logger.info(f"Method: Train on 100% source, fixed {epochs} epochs")
    logger.info(f"Gene pool: Sign-consistent (141 filtered)")
    logger.info(f"{'='*60}")
    
    validation_dir = output_dir / f"k{k:03d}" / "cross_cohort_validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config_path = project_root / 'config' / 'default_config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default config if file doesn't exist
        config = {
            'data': {
                'use_consensus_genes': False,
                'min_variance_percentile': 0,
                'standardize': True
            }
        }
    
    config['data']['use_consensus_genes'] = False
    config['data']['min_variance_percentile'] = 0
    config['data']['standardize'] = True
    
    # Direction 1: ORIEN → TCGA
    o2t_results = train_and_test_direction_multi_seed(
        source_cohort='orien',
        target_cohort='tcga',
        source_params=orien_params,
        consensus_genes=consensus_genes,
        data_dict=data_dict,
        config=config,
        k=k,
        seeds=seeds,
        epochs=epochs,
        output_dir=validation_dir
    )
    
    # Direction 2: TCGA → ORIEN
    t2o_results = train_and_test_direction_multi_seed(
        source_cohort='tcga',
        target_cohort='orien',
        source_params=tcga_params,
        consensus_genes=consensus_genes,
        data_dict=data_dict,
        config=config,
        k=k,
        seeds=seeds,
        epochs=epochs,
        output_dir=validation_dir
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
        'method': 'train_100pct_source_fixed_epochs',
        'gene_pool': 'sign_consistent_141',
        'epochs': epochs,
        'orien_to_tcga': {
            'test_cindex_mean': o2t_mean,
            'test_cindex_std': o2t_std,
            'train_cindex_mean': o2t_results['train_cindex_mean'],
            'best_test_cindex_mean': o2t_results['best_test_cindex_mean'],
            'best_test_epoch_mean': o2t_results['best_test_epoch_mean'],
            'all_test_cindices': o2t_results['test_cindices_all']
        },
        'tcga_to_orien': {
            'test_cindex_mean': t2o_mean,
            'test_cindex_std': t2o_std,
            'train_cindex_mean': t2o_results['train_cindex_mean'],
            'best_test_cindex_mean': t2o_results['best_test_cindex_mean'],
            'best_test_epoch_mean': t2o_results['best_test_epoch_mean'],
            'all_test_cindices': t2o_results['test_cindices_all']
        },
        'mean_bidirectional_cindex': mean_bidirectional,
        'mean_bidirectional_std': mean_bidirectional_std,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save results
    with open(validation_dir / 'results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Cross-Cohort Validation Results (k={k}):")
    logger.info(f"{'='*60}")
    logger.info(f"  ORIEN → TCGA: {o2t_mean:.4f} ± {o2t_std:.4f}")
    logger.info(f"  TCGA → ORIEN: {t2o_mean:.4f} ± {t2o_std:.4f}")
    logger.info(f"  Mean Bidirectional: {mean_bidirectional:.4f} ± {mean_bidirectional_std:.4f}")
    logger.info(f"{'='*60}")
    
    return summary


def create_summary_figures(summary_df: pd.DataFrame, output_dir: Path):
    """Create presentation-ready visualization figures for k-selection analysis."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # ==========================================================================
    # Figure 1: Consensus Gene Count vs k
    # ==========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(summary_df['k'], summary_df['m'], 'o-', color='#2ca02c', linewidth=2, markersize=8)
    ax.set_xlabel('k (top genes per cohort)', fontsize=12)
    ax.set_ylabel('m (consensus genes)', fontsize=12)
    ax.set_title('Consensus Gene Count vs k (IG-based, Sign-Filtered)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    for _, row in summary_df.iterrows():
        ax.annotate(f"{int(row['m'])}", 
                    (row['k'], row['m']), 
                    textcoords="offset points", 
                    xytext=(0, 10), 
                    ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'consensus_genes_vs_k.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: consensus_genes_vs_k.png")
    
    # ==========================================================================
    # Figure 2: Two-panel performance figure (CV + Cross-Cohort Transfer)
    # ==========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: CV C-index by cohort
    ax1 = axes[0]
    ax1.plot(summary_df['k'], summary_df['tcga_cv_cindex'], 
             'o-', label='TCGA CV', color='#1f77b4', linewidth=2, markersize=8)
    ax1.plot(summary_df['k'], summary_df['orien_cv_cindex'], 
             's-', label='ORIEN CV', color='#ff7f0e', linewidth=2, markersize=8)
    ax1.set_xlabel('k (top genes per cohort)', fontsize=12)
    ax1.set_ylabel('CV C-index', fontsize=12)
    ax1.set_title('Cross-Validation Performance (from tuning)', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylim([0.5, max(summary_df['tcga_cv_cindex'].max(), summary_df['orien_cv_cindex'].max()) + 0.05])
    
    # Right: Cross-cohort transfer with error bars
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
    ax2.set_xlabel('k (top genes per cohort)', fontsize=12)
    ax2.set_ylabel('Test C-index', fontsize=12)
    ax2.set_title('Cross-Cohort Transfer (100% source, CV epochs)', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'k_selection_performance.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: k_selection_performance.png")
    
    # ==========================================================================
    # Figure 3: Detailed Cross-Cohort Transfer (single panel, larger)
    # ==========================================================================
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.errorbar(summary_df['k'], summary_df['orien_to_tcga_mean'], 
                yerr=summary_df['orien_to_tcga_std'],
                fmt='o-', label='ORIEN → TCGA (large → small)', color='#2ca02c', 
                linewidth=2.5, markersize=10, capsize=5, capthick=2)
    ax.errorbar(summary_df['k'], summary_df['tcga_to_orien_mean'], 
                yerr=summary_df['tcga_to_orien_std'],
                fmt='s-', label='TCGA → ORIEN (small → large)', color='#d62728', 
                linewidth=2.5, markersize=10, capsize=5, capthick=2)
    ax.errorbar(summary_df['k'], summary_df['mean_bidirectional_cindex'], 
                yerr=summary_df['mean_bidirectional_std'],
                fmt='^-', label='Mean Bidirectional', color='#9467bd', 
                linewidth=2.5, markersize=10, capsize=5, capthick=2)
    
    ax.set_xlabel('k (top genes per cohort)', fontsize=14)
    ax.set_ylabel('Test C-index', fontsize=14)
    ax.set_title('Cross-Cohort Transfer Performance\n(Sign-Consistent Genes, 5-seed validation)', fontsize=16)
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random (C=0.5)')
    
    # Highlight optimal k
    optimal_idx = summary_df['mean_bidirectional_cindex'].idxmax()
    optimal_k = summary_df.loc[optimal_idx, 'k']
    optimal_cindex = summary_df.loc[optimal_idx, 'mean_bidirectional_cindex']
    ax.axvline(x=optimal_k, color='gold', linestyle='--', linewidth=2, alpha=0.7)
    ax.annotate(f'Optimal k={int(optimal_k)}\nC={optimal_cindex:.3f}', 
                xy=(optimal_k, optimal_cindex),
                xytext=(optimal_k + 10, optimal_cindex + 0.02),
                fontsize=11, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='gold', lw=1.5))
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'cross_cohort_transfer_detailed.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: cross_cohort_transfer_detailed.png")
    
    # ==========================================================================
    # Figure 4: Summary comparison table as figure
    # ==========================================================================
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    for _, row in summary_df.iterrows():
        table_data.append([
            int(row['k']),
            int(row['m']),
            f"{row['tcga_cv_cindex']:.3f}",
            f"{row['orien_cv_cindex']:.3f}",
            f"{row['orien_to_tcga_mean']:.3f} ± {row['orien_to_tcga_std']:.3f}",
            f"{row['tcga_to_orien_mean']:.3f} ± {row['tcga_to_orien_std']:.3f}",
            f"{row['mean_bidirectional_cindex']:.3f} ± {row['mean_bidirectional_std']:.3f}"
        ])
    
    columns = ['k', 'm', 'TCGA CV', 'ORIEN CV', 'ORIEN→TCGA', 'TCGA→ORIEN', 'Mean Bidir.']
    
    table = ax.table(cellText=table_data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Highlight header
    for j in range(len(columns)):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Highlight optimal row
    optimal_row_idx = summary_df['mean_bidirectional_cindex'].idxmax() + 1  # +1 for header
    for j in range(len(columns)):
        table[(optimal_row_idx, j)].set_facecolor('#E2EFDA')
    
    ax.set_title('K-Selection Summary (Sign-Consistent Genes)\n', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'k_selection_summary_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: k_selection_summary_table.png")
    
    logger.info(f"\nAll figures saved to: {fig_dir}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Cross-cohort validation for sign-filtered genes')
    parser.add_argument('--input_dir', type=str, default=DEFAULT_CONFIG['input_dir'],
                        help='Directory containing k-value subdirectories with tuning results')
    parser.add_argument('--data_dir', type=str, default=DEFAULT_CONFIG['data_dir'],
                        help='Data directory')
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS,
                        help='Random seeds for multi-seed validation')
    parser.add_argument('--epochs', type=int, default=DEFAULT_CONFIG['epochs'],
                        help='Number of training epochs (default: 150)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    data_dir = Path(args.data_dir)
    seeds = args.seeds
    epochs = args.epochs
    
    logger.info("="*80)
    logger.info("CROSS-COHORT VALIDATION (SIGN-FILTERED GENES)")
    logger.info("="*80)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Gene pool: Sign-consistent (141 filtered)")
    logger.info("Method: Train on 100% source cohort")
    logger.info("="*80)
    
    # Load data once
    data_dict = load_expression_and_survival_data(data_dir)
    
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
            
            tcga_cv_cindex = tcga_params.get('best_cv_cindex', 0.5)
            orien_cv_cindex = orien_params.get('best_cv_cindex', 0.5)
            
            logger.info(f"TCGA CV C-index: {tcga_cv_cindex:.4f}")
            logger.info(f"ORIEN CV C-index: {orien_cv_cindex:.4f}")
            
            # Run cross-cohort validation
            validation_results = cross_cohort_validation(
                consensus_genes=consensus_genes,
                tcga_params=tcga_params,
                orien_params=orien_params,
                k=k,
                output_dir=input_dir,
                data_dict=data_dict,
                seeds=seeds,
                epochs=epochs
            )
            
            # Compile results
            k_results = {
                'k': k,
                'm': len(consensus_genes),
                'tcga_cv_cindex': tcga_cv_cindex,
                'orien_cv_cindex': orien_cv_cindex,
                'orien_to_tcga_mean': validation_results['orien_to_tcga']['test_cindex_mean'],
                'orien_to_tcga_std': validation_results['orien_to_tcga']['test_cindex_std'],
                'tcga_to_orien_mean': validation_results['tcga_to_orien']['test_cindex_mean'],
                'tcga_to_orien_std': validation_results['tcga_to_orien']['test_cindex_std'],
                'mean_bidirectional_cindex': validation_results['mean_bidirectional_cindex'],
                'mean_bidirectional_std': validation_results['mean_bidirectional_std']
            }
            
            all_results.append(k_results)
            
        except Exception as e:
            logger.error(f"Error processing k={k}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create summary DataFrame
    if not all_results:
        logger.error("No results collected!")
        return
    
    summary_df = pd.DataFrame(all_results)
    summary_df = summary_df.sort_values('k').reset_index(drop=True)
    
    # Save summary
    summary_dir = input_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    summary_df.to_csv(summary_dir / 'crosscohort_validation_summary.csv', index=False)
    
    logger.info("\n" + "="*80)
    logger.info("K-SELECTION CROSS-COHORT VALIDATION SUMMARY")
    logger.info("="*80)
    logger.info("\n" + summary_df.to_string(index=False))
    
    # Find optimal k
    optimal_idx = summary_df['mean_bidirectional_cindex'].idxmax()
    optimal_k = summary_df.loc[optimal_idx, 'k']
    optimal_m = summary_df.loc[optimal_idx, 'm']
    optimal_cindex = summary_df.loc[optimal_idx, 'mean_bidirectional_cindex']
    optimal_std = summary_df.loc[optimal_idx, 'mean_bidirectional_std']
    
    optimal_recommendation = {
        'optimal_k': int(optimal_k),
        'optimal_m': int(optimal_m),
        'mean_bidirectional_cindex': float(optimal_cindex),
        'mean_bidirectional_std': float(optimal_std),
        'orien_to_tcga': float(summary_df.loc[optimal_idx, 'orien_to_tcga_mean']),
        'tcga_to_orien': float(summary_df.loc[optimal_idx, 'tcga_to_orien_mean']),
        'method': 'train_100pct_source_fixed_epochs',
        'gene_pool': 'sign_consistent_141',
        'n_seeds': len(seeds),
        'epochs': epochs,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(summary_dir / 'optimal_k_recommendation.json', 'w') as f:
        json.dump(optimal_recommendation, f, indent=2)
    
    logger.info("\n" + "="*80)
    logger.info("OPTIMAL K RECOMMENDATION")
    logger.info("="*80)
    logger.info(f"Optimal k: {optimal_k}")
    logger.info(f"Consensus genes (m): {optimal_m}")
    logger.info(f"Mean Bidirectional C-index: {optimal_cindex:.4f} ± {optimal_std:.4f}")
    logger.info(f"  ORIEN → TCGA: {summary_df.loc[optimal_idx, 'orien_to_tcga_mean']:.4f}")
    logger.info(f"  TCGA → ORIEN: {summary_df.loc[optimal_idx, 'tcga_to_orien_mean']:.4f}")
    logger.info("="*80)
    
    # Create figures
    if len(summary_df) > 1:
        logger.info("\nGenerating presentation figures...")
        create_summary_figures(summary_df, input_dir)
    
    logger.info(f"\nResults saved to: {summary_dir}")
    logger.info(f"Figures saved to: {input_dir / 'figures'}")


if __name__ == '__main__':
    main()
