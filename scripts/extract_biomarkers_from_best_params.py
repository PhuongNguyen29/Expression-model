"""
IMPROVED: Extract biomarkers using validation-based epoch determination.

This script:
1. Fixes trainer bug with valid_loader=None
2. Uses 80/20 split to find optimal epochs with early stopping
3. Trains on 100% of data for scaled number of epochs
4. Extracts feature importance
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

from src.data.dataset import SurvivalDataset
from torch.utils.data import DataLoader
from src.models.elastic_deepsurv import ElasticDeepSurv, ElasticDeepSurvTrainer
from src.utils.batch_samplers import StratifiedBatchSampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_architecture(best_params: dict) -> List[int]:
    """
    Parse architecture from various hyperparameter formats.
    
    Handles:
    - Single layer: 'layer1_size' → [64] or [128] or [256]
    - Two layers: 'architecture_2layer' → "256-128" → [256, 128]
    - Three layers: 'architecture_3layer' → "256-128-32" → [256, 128, 32]
    
    Args:
        best_params: Dictionary from best_params.json
        
    Returns:
        List of hidden layer sizes
    """
    # Single layer (TCGA typically uses this)
    if 'layer1_size' in best_params:
        hidden_sizes = [best_params['layer1_size']]
        logger.info(f"  Parsed 1-layer architecture: {hidden_sizes}")
        return hidden_sizes
    
    # Two layers
    if 'architecture_2layer' in best_params:
        hidden_sizes = [int(x) for x in best_params['architecture_2layer'].split('-')]
        logger.info(f"  Parsed 2-layer architecture: {hidden_sizes}")
        return hidden_sizes
    
    # Three layers (ORIEN typically uses this)
    if 'architecture_3layer' in best_params:
        hidden_sizes = [int(x) for x in best_params['architecture_3layer'].split('-')]
        logger.info(f"  Parsed 3-layer architecture: {hidden_sizes}")
        return hidden_sizes
    
    # Fallback (shouldn't reach here if best_params is correct)
    logger.warning("No architecture found in best_params, using default [256, 128]")
    return [256, 128]

def load_consensus_genes(consensus_file: str) -> List[str]:
    """Load consensus genes from file."""
    logger.info(f"Loading consensus genes from: {consensus_file}")
    
    if consensus_file.endswith('.txt'):
        with open(consensus_file, 'r') as f:
            genes = [line.strip() for line in f if line.strip()]
    elif consensus_file.endswith('.csv'):
        df = pd.read_csv(consensus_file)
        genes = df['gene_name'].tolist() if 'gene_name' in df.columns else df.iloc[:, 0].tolist()
    else:
        raise ValueError(f"Unknown file format: {consensus_file}")
    
    logger.info(f"Loaded {len(genes)} consensus genes")
    return genes


def compute_l2_feature_importance(model: ElasticDeepSurv) -> np.ndarray:
    """Compute L2 norm of first layer weights as feature importance."""
    first_layer = model.network[0]
    if not isinstance(first_layer, nn.Linear):
        raise TypeError(f"First layer is {type(first_layer)}, not nn.Linear")
    
    weights = first_layer.weight.data.cpu().numpy()
    importance = np.linalg.norm(weights, axis=0)
    
    logger.info(f"Computed importance for {len(importance)} genes")
    logger.info(f"Range: [{importance.min():.6f}, {importance.max():.6f}]")
    logger.info(f"Mean: {importance.mean():.6f}")
    
    return importance

def create_data_loader(
    dataset,
    events: np.ndarray,
    batch_size: int,
    shuffle: bool = True
):
    """
    Create data loader with appropriate sampling strategy.
    
    CRITICAL FIX: Use simple shuffling for small cohorts (<500 samples)
    to avoid 0-event batches that we encountered in Step 1.
    """
    n_samples = len(events)
    
    if n_samples >= 500:
        # Large cohort: Use stratified sampling
        logger.info(f"    Using StratifiedBatchSampler (n={n_samples})")
        sampler = StratifiedBatchSampler(
            events=events,
            batch_size=batch_size,
            shuffle=shuffle
        )
        loader = DataLoader(dataset, batch_sampler=sampler)
    else:
        # Small cohort: Use simple random shuffling
        logger.info(f"    Using simple random shuffling (n={n_samples})")
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )
    
    return loader

def determine_optimal_epochs(
    expr_standardized: pd.DataFrame,
    surv: pd.DataFrame,
    best_params: dict,
    cohort_name: str,
    seed: int,
    max_epochs: int = 150
) -> Tuple[int, float, int]:
    """
    Use validation set to determine optimal number of epochs.
    
    Args:
        seed: Random seed for train/val split
        
    Returns:
        Tuple of (best_epoch, best_val_cindex, train_batches_per_epoch)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"DETERMINING OPTIMAL EPOCHS - {cohort_name} (Seed {seed})")
    logger.info(f"{'='*60}")
    
    # Split into train/val with specified seed
    train_idx, val_idx = train_test_split(
        np.arange(len(surv)),
        test_size=0.2,
        stratify=surv['event'].values,
        random_state=seed  # ← Use provided seed
    )
    
    train_expr = expr_standardized.iloc[:, train_idx]
    val_expr = expr_standardized.iloc[:, val_idx]
    train_surv = surv.iloc[train_idx]
    val_surv = surv.iloc[val_idx]
    
    logger.info(f"  Train: {len(train_surv)} samples ({train_surv['event'].sum()} events)")
    logger.info(f"  Val: {len(val_surv)} samples ({val_surv['event'].sum()} events)")
    
    # Create datasets
    train_dataset = SurvivalDataset(train_expr, train_surv)
    val_dataset = SurvivalDataset(val_expr, val_surv)
    
    # Get batch size and create loaders
    batch_size = best_params.get('batch_size', 32)
    
    train_loader = create_data_loader(
        train_dataset,
        train_surv['event'].values,
        batch_size,
        shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Build model with CORRECT architecture parsing
    n_features = expr_standardized.shape[0]
    hidden_sizes = parse_architecture(best_params)  # ← CRITICAL FIX
    
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
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"  Device: {device}")
    
    trainer = ElasticDeepSurvTrainer(
        model=model,
        learning_rate=best_params.get('learning_rate', 1e-4),
        weight_decay=0.0,
        device=device
    )
    
    logger.info(f"  Training with validation for up to {max_epochs} epochs...")
    
    # Train with validation
    history = trainer.fit(
        train_loader=train_loader,
        valid_loader=val_loader,
        n_epochs=max_epochs,
        early_stopping_patience=20,
        verbose=False  # Reduce logging
    )
    
    # Get best epoch
    best_epoch = history.get('best_epoch', len(history['train_loss']))
    
    # Get best validation C-index
    cindex_key = 'valid_cindex' if 'valid_cindex' in history else 'valid_c_index'
    if cindex_key in history and len(history[cindex_key]) > 0:
        valid_cindices = [c for c in history[cindex_key] if c is not None]
        best_val_cindex = max(valid_cindices) if valid_cindices else 0.5
    else:
        best_val_cindex = 0.5
    
    logger.info(f"  ✓ Best epoch: {best_epoch}")
    logger.info(f"  ✓ Best validation C-index: {best_val_cindex:.4f}")
    
    return best_epoch, best_val_cindex, len(train_loader)


def train_on_full_dataset(
    expr_standardized: pd.DataFrame,
    surv: pd.DataFrame,
    best_params: dict,
    cohort_name: str,
    target_epochs: int,
    train_batches_per_epoch: int,
    seed: int
) -> Tuple[ElasticDeepSurv, np.ndarray, List[str], float]:
    """
    Train model on 100% of data for scaled number of epochs.
    
    Args:
        seed: Random seed for reproducibility
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING ON FULL {cohort_name} DATASET (Seed {seed})")
    logger.info(f"{'='*60}")
    
    # Set seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create full dataset
    full_dataset = SurvivalDataset(expr_standardized, surv)
    
    batch_size = best_params.get('batch_size', 32)
    
    # Create data loader
    full_loader = create_data_loader(
        full_dataset,
        surv['event'].values,
        batch_size,
        shuffle=True
    )
    
    # Calculate scaled epochs
    full_batches_per_epoch = len(full_loader)
    target_steps = target_epochs * train_batches_per_epoch
    scaled_epochs = max(1, int(target_steps / full_batches_per_epoch))
    
    logger.info(f"  Epoch scaling:")
    logger.info(f"    Target epoch from validation: {target_epochs}")
    logger.info(f"    Train batches/epoch (80%): {train_batches_per_epoch}")
    logger.info(f"    Full batches/epoch (100%): {full_batches_per_epoch}")
    logger.info(f"    Scaled epochs for full data: {scaled_epochs}")
    
    # Build model with CORRECT architecture parsing
    n_features = expr_standardized.shape[0]
    hidden_sizes = parse_architecture(best_params)  # ← CRITICAL FIX
    
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
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    trainer = ElasticDeepSurvTrainer(
        model=model,
        learning_rate=best_params.get('learning_rate', 1e-4),
        weight_decay=0.0,
        device=device
    )
    
    logger.info(f"  Training for {scaled_epochs} epochs (no validation)...")
    
    # Train on full data
    history = trainer.fit(
        train_loader=full_loader,
        valid_loader=None,
        n_epochs=scaled_epochs,
        early_stopping_patience=None,
        verbose=False
    )
    
    # Get final C-index
    cindex_key = 'train_cindex' if 'train_cindex' in history else 'valid_c_index'
    final_cindex = history[cindex_key][-1] if cindex_key in history else 0.5
    
    logger.info(f"  ✓ Final C-index: {final_cindex:.4f}")
    
    # Quality check
    if final_cindex < 0.58:
        logger.warning(f"  ⚠️  LOW C-INDEX: {final_cindex:.4f} - Biomarkers may be unreliable!")
    elif final_cindex < 0.60:
        logger.info(f"  ⚠️  MODERATE C-INDEX: {final_cindex:.4f}")
    else:
        logger.info(f"  ✅ GOOD C-INDEX: {final_cindex:.4f}")
    
    # Extract importance
    logger.info("  Computing feature importance...")
    importance = compute_l2_feature_importance(model)
    gene_names = expr_standardized.index.tolist()
    
    return model, importance, gene_names, final_cindex


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract biomarkers with multi-seed support'
    )
    parser.add_argument('--tcga_params', type=str, required=True,
                        help='Path to TCGA best_params.json from Step 1')
    parser.add_argument('--orien_params', type=str, required=True,
                        help='Path to ORIEN best_params.json from Step 1')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--consensus_genes', type=str, 
                        default='data/raw/consensus_genes_308.txt')
    parser.add_argument('--seeds', type=int, nargs='+', 
                        default=[42, 123, 456, 789, 1011],
                        help='Random seeds for multi-seed validation')
    parser.add_argument('--max_epochs', type=int, default=150)
    
    args = parser.parse_args()
    
    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"results_v2/02_biomarker_discovery/trained_models_{timestamp}"
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"BIOMARKER EXTRACTION WITH MULTI-SEED TRAINING")
    logger.info(f"{'='*70}")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"{'='*70}\n")
    
    # Load consensus genes
    consensus_genes = load_consensus_genes(args.consensus_genes)
    
    # Load hyperparameters
    logger.info("Loading best hyperparameters from Step 1...")
    with open(args.tcga_params, 'r') as f:
        tcga_params = json.load(f)
    with open(args.orien_params, 'r') as f:
        orien_params = json.load(f)
    
    logger.info(f"  TCGA params: {args.tcga_params}")
    logger.info(f"  ORIEN params: {args.orien_params}")
    
    # Load data
    logger.info("\nLoading expression data...")
    tcga_expr = pd.read_csv("data/raw/tcga_batch_corrected_2sv.csv", index_col=0)
    orien_expr = pd.read_csv("data/raw/orien_batch_corrected.csv", index_col=0)
    
    logger.info("Loading survival data...")
    surv_tcga = pd.read_csv("data/processed/surv_tcga_harmonized.csv", index_col=0)
    surv_orien = pd.read_csv("data/processed/surv_orien_harmonized.csv", index_col=0)
    
    # ========================================================================
    # MULTI-SEED TRAINING LOOP
    # ========================================================================
    
    all_results = {
        'tcga': {'models': [], 'importances': [], 'cindices': []},
        'orien': {'models': [], 'importances': [], 'cindices': []}
    }
    
    for seed_idx, seed in enumerate(args.seeds):
        logger.info(f"\n{'#'*70}")
        logger.info(f"# SEED {seed_idx+1}/{len(args.seeds)}: {seed}")
        logger.info(f"{'#'*70}\n")
        
        # ====================================================================
        # TCGA
        # ====================================================================
        logger.info(f"\n{'='*70}")
        logger.info(f"TCGA COHORT - Seed {seed}")
        logger.info(f"{'='*70}")
        
        # Filter to consensus genes
        tcga_genes_available = [g for g in consensus_genes if g in tcga_expr.index]
        tcga_expr_filtered = tcga_expr.loc[tcga_genes_available, :]
        
        # Standardize
        tcga_mean = tcga_expr_filtered.mean(axis=1).values.reshape(-1, 1)
        tcga_std = tcga_expr_filtered.std(axis=1).values.reshape(-1, 1)
        tcga_standardized = pd.DataFrame(
            (tcga_expr_filtered.values - tcga_mean) / (tcga_std + 1e-8),
            index=tcga_expr_filtered.index,
            columns=tcga_expr_filtered.columns
        )
        
        # Determine optimal epochs
        tcga_best_epoch, tcga_val_cindex, tcga_train_batches = determine_optimal_epochs(
            tcga_standardized, surv_tcga, tcga_params, 'TCGA', seed, args.max_epochs
        )
        
        # Train on full data
        tcga_model, tcga_importance, tcga_genes, tcga_final_cindex = train_on_full_dataset(
            tcga_standardized, surv_tcga, tcga_params, 'TCGA',
            tcga_best_epoch, tcga_train_batches, seed
        )
        
        # Store results
        all_results['tcga']['models'].append(tcga_model)
        all_results['tcga']['importances'].append(tcga_importance)
        all_results['tcga']['cindices'].append(tcga_final_cindex)
        
        # ====================================================================
        # ORIEN
        # ====================================================================
        logger.info(f"\n{'='*70}")
        logger.info(f"ORIEN COHORT - Seed {seed}")
        logger.info(f"{'='*70}")
        
        # Filter to consensus genes
        orien_genes_available = [g for g in consensus_genes if g in orien_expr.index]
        orien_expr_filtered = orien_expr.loc[orien_genes_available, :]
        
        # Standardize
        orien_mean = orien_expr_filtered.mean(axis=1).values.reshape(-1, 1)
        orien_std = orien_expr_filtered.std(axis=1).values.reshape(-1, 1)
        orien_standardized = pd.DataFrame(
            (orien_expr_filtered.values - orien_mean) / (orien_std + 1e-8),
            index=orien_expr_filtered.index,
            columns=orien_expr_filtered.columns
        )
        
        # Determine optimal epochs
        orien_best_epoch, orien_val_cindex, orien_train_batches = determine_optimal_epochs(
            orien_standardized, surv_orien, orien_params, 'ORIEN', seed, args.max_epochs
        )
        
        # Train on full data
        orien_model, orien_importance, orien_genes, orien_final_cindex = train_on_full_dataset(
            orien_standardized, surv_orien, orien_params, 'ORIEN',
            orien_best_epoch, orien_train_batches, seed
        )
        
        # Store results
        all_results['orien']['models'].append(orien_model)
        all_results['orien']['importances'].append(orien_importance)
        all_results['orien']['cindices'].append(orien_final_cindex)
        
        # Save per-seed models
        seed_dir = output_dir / f'seed_{seed}'
        seed_dir.mkdir(exist_ok=True)
        torch.save(tcga_model.state_dict(), seed_dir / 'tcga_model.pth')
        torch.save(orien_model.state_dict(), seed_dir / 'orien_model.pth')
        
        logger.info(f"\n✓ Seed {seed} complete:")
        logger.info(f"  TCGA C-index: {tcga_final_cindex:.4f}")
        logger.info(f"  ORIEN C-index: {orien_final_cindex:.4f}")
    
    # ========================================================================
    # AGGREGATE RESULTS ACROSS SEEDS
    # ========================================================================
    
    logger.info(f"\n{'='*70}")
    logger.info("AGGREGATING MULTI-SEED RESULTS")
    logger.info(f"{'='*70}")
    
    # Verify genes match across seeds
    assert tcga_genes == orien_genes, "Gene lists don't match!"
    gene_names = tcga_genes
    
    # Stack importances across seeds
    tcga_importances = np.array(all_results['tcga']['importances'])  # (n_seeds, n_genes)
    orien_importances = np.array(all_results['orien']['importances'])
    
    # Save importance scores for each seed
    for seed_idx, seed in enumerate(args.seeds):
        importance_df = pd.DataFrame({
            'gene_name': gene_names,
            'tcga_importance': tcga_importances[seed_idx],
            'orien_importance': orien_importances[seed_idx],
            'mean_importance': (tcga_importances[seed_idx] + orien_importances[seed_idx]) / 2
        }).sort_values('mean_importance', ascending=False)
        
        seed_dir = output_dir / f'seed_{seed}'
        importance_df.to_csv(seed_dir / 'gene_importances.csv', index=False)
    
    # Aggregate across seeds (mean)
    tcga_importance_mean = tcga_importances.mean(axis=0)
    orien_importance_mean = orien_importances.mean(axis=0)
    
    # Save aggregated importances
    aggregated_df = pd.DataFrame({
        'gene_name': gene_names,
        'tcga_importance_mean': tcga_importance_mean,
        'tcga_importance_std': tcga_importances.std(axis=0),
        'orien_importance_mean': orien_importance_mean,
        'orien_importance_std': orien_importances.std(axis=0),
        'overall_mean': (tcga_importance_mean + orien_importance_mean) / 2
    }).sort_values('overall_mean', ascending=False)
    
    aggregated_df.to_csv(output_dir / 'aggregated_gene_importances.csv', index=False)
    logger.info(f"✓ Saved: {output_dir / 'aggregated_gene_importances.csv'}")
    
    # Create summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'method': 'multi_seed_validation_based_training',
        'n_seeds': len(args.seeds),
        'seeds': args.seeds,
        'n_input_genes': len(consensus_genes),
        'tcga': {
            'n_samples': len(surv_tcga),
            'n_genes': len(tcga_genes),
            'architecture': parse_architecture(tcga_params),
            'cindices': [float(c) for c in all_results['tcga']['cindices']],
            'mean_cindex': float(np.mean(all_results['tcga']['cindices'])),
            'std_cindex': float(np.std(all_results['tcga']['cindices']))
        },
        'orien': {
            'n_samples': len(surv_orien),
            'n_genes': len(orien_genes),
            'architecture': parse_architecture(orien_params),
            'cindices': [float(c) for c in all_results['orien']['cindices']],
            'mean_cindex': float(np.mean(all_results['orien']['cindices'])),
            'std_cindex': float(np.std(all_results['orien']['cindices']))
        }
    }
    
    with open(output_dir / 'SUMMARY.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n{'='*70}")
    logger.info("COMPLETE!")
    logger.info(f"{'='*70}")
    logger.info(f"Results: {output_dir}")
    logger.info(f"\nPerformance Summary (Mean ± SD across {len(args.seeds)} seeds):")
    logger.info(f"  TCGA:  {summary['tcga']['mean_cindex']:.4f} ± {summary['tcga']['std_cindex']:.4f}")
    logger.info(f"  ORIEN: {summary['orien']['mean_cindex']:.4f} ± {summary['orien']['std_cindex']:.4f}")
    logger.info(f"{'='*70}\n")
    
    return summary


if __name__ == "__main__":
    summary = main()