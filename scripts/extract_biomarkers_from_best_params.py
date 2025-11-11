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


def determine_optimal_epochs(
    expr_standardized: pd.DataFrame,
    surv: pd.DataFrame,
    best_params: dict,
    cohort_name: str,
    max_epochs: int = 150
) -> Tuple[int, float]:
    """
    Use validation set to determine optimal number of epochs.
    
    Returns:
        Tuple of (best_epoch, best_val_cindex)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"DETERMINING OPTIMAL EPOCHS FOR {cohort_name}")
    logger.info(f"{'='*60}")
    
    # Split into train/val
    train_idx, val_idx = train_test_split(
        np.arange(len(surv)),
        test_size=0.2,
        stratify=surv['event'].values,
        random_state=42
    )
    
    train_expr = expr_standardized.iloc[:, train_idx]
    val_expr = expr_standardized.iloc[:, val_idx]
    train_surv = surv.iloc[train_idx]
    val_surv = surv.iloc[val_idx]
    
    logger.info(f"Train: {len(train_surv)} samples ({train_surv['event'].sum()} events)")
    logger.info(f"Val: {len(val_surv)} samples ({val_surv['event'].sum()} events)")
    
    # Create datasets
    train_dataset = SurvivalDataset(train_expr, train_surv)
    val_dataset = SurvivalDataset(val_expr, val_surv)
    
    # Get batch size
    batch_size = best_params.get('batch_size', 32)
    
    # Create stratified train loader
    train_sampler = StratifiedBatchSampler(
        events=train_surv['event'].values,
        batch_size=batch_size,
        shuffle=True
    )
    
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Build model
    n_features = expr_standardized.shape[0]
    hidden_sizes = [best_params['layer1_size']] if 'layer1_size' in best_params else [256]
    
    logger.info(f"Architecture: {n_features} → {' → '.join(map(str, hidden_sizes))} → 1")
    
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
    logger.info(f"Device: {device}")
    
    trainer = ElasticDeepSurvTrainer(
        model=model,
        learning_rate=best_params.get('learning_rate', 1e-4),
        weight_decay=0.0,
        device=device
    )
    
    logger.info(f"Training with validation for up to {max_epochs} epochs...")
    
    # Train with validation
    history = trainer.fit(
        train_loader=train_loader,
        valid_loader=val_loader,
        n_epochs=max_epochs,
        early_stopping_patience=20,
        verbose=True
    )
    
    # Get best epoch
    best_epoch = history.get('best_epoch', len(history['train_loss']))
    
    # Get best validation C-index
    cindex_key = 'valid_cindex' if 'valid_cindex' in history else 'valid_c_index'
    if cindex_key in history and len(history[cindex_key]) > 0:
        # Filter out None values
        valid_cindices = [c for c in history[cindex_key] if c is not None]
        best_val_cindex = max(valid_cindices) if valid_cindices else 0.5
    else:
        best_val_cindex = 0.5
    
    logger.info(f"\n{'='*60}")
    logger.info(f"VALIDATION RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Best validation C-index: {best_val_cindex:.4f}")
    logger.info(f"Train batches per epoch: {len(train_loader)}")
    logger.info(f"{'='*60}\n")
    
    return best_epoch, best_val_cindex, len(train_loader)


def train_on_full_dataset(
    expr_standardized: pd.DataFrame,
    surv: pd.DataFrame,
    best_params: dict,
    cohort_name: str,
    target_epochs: int,
    train_batches_per_epoch: int
) -> Tuple[ElasticDeepSurv, np.ndarray, List[str], float]:
    """
    Train model on 100% of data for scaled number of epochs.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING ON FULL {cohort_name} DATASET")
    logger.info(f"{'='*60}")
    
    # Create full dataset
    full_dataset = SurvivalDataset(expr_standardized, surv)
    
    batch_size = best_params.get('batch_size', 32)
    
    # Create stratified sampler for full data
    full_sampler = StratifiedBatchSampler(
        events=surv['event'].values,
        batch_size=batch_size,
        shuffle=True
    )
    
    full_loader = DataLoader(full_dataset, batch_sampler=full_sampler)
    
    # Calculate scaled epochs
    full_batches_per_epoch = len(full_loader)
    target_steps = target_epochs * train_batches_per_epoch
    scaled_epochs = max(1, int(target_steps / full_batches_per_epoch))
    
    logger.info(f"Epoch scaling calculation:")
    logger.info(f"  Target epoch from validation: {target_epochs}")
    logger.info(f"  Train batches/epoch (80%): {train_batches_per_epoch}")
    logger.info(f"  Full batches/epoch (100%): {full_batches_per_epoch}")
    logger.info(f"  Target update steps: {target_steps}")
    logger.info(f"  Scaled epochs for full data: {scaled_epochs}")
    
    # Build model
    n_features = expr_standardized.shape[0]
    hidden_sizes = [best_params['layer1_size']] if 'layer1_size' in best_params else [256]
    
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
    
    logger.info(f"Training for {scaled_epochs} epochs (no validation)...")
    
    # Train on full data
    history = trainer.fit(
        train_loader=full_loader,
        valid_loader=None,
        n_epochs=scaled_epochs,
        early_stopping_patience=None,
        verbose=True
    )
    
    # Safe history access
    cindex_key = 'train_cindex' if 'train_cindex' in history else 'valid_c_index'
    final_cindex = history[cindex_key][-1] if cindex_key in history else 0.5
    
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Final C-index: {final_cindex:.4f}")
    
    # Quality check
    if final_cindex < 0.58:
        logger.warning(f"⚠️  LOW C-INDEX: {final_cindex:.4f}")
        logger.warning("Biomarkers may be unreliable!")
    elif final_cindex < 0.60:
        logger.info(f"⚠️  MODERATE C-INDEX: {final_cindex:.4f}")
        logger.info("Biomarkers usable with caveats")
    else:
        logger.info(f"✅ GOOD C-INDEX: {final_cindex:.4f}")
    
    logger.info(f"{'='*60}\n")
    
    # Extract importance
    logger.info("Computing feature importance...")
    importance = compute_l2_feature_importance(model)
    gene_names = expr_standardized.index.tolist()
    
    return model, importance, gene_names, final_cindex


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='IMPROVED: Extract biomarkers with validation-based training'
    )
    parser.add_argument('--tcga_params', type=str, required=True)
    parser.add_argument('--orien_params', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--consensus_genes', type=str, default='data/raw/consensus_genes_308.txt')
    parser.add_argument('--max_epochs', type=int, default=150)
    
    args = parser.parse_args()
    
    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"results/biomarker_IMPROVED_{timestamp}"
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load consensus genes
    consensus_genes = load_consensus_genes(args.consensus_genes)
    
    # Load hyperparameters
    logger.info("Loading best hyperparameters...")
    with open(args.tcga_params, 'r') as f:
        tcga_params = json.load(f)
    with open(args.orien_params, 'r') as f:
        orien_params = json.load(f)
    
    # Load data
    logger.info("\nLoading raw expression data...")
    tcga_expr = pd.read_csv("data/raw/tcga_batch_corrected_2sv.csv", index_col=0)
    orien_expr = pd.read_csv("data/raw/orien_batch_corrected.csv", index_col=0)
    
    logger.info("Loading survival data...")
    surv_tcga = pd.read_csv("data/processed/surv_tcga_harmonized.csv", index_col=0)
    surv_orien = pd.read_csv("data/processed/surv_orien_harmonized.csv", index_col=0)
    
    # Process TCGA
    logger.info("\n" + "="*70)
    logger.info("TCGA COHORT")
    logger.info("="*70)
    
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
        tcga_standardized, surv_tcga, tcga_params, 'TCGA', args.max_epochs
    )
    
    # Train on full data
    tcga_model, tcga_importance, tcga_genes, tcga_final_cindex = train_on_full_dataset(
        tcga_standardized, surv_tcga, tcga_params, 'TCGA',
        tcga_best_epoch, tcga_train_batches
    )
    
    # Process ORIEN (same steps)
    logger.info("\n" + "="*70)
    logger.info("ORIEN COHORT")
    logger.info("="*70)
    
    orien_genes_available = [g for g in consensus_genes if g in orien_expr.index]
    orien_expr_filtered = orien_expr.loc[orien_genes_available, :]
    
    orien_mean = orien_expr_filtered.mean(axis=1).values.reshape(-1, 1)
    orien_std = orien_expr_filtered.std(axis=1).values.reshape(-1, 1)
    orien_standardized = pd.DataFrame(
        (orien_expr_filtered.values - orien_mean) / (orien_std + 1e-8),
        index=orien_expr_filtered.index,
        columns=orien_expr_filtered.columns
    )
    
    orien_best_epoch, orien_val_cindex, orien_train_batches = determine_optimal_epochs(
        orien_standardized, surv_orien, orien_params, 'ORIEN', args.max_epochs
    )
    
    orien_model, orien_importance, orien_genes, orien_final_cindex = train_on_full_dataset(
        orien_standardized, surv_orien, orien_params, 'ORIEN',
        orien_best_epoch, orien_train_batches
    )
    
    # Save results
    logger.info("\n" + "="*70)
    logger.info("SAVING RESULTS")
    logger.info("="*70)
    
    # Verify genes match
    assert tcga_genes == orien_genes, "Gene lists don't match!"
    gene_names = tcga_genes
    
    # Save importance scores
    importance_df = pd.DataFrame({
        'gene_name': gene_names,
        'tcga_importance': tcga_importance,
        'orien_importance': orien_importance,
        'mean_importance': (tcga_importance + orien_importance) / 2
    }).sort_values('mean_importance', ascending=False)
    
    importance_df.to_csv(output_dir / 'all_gene_importances.csv', index=False)
    logger.info(f"Saved: {output_dir / 'all_gene_importances.csv'}")
    
    # Save models
    torch.save(tcga_model.state_dict(), output_dir / 'tcga_model.pth')
    torch.save(orien_model.state_dict(), output_dir / 'orien_model.pth')
    
    # Create summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'method': 'validation_based_epoch_scaling',
        'n_input_genes': len(consensus_genes),
        'tcga': {
            'n_samples': len(surv_tcga),
            'n_genes': len(tcga_genes),
            'validation_cindex': float(tcga_val_cindex),
            'best_epoch_from_validation': int(tcga_best_epoch),
            'scaled_epochs_full_data': int(tcga_best_epoch * tcga_train_batches / (len(surv_tcga) // tcga_params.get('batch_size', 32))),
            'final_cindex': float(tcga_final_cindex),
            'n_params': sum(p.numel() for p in tcga_model.parameters())
        },
        'orien': {
            'n_samples': len(surv_orien),
            'n_genes': len(orien_genes),
            'validation_cindex': float(orien_val_cindex),
            'best_epoch_from_validation': int(orien_best_epoch),
            'final_cindex': float(orien_final_cindex),
            'n_params': sum(p.numel() for p in orien_model.parameters())
        }
    }
    
    with open(output_dir / 'SUMMARY.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n{'='*70}")
    logger.info("COMPLETE!")
    logger.info(f"{'='*70}")
    logger.info(f"Results: {output_dir}")
    logger.info(f"\nPerformance Summary:")
    logger.info(f"  TCGA: Val C-index={tcga_val_cindex:.4f} → Full C-index={tcga_final_cindex:.4f}")
    logger.info(f"  ORIEN: Val C-index={orien_val_cindex:.4f} → Full C-index={orien_final_cindex:.4f}")
    logger.info(f"{'='*70}")
    
    return summary


if __name__ == "__main__":
    summary = main()