"""
SIMPLIFIED Bidirectional Validation with Consensus Gene Support

This script:
1. Loads your 7 consensus genes
2. Filters expression data to only those genes
3. Trains models on each cohort
4. Performs bidirectional validation
5. Reports C-index for comparison with Chapter 2

Usage:
    python evaluate_bidirectional_SIMPLE.py \
        --tcga_params path/to/tcga_best_params.json \
        --orien_params path/to/orien_best_params.json \
        --consensus_genes path/to/consensus_genes.csv \
        --output_dir results/bidirectional/
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
import argparse

from src.data.dataset import SurvivalDataset
from torch.utils.data import DataLoader
from src.models.elastic_deepsurv import ElasticDeepSurv, ElasticDeepSurvTrainer
from src.utils.batch_samplers import StratifiedBatchSampler
from lifelines.utils import concordance_index

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


def train_and_test(
    train_expr: pd.DataFrame,
    train_surv: pd.DataFrame,
    test_expr: pd.DataFrame,
    test_surv: pd.DataFrame,
    best_params: dict,
    direction: str,
    n_epochs: int = 150
) -> dict:
    """Train on one cohort, test on another."""
    
    logger.info(f"\n{'='*70}")
    logger.info(f"EVALUATING: {direction}")
    logger.info(f"{'='*70}")
    logger.info(f"Train: {train_expr.shape[1]} samples, {train_expr.shape[0]} genes")
    logger.info(f"Test: {test_expr.shape[1]} samples, {test_expr.shape[0]} genes")
    
    # Create datasets
    train_dataset = SurvivalDataset(train_expr, train_surv)
    test_dataset = SurvivalDataset(test_expr, test_surv)
    
    # Batch sampler
    batch_size = best_params.get('batch_size', 48)
    train_sampler = StratifiedBatchSampler(
        events=train_surv['event'].values,
        batch_size=batch_size,
        min_events_per_batch=1,
        shuffle=True,
        drop_last=False
    )
    
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Build model
    n_features = train_expr.shape[0]
    
    # Parse architecture
    if 'architecture_2layer' in best_params:
        hidden_sizes = [int(x) for x in best_params['architecture_2layer'].split('-')]
    elif 'architecture_3layer' in best_params:
        hidden_sizes = [int(x) for x in best_params['architecture_3layer'].split('-')]
    elif 'architecture_1layer' in best_params:
        hidden_sizes = [int(best_params['architecture_1layer'])]
    else:
        hidden_sizes = [128, 64]
    
    # Adjust architecture if needed (hidden layer can't be larger than input)
    max_hidden = int(n_features * 0.8)
    hidden_sizes = [min(h, max_hidden) for h in hidden_sizes]
    
    logger.info(f"Architecture: {n_features} → {' → '.join(map(str, hidden_sizes))} → 1")
    
    model = ElasticDeepSurv(
        n_features=n_features,
        hidden_sizes=hidden_sizes,
        dropout=best_params.get('dropout', 0.3),
        activation=best_params.get('activation', 'relu'),
        batch_norm=best_params.get('batch_norm', False),
        weight_init=best_params.get('weight_init', 'kaiming_uniform'),
        l1_ratio=best_params.get('l1_ratio', 0.7),
        alpha=best_params.get('alpha', 0.001)
    )
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {n_params:,}")
    
    # Train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    adjusted_lr = best_params.get('learning_rate', 1e-4) * 0.5
    
    trainer = ElasticDeepSurvTrainer(
        model=model,
        learning_rate=adjusted_lr,
        weight_decay=0.0,
        device=device
    )
    
    logger.info(f"Training for {n_epochs} epochs...")
    history = trainer.fit(
        train_loader=train_loader,
        valid_loader=None,
        n_epochs=n_epochs,
        early_stopping_patience=None,
        verbose=False  # Less verbose to avoid clutter
    )
    
    # Print summary every 30 epochs
    for i in range(0, len(history['train_cindex']), 30):
        logger.info(f"Epoch {i+1}: C-index {history['train_cindex'][i]:.4f}")
    logger.info(f"Epoch {len(history['train_cindex'])}: C-index {history['train_cindex'][-1]:.4f}")
    
    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    model.eval()
    all_risks = []
    all_times = []
    all_events = []
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            times = batch['time'].cpu().numpy()
            events = batch['event'].cpu().numpy()
            
            log_hazards = model(features)
            risks = torch.exp(log_hazards).squeeze().cpu().numpy()
            
            all_risks.extend(risks if isinstance(risks, (list, np.ndarray)) else [risks])
            all_times.extend(times)
            all_events.extend(events)
    
    test_cindex = concordance_index(all_times, -np.array(all_risks), all_events)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"RESULTS: {direction}")
    logger.info(f"Test C-index: {test_cindex:.4f}")
    logger.info(f"{'='*70}\n")
    
    return {
        'direction': direction,
        'test_cindex': test_cindex,
        'train_cindex_final': history['train_cindex'][-1],
        'n_epochs': len(history['train_cindex']),
        'n_params': n_params
    }


def main():
    parser = argparse.ArgumentParser(description='Bidirectional validation with consensus genes')
    parser.add_argument('--tcga_params', required=True, help='TCGA best params JSON')
    parser.add_argument('--orien_params', required=True, help='ORIEN best params JSON')
    parser.add_argument('--consensus_genes', default=None, help='Consensus genes CSV (optional)')
    parser.add_argument('--output_dir', default=None, help='Output directory')
    parser.add_argument('--n_epochs', type=int, default=150, help='Training epochs')
    
    args = parser.parse_args()
    
    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"results/bidirectional_{timestamp}"
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load best params
    logger.info("Loading best hyperparameters...")
    with open(args.tcga_params) as f:
        tcga_params = json.load(f)
    with open(args.orien_params) as f:
        orien_params = json.load(f)
    
    # Load expression data
    logger.info("\nLoading expression data...")
    tcga_expr = pd.read_csv("data/raw/tcga_batch_corrected_2sv.csv", index_col=0)
    orien_expr = pd.read_csv("data/raw/orien_batch_corrected.csv", index_col=0)
    
    logger.info(f"TCGA raw: {tcga_expr.shape}")
    logger.info(f"ORIEN raw: {orien_expr.shape}")
    
    # Filter to consensus genes if provided
    if args.consensus_genes:
        logger.info(f"\nFiltering to consensus genes from: {args.consensus_genes}")
        consensus = pd.read_csv(args.consensus_genes)
        gene_list = consensus['gene_name'].tolist()
        logger.info(f"Consensus genes: {len(gene_list)}")
        
        # Filter both datasets
        tcga_available = [g for g in gene_list if g in tcga_expr.index]
        orien_available = [g for g in gene_list if g in orien_expr.index]
        
        common_genes = list(set(tcga_available) & set(orien_available))
        logger.info(f"Genes available in both cohorts: {len(common_genes)}")
        
        tcga_expr = tcga_expr.loc[common_genes, :]
        orien_expr = orien_expr.loc[common_genes, :]
        
        logger.info(f"TCGA filtered: {tcga_expr.shape}")
        logger.info(f"ORIEN filtered: {orien_expr.shape}")
    
    # Standardize (per-gene z-score)
    logger.info("\nStandardizing...")
    tcga_expr = tcga_expr.sub(tcga_expr.mean(axis=1), axis=0).div(tcga_expr.std(axis=1) + 1e-8, axis=0)
    orien_expr = orien_expr.sub(orien_expr.mean(axis=1), axis=0).div(orien_expr.std(axis=1) + 1e-8, axis=0)
    
    # Load survival data
    logger.info("Loading survival data...")
    surv_tcga = pd.read_csv("data/processed/surv_tcga_harmonized.csv", index_col=0)
    surv_orien = pd.read_csv("data/processed/surv_orien_harmonized.csv", index_col=0)
    
    # Direction 1: TCGA → ORIEN
    result1 = train_and_test(
        train_expr=tcga_expr,
        train_surv=surv_tcga,
        test_expr=orien_expr,
        test_surv=surv_orien,
        best_params=tcga_params,
        direction='TCGA_to_ORIEN',
        n_epochs=args.n_epochs
    )
    
    # Direction 2: ORIEN → TCGA
    result2 = train_and_test(
        train_expr=orien_expr,
        train_surv=surv_orien,
        test_expr=tcga_expr,
        test_surv=surv_tcga,
        best_params=orien_params,
        direction='ORIEN_to_TCGA',
        n_epochs=args.n_epochs
    )
    
    # Save results
    summary = {
        'timestamp': datetime.now().isoformat(),
        'n_genes': tcga_expr.shape[0],
        'consensus_genes_file': args.consensus_genes,
        'tcga_to_orien': result1,
        'orien_to_tcga': result2,
        'average_cindex': (result1['test_cindex'] + result2['test_cindex']) / 2
    }
    
    with open(output_dir / 'RESULTS.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print final summary
    logger.info("\n" + "="*70)
    logger.info("FINAL BIDIRECTIONAL VALIDATION RESULTS")
    logger.info("="*70)
    logger.info(f"Genes used: {tcga_expr.shape[0]}")
    logger.info(f"TCGA → ORIEN C-index: {result1['test_cindex']:.4f}")
    logger.info(f"ORIEN → TCGA C-index: {result2['test_cindex']:.4f}")
    logger.info(f"Average C-index: {summary['average_cindex']:.4f}")
    logger.info("="*70)
    logger.info(f"\nResults saved to: {output_dir}/RESULTS.json")
    logger.info("\nCompare with Chapter 2:")
    logger.info("  Cox TCGA → ORIEN: 0.68")
    logger.info("  Cox ORIEN → TCGA: 0.72")
    logger.info("="*70)
    
    return summary


if __name__ == "__main__":
    main()
