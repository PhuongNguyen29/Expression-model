"""
Final bidirectional evaluation with:
1. Proper preprocessing (no leakage)
2. Magnitude-based feature selection (soft sparsity)
3. Consensus biomarker identification

This script produces your Chapter 3 final results.
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
import yaml

from src.data.preprocessor import GeneExpressionPreprocessor
from src.data.dataset import SurvivalDataset
from torch.utils.data import DataLoader
from src.models.elastic_deepsurv import ElasticDeepSurv, ElasticDeepSurvTrainer
from src.utils.feature_selection import (
    compute_gene_importance_l2,
    compute_bidirectional_consensus,
    compare_with_chapter2_biomarkers,
    get_selected_gene_names
)
from lifelines.utils import concordance_index
from src.utils.batch_samplers import StratifiedBatchSampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_single_direction(
    train_expr_raw: pd.DataFrame,
    test_expr_raw: pd.DataFrame,
    train_surv: pd.DataFrame,
    test_surv: pd.DataFrame,
    best_params: dict,
    config: dict,
    direction_name: str,
    output_dir: str,
    n_epochs: int = 200
) -> dict:
    """
    Evaluate one direction with proper preprocessing.
    
    Args:
        train_expr_raw: RAW training expression data
        test_expr_raw: RAW test expression data  
        train_surv: Training survival data
        test_surv: Test survival data
        best_params: Best hyperparameters from tuning
        config: Preprocessor config
        direction_name: e.g., 'TCGA_to_ORIEN'
        output_dir: Where to save results
        n_epochs: Training epochs
        
    Returns:
        Dictionary with results
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"EVALUATING: {direction_name}")
    logger.info(f"{'='*60}")
    logger.info(f"Training cohort: {train_expr_raw.shape[1]} samples")
    logger.info(f"Test cohort: {test_expr_raw.shape[1]} samples")
    
    # CRITICAL: Fit preprocessor ONLY on training cohort
    preprocessor = GeneExpressionPreprocessor(config)
    train_processed = preprocessor.fit_transform_single_cohort(
        train_expr_raw,
        cohort_name=direction_name.split('_to_')[0]
    )
    test_processed = preprocessor.transform_single_cohort(test_expr_raw)
    
    logger.info(f"After preprocessing: {train_processed.shape[0]} genes")
    
    # Create datasets
    train_dataset = SurvivalDataset(train_processed, train_surv)
    test_dataset = SurvivalDataset(test_processed, test_surv)
    
    # Create data loaders
    batch_size = best_params.get('batch_size', 64)
    train_batch_sampler = StratifiedBatchSampler(
        events=train_surv['event'].values,
        batch_size=batch_size,
        min_events_per_batch=1,
        shuffle=True,
        drop_last=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    logger.info(f"Training batches: {len(train_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")
    
    # Build model with best hyperparameters
    n_features = train_processed.shape[0]
    
    # Parse hidden sizes from best_params
    if 'pattern' in best_params:
        hidden_sizes = [int(x) for x in best_params['pattern'].split('-')]
    elif 'h1' in best_params:
        hidden_sizes = [best_params['h1']]
        if 'h2' in best_params:
            hidden_sizes.append(best_params['h2'])
    else:
        hidden_sizes = [256, 128]  # Fallback
    
    model = ElasticDeepSurv(
        n_features=n_features,
        hidden_sizes=hidden_sizes,
        dropout=best_params.get('dropout', 0.3),
        activation=best_params.get('activation', 'relu'),
        batch_norm=best_params.get('batch_norm', True),
        weight_init=best_params.get('weight_init', 'kaiming_uniform'),
        l1_ratio=best_params.get('l1_ratio', 0.7),
        alpha=best_params.get('alpha', 0.001)
    )
    
    # Train model on 100% of training cohort
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = ElasticDeepSurvTrainer(
        model=model,
        learning_rate=best_params.get('learning_rate', 1e-4),
        weight_decay=0.0,
        device=device
    )
    
    logger.info(f"\nTraining model for {n_epochs} epochs...")
    history = trainer.fit(
        train_loader=train_loader,
        valid_loader=test_loader,  # Use test as validation for monitoring
        n_epochs=n_epochs,
        early_stopping_patience=20,
        verbose=True
    )
    
    # Evaluate on test set
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
            
            all_risks.extend(risks)
            all_times.extend(times)
            all_events.extend(events)
    
    test_cindex = concordance_index(all_times, -np.array(all_risks), all_events)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"RESULTS: {direction_name}")
    logger.info(f"{'='*60}")
    logger.info(f"Test C-index: {test_cindex:.4f}")
    logger.info(f"Training epochs: {len(history['train_loss'])}")
    logger.info(f"Best validation C-index: {max(history['valid_c_index']):.4f}")
    
    # Extract feature importance
    gene_names = train_processed.index.tolist()
    importance_scores = compute_gene_importance_l2(model, method='l2_norm')
    
    # Select top genes (top 5%)
    from src.utils.feature_selection import select_features_percentile
    selected_indices, selected_scores, threshold = select_features_percentile(
        importance_scores, gene_names, percentile=95.0
    )
    
    selected_genes_df = get_selected_gene_names(
        selected_indices, gene_names, importance_scores
    )
    
    # Save results
    direction_dir = Path(output_dir) / direction_name
    direction_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), direction_dir / 'model.pth')
    
    # Save selected genes
    selected_genes_df.to_csv(direction_dir / 'selected_genes.csv', index=False)
    
    # Save all importance scores
    importance_df = pd.DataFrame({
        'gene_name': gene_names,
        'importance': importance_scores
    }).sort_values('importance', ascending=False)
    importance_df.to_csv(direction_dir / 'all_gene_importances.csv', index=False)
    
    # Save metrics
    results = {
        'direction': direction_name,
        'test_cindex': float(test_cindex),
        'n_train_samples': train_expr_raw.shape[1],
        'n_test_samples': test_expr_raw.shape[1],
        'n_genes_after_preprocessing': n_features,
        'n_selected_genes': len(selected_indices),
        'selection_threshold': float(threshold),
        'best_params': best_params,
        'training_history': {
            'final_train_loss': history['train_loss'][-1],
            'best_valid_cindex': max(history['valid_c_index']),
            'n_epochs': len(history['train_loss'])
        }
    }
    
    with open(direction_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {direction_dir}")
    logger.info(f"{'='*60}\n")
    
    return {
        'cindex': test_cindex,
        'importance_scores': importance_scores,
        'gene_names': gene_names,
        'selected_genes': set(selected_genes_df['gene_name'].tolist()),
        'model': model,
        'results': results
    }


def main():
    """
    Run complete bidirectional evaluation.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Final bidirectional evaluation for Chapter 3'
    )
    parser.add_argument(
        '--tcga_params',
        type=str,
        required=True,
        help='Path to TCGA best_params.json from hyperparameter tuning'
    )
    parser.add_argument(
        '--orien_params',
        type=str,
        required=True,
        help='Path to ORIEN best_params.json from hyperparameter tuning'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for results'
    )
    parser.add_argument(
        '--chapter2_genes',
        type=str,
        default=None,
        help='Path to CSV with Chapter 2 consensus genes (optional)'
    )
    parser.add_argument(
        '--n_epochs',
        type=int,
        default=200,
        help='Number of training epochs'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"results/bidirectional_FINAL_{timestamp}"
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    with open('config/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load best hyperparameters
    with open(args.tcga_params, 'r') as f:
        tcga_best_params = json.load(f)
    
    with open(args.orien_params, 'r') as f:
        orien_best_params = json.load(f)
    
    # Load RAW data
    logger.info("Loading RAW expression data...")
    tcga_expr_raw = pd.read_csv("data/raw/tcga_batch_corrected_2sv.csv", index_col=0)
    orien_expr_raw = pd.read_csv("data/raw/orien_batch_corrected.csv", index_col=0)
    
    logger.info("Loading survival data...")
    surv_tcga = pd.read_csv("data/processed/surv_tcga_harmonized.csv", index_col=0)
    surv_orien = pd.read_csv("data/processed/surv_orien_harmonized.csv", index_col=0)
    
    # Evaluate both directions
    logger.info("\n" + "="*60)
    logger.info("STARTING BIDIRECTIONAL EVALUATION")
    logger.info("="*60 + "\n")
    
    # Direction 1: TCGA → ORIEN
    tcga_to_orien = evaluate_single_direction(
        train_expr_raw=tcga_expr_raw,
        test_expr_raw=orien_expr_raw,
        train_surv=surv_tcga,
        test_surv=surv_orien,
        best_params=tcga_best_params,
        config=config,
        direction_name='TCGA_to_ORIEN',
        output_dir=args.output_dir,
        n_epochs=args.n_epochs
    )
    
    # Direction 2: ORIEN → TCGA
    orien_to_tcga = evaluate_single_direction(
        train_expr_raw=orien_expr_raw,
        test_expr_raw=tcga_expr_raw,
        train_surv=surv_orien,
        test_surv=surv_tcga,
        best_params=orien_best_params,
        config=config,
        direction_name='ORIEN_to_TCGA',
        output_dir=args.output_dir,
        n_epochs=args.n_epochs
    )
    
    # Compute bidirectional consensus
    consensus_results = compute_bidirectional_consensus(
        model1_importance=tcga_to_orien['importance_scores'],
        model2_importance=orien_to_tcga['importance_scores'],
        gene_names=tcga_to_orien['gene_names'],  # Should be same as orien_to_tcga
        selection_method='percentile',
        percentile=95.0
    )
    
    # Save consensus genes
    consensus_results['consensus_df'].to_csv(
        Path(args.output_dir) / 'consensus_genes.csv',
        index=False
    )
    
    # Compare with Chapter 2 if provided
    if args.chapter2_genes:
        chapter2_df = pd.read_csv(args.chapter2_genes)
        chapter2_genes = chapter2_df['gene_name'].tolist()
        
        comparison = compare_with_chapter2_biomarkers(
            neural_net_genes=consensus_results['consensus_genes'],
            chapter2_genes=chapter2_genes
        )
        
        with open(Path(args.output_dir) / 'chapter2_comparison.json', 'w') as f:
            json.dump(comparison, f, indent=2)
    
    # Final summary
    summary = {
        'tcga_to_orien_cindex': tcga_to_orien['cindex'],
        'orien_to_tcga_cindex': orien_to_tcga['cindex'],
        'n_genes_tcga_model': len(tcga_to_orien['selected_genes']),
        'n_genes_orien_model': len(orien_to_tcga['selected_genes']),
        'n_consensus_genes': consensus_results['n_consensus'],
        'overlap_rate': consensus_results['overlap_rate'],
        'jaccard_index': consensus_results['jaccard_index']
    }
    
    with open(Path(args.output_dir) / 'FINAL_SUMMARY.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print final results
    logger.info("\n" + "="*60)
    logger.info("FINAL CHAPTER 3 RESULTS")
    logger.info("="*60)
    logger.info(f"TCGA → ORIEN C-index: {tcga_to_orien['cindex']:.4f}")
    logger.info(f"ORIEN → TCGA C-index: {orien_to_tcga['cindex']:.4f}")
    logger.info(f"Consensus genes: {consensus_results['n_consensus']}")
    logger.info(f"Overlap rate: {consensus_results['overlap_rate']:.1%}")
    logger.info("="*60 + "\n")
    
    logger.info(f"All results saved to: {args.output_dir}")
    
    return summary


if __name__ == "__main__":
    summary = main()