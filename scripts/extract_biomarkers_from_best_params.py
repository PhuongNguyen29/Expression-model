"""
FIXED: Extract biomarkers from ElasticDeepSurv models by retraining with best hyperparameters.

FIXES:
1. Uses consensus_genes_308.txt to filter to 308 genes (matches Chapter 2)
2. Handles None validation loader properly
3. Proper preprocessing with consensus genes

This script:
1. Loads best hyperparameters from Optuna optimization
2. Retrains models on 100% of each cohort using 308 consensus genes
3. Extracts feature importance using L2 norm
4. Identifies top genes and consensus genes
5. Compares with Chapter 2 biomarkers
"""

import sys
sys.path.append('.')

import torch
import pandas as pd
import numpy as np
import logging
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Set

from src.data.preprocessor import GeneExpressionPreprocessor
from src.data.dataset import SurvivalDataset
from torch.utils.data import DataLoader
from src.models.elastic_deepsurv import ElasticDeepSurv, ElasticDeepSurvTrainer
from src.utils.batch_samplers import StratifiedBatchSampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_consensus_genes(consensus_file: str) -> List[str]:
    """Load consensus genes from Chapter 2."""
    logger.info(f"Loading consensus genes from: {consensus_file}")
    
    if consensus_file.endswith('.txt'):
        with open(consensus_file, 'r') as f:
            genes = [line.strip() for line in f if line.strip()]
    elif consensus_file.endswith('.csv'):
        df = pd.read_csv(consensus_file)
        if 'gene_name' in df.columns:
            genes = df['gene_name'].tolist()
        else:
            genes = df.iloc[:, 0].tolist()
    else:
        raise ValueError(f"Unknown file format: {consensus_file}")
    
    logger.info(f"Loaded {len(genes)} consensus genes")
    return genes


def compute_l2_feature_importance(model: ElasticDeepSurv) -> np.ndarray:
    """
    Compute L2 norm of first layer weights as feature importance.
    
    Standard method from Simonyan et al. (2014) "Deep Inside CNNs"
    """
    first_layer = model.network[0]  # First linear layer
    weights = first_layer.weight.data.cpu().numpy()  # Shape: (hidden_size, n_genes)
    
    # L2 norm across output dimension
    importance = np.linalg.norm(weights, axis=0)  # Shape: (n_genes,)
    
    return importance


def train_model_on_cohort(
    expr_raw: pd.DataFrame,
    surv: pd.DataFrame,
    best_params: dict,
    cohort_name: str,
    consensus_genes: List[str],
    n_epochs: int = 150
) -> Tuple[ElasticDeepSurv, np.ndarray, List[str]]:
    """
    Train model on 100% of cohort data using consensus genes only.
    
    Args:
        expr_raw: Raw expression data (genes × samples)
        surv: Survival data (samples × [time, event])
        best_params: Best hyperparameters from Optuna
        cohort_name: 'TCGA' or 'ORIEN'
        consensus_genes: List of genes to use
        n_epochs: Training epochs
        
    Returns:
        Tuple of (trained_model, importance_scores, gene_names)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING {cohort_name} MODEL")
    logger.info(f"{'='*60}")
    logger.info(f"Samples: {expr_raw.shape[1]}")
    logger.info(f"Raw genes: {expr_raw.shape[0]}")
    
    # Filter to consensus genes FIRST
    available_genes = set(expr_raw.index)
    consensus_in_data = [g for g in consensus_genes if g in available_genes]
    
    logger.info(f"Consensus genes available in data: {len(consensus_in_data)}/{len(consensus_genes)}")
    
    if len(consensus_in_data) < len(consensus_genes):
        missing = set(consensus_genes) - set(consensus_in_data)
        logger.warning(f"Missing {len(missing)} genes: {list(missing)[:5]}...")
    
    # Filter expression data to consensus genes
    expr_filtered = expr_raw.loc[consensus_in_data, :]
    
    logger.info(f"Expression matrix after consensus filtering: {expr_filtered.shape}")
    
    # Standardize (per-gene z-score)
    expr_mean = expr_filtered.mean(axis=1).values.reshape(-1, 1)
    expr_std = expr_filtered.std(axis=1).values.reshape(-1, 1)
    expr_standardized = pd.DataFrame(
        (expr_filtered.values - expr_mean) / (expr_std + 1e-8),
        index=expr_filtered.index,
        columns=expr_filtered.columns
)
    
    logger.info(f"Standardized: mean={expr_standardized.values.mean():.4f}, std={expr_standardized.values.std():.4f}")
    
    # Create dataset
    dataset = SurvivalDataset(expr_standardized, surv)
    
    # Create batch sampler
    batch_size = best_params.get('batch_size', 64)
    batch_sampler = StratifiedBatchSampler(
        events=surv['event'].values,
        batch_size=batch_size,
        min_events_per_batch=1,
        shuffle=True,
        drop_last=False
    )
    
    # Create data loader
    data_loader = DataLoader(dataset, batch_sampler=batch_sampler)
    
    logger.info(f"Batches per epoch: {len(data_loader)}")
    
    # Build model
    n_features = expr_standardized.shape[0]
    
    # Parse architecture from best_params
    if 'architecture_2layer' in best_params:
        hidden_sizes = [int(x) for x in best_params['architecture_2layer'].split('-')]
    elif 'architecture_3layer' in best_params:
        hidden_sizes = [int(x) for x in best_params['architecture_3layer'].split('-')]
    elif 'architecture_1layer' in best_params:
        hidden_sizes = [int(best_params['architecture_1layer'])]
    else:
        # Fallback
        hidden_sizes = [256, 128]
        logger.warning(f"Could not parse architecture, using default: {hidden_sizes}")
    
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
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {n_params:,}")
    
    # Train model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Training on: {device}")
    
    trainer = ElasticDeepSurvTrainer(
        model=model,
        learning_rate=best_params.get('learning_rate', 1e-4),
        weight_decay=0.0,
        device=device
    )
    
    logger.info(f"Training for {n_epochs} epochs...")
    
    # FIXED: Pass valid_loader=None explicitly and handle in trainer
    history = trainer.fit(
        train_loader=data_loader,
        valid_loader=None,  # No validation, using 100% data
        n_epochs=n_epochs,
        early_stopping_patience=None,  # Disable early stopping
        verbose=True
    )
    
    logger.info(f"Training complete!")
    logger.info(f"Final train loss: {history['train_loss'][-1]:.4f}")
    
    # Extract feature importance
    logger.info("Computing feature importance (L2 norm)...")
    importance_scores = compute_l2_feature_importance(model)
    
    gene_names = expr_standardized.index.tolist()
    
    return model, importance_scores, gene_names


def select_top_genes(
    importance_scores: np.ndarray,
    gene_names: List[str],
    method: str = 'percentile',
    percentile: float = 95.0,
    top_n: int = None
) -> Tuple[List[str], np.ndarray]:
    """
    Select top genes based on importance scores.
    """
    if method == 'percentile':
        threshold = np.percentile(importance_scores, percentile)
        selected_indices = np.where(importance_scores >= threshold)[0]
    elif method == 'top_n':
        selected_indices = np.argsort(importance_scores)[-top_n:]
    else:
        raise ValueError(f"Unknown method: {method}")
    
    selected_genes = [gene_names[i] for i in selected_indices]
    selected_scores = importance_scores[selected_indices]
    
    # Sort by importance descending
    sort_idx = np.argsort(selected_scores)[::-1]
    selected_genes = [selected_genes[i] for i in sort_idx]
    selected_scores = selected_scores[sort_idx]
    
    return selected_genes, selected_scores


def compute_consensus_genes(
    tcga_genes: List[str],
    orien_genes: List[str]
) -> Dict:
    """
    Compute consensus genes (intersection) between two cohorts.
    """
    tcga_set = set(tcga_genes)
    orien_set = set(orien_genes)
    
    consensus = tcga_set & orien_set  # Intersection
    tcga_only = tcga_set - orien_set
    orien_only = orien_set - tcga_set
    
    # Jaccard index
    union = tcga_set | orien_set
    jaccard = len(consensus) / len(union) if len(union) > 0 else 0.0
    
    # Overlap rate
    overlap_rate = len(consensus) / min(len(tcga_set), len(orien_set)) if min(len(tcga_set), len(orien_set)) > 0 else 0.0
    
    return {
        'consensus_genes': sorted(list(consensus)),
        'n_consensus': len(consensus),
        'n_tcga_only': len(tcga_only),
        'n_orien_only': len(orien_only),
        'jaccard_index': jaccard,
        'overlap_rate': overlap_rate,
        'tcga_only_genes': sorted(list(tcga_only)),
        'orien_only_genes': sorted(list(orien_only))
    }


def compare_with_chapter2(
    neural_consensus: List[str],
    chapter2_genes: List[str]
) -> Dict:
    """
    Compare neural network consensus genes with Chapter 2 Cox genes.
    """
    neural_set = set(neural_consensus)
    chapter2_set = set(chapter2_genes)
    
    overlap = neural_set & chapter2_set
    neural_only = neural_set - chapter2_set
    chapter2_only = chapter2_set - neural_set
    
    jaccard = len(overlap) / len(neural_set | chapter2_set) if len(neural_set | chapter2_set) > 0 else 0.0
    
    return {
        'overlap_genes': sorted(list(overlap)),
        'n_overlap': len(overlap),
        'overlap_percentage': 100.0 * len(overlap) / len(chapter2_set) if len(chapter2_set) > 0 else 0.0,
        'jaccard_index': jaccard,
        'neural_only': sorted(list(neural_only)),
        'chapter2_only': sorted(list(chapter2_only))
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract biomarkers from ElasticDeepSurv models using 308 consensus genes'
    )
    parser.add_argument(
        '--tcga_params',
        type=str,
        required=True,
        help='Path to TCGA best_params.json'
    )
    parser.add_argument(
        '--orien_params',
        type=str,
        required=True,
        help='Path to ORIEN best_params.json'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory'
    )
    parser.add_argument(
        '--consensus_genes',
        type=str,
        default='data/raw/consensus_genes_308.txt',
        help='Path to consensus genes file (308 genes from Chapter 2)'
    )
    parser.add_argument(
        '--selection_method',
        type=str,
        default='percentile',
        choices=['percentile', 'top_n'],
        help='Method for selecting top genes'
    )
    parser.add_argument(
        '--percentile',
        type=float,
        default=95.0,
        help='Percentile threshold (e.g., 95.0 for top 5%)'
    )
    parser.add_argument(
        '--top_n',
        type=int,
        default=20,
        help='Number of top genes (if method=top_n)'
    )
    parser.add_argument(
        '--n_epochs',
        type=int,
        default=150,
        help='Number of training epochs'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"results/biomarker_extraction_{timestamp}"
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load consensus genes
    consensus_genes = load_consensus_genes(args.consensus_genes)
    
    # Load best hyperparameters
    logger.info("Loading best hyperparameters...")
    with open(args.tcga_params, 'r') as f:
        tcga_params = json.load(f)
    
    with open(args.orien_params, 'r') as f:
        orien_params = json.load(f)
    
    logger.info(f"TCGA best params: {tcga_params}")
    logger.info(f"ORIEN best params: {orien_params}")
    
    # Load raw data
    logger.info("\nLoading raw expression data...")
    tcga_expr = pd.read_csv("data/raw/tcga_batch_corrected_2sv.csv", index_col=0)
    orien_expr = pd.read_csv("data/raw/orien_batch_corrected.csv", index_col=0)
    
    logger.info("Loading survival data...")
    surv_tcga = pd.read_csv("data/processed/surv_tcga_harmonized.csv", index_col=0)
    surv_orien = pd.read_csv("data/processed/surv_orien_harmonized.csv", index_col=0)
    
    # Train TCGA model and extract biomarkers
    tcga_model, tcga_importance, tcga_genes = train_model_on_cohort(
        expr_raw=tcga_expr,
        surv=surv_tcga,
        best_params=tcga_params,
        cohort_name='TCGA',
        consensus_genes=consensus_genes,
        n_epochs=args.n_epochs
    )
    
    # Train ORIEN model and extract biomarkers
    orien_model, orien_importance, orien_genes = train_model_on_cohort(
        expr_raw=orien_expr,
        surv=surv_orien,
        best_params=orien_params,
        cohort_name='ORIEN',
        consensus_genes=consensus_genes,
        n_epochs=args.n_epochs
    )
    
    # Verify gene lists match
    assert tcga_genes == orien_genes, "Gene lists don't match after preprocessing!"
    gene_names = tcga_genes
    
    # Save all importance scores
    logger.info("\nSaving importance scores...")
    importance_df = pd.DataFrame({
        'gene_name': gene_names,
        'tcga_importance': tcga_importance,
        'orien_importance': orien_importance,
        'mean_importance': (tcga_importance + orien_importance) / 2
    }).sort_values('mean_importance', ascending=False)
    
    importance_df.to_csv(output_dir / 'all_gene_importances.csv', index=False)
    logger.info(f"Saved: {output_dir / 'all_gene_importances.csv'}")
    
    # Select top genes from each cohort
    logger.info("\nSelecting top genes...")
    
    if args.selection_method == 'percentile':
        tcga_selected, tcga_scores = select_top_genes(
            tcga_importance, gene_names, 
            method='percentile', percentile=args.percentile
        )
        orien_selected, orien_scores = select_top_genes(
            orien_importance, gene_names,
            method='percentile', percentile=args.percentile
        )
        selection_params = f"top {100-args.percentile:.1f}%"
    else:
        tcga_selected, tcga_scores = select_top_genes(
            tcga_importance, gene_names,
            method='top_n', top_n=args.top_n
        )
        orien_selected, orien_scores = select_top_genes(
            orien_importance, gene_names,
            method='top_n', top_n=args.top_n
        )
        selection_params = f"top {args.top_n}"
    
    logger.info(f"TCGA selected genes: {len(tcga_selected)}")
    logger.info(f"ORIEN selected genes: {len(orien_selected)}")
    
    # Save cohort-specific genes
    pd.DataFrame({
        'gene_name': tcga_selected,
        'importance': tcga_scores
    }).to_csv(output_dir / 'tcga_selected_genes.csv', index=False)
    
    pd.DataFrame({
        'gene_name': orien_selected,
        'importance': orien_scores
    }).to_csv(output_dir / 'orien_selected_genes.csv', index=False)
    
    # Compute consensus genes
    logger.info("\nComputing consensus genes (TCGA ∩ ORIEN)...")
    consensus_results = compute_consensus_genes(tcga_selected, orien_selected)
    
    logger.info(f"Consensus genes: {consensus_results['n_consensus']}")
    logger.info(f"TCGA-only genes: {consensus_results['n_tcga_only']}")
    logger.info(f"ORIEN-only genes: {consensus_results['n_orien_only']}")
    logger.info(f"Jaccard index: {consensus_results['jaccard_index']:.3f}")
    logger.info(f"Overlap rate: {consensus_results['overlap_rate']:.1%}")
    
    # Save consensus genes
    if consensus_results['n_consensus'] > 0:
        consensus_df = pd.DataFrame({
            'gene_name': consensus_results['consensus_genes']
        })
        consensus_df.to_csv(output_dir / 'consensus_genes.csv', index=False)
        logger.info(f"Saved: {output_dir / 'consensus_genes.csv'}")
        logger.info(f"Consensus genes: {', '.join(consensus_results['consensus_genes'])}")
    else:
        logger.warning("WARNING: No consensus genes found!")
    
    # Compare with Chapter 2 (which is the input consensus genes)
    logger.info("\nNote: Chapter 2 used these same 308 genes, so we're comparing")
    logger.info("which subset the neural network selected vs Cox regression.")
    
    # Save models
    logger.info("\nSaving trained models...")
    torch.save(tcga_model.state_dict(), output_dir / 'tcga_model.pth')
    torch.save(orien_model.state_dict(), output_dir / 'orien_model.pth')
    
    # Create summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'selection_method': args.selection_method,
        'selection_params': selection_params,
        'n_input_genes': len(gene_names),
        'tcga': {
            'n_samples': 339,
            'n_selected_genes': len(tcga_selected),
            'architecture': tcga_params.get('architecture_2layer', 'unknown'),
            'n_params': sum(p.numel() for p in tcga_model.parameters())
        },
        'orien': {
            'n_samples': 1112,
            'n_selected_genes': len(orien_selected),
            'architecture': orien_params.get('architecture_2layer', 'unknown'),
            'n_params': sum(p.numel() for p in orien_model.parameters())
        },
        'consensus': consensus_results
    }
    
    with open(output_dir / 'SUMMARY.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print final summary
    logger.info("\n" + "="*60)
    logger.info("BIOMARKER EXTRACTION COMPLETE")
    logger.info("="*60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Input genes (Chapter 2 consensus): {len(gene_names)}")
    logger.info(f"TCGA selected: {len(tcga_selected)} genes")
    logger.info(f"ORIEN selected: {len(orien_selected)} genes")
    logger.info(f"Neural network consensus: {consensus_results['n_consensus']} genes")
    logger.info("="*60 + "\n")
    
    logger.info("\n📋 NEXT STEPS:")
    if consensus_results['n_consensus'] > 0:
        logger.info("1. ✅ Good consensus! Proceed to bidirectional validation")
        logger.info(f"2. Use: {output_dir}/consensus_genes.csv")
    else:
        logger.info("1. ⚠️ No consensus genes - this is a valid finding!")
        logger.info("2. Consider: Use all 308 genes for bidirectional validation")
        logger.info("3. Or: Adjust selection threshold and re-run")
    
    return summary


if __name__ == "__main__":
    summary = main()