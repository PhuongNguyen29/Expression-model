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
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn

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
    try:
        first_layer = model.network[0]
        
        # Verify it's actually a Linear layer
        if not isinstance(first_layer, nn.Linear):
            raise TypeError(f"First layer is {type(first_layer)}, not nn.Linear")
        
    except (IndexError, AttributeError) as e:
        raise ValueError(f"Could not access first layer: {e}")
    
    weights = first_layer.weight.data.cpu().numpy()  # [hidden_size, n_genes]
    
    # Verify shape
    logger.info(f"First layer weights shape: {weights.shape}")
    
    # L2 norm across output dimension (axis=0)
    importance = np.linalg.norm(weights, axis=0)  # [n_genes]
    
    logger.info(f"Computed importance for {len(importance)} genes")
    logger.info(f"Importance range: [{importance.min():.6f}, {importance.max():.6f}]")
    logger.info(f"Mean importance: {importance.mean():.6f}")
    
    return importance


def train_model_on_cohort(
    expr_raw: pd.DataFrame,
    surv: pd.DataFrame,
    best_params: dict,
    cohort_name: str,
    consensus_genes: List[str],
    n_epochs: int = 150
) -> Tuple[ElasticDeepSurv, np.ndarray, List[str], float]:
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
    
    # ============================================================
    # Use ALL data for biomarker extraction (no train/val split)
    # ============================================================
    # Standardize all data (per-gene z-score)
    expr_mean = expr_filtered.mean(axis=1).values.reshape(-1, 1)
    expr_std = expr_filtered.std(axis=1).values.reshape(-1, 1)
    expr_standardized = pd.DataFrame(
        (expr_filtered.values - expr_mean) / (expr_std + 1e-8),
        index=expr_filtered.index,
        columns=expr_filtered.columns
    )
    
    logger.info(f"Standardized: mean={expr_standardized.values.mean():.4f}, std={expr_standardized.values.std():.4f}")
    
    # Create full dataset (all samples)
    full_dataset = SurvivalDataset(expr_standardized, surv)
    
    # Get batch size
    batch_size = best_params.get('batch_size', 32)
    
    # Create data loader (no stratified sampling needed for full data)
    full_loader = DataLoader(
        full_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=False
    )
    
    logger.info(f"Using all {len(full_dataset)} samples")
    logger.info(f"Batches per epoch: {len(full_loader)}")
    
    # Build model
    n_features = expr_standardized.shape[0]
    
    # Parse architecture from best_params
    if 'layer1_size' in best_params:
        hidden_sizes = [best_params['layer1_size']]
    elif 'architecture_2layer' in best_params:
        hidden_sizes = [int(x) for x in best_params['architecture_2layer'].split('-')]
    elif 'architecture_3layer' in best_params:
        hidden_sizes = [int(x) for x in best_params['architecture_3layer'].split('-')]
    else:
        hidden_sizes = [256, 64]
        logger.warning(f"Could not parse architecture, using default: {hidden_sizes}")
    
    logger.info(f"Architecture: {n_features} → {' → '.join(map(str, hidden_sizes))} → 1")
    
    # Get hyperparameters from CV (use directly - no validation split)
    alpha_cv = best_params.get('alpha', 0.001)
    learning_rate = best_params.get('learning_rate', 1e-4)
    
    logger.info(f"\nUsing hyperparameters from CV:")
    logger.info(f"  Lambda (alpha): {alpha_cv:.6f}")
    logger.info(f"  Learning rate: {learning_rate:.6f}")
    logger.info(f"  (Training on 100% data for maximum biomarker stability)")
    
    # Create model
    model = ElasticDeepSurv(
        n_features=n_features,
        hidden_sizes=hidden_sizes,
        dropout=best_params.get('dropout', 0.3),
        activation=best_params.get('activation', 'relu'),
        batch_norm=best_params.get('batch_norm', False),
        weight_init=best_params.get('weight_init', 'kaiming_uniform'),
        l1_ratio=best_params.get('l1_ratio', 0.7),
        alpha=alpha_cv
    )
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {n_params:,}")
    
    # Setup training
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Training device: {device}")
    
    trainer = ElasticDeepSurvTrainer(
        model=model,
        learning_rate=learning_rate,
        weight_decay=0.0,
        device=device
    )
    
    logger.info(f"Training for {n_epochs} epochs (no validation split)...")

    # Train without validation
    history = trainer.fit(
        train_loader=full_loader,
        valid_loader=None,
        n_epochs=n_epochs,
        early_stopping_patience=None,
        verbose=True
    )

    # ============================================================
    # SAFE HISTORY ACCESS - Handle key name variations
    # ============================================================
    logger.info("\nChecking training history...")
    logger.info(f"History type: {type(history)}")

    if history is None:
        raise RuntimeError("trainer.fit() returned None!")

    if not isinstance(history, dict):
        raise RuntimeError(f"trainer.fit() returned {type(history)}, expected dict!")

    logger.info(f"History keys: {list(history.keys())}")

    # Handle key name variations ('train_cindex' vs 'valid_c_index')
    cindex_key = None
    possible_keys = ['train_cindex', 'valid_cindex', 'valid_c_index']
    for key in possible_keys:
        if key in history and len(history[key]) > 0:
            cindex_key = key
            break

    if cindex_key is None:
        raise RuntimeError(f"No C-index key found! Available: {list(history.keys())}")

    # Check if lists are non-empty
    if 'train_loss' not in history or len(history['train_loss']) == 0:
        raise RuntimeError("history['train_loss'] is empty!")

    logger.info(f"✅ History valid - {len(history['train_loss'])} epochs recorded")
    logger.info(f"✅ Using C-index key: '{cindex_key}'")

    # Safe access
    final_train_loss = history['train_loss'][-1]
    final_train_cindex = history[cindex_key][-1]

    logger.info(f"Final training loss: {final_train_loss:.4f}")
    logger.info(f"Final training C-index: {final_train_cindex:.4f}")

    # Quality warning
    if final_train_cindex < 0.58:
        logger.warning(f"\n{'='*60}")
        logger.warning(f"⚠️  LOW C-INDEX WARNING: {final_train_cindex:.4f} < 0.58")
        logger.warning(f"{'='*60}")
        logger.warning("Model performance is poor. Biomarkers may be unreliable.")
        logger.warning("Consider implementing proper validation-based training.")
        logger.warning(f"{'='*60}\n")

    
    # Check sparsity
    try:
        sparsity_info = model.get_sparsity_info()
        final_sparsity = sparsity_info['sparsity_ratio']
        logger.info(f"Final sparsity: {final_sparsity:.1%}")
    except:
        final_sparsity = 0.0
        logger.warning("Could not compute sparsity")

    
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING COMPLETE - QUALITY CHECK")
    logger.info(f"{'='*60}")
    logger.info(f"Final train C-index: {final_train_cindex:.4f}")
    logger.info(f"Final sparsity: {final_sparsity:.1%}")
    logger.info(f"Best epoch: {history['best_epoch'] if 'best_epoch' in history else 'N/A'}")
    
    # Quality thresholds
    MIN_CINDEX = 0.58
    MIN_SPARSITY = 0.01
    
    quality_issues = []
    
    
        # Check C-index
    if final_train_cindex < MIN_CINDEX:
        quality_issues.append(
            f"Low training C-index: {final_train_cindex:.4f} < {MIN_CINDEX:.4f}"
        )

    # Check sparsity
    if final_sparsity < MIN_SPARSITY:
        quality_issues.append(
            f"No sparsity achieved: {final_sparsity:.1%} < {MIN_SPARSITY:.1%}"
        )
        
    
    if quality_issues:
        logger.warning(f"\n{'='*60}")
        logger.warning(f"MODEL QUALITY ISSUES DETECTED:")
        for issue in quality_issues:
            logger.warning(f"  {issue}")
        logger.warning(f"{'='*60}")
        logger.warning(f"\nBiomarkers may be unreliable. Consider:")
        logger.warning(f"  1. Re-running hyperparameter search")
        logger.warning(f"  2. Using weaker lambda")
        logger.warning(f"  3. Falling back to Cox elastic net")
        logger.warning(f"{'='*60}\n")
        
        # Stop if completely failed
       
    else:
        logger.info(f"All quality checks passed!")
    
    logger.info(f"{'='*60}\n")
    
    # ============================================================
    # STEP 8: Extract feature importance
    # ============================================================
    logger.info("Computing feature importance (L2 norm of first layer weights)...")
    importance_scores = compute_l2_feature_importance(model)
    gene_names = expr_standardized.index.tolist()
    
    # Log importance statistics
    logger.info(f"Importance statistics:")
    logger.info(f"  Min: {importance_scores.min():.6f}")
    logger.info(f"  Max: {importance_scores.max():.6f}")
    logger.info(f"  Mean: {importance_scores.mean():.6f}")
    logger.info(f"  Median: {np.median(importance_scores):.6f}")
    logger.info(f"  Genes with importance > 0: {(importance_scores > 1e-6).sum()}/{len(importance_scores)}")
    
    return model, importance_scores, gene_names, final_train_cindex

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
        description='Extract biomarker importance from ElasticDeepSurv models using 308 consensus genes'
    )
    parser.add_argument('--tcga_params', type=str, required=True)
    parser.add_argument('--orien_params', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--consensus_genes', type=str, default='data/raw/consensus_genes_308.txt')
    parser.add_argument('--n_epochs', type=int, default=150)
    
    args = parser.parse_args()
    
    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"results/biomarker_importance_{timestamp}"
    
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
    
    # Load raw data
    logger.info("\nLoading raw expression data...")
    tcga_expr = pd.read_csv("data/raw/tcga_batch_corrected_2sv.csv", index_col=0)
    orien_expr = pd.read_csv("data/raw/orien_batch_corrected.csv", index_col=0)
    
    logger.info("Loading survival data...")
    surv_tcga = pd.read_csv("data/processed/surv_tcga_harmonized.csv", index_col=0)
    surv_orien = pd.read_csv("data/processed/surv_orien_harmonized.csv", index_col=0)
    
    # Train TCGA model
    logger.info("\n" + "="*60)
    logger.info("TRAINING TCGA MODEL")
    logger.info("="*60)
    tcga_model, tcga_importance, tcga_genes, tcga_train_cindex = train_model_on_cohort(
        expr_raw=tcga_expr,
        surv=surv_tcga,
        best_params=tcga_params,
        cohort_name='TCGA',
        consensus_genes=consensus_genes,
        n_epochs=args.n_epochs
    )
    
    # Train ORIEN model
    logger.info("\n" + "="*60)
    logger.info("TRAINING ORIEN MODEL")
    logger.info("="*60)
    orien_model, orien_importance, orien_genes, orien_train_cindex = train_model_on_cohort(
        expr_raw=orien_expr,
        surv=surv_orien,
        best_params=orien_params,
        cohort_name='ORIEN',
        consensus_genes=consensus_genes,
        n_epochs=args.n_epochs
    )
    
    # Verify gene lists match
    assert tcga_genes == orien_genes, "Gene lists don't match!"
    gene_names = tcga_genes
    
    # Save ALL importance scores
    logger.info("\nSaving importance scores for ALL genes...")
    importance_df = pd.DataFrame({
        'gene_name': gene_names,
        'tcga_importance': tcga_importance,
        'orien_importance': orien_importance,
        'mean_importance': (tcga_importance + orien_importance) / 2
    }).sort_values('mean_importance', ascending=False)
    
    importance_df.to_csv(output_dir / 'all_gene_importances.csv', index=False)
    logger.info(f"Saved: {output_dir / 'all_gene_importances.csv'}")
    
    # Save models
    logger.info("\nSaving trained models...")
    torch.save(tcga_model.state_dict(), output_dir / 'tcga_model.pth')
    torch.save(orien_model.state_dict(), output_dir / 'orien_model.pth')
    
    # Create summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'n_input_genes': len(consensus_genes),
        'tcga': {
            'n_samples': len(surv_tcga),
            'n_genes': len(tcga_genes),
            'final_train_cindex': float(tcga_train_cindex),
            'n_params': sum(p.numel() for p in tcga_model.parameters()),
            'importance_stats': {
                'min': float(tcga_importance.min()),
                'max': float(tcga_importance.max()),
                'mean': float(tcga_importance.mean()),
                'median': float(np.median(tcga_importance)),
                'std': float(tcga_importance.std())
            }
        },
        'orien': {
            'n_samples': len(surv_orien),
            'n_genes': len(orien_genes),
            'final_train_cindex': float(orien_train_cindex),
            'n_params': sum(p.numel() for p in orien_model.parameters()),
            'importance_stats': {
                'min': float(orien_importance.min()),
                'max': float(orien_importance.max()),
                'mean': float(orien_importance.mean()),
                'median': float(np.median(orien_importance)),
                'std': float(orien_importance.std())
            }
        }
    }
    
    # Save summary
    with open(output_dir / 'SUMMARY.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info("BIOMARKER EXTRACTION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"\nOutputs:")
    logger.info(f"  - all_gene_importances.csv ({len(gene_names)} genes)")
    logger.info(f"  - tcga_model.pth")
    logger.info(f"  - orien_model.pth")
    logger.info(f"  - SUMMARY.json")
    logger.info(f"\nNext step:")
    logger.info(f"  Use these importance scores to test different k values")
    logger.info(f"  and perform bidirectional cross-cohort validation")
    logger.info(f"{'='*60}")
    
    return summary


if __name__ == "__main__":
    summary = main()