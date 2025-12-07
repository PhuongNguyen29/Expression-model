#!/usr/bin/env python3
"""
Experiment 1A: ElasticDeepSurv on 20 Cox Genes

Objective: Test if deep learning can match penalized Cox performance (0.72/0.67)
           using identical features.

Protocol:
- Cross-cohort validation (train 100% source → test 100% target)
- Both directions: ORIEN→TCGA and TCGA→ORIEN
- 5 seeds for variance estimation
- Fixed hyperparameters (Option C)

Expected comparison:
- Penalized Cox (20 genes): 0.72 / 0.67
- ElasticDeepSurv (20 genes): ? / ?

Based on step2_2_cross_cohort_validation.py methodology
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import json
from typing import List, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from lifelines.utils import concordance_index

from src.data.dataset import SurvivalDataset
from src.models.elastic_deepsurv import ElasticDeepSurv, ElasticDeepSurvTrainer
from src.utils.batch_samplers import StratifiedBatchSampler

# ============================================================
# Configuration
# ============================================================

EXPERIMENT_NAME = "exp1a_deepsurv_20genes"
GENE_FILE = "data/raw/cox_consensus_genes_20.txt"
OUTPUT_DIR = Path(f"results_v2/05_ablation_experiments/{EXPERIMENT_NAME}")

# Data files (matching step2_2)
TCGA_EXPR_FILE = "data/raw/tcga_batch_corrected_2sv.csv"
ORIEN_EXPR_FILE = "data/raw/orien_batch_corrected.csv"
TCGA_SURV_FILE = "data/processed/surv_tcga_harmonized.csv"
ORIEN_SURV_FILE = "data/processed/surv_orien_harmonized.csv"

# Fixed hyperparameters (Option C - scaled for 20 genes)
FIXED_CONFIG = {
    'hidden_sizes': [32],
    'dropout': 0.3,
    'learning_rate': 1e-4,
    'batch_size': 32,
    'alpha': 0.001,
    'l1_ratio': 0.5,
    'activation': 'relu',
    'batch_norm': True,
    'weight_init': 'kaiming_normal',
    'max_epochs': 100,
    'early_stopping_patience': 20
}

SEEDS = [42, 123, 456, 789, 1011]

# ============================================================
# Setup Logging
# ============================================================

def setup_logging(output_dir: Path):
    """Setup logging configuration."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / f"{EXPERIMENT_NAME}_{datetime.now():%Y%m%d_%H%M%S}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# ============================================================
# Utility Functions
# ============================================================

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_genes(gene_file: str) -> List[str]:
    """Load gene list from file."""
    with open(gene_file, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]
    return genes


def load_data(gene_list: List[str], logger) -> Dict:
    """Load expression and survival data, filtered to specified genes."""
    logger.info("Loading data files...")
    
    # Load expression data
    tcga_expr = pd.read_csv(TCGA_EXPR_FILE, index_col=0)
    orien_expr = pd.read_csv(ORIEN_EXPR_FILE, index_col=0)
    
    # Load survival data
    tcga_surv = pd.read_csv(TCGA_SURV_FILE, index_col=0)
    orien_surv = pd.read_csv(ORIEN_SURV_FILE, index_col=0)
    
    logger.info(f"  TCGA expression: {tcga_expr.shape[0]} genes × {tcga_expr.shape[1]} samples")
    logger.info(f"  ORIEN expression: {orien_expr.shape[0]} genes × {orien_expr.shape[1]} samples")
    
    # Filter to specified genes
    available_genes = [g for g in gene_list if g in tcga_expr.index and g in orien_expr.index]
    
    if len(available_genes) != len(gene_list):
        missing = set(gene_list) - set(available_genes)
        logger.warning(f"  Missing {len(missing)} genes: {list(missing)[:5]}...")
    
    logger.info(f"  Using {len(available_genes)} genes")
    
    tcga_expr = tcga_expr.loc[available_genes]
    orien_expr = orien_expr.loc[available_genes]
    
    # Align samples between expression and survival data
    tcga_samples = list(set(tcga_expr.columns) & set(tcga_surv.index))
    orien_samples = list(set(orien_expr.columns) & set(orien_surv.index))
    
    tcga_expr = tcga_expr[tcga_samples]
    orien_expr = orien_expr[orien_samples]
    tcga_surv = tcga_surv.loc[tcga_samples]
    orien_surv = orien_surv.loc[orien_samples]
    
    logger.info(f"  TCGA final: {tcga_expr.shape[0]} genes × {tcga_expr.shape[1]} samples")
    logger.info(f"  ORIEN final: {orien_expr.shape[0]} genes × {orien_expr.shape[1]} samples")
    logger.info(f"  TCGA events: {tcga_surv['event'].sum()}/{len(tcga_surv)} ({100*tcga_surv['event'].mean():.1f}%)")
    logger.info(f"  ORIEN events: {orien_surv['event'].sum()}/{len(orien_surv)} ({100*orien_surv['event'].mean():.1f}%)")
    
    return {
        'tcga_expr': tcga_expr,
        'orien_expr': orien_expr,
        'tcga_surv': tcga_surv,
        'orien_surv': orien_surv,
        'genes': available_genes
    }


def standardize_data(train_expr: pd.DataFrame, test_expr: pd.DataFrame) -> tuple:
    """
    Z-score standardization: fit on train, transform both.
    
    Args:
        train_expr: Training expression data (genes × samples)
        test_expr: Test expression data (genes × samples)
    
    Returns:
        Standardized train and test DataFrames
    """
    # Transpose to samples × genes for sklearn
    train_T = train_expr.T
    test_T = test_expr.T
    
    # Fit scaler on training data
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_T)
    test_scaled = scaler.transform(test_T)
    
    # Convert back to DataFrames (genes × samples)
    train_standardized = pd.DataFrame(
        train_scaled.T,
        index=train_expr.index,
        columns=train_expr.columns
    )
    test_standardized = pd.DataFrame(
        test_scaled.T,
        index=test_expr.index,
        columns=test_expr.columns
    )
    
    return train_standardized, test_standardized


# ============================================================
# Training and Evaluation
# ============================================================

def train_and_evaluate_direction(
    source_expr: pd.DataFrame,
    source_surv: pd.DataFrame,
    target_expr: pd.DataFrame,
    target_surv: pd.DataFrame,
    source_name: str,
    target_name: str,
    seed: int,
    config: Dict,
    logger,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict:
    """
    Train on source cohort, evaluate on target cohort.
    
    Uses 80/20 split on source for early stopping (matching step2_2).
    """
    set_seed(seed)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Seed {seed}: {source_name} → {target_name}")
    logger.info(f"{'='*60}")
    
    # Split source into 80% train, 20% validation for early stopping
    source_samples = source_expr.columns.tolist()
    source_events = source_surv.loc[source_samples, 'event'].values
    
    train_samples, val_samples = train_test_split(
        source_samples,
        test_size=0.2,
        random_state=seed,
        stratify=source_events
    )
    
    logger.info(f"  Source split: {len(train_samples)} train, {len(val_samples)} validation")
    
    # Extract train and validation data
    train_expr = source_expr[train_samples]
    val_expr = source_expr[val_samples]
    train_surv = source_surv.loc[train_samples]
    val_surv = source_surv.loc[val_samples]
    
    # Standardize: fit on train, transform val and target
    train_standardized, val_standardized = standardize_data(train_expr, val_expr)
    _, target_standardized = standardize_data(train_expr, target_expr)
    
    logger.info(f"  After standardization:")
    logger.info(f"    Train: {train_standardized.shape[0]} genes × {train_standardized.shape[1]} samples")
    logger.info(f"    Val: {val_standardized.shape[0]} genes × {val_standardized.shape[1]} samples")
    logger.info(f"    Target: {target_standardized.shape[0]} genes × {target_standardized.shape[1]} samples")
    
    # Create datasets
    train_dataset = SurvivalDataset(train_standardized, train_surv)
    val_dataset = SurvivalDataset(val_standardized, val_surv)
    target_dataset = SurvivalDataset(target_standardized, target_surv)
    
    # Create dataloaders
    n_train_samples = len(train_dataset)
    train_events = train_surv['event'].values
    
    if n_train_samples < 400:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=0
        )
    else:
        train_sampler = StratifiedBatchSampler(
            events=train_events,
            batch_size=config['batch_size'],
            min_events_per_batch=2,
            shuffle=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=0
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    target_loader = DataLoader(
        target_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    n_features = train_standardized.shape[0]
    model = ElasticDeepSurv(
        n_features=n_features,
        hidden_sizes=config['hidden_sizes'],
        dropout=config['dropout'],
        activation=config['activation'],
        batch_norm=config['batch_norm'],
        weight_init=config['weight_init'],
        l1_ratio=config['l1_ratio'],
        alpha=config['alpha']
    )
    
    logger.info(f"  Model: {n_features} → {config['hidden_sizes']} → 1")
    
    # Create trainer
    trainer = ElasticDeepSurvTrainer(
        model=model,
        learning_rate=config['learning_rate'],
        device=device
    )
    
    # Train with early stopping on validation set
    history = trainer.fit(
        train_loader=train_loader,
        valid_loader=val_loader,
        n_epochs=config['max_epochs'],
        early_stopping_patience=config['early_stopping_patience'],
        verbose=False
    )
    
    # Evaluate on all sets
    _, _, _, train_cindex = trainer.evaluate(train_loader)
    _, _, _, val_cindex = trainer.evaluate(val_loader)
    _, _, _, test_cindex = trainer.evaluate(target_loader)
    
    logger.info(f"  Results:")
    logger.info(f"    Train C-index: {train_cindex:.4f}")
    logger.info(f"    Val C-index: {val_cindex:.4f}")
    logger.info(f"    Test C-index ({target_name}): {test_cindex:.4f}")
    
    # Get best epoch from history
    best_epoch = history.get('best_epoch', len(history.get('train_loss', [])))
    
    return {
        'seed': seed,
        'source': source_name,
        'target': target_name,
        'train_cindex': float(train_cindex),
        'val_cindex': float(val_cindex),
        'test_cindex': float(test_cindex),
        'best_epoch': best_epoch,
        'n_train': len(train_samples),
        'n_val': len(val_samples),
        'n_target': len(target_surv),
        'n_features': n_features
    }


# ============================================================
# Main Execution
# ============================================================

def main():
    # Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "orien_to_tcga").mkdir(exist_ok=True)
    (OUTPUT_DIR / "tcga_to_orien").mkdir(exist_ok=True)
    
    logger = setup_logging(OUTPUT_DIR)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info("="*70)
    logger.info(f"EXPERIMENT 1A: ElasticDeepSurv on 20 Cox Genes")
    logger.info("="*70)
    logger.info(f"Device: {device}")
    logger.info(f"Seeds: {SEEDS}")
    logger.info(f"Output: {OUTPUT_DIR}")
    logger.info(f"\nFixed Hyperparameters:")
    for key, value in FIXED_CONFIG.items():
        logger.info(f"  {key}: {value}")
    
    # Load genes
    logger.info(f"\nLoading genes from {GENE_FILE}")
    genes = load_genes(GENE_FILE)
    logger.info(f"  Loaded {len(genes)} genes")
    
    # Load data
    data = load_data(genes, logger)
    
    # Run both directions for all seeds
    all_results = []
    
    # Direction 1: ORIEN → TCGA
    logger.info("\n" + "="*70)
    logger.info("DIRECTION 1: ORIEN → TCGA")
    logger.info("="*70)
    
    orien_to_tcga_results = []
    for seed in SEEDS:
        result = train_and_evaluate_direction(
            source_expr=data['orien_expr'],
            source_surv=data['orien_surv'],
            target_expr=data['tcga_expr'],
            target_surv=data['tcga_surv'],
            source_name='ORIEN',
            target_name='TCGA',
            seed=seed,
            config=FIXED_CONFIG,
            logger=logger,
            device=device
        )
        orien_to_tcga_results.append(result)
        
        # Save individual result
        with open(OUTPUT_DIR / "orien_to_tcga" / f"seed{seed}_results.json", 'w') as f:
            json.dump(result, f, indent=2)
    
    all_results.extend(orien_to_tcga_results)
    
    # Direction 2: TCGA → ORIEN
    logger.info("\n" + "="*70)
    logger.info("DIRECTION 2: TCGA → ORIEN")
    logger.info("="*70)
    
    tcga_to_orien_results = []
    for seed in SEEDS:
        result = train_and_evaluate_direction(
            source_expr=data['tcga_expr'],
            source_surv=data['tcga_surv'],
            target_expr=data['orien_expr'],
            target_surv=data['orien_surv'],
            source_name='TCGA',
            target_name='ORIEN',
            seed=seed,
            config=FIXED_CONFIG,
            logger=logger,
            device=device
        )
        tcga_to_orien_results.append(result)
        
        # Save individual result
        with open(OUTPUT_DIR / "tcga_to_orien" / f"seed{seed}_results.json", 'w') as f:
            json.dump(result, f, indent=2)
    
    all_results.extend(tcga_to_orien_results)
    
    # ============================================================
    # Summary Statistics
    # ============================================================
    
    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    
    # ORIEN → TCGA summary
    o2t_test_cindices = [r['test_cindex'] for r in orien_to_tcga_results]
    o2t_mean = np.mean(o2t_test_cindices)
    o2t_std = np.std(o2t_test_cindices)
    
    # TCGA → ORIEN summary
    t2o_test_cindices = [r['test_cindex'] for r in tcga_to_orien_results]
    t2o_mean = np.mean(t2o_test_cindices)
    t2o_std = np.std(t2o_test_cindices)
    
    logger.info(f"\nORIEN → TCGA: {o2t_mean:.4f} ± {o2t_std:.4f}")
    logger.info(f"  Individual: {[f'{c:.4f}' for c in o2t_test_cindices]}")
    
    logger.info(f"\nTCGA → ORIEN: {t2o_mean:.4f} ± {t2o_std:.4f}")
    logger.info(f"  Individual: {[f'{c:.4f}' for c in t2o_test_cindices]}")
    
    logger.info(f"\nComparison to Penalized Cox (20 genes):")
    logger.info(f"  Penalized Cox ORIEN→TCGA: 0.72")
    logger.info(f"  ElasticDeepSurv ORIEN→TCGA: {o2t_mean:.4f} ± {o2t_std:.4f}")
    logger.info(f"  Difference: {o2t_mean - 0.72:+.4f}")
    logger.info(f"\n  Penalized Cox TCGA→ORIEN: 0.67")
    logger.info(f"  ElasticDeepSurv TCGA→ORIEN: {t2o_mean:.4f} ± {t2o_std:.4f}")
    logger.info(f"  Difference: {t2o_mean - 0.67:+.4f}")
    
    # Save summary
    summary = {
        'experiment': EXPERIMENT_NAME,
        'n_genes': len(data['genes']),
        'genes': data['genes'],
        'config': FIXED_CONFIG,
        'seeds': SEEDS,
        'orien_to_tcga': {
            'mean': float(o2t_mean),
            'std': float(o2t_std),
            'individual': o2t_test_cindices
        },
        'tcga_to_orien': {
            'mean': float(t2o_mean),
            'std': float(t2o_std),
            'individual': t2o_test_cindices
        },
        'comparison_to_cox': {
            'cox_orien_to_tcga': 0.72,
            'cox_tcga_to_orien': 0.67,
            'deepsurv_orien_to_tcga': float(o2t_mean),
            'deepsurv_tcga_to_orien': float(t2o_mean),
            'diff_orien_to_tcga': float(o2t_mean - 0.72),
            'diff_tcga_to_orien': float(t2o_mean - 0.67)
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open(OUTPUT_DIR / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save all results as CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "all_results.csv", index=False)
    
    logger.info(f"\nResults saved to: {OUTPUT_DIR}")
    logger.info("="*70)
    logger.info("EXPERIMENT 1A COMPLETE")
    logger.info("="*70)


if __name__ == "__main__":
    main()
