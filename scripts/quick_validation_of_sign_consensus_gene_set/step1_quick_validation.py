#!/usr/bin/env python3
"""
Step 1: Quick Validation - Cross-Cohort Performance Comparison

Purpose: Compare cross-cohort C-index for:
- Set A: 68 consensus genes (magnitude-only)
- Set B: 26 sign-consistent genes (magnitude + sign filter)

Uses YOUR existing ElasticDeepSurv and SurvivalDataset classes for consistency.

Protocol:
- Train on source cohort (100% data) using pre-tuned hyperparameters
- Test on target cohort (100% data)
- Both directions: TCGA→ORIEN and ORIEN→TCGA
- 5 random seeds for robust estimation

Output:
    results_v2/07_sign_filter_validation/
    ├── set_A_68genes/
    │   ├── tcga_to_orien/seed_{seed}_results.json
    │   ├── orien_to_tcga/seed_{seed}_results.json
    │   └── summary.json
    ├── set_B_26genes/
    │   └── ...
    └── logs/

Author: [Your Name]
Date: 2024-12

Usage:
    python step1_quick_validation.py --gene_set A
    python step1_quick_validation.py --gene_set B
    python step1_quick_validation.py --gene_set both
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from lifelines.utils import concordance_index

# Import YOUR existing classes
from src.models.elastic_deepsurv import ElasticDeepSurv, ElasticDeepSurvTrainer
from src.data.dataset import SurvivalDataset

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    # Input files (relative to project root)
    'inputs_dir': 'results_v2/07_sign_filter_validation/inputs',
    'output_dir': 'results_v2/07_sign_filter_validation',
    
    # Hyperparameter files (will be loaded from inputs/validation_config.json)
    'best_params_tcga': 'results_v2/02b_biomarker_discovery_ig/k_selection_with_tuning/k140/hyperparameter_tuning/tcga/best_params.json',
    'best_params_orien': 'results_v2/02b_biomarker_discovery_ig/k_selection_with_tuning/k140/hyperparameter_tuning/orien/best_params.json',
    
    # Data directory (passed to load_expression_and_survival_data)
    'data_dir': 'data',
    
    # Validation settings
    'seeds': [42, 123, 456, 789, 1011],
    'max_epochs': 200,
    'patience': 20,  # Early stopping patience
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Setup logging
logger = logging.getLogger(__name__)


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================
def load_expression_and_survival_data(data_dir: str) -> Dict:
    """
    Load all expression and survival data once.
    
    MATCHES YOUR EXISTING CODE in run_step2b_crosscohort_validation_only_ig.py
    
    Returns:
        Dict with keys: tcga_expr_raw, orien_expr_raw, tcga_surv, orien_surv
    """
    from pathlib import Path
    data_dir = Path(data_dir)
    
    logger.info("Loading expression and survival data...")
    
    # Load exactly as your existing code does (index_col=0)
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


def subset_expression_to_genes(expr_df: pd.DataFrame, gene_list: list) -> pd.DataFrame:
    """
    Subset expression data to specified genes.
    
    Args:
        expr_df: Expression DataFrame (genes × samples)
        gene_list: List of gene IDs to select
        
    Returns:
        Expression DataFrame subset to genes (genes × samples)
    """
    # Handle gene selection
    available_genes = [g for g in gene_list if g in expr_df.index]
    if len(available_genes) < len(gene_list):
        missing = set(gene_list) - set(available_genes)
        logger.warning(f"Missing {len(missing)} genes from expression data")
    
    return expr_df.loc[available_genes]


def create_dataloader(expr_df: pd.DataFrame, surv_df: pd.DataFrame, 
                      batch_size: int = 32, shuffle: bool = True) -> Tuple:
    """
    Create DataLoader using YOUR SurvivalDataset class.
    
    Args:
        expr_df: Expression data (genes × samples)
        surv_df: Survival data with index = sample IDs
        
    Returns:
        Tuple of (DataLoader, SurvivalDataset)
    """
    dataset = SurvivalDataset(expr_df, surv_df)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=True if CONFIG['device'] == 'cuda' else False
    )
    return loader, dataset


# ============================================================================
# MODEL CREATION
# ============================================================================
def create_model(n_features: int, params: dict, seed: int) -> ElasticDeepSurv:
    """
    Create ElasticDeepSurv model with specified parameters.
    
    Args:
        n_features: Number of input features (genes)
        params: Hyperparameters from best_params.json
        seed: Random seed
        
    Returns:
        ElasticDeepSurv model
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Determine hidden sizes
    n_layers = params.get('n_layers', 1)
    hidden_sizes = []
    for i in range(1, n_layers + 1):
        layer_key = f'layer{i}_size'
        if layer_key in params:
            hidden_sizes.append(params[layer_key])
        elif 'layer1_size' in params and i == 1:
            hidden_sizes.append(params['layer1_size'])
    
    # Default if nothing specified
    if not hidden_sizes:
        hidden_sizes = [64]
    
    model = ElasticDeepSurv(
        n_features=n_features,
        hidden_sizes=hidden_sizes,
        dropout=params.get('dropout', 0.3),
        activation=params.get('activation', 'elu'),
        batch_norm=params.get('batch_norm', False),
        weight_init=params.get('weight_init', 'kaiming_uniform'),
        l1_ratio=params.get('l1_ratio', 0.5),
        alpha=params.get('alpha', 0.001)
    )
    
    return model


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================
def train_and_evaluate(train_loader, test_loader, train_dataset, test_dataset,
                       params: dict, seed: int) -> dict:
    """
    Train on source cohort, evaluate on target cohort.
    
    Uses YOUR ElasticDeepSurvTrainer class.
    
    Returns:
        dict with train_cindex, test_cindex, best_epoch
    """
    device = CONFIG['device']
    
    # Create model
    n_features = train_dataset.n_features
    model = create_model(n_features, params, seed)
    
    # Create trainer
    trainer = ElasticDeepSurvTrainer(
        model=model,
        learning_rate=params.get('learning_rate', 1e-4),
        weight_decay=0.0,  # Regularization handled by ElasticDeepSurv
        scheduler_patience=10,
        device=device
    )
    
    # Train (no validation split - using full source cohort)
    # We'll use training data as validation for early stopping
    logger.info(f"Training with seed {seed}...")
    history = trainer.fit(
        train_loader=train_loader,
        valid_loader=train_loader,  # Use train as validation for this quick test
        n_epochs=CONFIG['max_epochs'],
        early_stopping_patience=CONFIG['patience'],
        verbose=False
    )
    
    # Evaluate on training data (source cohort)
    _, _, _, train_cindex = trainer.evaluate(train_loader)
    
    # Evaluate on test data (target cohort)
    _, _, _, test_cindex = trainer.evaluate(test_loader)
    
    best_epoch = history.get('best_epoch', len(history['train_cindex']))
    
    return {
        'train_cindex': float(train_cindex),
        'test_cindex': float(test_cindex),
        'best_epoch': int(best_epoch),
        'total_epochs': len(history['train_cindex'])
    }


# ============================================================================
# MAIN VALIDATION
# ============================================================================
def run_validation(gene_set_name: str, gene_set_file: str, output_subdir: str):
    """Run cross-cohort validation for a gene set."""
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Validating Gene Set: {gene_set_name}")
    logger.info(f"{'='*70}")
    
    # Load gene list
    with open(gene_set_file, 'r') as f:
        gene_list = [line.strip() for line in f if line.strip()]
    logger.info(f"Loaded {len(gene_list)} genes from {gene_set_file}")
    
    # Load hyperparameters
    with open(CONFIG['best_params_tcga'], 'r') as f:
        params_tcga = json.load(f)['best_params']
    with open(CONFIG['best_params_orien'], 'r') as f:
        params_orien = json.load(f)['best_params']
    
    # Load all data using the SAME approach as your existing code
    data_dict = load_expression_and_survival_data('data')
    
    # Subset expression data to gene list
    expr_tcga = subset_expression_to_genes(data_dict['tcga_expr_raw'], gene_list)
    expr_orien = subset_expression_to_genes(data_dict['orien_expr_raw'], gene_list)
    
    logger.info(f"\nExpression data subset:")
    logger.info(f"  TCGA: {expr_tcga.shape[1]} samples, {expr_tcga.shape[0]} genes")
    logger.info(f"  ORIEN: {expr_orien.shape[1]} samples, {expr_orien.shape[0]} genes")
    
    # Get survival data
    surv_tcga = data_dict['tcga_surv']
    surv_orien = data_dict['orien_surv']
    
    # Create dataloaders
    batch_size = params_tcga.get('batch_size', 32)
    
    loader_tcga, dataset_tcga = create_dataloader(
        expr_tcga, surv_tcga, batch_size=batch_size, shuffle=True
    )
    loader_orien, dataset_orien = create_dataloader(
        expr_orien, surv_orien, batch_size=batch_size, shuffle=True
    )
    
    logger.info(f"\nDatasets created:")
    logger.info(f"  TCGA: {len(dataset_tcga)} samples, {dataset_tcga.n_features} features")
    logger.info(f"  ORIEN: {len(dataset_orien)} samples, {dataset_orien.n_features} features")
    
    # Create output directories
    os.makedirs(os.path.join(output_subdir, 'tcga_to_orien'), exist_ok=True)
    os.makedirs(os.path.join(output_subdir, 'orien_to_tcga'), exist_ok=True)
    
    results = {
        'gene_set': gene_set_name,
        'n_genes': len(gene_list),
        'tcga_to_orien': [],
        'orien_to_tcga': []
    }
    
    # Direction 1: TCGA → ORIEN
    logger.info("\n--- Direction: TCGA → ORIEN ---")
    for seed in CONFIG['seeds']:
        logger.info(f"  Seed {seed}...")
        
        # Recreate loaders with new seed for shuffling
        torch.manual_seed(seed)
        loader_tcga_train, _ = create_dataloader(
            expr_tcga, surv_tcga, batch_size=batch_size, shuffle=True
        )
        loader_orien_test, _ = create_dataloader(
            expr_orien, surv_orien, batch_size=batch_size, shuffle=False
        )
        
        result = train_and_evaluate(
            train_loader=loader_tcga_train,
            test_loader=loader_orien_test,
            train_dataset=dataset_tcga,
            test_dataset=dataset_orien,
            params=params_tcga,
            seed=seed
        )
        result['seed'] = seed
        results['tcga_to_orien'].append(result)
        
        # Save individual result
        result_file = os.path.join(output_subdir, 'tcga_to_orien', f'seed_{seed}_results.json')
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"    Train C-index: {result['train_cindex']:.4f}, "
                   f"Test C-index: {result['test_cindex']:.4f}")
    
    # Direction 2: ORIEN → TCGA
    logger.info("\n--- Direction: ORIEN → TCGA ---")
    for seed in CONFIG['seeds']:
        logger.info(f"  Seed {seed}...")
        
        # Recreate loaders with new seed for shuffling
        torch.manual_seed(seed)
        loader_orien_train, _ = create_dataloader(
            expr_orien, surv_orien, batch_size=batch_size, shuffle=True
        )
        loader_tcga_test, _ = create_dataloader(
            expr_tcga, surv_tcga, batch_size=batch_size, shuffle=False
        )
        
        result = train_and_evaluate(
            train_loader=loader_orien_train,
            test_loader=loader_tcga_test,
            train_dataset=dataset_orien,
            test_dataset=dataset_tcga,
            params=params_orien,
            seed=seed
        )
        result['seed'] = seed
        results['orien_to_tcga'].append(result)
        
        # Save individual result
        result_file = os.path.join(output_subdir, 'orien_to_tcga', f'seed_{seed}_results.json')
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"    Train C-index: {result['train_cindex']:.4f}, "
                   f"Test C-index: {result['test_cindex']:.4f}")
    
    # Compute summary statistics
    tcga_to_orien_cindices = [r['test_cindex'] for r in results['tcga_to_orien']]
    orien_to_tcga_cindices = [r['test_cindex'] for r in results['orien_to_tcga']]
    
    summary = {
        'gene_set': gene_set_name,
        'n_genes': len(gene_list),
        'tcga_to_orien': {
            'mean': float(np.mean(tcga_to_orien_cindices)),
            'std': float(np.std(tcga_to_orien_cindices)),
            'min': float(np.min(tcga_to_orien_cindices)),
            'max': float(np.max(tcga_to_orien_cindices)),
            'all_cindices': tcga_to_orien_cindices
        },
        'orien_to_tcga': {
            'mean': float(np.mean(orien_to_tcga_cindices)),
            'std': float(np.std(orien_to_tcga_cindices)),
            'min': float(np.min(orien_to_tcga_cindices)),
            'max': float(np.max(orien_to_tcga_cindices)),
            'all_cindices': orien_to_tcga_cindices
        },
        'overall': {
            'mean': float(np.mean(tcga_to_orien_cindices + orien_to_tcga_cindices)),
            'std': float(np.std(tcga_to_orien_cindices + orien_to_tcga_cindices))
        }
    }
    
    # Save summary
    summary_file = os.path.join(output_subdir, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"SUMMARY: {gene_set_name}")
    logger.info(f"{'='*70}")
    logger.info(f"TCGA → ORIEN: {summary['tcga_to_orien']['mean']:.4f} ± {summary['tcga_to_orien']['std']:.4f}")
    logger.info(f"ORIEN → TCGA: {summary['orien_to_tcga']['mean']:.4f} ± {summary['orien_to_tcga']['std']:.4f}")
    logger.info(f"Overall:      {summary['overall']['mean']:.4f} ± {summary['overall']['std']:.4f}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Quick validation for sign-filter comparison')
    parser.add_argument('--gene_set', type=str, choices=['A', 'B', 'both'], default='both',
                       help='Gene set to validate: A (68 genes), B (26 genes), or both')
    args = parser.parse_args()
    
    # Setup logging
    log_dir = os.path.join(CONFIG['output_dir'], 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'validation_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"Device: {CONFIG['device']}")
    logger.info(f"Seeds: {CONFIG['seeds']}")
    
    summaries = {}
    
    # Validate Set A (68 genes)
    if args.gene_set in ['A', 'both']:
        set_a_file = os.path.join(CONFIG['inputs_dir'], 'gene_set_A_68_magnitude_only.txt')
        set_a_output = os.path.join(CONFIG['output_dir'], 'set_A_68genes')
        summaries['set_A'] = run_validation('Set_A_68_magnitude_only', set_a_file, set_a_output)
    
    # Validate Set B (26 genes)
    if args.gene_set in ['B', 'both']:
        set_b_file = os.path.join(CONFIG['inputs_dir'], 'gene_set_B_26_sign_consistent.txt')
        set_b_output = os.path.join(CONFIG['output_dir'], 'set_B_26genes')
        summaries['set_B'] = run_validation('Set_B_26_sign_consistent', set_b_file, set_b_output)
    
    # Final comparison (if both sets validated)
    if len(summaries) == 2:
        logger.info("\n" + "=" * 70)
        logger.info("FINAL COMPARISON")
        logger.info("=" * 70)
        
        set_a = summaries['set_A']
        set_b = summaries['set_B']
        
        logger.info(f"""
                        Set A (68 genes)    Set B (26 genes)    Difference
                        ----------------    ----------------    ----------
TCGA → ORIEN:           {set_a['tcga_to_orien']['mean']:.4f} ± {set_a['tcga_to_orien']['std']:.4f}    {set_b['tcga_to_orien']['mean']:.4f} ± {set_b['tcga_to_orien']['std']:.4f}    {set_b['tcga_to_orien']['mean'] - set_a['tcga_to_orien']['mean']:+.4f}
ORIEN → TCGA:           {set_a['orien_to_tcga']['mean']:.4f} ± {set_a['orien_to_tcga']['std']:.4f}    {set_b['orien_to_tcga']['mean']:.4f} ± {set_b['orien_to_tcga']['std']:.4f}    {set_b['orien_to_tcga']['mean'] - set_a['orien_to_tcga']['mean']:+.4f}
Overall:                {set_a['overall']['mean']:.4f} ± {set_a['overall']['std']:.4f}    {set_b['overall']['mean']:.4f} ± {set_b['overall']['std']:.4f}    {set_b['overall']['mean'] - set_a['overall']['mean']:+.4f}
""")
        
        # Interpretation
        diff = set_b['overall']['mean'] - set_a['overall']['mean']
        if abs(diff) < 0.01:
            logger.info("INTERPRETATION: Performance is essentially equivalent.")
            logger.info("RECOMMENDATION: Use Set B (26 sign-consistent genes) for better interpretability.")
        elif diff > 0:
            logger.info("INTERPRETATION: Set B (26 genes) performs BETTER than Set A (68 genes)!")
            logger.info("RECOMMENDATION: Use Set B - fewer genes with better performance and interpretability.")
        else:
            logger.info(f"INTERPRETATION: Set A (68 genes) performs better by {-diff:.4f}.")
            if abs(diff) < 0.02:
                logger.info("RECOMMENDATION: Consider Set B anyway - small performance cost for much better interpretability.")
            else:
                logger.info("RECOMMENDATION: Trade-off required - report both, discuss in dissertation.")
    
    logger.info(f"\nResults saved to: {CONFIG['output_dir']}")
    logger.info(f"Log file: {log_file}")


if __name__ == '__main__':
    main()
