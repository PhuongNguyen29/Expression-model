#!/usr/bin/env python3
"""
Script: evaluate_transfer_learning.py
Purpose: Evaluate transfer learning models and compare with Chapter 3 baseline
Status: ACTIVE (Chapter 4 - Transfer Learning Evaluation)
Author: Phuong
Created: 2024-11-15

This script:
1. Loads transfer-learned models (ORIEN→TCGA and TCGA→ORIEN)
2. Trains baseline models from scratch (for fair comparison)
3. Evaluates cross-cohort performance for both approaches
4. Computes improvement metrics
5. Generates comparison tables and visualizations

Usage:
    python scripts/evaluate_transfer_learning.py \
        --transfer_dir results/transfer_learning/orien_to_tcga_seed42_20251114_235110 \
        --seed 42
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lifelines.utils import concordance_index

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset import SurvivalDataset
from torch.utils.data import DataLoader
from src.models.elastic_deepsurv import ElasticDeepSurv, ElasticDeepSurvTrainer
from src.utils.batch_samplers import StratifiedBatchSampler


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_consensus_genes(filepath: str = 'data/raw/consensus_genes_308.txt'):
    """Load 308 consensus genes."""
    with open(filepath, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]
    return genes


def load_cohort_data(cohort_name: str):
    """
    Load batch-corrected data and filter to 308 consensus genes.
    Same as transfer_learning_trainer.py
    """
    # Load batch-corrected expression data
    if cohort_name.lower() == 'tcga':
        expr_file = "data/raw/tcga_batch_corrected_2sv.csv"
        surv_file = "data/processed/surv_tcga_harmonized.csv"
    else:  # orien
        expr_file = "data/raw/orien_batch_corrected.csv"
        surv_file = "data/processed/surv_orien_harmonized.csv"
    
    expression = pd.read_csv(expr_file, index_col=0)
    
    # Filter to 308 consensus genes
    consensus_genes = load_consensus_genes()
    available_genes = [g for g in consensus_genes if g in expression.index]
    expression = expression.loc[available_genes]
    
    # Load survival data
    survival = pd.read_csv(surv_file)
    if 'sampleID' in survival.columns:
        survival = survival.set_index('sampleID')
    
    # Align samples
    common_samples = list(set(expression.columns) & set(survival.index))
    common_samples = sorted(common_samples)
    
    expression = expression[common_samples]
    survival = survival.loc[common_samples]
    
    return {
        'expression': expression,
        'survival': survival
    }


def parse_optuna_params(optuna_params: Dict) -> Dict:
    """
    Convert Optuna hyperparameter format to model-compatible format.
    Same as transfer_learning_trainer.py
    """
    params = optuna_params.copy()
    
    n_layers = params.get('n_layers', 1)
    
    if n_layers == 1:
        layer1_size = params.get('layer1_size', 128)
        hidden_sizes = [layer1_size]
    elif n_layers == 2:
        arch_str = params.get('architecture_2layer', '256-64')
        hidden_sizes = [int(x) for x in arch_str.split('-')]
    else:
        raise ValueError(f"Unsupported n_layers: {n_layers}")
    
    params['hidden_sizes'] = hidden_sizes
    return params


# ============================================================================
# BASELINE TRAINING (FROM SCRATCH)
# ============================================================================

def train_baseline_model(
    data: Dict,
    params: Dict,
    n_epochs: int = 100,
    learning_rate: float = 1e-4,
    seed: int = 42,
    device: str = 'cuda'
) -> Tuple[ElasticDeepSurv, float]:
    """
    Train baseline model from scratch (no transfer learning).
    
    This provides the Chapter 3-style baseline for comparison.
    Trains on ENTIRE cohort with same methodology as transfer learning.
    
    Args:
        data: Dict with 'expression' and 'survival'
        params: Hyperparameters
        n_epochs: Training epochs
        learning_rate: Learning rate
        seed: Random seed
        device: Device
        
    Returns:
        (trained_model, training_c_index)
    """
    set_all_seeds(seed)
    
    # Create dataset
    dataset = SurvivalDataset(data['expression'], data['survival'])
    
    # Create data loader
    sampler = StratifiedBatchSampler(
        events=dataset.y_event,
        batch_size=params.get('batch_size', 32),
        shuffle=True
    )
    
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=True if device == 'cuda' else False
    )
    
    # Create model
    model = ElasticDeepSurv(
        n_features=dataset.n_features,
        hidden_sizes=params.get('hidden_sizes', [128]),
        dropout=params.get('dropout', 0.3),
        l1_ratio=params.get('l1_ratio', 0.7),
        alpha=params.get('alpha', 0.01)
    )
    
    # Create trainer
    trainer = ElasticDeepSurvTrainer(
        model=model,
        learning_rate=learning_rate,
        weight_decay=0.0,
        scheduler_patience=10,
        device=device
    )
    
    # Train
    history = trainer.fit(
        train_loader=loader,
        valid_loader=None,
        n_epochs=n_epochs,
        early_stopping_patience=20,
        verbose=False  # Suppress detailed output
    )
    
    final_cindex = history['train_cindex'][-1]
    
    return model, final_cindex


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_model_cross_cohort(
    model: ElasticDeepSurv,
    test_data: Dict,
    device: str = 'cuda'
) -> float:
    """
    Evaluate model on a different cohort (cross-cohort validation).
    
    Args:
        model: Trained model
        test_data: Dict with 'expression' and 'survival' from test cohort
        device: Device
        
    Returns:
        C-index on test cohort
    """
    model.to(device)
    model.eval()
    
    # Create test dataset
    test_dataset = SurvivalDataset(test_data['expression'], test_data['survival'])
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    all_risks = []
    all_times = []
    all_events = []
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            times = batch['time'].cpu().numpy()
            events = batch['event'].cpu().numpy()
            
            # Get risk predictions
            log_hazards = model(features)
            risks = torch.exp(log_hazards).cpu().numpy().flatten()
            
            all_risks.extend(risks)
            all_times.extend(times)
            all_events.extend(events)
    
    # Compute C-index
    cindex = concordance_index(all_times, -np.array(all_risks), all_events)
    
    return cindex


# ============================================================================
# MAIN EVALUATION PIPELINE
# ============================================================================

def evaluate_transfer_learning(
    transfer_dir: str,
    reverse_transfer_dir: str = None,
    seed: int = 42,
    device: str = 'cuda'
):
    """
    Complete evaluation pipeline comparing transfer learning with baseline.
    
    Args:
        transfer_dir: Directory with ORIEN→TCGA transfer learning results
        reverse_transfer_dir: Directory with TCGA→ORIEN transfer (optional)
        seed: Random seed
        device: Device
    """
    
    print(f"\n{'='*60}")
    print("TRANSFER LEARNING EVALUATION - CHAPTER 4")
    print(f"{'='*60}")
    print(f"Transfer directory: {transfer_dir}")
    print(f"Random seed: {seed}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f'results/transfer_learning_evaluation_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    set_all_seeds(seed)
    
    # ========================================
    # Load Data
    # ========================================
    
    print("Loading cohort data...")
    tcga_data = load_cohort_data('tcga')
    orien_data = load_cohort_data('orien')
    print(f"✓ TCGA: {tcga_data['expression'].shape}")
    print(f"✓ ORIEN: {orien_data['expression'].shape}\n")
    
    # ========================================
    # Load Hyperparameters
    # ========================================
    
    # Infer hyperparameter paths from transfer directory structure
    transfer_path = Path(transfer_dir)
    
    # Try to find best_params.json files
    # Assume they're in the typical Chapter 3 location
    tcga_params_file = "results/hyperparam_FIXED_tcga_20251109_194909/best_params.json"
    orien_params_file = "results/hyperparam_FIXED_orien_20251109_195430/best_params.json"
    
    if not Path(tcga_params_file).exists() or not Path(orien_params_file).exists():
        print(f"⚠️  Could not find hyperparameter files in default locations")
        print(f"Please provide paths via --tcga_params and --orien_params")
        return
    
    with open(tcga_params_file, 'r') as f:
        tcga_params = parse_optuna_params(json.load(f))
    
    with open(orien_params_file, 'r') as f:
        orien_params = parse_optuna_params(json.load(f))
    
    print(f"Loaded hyperparameters:")
    print(f"  TCGA: {tcga_params['hidden_sizes']}")
    print(f"  ORIEN: {orien_params['hidden_sizes']}\n")
    
    # ========================================
    # Load Transfer-Learned Models
    # ========================================
    
    print("Loading transfer-learned models...")
    
    # ORIEN→TCGA transfer
    tcga_transfer_file = transfer_path / f"tcga_finetuned_seed{seed}.pth"
    
    if not tcga_transfer_file.exists():
        print(f"✗ Transfer model not found: {tcga_transfer_file}")
        return
    
    tcga_transfer_checkpoint = torch.load(tcga_transfer_file, map_location='cpu')
    
    # Create model and load weights
    # CRITICAL: Transfer-learned model uses SOURCE architecture (ORIEN)
    # because it was pre-trained on ORIEN, then fine-tuned on TCGA
    tcga_transfer_model = ElasticDeepSurv(
        n_features=tcga_data['expression'].shape[0],
        hidden_sizes=orien_params['hidden_sizes'],  # ← Use ORIEN architecture!
        dropout=orien_params.get('dropout', 0.3),
        l1_ratio=orien_params.get('l1_ratio', 0.7),
        alpha=orien_params.get('alpha', 0.01)
    )
    
    tcga_transfer_model.load_state_dict(tcga_transfer_checkpoint['model_state_dict'])
    print(f"✓ Loaded ORIEN→TCGA transfer model")
    print(f"  Architecture: {orien_params['hidden_sizes']} (inherited from ORIEN)")
    
    # Get training metrics from checkpoint
    tcga_transfer_train_cindex = tcga_transfer_checkpoint['finetune_metrics']['c_index']
    
    # Load reverse direction if provided (TCGA→ORIEN)
    orien_transfer_model = None
    orien_transfer_train_cindex = None
    
    if reverse_transfer_dir:
        print(f"\nLoading reverse transfer model (TCGA→ORIEN)...")
        reverse_path = Path(reverse_transfer_dir)
        orien_transfer_file = reverse_path / f"orien_finetuned_seed{seed}.pth"
        
        if orien_transfer_file.exists():
            orien_transfer_checkpoint = torch.load(orien_transfer_file, map_location='cpu')
            
            # Create model with TCGA architecture (source for this direction)
            orien_transfer_model = ElasticDeepSurv(
                n_features=orien_data['expression'].shape[0],
                hidden_sizes=tcga_params['hidden_sizes'],  # Use TCGA architecture
                dropout=tcga_params.get('dropout', 0.3),
                l1_ratio=tcga_params.get('l1_ratio', 0.7),
                alpha=tcga_params.get('alpha', 0.01)
            )
            
            orien_transfer_model.load_state_dict(orien_transfer_checkpoint['model_state_dict'])
            orien_transfer_train_cindex = orien_transfer_checkpoint['finetune_metrics']['c_index']
            
            print(f"✓ Loaded TCGA→ORIEN transfer model")
            print(f"  Architecture: {tcga_params['hidden_sizes']} (inherited from TCGA)")
        else:
            print(f"⚠️  Reverse transfer model not found: {orien_transfer_file}")
    else:
        print(f"\n⚠️  Reverse transfer directory not provided, skipping TCGA→ORIEN evaluation")
    
    # ========================================
    # Train Baseline Models (From Scratch)
    # ========================================
    
    print("\nTraining baseline models from scratch...")
    print("(This provides Chapter 3-style comparison)")
    
    print("\n  Training TCGA baseline (from scratch)...")
    tcga_baseline_model, tcga_baseline_train_cindex = train_baseline_model(
        data=tcga_data,
        params=tcga_params,
        n_epochs=100,
        learning_rate=1e-4,
        seed=seed,
        device=device
    )
    print(f"    ✓ TCGA baseline training C-index: {tcga_baseline_train_cindex:.4f}")
    
    print("\n  Training ORIEN baseline (from scratch)...")
    orien_baseline_model, orien_baseline_train_cindex = train_baseline_model(
        data=orien_data,
        params=orien_params,
        n_epochs=100,
        learning_rate=1e-4,
        seed=seed,
        device=device
    )
    print(f"    ✓ ORIEN baseline training C-index: {orien_baseline_train_cindex:.4f}")
    
    # ========================================
    # Cross-Cohort Evaluation
    # ========================================
    
    print(f"\n{'='*60}")
    print("CROSS-COHORT EVALUATION")
    print(f"{'='*60}\n")
    
    # Baseline: TCGA model tested on ORIEN
    print("Evaluating TCGA baseline on ORIEN...")
    baseline_tcga_on_orien = evaluate_model_cross_cohort(
        tcga_baseline_model, orien_data, device
    )
    print(f"  Baseline C-index: {baseline_tcga_on_orien:.4f}")
    
    # Transfer: TCGA transfer model tested on ORIEN
    print("\nEvaluating TCGA transfer model on ORIEN...")
    transfer_tcga_on_orien = evaluate_model_cross_cohort(
        tcga_transfer_model, orien_data, device
    )
    print(f"  Transfer C-index: {transfer_tcga_on_orien:.4f}")
    
    # Baseline: ORIEN model tested on TCGA
    print("\nEvaluating ORIEN baseline on TCGA...")
    baseline_orien_on_tcga = evaluate_model_cross_cohort(
        orien_baseline_model, tcga_data, device
    )
    print(f"  Baseline C-index: {baseline_orien_on_tcga:.4f}")
    
    # Transfer: ORIEN transfer model tested on TCGA (if available)
    transfer_orien_on_tcga = None
    if orien_transfer_model:
        print("\nEvaluating ORIEN transfer model on TCGA...")
        transfer_orien_on_tcga = evaluate_model_cross_cohort(
            orien_transfer_model, tcga_data, device
        )
        print(f"  Transfer C-index: {transfer_orien_on_tcga:.4f}")
    
    # ========================================
    # Compute Improvements
    # ========================================
    
    improvement_tcga = transfer_tcga_on_orien - baseline_tcga_on_orien
    improvement_pct_tcga = (improvement_tcga / baseline_tcga_on_orien) * 100
    
    improvement_orien = None
    improvement_pct_orien = None
    avg_improvement = None
    
    if transfer_orien_on_tcga:
        improvement_orien = transfer_orien_on_tcga - baseline_orien_on_tcga
        improvement_pct_orien = (improvement_orien / baseline_orien_on_tcga) * 100
        
        # Compute bidirectional average
        baseline_avg = (baseline_tcga_on_orien + baseline_orien_on_tcga) / 2
        transfer_avg = (transfer_tcga_on_orien + transfer_orien_on_tcga) / 2
        avg_improvement = transfer_avg - baseline_avg
    
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}\n")
    
    print("Direction 1: TCGA model tested on ORIEN (ORIEN→TCGA transfer)")
    print(f"  Baseline (from scratch): {baseline_tcga_on_orien:.4f}")
    print(f"  Transfer learning:       {transfer_tcga_on_orien:.4f}")
    print(f"  Improvement:             {improvement_tcga:+.4f} ({improvement_pct_tcga:+.1f}%)")
    
    print("\nDirection 2: ORIEN model tested on TCGA (TCGA→ORIEN transfer)")
    print(f"  Baseline (from scratch): {baseline_orien_on_tcga:.4f}")
    if transfer_orien_on_tcga:
        print(f"  Transfer learning:       {transfer_orien_on_tcga:.4f}")
        print(f"  Improvement:             {improvement_orien:+.4f} ({improvement_pct_orien:+.1f}%)")
    else:
        print(f"  Transfer learning:       Not computed")
    
    if avg_improvement is not None:
        print(f"\nBidirectional Average:")
        print(f"  Baseline: {baseline_avg:.4f}")
        print(f"  Transfer: {transfer_avg:.4f}")
        print(f"  Improvement: {avg_improvement:+.4f} ({avg_improvement/baseline_avg*100:+.1f}%)")
    
    # ========================================
    # Compare with Chapter 3 Best Results
    # ========================================
    
    print(f"\n{'='*60}")
    print("COMPARISON WITH CHAPTER 3 (308 genes)")
    print(f"{'='*60}\n")
    
    # Your Chapter 3 best results (from the table you provided)
    chapter3_best = {
        'k95_avg': 0.6311,
        'k95_tcga_on_orien': 0.6255,
        'k95_orien_on_tcga': 0.6367,
        'k140_orien_on_tcga': 0.6979  # Best single direction
    }
    
    print(f"Chapter 3 Best (k=95, 28 genes):")
    print(f"  Average C-index: {chapter3_best['k95_avg']:.4f}")
    print(f"  TCGA→ORIEN: {chapter3_best['k95_tcga_on_orien']:.4f}")
    print(f"  ORIEN→TCGA: {chapter3_best['k95_orien_on_tcga']:.4f}")
    
    print(f"\nChapter 4 Transfer Learning (308 genes):")
    print(f"  TCGA→ORIEN: {transfer_tcga_on_orien:.4f}")
    
    if transfer_tcga_on_orien > chapter3_best['k95_tcga_on_orien']:
        improvement = transfer_tcga_on_orien - chapter3_best['k95_tcga_on_orien']
        print(f"  ✓ Improvement: +{improvement:.4f} (+{improvement/chapter3_best['k95_tcga_on_orien']*100:.1f}%)")
    else:
        decline = chapter3_best['k95_tcga_on_orien'] - transfer_tcga_on_orien
        print(f"  ✗ Decline: -{decline:.4f} (-{decline/chapter3_best['k95_tcga_on_orien']*100:.1f}%)")
    
    # ========================================
    # Save Results
    # ========================================
    
    results = {
        'seed': seed,
        'n_genes': tcga_data['expression'].shape[0],
        'baseline': {
            'tcga_on_orien': baseline_tcga_on_orien,
            'orien_on_tcga': baseline_orien_on_tcga,
            'tcga_train_cindex': tcga_baseline_train_cindex,
            'orien_train_cindex': orien_baseline_train_cindex
        },
        'transfer': {
            'tcga_on_orien': transfer_tcga_on_orien,
            'tcga_train_cindex': tcga_transfer_train_cindex
        },
        'improvement': {
            'tcga_on_orien_absolute': improvement_tcga,
            'tcga_on_orien_percent': improvement_pct_tcga
        },
        'chapter3_comparison': {
            'chapter3_best_avg': chapter3_best['k95_avg'],
            'chapter4_transfer': transfer_tcga_on_orien,
            'difference': transfer_tcga_on_orien - chapter3_best['k95_tcga_on_orien']
        }
    }
    
    # Save to JSON
    results_file = output_dir / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {results_file}")
    print(f"{'='*60}\n")
    
    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Transfer Learning vs. Baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--transfer_dir', type=str, required=True,
                       help='Directory containing transfer learning results (ORIEN→TCGA)')
    parser.add_argument('--reverse_transfer_dir', type=str, default=None,
                       help='Directory containing reverse transfer (TCGA→ORIEN)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use (default: cuda)')
    
    args = parser.parse_args()
    
    results = evaluate_transfer_learning(
        transfer_dir=args.transfer_dir,
        reverse_transfer_dir=args.reverse_transfer_dir,
        seed=args.seed,
        device=args.device
    )