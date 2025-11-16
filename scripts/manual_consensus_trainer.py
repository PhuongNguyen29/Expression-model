#!/usr/bin/env python3
"""
Script: manual_consensus_trainer.py
Purpose: Train transfer learning models with consensus genes from k-sweep
Status: ACTIVE (Chapter 4 - Manual consensus gene evaluation)
Author: Phuong
Created: 2024-11-15

This script allows quick manual testing of specific k values from the k-sweep
analysis. It uses the proven ElasticDeepSurvTrainer infrastructure from your
existing transfer_learning_trainer.py.

Usage:
    # Test k=120 (55 consensus genes)
    python scripts/manual_consensus_trainer.py --k 120 --seed 42
    
    # Test k=100 (37 consensus genes)
    python scripts/manual_consensus_trainer.py --k 100 --seed 42
    
    # Test k=140 (75 consensus genes)
    python scripts/manual_consensus_trainer.py --k 140 --seed 42
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import your proven classes
from src.models.elastic_deepsurv import ElasticDeepSurv, ElasticDeepSurvTrainer


def set_all_seeds(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


def load_consensus_genes(k, gene_lists_dir='results/biomarker_ksweep_transfer/gene_lists'):
    """
    Load consensus genes for a specific k value.
    
    Args:
        k: K value (e.g., 120)
        gene_lists_dir: Directory containing gene lists
        
    Returns:
        List of gene names
    """
    gene_file = Path(gene_lists_dir) / f'k{k}_bidirectional.txt'
    
    if not gene_file.exists():
        raise FileNotFoundError(
            f"Gene list not found: {gene_file}\n"
            f"Make sure you've run the k-sweep analysis first."
        )
    
    with open(gene_file, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]
    
    return genes


def load_cohort_data_with_genes(cohort_name, consensus_genes):
    """
    Load cohort data filtered to specific consensus genes.
    
    Based on your transfer_learning_trainer.py load_cohort_data function.
    
    Args:
        cohort_name: 'tcga' or 'orien'
        consensus_genes: List of gene names to keep
        
    Returns:
        Dict with 'expression' (genes × samples) and 'survival' DataFrames
    """
    # Load batch-corrected expression data
    if cohort_name.lower() == 'tcga':
        expr_file = "data/raw/tcga_batch_corrected_2sv.csv"
        surv_file = "data/processed/surv_tcga_harmonized.csv"
    else:  # orien
        expr_file = "data/raw/orien_batch_corrected.csv"
        surv_file = "data/processed/surv_orien_harmonized.csv"
    
    print(f"Loading {cohort_name.upper()} data...")
    
    # Load expression
    expression = pd.read_csv(expr_file, index_col=0)
    print(f"  Raw expression: {expression.shape[0]} genes × {expression.shape[1]} samples")
    
    # Filter to consensus genes
    available_genes = [g for g in consensus_genes if g in expression.index]
    missing_genes = set(consensus_genes) - set(available_genes)
    
    if missing_genes:
        print(f"  ⚠️  Missing {len(missing_genes)} genes from consensus list")
    
    expression = expression.loc[available_genes]
    print(f"  After filtering: {len(expression)} genes × {expression.shape[1]} samples")
    
    # Load survival data
    survival = pd.read_csv(surv_file)
    if 'sampleID' in survival.columns:
        survival = survival.set_index('sampleID')
    
    # Align samples
    common_samples = list(set(expression.columns) & set(survival.index))
    common_samples = sorted(common_samples)
    
    expression = expression[common_samples]
    survival = survival.loc[common_samples]
    
    print(f"  Final: {expression.shape[0]} genes × {len(common_samples)} samples")
    print(f"  Events: {survival['event'].sum()} ({100*survival['event'].mean():.1f}%)\n")
    
    return {
        'expression': expression,
        'survival': survival
    }


def train_source_model(
    source_data,
    hyperparams,
    n_epochs=100,
    learning_rate=1e-4,
    seed=42,
    device='cuda'
):
    """
    Pre-train model on source cohort.
    
    Based on your transfer_learning_trainer.py train_source_model function.
    """
    print(f"\n{'='*80}")
    print("PHASE 1: PRE-TRAINING ON SOURCE COHORT")
    print(f"{'='*80}\n")
    
    set_all_seeds(seed)
    
    # Create dataset
    from src.data.dataset import SurvivalDataset
    from torch.utils.data import DataLoader
    from src.utils.batch_samplers import StratifiedBatchSampler
    
    source_dataset = SurvivalDataset(
        source_data['expression'],
        source_data['survival']
    )
    
    print(f"Source dataset: {len(source_dataset)} samples (full cohort)")
    print(f"  Features: {source_dataset.n_features} genes")
    print(f"  Event rate: {source_dataset.y_event.mean():.2%}")
    
    # Create data loader
    train_sampler = StratifiedBatchSampler(
        events=source_dataset.y_event,
        batch_size=hyperparams.get('batch_size', 32),
        shuffle=True
    )
    
    train_loader = DataLoader(
        source_dataset,
        batch_sampler=train_sampler,
        num_workers=0,
        pin_memory=True if device == 'cuda' else False
    )
    
    # Create model
    model = ElasticDeepSurv(
        n_features=source_dataset.n_features,
        hidden_sizes=hyperparams.get('hidden_sizes', [128, 64]),
        dropout=hyperparams.get('dropout', 0.3),
        l1_ratio=hyperparams.get('l1_ratio', 0.7),
        alpha=hyperparams.get('alpha', 0.01)
    )
    
    print(f"\nModel architecture:")
    print(f"  Input: {source_dataset.n_features} genes")
    print(f"  Hidden: {hyperparams.get('hidden_sizes', [128, 64])}")
    print(f"  Dropout: {hyperparams.get('dropout', 0.3)}")
    
    # Create trainer
    trainer = ElasticDeepSurvTrainer(
        model=model,
        learning_rate=learning_rate,
        weight_decay=0.0,
        scheduler_patience=10,
        device=device
    )
    
    print(f"\nTraining configuration:")
    print(f"  Epochs: {n_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Device: {device}\n")
    
    # Train
    history = trainer.fit(
        train_loader=train_loader,
        valid_loader=None,
        n_epochs=n_epochs,
        early_stopping_patience=20,
        verbose=True
    )
    
    # Get metrics
    final_cindex = history['train_cindex'][-1]
    best_epoch = history.get('best_epoch', len(history['train_cindex']))
    
    print(f"\n{'='*80}")
    print("PRE-TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Best epoch: {best_epoch}")
    print(f"Final C-index: {final_cindex:.4f}")
    print(f"{'='*80}\n")
    
    return model, {'c_index': final_cindex, 'best_epoch': best_epoch, 'history': history}


def finetune_target_model(
    pretrained_model,
    target_data,
    hyperparams,
    n_epochs=50,
    learning_rate=1e-5,
    seed=42,
    device='cuda'
):
    """
    Fine-tune model on target cohort.
    
    Based on your transfer_learning_trainer.py finetune_target_model function.
    """
    print(f"\n{'='*80}")
    print("PHASE 2: FINE-TUNING ON TARGET COHORT")
    print(f"{'='*80}\n")
    
    set_all_seeds(seed)
    
    # Create dataset
    from src.data.dataset import SurvivalDataset
    from torch.utils.data import DataLoader
    from src.utils.batch_samplers import StratifiedBatchSampler
    
    target_dataset = SurvivalDataset(
        target_data['expression'],
        target_data['survival']
    )
    
    print(f"Target dataset: {len(target_dataset)} samples (full cohort)")
    print(f"  Features: {target_dataset.n_features} genes")
    print(f"  Event rate: {target_dataset.y_event.mean():.2%}")
    
    # Create data loader
    train_sampler = StratifiedBatchSampler(
        events=target_dataset.y_event,
        batch_size=hyperparams.get('batch_size', 32),
        shuffle=True
    )
    
    train_loader = DataLoader(
        target_dataset,
        batch_sampler=train_sampler,
        num_workers=0,
        pin_memory=True if device == 'cuda' else False
    )
    
    # Create trainer with pre-trained model
    trainer = ElasticDeepSurvTrainer(
        model=pretrained_model,
        learning_rate=learning_rate,
        weight_decay=0.0,
        scheduler_patience=10,
        device=device
    )
    
    print(f"\nFine-tuning configuration:")
    print(f"  Epochs: {n_epochs}")
    print(f"  Learning rate: {learning_rate} (lower for fine-tuning)")
    print(f"  Device: {device}\n")
    
    # Fine-tune
    history = trainer.fit(
        train_loader=train_loader,
        valid_loader=None,
        n_epochs=n_epochs,
        early_stopping_patience=15,
        verbose=True
    )
    
    # Get metrics
    final_cindex = history['train_cindex'][-1]
    best_epoch = history.get('best_epoch', len(history['train_cindex']))
    
    print(f"\n{'='*80}")
    print("FINE-TUNING COMPLETE")
    print(f"{'='*80}")
    print(f"Best epoch: {best_epoch}")
    print(f"Final C-index: {final_cindex:.4f}")
    print(f"{'='*80}\n")
    
    return pretrained_model, {'c_index': final_cindex, 'best_epoch': best_epoch, 'history': history}


def run_bidirectional_transfer(
    k,
    gene_lists_dir,
    output_dir,
    seed=42,
    pretrain_epochs=100,
    finetune_epochs=50,
    pretrain_lr=1e-4,
    finetune_lr=1e-5,
    device='cuda'
):
    """
    Run bidirectional transfer learning for a specific k value.
    
    Args:
        k: K value (number of top genes extracted in k-sweep)
        gene_lists_dir: Directory with consensus gene lists
        output_dir: Output directory for results
        seed: Random seed
        pretrain_epochs: Epochs for pre-training
        finetune_epochs: Epochs for fine-tuning
        pretrain_lr: Learning rate for pre-training
        finetune_lr: Learning rate for fine-tuning
        device: 'cuda' or 'cpu'
    """
    # Setup
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print(f"\n{'='*80}")
    print(f"CONSENSUS GENE TRANSFER LEARNING EVALUATION")
    print(f"{'='*80}")
    print(f"K value: {k}")
    print(f"Gene lists directory: {gene_lists_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {seed}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")
    
    # Load consensus genes
    print(f"Loading consensus genes for k={k}...")
    consensus_genes = load_consensus_genes(k, gene_lists_dir)
    n_genes = len(consensus_genes)
    print(f"✓ Loaded {n_genes} consensus genes\n")
    
    # Save gene list
    with open(output_path / 'consensus_genes.txt', 'w') as f:
        f.write('\n'.join(consensus_genes))
    
    # Load data for both cohorts
    print(f"{'='*80}")
    print("LOADING DATA")
    print(f"{'='*80}\n")
    
    tcga_data = load_cohort_data_with_genes('tcga', consensus_genes)
    orien_data = load_cohort_data_with_genes('orien', consensus_genes)
    
    # Standard hyperparameters (from your Chapter 3 tuning)
    hyperparams = {
        'hidden_sizes': [128, 64],
        'dropout': 0.3,
        'batch_size': 32,
        'l1_ratio': 0.7,
        'alpha': 0.01
    }
    
    results = {
        'k': k,
        'n_genes': n_genes,
        'consensus_genes': consensus_genes,
        'seed': seed,
        'timestamp': timestamp
    }
    
    # ========================================
    # Direction 1: TCGA → ORIEN
    # ========================================
    
    print(f"\n{'='*80}")
    print(f"DIRECTION 1: TCGA → ORIEN")
    print(f"{'='*80}\n")
    
    tcga_to_orien_dir = output_path / 'tcga_to_orien'
    tcga_to_orien_dir.mkdir(exist_ok=True)
    
    # Pre-train on TCGA
    pretrained_tcga, pretrain_metrics_tcga = train_source_model(
        source_data=tcga_data,
        hyperparams=hyperparams,
        n_epochs=pretrain_epochs,
        learning_rate=pretrain_lr,
        seed=seed,
        device=device
    )
    
    # Fine-tune on ORIEN
    finetuned_tcga_orien, finetune_metrics_orien = finetune_target_model(
        pretrained_model=pretrained_tcga,
        target_data=orien_data,
        hyperparams=hyperparams,
        n_epochs=finetune_epochs,
        learning_rate=finetune_lr,
        seed=seed,
        device=device
    )
    
    # Save model
    torch.save({
        'model_state_dict': finetuned_tcga_orien.state_dict(),
        'n_features': n_genes,
        'pretrain_cindex': pretrain_metrics_tcga['c_index'],
        'finetune_cindex': finetune_metrics_orien['c_index'],
        'seed': seed
    }, tcga_to_orien_dir / f'model_seed{seed}.pth')
    
    results['tcga_to_orien'] = {
        'pretrain_cindex': pretrain_metrics_tcga['c_index'],
        'finetune_cindex': finetune_metrics_orien['c_index']
    }
    
    # ========================================
    # Direction 2: ORIEN → TCGA
    # ========================================
    
    print(f"\n{'='*80}")
    print(f"DIRECTION 2: ORIEN → TCGA")
    print(f"{'='*80}\n")
    
    orien_to_tcga_dir = output_path / 'orien_to_tcga'
    orien_to_tcga_dir.mkdir(exist_ok=True)
    
    # Pre-train on ORIEN
    pretrained_orien, pretrain_metrics_orien = train_source_model(
        source_data=orien_data,
        hyperparams=hyperparams,
        n_epochs=pretrain_epochs,
        learning_rate=pretrain_lr,
        seed=seed,
        device=device
    )
    
    # Fine-tune on TCGA
    finetuned_orien_tcga, finetune_metrics_tcga = finetune_target_model(
        pretrained_model=pretrained_orien,
        target_data=tcga_data,
        hyperparams=hyperparams,
        n_epochs=finetune_epochs,
        learning_rate=finetune_lr,
        seed=seed,
        device=device
    )
    
    # Save model
    torch.save({
        'model_state_dict': finetuned_orien_tcga.state_dict(),
        'n_features': n_genes,
        'pretrain_cindex': pretrain_metrics_orien['c_index'],
        'finetune_cindex': finetune_metrics_tcga['c_index'],
        'seed': seed
    }, orien_to_tcga_dir / f'model_seed{seed}.pth')
    
    results['orien_to_tcga'] = {
        'pretrain_cindex': pretrain_metrics_orien['c_index'],
        'finetune_cindex': finetune_metrics_tcga['c_index']
    }
    
    # ========================================
    # Summary
    # ========================================
    
    avg_cindex = (
        finetune_metrics_orien['c_index'] + 
        finetune_metrics_tcga['c_index']
    ) / 2
    
    results['average_cindex'] = avg_cindex
    
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS FOR k={k}")
    print(f"{'='*80}")
    print(f"Number of genes: {n_genes}")
    print(f"\nTCGA → ORIEN:")
    print(f"  Pre-training (TCGA): {pretrain_metrics_tcga['c_index']:.4f}")
    print(f"  Fine-tuning (ORIEN): {finetune_metrics_orien['c_index']:.4f}")
    print(f"\nORIEN → TCGA:")
    print(f"  Pre-training (ORIEN): {pretrain_metrics_orien['c_index']:.4f}")
    print(f"  Fine-tuning (TCGA): {finetune_metrics_tcga['c_index']:.4f}")
    print(f"\nAverage C-index: {avg_cindex:.4f}")
    print(f"{'='*80}\n")
    
    # Save results
    with open(output_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to: {output_path}/results.json")
    print(f"✓ Models saved to: {output_path}/*/model_seed{seed}.pth")
    print(f"✓ Gene list saved to: {output_path}/consensus_genes.txt\n")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Manual consensus gene transfer learning evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test k=120 (55 consensus genes - most promising)
  python scripts/manual_consensus_trainer.py --k 120 --seed 42
  
  # Test k=100 (37 consensus genes - balanced)
  python scripts/manual_consensus_trainer.py --k 100 --seed 42
  
  # Test k=140 (75 consensus genes - high stability)
  python scripts/manual_consensus_trainer.py --k 140 --seed 42
  
  # Custom output directory
  python scripts/manual_consensus_trainer.py --k 120 \\
      --output_dir results/manual_k120_test \\
      --seed 42
        """
    )
    
    parser.add_argument('--k', type=int, required=True,
                       help='K value from k-sweep (e.g., 120)')
    parser.add_argument('--gene_lists_dir', type=str,
                       default='results/biomarker_ksweep_transfer/gene_lists',
                       help='Directory with consensus gene lists')
    parser.add_argument('--output_dir', type=str,
                       default=None,
                       help='Output directory (default: results/manual_consensus_k{K})')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--pretrain_epochs', type=int, default=100,
                       help='Pre-training epochs (default: 100)')
    parser.add_argument('--finetune_epochs', type=int, default=50,
                       help='Fine-tuning epochs (default: 50)')
    parser.add_argument('--pretrain_lr', type=float, default=1e-4,
                       help='Pre-training learning rate (default: 1e-4)')
    parser.add_argument('--finetune_lr', type=float, default=1e-5,
                       help='Fine-tuning learning rate (default: 1e-5)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device (default: cuda)')
    
    args = parser.parse_args()
    
    # Set default output directory
    if args.output_dir is None:
        args.output_dir = f'results/manual_consensus_k{args.k}'
    
    # Run evaluation
    results = run_bidirectional_transfer(
        k=args.k,
        gene_lists_dir=args.gene_lists_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        pretrain_lr=args.pretrain_lr,
        finetune_lr=args.finetune_lr,
        device=args.device
    )
    
    print(f"\n✅ Evaluation complete for k={args.k}!")
    print(f"📁 Results in: {args.output_dir}/")
    print(f"\n🎯 Average C-index: {results['average_cindex']:.4f}")
    print(f"\nNext steps:")
    print(f"  1. Test other k values (e.g., k=100, k=140)")
    print(f"  2. Compare C-index across k values")
    print(f"  3. Select optimal k for dissertation")
    print(f"  4. Run multi-seed validation with optimal k\n")
