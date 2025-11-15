#!/usr/bin/env python3
"""
Script: transfer_learning_trainer.py
Purpose: Pre-train models on source cohort and fine-tune on target cohort for Chapter 4
Status: ACTIVE (Chapter 4 - Transfer Learning)
Author: Phuong
Created: 2024-11-14

Main Functions:
- save_pretrained_model(): Save pre-trained model weights and metadata
- load_pretrained_model(): Load pre-trained weights into target model
- train_source_model(): Pre-train on large source cohort (ORIEN)
- finetune_target_model(): Fine-tune on small target cohort (TCGA)

Dependencies:
- src/models/elastic_deepsurv.py
- src/data/data_loader.py
- Best hyperparameters from Chapter 3 (hyperparam_tuning_elastic_FIXED.py)

Usage:
    python scripts/transfer_learning_trainer.py \
        --source_cohort orien \
        --target_cohort tcga \
        --source_params results/hyperparam_FIXED_orien_20251109_195430/best_params.json \
        --target_params results/hyperparam_FIXED_tcga_20251109_194909/best_params.json \
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

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Note: We don't need ModelFactory or DataFactory here because we:
# - Load data directly using pandas (see load_cohort_data function)
# - Create models directly using ElasticDeepSurv class
# These imports will be done inside the functions where needed


# ============================================================================
# STEP 1: CHECKPOINT MANAGEMENT FUNCTIONS
# ============================================================================

def save_pretrained_model(model, save_dir, cohort_name, hyperparams, 
                          train_metrics, seed):
    """
    Save pre-trained model with all necessary metadata for transfer learning.
    
    This function saves:
    1. Model state dict (weights and biases)
    2. Model architecture info (for verification during loading)
    3. Training hyperparameters used
    4. Training metrics (C-index, loss history)
    5. Random seed for reproducibility
    
    Args:
        model: Trained ElasticDeepSurv model
        save_dir: Directory to save checkpoint
        cohort_name: Name of source cohort (e.g., 'orien', 'tcga')
        hyperparams: Dict of hyperparameters used for training
        train_metrics: Dict of training metrics (c_index, loss, etc.)
        seed: Random seed used for training
        
    Returns:
        checkpoint_path: Path to saved checkpoint file
        
    Reference:
        Yosinski et al. (2014): "How transferable are features in deep neural networks?"
        - Emphasizes saving complete architecture info for successful transfer
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create checkpoint dictionary
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_architecture': {
            'n_features': model.network[0].in_features,
            'hidden_sizes': hyperparams.get('hidden_sizes', [128]),
            'dropout': hyperparams.get('dropout', 0.3),
            'l1_penalty': hyperparams.get('l1_penalty', 0.0),
            'l2_penalty': hyperparams.get('l2_penalty', 0.0)
        },
        'hyperparameters': hyperparams,
        'train_metrics': train_metrics,
        'cohort': cohort_name,
        'seed': seed,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
    }
    
    # Save checkpoint
    checkpoint_path = os.path.join(
        save_dir, 
        f'{cohort_name}_pretrained_seed{seed}.pth'
    )
    torch.save(checkpoint, checkpoint_path)
    
    print(f"\n{'='*60}")
    print(f"Pre-trained model saved: {checkpoint_path}")
    print(f"{'='*60}")
    print(f"Source cohort: {cohort_name}")
    print(f"Architecture: {checkpoint['model_architecture']['n_features']} genes → " +
          f"{checkpoint['model_architecture']['hidden_sizes']} hidden")
    print(f"Training C-index: {train_metrics.get('c_index', 'N/A'):.4f}")
    print(f"Random seed: {seed}")
    print(f"{'='*60}\n")
    
    return checkpoint_path


def load_pretrained_model(checkpoint_path, target_model):
    """
    Load pre-trained weights into target model with architecture verification.
    
    This is the CORE transfer learning function. It:
    1. Loads pre-trained checkpoint
    2. Verifies architecture compatibility (critical!)
    3. Transfers weights from source to target model
    4. Returns metadata for logging
    
    Args:
        checkpoint_path: Path to pre-trained model checkpoint
        target_model: Initialized target model to receive weights
        
    Returns:
        loaded_metadata: Dict containing checkpoint metadata
        
    Raises:
        ValueError: If architectures are incompatible
        
    Reference:
        Ganchev et al. (2011): "Transfer learning of classification rules"
        - Transfer requires identical feature spaces
        
    Critical Note:
        Source and target MUST have same n_features (308 genes in your case)
        Hidden layer sizes should also match for full weight transfer
    """
    print(f"\n{'='*60}")
    print(f"Loading pre-trained model from: {checkpoint_path}")
    print(f"{'='*60}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract architecture info
    source_arch = checkpoint['model_architecture']
    target_arch = {
        'n_features': target_model.network[0].in_features,
        'hidden_sizes': [layer.out_features for layer in target_model.network 
                        if isinstance(layer, nn.Linear)][:-1]  # Exclude output layer
    }
    
    # Verify architecture compatibility
    print(f"\nArchitecture Verification:")
    print(f"  Source: {source_arch['n_features']} genes → {source_arch['hidden_sizes']} hidden")
    print(f"  Target: {target_arch['n_features']} genes → {target_arch['hidden_sizes']} hidden")
    
    if source_arch['n_features'] != target_arch['n_features']:
        raise ValueError(
            f"Architecture mismatch: Source has {source_arch['n_features']} features, "
            f"but target has {target_arch['n_features']} features. "
            f"Transfer learning requires IDENTICAL input dimensions."
        )
    
    if source_arch['hidden_sizes'] != target_arch['hidden_sizes']:
        print(f"\n⚠️  WARNING: Hidden layer sizes differ!")
        print(f"  This may reduce transfer effectiveness.")
        print(f"  Consider using matching architectures for both cohorts.\n")
    
    # Load pre-trained weights into target model
    target_model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"\n✓ Successfully transferred weights from {checkpoint['cohort']} cohort")
    print(f"✓ Source model C-index: {checkpoint['train_metrics'].get('c_index', 'N/A'):.4f}")
    print(f"✓ Pre-training seed: {checkpoint['seed']}")
    print(f"{'='*60}\n")
    
    # Return metadata for logging
    return {
        'source_cohort': checkpoint['cohort'],
        'source_c_index': checkpoint['train_metrics'].get('c_index'),
        'source_seed': checkpoint['seed'],
        'architecture': source_arch
    }


def verify_weight_transfer(source_model, target_model, layer_idx=0, n_params=5):
    """
    Verify that weights were actually transferred (debugging utility).
    
    Compares a few parameters between source and target models to confirm
    successful weight transfer. Useful for debugging.
    
    Args:
        source_model: Pre-trained source model
        target_model: Target model after weight loading
        layer_idx: Which layer to check (default: 0 = first layer)
        n_params: Number of parameters to display (default: 5)
        
    Returns:
        bool: True if weights match, False otherwise
    """
    source_params = list(source_model.parameters())[layer_idx].detach().cpu().numpy()
    target_params = list(target_model.parameters())[layer_idx].detach().cpu().numpy()
    
    print(f"\nWeight Transfer Verification (Layer {layer_idx}):")
    print(f"  Source weights (first {n_params}): {source_params.flatten()[:n_params]}")
    print(f"  Target weights (first {n_params}): {target_params.flatten()[:n_params]}")
    
    if np.allclose(source_params, target_params):
        print(f"  ✓ Weights successfully transferred!")
        return True
    else:
        print(f"  ✗ WARNING: Weights do NOT match!")
        return False


# ============================================================================
# STEP 2: PRE-TRAINING AND FINE-TUNING FUNCTIONS
# ============================================================================

def train_source_model(
    source_data,
    source_params,
    n_epochs=100,
    learning_rate=1e-4,
    seed=42,
    device='cuda',
    output_dir='results/transfer_learning'
):
    """
    Pre-train model on source cohort (typically ORIEN - the larger cohort).
    
    This is STEP 1 of transfer learning: training from scratch on the source
    cohort to learn general survival patterns from genomic data.
    
    Args:
        source_data: Dict with 'expression' (genes × samples) and 'survival' DataFrames
        source_params: Dict with best hyperparameters from Chapter 3
        n_epochs: Number of training epochs (default: 100)
        learning_rate: Learning rate for Adam optimizer (default: 1e-4)
        seed: Random seed for reproducibility
        device: Device to train on ('cuda' or 'cpu')
        output_dir: Directory to save pre-trained model
        
    Returns:
        pretrained_model: Trained model
        train_metrics: Dict with training metrics (c_index, loss, etc.)
        
    Reference:
        Pan & Yang (2010): "A survey on transfer learning"
        - Pre-training on large source dataset provides good initialization
    """
    print(f"\n{'='*60}")
    print("STEP 1: PRE-TRAINING ON SOURCE COHORT")
    print(f"{'='*60}")
    
    # Set random seeds
    set_all_seeds(seed)
    
    # Create datasets
    from src.data.dataset import SurvivalDataset
    from torch.utils.data import DataLoader
    from src.utils.batch_samplers import StratifiedBatchSampler
    
    source_dataset = SurvivalDataset(
        source_data['expression'],
        source_data['survival']
    )
    
    print(f"Source dataset: {len(source_dataset)} samples, "
          f"{source_dataset.n_features} features")
    print(f"Event rate: {source_dataset.y_event.mean():.2%}")
    
    # Create train/validation split
    train_dataset, valid_dataset = source_dataset.create_train_valid_split(
        valid_size=0.2,
        random_seed=seed
    )
    
    # Create data loaders with stratified sampling
    train_sampler = StratifiedBatchSampler(
        events=source_dataset.y_event[train_dataset.indices],
        batch_size=source_params.get('batch_size', 32),
        shuffle=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=0,
        pin_memory=True if device == 'cuda' else False
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=source_params.get('batch_size', 32),
        shuffle=False,
        num_workers=0,
        pin_memory=True if device == 'cuda' else False
    )
    
    # Create model
    from src.models.elastic_deepsurv import ElasticDeepSurv, ElasticDeepSurvTrainer
    
    model = ElasticDeepSurv(
        n_features=source_dataset.n_features,
        hidden_sizes=source_params.get('hidden_sizes', [128]),
        dropout=source_params.get('dropout', 0.3),
        l1_ratio=source_params.get('l1_ratio', 0.7),
        alpha=source_params.get('alpha', 0.01)
    )
    
    print(f"\nModel architecture:")
    print(f"  Input: {source_dataset.n_features} genes")
    print(f"  Hidden: {source_params.get('hidden_sizes', [128])}")
    print(f"  Dropout: {source_params.get('dropout', 0.3)}")
    print(f"  L1 ratio: {source_params.get('l1_ratio', 0.7)}")
    print(f"  Alpha: {source_params.get('alpha', 0.01)}")
    
    # Create trainer
    trainer = ElasticDeepSurvTrainer(
        model=model,
        learning_rate=learning_rate,
        weight_decay=0.0,  # Use elastic net in loss instead
        scheduler_patience=10,
        device=device
    )
    
    print(f"\nTraining configuration:")
    print(f"  Epochs: {n_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Early stopping patience: 20")
    print(f"  Device: {device}")
    
    # Train model
    print(f"\n{'='*60}")
    print("Starting pre-training...")
    print(f"{'='*60}\n")
    
    history = trainer.fit(
        train_loader=train_loader,
        valid_loader=valid_loader,
        n_epochs=n_epochs,
        early_stopping_patience=20,
        verbose=True
    )
    
    # Get final metrics
    final_train_cindex = history['train_cindex'][-1]
    final_valid_cindex = history['valid_c_index'][-1]
    best_epoch = history.get('best_epoch', len(history['train_cindex']))
    
    print(f"\n{'='*60}")
    print("PRE-TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best epoch: {best_epoch}")
    print(f"Final training C-index: {final_train_cindex:.4f}")
    print(f"Final validation C-index: {final_valid_cindex:.4f}")
    print(f"{'='*60}\n")
    
    # Prepare metrics for saving
    train_metrics = {
        'c_index': final_valid_cindex,
        'train_c_index': final_train_cindex,
        'best_epoch': best_epoch,
        'history': history
    }
    
    return model, train_metrics


def finetune_target_model(
    pretrained_model,
    target_data,
    target_params,
    n_epochs=30,
    learning_rate=1e-5,
    seed=42,
    device='cuda',
    output_dir='results/transfer_learning'
):
    """
    Fine-tune pre-trained model on target cohort (typically TCGA - the smaller cohort).
    
    This is STEP 2 of transfer learning: adapting the pre-trained model to the
    target cohort while preserving learned knowledge from source cohort.
    
    CRITICAL: Uses 10× smaller learning rate than pre-training to avoid
    catastrophic forgetting of source knowledge.
    
    Args:
        pretrained_model: Pre-trained model from source cohort
        target_data: Dict with 'expression' and 'survival' DataFrames
        target_params: Dict with target cohort hyperparameters
        n_epochs: Number of fine-tuning epochs (default: 30, 3× less than pre-training)
        learning_rate: Learning rate (default: 1e-5, 10× smaller than pre-training)
        seed: Random seed
        device: Device to train on
        output_dir: Directory to save fine-tuned model
        
    Returns:
        finetuned_model: Fine-tuned model
        finetune_metrics: Dict with fine-tuning metrics
        
    Reference:
        Yosinski et al. (2014): "How transferable are features in deep neural networks?"
        - Recommends smaller learning rate for fine-tuning to preserve transferred features
        - Fewer epochs needed as model starts from good initialization
    """
    print(f"\n{'='*60}")
    print("STEP 2: FINE-TUNING ON TARGET COHORT")
    print(f"{'='*60}")
    
    # Set random seeds
    set_all_seeds(seed)
    
    # Create datasets
    from src.data.dataset import SurvivalDataset
    from torch.utils.data import DataLoader
    from src.utils.batch_samplers import StratifiedBatchSampler
    
    target_dataset = SurvivalDataset(
        target_data['expression'],
        target_data['survival']
    )
    
    print(f"Target dataset: {len(target_dataset)} samples, "
          f"{target_dataset.n_features} features")
    print(f"Event rate: {target_dataset.y_event.mean():.2%}")
    
    # Verify architecture compatibility
    if target_dataset.n_features != pretrained_model.network[0].in_features:
        raise ValueError(
            f"Architecture mismatch! "
            f"Pre-trained model expects {pretrained_model.network[0].in_features} features, "
            f"but target data has {target_dataset.n_features} features. "
            f"Both cohorts must use the same gene set (308 consensus genes)."
        )
    
    print(f"✓ Architecture compatible: {target_dataset.n_features} genes")
    
    # Create train/validation split
    train_dataset, valid_dataset = target_dataset.create_train_valid_split(
        valid_size=0.2,
        random_seed=seed
    )
    
    # Create data loaders
    train_sampler = StratifiedBatchSampler(
        events=target_dataset.y_event[train_dataset.indices],
        batch_size=target_params.get('batch_size', 32),
        shuffle=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=0,
        pin_memory=True if device == 'cuda' else False
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=target_params.get('batch_size', 32),
        shuffle=False,
        num_workers=0,
        pin_memory=True if device == 'cuda' else False
    )
    
    # Create new trainer with SMALLER learning rate
    from src.models.elastic_deepsurv import ElasticDeepSurvTrainer
    
    # Model already has pre-trained weights loaded
    trainer = ElasticDeepSurvTrainer(
        model=pretrained_model,  # Use pre-trained model
        learning_rate=learning_rate,  # 10× smaller than pre-training
        weight_decay=0.0,
        scheduler_patience=10,
        device=device
    )
    
    print(f"\nFine-tuning configuration:")
    print(f"  Epochs: {n_epochs} (vs {100} for pre-training)")
    print(f"  Learning rate: {learning_rate} (vs {1e-4} for pre-training)")
    print(f"  Early stopping patience: 15")
    print(f"  Device: {device}")
    print(f"\n⚠️  Using 10× smaller LR to preserve pre-trained knowledge")
    
    # Fine-tune model
    print(f"\n{'='*60}")
    print("Starting fine-tuning...")
    print(f"{'='*60}\n")
    
    history = trainer.fit(
        train_loader=train_loader,
        valid_loader=valid_loader,
        n_epochs=n_epochs,
        early_stopping_patience=15,  # Slightly less patience for fine-tuning
        verbose=True
    )
    
    # Get final metrics
    final_train_cindex = history['train_cindex'][-1]
    final_valid_cindex = history['valid_c_index'][-1]
    best_epoch = history.get('best_epoch', len(history['train_cindex']))
    
    print(f"\n{'='*60}")
    print("FINE-TUNING COMPLETE")
    print(f"{'='*60}")
    print(f"Best epoch: {best_epoch}")
    print(f"Final training C-index: {final_train_cindex:.4f}")
    print(f"Final validation C-index: {final_valid_cindex:.4f}")
    print(f"{'='*60}\n")
    
    # Prepare metrics
    finetune_metrics = {
        'c_index': final_valid_cindex,
        'train_c_index': final_train_cindex,
        'best_epoch': best_epoch,
        'history': history
    }
    
    return pretrained_model, finetune_metrics


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def set_all_seeds(seed: int):
    """
    Set all random seeds for reproducibility.
    
    Critical for multi-seed validation to ensure fair comparison.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_hyperparameters(params_path: str) -> Dict:
    """
    Load hyperparameters from best_params.json file.
    
    Args:
        params_path: Path to best_params.json from Chapter 3
        
    Returns:
        Dictionary with hyperparameters in model-compatible format
    """
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    # Convert your Optuna hyperparameter format to model format
    params = parse_optuna_params(params)
    
    return params


def parse_optuna_params(optuna_params: Dict) -> Dict:
    """
    Convert Optuna hyperparameter format from Chapter 3 to model-compatible format.
    
    Your Chapter 3 format:
        - n_layers: 1 or 2
        - layer1_size: int (for 1-layer)
        - architecture_2layer: "256-64" (for 2-layer)
    
    Model expects:
        - hidden_sizes: [256] or [256, 64]
    
    Args:
        optuna_params: Dict from best_params.json
        
    Returns:
        Dict with converted parameters
    """
    params = optuna_params.copy()
    
    # Parse architecture based on n_layers
    n_layers = params.get('n_layers', 1)
    
    if n_layers == 1:
        # Single layer: use layer1_size
        layer1_size = params.get('layer1_size', 128)
        hidden_sizes = [layer1_size]
    elif n_layers == 2:
        # Two layers: parse architecture_2layer string
        arch_str = params.get('architecture_2layer', '256-64')
        hidden_sizes = [int(x) for x in arch_str.split('-')]
    else:
        raise ValueError(f"Unsupported n_layers: {n_layers}")
    
    # Add to params dict
    params['hidden_sizes'] = hidden_sizes
    
    # Log the conversion
    print(f"  Parsed architecture: n_layers={n_layers} → hidden_sizes={hidden_sizes}")
    
    return params


def load_cohort_data(cohort_name: str, data_dir: str = 'data/raw') -> Dict:
    """
    Load expression and survival data for a cohort.
    
    CRITICAL: This loads the SAME 308 consensus genes used in Chapter 3
    for fair comparison with baseline results.
    
    Args:
        cohort_name: 'tcga' or 'orien'
        data_dir: Directory containing data files (default: 'data/raw')
        
    Returns:
        Dict with 'expression' and 'survival' DataFrames
    """
    data_path = Path(data_dir)
    
    # Load 308 consensus gene expression data (SAME as Chapter 3)
    # These files should contain only the 308 genes used in your experiments
    expr_file = data_path / f"{cohort_name}_consensus_308.csv"
    
    if not expr_file.exists():
        raise FileNotFoundError(
            f"Could not find consensus gene file: {expr_file}\n"
            f"Expected file with 308 genes used in Chapter 3.\n"
            f"Available files in {data_path}:\n" + 
            "\n".join([f"  - {f.name}" for f in data_path.glob(f"{cohort_name}*.csv")])
        )
    
    expression = pd.read_csv(expr_file, index_col=0)
    
    # Verify we have 308 genes
    if expression.shape[0] != 308:
        raise ValueError(
            f"Expected 308 consensus genes, but found {expression.shape[0]} genes in {expr_file}\n"
            f"Transfer learning requires the SAME gene set as Chapter 3 for fair comparison."
        )
    
    # Load survival data (harmonized files in processed directory)
    surv_file = Path('data/processed') / f"surv_{cohort_name}_harmonized.csv"
    if not surv_file.exists():
        # Try raw directory
        surv_file = data_path / f"surv_{cohort_name}_update.csv" if cohort_name == 'orien' else data_path / f"surv_{cohort_name}.csv"
        if not surv_file.exists():
            raise FileNotFoundError(f"Could not find survival data for {cohort_name}")
    
    survival = pd.read_csv(surv_file)
    if 'sampleID' in survival.columns:
        survival = survival.set_index('sampleID')
    
    print(f"Loaded {cohort_name.upper()} data:")
    print(f"  Expression: {expression.shape} (genes × samples)")
    print(f"  ✓ Verified: 308 consensus genes (same as Chapter 3)")
    print(f"  Survival: {len(survival)} samples")
    print(f"  Events: {survival['event'].sum()} ({100*survival['event'].mean():.1f}%)")
    
    return {
        'expression': expression,
        'survival': survival
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transfer Learning Trainer for Chapter 4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # ORIEN → TCGA transfer (typical use case)
  python scripts/transfer_learning_trainer.py \\
      --source_cohort orien \\
      --target_cohort tcga \\
      --source_params results/hyperparam_FIXED_orien_20251109_195430/best_params.json \\
      --target_params results/hyperparam_FIXED_tcga_20251109_194909/best_params.json \\
      --seed 42

  # TCGA → ORIEN transfer (reverse direction)
  python scripts/transfer_learning_trainer.py \\
      --source_cohort tcga \\
      --target_cohort orien \\
      --source_params results/hyperparam_FIXED_tcga_20251109_194909/best_params.json \\
      --target_params results/hyperparam_FIXED_orien_20251109_195430/best_params.json \\
      --seed 42
        """
    )
    
    # Cohort arguments
    parser.add_argument('--source_cohort', type=str, required=True,
                       choices=['orien', 'tcga'],
                       help='Source cohort for pre-training (usually orien)')
    parser.add_argument('--target_cohort', type=str, required=True,
                       choices=['orien', 'tcga'],
                       help='Target cohort for fine-tuning (usually tcga)')
    
    # Hyperparameter files from Chapter 3
    parser.add_argument('--source_params', type=str, required=True,
                       help='Path to source cohort best_params.json')
    parser.add_argument('--target_params', type=str, required=True,
                       help='Path to target cohort best_params.json')
    
    # Data directory
    parser.add_argument('--data_dir', type=str, default='data/processed',
                       help='Directory containing preprocessed data (default: data/processed)')
    
    # Training parameters
    parser.add_argument('--pretrain_epochs', type=int, default=100,
                       help='Number of epochs for pre-training (default: 100)')
    parser.add_argument('--finetune_epochs', type=int, default=30,
                       help='Number of epochs for fine-tuning (default: 30)')
    parser.add_argument('--pretrain_lr', type=float, default=1e-4,
                       help='Learning rate for pre-training (default: 1e-4)')
    parser.add_argument('--finetune_lr', type=float, default=1e-5,
                       help='Learning rate for fine-tuning (default: 1e-5)')
    
    # Random seed
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    # Output directory
    parser.add_argument('--output_dir', type=str, 
                       default='results/transfer_learning',
                       help='Output directory for results')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for training (default: cuda)')
    
    args = parser.parse_args()
    
    # Validate cohorts are different
    if args.source_cohort == args.target_cohort:
        raise ValueError("Source and target cohorts must be different!")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(
        args.output_dir,
        f"{args.source_cohort}_to_{args.target_cohort}_seed{args.seed}_{timestamp}"
    )
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"TRANSFER LEARNING TRAINER - CHAPTER 4")
    print(f"{'='*60}")
    print(f"Transfer direction: {args.source_cohort.upper()} → {args.target_cohort.upper()}")
    print(f"Pre-training: {args.pretrain_epochs} epochs @ LR={args.pretrain_lr}")
    print(f"Fine-tuning: {args.finetune_epochs} epochs @ LR={args.finetune_lr}")
    print(f"Random seed: {args.seed}")
    print(f"Output directory: {run_dir}")
    print(f"Device: {args.device}")
    print(f"{'='*60}\n")
    
    # Load hyperparameters
    print("Loading hyperparameters from Chapter 3...")
    source_params = load_hyperparameters(args.source_params)
    target_params = load_hyperparameters(args.target_params)
    print(f"✓ Source params loaded: {args.source_params}")
    print(f"✓ Target params loaded: {args.target_params}\n")
    
    # Load data
    print("Loading cohort data...")
    source_data = load_cohort_data(args.source_cohort, args.data_dir)
    target_data = load_cohort_data(args.target_cohort, args.data_dir)
    print()
    
    # Verify same number of genes
    n_source_genes = source_data['expression'].shape[0]
    n_target_genes = target_data['expression'].shape[0]
    
    if n_source_genes != n_target_genes:
        raise ValueError(
            f"Gene count mismatch! "
            f"Source has {n_source_genes} genes, target has {n_target_genes} genes. "
            f"Both cohorts must use the same 308 consensus genes."
        )
    
    print(f"✓ Both cohorts have {n_source_genes} genes (architecture compatible)\n")
    
    # ========================================
    # STEP 1: Pre-train on source cohort
    # ========================================
    
    pretrained_model, pretrain_metrics = train_source_model(
        source_data=source_data,
        source_params=source_params,
        n_epochs=args.pretrain_epochs,
        learning_rate=args.pretrain_lr,
        seed=args.seed,
        device=args.device,
        output_dir=run_dir
    )
    
    # Save pre-trained model
    pretrained_checkpoint = save_pretrained_model(
        model=pretrained_model,
        save_dir=run_dir,
        cohort_name=args.source_cohort,
        hyperparams=source_params,
        train_metrics=pretrain_metrics,
        seed=args.seed
    )
    
    # ========================================
    # STEP 2: Fine-tune on target cohort
    # ========================================
    
    finetuned_model, finetune_metrics = finetune_target_model(
        pretrained_model=pretrained_model,
        target_data=target_data,
        target_params=target_params,
        n_epochs=args.finetune_epochs,
        learning_rate=args.finetune_lr,
        seed=args.seed,
        device=args.device,
        output_dir=run_dir
    )
    
    # Save fine-tuned model
    finetuned_path = os.path.join(
        run_dir,
        f"{args.target_cohort}_finetuned_seed{args.seed}.pth"
    )
    
    checkpoint = {
        'model_state_dict': finetuned_model.state_dict(),
        'pretrained_from': args.source_cohort,
        'finetuned_on': args.target_cohort,
        'pretrain_metrics': pretrain_metrics,
        'finetune_metrics': finetune_metrics,
        'seed': args.seed,
        'timestamp': timestamp
    }
    
    torch.save(checkpoint, finetuned_path)
    
    print(f"\n{'='*60}")
    print("TRANSFER LEARNING COMPLETE")
    print(f"{'='*60}")
    print(f"Pre-trained model: {pretrained_checkpoint}")
    print(f"Fine-tuned model: {finetuned_path}")
    print(f"\nResults:")
    print(f"  Pre-training ({args.source_cohort.upper()}) C-index: "
          f"{pretrain_metrics['c_index']:.4f}")
    print(f"  Fine-tuning ({args.target_cohort.upper()}) C-index: "
          f"{finetune_metrics['c_index']:.4f}")
    print(f"\nNext steps:")
    print(f"  1. Evaluate on held-out test set")
    print(f"  2. Compare with Chapter 3 baseline (training from scratch)")
    print(f"  3. Run multi-seed validation (seeds: 42, 123, 456, 789, 1011)")
    print(f"{'='*60}\n")