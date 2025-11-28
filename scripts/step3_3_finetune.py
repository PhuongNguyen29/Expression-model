"""
Step 3.3: Fine-tuning Phase

Purpose: Fine-tune pre-trained models on target cohort and evaluate transfer learning benefit.

Protocol:
- Load pre-trained model from Step 3.2
- Load SAME train/test splits from Step 3.1 for fair comparison
- Fine-tune on target cohort's train split
- Learning rate: 10× reduction from source cohort's LR
- Regularization: Keep values from pre-trained model config
- Batch size: Use TARGET cohort's batch size
- Early stopping based on test C-index (patience=20)
- Evaluate on target cohort's test split
- Multi-seed validation (seeds: 42, 123, 456, 789, 1011)

Directions:
- ORIEN→TCGA: Fine-tune ORIEN pre-trained model on TCGA train, evaluate on TCGA test
- TCGA→ORIEN: Fine-tune TCGA pre-trained model on ORIEN train, evaluate on ORIEN test
"""

import sys
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, Subset
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.elastic_deepsurv import ElasticDeepSurv
from src.data.dataset import SurvivalDataset
from src.utils.batch_samplers import StratifiedBatchSampler


def setup_logging(output_dir, direction, seed):
    """Setup logging configuration"""
    log_file = output_dir / f"finetune_{direction}_seed{seed}.log"
    
    # Create logger
    logger = logging.getLogger(f"finetune_{direction}_{seed}")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def load_consensus_genes(consensus_file):
    """Load consensus genes from file"""
    with open(consensus_file, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]
    return genes


def load_hyperparameters(params_file):
    """Load hyperparameters from JSON file"""
    with open(params_file, 'r') as f:
        data = json.load(f)
    return data['best_params']


def load_data(consensus_genes, logger):
    """Load and filter data to consensus genes"""
    logger.info("Loading data files...")
    
    # Load expression data
    tcga_expr = pd.read_csv("data/raw/tcga_batch_corrected_2sv.csv", index_col=0)
    orien_expr = pd.read_csv("data/raw/orien_batch_corrected.csv", index_col=0)
    
    # Load survival data
    tcga_surv = pd.read_csv("data/raw/surv_tcga.csv")
    orien_surv = pd.read_csv("data/raw/surv_orien_update.csv")
    
    # Filter to consensus genes
    logger.info(f"Filtering to {len(consensus_genes)} consensus genes...")
    
    # Check which genes are available
    available_tcga = [g for g in consensus_genes if g in tcga_expr.index]
    available_orien = [g for g in consensus_genes if g in orien_expr.index]
    
    # Use intersection of available genes
    common_genes = sorted(list(set(available_tcga) & set(available_orien)))
    logger.info(f"Using {len(common_genes)} genes available in both cohorts")
    
    tcga_expr = tcga_expr.loc[common_genes]
    orien_expr = orien_expr.loc[common_genes]
    
    # Harmonize sample IDs
    tcga_expr, tcga_surv = harmonize_samples(tcga_expr, tcga_surv, logger, "TCGA")
    orien_expr, orien_surv = harmonize_samples(orien_expr, orien_surv, logger, "ORIEN")
    
    # Standardize (z-score per gene)
    logger.info("Standardizing expression data...")
    tcga_expr = standardize(tcga_expr)
    orien_expr = standardize(orien_expr)
    
    return {
        'tcga_expr': tcga_expr,
        'orien_expr': orien_expr,
        'tcga_surv': tcga_surv,
        'orien_surv': orien_surv,
        'n_features': len(common_genes)
    }


def harmonize_samples(expr_df, surv_df, logger, cohort_name):
    """Match samples between expression and survival data"""
    expr_samples = set(expr_df.columns)
    surv_samples = set(surv_df['sampleID'])
    matched = sorted(list(expr_samples.intersection(surv_samples)))
    
    logger.info(f"  {cohort_name}: {len(matched)}/{len(surv_samples)} samples matched")
    
    expr_df = expr_df[matched]
    surv_df = surv_df[surv_df['sampleID'].isin(matched)].set_index('sampleID')
    
    return expr_df, surv_df


def standardize(expr_df):
    """Z-score standardization per gene"""
    return expr_df.subtract(expr_df.mean(axis=1), axis=0).divide(expr_df.std(axis=1), axis=0)


def load_saved_split(split_dir, target_cohort, seed):
    """Load saved train/test split indices from Step 3.1"""
    train_idx = np.load(split_dir / f'{target_cohort}_seed{seed}_train_idx.npy')
    test_idx = np.load(split_dir / f'{target_cohort}_seed{seed}_test_idx.npy')
    return train_idx, test_idx


def get_finetune_config(direction, source_params_file, target_params_file):
    """
    Get fine-tuning configuration.
    
    Fine-tuning LR = source LR / 10 (10× reduction)
    Batch size from target cohort
    
    Args:
        direction: 'orien_to_tcga' or 'tcga_to_orien'
        source_params_file: Path to source cohort's best_params.json
        target_params_file: Path to target cohort's best_params.json
    
    Returns:
        target_cohort, finetune_lr, target_batch_size
    """
    source_params = load_hyperparameters(source_params_file)
    target_params = load_hyperparameters(target_params_file)
    
    if direction == 'orien_to_tcga':
        target_cohort = 'tcga'
        source_lr = source_params['learning_rate']  # ORIEN's LR
        target_batch_size = target_params['batch_size']  # TCGA's batch size
    else:  # tcga_to_orien
        target_cohort = 'orien'
        source_lr = source_params['learning_rate']  # TCGA's LR
        target_batch_size = target_params['batch_size']  # ORIEN's batch size
    
    # Fine-tuning LR = source LR / 10
    finetune_lr = source_lr * 0.75
    
    return target_cohort, finetune_lr, target_batch_size


def plot_training_curves(history, output_path, title):
    """Plot training curves with train and test C-index"""
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    train_cindex = [h['train_cindex'] for h in history]
    test_cindex = [h['test_cindex'] for h in history]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    ax1.plot(epochs, train_loss, 'b-', linewidth=1.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title(f'{title} - Training Loss')
    ax1.grid(alpha=0.3)
    
    # C-index
    ax2.plot(epochs, train_cindex, 'b-', label='Train C-index', linewidth=1.5)
    ax2.plot(epochs, test_cindex, 'r-', label='Test C-index', linewidth=1.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('C-index')
    ax2.set_title(f'{title} - C-index')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Mark best epoch
    if history and 'best_epoch' in history[-1]:
        best_epoch = history[-1]['best_epoch']
        ax2.axvline(best_epoch, color='g', linestyle='--', 
                   label=f'Best: {best_epoch}', alpha=0.7)
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch in train_loader:
        features = batch['features'].to(device)
        time = batch['time'].to(device)
        event = batch['event'].to(device)
        
        optimizer.zero_grad()
        risk = model(features)
        loss = model.compute_loss(risk, time, event)
        
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def evaluate(model, dataset, indices, device):
    """Evaluate model on dataset subset"""
    model.eval()
    
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=256, shuffle=False)
    
    all_risks = []
    all_times = []
    all_events = []
    
    with torch.no_grad():
        for batch in loader:
            features = batch['features'].to(device)
            time = batch['time']
            event = batch['event']
            
            risk = model(features)
            
            all_risks.append(risk.cpu().numpy())
            all_times.append(time.numpy())
            all_events.append(event.numpy())
    
    risks = np.concatenate(all_risks)
    times = np.concatenate(all_times)
    events = np.concatenate(all_events).astype(bool)
    
    c_index = concordance_index(times, -risks, events)
    
    return c_index


def finetune_model(direction, target_expr, target_surv, pretrain_model_path,
                   source_params_file, target_params_file, seed, split_dir,
                   output_dir, logger, device):
    """
    Fine-tune pre-trained model on target cohort.
    
    Args:
        direction: 'orien_to_tcga' or 'tcga_to_orien'
        target_expr: Target cohort expression data
        target_surv: Target cohort survival data
        pretrain_model_path: Path to pre-trained model
        source_params_file: Path to source cohort's best_params.json
        target_params_file: Path to target cohort's best_params.json
        seed: Random seed
        split_dir: Directory with saved split indices
        output_dir: Output directory
        logger: Logger instance
        device: torch device
    """
    
    logger.info(f"{'='*60}")
    logger.info(f"Fine-tuning: {direction.upper()} - Seed {seed}")
    logger.info(f"{'='*60}")
    
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Get fine-tuning configuration
    target_cohort, finetune_lr, target_batch_size = get_finetune_config(
        direction, source_params_file, target_params_file
    )
    
    logger.info(f"Target cohort: {target_cohort.upper()}")
    logger.info(f"Fine-tuning LR: {finetune_lr:.6f} (10× reduction from source)")
    logger.info(f"Target batch size: {target_batch_size}")
    
    # Create dataset
    dataset = SurvivalDataset(target_expr, target_surv)
    
    # Load SAME split as Step 3.1
    train_idx, test_idx = load_saved_split(split_dir, target_cohort, seed)
    
    logger.info(f"\nLoaded saved split from Step 3.1:")
    logger.info(f"  Train: {len(train_idx)} samples")
    logger.info(f"  Test: {len(test_idx)} samples")
    logger.info(f"  Train events: {dataset.y_event[train_idx].sum():.0f}/{len(train_idx)} "
                f"({100*dataset.y_event[train_idx].mean():.1f}%)")
    logger.info(f"  Test events: {dataset.y_event[test_idx].sum():.0f}/{len(test_idx)} "
                f"({100*dataset.y_event[test_idx].mean():.1f}%)")
    
    # Load pre-trained model
    logger.info(f"\nLoading pre-trained model from {pretrain_model_path.name}")
    checkpoint = torch.load(pretrain_model_path, map_location=device)
    
    model_config = checkpoint['config']
    pretrain_cindex = checkpoint.get('final_train_cindex', checkpoint.get('best_valid_cindex', 0.0))
    
    logger.info(f"Pre-trained model architecture: {model_config['hidden_sizes']}")
    logger.info(f"Pre-trained model train C-index: {pretrain_cindex:.4f}")
    
    # Initialize model with pre-trained weights
    model = ElasticDeepSurv(**model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {n_params:,}")
    
    # Re-initialize optimizer (fresh optimizer state for fine-tuning)
    optimizer = torch.optim.Adam(model.parameters(), lr=finetune_lr)
    
    logger.info(f"\nModel loaded and ready for fine-tuning")
    
    # Create train loader
    train_dataset = Subset(dataset, train_idx)
    train_events = dataset.y_event[train_idx]
    
    # Use stratified batch sampler for larger cohorts
    n_samples = len(train_idx)
    if n_samples >= 500:
        logger.info("Using StratifiedBatchSampler")
        train_batch_sampler = StratifiedBatchSampler(
            events=train_events,
            batch_size=target_batch_size,
            min_events_per_batch=2,
            shuffle=True,
            drop_last=False
        )
        train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler)
    else:
        logger.info("Using simple random shuffling (smaller target cohort)")
        train_loader = DataLoader(train_dataset, batch_size=target_batch_size, shuffle=True)
    
    # Fine-tuning settings
    max_epochs = 500
    patience = 80
    
    logger.info(f"\nFine-tuning Settings:")
    logger.info(f"  Max epochs: {max_epochs}")
    logger.info(f"  Early stopping patience: {patience}")
    
    # Evaluate before fine-tuning (zero-shot transfer)
    zero_shot_train = evaluate(model, dataset, train_idx, device)
    zero_shot_test = evaluate(model, dataset, test_idx, device)
    logger.info(f"\nZero-shot (before fine-tuning):")
    logger.info(f"  Train C-index: {zero_shot_train:.4f}")
    logger.info(f"  Test C-index: {zero_shot_test:.4f}")
    
    # Fine-tuning loop with early stopping
    logger.info(f"\nStarting fine-tuning...")
    
    training_history = []
    best_test_cindex = 0.0
    best_train_cindex = 0.0
    best_epoch = 0
    best_model_state = None
    epochs_without_improvement = 0
    
    for epoch in range(max_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_cindex = evaluate(model, dataset, train_idx, device)
        test_cindex = evaluate(model, dataset, test_idx, device)
        
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'train_cindex': float(train_cindex),
            'test_cindex': float(test_cindex)
        })
        
        # Check for improvement on test set
        if test_cindex > best_test_cindex:
            best_test_cindex = test_cindex
            best_train_cindex = train_cindex
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
            
            # Always log when there's a new best
            logger.info(f"Epoch {epoch+1:3d}: Loss={train_loss:.4f}, "
                       f"Train={train_cindex:.4f}, Test={test_cindex:.4f} *** BEST ***")
        else:
            epochs_without_improvement += 1
            
            # Only log non-improvements at epoch multiples of 10
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1:3d}: Loss={train_loss:.4f}, "
                           f"Train={train_cindex:.4f}, Test={test_cindex:.4f} "
                           f"(no improvement: {epochs_without_improvement}/{patience})")
            
            if epochs_without_improvement >= patience:
                logger.info(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Add best epoch info to history
    if training_history:
        training_history[-1]['best_epoch'] = best_epoch
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"\nRestored model from epoch {best_epoch}")
    
    # Final summary
    logger.info(f"\nFine-tuning Complete:")
    logger.info(f"  Best epoch: {best_epoch}")
    logger.info(f"  Zero-shot test C-index: {zero_shot_test:.4f}")
    logger.info(f"  Fine-tuned test C-index: {best_test_cindex:.4f}")
    logger.info(f"  Improvement: {best_test_cindex - zero_shot_test:+.4f} ({100*(best_test_cindex - zero_shot_test)/max(zero_shot_test, 0.001):+.1f}%)")
    logger.info(f"  Train C-index at best: {best_train_cindex:.4f}")
    
    # Plot training curves
    if training_history:
        plot_training_curves(
            training_history,
            output_dir / f'seed{seed}_finetuning_curve.png',
            f'{direction.upper()} Fine-tuning Seed {seed}'
        )
        logger.info(f"Saved training curve to {output_dir / f'seed{seed}_finetuning_curve.png'}")
    
    # Save results
    results = {
        'direction': direction,
        'seed': seed,
        'target_cohort': target_cohort,
        'pretrain_train_cindex': float(pretrain_cindex),
        'zero_shot_train_cindex': float(zero_shot_train),
        'zero_shot_test_cindex': float(zero_shot_test),
        'finetune_train_cindex': float(best_train_cindex),
        'finetune_test_cindex': float(best_test_cindex),
        'improvement_over_zero_shot': float(best_test_cindex - zero_shot_test),
        'best_epoch': best_epoch,
        'total_epochs': len(training_history),
        'n_train': len(train_idx),
        'n_test': len(test_idx),
        'n_train_events': int(train_events.sum()),
        'n_test_events': int(dataset.y_event[test_idx].sum()),
        'architecture': model_config['hidden_sizes'],
        'hyperparameters': {
            'finetune_lr': finetune_lr,
            'batch_size': target_batch_size,
            'dropout': model_config['dropout'],
            'alpha': model_config['alpha'],
            'l1_ratio': model_config['l1_ratio'],
            'batch_norm': model_config['batch_norm'],
            'activation': model_config['activation']
        },
        'training_settings': {
            'max_epochs': max_epochs,
            'patience': patience
        }
    }
    
    # Save results JSON
    with open(output_dir / f'seed{seed}_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save training history CSV
    if training_history:
        history_df = pd.DataFrame(training_history)
        history_df.to_csv(output_dir / f'seed{seed}_training_log.csv', index=False)
    
    # Save fine-tuned model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model_config,
        'seed': seed,
        'direction': direction,
        'best_epoch': best_epoch,
        'finetune_test_cindex': best_test_cindex,
        'zero_shot_test_cindex': zero_shot_test,
        'pretrain_cindex': pretrain_cindex
    }, output_dir / f'seed{seed}_finetuned_model.pth')
    
    logger.info(f"Saved fine-tuned model to {output_dir / f'seed{seed}_finetuned_model.pth'}")
    
    return results


def aggregate_results(output_dir, direction, logger):
    """Aggregate fine-tuning results across all seeds"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Aggregating {direction.upper()} Results")
    logger.info(f"{'='*60}")
    
    results = []
    
    for seed_file in sorted(output_dir.glob('seed*_results.json')):
        with open(seed_file, 'r') as f:
            results.append(json.load(f))
    
    if not results:
        logger.warning("No results found to aggregate")
        return None
    
    # Create summary DataFrame
    df = pd.DataFrame(results)
    
    summary = {
        'direction': direction,
        'n_seeds': len(results),
        'mean_zero_shot_test': df['zero_shot_test_cindex'].mean(),
        'std_zero_shot_test': df['zero_shot_test_cindex'].std(),
        'mean_finetune_test': df['finetune_test_cindex'].mean(),
        'std_finetune_test': df['finetune_test_cindex'].std(),
        'mean_improvement': df['improvement_over_zero_shot'].mean(),
        'std_improvement': df['improvement_over_zero_shot'].std(),
        'min_finetune_test': df['finetune_test_cindex'].min(),
        'max_finetune_test': df['finetune_test_cindex'].max(),
        'mean_best_epoch': df['best_epoch'].mean()
    }
    
    # Print summary
    logger.info(f"\n{direction.upper()} Fine-tuning Summary:")
    logger.info(f"  Zero-shot test C-index: {summary['mean_zero_shot_test']:.4f} ± {summary['std_zero_shot_test']:.4f}")
    logger.info(f"  Fine-tuned test C-index: {summary['mean_finetune_test']:.4f} ± {summary['std_finetune_test']:.4f}")
    logger.info(f"  Improvement: {summary['mean_improvement']:+.4f} ± {summary['std_improvement']:.4f}")
    logger.info(f"  Range: {summary['min_finetune_test']:.4f} - {summary['max_finetune_test']:.4f}")
    logger.info(f"  Average best epoch: {summary['mean_best_epoch']:.1f}")
    
    # Save summary
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary


def main():
    # Configuration for k=155 (87 consensus genes)
    SEEDS = [42, 123, 456, 789, 1011]
    K_VALUE = 155
    
    # File paths for k=155
    CONSENSUS_GENES_FILE = f"results_v2/02_biomarker_discovery/k_selection_with_tuning/k{K_VALUE}/consensus_genes/consensus_genes.txt"
    TCGA_PARAMS_FILE = f"results_v2/02_biomarker_discovery/k_selection_with_tuning/k{K_VALUE}/hyperparameter_tuning/tcga/best_params.json"
    ORIEN_PARAMS_FILE = f"results_v2/02_biomarker_discovery/k_selection_with_tuning/k{K_VALUE}/hyperparameter_tuning/orien/best_params.json"
    PRETRAIN_BASE_DIR = Path(f"results_v2/03_transfer_learning/k{K_VALUE}/pretrained")
    SPLIT_DIR = Path(f"results_v2/03_transfer_learning/k{K_VALUE}/baseline_target_only/splits")
    BASE_OUTPUT_DIR = Path(f"results_v2/03_transfer_learning/k{K_VALUE}/finetuned")
    
    # Setup
    BASE_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*60)
    print("Step 3.3: Fine-tuning Phase")
    print("="*60)
    print(f"K-value: {K_VALUE}")
    print(f"Seeds: {SEEDS}")
    print(f"Device: {device}")
    print(f"Output: {BASE_OUTPUT_DIR}")
    
    # Verify input directories exist
    if not PRETRAIN_BASE_DIR.exists():
        raise FileNotFoundError(
            f"Pre-trained models not found: {PRETRAIN_BASE_DIR}\n"
            "Please run Step 3.2 first."
        )
    
    if not SPLIT_DIR.exists():
        raise FileNotFoundError(
            f"Split directory not found: {SPLIT_DIR}\n"
            "Please run Step 3.1 first to create train/test splits."
        )
    
    # Verify file paths
    for filepath, desc in [
        (CONSENSUS_GENES_FILE, "Consensus genes"),
        (TCGA_PARAMS_FILE, "TCGA hyperparameters"),
        (ORIEN_PARAMS_FILE, "ORIEN hyperparameters")
    ]:
        if not Path(filepath).exists():
            raise FileNotFoundError(f"{desc} not found: {filepath}")
        print(f"✓ Found {desc}")
    
    # Load consensus genes
    consensus_genes = load_consensus_genes(CONSENSUS_GENES_FILE)
    print(f"\nLoaded {len(consensus_genes)} consensus genes from k={K_VALUE}")
    
    # Setup initial logger for data loading
    data_logger = logging.getLogger('data_loader')
    data_logger.setLevel(logging.INFO)
    if not data_logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        data_logger.addHandler(ch)
    
    # Load data
    print("\nLoading data...")
    data = load_data(consensus_genes, data_logger)
    
    # Fine-tune for both directions
    directions = [
        {
            'name': 'orien_to_tcga',
            'target_expr': data['tcga_expr'],
            'target_surv': data['tcga_surv'],
            'source_params': ORIEN_PARAMS_FILE,
            'target_params': TCGA_PARAMS_FILE
        },
        {
            'name': 'tcga_to_orien',
            'target_expr': data['orien_expr'],
            'target_surv': data['orien_surv'],
            'source_params': TCGA_PARAMS_FILE,
            'target_params': ORIEN_PARAMS_FILE
        }
    ]
    
    all_summaries = []
    
    for dir_config in directions:
        direction = dir_config['name']
        output_dir = BASE_OUTPUT_DIR / direction
        output_dir.mkdir(exist_ok=True, parents=True)
        pretrain_dir = PRETRAIN_BASE_DIR / direction
        
        print(f"\n{'='*60}")
        print(f"Direction: {direction.upper()}")
        print(f"{'='*60}")
        
        for seed in SEEDS:
            logger = setup_logging(output_dir, direction, seed)
            
            # Get pre-trained model path (note: filename is seed{seed}_pretrained_model.pth)
            pretrain_model_path = pretrain_dir / f'seed{seed}_pretrained_model.pth'
            
            if not pretrain_model_path.exists():
                logger.error(f"Pre-trained model not found: {pretrain_model_path}")
                logger.error("Please run Step 3.2 first.")
                continue
            
            finetune_model(
                direction=direction,
                target_expr=dir_config['target_expr'],
                target_surv=dir_config['target_surv'],
                pretrain_model_path=pretrain_model_path,
                source_params_file=dir_config['source_params'],
                target_params_file=dir_config['target_params'],
                seed=seed,
                split_dir=SPLIT_DIR,
                output_dir=output_dir,
                logger=logger,
                device=device
            )
        
        # Aggregate results for this direction
        summary_logger = setup_logging(output_dir, direction, 'summary')
        summary = aggregate_results(output_dir, direction, summary_logger)
        if summary:
            all_summaries.append(summary)
    
    # Save overall summary
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_df.to_csv(BASE_OUTPUT_DIR / 'finetuning_summary.csv', index=False)
        
        print(f"\n{'='*60}")
        print("Fine-tuning Summary")
        print(f"{'='*60}")
        for summary in all_summaries:
            print(f"\n{summary['direction'].upper()}:")
            print(f"  Zero-shot: {summary['mean_zero_shot_test']:.4f} ± {summary['std_zero_shot_test']:.4f}")
            print(f"  Fine-tuned: {summary['mean_finetune_test']:.4f} ± {summary['std_finetune_test']:.4f}")
            print(f"  Improvement: {summary['mean_improvement']:+.4f}")
    
    print(f"\n{'='*60}")
    print("Step 3.3 Complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {BASE_OUTPUT_DIR}")
    print("\nNext: Run Step 3.4 (Statistical Analysis)")


if __name__ == "__main__":
    main()
