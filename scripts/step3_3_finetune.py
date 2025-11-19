"""
Step 3.3: Fine-tuning Phase

Purpose: Fine-tune pre-trained models on target cohort and evaluate transfer learning benefit.

Protocol:
- Load pre-trained model from Step 3.2
- Load SAME train/test splits from Step 3.1 for fair comparison
- Fine-tune on target cohort's train split
- Learning rate: 10× reduction from source cohort's LR
- Regularization: Keep SOURCE cohort's values (dropout, alpha, l1_ratio)
- Batch size: Use TARGET cohort's batch size
- Fixed 40 epochs (no early stopping)
- Evaluate on target cohort's test split
- Multi-seed validation (seeds: 42, 123, 456, 789, 1011)

Directions:
- ORIEN→TCGA: Fine-tune ORIEN pre-trained model on TCGA
- TCGA→ORIEN: Fine-tune TCGA pre-trained model on ORIEN
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

def plot_training_curves(history, output_path, title, show_validation=False):
    """Plot training curves"""
    
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    
    if show_validation:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        ax1.plot(epochs, train_loss, 'b-', label='Train Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'{title} - Loss')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # C-index
        train_cindex = [h['train_cindex'] for h in history]
        valid_cindex = [h['valid_cindex'] for h in history]
        ax2.plot(epochs, train_cindex, 'b-', label='Train C-index')
        ax2.plot(epochs, valid_cindex, 'r-', label='Valid C-index')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('C-index')
        ax2.set_title(f'{title} - C-index')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # Mark best epoch if available
        if 'best_epoch' in history[-1]:
            best_epoch = history[-1]['best_epoch']
            ax2.axvline(best_epoch, color='g', linestyle='--', 
                       label=f'Best Epoch: {best_epoch}')
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))
        ax1.plot(epochs, train_loss, 'b-')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'{title} - Training Loss')
        ax1.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

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
    tcga_expr = tcga_expr.loc[tcga_expr.index.isin(consensus_genes)]
    orien_expr = orien_expr.loc[orien_expr.index.isin(consensus_genes)]
    
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
        'orien_surv': orien_surv
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
    
    Args:
        direction: 'orien_to_tcga' or 'tcga_to_orien'
        source_params_file: Path to source cohort's best_params.json
        target_params_file: Path to target cohort's best_params.json
    
    Returns:
        target_cohort, finetune_lr, target_batch_size
    """
    with open(source_params_file, 'r') as f:
        source_params = json.load(f)
    
    with open(target_params_file, 'r') as f:
        target_params = json.load(f)
    
    if direction == 'orien_to_tcga':
        target_cohort = 'tcga'
        source_lr = source_params['learning_rate']  # ORIEN's LR
        target_batch_size = target_params['batch_size']  # TCGA's batch size
    else:  # tcga_to_orien
        target_cohort = 'orien'
        source_lr = source_params['learning_rate']  # TCGA's LR
        target_batch_size = target_params['batch_size']  # ORIEN's batch size
    
    # Fine-tuning LR = 0.2 × source LR
    finetune_lr = source_lr / 3
    
    return target_cohort, finetune_lr, target_batch_size


def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    for batch in train_loader:
        features = batch['features'].to(device)
        time = batch['time'].to(device)
        event = batch['event'].to(device)
        
        optimizer.zero_grad()
        risk = model(features)
        loss = model.compute_loss(risk, time, event)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


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
    
    # Get fine-tuning configuration
    target_cohort, finetune_lr, target_batch_size = get_finetune_config(
        direction, source_params_file, target_params_file
    )
    
    logger.info(f"Target cohort: {target_cohort.upper()}")
    logger.info(f"Fine-tuning LR: {finetune_lr:.6f} (10× reduction)")
    logger.info(f"Target batch size: {target_batch_size}")
    
    # Create dataset
    dataset = SurvivalDataset(target_expr, target_surv)
    
    # Load SAME split as Step 3.1
    train_idx, test_idx = load_saved_split(split_dir, target_cohort, seed)
    
    logger.info(f"Loaded saved split from Step 3.1:")
    logger.info(f"  Train: {len(train_idx)} samples")
    logger.info(f"  Test: {len(test_idx)} samples")
    logger.info(f"  Train events: {dataset.y_event[train_idx].sum()}/{len(train_idx)} "
                f"({100*dataset.y_event[train_idx].mean():.1f}%)")
    logger.info(f"  Test events: {dataset.y_event[test_idx].sum()}/{len(test_idx)} "
                f"({100*dataset.y_event[test_idx].mean():.1f}%)")
    
    # Load pre-trained model
    logger.info(f"\nLoading pre-trained model from {pretrain_model_path.name}")
    checkpoint = torch.load(pretrain_model_path, map_location=device)
    
    model_config = checkpoint['config']
    logger.info(f"Pre-trained model architecture: {model_config['hidden_sizes']}")
    logger.info(f"Pre-trained model C-index: {checkpoint['best_valid_cindex']:.4f}")
    
    # Initialize model with pre-trained weights
    model = ElasticDeepSurv(**model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Re-initialize optimizer (fresh optimizer state)
    optimizer = torch.optim.Adam(model.parameters(), lr=finetune_lr)
    
    logger.info(f"\nModel successfully loaded and ready for fine-tuning")
    logger.info(f"All layers trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")
    
    # Create train loader
    train_dataset = Subset(dataset, train_idx)
    train_events = dataset.y_event[train_idx]
    
    # Use stratified batch sampler for large cohorts, simple shuffle for small
    n_samples = len(train_idx)
    if n_samples >= 500:
        logger.info("Using StratifiedBatchSampler (large cohort)")
        train_batch_sampler = StratifiedBatchSampler(
            events=train_events,
            batch_size=target_batch_size,
            min_events_per_batch=2,
            shuffle=True,
            drop_last=False
        )
        train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler)
    else:
        logger.info("Using simple random shuffling (small cohort)")
        train_loader = DataLoader(train_dataset, batch_size=target_batch_size, shuffle=True)
    
    # Fine-tuning loop (40 epochs fixed)
    logger.info(f"\nStarting fine-tuning (40 epochs)...")
    
    training_history = []
    best_train_cindex = 0.0
    
    for epoch in range(200)):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            train_cindex = evaluate(model, dataset, train_idx, device)
            logger.info(f"Epoch {epoch+1:2d}: Loss={train_loss:.4f}, Train C-index={train_cindex:.4f}")
            best_train_cindex = max(best_train_cindex, train_cindex)
            
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': float(train_loss),
                'train_cindex': float(train_cindex)
            })
    
    # Final evaluation on test set
    test_cindex = evaluate(model, dataset, test_idx, device)
    
    logger.info(f"\nFine-tuning Complete:")
    logger.info(f"  Best train C-index: {best_train_cindex:.4f}")
    logger.info(f"  Test C-index: {test_cindex:.4f}")
    
    # Calculate improvement over zero-shot (if available)
    # Note: Zero-shot results should come from Step 2.2B
    # For now, we'll just report the test C-index
    
    if training_history:  # Only plot if we have history
        plot_training_curves(
            training_history,
            output_dir / f'seed{seed}_finetuning_curve.png',
            f'{direction.upper()} Fine-tuning Seed {seed}',
            show_validation=False
        )
        logger.info(f"Saved training curve to {output_dir / f'seed{seed}_finetuning_curve.png'}")
    
    # Save results
    results = {
        'direction': direction,
        'seed': seed,
        'target_cohort': target_cohort,
        'pretrain_valid_cindex': float(checkpoint['best_valid_cindex']),
        'finetune_train_cindex': float(best_train_cindex),
        'finetune_test_cindex': float(test_cindex),
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
            'batch_norm': model_config['batch_norm']
        },
        'training_history': training_history
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
        'test_cindex': test_cindex,
        'pretrain_cindex': checkpoint['best_valid_cindex']
    }, output_dir / f'seed{seed}_finetune_model.pth')
    
    logger.info(f"Saved fine-tuned model to {output_dir / f'seed{seed}_finetune_model.pth'}")
    
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
    
    # Create summary DataFrame
    df = pd.DataFrame(results)
    
    summary = {
        'direction': direction,
        'n_seeds': len(results),
        'mean_test_cindex': df['finetune_test_cindex'].mean(),
        'std_test_cindex': df['finetune_test_cindex'].std(),
        'min_test_cindex': df['finetune_test_cindex'].min(),
        'max_test_cindex': df['finetune_test_cindex'].max(),
        'mean_pretrain_cindex': df['pretrain_valid_cindex'].mean()
    }
    
    # Print summary
    logger.info(f"\n{direction.upper()} Fine-tuning Summary:")
    logger.info(f"  Test C-index: {summary['mean_test_cindex']:.4f} ± {summary['std_test_cindex']:.4f}")
    logger.info(f"  Range: {summary['min_test_cindex']:.4f} - {summary['max_test_cindex']:.4f}")
    logger.info(f"  Mean pre-train C-index: {summary['mean_pretrain_cindex']:.4f}")
    
    return summary


def main():
    # Configuration
    SEEDS = [42, 123, 456, 789, 1011]
    CONSENSUS_GENES_FILE = "results_v2/02_biomarker_discovery/ksweep_analysis/gene_lists/k120_consensus.txt"
    TCGA_PARAMS_FILE = "results_v2/01_hyperparameter_tuning/tcga_308genes/best_params.json"
    ORIEN_PARAMS_FILE = "results_v2/01_hyperparameter_tuning/orien_308genes/best_params.json"
    PRETRAIN_BASE_DIR = Path("results_v2/03_transfer_learning/pretraining")
    SPLIT_DIR = Path("results_v2/03_transfer_learning/baseline2_target_only/splits")
    BASE_OUTPUT_DIR = Path("results_v2/03_transfer_learning/finetuning")
    
    # Setup
    BASE_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*60)
    print("Step 3.3: Fine-tuning Phase")
    print("="*60)
    print(f"Seeds: {SEEDS}")
    print(f"Device: {device}")
    print(f"Output: {BASE_OUTPUT_DIR}")
    
    # Verify split directory exists
    if not SPLIT_DIR.exists():
        raise FileNotFoundError(
            f"Split directory not found: {SPLIT_DIR}\n"
            "Please run Step 3.1 first to create train/test splits."
        )
    
    # Load consensus genes
    consensus_genes = load_consensus_genes(CONSENSUS_GENES_FILE)
    print(f"\nLoaded {len(consensus_genes)} consensus genes from k=120")
    
    # Load data
    print("\nLoading data...")
    data = load_data(consensus_genes, logging.getLogger('data_loader'))
    
    # Fine-tune for both directions
    directions = [
        ('orien_to_tcga', data['tcga_expr'], data['tcga_surv'], 
         ORIEN_PARAMS_FILE, TCGA_PARAMS_FILE),
        ('tcga_to_orien', data['orien_expr'], data['orien_surv'],
         TCGA_PARAMS_FILE, ORIEN_PARAMS_FILE)
    ]
    
    all_summaries = []
    
    for direction, target_expr, target_surv, source_params, target_params in directions:
        output_dir = BASE_OUTPUT_DIR / direction
        output_dir.mkdir(exist_ok=True, parents=True)
        pretrain_dir = PRETRAIN_BASE_DIR / direction
        
        print(f"\n{'='*60}")
        print(f"Direction: {direction.upper()}")
        print(f"{'='*60}")
        
        for seed in SEEDS:
            logger = setup_logging(output_dir, direction, seed)
            
            # Get pre-trained model path
            pretrain_model_path = pretrain_dir / f'seed{seed}_pretrain_model.pth'
            
            if not pretrain_model_path.exists():
                logger.error(f"Pre-trained model not found: {pretrain_model_path}")
                logger.error("Please run Step 3.2 first.")
                continue
            
            finetune_model(
                direction=direction,
                target_expr=target_expr,
                target_surv=target_surv,
                pretrain_model_path=pretrain_model_path,
                source_params_file=source_params,
                target_params_file=target_params,
                seed=seed,
                split_dir=SPLIT_DIR,
                output_dir=output_dir,
                logger=logger,
                device=device
            )
        
        # Aggregate results for this direction
        summary_logger = logging.getLogger(f'{direction}_summary')
        summary = aggregate_results(output_dir, direction, summary_logger)
        all_summaries.append(summary)
    
    # Save overall summary
    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(BASE_OUTPUT_DIR / 'finetuning_summary.csv', index=False)
    
    print(f"\n{'='*60}")
    print("Step 3.3 Complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {BASE_OUTPUT_DIR}")
    print("\nNext: Run Step 3.4 (Statistical Analysis)")


if __name__ == "__main__":
    main()
