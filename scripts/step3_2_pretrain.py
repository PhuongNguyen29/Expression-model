"""
Step 3.2: Pre-training Phase

Purpose: Pre-train models on source cohort to learn generalizable features.

Protocol:
- Train on SOURCE cohort using TARGET cohort's architecture
- Use 80/20 train/validation split for early stopping
- Early stopping patience: 20 epochs
- Maximum epochs: 100
- Save best pre-trained model based on validation C-index
- Multi-seed validation (seeds: 42, 123, 456, 789, 1011)

Directions:
- ORIEN→TCGA: Pre-train on ORIEN using TCGA architecture [48, 24]
- TCGA→ORIEN: Pre-train on TCGA using ORIEN architecture [96, 48]

Hyperparameters:
- Use SOURCE cohort's Step 1 hyperparameters
- Use TARGET cohort's architecture
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
from sklearn.model_selection import train_test_split
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
    log_file = output_dir / f"pretrain_{direction}_seed{seed}.log"
    
    # Create logger
    logger = logging.getLogger(f"pretrain_{direction}_{seed}")
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


def create_stratified_split(dataset, test_size=0.2, random_state=42):
    """Create stratified train/validation split"""
    indices = np.arange(len(dataset))
    events = dataset.y_event
    
    train_idx, valid_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=events,
        random_state=random_state
    )
    
    return train_idx, valid_idx


def get_pretrain_config(direction, source_params_file, n_features=51):
    """
    Get pre-training configuration.
    
    Args:
        direction: 'orien_to_tcga' or 'tcga_to_orien'
        source_params_file: Path to source cohort's best_params.json
        n_features: Number of input features (51 consensus genes)
    
    Returns:
        model_config, batch_size, learning_rate
    """
    with open(source_params_file, 'r') as f:
        source_params = json.load(f)
    
    # Use TARGET cohort's architecture but SOURCE cohort's hyperparameters
    if direction == 'orien_to_tcga':
        # Pre-train on ORIEN, use TCGA architecture [48, 24]
        architecture = [48, 24]
    else:  # tcga_to_orien
        # Pre-train on TCGA, use ORIEN architecture [96, 48]
        architecture = [96, 48]
    
    config = {
        'n_features': n_features,
        'hidden_sizes': architecture,
        'dropout': source_params['dropout'],
        'activation': source_params['activation'],
        'batch_norm': source_params['batch_norm'],
        'alpha': source_params['alpha'],
        'l1_ratio': source_params['l1_ratio'],
        'weight_init': 'xavier_uniform'
    }
    
    batch_size = source_params['batch_size']
    learning_rate = source_params['learning_rate']
    
    return config, batch_size, learning_rate


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


def pretrain_model(direction, source_expr, source_surv, source_params_file, 
                   seed, output_dir, logger, device):
    """
    Pre-train model on source cohort.
    
    Args:
        direction: 'orien_to_tcga' or 'tcga_to_orien'
        source_expr: Source cohort expression data
        source_surv: Source cohort survival data
        source_params_file: Path to source cohort's best_params.json
        seed: Random seed
        output_dir: Output directory
        logger: Logger instance
        device: torch device
    """
    
    logger.info(f"{'='*60}")
    logger.info(f"Pre-training: {direction.upper()} - Seed {seed}")
    logger.info(f"{'='*60}")
    
    # Create dataset
    dataset = SurvivalDataset(source_expr, source_surv)
    
    # Create train/validation split
    train_idx, valid_idx = create_stratified_split(
        dataset,
        test_size=0.2,
        random_state=seed
    )
    
    logger.info(f"Split: Train={len(train_idx)}, Validation={len(valid_idx)}")
    logger.info(f"  Train events: {dataset.y_event[train_idx].sum()}/{len(train_idx)} "
                f"({100*dataset.y_event[train_idx].mean():.1f}%)")
    logger.info(f"  Valid events: {dataset.y_event[valid_idx].sum()}/{len(valid_idx)} "
                f"({100*dataset.y_event[valid_idx].mean():.1f}%)")
    
    # Get model configuration
    config, batch_size, learning_rate = get_pretrain_config(
        direction,
        source_params_file,
        n_features=len(dataset.gene_names)
    )
    
    logger.info(f"Model architecture: {config['hidden_sizes']}")
    logger.info(f"Hyperparameters from SOURCE cohort:")
    logger.info(f"  Learning rate: {learning_rate:.6f}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Dropout: {config['dropout']}")
    logger.info(f"  Alpha: {config['alpha']:.6f}")
    logger.info(f"  L1 ratio: {config['l1_ratio']}")
    logger.info(f"  Batch norm: {config['batch_norm']}")
    
    # Initialize model
    model = ElasticDeepSurv(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create train loader
    train_dataset = Subset(dataset, train_idx)
    train_events = dataset.y_event[train_idx]
    
    # Use stratified batch sampler for large cohorts, simple shuffle for small
    n_samples = len(train_idx)
    if n_samples >= 500:
        logger.info("Using StratifiedBatchSampler (large cohort)")
        train_batch_sampler = StratifiedBatchSampler(
            events=train_events,
            batch_size=batch_size,
            min_events_per_batch=2,
            shuffle=True,
            drop_last=False
        )
        train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler)
    else:
        logger.info("Using simple random shuffling (small cohort)")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop with early stopping
    best_valid_cindex = 0.0
    best_epoch = 0
    patience = 20
    patience_counter = 0
    max_epochs = 100
    
    training_history = []
    
    logger.info(f"\nStarting pre-training (max {max_epochs} epochs, patience {patience})...")
    
    for epoch in range(max_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_cindex = evaluate(model, dataset, train_idx, device)
        valid_cindex = evaluate(model, dataset, valid_idx, device)
        
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'train_cindex': float(train_cindex),
            'valid_cindex': float(valid_cindex)
        })
        
        # Check for improvement
        if valid_cindex > best_valid_cindex:
            best_valid_cindex = valid_cindex
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save best model
            best_model_state = model.state_dict().copy()
            
            logger.info(f"Epoch {epoch+1:3d}: Loss={train_loss:.4f}, "
                       f"Train C-index={train_cindex:.4f}, "
                       f"Valid C-index={valid_cindex:.4f} *** NEW BEST ***")
        else:
            patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1:3d}: Loss={train_loss:.4f}, "
                           f"Train C-index={train_cindex:.4f}, "
                           f"Valid C-index={valid_cindex:.4f} "
                           f"(patience: {patience_counter}/{patience})")
            
            if patience_counter >= patience:
                logger.info(f"\nEarly stopping triggered at epoch {epoch+1}")
                logger.info(f"Best validation C-index: {best_valid_cindex:.4f} at epoch {best_epoch}")
                break
    
    # Restore best model
    model.load_state_dict(best_model_state)
    
    # Final evaluation
    final_train_cindex = evaluate(model, dataset, train_idx, device)
    final_valid_cindex = evaluate(model, dataset, valid_idx, device)
    
    logger.info(f"\nPre-training Complete:")
    logger.info(f"  Best epoch: {best_epoch}")
    logger.info(f"  Final train C-index: {final_train_cindex:.4f}")
    logger.info(f"  Final valid C-index: {final_valid_cindex:.4f}")
    
    
    if training_history:  # Only plot if we have history
        plot_training_curves(
            training_history,
            output_dir / f'seed{seed}_pretraining_curve.png',
            f'{direction.upper()} Seed {seed}',
            show_validation=True
        )
        logger.info(f"Saved training curve to {output_dir / f'seed{seed}_pretraining_curve.png'}")
        
    # Save results
    results = {
        'direction': direction,
        'seed': seed,
        'best_epoch': best_epoch,
        'best_valid_cindex': float(best_valid_cindex),
        'final_train_cindex': float(final_train_cindex),
        'final_valid_cindex': float(final_valid_cindex),
        'n_train': len(train_idx),
        'n_valid': len(valid_idx),
        'n_train_events': int(train_events.sum()),
        'n_valid_events': int(dataset.y_event[valid_idx].sum()),
        'architecture': config['hidden_sizes'],
        'hyperparameters': {
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'dropout': config['dropout'],
            'alpha': config['alpha'],
            'l1_ratio': config['l1_ratio'],
            'batch_norm': config['batch_norm']
        },
        'training_history': training_history
    }
    
    # Save results JSON
    with open(output_dir / f'seed{seed}_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save training history CSV
    history_df = pd.DataFrame(training_history)
    history_df.to_csv(output_dir / f'seed{seed}_training_log.csv', index=False)
    
    # Save pre-trained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'seed': seed,
        'direction': direction,
        'best_epoch': best_epoch,
        'best_valid_cindex': best_valid_cindex
    }, output_dir / f'seed{seed}_pretrain_model.pth')
    
    logger.info(f"Saved pre-trained model to {output_dir / f'seed{seed}_pretrain_model.pth'}")
    
    return results


def aggregate_results(output_dir, direction, logger):
    """Aggregate pre-training results across all seeds"""
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
        'mean_valid_cindex': df['best_valid_cindex'].mean(),
        'std_valid_cindex': df['best_valid_cindex'].std(),
        'min_valid_cindex': df['best_valid_cindex'].min(),
        'max_valid_cindex': df['best_valid_cindex'].max(),
        'mean_best_epoch': df['best_epoch'].mean()
    }
    
    # Print summary
    logger.info(f"\n{direction.upper()} Pre-training Summary:")
    logger.info(f"  Validation C-index: {summary['mean_valid_cindex']:.4f} ± {summary['std_valid_cindex']:.4f}")
    logger.info(f"  Range: {summary['min_valid_cindex']:.4f} - {summary['max_valid_cindex']:.4f}")
    logger.info(f"  Average best epoch: {summary['mean_best_epoch']:.1f}")
    
    return summary


def main():
    # Configuration
    SEEDS = [42, 123, 456, 789, 1011]
    CONSENSUS_GENES_FILE = "results_v2/02_biomarker_discovery/ksweep_analysis/gene_lists/k120_consensus.txt"
    TCGA_PARAMS_FILE = "results_v2/01_hyperparameter_tuning/tcga_308genes/best_params.json"
    ORIEN_PARAMS_FILE = "results_v2/01_hyperparameter_tuning/orien_308genes/best_params.json"
    BASE_OUTPUT_DIR = Path("results_v2/03_transfer_learning/pretraining")
    
    # Setup
    BASE_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*60)
    print("Step 3.2: Pre-training Phase")
    print("="*60)
    print(f"Seeds: {SEEDS}")
    print(f"Device: {device}")
    print(f"Output: {BASE_OUTPUT_DIR}")
    
    # Load consensus genes
    consensus_genes = load_consensus_genes(CONSENSUS_GENES_FILE)
    print(f"\nLoaded {len(consensus_genes)} consensus genes from k=120")
    
    # Load data
    print("\nLoading data...")
    data = load_data(consensus_genes, logging.getLogger('data_loader'))
    
    # Pre-train for both directions
    directions = [
        ('orien_to_tcga', data['orien_expr'], data['orien_surv'], ORIEN_PARAMS_FILE),
        ('tcga_to_orien', data['tcga_expr'], data['tcga_surv'], TCGA_PARAMS_FILE)
    ]
    
    all_summaries = []
    
    for direction, source_expr, source_surv, source_params in directions:
        output_dir = BASE_OUTPUT_DIR / direction
        output_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"\n{'='*60}")
        print(f"Direction: {direction.upper()}")
        print(f"{'='*60}")
        
        for seed in SEEDS:
            logger = setup_logging(output_dir, direction, seed)
            
            pretrain_model(
                direction=direction,
                source_expr=source_expr,
                source_surv=source_surv,
                source_params_file=source_params,
                seed=seed,
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
    summary_df.to_csv(BASE_OUTPUT_DIR / 'pretraining_summary.csv', index=False)
    
    print(f"\n{'='*60}")
    print("Step 3.2 Complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {BASE_OUTPUT_DIR}")
    print("\nNext: Run Step 3.3 (Fine-tuning)")


if __name__ == "__main__":
    main()
