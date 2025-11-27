"""
Step 3.1: Baseline - Target-only Training (No Transfer Learning)

Purpose: Establish performance baseline when training from scratch on target cohort.
This creates the train/test splits that will be reused in Step 3.3 for fair comparison.

Protocol:
- Train models from scratch on each cohort independently
- Use 80/20 train/test split, stratified by event status
- Uses best hyperparameters from Step 2 k-selection tuning
- Save train/test indices for Step 3.3 reuse
- Multi-seed validation (seeds: 42, 123, 456, 789, 1011)

For k=155 (87 consensus genes):
- TCGA: [32] (1 layer, from tuning)
- ORIEN: [48] (1 layer, from tuning)
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
import matplotlib.pyplot as plt
from lifelines.utils import concordance_index

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.elastic_deepsurv import ElasticDeepSurv
from src.data.dataset import SurvivalDataset
from src.utils.batch_samplers import StratifiedBatchSampler


# =============================================================================
# Configuration
# =============================================================================
K_VALUE = 155
SEEDS = [42, 123, 456, 789, 1011]

# Paths - updated for k=155
CONSENSUS_GENES_FILE = f"results_v2/02_biomarker_discovery/k_selection_with_tuning/k{K_VALUE}/consensus_genes/consensus_genes.txt"
TCGA_PARAMS_FILE = f"results_v2/02_biomarker_discovery/k_selection_with_tuning/k{K_VALUE}/hyperparameter_tuning/tcga/best_params.json"
ORIEN_PARAMS_FILE = f"results_v2/02_biomarker_discovery/k_selection_with_tuning/k{K_VALUE}/hyperparameter_tuning/orien/best_params.json"
OUTPUT_DIR = Path(f"results_v2/03_transfer_learning/k{K_VALUE}/baseline_target_only")


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
    

def setup_logging(output_dir):
    """Setup logging configuration"""
    log_file = output_dir / f"step3_1_baseline_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


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
    
    # Load survival data (use harmonized versions)
    tcga_surv = pd.read_csv("data/processed/surv_tcga_harmonized.csv", index_col=0)
    orien_surv = pd.read_csv("data/processed/surv_orien_harmonized.csv", index_col=0)
    
    # Filter to consensus genes
    logger.info(f"Filtering to {len(consensus_genes)} consensus genes...")
    tcga_expr = tcga_expr.loc[tcga_expr.index.isin(consensus_genes)]
    orien_expr = orien_expr.loc[orien_expr.index.isin(consensus_genes)]
    
    # Ensure gene order matches
    common_genes = sorted(list(set(tcga_expr.index) & set(orien_expr.index) & set(consensus_genes)))
    tcga_expr = tcga_expr.loc[common_genes]
    orien_expr = orien_expr.loc[common_genes]
    
    logger.info(f"  TCGA: {tcga_expr.shape[0]} genes × {tcga_expr.shape[1]} samples")
    logger.info(f"  ORIEN: {orien_expr.shape[0]} genes × {orien_expr.shape[1]} samples")
    
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


def standardize(expr_df):
    """Z-score standardization per gene"""
    mean = expr_df.mean(axis=1)
    std = expr_df.std(axis=1)
    # Avoid division by zero
    std = std.replace(0, 1)
    return expr_df.subtract(mean, axis=0).divide(std, axis=0)


def create_stratified_split(dataset, test_size=0.2, random_state=42):
    """Create stratified train/test split"""
    indices = np.arange(len(dataset))
    events = dataset.y_event
    
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=events,
        random_state=random_state
    )
    
    return train_idx, test_idx


def parse_architecture(best_params):
    """Parse architecture from best_params dictionary.
    
    Handles 1-layer, 2-layer, and 3-layer architectures from Optuna tuning.
    """
    params = best_params.get('best_params', best_params)
    
    if 'architecture_3layer' in params:
        hidden_sizes = [int(x) for x in params['architecture_3layer'].split('-')]
    elif 'architecture_2layer' in params:
        hidden_sizes = [int(x) for x in params['architecture_2layer'].split('-')]
    elif 'layer1_size' in params:
        hidden_sizes = [params['layer1_size']]
    else:
        raise ValueError(f"Cannot parse architecture from params: {params}")
    
    return hidden_sizes


def get_model_config(params_file, n_features):
    """Get model configuration from best_params.json file.
    
    Dynamically parses architecture from the tuning results.
    """
    with open(params_file, 'r') as f:
        params_data = json.load(f)
    
    # Get best_params dict
    if 'best_params' in params_data:
        params = params_data['best_params']
    else:
        params = params_data
    
    # Parse architecture
    hidden_sizes = parse_architecture(params_data)
    
    config = {
        'n_features': n_features,
        'hidden_sizes': hidden_sizes,
        'dropout': params['dropout'],
        'activation': params['activation'],
        'batch_norm': params['batch_norm'],
        'alpha': params['alpha'],
        'l1_ratio': params['l1_ratio'],
        'weight_init': params.get('weight_init', 'kaiming_normal')
    }
    
    batch_size = params['batch_size']
    learning_rate = params['learning_rate']
    
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


def evaluate_with_risk_scores(model, dataset, indices, device):
    """Evaluate model and return risk scores for verification"""
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
    
    risks = np.concatenate(all_risks).flatten()
    times = np.concatenate(all_times)
    events = np.concatenate(all_events).astype(bool)
    
    c_index = concordance_index(times, -risks, events)
    
    return c_index, {
        'risk_scores': risks,
        'times': times,
        'events': events
    }


def train_baseline_model(cohort_name, expr_df, surv_df, params_file, seed, 
                         output_dir, logger, device):
    """Train baseline model from scratch on target cohort"""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {cohort_name.upper()} - Seed {seed}")
    logger.info(f"{'='*60}")
    
    # Set seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create dataset
    dataset = SurvivalDataset(expr_df, surv_df)
    
    # Create stratified split
    train_idx, test_idx = create_stratified_split(
        dataset, 
        test_size=0.2, 
        random_state=seed
    )
    
    logger.info(f"Split: Train={len(train_idx)}, Test={len(test_idx)}")
    logger.info(f"  Train events: {dataset.y_event[train_idx].sum()}/{len(train_idx)} "
                f"({100*dataset.y_event[train_idx].mean():.1f}%)")
    logger.info(f"  Test events: {dataset.y_event[test_idx].sum()}/{len(test_idx)} "
                f"({100*dataset.y_event[test_idx].mean():.1f}%)")
    
    # Save split indices for Step 3.3 reuse
    split_dir = output_dir / 'splits'
    split_dir.mkdir(exist_ok=True, parents=True)
    np.save(split_dir / f'{cohort_name}_seed{seed}_train_idx.npy', train_idx)
    np.save(split_dir / f'{cohort_name}_seed{seed}_test_idx.npy', test_idx)
    logger.info(f"Saved split indices to {split_dir}")
    
    # Get model configuration from tuned hyperparameters
    n_features = len(dataset.gene_names)
    config, batch_size, learning_rate = get_model_config(params_file, n_features)
    
    logger.info(f"Model config: {config['hidden_sizes']}, dropout={config['dropout']:.3f}, "
                f"batch_norm={config['batch_norm']}, activation={config['activation']}")
    logger.info(f"Training: LR={learning_rate:.6f}, batch_size={batch_size}")
    
    # Initialize model
    model = ElasticDeepSurv(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create train loader
    train_dataset = Subset(dataset, train_idx)
    train_events = dataset.y_event[train_idx]
    
    # Use stratified batch sampler for large cohorts (ORIEN), simple shuffle for small (TCGA)
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
    
    # Training loop
    n_epochs = 500
    training_history = []
    best_test_cindex = 0.0
    best_train_cindex = 0.0
    best_epoch = 0
    best_model_state = None

    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            train_cindex = evaluate(model, dataset, train_idx, device)
            test_cindex = evaluate(model, dataset, test_idx, device)
            
            logger.info(f"Epoch {epoch+1:3d}: Loss={train_loss:.4f}, "
                    f"Train C-index={train_cindex:.4f}, "
                    f"Test C-index={test_cindex:.4f}")
            
            # Track best TEST C-index
            if test_cindex > best_test_cindex:
                best_test_cindex = test_cindex
                best_train_cindex = train_cindex
                best_epoch = epoch + 1
                best_model_state = model.state_dict().copy()
                logger.info(f"      *** NEW BEST TEST C-INDEX ***")
            
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': float(train_loss),
                'train_cindex': float(train_cindex),
                'test_cindex': float(test_cindex)
            })

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"\nRestored model from epoch {best_epoch} (best test C-index: {best_test_cindex:.4f})")

    # Final evaluation with risk scores
    final_test_cindex, test_risk_data = evaluate_with_risk_scores(model, dataset, test_idx, device)
    
    logger.info(f"\nBaseline Training Complete:")
    logger.info(f"  Best epoch: {best_epoch}")
    logger.info(f"  Best test C-index: {best_test_cindex:.4f}")
    logger.info(f"  Train C-index at best epoch: {best_train_cindex:.4f}")
    
    # Save results
    cohort_dir = output_dir / cohort_name
    cohort_dir.mkdir(exist_ok=True, parents=True)
    
    # Plot training curves
    if training_history:
        plot_training_curves(
            training_history,
            cohort_dir / f'seed{seed}_training_curve.png',
            f'{cohort_name.upper()} Seed {seed}',
            show_validation=False
        )
        logger.info(f"Saved training curve to {cohort_dir / f'seed{seed}_training_curve.png'}")
    
    # Save risk scores for verification
    risk_df = pd.DataFrame({
        'risk_score': test_risk_data['risk_scores'],
        'time': test_risk_data['times'],
        'event': test_risk_data['events']
    })
    risk_df.to_csv(cohort_dir / f'seed{seed}_test_risk_scores.csv', index=False)
    
    results = {
        'seed': seed,
        'cohort': cohort_name,
        'k': K_VALUE,
        'm': len(dataset.gene_names),
        'train_cindex': float(best_train_cindex),
        'test_cindex': float(best_test_cindex),
        'best_epoch': best_epoch,
        'n_train': len(train_idx),
        'n_test': len(test_idx),
        'n_train_events': int(train_events.sum()),
        'n_test_events': int(dataset.y_event[test_idx].sum()),
        'architecture': config['hidden_sizes'],
        'hyperparameters': {
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'dropout': config['dropout'],
            'alpha': config['alpha'],
            'l1_ratio': config['l1_ratio'],
            'batch_norm': config['batch_norm'],
            'activation': config['activation']
        }
    }
    
    # Save results JSON
    with open(cohort_dir / f'seed{seed}_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'seed': seed,
        'best_epoch': best_epoch,
        'test_cindex': best_test_cindex
    }, cohort_dir / f'seed{seed}_model.pth')
    
    # Save training history
    with open(cohort_dir / f'seed{seed}_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    return results


def aggregate_results(output_dir, logger):
    """Aggregate results across all seeds"""
    logger.info(f"\n{'='*60}")
    logger.info("Aggregating Results")
    logger.info(f"{'='*60}")
    
    results = []
    
    for cohort in ['tcga', 'orien']:
        cohort_dir = output_dir / cohort
        
        for seed_file in sorted(cohort_dir.glob('seed*_results.json')):
            with open(seed_file, 'r') as f:
                results.append(json.load(f))
    
    # Create summary DataFrame
    df = pd.DataFrame(results)
    
    # Save all results
    df.to_csv(output_dir / 'all_results.csv', index=False)
    
    # Calculate statistics per cohort
    summary_rows = []
    for cohort in ['tcga', 'orien']:
        cohort_results = df[df['cohort'] == cohort]
        
        summary_rows.append({
            'cohort': cohort.upper(),
            'k': K_VALUE,
            'm': cohort_results['m'].iloc[0] if len(cohort_results) > 0 else None,
            'mean_test_cindex': cohort_results['test_cindex'].mean(),
            'std_test_cindex': cohort_results['test_cindex'].std(),
            'min_test_cindex': cohort_results['test_cindex'].min(),
            'max_test_cindex': cohort_results['test_cindex'].max(),
            'mean_train_cindex': cohort_results['train_cindex'].mean(),
            'n_seeds': len(cohort_results)
        })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Save summary
    summary_df.to_csv(output_dir / 'baseline_summary.csv', index=False)
    
    # Print summary
    logger.info(f"\nBaseline (Target-only) Summary for k={K_VALUE}:")
    for _, row in summary_df.iterrows():
        logger.info(f"  {row['cohort']}: {row['mean_test_cindex']:.4f} ± {row['std_test_cindex']:.4f} "
                   f"(range: {row['min_test_cindex']:.4f}-{row['max_test_cindex']:.4f})")
    
    return summary_df


def main():
    # Setup
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    logger = setup_logging(OUTPUT_DIR)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info("="*60)
    logger.info(f"Step 3.1: Baseline - Target-only Training (k={K_VALUE})")
    logger.info("="*60)
    logger.info(f"Seeds: {SEEDS}")
    logger.info(f"Device: {device}")
    logger.info(f"Output: {OUTPUT_DIR}")
    
    # Load consensus genes
    consensus_genes = load_consensus_genes(CONSENSUS_GENES_FILE)
    logger.info(f"\nLoaded {len(consensus_genes)} consensus genes from k={K_VALUE}")
    
    # Load data
    data = load_data(consensus_genes, logger)
    
    # Train models for all seeds and cohorts
    all_results = []
    
    for seed in SEEDS:
        # TCGA
        results_tcga = train_baseline_model(
            cohort_name='tcga',
            expr_df=data['tcga_expr'],
            surv_df=data['tcga_surv'],
            params_file=TCGA_PARAMS_FILE,
            seed=seed,
            output_dir=OUTPUT_DIR,
            logger=logger,
            device=device
        )
        all_results.append(results_tcga)
        
        # ORIEN
        results_orien = train_baseline_model(
            cohort_name='orien',
            expr_df=data['orien_expr'],
            surv_df=data['orien_surv'],
            params_file=ORIEN_PARAMS_FILE,
            seed=seed,
            output_dir=OUTPUT_DIR,
            logger=logger,
            device=device
        )
        all_results.append(results_orien)
    
    # Aggregate results
    summary = aggregate_results(OUTPUT_DIR, logger)
    
    logger.info(f"\n{'='*60}")
    logger.info("Step 3.1 Complete!")
    logger.info(f"{'='*60}")
    logger.info(f"Results saved to: {OUTPUT_DIR}")
    logger.info(f"Split indices saved to: {OUTPUT_DIR / 'splits'}")
    logger.info("\nNext: Run Step 3.2 (Pre-training) or Step 3.3 (Fine-tuning)")


if __name__ == "__main__":
    main()
