"""
Step 3.2: Pre-training Phase

Purpose: Pre-train models on FULL source cohort to learn generalizable features.

Protocol:
- Train on FULL SOURCE cohort (no split - proper transfer learning)
- Use TARGET cohort's architecture with SOURCE cohort's hyperparameters
- Convergence-based stopping: stop if loss doesn't decrease by >0.001 for 20 epochs
- Maximum epochs: 100
- Save final model after convergence
- Multi-seed validation (seeds: 42, 123, 456, 789, 1011)

Directions:
- ORIEN→TCGA: Pre-train on full ORIEN (1,112 samples) using TCGA architecture [32]
- TCGA→ORIEN: Pre-train on full TCGA (339 samples) using ORIEN architecture [48]

Hyperparameters:
- Architecture: From TARGET cohort's k=155 tuned params
- Training params (lr, dropout, alpha, etc.): From SOURCE cohort's k=155 tuned params

References:
- Transfer learning pretraining should use full source data (Yosinski et al., 2014)
- Regularization (dropout, elastic net) prevents overfitting without validation set
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
from torch.utils.data import DataLoader
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


def load_hyperparameters(params_file):
    """Load hyperparameters from JSON file"""
    with open(params_file, 'r') as f:
        data = json.load(f)
    return data['best_params']


def get_architecture_from_params(params):
    """Extract architecture (hidden layer sizes) from params"""
    n_layers = params.get('n_layers', 1)
    
    if n_layers == 1:
        return [params['layer1_size']]
    elif n_layers == 2:
        return [params['layer1_size'], params['layer2_size']]
    elif n_layers == 3:
        return [params['layer1_size'], params['layer2_size'], params['layer3_size']]
    else:
        raise ValueError(f"Unsupported n_layers: {n_layers}")


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
    
    if len(available_tcga) != len(consensus_genes):
        missing = set(consensus_genes) - set(available_tcga)
        logger.warning(f"Missing {len(missing)} genes in TCGA: {list(missing)[:5]}...")
    
    if len(available_orien) != len(consensus_genes):
        missing = set(consensus_genes) - set(available_orien)
        logger.warning(f"Missing {len(missing)} genes in ORIEN: {list(missing)[:5]}...")
    
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


def get_pretrain_config(direction, source_params_file, target_params_file, n_features):
    """
    Get pre-training configuration.
    
    Uses TARGET architecture with SOURCE hyperparameters.
    
    Args:
        direction: 'orien_to_tcga' or 'tcga_to_orien'
        source_params_file: Path to source cohort's best_params.json
        target_params_file: Path to target cohort's best_params.json
        n_features: Number of input features
    
    Returns:
        model_config, batch_size, learning_rate
    """
    source_params = load_hyperparameters(source_params_file)
    target_params = load_hyperparameters(target_params_file)
    
    # Use TARGET architecture
    architecture = get_architecture_from_params(target_params)
    
    # Use SOURCE hyperparameters for training
    config = {
        'n_features': n_features,
        'hidden_sizes': architecture,
        'dropout': source_params['dropout'],
        'activation': source_params['activation'],
        'batch_norm': source_params['batch_norm'],
        'alpha': source_params['alpha'],
        'l1_ratio': source_params['l1_ratio'],
        'weight_init': source_params.get('weight_init', 'xavier_uniform')
    }
    
    batch_size = source_params['batch_size']
    learning_rate = source_params['learning_rate']
    
    return config, batch_size, learning_rate


def plot_training_curves(history, output_path, title):
    """Plot training curves"""
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    train_cindex = [h['train_cindex'] for h in history]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    ax1.plot(epochs, train_loss, 'b-', linewidth=1.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title(f'{title} - Training Loss')
    ax1.grid(alpha=0.3)
    
    # C-index
    ax2.plot(epochs, train_cindex, 'g-', linewidth=1.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Training C-index')
    ax2.set_title(f'{title} - Training C-index')
    ax2.grid(alpha=0.3)
    
    # Mark convergence point if early stopped
    if len(history) > 0 and 'converged_epoch' in history[-1]:
        converged_epoch = history[-1]['converged_epoch']
        if converged_epoch is not None:
            ax1.axvline(converged_epoch, color='r', linestyle='--', 
                       label=f'Converged: {converged_epoch}')
            ax2.axvline(converged_epoch, color='r', linestyle='--',
                       label=f'Converged: {converged_epoch}')
            ax1.legend()
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


def evaluate_training(model, dataset, device):
    """Evaluate model on full training data"""
    model.eval()
    
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    all_risks = []
    all_times = []
    all_events = []
    
    with torch.no_grad():
        for batch in loader:
            features = batch['features'].to(device)
            risk = model(features)
            
            all_risks.append(risk.cpu().numpy())
            all_times.append(batch['time'].numpy())
            all_events.append(batch['event'].numpy())
    
    risks = np.concatenate(all_risks)
    times = np.concatenate(all_times)
    events = np.concatenate(all_events).astype(bool)
    
    c_index = concordance_index(times, -risks, events)
    
    return c_index


def pretrain_model(direction, source_expr, source_surv, source_params_file, 
                   target_params_file, n_features, seed, output_dir, logger, device):
    """
    Pre-train model on FULL source cohort.
    
    Args:
        direction: 'orien_to_tcga' or 'tcga_to_orien'
        source_expr: Source cohort expression data (full cohort)
        source_surv: Source cohort survival data (full cohort)
        source_params_file: Path to source cohort's best_params.json
        target_params_file: Path to target cohort's best_params.json
        n_features: Number of input features
        seed: Random seed
        output_dir: Output directory
        logger: Logger instance
        device: torch device
    """
    
    logger.info(f"{'='*60}")
    logger.info(f"Pre-training: {direction.upper()} - Seed {seed}")
    logger.info(f"{'='*60}")
    
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create dataset from FULL source cohort
    dataset = SurvivalDataset(source_expr, source_surv)
    
    logger.info(f"Training on FULL source cohort: {len(dataset)} samples")
    logger.info(f"  Events: {dataset.y_event.sum()}/{len(dataset)} "
                f"({100*dataset.y_event.mean():.1f}%)")
    
    # Get model configuration
    config, batch_size, learning_rate = get_pretrain_config(
        direction,
        source_params_file,
        target_params_file,
        n_features
    )
    
    logger.info(f"\nModel Configuration:")
    logger.info(f"  Architecture (from TARGET): {config['hidden_sizes']}")
    logger.info(f"  Input features: {config['n_features']}")
    logger.info(f"\nTraining Hyperparameters (from SOURCE):")
    logger.info(f"  Learning rate: {learning_rate:.6f}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Dropout: {config['dropout']:.4f}")
    logger.info(f"  Alpha: {config['alpha']:.6f}")
    logger.info(f"  L1 ratio: {config['l1_ratio']:.4f}")
    logger.info(f"  Batch norm: {config['batch_norm']}")
    logger.info(f"  Activation: {config['activation']}")
    
    # Initialize model
    model = ElasticDeepSurv(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total parameters: {n_params:,}")
    
    # Create data loader
    events = dataset.y_event
    n_samples = len(dataset)
    
    # Use stratified batch sampler for larger cohorts
    if n_samples >= 500:
        logger.info("\nUsing StratifiedBatchSampler")
        train_batch_sampler = StratifiedBatchSampler(
            events=events,
            batch_size=batch_size,
            min_events_per_batch=2,
            shuffle=True,
            drop_last=False
        )
        train_loader = DataLoader(dataset, batch_sampler=train_batch_sampler)
    else:
        logger.info("\nUsing simple random shuffling (smaller cohort)")
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training parameters
    max_epochs = 100
    convergence_threshold = 0.001
    patience = 20
    
    logger.info(f"\nTraining Settings:")
    logger.info(f"  Max epochs: {max_epochs}")
    logger.info(f"  Convergence threshold: {convergence_threshold}")
    logger.info(f"  Patience: {patience} epochs")
    
    # Training loop with convergence-based stopping
    training_history = []
    best_loss = float('inf')
    epochs_without_improvement = 0
    converged_epoch = None
    
    logger.info(f"\nStarting pre-training...")
    
    for epoch in range(max_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_cindex = evaluate_training(model, dataset, device)
        
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'train_cindex': float(train_cindex)
        })
        
        # Check for convergence
        loss_improvement = best_loss - train_loss
        
        if loss_improvement > convergence_threshold:
            best_loss = train_loss
            epochs_without_improvement = 0
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1:3d}: Loss={train_loss:.4f}, "
                           f"C-index={train_cindex:.4f}")
        else:
            epochs_without_improvement += 1
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1:3d}: Loss={train_loss:.4f}, "
                           f"C-index={train_cindex:.4f} "
                           f"(no improvement: {epochs_without_improvement}/{patience})")
            
            if epochs_without_improvement >= patience:
                converged_epoch = epoch + 1 - patience
                logger.info(f"\nConverged at epoch {converged_epoch} "
                           f"(stopped at epoch {epoch+1})")
                break
    
    if converged_epoch is None:
        converged_epoch = max_epochs
        logger.info(f"\nReached max epochs ({max_epochs})")
    
    # Add convergence info to history
    if training_history:
        training_history[-1]['converged_epoch'] = converged_epoch
    
    # Final evaluation
    final_train_cindex = evaluate_training(model, dataset, device)
    final_train_loss = training_history[-1]['train_loss'] if training_history else 0.0
    
    logger.info(f"\nPre-training Complete:")
    logger.info(f"  Final epoch: {len(training_history)}")
    logger.info(f"  Converged at epoch: {converged_epoch}")
    logger.info(f"  Final train loss: {final_train_loss:.4f}")
    logger.info(f"  Final train C-index: {final_train_cindex:.4f}")
    
    # Plot training curves
    if training_history:
        plot_training_curves(
            training_history,
            output_dir / f'seed{seed}_training_curve.png',
            f'{direction.upper()} Seed {seed}'
        )
        logger.info(f"Saved training curve to {output_dir / f'seed{seed}_training_curve.png'}")
    
    # Save results
    results = {
        'direction': direction,
        'seed': seed,
        'n_samples': len(dataset),
        'n_events': int(dataset.y_event.sum()),
        'n_features': n_features,
        'converged_epoch': converged_epoch,
        'total_epochs': len(training_history),
        'final_train_loss': float(final_train_loss),
        'final_train_cindex': float(final_train_cindex),
        'architecture': config['hidden_sizes'],
        'hyperparameters': {
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'dropout': config['dropout'],
            'alpha': config['alpha'],
            'l1_ratio': config['l1_ratio'],
            'batch_norm': config['batch_norm'],
            'activation': config['activation']
        },
        'training_settings': {
            'max_epochs': max_epochs,
            'convergence_threshold': convergence_threshold,
            'patience': patience
        }
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
        'converged_epoch': converged_epoch,
        'final_train_cindex': final_train_cindex,
        'n_features': n_features
    }, output_dir / f'seed{seed}_pretrained_model.pth')
    
    logger.info(f"Saved pre-trained model to {output_dir / f'seed{seed}_pretrained_model.pth'}")
    
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
    
    if not results:
        logger.warning("No results found to aggregate")
        return None
    
    # Create summary DataFrame
    df = pd.DataFrame(results)
    
    summary = {
        'direction': direction,
        'n_seeds': len(results),
        'mean_train_cindex': df['final_train_cindex'].mean(),
        'std_train_cindex': df['final_train_cindex'].std(),
        'min_train_cindex': df['final_train_cindex'].min(),
        'max_train_cindex': df['final_train_cindex'].max(),
        'mean_converged_epoch': df['converged_epoch'].mean(),
        'mean_train_loss': df['final_train_loss'].mean()
    }
    
    # Print summary
    logger.info(f"\n{direction.upper()} Pre-training Summary:")
    logger.info(f"  Training C-index: {summary['mean_train_cindex']:.4f} ± {summary['std_train_cindex']:.4f}")
    logger.info(f"  Range: {summary['min_train_cindex']:.4f} - {summary['max_train_cindex']:.4f}")
    logger.info(f"  Average convergence epoch: {summary['mean_converged_epoch']:.1f}")
    logger.info(f"  Average final loss: {summary['mean_train_loss']:.4f}")
    
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
    BASE_OUTPUT_DIR = Path(f"results_v2/03_transfer_learning/k{K_VALUE}/pretrained")
    
    # Setup
    BASE_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*60)
    print("Step 3.2: Pre-training Phase")
    print("="*60)
    print(f"K-value: {K_VALUE}")
    print(f"Seeds: {SEEDS}")
    print(f"Device: {device}")
    print(f"Output: {BASE_OUTPUT_DIR}")
    
    # Verify input files exist
    for filepath, desc in [
        (CONSENSUS_GENES_FILE, "Consensus genes"),
        (TCGA_PARAMS_FILE, "TCGA hyperparameters"),
        (ORIEN_PARAMS_FILE, "ORIEN hyperparameters")
    ]:
        if not Path(filepath).exists():
            raise FileNotFoundError(f"{desc} not found: {filepath}")
        print(f"✓ Found {desc}: {filepath}")
    
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
    n_features = data['n_features']
    print(f"Using {n_features} features")
    
    # Pre-train for both directions
    directions = [
        {
            'name': 'orien_to_tcga',
            'source_expr': data['orien_expr'],
            'source_surv': data['orien_surv'],
            'source_params': ORIEN_PARAMS_FILE,
            'target_params': TCGA_PARAMS_FILE
        },
        {
            'name': 'tcga_to_orien',
            'source_expr': data['tcga_expr'],
            'source_surv': data['tcga_surv'],
            'source_params': TCGA_PARAMS_FILE,
            'target_params': ORIEN_PARAMS_FILE
        }
    ]
    
    all_summaries = []
    
    for dir_config in directions:
        direction = dir_config['name']
        output_dir = BASE_OUTPUT_DIR / direction
        output_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"\n{'='*60}")
        print(f"Direction: {direction.upper()}")
        print(f"{'='*60}")
        
        for seed in SEEDS:
            logger = setup_logging(output_dir, direction, seed)
            
            pretrain_model(
                direction=direction,
                source_expr=dir_config['source_expr'],
                source_surv=dir_config['source_surv'],
                source_params_file=dir_config['source_params'],
                target_params_file=dir_config['target_params'],
                n_features=n_features,
                seed=seed,
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
        summary_df.to_csv(BASE_OUTPUT_DIR / 'pretraining_summary.csv', index=False)
        
        print(f"\n{'='*60}")
        print("Pre-training Summary")
        print(f"{'='*60}")
        for summary in all_summaries:
            print(f"\n{summary['direction'].upper()}:")
            print(f"  Train C-index: {summary['mean_train_cindex']:.4f} ± {summary['std_train_cindex']:.4f}")
            print(f"  Avg convergence: {summary['mean_converged_epoch']:.1f} epochs")
    
    print(f"\n{'='*60}")
    print("Step 3.2 Complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {BASE_OUTPUT_DIR}")
    print("\nNext: Run Step 3.3 (Fine-tuning)")


if __name__ == "__main__":
    main()
