"""
Training script for DeepSurv model
Implements bidirectional transfer experiments: TCGA → ORIEN and ORIEN → TCGA
"""

import sys
sys.path.append('.')

import torch
import pandas as pd
import numpy as np
import logging
import yaml
import os
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Your existing modules
from src.data.dataset import SurvivalDataset, CombinedSurvivalDataset
from torch.utils.data import DataLoader

# DeepSurv model (from the previous artifact - save as src/models/deepsurv.py)
from src.models.deepsurv import DeepSurv, DeepSurvTrainer, calculate_concordance_index

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
def set_seed(seed=42):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data():
    """Load preprocessed data from disk."""
    logger.info("Loading preprocessed data...")
    
    # Load expression data
    tcga_expr = pd.read_csv("data/processed/tcga_preprocessed.csv", index_col=0)
    orien_expr = pd.read_csv("data/processed/orien_preprocessed.csv", index_col=0)
    
    # Load survival data
    surv_tcga = pd.read_csv("data/processed/surv_tcga_harmonized.csv", index_col=0)
    surv_orien = pd.read_csv("data/processed/surv_orien_harmonized.csv", index_col=0)
    
    logger.info(f"Loaded TCGA: {tcga_expr.shape[0]} samples, {tcga_expr.shape[1]} genes")
    logger.info(f"Loaded ORIEN: {orien_expr.shape[0]} samples, {orien_expr.shape[1]} genes")
    
    return tcga_expr, orien_expr, surv_tcga, surv_orien

def create_dataloaders(expression_df, survival_df, batch_size=32, valid_split=0.2):
    """Create train and validation dataloaders."""
    dataset = SurvivalDataset(expression_df, survival_df)
    
    # Split into train/validation
    n_samples = len(dataset)
    n_valid = int(n_samples * valid_split)
    n_train = n_samples - n_valid
    
    train_dataset, valid_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_valid],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Check if using MPS (Apple Silicon) to avoid pin_memory warning
    use_pin_memory = torch.cuda.is_available()  # Only use pin_memory with CUDA
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=use_pin_memory
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=use_pin_memory
    )
    
    return train_loader, valid_loader

def train_deepsurv_model(
    train_loader,
    valid_loader,
    n_features,
    config,
    experiment_name="deepsurv"
):
    """Train a DeepSurv model."""
    
    # Create model
    model = DeepSurv(
        n_features=n_features,
        hidden_sizes=config['hidden_sizes'],
        dropout=config['dropout'],
        activation=config['activation'],
        batch_norm=config['batch_norm']
    )
    
    # Create trainer
    trainer = DeepSurvTrainer(
        model=model,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Train model
    logger.info(f"Starting training for {experiment_name}...")
    history = trainer.fit(
        train_loader=train_loader,
        valid_loader=valid_loader,
        n_epochs=config['n_epochs'],
        early_stopping_patience=config['early_stopping_patience'],
        verbose=True
    )
    
    return model, trainer, history

def evaluate_transfer(model, test_loader, cohort_name):
    """Evaluate model on test cohort."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Calculate C-index
    c_index = calculate_concordance_index(model, test_loader, device)
    
    # Get predictions for additional metrics
    all_risks = []
    all_times = []
    all_events = []
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            risks = model.predict_risk(features).cpu().numpy()
            
            all_risks.extend(risks)
            all_times.extend(batch['time'].numpy())
            all_events.extend(batch['event'].numpy())
    
    results = {
        'cohort': cohort_name,
        'c_index': c_index,
        'n_samples': len(all_risks),
        'n_events': sum(all_events),
        'median_risk': np.median(all_risks),
        'risk_std': np.std(all_risks)
    }
    
    return results, np.array(all_risks)

def run_bidirectional_experiments():
    """Run all bidirectional transfer experiments."""
    
    # Set seed
    set_seed(42)
    
    # Load configuration
    config = {
        'hidden_sizes': [512, 256],  # Following DeepSurv paper
        'dropout': 0.4,
        'activation': 'relu',
        'batch_norm': True,
        'learning_rate': 0.001,
        'weight_decay': 0.01,
        'batch_size': 32,
        'n_epochs': 200,
        'early_stopping_patience': 20
    }
    
    # Load data
    tcga_expr, orien_expr, surv_tcga, surv_orien = load_data()
    
    # Data is stored as genes × samples, so genes are in rows
    n_features = tcga_expr.shape[0]  # 14,778 genes
    logger.info(f"Number of features (genes): {n_features}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/deepsurv_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    with open(f"{output_dir}/config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    results = {}
    
    # ============================================================
    # Experiment 1: Train on TCGA → Test on ORIEN
    # ============================================================
    logger.info("="*60)
    logger.info("EXPERIMENT 1: TCGA → ORIEN")
    logger.info("="*60)
    
    # Create TCGA dataloaders
    tcga_train_loader, tcga_valid_loader = create_dataloaders(
        tcga_expr, surv_tcga, config['batch_size']
    )
    
    # Create ORIEN test set (full dataset for testing)
    orien_dataset = SurvivalDataset(orien_expr, surv_orien)
    orien_test_loader = DataLoader(
        orien_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )
    
    # Train on TCGA
    tcga_model, tcga_trainer, tcga_history = train_deepsurv_model(
        tcga_train_loader,
        tcga_valid_loader,
        n_features,
        config,
        "TCGA_training"
    )
    
    # Evaluate on TCGA (in-domain)
    tcga_test_results, _ = evaluate_transfer(tcga_model, tcga_valid_loader, "TCGA_validation")
    logger.info(f"TCGA Validation C-index: {tcga_test_results['c_index']:.4f}")
    
    # Evaluate on ORIEN (transfer)
    tcga_to_orien_results, tcga_to_orien_risks = evaluate_transfer(
        tcga_model, orien_test_loader, "ORIEN_test"
    )
    logger.info(f"TCGA → ORIEN C-index: {tcga_to_orien_results['c_index']:.4f}")
    
    results['tcga_to_orien'] = {
        'training_history': tcga_history,
        'validation_performance': tcga_test_results,
        'transfer_performance': tcga_to_orien_results
    }
    
    # Save TCGA model
    torch.save(tcga_model.state_dict(), f"{output_dir}/tcga_model.pth")
    
    # ============================================================
    # Experiment 2: Train on ORIEN → Test on TCGA
    # ============================================================
    logger.info("="*60)
    logger.info("EXPERIMENT 2: ORIEN → TCGA")
    logger.info("="*60)
    
    # Create ORIEN dataloaders
    orien_train_loader, orien_valid_loader = create_dataloaders(
        orien_expr, surv_orien, config['batch_size']
    )
    
    # Create TCGA test set
    tcga_dataset = SurvivalDataset(tcga_expr, surv_tcga)
    tcga_test_loader = DataLoader(
        tcga_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )
    
    # Train on ORIEN
    orien_model, orien_trainer, orien_history = train_deepsurv_model(
        orien_train_loader,
        orien_valid_loader,
        n_features,
        config,
        "ORIEN_training"
    )
    
    # Evaluate on ORIEN (in-domain)
    orien_test_results, _ = evaluate_transfer(orien_model, orien_valid_loader, "ORIEN_validation")
    logger.info(f"ORIEN Validation C-index: {orien_test_results['c_index']:.4f}")
    
    # Evaluate on TCGA (transfer)
    orien_to_tcga_results, orien_to_tcga_risks = evaluate_transfer(
        orien_model, tcga_test_loader, "TCGA_test"
    )
    logger.info(f"ORIEN → TCGA C-index: {orien_to_tcga_results['c_index']:.4f}")
    
    results['orien_to_tcga'] = {
        'training_history': orien_history,
        'validation_performance': orien_test_results,
        'transfer_performance': orien_to_tcga_results
    }
    
    # Save ORIEN model
    torch.save(orien_model.state_dict(), f"{output_dir}/orien_model.pth")
    
    # ============================================================
    # Experiment 3: Train on Combined → Test on both
    # ============================================================
    logger.info("="*60)
    logger.info("EXPERIMENT 3: COMBINED TRAINING")
    logger.info("="*60)
    
    # Create combined dataset
    combined_dataset = CombinedSurvivalDataset(
        tcga_expr, surv_tcga, orien_expr, surv_orien
    )
    
    # Split combined dataset
    n_samples = len(combined_dataset)
    n_valid = int(n_samples * 0.2)
    n_train = n_samples - n_valid
    
    train_dataset, valid_dataset = torch.utils.data.random_split(
        combined_dataset, [n_train, n_valid],
        generator=torch.Generator().manual_seed(42)
    )
    
    combined_train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True
    )
    combined_valid_loader = DataLoader(
        valid_dataset, batch_size=config['batch_size'], shuffle=False
    )
    
    # Train on combined
    combined_model, combined_trainer, combined_history = train_deepsurv_model(
        combined_train_loader,
        combined_valid_loader,
        n_features,
        config,
        "Combined_training"
    )
    
    # Evaluate on both cohorts
    combined_tcga_results, _ = evaluate_transfer(combined_model, tcga_test_loader, "TCGA_combined")
    combined_orien_results, _ = evaluate_transfer(combined_model, orien_test_loader, "ORIEN_combined")
    
    logger.info(f"Combined → TCGA C-index: {combined_tcga_results['c_index']:.4f}")
    logger.info(f"Combined → ORIEN C-index: {combined_orien_results['c_index']:.4f}")
    
    results['combined'] = {
        'training_history': combined_history,
        'tcga_performance': combined_tcga_results,
        'orien_performance': combined_orien_results
    }
    
    # Save combined model
    torch.save(combined_model.state_dict(), f"{output_dir}/combined_model.pth")
    
    # ============================================================
    # Generate Summary Report
    # ============================================================
    logger.info("="*60)
    logger.info("SUMMARY OF RESULTS")
    logger.info("="*60)
    
    summary = {
        'TCGA → ORIEN': {
            'In-domain (TCGA)': results['tcga_to_orien']['validation_performance']['c_index'],
            'Transfer (ORIEN)': results['tcga_to_orien']['transfer_performance']['c_index']
        },
        'ORIEN → TCGA': {
            'In-domain (ORIEN)': results['orien_to_tcga']['validation_performance']['c_index'],
            'Transfer (TCGA)': results['orien_to_tcga']['transfer_performance']['c_index']
        },
        'Combined Training': {
            'TCGA': results['combined']['tcga_performance']['c_index'],
            'ORIEN': results['combined']['orien_performance']['c_index']
        }
    }
    
    # Print summary table
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY (C-index)")
    print("="*60)
    for exp_name, exp_results in summary.items():
        print(f"\n{exp_name}:")
        for metric_name, c_index in exp_results.items():
            print(f"  {metric_name:.<30} {c_index:.4f}")
    
    # Calculate bidirectional stability
    tcga_to_orien_cindex = results['tcga_to_orien']['transfer_performance']['c_index']
    orien_to_tcga_cindex = results['orien_to_tcga']['transfer_performance']['c_index']
    stability = 1.0 - abs(tcga_to_orien_cindex - orien_to_tcga_cindex)
    
    print("\n" + "="*60)
    print(f"BIDIRECTIONAL STABILITY SCORE: {stability:.4f}")
    print(f"(1.0 = perfect stability, 0.0 = complete instability)")
    print("="*60)
    
    # Save all results
    with open(f"{output_dir}/results.json", 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        json.dump(convert_to_serializable(results), f, indent=2)
    
    # Plot training curves
    plot_training_curves(results, output_dir)
    
    logger.info(f"\nResults saved to: {output_dir}")
    
    return results

def plot_training_curves(results, output_dir):
    """Plot training and validation curves."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    experiments = [
        ('tcga_to_orien', 'TCGA Training'),
        ('orien_to_tcga', 'ORIEN Training'),
        ('combined', 'Combined Training')
    ]
    
    for idx, (exp_key, exp_name) in enumerate(experiments):
        if exp_key in results:
            history = results[exp_key]['training_history']
            
            # Plot loss
            axes[0, idx].plot(history['train_loss'], label='Train', alpha=0.8)
            axes[0, idx].plot(history['valid_loss'], label='Valid', alpha=0.8)
            axes[0, idx].set_title(f'{exp_name} - Loss')
            axes[0, idx].set_xlabel('Epoch')
            axes[0, idx].set_ylabel('Cox Loss')
            axes[0, idx].legend()
            axes[0, idx].grid(True, alpha=0.3)
            
            # Plot C-index
            axes[1, idx].plot(history['train_cindex'], label='Train', alpha=0.8)
            axes[1, idx].plot(history['valid_cindex'], label='Valid', alpha=0.8)
            axes[1, idx].set_title(f'{exp_name} - C-index')
            axes[1, idx].set_xlabel('Epoch')
            axes[1, idx].set_ylabel('C-index')
            axes[1, idx].legend()
            axes[1, idx].grid(True, alpha=0.3)
            axes[1, idx].set_ylim([0.5, 1.0])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_curves.png", dpi=150)
    plt.close()
    
    logger.info(f"Training curves saved to {output_dir}/training_curves.png")


if __name__ == "__main__":
    # Run experiments
    results = run_bidirectional_experiments()
    
    # Print final message
    print("\n" + "="*60)
    print("DEEPSURV BASELINE EXPERIMENTS COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Analyze feature importance using gradient-based methods")
    print("2. Compare with Cox-PASNet for pathway-guided analysis")
    print("3. Identify stable biomarkers across both directions")
    print("4. Compare with Chapter 2 gene signatures")
    print("="*60)