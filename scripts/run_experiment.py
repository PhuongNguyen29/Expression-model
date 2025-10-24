"""
Unified Experiment Runner
Single entry point for running any model + dataset combination
Usage: python scripts/run_experiment.py --config config/experiments/deepsurv_full.yaml
"""

import sys
sys.path.append('.')

import argparse
import yaml
import torch
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
import json
from pathlib import Path

# Import factories
from src.data.data_factory import load_dataset_from_config
from src.models.model_factory import create_model_from_config
from src.data.dataset import SurvivalDataset, CombinedSurvivalDataset
from torch.utils.data import DataLoader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dataloaders(expression_df, survival_df, config):
    """Create train and validation dataloaders."""
    dataset = SurvivalDataset(expression_df, survival_df)
    
    # Get split ratio from config
    valid_split = config['training'].get('valid_split', 0.2)
    
    # Split into train/validation
    n_samples = len(dataset)
    n_valid = int(n_samples * valid_split)
    n_train = n_samples - n_valid
    
    seed = config['experiment'].get('seed', 42)
    train_dataset, valid_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_valid],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Get batch size and num_workers from config
    batch_size = config['training'].get('batch_size', 32)
    num_workers = config['compute'].get('num_workers', 2)
    
    # Only use pin_memory with CUDA
    use_pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    logger.info(f"Created dataloaders: Train={len(train_dataset)}, Valid={len(valid_dataset)}")
    
    return train_loader, valid_loader


def evaluate_model(model, test_loader, cohort_name, device):
    """Evaluate model on test set."""
    from src.models.deepsurv import calculate_concordance_index
    
    model = model.to(device)
    c_index = calculate_concordance_index(model, test_loader, device)
    
    # Get predictions
    model.eval()
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
        'c_index': float(c_index),
        'n_samples': len(all_risks),
        'n_events': int(sum(all_events)),
        'median_risk': float(np.median(all_risks)),
        'risk_std': float(np.std(all_risks))
    }
    
    return results


def run_bidirectional_experiments(config, data):
    """
    Run bidirectional transfer experiments.
    
    Three experiments:
    1. Train on TCGA → Test on ORIEN
    2. Train on ORIEN → Test on TCGA  

    """
    
    # Get experiment config
    exp_config = config['experiment']
    training_config = config['training']
    eval_config = config['evaluation']
    
    # Unpack data
    tcga_expr = data['tcga_expr']
    orien_expr = data['orien_expr']
    surv_tcga = data['surv_tcga']
    surv_orien = data['surv_orien']
    
    # Get number of features (genes in rows)
    n_features = tcga_expr.shape[0]
    logger.info(f"Number of features: {n_features}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = exp_config['name']
    output_dir = Path(config['paths']['results_dir']) / f"{exp_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Results will be saved to: {output_dir}")
    
    # Save config
    with open(output_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    results = {}
    directions = eval_config.get('directions', ['tcga_to_orien', 'orien_to_tcga'])
    
    # Determine device
    device_config = config.get('compute', {}).get('device', 'auto')
    if device_config == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = device_config
    
    logger.info(f"Using device: {device}")
    
    # ============================================================
    # Experiment 1: TCGA → ORIEN
    # ============================================================
    if 'tcga_to_orien' in directions:
        logger.info("="*60)
        logger.info("EXPERIMENT 1: TCGA → ORIEN")
        logger.info("="*60)
        
        # Create TCGA dataloaders
        tcga_train_loader, tcga_valid_loader = create_dataloaders(
            tcga_expr, surv_tcga, config
        )
        
        # Create ORIEN test loader
        orien_dataset = SurvivalDataset(orien_expr, surv_orien)
        orien_test_loader = DataLoader(
            orien_dataset,
            batch_size=training_config['batch_size'],
            shuffle=False
        )
        
        # Create model and trainer
        tcga_model, tcga_trainer = create_model_from_config(config, n_features)
        
        # Train
        logger.info("Training on TCGA...")
        tcga_history = tcga_trainer.fit(
            train_loader=tcga_train_loader,
            valid_loader=tcga_valid_loader,
            n_epochs=training_config['num_epochs'],
            early_stopping_patience=training_config['early_stopping_patience'],
            verbose=True
        )
        
        # Evaluate
        tcga_valid_results = evaluate_model(tcga_model, tcga_valid_loader, "TCGA_valid", device)
        tcga_to_orien_results = evaluate_model(tcga_model, orien_test_loader, "ORIEN_test", device)
        
        logger.info(f"TCGA Validation C-index: {tcga_valid_results['c_index']:.4f}")
        logger.info(f"TCGA → ORIEN C-index: {tcga_to_orien_results['c_index']:.4f}")
        
        results['tcga_to_orien'] = {
            'training_history': tcga_history,
            'validation_performance': tcga_valid_results,
            'transfer_performance': tcga_to_orien_results
        }
        
        # Save model
        torch.save(tcga_model.state_dict(), output_dir / "tcga_model.pth")
    
    # ============================================================
    # Experiment 2: ORIEN → TCGA
    # ============================================================
    if 'orien_to_tcga' in directions:
        logger.info("="*60)
        logger.info("EXPERIMENT 2: ORIEN → TCGA")
        logger.info("="*60)
        
        # Create ORIEN dataloaders
        orien_train_loader, orien_valid_loader = create_dataloaders(
            orien_expr, surv_orien, config
        )
        
        # Create TCGA test loader
        tcga_dataset = SurvivalDataset(tcga_expr, surv_tcga)
        tcga_test_loader = DataLoader(
            tcga_dataset,
            batch_size=training_config['batch_size'],
            shuffle=False
        )
        
        # Create model and trainer
        orien_model, orien_trainer = create_model_from_config(config, n_features)
        
        # Train
        logger.info("Training on ORIEN...")
        orien_history = orien_trainer.fit(
            train_loader=orien_train_loader,
            valid_loader=orien_valid_loader,
            n_epochs=training_config['num_epochs'],
            early_stopping_patience=training_config['early_stopping_patience'],
            verbose=True
        )
        
        # Evaluate
        orien_valid_results = evaluate_model(orien_model, orien_valid_loader, "ORIEN_valid", device)
        orien_to_tcga_results = evaluate_model(orien_model, tcga_test_loader, "TCGA_test", device)
        
        logger.info(f"ORIEN Validation C-index: {orien_valid_results['c_index']:.4f}")
        logger.info(f"ORIEN → TCGA C-index: {orien_to_tcga_results['c_index']:.4f}")
        
        results['orien_to_tcga'] = {
            'training_history': orien_history,
            'validation_performance': orien_valid_results,
            'transfer_performance': orien_to_tcga_results
        }
        
        # Save model
        torch.save(orien_model.state_dict(), output_dir / "orien_model.pth")
    
    
    
    # ============================================================
    # Generate Summary
    # ============================================================
    logger.info("="*60)
    logger.info("SUMMARY OF RESULTS")
    logger.info("="*60)
    
    summary = {}
    if 'tcga_to_orien' in results:
        summary['TCGA → ORIEN'] = {
            'In-domain (TCGA)': results['tcga_to_orien']['validation_performance']['c_index'],
            'Transfer (ORIEN)': results['tcga_to_orien']['transfer_performance']['c_index']
        }
    
    if 'orien_to_tcga' in results:
        summary['ORIEN → TCGA'] = {
            'In-domain (ORIEN)': results['orien_to_tcga']['validation_performance']['c_index'],
            'Transfer (TCGA)': results['orien_to_tcga']['transfer_performance']['c_index']
        }
    
    
    # Print summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY (C-index)")
    print("="*60)
    for exp_name, exp_results in summary.items():
        print(f"\n{exp_name}:")
        for metric_name, c_index in exp_results.items():
            print(f"  {metric_name:.<30} {c_index:.4f}")
    
    # Calculate bidirectional stability if available
    if 'tcga_to_orien' in results and 'orien_to_tcga' in results:
        tcga_to_orien_c = results['tcga_to_orien']['transfer_performance']['c_index']
        orien_to_tcga_c = results['orien_to_tcga']['transfer_performance']['c_index']
        stability = 1.0 - abs(tcga_to_orien_c - orien_to_tcga_c)
        
        print("\n" + "="*60)
        print(f"BIDIRECTIONAL STABILITY SCORE: {stability:.4f}")
        print("(1.0 = perfect stability, 0.0 = complete instability)")
        print("="*60)
    
    # Save results
    with open(output_dir / "results.json", 'w') as f:
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
    
    logger.info(f"\nResults saved to: {output_dir}")
    
    return results, output_dir


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run survival analysis experiment')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to experiment config file (e.g., config/experiments/deepsurv_full.yaml)'
    )
    
    args = parser.parse_args()
    
    # Load config
    logger.info(f"Loading config from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed
    seed = config['experiment'].get('seed', 42)
    set_seed(seed)
    logger.info(f"Set random seed: {seed}")
    
    # Log experiment details
    logger.info("="*60)
    logger.info(f"Experiment: {config['experiment']['name']}")
    logger.info(f"Description: {config['experiment']['description']}")
    logger.info(f"Dataset: {config['dataset']['name']}")
    logger.info(f"Model: {config['model']['type']}")
    logger.info("="*60)
    
    # Load dataset
    logger.info("\nLoading dataset...")
    data = load_dataset_from_config(config)
    
    # Run experiments
    logger.info("\nStarting experiments...")
    results, output_dir = run_bidirectional_experiments(config, data)
    
    # Final message
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE!")
    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    print(f"Config: {output_dir}/config.yaml")
    print(f"Results: {output_dir}/results.json")
    print("="*60)


if __name__ == "__main__":
    main()
