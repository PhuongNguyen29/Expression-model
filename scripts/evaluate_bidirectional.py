"""
Bidirectional evaluation: Test TCGA model on ORIEN and vice versa
"""

import sys
sys.path.append('.')

import torch
import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path

from src.data.dataset import SurvivalDataset
from torch.utils.data import DataLoader
from src.models.deepsurv import DeepSurv, calculate_concordance_index
from lifelines.utils import concordance_index

def load_model_from_params(best_params: dict, n_features: int, device: str) -> DeepSurv:
    """Reconstruct model from saved hyperparameters."""
    
    # Reconstruct architecture
    hidden_sizes = []
    n_layers = best_params['n_layers']
    
    if 'single_layer_size' in best_params:
        # TCGA 1-layer
        hidden_sizes = [best_params['single_layer_size']]
    elif 'first_layer_size' in best_params:
        # TCGA 2-layer
        hidden_sizes = [best_params['first_layer_size'], best_params['second_layer_size']]
    elif 'pattern_2' in best_params:
        # ORIEN 2-layer
        hidden_sizes = [int(x) for x in best_params['pattern_2'].split('-')]
    elif 'pattern_3' in best_params:
        # ORIEN 3-layer
        hidden_sizes = [int(x) for x in best_params['pattern_3'].split('-')]
    else:
        raise ValueError(f"Cannot reconstruct architecture from: {best_params}")
    
    # Create model
    model = DeepSurv(
        n_features=n_features,
        hidden_sizes=hidden_sizes,
        dropout=best_params['dropout'],
        activation=best_params['activation'],
        batch_norm=best_params.get('batch_norm', False),
        weight_init=best_params['weight_init']
    )
    
    return model.to(device)


def evaluate_model(model, data_loader, device):
    """Evaluate model and return C-index."""
    model.eval()
    
    all_risks = []
    all_times = []
    all_events = []
    
    with torch.no_grad():
        for batch in data_loader:
            features = batch['features'].to(device)
            times = batch['time']
            events = batch['event']
            
            # Get predictions
            log_hazards = model(features)
            risks = torch.exp(log_hazards).squeeze().cpu().numpy()
            
            all_risks.extend(risks)
            all_times.extend(times.numpy())
            all_events.extend(events.numpy())
    
    # Calculate C-index
    c_index = concordance_index(all_times, -np.array(all_risks), all_events)
    return c_index


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading data...")
    tcga_expr = pd.read_csv("data/processed/tcga_preprocessed.csv", index_col=0)
    orien_expr = pd.read_csv("data/processed/orien_preprocessed.csv", index_col=0)
    surv_tcga = pd.read_csv("data/processed/surv_tcga_harmonized.csv", index_col=0)
    surv_orien = pd.read_csv("data/processed/surv_orien_harmonized.csv", index_col=0)
    
    print(f"TCGA: {tcga_expr.shape[1]} samples, {tcga_expr.shape[0]} features")
    print(f"ORIEN: {orien_expr.shape[1]} samples, {orien_expr.shape[0]} features")
    
    results = {}
    
    # ===== TCGA → ORIEN =====
    print("\n" + "="*60)
    print("EVALUATING: TCGA model → ORIEN data")
    print("="*60)
    
    tcga_results_dir = Path(args.tcga_model_dir)
    
    # Load TCGA model
    with open(tcga_results_dir / "best_params.json", 'r') as f:
        tcga_params = json.load(f)
    
    tcga_model = load_model_from_params(tcga_params, tcga_expr.shape[0], device)
    tcga_model.load_state_dict(torch.load(
        tcga_results_dir / f"final_model_tcga.pth",
        map_location=device
    ))
    
    print(f"Loaded TCGA model: {tcga_params.get('single_layer_size', tcga_params.get('first_layer_size'))}")
    
    # Test on ORIEN
    orien_dataset = SurvivalDataset(orien_expr, surv_orien)
    orien_loader = DataLoader(orien_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    tcga_to_orien_cindex = evaluate_model(tcga_model, orien_loader, device)
    print(f"✓ TCGA → ORIEN C-index: {tcga_to_orien_cindex:.4f}")
    
    results['tcga_to_orien'] = {
        'c_index': tcga_to_orien_cindex,
        'source': 'TCGA (339 samples)',
        'target': 'ORIEN (1112 samples)',
        'model_params': tcga_params
    }
    
    # ===== ORIEN → TCGA =====
    print("\n" + "="*60)
    print("EVALUATING: ORIEN model → TCGA data")
    print("="*60)
    
    orien_results_dir = Path(args.orien_model_dir)
    
    # Load ORIEN model
    with open(orien_results_dir / "best_params.json", 'r') as f:
        orien_params = json.load(f)
    
    orien_model = load_model_from_params(orien_params, orien_expr.shape[0], device)
    orien_model.load_state_dict(torch.load(
        orien_results_dir / f"final_model_orien.pth",
        map_location=device
    ))
    
    print(f"Loaded ORIEN model: {orien_params.get('pattern_2', orien_params.get('pattern_3'))}")
    
    # Test on TCGA
    tcga_dataset = SurvivalDataset(tcga_expr, surv_tcga)
    tcga_loader = DataLoader(tcga_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    orien_to_tcga_cindex = evaluate_model(orien_model, tcga_loader, device)
    print(f"✓ ORIEN → TCGA C-index: {orien_to_tcga_cindex:.4f}")
    
    results['orien_to_tcga'] = {
        'c_index': orien_to_tcga_cindex,
        'source': 'ORIEN (1112 samples)',
        'target': 'TCGA (339 samples)',
        'model_params': orien_params
    }
    
    # ===== SUMMARY =====
    print("\n" + "="*60)
    print("BIDIRECTIONAL EVALUATION SUMMARY")
    print("="*60)
    print(f"TCGA → ORIEN: {tcga_to_orien_cindex:.4f}")
    print(f"ORIEN → TCGA: {orien_to_tcga_cindex:.4f}")
    print("="*60)
    
    # Save results
    output_file = Path(args.output_dir) / "bidirectional_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    # Compare to Chapter 2
    print("\n" + "="*60)
    print("COMPARISON TO CHAPTER 2 (Penalized Cox)")
    print("="*60)
    print("Chapter 2 Results:")
    print("  TCGA → ORIEN: 0.72")
    print("  ORIEN → TCGA: 0.68")
    print("\nDeepSurv Results:")
    print(f"  TCGA → ORIEN: {tcga_to_orien_cindex:.4f}")
    print(f"  ORIEN → TCGA: {orien_to_tcga_cindex:.4f}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bidirectional evaluation of DeepSurv models')
    parser.add_argument('--tcga_model_dir', type=str, required=True,
                       help='Directory with TCGA model results')
    parser.add_argument('--orien_model_dir', type=str, required=True,
                       help='Directory with ORIEN model results')
    parser.add_argument('--output_dir', type=str, default='results/bidirectional_evaluation',
                       help='Output directory for results')
    
    args = parser.parse_args()
    main(args)