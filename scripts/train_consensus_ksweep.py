#!/usr/bin/env python3
"""
Script: train_consensus_ksweep.py
Purpose: Train and evaluate transfer learning with consensus genes from k-sweep
Status: ACTIVE (Chapter 4 - Performance-based k selection)
Author: Phuong
Created: 2024-11-15

This script:
1. Loops through selected k values from k-sweep results
2. Loads bidirectional consensus genes for each k
3. Trains transfer learning models using ONLY those genes
4. Evaluates cross-cohort performance (C-index)
5. Generates comparison table to select optimal k
6. Recommends k value that maximizes C-index

Methodology:
- Uses same training protocol as full 308-gene models
- Bidirectional validation (TCGA→ORIEN, ORIEN→TCGA)
- Single seed (42) for consistency with k-sweep
- Final selection based on C-index, not just stability

Reference:
- Guyon & Elisseeff (2003): Feature selection validated by model performance
- Your Chapter 3: Same k-sweep → retrain → evaluate methodology

Usage:
    python scripts/train_consensus_ksweep.py \
        --k_values 90 95 100 120 140 150 \
        --gene_lists_dir results/biomarker_ksweep_transfer/gene_lists \
        --output_dir results/consensus_ksweep_evaluation \
        --seed 42
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.elastic_deepsurv import ElasticDeepSurv
from src.data.dataset import SurvivalDataset
from src.utils.batch_samplers import StratifiedBatchSampler


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_consensus_genes(filepath: Path) -> List[str]:
    """Load consensus genes from text file."""
    with open(filepath, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]
    return genes


def load_data(
    cohort: str,
    genes: List[str]
) -> Dict:
    """
    Load and filter data to only include consensus genes.
    
    Follows same structure as transfer_learning_trainer.py:
    1. Load batch-corrected raw data (genes × samples)
    2. Filter to consensus genes only
    3. Load harmonized survival data
    4. Return expression in (genes × samples) format - SurvivalDataset handles it
    
    Args:
        cohort: 'tcga' or 'orien'
        genes: List of gene names to keep
        
    Returns:
        Dict with 'expression' (genes × samples) and 'survival' DataFrames
    """
    # Load batch-corrected expression data (same as Chapter 3)
    if cohort.lower() == 'tcga':
        expr_file = "data/raw/tcga_batch_corrected_2sv.csv"
        surv_file = "data/processed/surv_tcga_harmonized.csv"
    else:  # orien
        expr_file = "data/raw/orien_batch_corrected.csv"
        surv_file = "data/processed/surv_orien_harmonized.csv"
    
    # Load expression data (genes × samples)
    expression = pd.read_csv(expr_file, index_col=0)
    
    # Filter to consensus genes only
    available_genes = [g for g in genes if g in expression.index]
    expression = expression.loc[available_genes]
    
    # Load survival data
    survival = pd.read_csv(surv_file)
    if 'sampleID' in survival.columns:
        survival = survival.set_index('sampleID')
    
    # Align samples (only keep samples present in both expression and survival)
    common_samples = list(set(expression.columns) & set(survival.index))
    common_samples = sorted(common_samples)
    
    expression = expression[common_samples]
    survival = survival.loc[common_samples]
    
    return {
        'expression': expression,  # genes × samples format
        'survival': survival
    }


def create_model(n_features: int, device: str) -> ElasticDeepSurv:
    """
    Create ElasticDeepSurv model with standard hyperparameters.
    
    Uses same hyperparameters as Chapter 3/4 transfer learning.
    """
    model = ElasticDeepSurv(
        n_features=n_features,
        hidden_sizes=[128, 64],  # Standard architecture from Chapter 3
        dropout=0.3,
        l1_ratio=0.7,
        alpha=0.01
    )
    return model.to(device)


def train_model(
    model: ElasticDeepSurv,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int = 100,
    lr: float = 0.001,
    patience: int = 20,
    verbose: bool = True
) -> Dict:
    """
    Train model with early stopping.
    
    Returns:
        Dictionary with training history and best metrics
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    
    best_val_cindex = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_cindex': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            # Batch is a tuple: (x, y) where y is a dict/array with time and event
            x = batch[0].to(device)
            y = batch[1]  # This is the survival data
            
            # Extract time and event from y
            if isinstance(y, dict):
                time = y['time'].to(device)
                event = y['event'].to(device)
            else:
                # y is a 2D array/tensor: [:, 0] is time, [:, 1] is event
                time = y[:, 0].to(device)
                event = y[:, 1].to(device)
            
            optimizer.zero_grad()
            risk = model(x)
            loss = model.loss(risk, time, event)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        
        # Validation
        model.eval()
        val_cindex = evaluate_cindex(model, val_loader, device)
        
        history['train_loss'].append(avg_train_loss)
        history['val_cindex'].append(val_cindex)
        
        # Early stopping check
        if val_cindex > best_val_cindex:
            best_val_cindex = val_cindex
            best_epoch = epoch
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}: Loss={avg_train_loss:.4f}, "
                  f"Val C-index={val_cindex:.4f}, Best={best_val_cindex:.4f}")
        
        # Early stopping
        if patience_counter >= patience:
            if verbose:
                print(f"    Early stopping at epoch {epoch+1}")
            break
    
    # Restore best model
    model.load_state_dict(best_state)
    
    return {
        'best_val_cindex': best_val_cindex,
        'best_epoch': best_epoch,
        'history': history
    }


def evaluate_cindex(
    model: ElasticDeepSurv,
    data_loader: DataLoader,
    device: str
) -> float:
    """Evaluate concordance index on a dataset."""
    model.eval()
    
    all_risks = []
    all_times = []
    all_events = []
    
    with torch.no_grad():
        for batch in data_loader:
            x = batch[0].to(device)
            y = batch[1]
            
            # Extract time and event from y
            if isinstance(y, dict):
                time = y['time'].cpu().numpy()
                event = y['event'].cpu().numpy()
            else:
                # y is a 2D array/tensor
                time = y[:, 0].cpu().numpy() if isinstance(y, torch.Tensor) else y[:, 0]
                event = y[:, 1].cpu().numpy() if isinstance(y, torch.Tensor) else y[:, 1]
            
            risk = model(x)
            
            all_risks.extend(risk.cpu().numpy())
            all_times.extend(time)
            all_events.extend(event)
    
    # Compute C-index
    all_risks = np.array(all_risks)
    all_times = np.array(all_times)
    all_events = np.array(all_events)
    
    # Simple C-index calculation
    concordant = 0
    permissible = 0
    
    for i in range(len(all_risks)):
        if all_events[i] == 0:
            continue
        
        for j in range(len(all_risks)):
            if all_times[j] > all_times[i]:
                permissible += 1
                if all_risks[i] > all_risks[j]:
                    concordant += 1
    
    cindex = concordant / permissible if permissible > 0 else 0.5
    return cindex


def transfer_learning_pipeline(
    source_cohort: str,
    target_cohort: str,
    genes: List[str],
    seed: int,
    device: str,
    output_dir: Path,
    verbose: bool = True
) -> Dict:
    """
    Complete transfer learning pipeline for one direction.
    
    Args:
        source_cohort: 'tcga' or 'orien' (for pre-training)
        target_cohort: 'orien' or 'tcga' (for fine-tuning)
        genes: List of consensus genes to use
        seed: Random seed
        device: 'cuda' or 'cpu'
        output_dir: Where to save results
        verbose: Print progress
        
    Returns:
        Dictionary with results
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if verbose:
        print(f"\n  Transfer Learning: {source_cohort.upper()}→{target_cohort.upper()}")
        print(f"    Using {len(genes)} consensus genes")
    
    # ========================================
    # Load data
    # ========================================
    
    if verbose:
        print(f"    Loading data...")
    
    source_data = load_data(source_cohort, genes)
    target_data = load_data(target_cohort, genes)
    
    n_genes = len(source_data['expression'])  # Number of rows (genes)
    n_source_samples = len(source_data['expression'].columns)  # Number of columns (samples)
    n_target_samples = len(target_data['expression'].columns)
    
    if verbose:
        print(f"    Source ({source_cohort}): {n_source_samples} samples, {n_genes} genes")
        print(f"    Target ({target_cohort}): {n_target_samples} samples, {n_genes} genes")
    
    # ========================================
    # Create datasets
    # ========================================
    
    # Split target into train/val (80/20) based on sample IDs
    from sklearn.model_selection import train_test_split
    
    sample_ids = list(target_data['expression'].columns)
    train_sample_ids, val_sample_ids = train_test_split(
        sample_ids,
        test_size=0.2,
        random_state=seed,
        stratify=target_data['survival']['event'].values
    )
    
    # Create train/val splits
    target_train_expr = target_data['expression'][train_sample_ids]
    target_train_surv = target_data['survival'].loc[train_sample_ids]
    target_val_expr = target_data['expression'][val_sample_ids]
    target_val_surv = target_data['survival'].loc[val_sample_ids]
    
    # Create datasets (SurvivalDataset expects genes × samples format)
    source_dataset = SurvivalDataset(
        source_data['expression'],
        source_data['survival']
    )
    target_train_dataset = SurvivalDataset(target_train_expr, target_train_surv)
    target_val_dataset = SurvivalDataset(target_val_expr, target_val_surv)
    
    # Create data loaders
    source_sampler = StratifiedBatchSampler(
        events=source_dataset.y_event,
        batch_size=32,
        shuffle=True
    )
    
    source_loader = DataLoader(
        source_dataset,
        batch_sampler=source_sampler
    )
    
    target_train_sampler = StratifiedBatchSampler(
        events=target_train_dataset.y_event,
        batch_size=32,
        shuffle=True
    )
    
    target_train_loader = DataLoader(
        target_train_dataset,
        batch_sampler=target_train_sampler
    )
    
    target_val_loader = DataLoader(target_val_dataset, batch_size=32, shuffle=False)
    
    # ========================================
    # Phase 1: Pre-train on source
    # ========================================
    
    if verbose:
        print(f"\n    Phase 1: Pre-training on {source_cohort.upper()}...")
    
    model = create_model(n_genes, device)
    
    pretrain_results = train_model(
        model=model,
        train_loader=source_loader,
        val_loader=source_loader,  # Use source as validation during pre-training
        device=device,
        epochs=100,
        lr=0.001,
        patience=20,
        verbose=verbose
    )
    
    if verbose:
        print(f"    Pre-training complete: C-index = {pretrain_results['best_val_cindex']:.4f}")
    
    # ========================================
    # Phase 2: Fine-tune on target
    # ========================================
    
    if verbose:
        print(f"\n    Phase 2: Fine-tuning on {target_cohort.upper()}...")
    
    finetune_results = train_model(
        model=model,
        train_loader=target_train_loader,
        val_loader=target_val_loader,
        device=device,
        epochs=50,
        lr=0.0001,  # Lower learning rate for fine-tuning
        patience=15,
        verbose=verbose
    )
    
    if verbose:
        print(f"    Fine-tuning complete: C-index = {finetune_results['best_val_cindex']:.4f}")
    
    # ========================================
    # Evaluate on full target cohort
    # ========================================
    
    target_full_dataset = SurvivalDataset(target_data['expression'], target_data['survival'])
    target_full_loader = DataLoader(target_full_dataset, batch_size=32, shuffle=False)
    
    final_cindex = evaluate_cindex(model, target_full_loader, device)
    
    if verbose:
        print(f"    Final evaluation on full target: C-index = {final_cindex:.4f}")
    
    # ========================================
    # Save model
    # ========================================
    
    model_path = output_dir / f'{source_cohort}_to_{target_cohort}_seed{seed}.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'n_features': n_genes,
        'genes': genes,
        'pretrain_cindex': pretrain_results['best_val_cindex'],
        'finetune_cindex': finetune_results['best_val_cindex'],
        'final_cindex': final_cindex,
        'seed': seed
    }, model_path)
    
    return {
        'source': source_cohort,
        'target': target_cohort,
        'n_genes': n_genes,
        'pretrain_cindex': pretrain_results['best_val_cindex'],
        'finetune_cindex': finetune_results['best_val_cindex'],
        'final_cindex': final_cindex,
        'pretrain_epochs': pretrain_results['best_epoch'],
        'finetune_epochs': finetune_results['best_epoch']
    }


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def run_consensus_ksweep(
    k_values: List[int],
    gene_lists_dir: str,
    output_dir: str,
    seed: int = 42,
    device: str = None
):
    """
    Train and evaluate transfer learning for multiple k values.
    
    Args:
        k_values: List of k values to test
        gene_lists_dir: Directory containing consensus gene lists
        output_dir: Output directory for results
        seed: Random seed
        device: 'cuda' or 'cpu' (auto-detect if None)
    """
    # Setup
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    gene_lists_path = Path(gene_lists_dir)
    
    print(f"{'='*80}")
    print("CONSENSUS K-SWEEP EVALUATION")
    print(f"{'='*80}\n")
    
    print(f"Configuration:")
    print(f"  K values to test: {k_values}")
    print(f"  Gene lists directory: {gene_lists_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Random seed: {seed}")
    print(f"  Device: {device}")
    print()
    
    # ========================================
    # Run transfer learning for each k
    # ========================================
    
    all_results = []
    
    for i, k in enumerate(k_values, 1):
        print(f"\n{'='*80}")
        print(f"K-VALUE {i}/{len(k_values)}: k = {k}")
        print(f"{'='*80}")
        
        # Load consensus genes for this k
        gene_file = gene_lists_path / f'k{k}_bidirectional.txt'
        
        if not gene_file.exists():
            print(f"  ⚠️  Gene list not found: {gene_file}")
            print(f"      Skipping k={k}")
            continue
        
        genes = load_consensus_genes(gene_file)
        print(f"  Loaded {len(genes)} consensus genes from k={k}")
        
        # Create k-specific output directory
        k_output_dir = output_path / f'k{k}'
        k_output_dir.mkdir(exist_ok=True)
        
        # Run both directions
        try:
            # TCGA → ORIEN
            tcga_to_orien = transfer_learning_pipeline(
                source_cohort='tcga',
                target_cohort='orien',
                genes=genes,
                seed=seed,
                device=device,
                output_dir=k_output_dir,
                verbose=True
            )
            
            # ORIEN → TCGA
            orien_to_tcga = transfer_learning_pipeline(
                source_cohort='orien',
                target_cohort='tcga',
                genes=genes,
                seed=seed,
                device=device,
                output_dir=k_output_dir,
                verbose=True
            )
            
            # Compute average
            avg_cindex = (tcga_to_orien['final_cindex'] + orien_to_tcga['final_cindex']) / 2
            
            # Store results
            result = {
                'k': k,
                'n_genes': len(genes),
                'tcga_to_orien_cindex': tcga_to_orien['final_cindex'],
                'orien_to_tcga_cindex': orien_to_tcga['final_cindex'],
                'average_cindex': avg_cindex,
                'tcga_to_orien_details': tcga_to_orien,
                'orien_to_tcga_details': orien_to_tcga
            }
            
            all_results.append(result)
            
            print(f"\n  ✓ Results for k={k}:")
            print(f"    TCGA→ORIEN: {tcga_to_orien['final_cindex']:.4f}")
            print(f"    ORIEN→TCGA: {orien_to_tcga['final_cindex']:.4f}")
            print(f"    Average: {avg_cindex:.4f}")
            
            # Save k-specific results
            with open(k_output_dir / 'results.json', 'w') as f:
                json.dump(result, f, indent=2)
        
        except Exception as e:
            print(f"\n  ❌ Error processing k={k}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # ========================================
    # Generate comparison table
    # ========================================
    
    print(f"\n{'='*80}")
    print("GENERATING COMPARISON TABLE")
    print(f"{'='*80}\n")
    
    if not all_results:
        print("❌ No results to compare!")
        return
    
    # Create summary DataFrame
    summary_data = []
    for r in all_results:
        summary_data.append({
            'k': r['k'],
            'n_genes': r['n_genes'],
            'TCGA_to_ORIEN': f"{r['tcga_to_orien_cindex']:.4f}",
            'ORIEN_to_TCGA': f"{r['orien_to_tcga_cindex']:.4f}",
            'Average': f"{r['average_cindex']:.4f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('k')
    
    print(summary_df.to_string(index=False))
    
    # Find best k
    best_result = max(all_results, key=lambda r: r['average_cindex'])
    
    print(f"\n{'='*80}")
    print("RECOMMENDATION")
    print(f"{'='*80}\n")
    
    print(f"🎯 OPTIMAL k = {best_result['k']}")
    print(f"   - Number of genes: {best_result['n_genes']}")
    print(f"   - TCGA→ORIEN C-index: {best_result['tcga_to_orien_cindex']:.4f}")
    print(f"   - ORIEN→TCGA C-index: {best_result['orien_to_tcga_cindex']:.4f}")
    print(f"   - Average C-index: {best_result['average_cindex']:.4f}")
    print(f"   - Rationale: Highest average C-index across both directions")
    
    # ========================================
    # Save final results
    # ========================================
    
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}\n")
    
    # Save summary table
    summary_df.to_csv(output_path / 'consensus_ksweep_summary.csv', index=False)
    print(f"✓ Summary table: consensus_ksweep_summary.csv")
    
    # Save full results
    with open(output_path / 'consensus_ksweep_full_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✓ Full results: consensus_ksweep_full_results.json")
    
    # Save recommendation
    with open(output_path / 'RECOMMENDATION.txt', 'w') as f:
        f.write(f"OPTIMAL K-VALUE SELECTION\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Recommended k: {best_result['k']}\n")
        f.write(f"Number of genes: {best_result['n_genes']}\n")
        f.write(f"TCGA→ORIEN C-index: {best_result['tcga_to_orien_cindex']:.4f}\n")
        f.write(f"ORIEN→TCGA C-index: {best_result['orien_to_tcga_cindex']:.4f}\n")
        f.write(f"Average C-index: {best_result['average_cindex']:.4f}\n\n")
        f.write(f"This k value maximizes predictive performance while maintaining\n")
        f.write(f"reasonable gene count for clinical translation.\n")
    
    print(f"✓ Recommendation: RECOMMENDATION.txt")
    
    print(f"\n{'='*80}")
    print("CONSENSUS K-SWEEP EVALUATION COMPLETE")
    print(f"{'='*80}\n")
    
    return summary_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate transfer learning with consensus genes from k-sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Test representative k values (recommended)
  python scripts/train_consensus_ksweep.py \
      --k_values 90 95 100 120 140 \
      --gene_lists_dir results/biomarker_ksweep_transfer/gene_lists \
      --output_dir results/consensus_ksweep_evaluation
  
  # Test all k values (comprehensive but slower)
  python scripts/train_consensus_ksweep.py \
      --k_values 60 70 80 90 95 100 110 120 130 140 150 \
      --gene_lists_dir results/biomarker_ksweep_transfer/gene_lists \
      --output_dir results/consensus_ksweep_evaluation_full
        """
    )
    
    parser.add_argument('--k_values', type=int, nargs='+',
                       default=[90, 95, 100, 120, 140],
                       help='K values to test (default: 90 95 100 120 140)')
    parser.add_argument('--gene_lists_dir', type=str,
                       default='results/biomarker_ksweep_transfer/gene_lists',
                       help='Directory containing consensus gene lists')
    parser.add_argument('--output_dir', type=str,
                       default='results/consensus_ksweep_evaluation',
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu, default: auto-detect)')
    
    args = parser.parse_args()
    
    # Run evaluation
    summary_df = run_consensus_ksweep(
        k_values=args.k_values,
        gene_lists_dir=args.gene_lists_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device
    )
    
    print("\n✅ Consensus k-sweep evaluation completed successfully!")
    print(f"📁 Results saved in: {args.output_dir}/")
