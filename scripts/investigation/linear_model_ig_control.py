"""
Linear Model Control Experiment for IG Sign Flipping Diagnostic
===============================================================

Purpose: Determine whether IG sign heterogeneity (~50% consistency) is driven by
neural network nonlinearity or by data structure (correlations, subpopulations).

Method:
  1. Train a LINEAR ElasticDeepSurv (hidden_sizes=[], i.e., input -> linear -> output)
     This is mathematically equivalent to Cox PH with elastic net penalty.
  2. Compute Integrated Gradients on the linear model using same pipeline.
  3. Compare per-patient sign consistency between linear and nonlinear models.

Interpretation:
  - If linear model also shows ~50% sign consistency:
    → Heterogeneity is in the DATA (correlations, patient subpopulations)
    → Not an artifact of nonlinearity
  - If linear model shows ~80-90% sign consistency:
    → Nonlinearity is driving sign flipping
    → Claims about "patient-level heterogeneity" need major revision

References:
  - Sundararajan et al. (2017) "Axiomatic Attribution for Deep Networks" - ICML
  - Sundararajan & Najmi (2020) "The many Shapley values for model explanation" - ICML
  - Katzman et al. (2018) "DeepSurv" - BMC Medical Research Methodology

Usage:
  python linear_model_ig_control.py --seed 42
  python linear_model_ig_control.py --seed 42 --n_steps 300

  To run all 5 seeds:
  for seed in 42 123 456 789 1011; do
      python linear_model_ig_control.py --seed $seed --n_steps 300
  done

Author: Phuong Nguyen
"""

import sys
import os
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
import sys
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from captum.attr import IntegratedGradients

from src.models.elastic_deepsurv import ElasticDeepSurv, ElasticDeepSurvTrainer
from src.data.dataset import SurvivalDataset
from src.data.batch_samplers import StratifiedBatchSampler

# =============================================================================
# Configuration (same paths as compute_shap_ig_importance.py)
# =============================================================================

DATA_DIR = Path("data")
RESULTS_BASE = Path("results_v2")

# Reuse existing paths
CONSENSUS_GENES_FILE = DATA_DIR / "raw" / "consensus_genes_308.txt"
TCGA_EXPR_FILE = DATA_DIR / "raw" / "tcga_batch_corrected_2sv.csv"
ORIEN_EXPR_FILE = DATA_DIR / "raw" / "orien_batch_corrected.csv"
TCGA_SURV_FILE = DATA_DIR / "processed" / "surv_tcga_harmonized.csv"
ORIEN_SURV_FILE = DATA_DIR / "processed" / "surv_orien_harmonized.csv"

# Nonlinear model paths (for loading existing IG results)
NONLINEAR_IG_DIR = RESULTS_BASE / "06_importance_methods"

# Output
OUTPUT_DIR = RESULTS_BASE / "07_linear_control_experiment"


# =============================================================================
# Setup
# =============================================================================

def setup_logging(output_dir: Path, seed: int) -> logging.Logger:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"linear_control_seed{seed}_{timestamp}.log"
    
    logger = logging.getLogger("linear_control")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def load_consensus_genes(filepath: Path) -> list:
    with open(filepath, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]
    return genes


# =============================================================================
# Data Loading (reuses same logic as compute_shap_ig_importance.py)
# =============================================================================

def load_data(expr_file, surv_file, consensus_genes, logger):
    """Load expression and survival data, return standardized tensor + metadata."""
    
    logger.info(f"Loading expression data from {expr_file.name}")
    expr_df = pd.read_csv(expr_file, index_col=0)
    surv_df = pd.read_csv(surv_file, index_col=0)
    
    # Filter to consensus genes
    available_genes = [g for g in consensus_genes if g in expr_df.index]
    logger.info(f"  {len(available_genes)}/{len(consensus_genes)} consensus genes found")
    expr_df = expr_df.loc[available_genes]
    
    # Align samples
    common_samples = sorted(set(expr_df.columns) & set(surv_df.index))
    expr_df = expr_df[common_samples]
    surv_df = surv_df.loc[common_samples]
    
    logger.info(f"  {len(common_samples)} samples, {len(available_genes)} genes")
    logger.info(f"  Events: {int(surv_df['event'].sum())}/{len(common_samples)} "
                f"({100*surv_df['event'].mean():.1f}%)")
    
    # Z-score standardization (same as original pipeline)
    expr_mean = expr_df.mean(axis=1)
    expr_std = expr_df.std(axis=1).replace(0, 1)
    expr_standardized = expr_df.subtract(expr_mean, axis=0).divide(expr_std, axis=0)
    
    # Tensor (samples x genes)
    expr_tensor = torch.tensor(expr_standardized.values.T, dtype=torch.float32)
    
    return expr_df, expr_standardized, surv_df, expr_tensor, common_samples, available_genes


# =============================================================================
# Train Linear Model
# =============================================================================

def train_linear_model(expr_df, surv_df, cohort, seed, logger, device,
                       n_epochs=200, early_stopping_patience=30):
    """
    Train a LINEAR ElasticDeepSurv (hidden_sizes=[]).
    
    Uses same training procedure as nonlinear model:
    - Adam optimizer
    - Early stopping on validation C-index
    - Elastic net regularization
    - Stratified batch sampling
    
    Hyperparameters: Use the same alpha, l1_ratio, learning_rate from the
    nonlinear model's best_params.json for fair comparison.
    """
    
    # Load hyperparameters from the nonlinear model (for fair comparison)
    if cohort == "tcga":
        params_file = RESULTS_BASE / "01_hyperparameter_tuning" / "tcga_308genes" / "best_params.json"
    else:
        params_file = RESULTS_BASE / "01_hyperparameter_tuning" / "orien_308genes" / "best_params.json"
    
    with open(params_file, 'r') as f:
        nl_params = json.load(f)
    
    logger.info(f"\nNonlinear model params (for reference):")
    logger.info(f"  alpha: {nl_params['alpha']}")
    logger.info(f"  l1_ratio: {nl_params['l1_ratio']}")
    logger.info(f"  learning_rate: {nl_params['learning_rate']}")
    logger.info(f"  batch_size: {nl_params['batch_size']}")
    
    n_features = len(expr_df.index)
    batch_size = nl_params['batch_size']
    
    # Create linear model: input -> output (NO hidden layers)
    model = ElasticDeepSurv(
        n_features=n_features,
        hidden_sizes=[],          # <-- THIS IS THE KEY DIFFERENCE
        dropout=0.0,              # No dropout needed for linear model
        activation='relu',        # Irrelevant (no hidden layers)
        batch_norm=False,         # No batch norm for linear model
        weight_init='xavier_normal',
        l1_ratio=nl_params['l1_ratio'],
        alpha=nl_params['alpha']
    )
    
    logger.info(f"\nLinear model architecture:")
    logger.info(f"  {n_features} -> 1 (NO hidden layers)")
    logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"  This is equivalent to Cox PH with elastic net penalty")
    
    # Create dataset
    dataset = SurvivalDataset(expr_df, surv_df)
    
    # Train/validation split (80/20 stratified by event)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    events = dataset.y_event
    n_samples = len(dataset)
    indices = np.arange(n_samples)
    
    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, stratify=events, random_state=seed
    )
    
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    
    # Create data loaders with stratified batch sampling
    train_events = events[train_idx]
    train_sampler = StratifiedBatchSampler(
        events=train_events,
        batch_size=batch_size,
        min_events_per_batch=2,
        shuffle=True
    )
    
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Train
    trainer = ElasticDeepSurvTrainer(
        model=model,
        learning_rate=nl_params['learning_rate'],
        device=device
    )
    
    logger.info(f"\nTraining linear model (max {n_epochs} epochs, patience {early_stopping_patience})...")
    
    history = trainer.fit(
        train_loader=train_loader,
        valid_loader=val_loader,
        n_epochs=n_epochs,
        early_stopping_patience=early_stopping_patience,
        verbose=False
    )
    
    best_cindex = max(history['valid_c_index'])
    best_epoch = history.get('best_epoch', len(history['train_loss']))
    
    logger.info(f"  Training complete: best C-index = {best_cindex:.4f} at epoch {best_epoch}")
    
    model.eval()
    return model, history


# =============================================================================
# Compute IG on Linear Model
# =============================================================================

def compute_ig_linear(model, expr_tensor, gene_names, sample_ids, device, logger,
                      n_steps=300):
    """Compute Integrated Gradients on the linear model (same method as nonlinear)."""
    
    logger.info(f"\nComputing IG for linear model (n_steps={n_steps})...")
    
    model = model.to(device)
    model.eval()
    expr_tensor = expr_tensor.to(device)
    
    # Same baseline as nonlinear: cohort mean
    baseline = expr_tensor.mean(dim=0, keepdim=True)
    baseline_expanded = baseline.expand(expr_tensor.shape[0], -1)
    
    logger.info(f"  Baseline L2 norm: {torch.norm(baseline):.6f}")
    
    ig = IntegratedGradients(model)
    
    # Compute in batches
    batch_size = 100
    n_samples = expr_tensor.shape[0]
    all_attributions = []
    
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch_inputs = expr_tensor[i:end_idx]
        batch_baselines = baseline_expanded[i:end_idx]
        
        attributions = ig.attribute(
            batch_inputs,
            baselines=batch_baselines,
            n_steps=n_steps,
            return_convergence_delta=False
        )
        all_attributions.append(attributions.cpu())
        
        if (i // batch_size + 1) % 5 == 0 or end_idx == n_samples:
            logger.info(f"    Processed {end_idx}/{n_samples} samples")
    
    attributions = torch.cat(all_attributions, dim=0).numpy()
    
    logger.info(f"  Attributions shape: {attributions.shape}")
    
    # Convergence check: for linear model, IG should be exact
    # IG(x) = weight * (x - baseline) for linear models
    # Verify this holds
    with torch.no_grad():
        outputs = model(expr_tensor).cpu().numpy().flatten()
        baseline_output = model(baseline.to(device)).cpu().numpy().flatten()[0]
    
    expected_diff = outputs - baseline_output
    actual_sum = attributions.sum(axis=1)
    relative_error = np.abs(expected_diff - actual_sum) / (np.abs(expected_diff) + 1e-7)
    
    logger.info(f"\n  Convergence check (should be near-perfect for linear model):")
    logger.info(f"    Mean relative error: {relative_error.mean():.6f}")
    logger.info(f"    Max relative error:  {relative_error.max():.6f}")
    logger.info(f"    % within 1% tolerance: {100*(relative_error < 0.01).mean():.1f}%")
    
    return attributions


# =============================================================================
# Sign Consistency Analysis
# =============================================================================

def compute_sign_consistency(attributions, gene_names, logger, label=""):
    """Compute per-gene sign consistency (same metric as your R code)."""
    
    n_samples, n_genes = attributions.shape
    results = []
    
    for j in range(n_genes):
        vals = attributions[:, j]
        mean_ig = np.mean(vals)
        mean_sign = np.sign(mean_ig)
        
        if mean_sign == 0:
            consistency = 0.5
        else:
            consistency = np.sum(np.sign(vals) == mean_sign) / n_samples
        
        results.append({
            'gene': gene_names[j],
            'mean_ig': mean_ig,
            'mean_abs_ig': np.mean(np.abs(vals)),
            'sign_consistency': consistency
        })
    
    df = pd.DataFrame(results)
    
    logger.info(f"\n  Sign Consistency ({label}):")
    logger.info(f"    Mean:   {df['sign_consistency'].mean():.4f}")
    logger.info(f"    Median: {df['sign_consistency'].median():.4f}")
    logger.info(f"    Range:  [{df['sign_consistency'].min():.4f}, {df['sign_consistency'].max():.4f}]")
    logger.info(f"    Genes with consistency > 0.6: {(df['sign_consistency'] > 0.6).sum()}/{n_genes}")
    logger.info(f"    Genes with consistency > 0.7: {(df['sign_consistency'] > 0.7).sum()}/{n_genes}")
    
    return df


# =============================================================================
# Load Existing Nonlinear IG Results
# =============================================================================

def load_nonlinear_attributions(seed, cohort, consensus_genes_58, logger):
    """Load per-sample IG attributions from the existing nonlinear model."""
    
    attr_file = (NONLINEAR_IG_DIR / f"seed_{seed}" / 
                 "per_sample_attributions" / f"{cohort}_attributions_per_sample.csv")
    
    if not attr_file.exists():
        logger.warning(f"  Nonlinear attributions not found: {attr_file}")
        return None
    
    df = pd.read_csv(attr_file, index_col=0)
    logger.info(f"  Loaded nonlinear IG: {df.shape[0]} samples x {df.shape[1]} genes")
    
    # Filter to 58 consensus genes (if available)
    available = [g for g in consensus_genes_58 if g in df.columns]
    if len(available) < len(consensus_genes_58):
        logger.info(f"  Filtered to {len(available)}/{len(consensus_genes_58)} consensus genes")
    
    return df[available].values, available


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Linear model control experiment for IG sign flipping diagnostic'
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--n_steps', type=int, default=300,
                        help='IG integration steps (default: 300, same as nonlinear)')
    parser.add_argument('--consensus_58_file', type=str, default=None,
                        help='Path to file listing the 58 sign-consistent genes '
                             '(if not provided, will use all 308)')
    
    args = parser.parse_args()
    
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    seed = args.seed
    seed_dir = OUTPUT_DIR / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(OUTPUT_DIR, seed)
    
    logger.info("=" * 70)
    logger.info("LINEAR MODEL CONTROL EXPERIMENT")
    logger.info("Testing: Is IG sign flipping from nonlinearity or data structure?")
    logger.info("=" * 70)
    logger.info(f"Seed: {seed}")
    logger.info(f"Device: {device}")
    logger.info(f"IG n_steps: {args.n_steps}")
    
    # Load consensus genes
    consensus_genes = load_consensus_genes(CONSENSUS_GENES_FILE)
    logger.info(f"Loaded {len(consensus_genes)} consensus genes (308)")
    
    # Load 58 sign-consistent genes if provided
    if args.consensus_58_file and Path(args.consensus_58_file).exists():
        consensus_58 = load_consensus_genes(Path(args.consensus_58_file))
        logger.info(f"Loaded {len(consensus_58)} sign-consistent genes for comparison")
    else:
        consensus_58 = None
        logger.info("No 58-gene file provided; will compute consistency on all 308 genes")
    
    # =================================================================
    # Process each cohort
    # =================================================================
    
    all_comparisons = []
    
    for cohort, expr_file, surv_file in [
        ("tcga", TCGA_EXPR_FILE, TCGA_SURV_FILE),
        ("orien", ORIEN_EXPR_FILE, ORIEN_SURV_FILE)
    ]:
        logger.info(f"\n{'='*70}")
        logger.info(f"COHORT: {cohort.upper()}")
        logger.info(f"{'='*70}")
        
        # Load data
        expr_df, expr_std, surv_df, expr_tensor, sample_ids, gene_names = load_data(
            expr_file, surv_file, consensus_genes, logger
        )
        
        # Train linear model
        linear_model, history = train_linear_model(
            expr_df, surv_df, cohort, seed, logger, device
        )
        
        # Save linear model
        model_path = seed_dir / f"{cohort}_linear_model.pth"
        torch.save(linear_model.state_dict(), model_path)
        logger.info(f"  Saved linear model: {model_path}")
        
        # Compute IG on linear model
        linear_attr = compute_ig_linear(
            linear_model, expr_tensor, gene_names, sample_ids,
            device, logger, n_steps=args.n_steps
        )
        
        # Save linear IG attributions
        linear_attr_df = pd.DataFrame(linear_attr, index=sample_ids, columns=gene_names)
        linear_attr_df.index.name = 'sample_id'
        linear_attr_df.to_csv(seed_dir / f"{cohort}_linear_attributions_per_sample.csv")
        
        # Sign consistency for LINEAR model (all 308 genes)
        linear_consistency = compute_sign_consistency(
            linear_attr, gene_names, logger, label=f"{cohort.upper()} LINEAR"
        )
        linear_consistency['model'] = 'linear'
        linear_consistency['cohort'] = cohort
        linear_consistency['seed'] = seed
        
        # Load nonlinear attributions for comparison
        genes_to_compare = consensus_58 if consensus_58 else gene_names
        
        nl_data = load_nonlinear_attributions(seed, cohort, genes_to_compare, logger)
        
        if nl_data is not None:
            nl_attr, nl_genes = nl_data
            
            # Sign consistency for NONLINEAR model (same genes)
            nl_consistency = compute_sign_consistency(
                nl_attr, nl_genes, logger, label=f"{cohort.upper()} NONLINEAR"
            )
            nl_consistency['model'] = 'nonlinear'
            nl_consistency['cohort'] = cohort
            nl_consistency['seed'] = seed
            
            # Also compute linear consistency for same gene subset
            gene_idx = [gene_names.index(g) for g in nl_genes if g in gene_names]
            linear_subset_attr = linear_attr[:, gene_idx]
            linear_subset_genes = [gene_names[i] for i in gene_idx]
            
            linear_subset_consistency = compute_sign_consistency(
                linear_subset_attr, linear_subset_genes, logger,
                label=f"{cohort.upper()} LINEAR (subset of {len(nl_genes)} genes)"
            )
            
            # Direct comparison
            merged = pd.merge(
                linear_subset_consistency[['gene', 'sign_consistency']].rename(
                    columns={'sign_consistency': 'linear_consistency'}),
                nl_consistency[['gene', 'sign_consistency']].rename(
                    columns={'sign_consistency': 'nonlinear_consistency'}),
                on='gene'
            )
            merged['difference'] = merged['linear_consistency'] - merged['nonlinear_consistency']
            merged['cohort'] = cohort
            merged['seed'] = seed
            
            all_comparisons.append(merged)
            
            logger.info(f"\n  === DIRECT COMPARISON ({cohort.upper()}) ===")
            logger.info(f"  Linear mean consistency:    {merged['linear_consistency'].mean():.4f}")
            logger.info(f"  Nonlinear mean consistency: {merged['nonlinear_consistency'].mean():.4f}")
            logger.info(f"  Difference (linear - NL):   {merged['difference'].mean():.4f}")
            
            if merged['difference'].mean() > 0.15:
                logger.info(f"  >>> LARGE DIFFERENCE: Nonlinearity is driving sign flipping")
            elif merged['difference'].mean() > 0.05:
                logger.info(f"  >>> MODERATE DIFFERENCE: Nonlinearity contributes partially")
            else:
                logger.info(f"  >>> SMALL DIFFERENCE: Sign flipping is in the data, not the architecture")
        
        # Save per-cohort results
        linear_consistency.to_csv(seed_dir / f"{cohort}_linear_sign_consistency.csv", index=False)
        
        if nl_data is not None:
            merged.to_csv(seed_dir / f"{cohort}_linear_vs_nonlinear_comparison.csv", index=False)
    
    # =================================================================
    # Summary across cohorts
    # =================================================================
    
    if all_comparisons:
        all_comp = pd.concat(all_comparisons, ignore_index=True)
        all_comp.to_csv(seed_dir / "all_comparisons.csv", index=False)
        
        logger.info(f"\n{'='*70}")
        logger.info("OVERALL SUMMARY")
        logger.info(f"{'='*70}")
        
        for cohort in ['tcga', 'orien']:
            sub = all_comp[all_comp['cohort'] == cohort]
            if len(sub) > 0:
                logger.info(f"\n{cohort.upper()}:")
                logger.info(f"  Linear mean:    {sub['linear_consistency'].mean():.4f}")
                logger.info(f"  Nonlinear mean: {sub['nonlinear_consistency'].mean():.4f}")
                logger.info(f"  Mean difference: {sub['difference'].mean():.4f}")
        
        overall_diff = all_comp['difference'].mean()
        
        logger.info(f"\n{'='*70}")
        logger.info("CONCLUSION")
        logger.info(f"{'='*70}")
        
        if overall_diff > 0.15:
            logger.info("RESULT: Nonlinearity is the PRIMARY driver of sign flipping.")
            logger.info("ACTION: IG sign-based heterogeneity claims need major revision.")
            logger.info("        Consider using Cox coefficients for directional interpretation.")
        elif overall_diff > 0.05:
            logger.info("RESULT: Nonlinearity PARTIALLY contributes to sign flipping.")
            logger.info("ACTION: Both architecture and data contribute. Report as mixed evidence.")
            logger.info("        Acknowledge in limitations. Focus on magnitude, not sign.")
        else:
            logger.info("RESULT: Sign flipping is in the DATA, not the architecture.")
            logger.info("ACTION: Heterogeneity likely reflects genuine biological variation")
            logger.info("        (gene correlations, patient subpopulations, interaction effects).")
            logger.info("        This supports IG-based heterogeneity claims, but gene correlation")
            logger.info("        effects (Diagnostic 2) should still be acknowledged.")
    
    logger.info(f"\nResults saved to: {seed_dir}")
    logger.info("DONE")


if __name__ == "__main__":
    main()
