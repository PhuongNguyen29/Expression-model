"""
Compute Feature Importance using Integrated Gradients and SHAP

This script computes gene importance scores using:
1. Integrated Gradients (Captum) - primary method
2. SHAP GradientExplainer - for validation

Replaces L2 norm importance which showed compression issues,
especially for ORIEN's 3-layer architecture.

References:
- Sundararajan et al. (2017) "Axiomatic Attribution for Deep Networks" - ICML
- Lundberg & Lee (2017) "A Unified Approach to Interpreting Model Predictions" - NeurIPS

Usage:
    python compute_shap_ig_importance.py --seed 42

Author: Phuong Nguyen
Date: December 2024
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Captum for Integrated Gradients
from captum.attr import IntegratedGradients

# SHAP for comparison
import shap

# Project imports
from src.models.elastic_deepsurv import ElasticDeepSurv


# =============================================================================
# Configuration
# =============================================================================

N_FEATURES = 308  # Number of consensus genes

# Paths
DATA_DIR = Path("data")
RESULTS_BASE = Path("results_v2")
HYPERPARAMS_DIR = RESULTS_BASE / "01_hyperparameter_tuning"
BIOMARKER_DIR = RESULTS_BASE / "02_biomarker_discovery"
OUTPUT_DIR = RESULTS_BASE / "06_importance_methods"

# Gene files
CONSENSUS_GENES_FILE = DATA_DIR / "raw" / "consensus_genes_308.txt"
COX_GENES_FILE = DATA_DIR / "raw" / "cox_consensus_genes_20.txt"

# Expression data
TCGA_EXPR_FILE = DATA_DIR / "raw" / "tcga_batch_corrected_2sv.csv"
ORIEN_EXPR_FILE = DATA_DIR / "raw" / "orien_batch_corrected.csv"

# Survival data (for sample alignment)
TCGA_SURV_FILE = DATA_DIR / "processed" / "surv_tcga_harmonized.csv"
ORIEN_SURV_FILE = DATA_DIR / "processed" / "surv_orien_harmonized.csv"


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(output_dir: Path, seed: int) -> logging.Logger:
    """Setup logging configuration."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"compute_importance_seed{seed}_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger("importance_computation")
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


# =============================================================================
# Data Loading
# =============================================================================

def load_consensus_genes(filepath: Path) -> list:
    """Load consensus gene list."""
    with open(filepath, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]
    return genes


def load_expression_data(
    expr_file: Path,
    surv_file: Path,
    consensus_genes: list,
    logger: logging.Logger
) -> tuple:
    """
    Load and preprocess expression data.
    
    Returns:
        expr_df: Expression DataFrame (genes x samples)
        expr_tensor: Standardized expression tensor (samples x genes)
        sample_ids: List of sample IDs
    """
    logger.info(f"Loading expression data from {expr_file.name}")
    
    # Load expression
    expr_df = pd.read_csv(expr_file, index_col=0)
    
    # Load survival for sample alignment
    surv_df = pd.read_csv(surv_file, index_col=0)
    
    # Filter to consensus genes
    available_genes = [g for g in consensus_genes if g in expr_df.index]
    if len(available_genes) != len(consensus_genes):
        missing = set(consensus_genes) - set(available_genes)
        logger.warning(f"Missing {len(missing)} genes: {list(missing)[:5]}...")
    
    expr_df = expr_df.loc[available_genes]
    
    # Align samples with survival data
    common_samples = list(set(expr_df.columns) & set(surv_df.index))
    common_samples = sorted(common_samples)
    expr_df = expr_df[common_samples]
    
    logger.info(f"  Shape: {expr_df.shape[0]} genes x {expr_df.shape[1]} samples")
    
    # Standardize (z-score per gene)
    expr_mean = expr_df.mean(axis=1)
    expr_std = expr_df.std(axis=1)
    expr_std = expr_std.replace(0, 1)  # Avoid division by zero
    
    expr_standardized = expr_df.subtract(expr_mean, axis=0).divide(expr_std, axis=0)
    
    # Convert to tensor (samples x genes for model input)
    expr_tensor = torch.tensor(
        expr_standardized.values.T,  # Transpose: samples x genes
        dtype=torch.float32
    )
    
    logger.info(f"  Tensor shape: {expr_tensor.shape}")
    logger.info(f"  Mean: {expr_tensor.mean():.4f}, Std: {expr_tensor.std():.4f}")
    
    return expr_df, expr_tensor, common_samples, available_genes


# =============================================================================
# Model Loading
# =============================================================================

def load_model(
    model_path: Path,
    params_path: Path,
    n_features: int,
    logger: logging.Logger
) -> ElasticDeepSurv:
    """
    Load trained model with correct architecture.
    
    Note: Due to a bug in ElasticDeepSurv, all models were trained with
    batch_norm=True regardless of config. We reconstruct accordingly.
    """
    logger.info(f"Loading model from {model_path.name}")
    
    # Load hyperparameters
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    # Parse architecture
    if 'architecture_3layer' in params:
        hidden_sizes = [int(x) for x in params['architecture_3layer'].split('-')]
    elif 'architecture_2layer' in params:
        hidden_sizes = [int(x) for x in params['architecture_2layer'].split('-')]
    elif 'layer1_size' in params:
        hidden_sizes = [params['layer1_size']]
    else:
        raise ValueError(f"Cannot parse architecture from {params}")
    
    logger.info(f"  Architecture: {n_features} -> {hidden_sizes} -> 1")
    
    # Create model
    # Note: batch_norm=True for all due to ElasticDeepSurv bug
    # The parent DeepSurv class uses default batch_norm=True
    model = ElasticDeepSurv(
        n_features=n_features,
        hidden_sizes=hidden_sizes,
        dropout=params.get('dropout', 0.3),
        activation=params.get('activation', 'relu'),
        batch_norm=True,  # Always True due to bug - matches saved checkpoints
        weight_init=params.get('weight_init', 'kaiming_uniform'),
        l1_ratio=params.get('l1_ratio', 0.5),
        alpha=params.get('alpha', 0.001)
    )
    
    # Load weights
    state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict)
    
    # Set to evaluation mode (critical for batch norm)
    model.eval()
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Parameters: {n_params:,}")
    
    return model


# =============================================================================
# Integrated Gradients Computation
# =============================================================================

def compute_integrated_gradients(
    model: nn.Module,
    expr_tensor: torch.Tensor,
    gene_names: list,
    device: str,
    logger: logging.Logger,
    n_steps: int = 50
) -> dict:
    """
    Compute Integrated Gradients attributions.
    
    Args:
        model: Trained ElasticDeepSurv model
        expr_tensor: Expression tensor (samples x genes)
        gene_names: List of gene names
        device: 'cuda' or 'cpu'
        logger: Logger instance
        n_steps: Number of integration steps (default 50)
        
    Returns:
        Dictionary with attribution results
    """
    logger.info("Computing Integrated Gradients...")
    logger.info(f"  Samples: {expr_tensor.shape[0]}")
    logger.info(f"  Genes: {expr_tensor.shape[1]}")
    logger.info(f"  Integration steps: {n_steps}")
    
    # Move model and data to device
    model = model.to(device)
    model.eval()
    expr_tensor = expr_tensor.to(device)
    
    # Compute mean baseline (average patient)
    baseline = expr_tensor.mean(dim=0, keepdim=True)  # Shape: (1, n_genes)
    logger.info(f"  Baseline shape: {baseline.shape}")
    logger.info(f"  Baseline mean: {baseline.mean():.4f}")
    
    # Expand baseline to match input size for batch processing
    baseline_expanded = baseline.expand(expr_tensor.shape[0], -1)
    
    # Create Integrated Gradients attributor
    ig = IntegratedGradients(model)
    
    # Compute attributions
    # Note: target=None for single-output regression
    logger.info("  Computing attributions (this may take a few minutes)...")
    
    # Process in batches to avoid memory issues
    batch_size = 100
    n_samples = expr_tensor.shape[0]
    all_attributions = []
    
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch_inputs = expr_tensor[i:end_idx]
        batch_baselines = baseline_expanded[i:end_idx]
        
        with torch.no_grad():
            # Need gradients for IG
            pass
        
        attributions = ig.attribute(
            batch_inputs,
            baselines=batch_baselines,
            n_steps=n_steps,
            return_convergence_delta=False
        )
        
        all_attributions.append(attributions.cpu())
        
        if (i // batch_size + 1) % 5 == 0:
            logger.info(f"    Processed {end_idx}/{n_samples} samples")
    
    # Concatenate all attributions
    attributions = torch.cat(all_attributions, dim=0)
    logger.info(f"  Attributions shape: {attributions.shape}")
    
    # Convert to numpy
    attributions_np = attributions.numpy()
    
    # Compute aggregated importance scores
    # 1. Mean absolute attribution (magnitude)
    importance_magnitude = np.abs(attributions_np).mean(axis=0)
    
    # 2. Signed mean attribution (direction)
    importance_signed = attributions_np.mean(axis=0)
    
    # 3. Standard deviation (variability across samples)
    importance_std = attributions_np.std(axis=0)
    
    # Log statistics
    logger.info(f"\n  Importance Statistics (Magnitude):")
    logger.info(f"    Range: [{importance_magnitude.min():.6f}, {importance_magnitude.max():.6f}]")
    logger.info(f"    Mean: {importance_magnitude.mean():.6f}")
    logger.info(f"    Std: {importance_magnitude.std():.6f}")
    logger.info(f"    CV: {importance_magnitude.std() / importance_magnitude.mean():.4f}")
    
    logger.info(f"\n  Importance Statistics (Signed):")
    logger.info(f"    Range: [{importance_signed.min():.6f}, {importance_signed.max():.6f}]")
    logger.info(f"    Mean: {importance_signed.mean():.6f}")
    logger.info(f"    Positive (hazardous): {(importance_signed > 0).sum()}")
    logger.info(f"    Negative (protective): {(importance_signed < 0).sum()}")
    
    # Create results dictionary
    results = {
        'attributions_per_sample': attributions_np,  # (n_samples, n_genes)
        'importance_magnitude': importance_magnitude,  # (n_genes,)
        'importance_signed': importance_signed,  # (n_genes,)
        'importance_std': importance_std,  # (n_genes,)
        'gene_names': gene_names,
        'n_samples': n_samples,
        'n_steps': n_steps,
        'baseline_type': 'mean'
    }
    
    return results


# =============================================================================
# SHAP GradientExplainer Computation
# =============================================================================

def compute_shap_gradientexplainer(
    model: nn.Module,
    expr_tensor: torch.Tensor,
    gene_names: list,
    device: str,
    logger: logging.Logger,
    n_background: int = 100
) -> dict:
    """
    Compute SHAP values using GradientExplainer.
    
    GradientExplainer is more robust to batch normalization than DeepExplainer.
    
    Args:
        model: Trained ElasticDeepSurv model
        expr_tensor: Expression tensor (samples x genes)
        gene_names: List of gene names
        device: 'cuda' or 'cpu'
        logger: Logger instance
        n_background: Number of background samples
        
    Returns:
        Dictionary with SHAP results
    """
    logger.info("Computing SHAP GradientExplainer...")
    logger.info(f"  Background samples: {n_background}")
    
    # Move model to device and set to eval
    model = model.to(device)
    model.eval()
    
    # Select background samples (random subset)
    n_samples = expr_tensor.shape[0]
    if n_background >= n_samples:
        background_idx = np.arange(n_samples)
    else:
        np.random.seed(42)  # Reproducibility
        background_idx = np.random.choice(n_samples, n_background, replace=False)
    
    background = expr_tensor[background_idx].to(device)
    logger.info(f"  Background shape: {background.shape}")
    
    # Create GradientExplainer
    try:
        explainer = shap.GradientExplainer(model, background)
        logger.info("  GradientExplainer created successfully")
    except Exception as e:
        logger.error(f"  Failed to create GradientExplainer: {e}")
        return None
    
    # Compute SHAP values
    logger.info("  Computing SHAP values (this may take several minutes)...")
    
    # Move all data to device
    expr_device = expr_tensor.to(device)
    
    try:
        shap_values = explainer.shap_values(expr_device)
        
        # Handle different return formats
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # Convert to numpy if tensor
        if torch.is_tensor(shap_values):
            shap_values = shap_values.cpu().numpy()
            
    except Exception as e:
        logger.error(f"  Failed to compute SHAP values: {e}")
        logger.info("  Trying with smaller batches...")
        
        # Try batch processing
        batch_size = 50
        all_shap = []
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch = expr_tensor[i:end_idx].to(device)
            
            try:
                batch_shap = explainer.shap_values(batch)
                if isinstance(batch_shap, list):
                    batch_shap = batch_shap[0]
                if torch.is_tensor(batch_shap):
                    batch_shap = batch_shap.cpu().numpy()
                all_shap.append(batch_shap)
                
                if (i // batch_size + 1) % 5 == 0:
                    logger.info(f"    Processed {end_idx}/{n_samples} samples")
                    
            except Exception as batch_e:
                logger.error(f"  Batch {i}-{end_idx} failed: {batch_e}")
                return None
        
        shap_values = np.concatenate(all_shap, axis=0)
    
    logger.info(f"  SHAP values shape: {shap_values.shape}")
    
    # Compute aggregated importance
    importance_magnitude = np.abs(shap_values).mean(axis=0)
    importance_signed = shap_values.mean(axis=0)
    importance_std = shap_values.std(axis=0)
    
    # Log statistics
    logger.info(f"\n  SHAP Importance Statistics (Magnitude):")
    logger.info(f"    Range: [{importance_magnitude.min():.6f}, {importance_magnitude.max():.6f}]")
    logger.info(f"    Mean: {importance_magnitude.mean():.6f}")
    logger.info(f"    Std: {importance_magnitude.std():.6f}")
    logger.info(f"    CV: {importance_magnitude.std() / importance_magnitude.mean():.4f}")
    
    results = {
        'shap_values_per_sample': shap_values,
        'importance_magnitude': importance_magnitude,
        'importance_signed': importance_signed,
        'importance_std': importance_std,
        'gene_names': gene_names,
        'n_samples': n_samples,
        'n_background': n_background
    }
    
    return results


# =============================================================================
# L2 Importance (for comparison)
# =============================================================================

def compute_l2_importance(
    model: nn.Module,
    gene_names: list,
    logger: logging.Logger
) -> dict:
    """
    Compute L2 norm importance from first layer weights.
    This is the current method - included for comparison.
    """
    logger.info("Computing L2 norm importance (current method)...")
    
    # Get first layer
    first_layer = model.network[0]
    if not isinstance(first_layer, nn.Linear):
        raise TypeError(f"First layer is {type(first_layer)}, not nn.Linear")
    
    weights = first_layer.weight.data.cpu().numpy()
    importance = np.linalg.norm(weights, axis=0)
    
    logger.info(f"  Weight shape: {weights.shape}")
    logger.info(f"  Importance range: [{importance.min():.6f}, {importance.max():.6f}]")
    logger.info(f"  CV: {importance.std() / importance.mean():.4f}")
    
    return {
        'importance': importance,
        'gene_names': gene_names
    }


# =============================================================================
# Analysis Functions
# =============================================================================

def compare_importance_methods(
    ig_results: dict,
    shap_results: dict,
    l2_results: dict,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Compare rankings between different importance methods.
    """
    logger.info("\n" + "="*60)
    logger.info("COMPARING IMPORTANCE METHODS")
    logger.info("="*60)
    
    gene_names = ig_results['gene_names']
    
    # Get importance scores
    ig_importance = ig_results['importance_magnitude']
    l2_importance = l2_results['importance']
    
    # SHAP may have failed
    if shap_results is not None:
        shap_importance = shap_results['importance_magnitude']
    else:
        shap_importance = np.full(len(gene_names), np.nan)
    
    # Create rankings (1 = most important)
    ig_ranks = stats.rankdata(-ig_importance)  # Negative for descending
    l2_ranks = stats.rankdata(-l2_importance)
    
    if shap_results is not None:
        shap_ranks = stats.rankdata(-shap_importance)
    else:
        shap_ranks = np.full(len(gene_names), np.nan)
    
    # Compute rank correlations
    logger.info("\nSpearman Rank Correlations:")
    
    # IG vs L2
    corr_ig_l2, p_ig_l2 = stats.spearmanr(ig_importance, l2_importance)
    logger.info(f"  IG vs L2:   rho = {corr_ig_l2:.4f}, p = {p_ig_l2:.2e}")
    
    # IG vs SHAP
    if shap_results is not None:
        corr_ig_shap, p_ig_shap = stats.spearmanr(ig_importance, shap_importance)
        logger.info(f"  IG vs SHAP: rho = {corr_ig_shap:.4f}, p = {p_ig_shap:.2e}")
        
        corr_shap_l2, p_shap_l2 = stats.spearmanr(shap_importance, l2_importance)
        logger.info(f"  SHAP vs L2: rho = {corr_shap_l2:.4f}, p = {p_shap_l2:.2e}")
    else:
        corr_ig_shap = np.nan
        corr_shap_l2 = np.nan
    fixed_names = []
    for g in gene_names:
        if hasattr(g, 'item'):
            fixed_names.append(g.item())
        elif isinstance(g, np.ndarray):
            fixed_names.append(str(g.flatten()[0]))
        else:
            fixed_names.append(str(g))
    gene_names = fixed_names

    # Top-50 agreement
    ig_top50 = set(gene_names[i] for i in [np.argsort(-ig_importance)[:50]])
    l2_top50 = set(gene_names[i] for i in [np.argsort(-l2_importance)[:50]])
    
    overlap_ig_l2 = len(ig_top50 & l2_top50)
    logger.info(f"\nTop-50 Gene Overlap:")
    logger.info(f"  IG vs L2: {overlap_ig_l2}/50 ({100*overlap_ig_l2/50:.1f}%)")
    
    if shap_results is not None:
        shap_top50 = set(gene_names[i] for i in [np.argsort(-shap_importance)[:50]])
        overlap_ig_shap = len(ig_top50 & shap_top50)
        overlap_shap_l2 = len(shap_top50 & l2_top50)
        logger.info(f"  IG vs SHAP: {overlap_ig_shap}/50 ({100*overlap_ig_shap/50:.1f}%)")
        logger.info(f"  SHAP vs L2: {overlap_shap_l2}/50 ({100*overlap_shap_l2/50:.1f}%)")
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'gene': gene_names,
        'ig_importance': ig_importance,
        'ig_rank': ig_ranks,
        'ig_signed': ig_results['importance_signed'],
        'shap_importance': shap_importance,
        'shap_rank': shap_ranks,
        'l2_importance': l2_importance,
        'l2_rank': l2_ranks
    })
    
    comparison_df = comparison_df.sort_values('ig_importance', ascending=False)
    
    return comparison_df


def check_cox_gene_overlap(
    comparison_df: pd.DataFrame,
    cox_genes_file: Path,
    logger: logging.Logger
) -> dict:
    """
    Check how many Cox consensus genes are captured at different k values.
    """
    logger.info("\n" + "="*60)
    logger.info("COX GENE OVERLAP ANALYSIS")
    logger.info("="*60)
    
    # Load Cox genes
    cox_genes = load_consensus_genes(cox_genes_file)
    logger.info(f"Cox consensus genes: {len(cox_genes)}")
    
    # Check overlap at different k values
    k_values = [20, 30, 50, 75, 100, 150]
    
    results = {'k': [], 'ig_overlap': [], 'l2_overlap': [], 'shap_overlap': []}
    
    logger.info("\nOverlap with Cox genes at different k:")
    logger.info("-" * 50)
    
    for k in k_values:
        # Top-k genes by each method
        ig_topk = set(comparison_df.nsmallest(k, 'ig_rank')['gene'])
        l2_topk = set(comparison_df.nsmallest(k, 'l2_rank')['gene'])
        
        ig_overlap = len(ig_topk & set(cox_genes))
        l2_overlap = len(l2_topk & set(cox_genes))
        
        results['k'].append(k)
        results['ig_overlap'].append(ig_overlap)
        results['l2_overlap'].append(l2_overlap)
        
        # SHAP if available
        if not comparison_df['shap_rank'].isna().all():
            shap_topk = set(comparison_df.nsmallest(k, 'shap_rank')['gene'])
            shap_overlap = len(shap_topk & set(cox_genes))
            results['shap_overlap'].append(shap_overlap)
        else:
            results['shap_overlap'].append(np.nan)
        
        logger.info(f"  k={k:3d}: IG={ig_overlap:2d}/20, L2={l2_overlap:2d}/20, "
                   f"SHAP={results['shap_overlap'][-1] if not np.isnan(results['shap_overlap'][-1]) else 'N/A'}")
    
    # Which Cox genes are captured by IG but not L2?
    ig_top100 = set(comparison_df.nsmallest(100, 'ig_rank')['gene'])
    l2_top100 = set(comparison_df.nsmallest(100, 'l2_rank')['gene'])
    
    cox_in_ig_not_l2 = set(cox_genes) & ig_top100 - l2_top100
    cox_in_l2_not_ig = set(cox_genes) & l2_top100 - ig_top100
    
    logger.info(f"\nCox genes in IG top-100 but not L2 top-100: {cox_in_ig_not_l2}")
    logger.info(f"Cox genes in L2 top-100 but not IG top-100: {cox_in_l2_not_ig}")
    
    return results


# =============================================================================
# Save Results
# =============================================================================

def save_results(
    cohort: str,
    ig_results: dict,
    shap_results: dict,
    l2_results: dict,
    comparison_df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger
):
    """Save all results to files."""
    logger.info(f"\nSaving results to {output_dir}")
    
    # Save IG importance
    ig_df = pd.DataFrame({
        'gene': ig_results['gene_names'],
        'importance_magnitude': ig_results['importance_magnitude'],
        'importance_signed': ig_results['importance_signed'],
        'importance_std': ig_results['importance_std']
    }).sort_values('importance_magnitude', ascending=False)
    
    ig_df.to_csv(output_dir / f'{cohort}_ig_importance.csv', index=False)
    logger.info(f"  Saved: {cohort}_ig_importance.csv")
    
    # Save SHAP importance (if available)
    if shap_results is not None:
        shap_df = pd.DataFrame({
            'gene': shap_results['gene_names'],
            'importance_magnitude': shap_results['importance_magnitude'],
            'importance_signed': shap_results['importance_signed'],
            'importance_std': shap_results['importance_std']
        }).sort_values('importance_magnitude', ascending=False)
        
        shap_df.to_csv(output_dir / f'{cohort}_shap_importance.csv', index=False)
        logger.info(f"  Saved: {cohort}_shap_importance.csv")
    
    # Save L2 importance
    l2_df = pd.DataFrame({
        'gene': l2_results['gene_names'],
        'importance': l2_results['importance']
    }).sort_values('importance', ascending=False)
    
    l2_df.to_csv(output_dir / f'{cohort}_l2_importance.csv', index=False)
    logger.info(f"  Saved: {cohort}_l2_importance.csv")
    
    # Save comparison
    comparison_df.to_csv(output_dir / f'{cohort}_method_comparison.csv', index=False)
    logger.info(f"  Saved: {cohort}_method_comparison.csv")


# =============================================================================
# Main Execution
# =============================================================================

def process_cohort(
    cohort: str,
    expr_file: Path,
    surv_file: Path,
    model_path: Path,
    params_path: Path,
    consensus_genes: list,
    output_dir: Path,
    device: str,
    logger: logging.Logger
) -> pd.DataFrame:
    """Process a single cohort."""
    logger.info("\n" + "="*70)
    logger.info(f"PROCESSING {cohort.upper()} COHORT")
    logger.info("="*70)
    
    # Load data
    expr_df, expr_tensor, sample_ids, gene_names = load_expression_data(
        expr_file, surv_file, consensus_genes, logger
    )
    
    # Load model
    model = load_model(model_path, params_path, len(gene_names), logger)
    
    # Compute L2 importance (current method)
    l2_results = compute_l2_importance(model, gene_names, logger)
    
    # Compute Integrated Gradients
    ig_results = compute_integrated_gradients(
        model, expr_tensor, gene_names, device, logger
    )
    
    # Compute SHAP GradientExplainer
    shap_results = compute_shap_gradientexplainer(
        model, expr_tensor, gene_names, device, logger
    )
    
    # Compare methods
    comparison_df = compare_importance_methods(
        ig_results, shap_results, l2_results, logger
    )
    
    # Save results
    save_results(
        cohort, ig_results, shap_results, l2_results,
        comparison_df, output_dir, logger
    )
    
    return comparison_df


def main():
    parser = argparse.ArgumentParser(
        description='Compute feature importance using Integrated Gradients and SHAP'
    )
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for model checkpoint (default: 42)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: cuda, cpu, or auto (default: auto)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Create output directory
    seed_output_dir = OUTPUT_DIR / f"seed_{args.seed}"
    seed_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(OUTPUT_DIR, args.seed)
    
    logger.info("="*70)
    logger.info("FEATURE IMPORTANCE COMPUTATION")
    logger.info("Integrated Gradients + SHAP GradientExplainer")
    logger.info("="*70)
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Device: {device}")
    logger.info(f"Output: {seed_output_dir}")
    logger.info("="*70)
    
    # Load consensus genes
    consensus_genes = load_consensus_genes(CONSENSUS_GENES_FILE)
    logger.info(f"\nLoaded {len(consensus_genes)} consensus genes")
    
    # Model paths
    tcga_model_path = BIOMARKER_DIR / f"seed_{args.seed}" / "tcga_model.pth"
    orien_model_path = BIOMARKER_DIR / f"seed_{args.seed}" / "orien_model.pth"
    tcga_params_path = HYPERPARAMS_DIR / "tcga_308genes" / "best_params.json"
    orien_params_path = HYPERPARAMS_DIR / "orien_308genes" / "best_params.json"
    
    # Verify files exist
    for path, name in [
        (tcga_model_path, "TCGA model"),
        (orien_model_path, "ORIEN model"),
        (tcga_params_path, "TCGA params"),
        (orien_params_path, "ORIEN params"),
        (CONSENSUS_GENES_FILE, "Consensus genes"),
        (COX_GENES_FILE, "Cox genes")
    ]:
        if not path.exists():
            logger.error(f"{name} not found: {path}")
            sys.exit(1)
        logger.info(f"Found: {name}")
    
    # Process TCGA
    tcga_comparison = process_cohort(
        cohort='tcga',
        expr_file=TCGA_EXPR_FILE,
        surv_file=TCGA_SURV_FILE,
        model_path=tcga_model_path,
        params_path=tcga_params_path,
        consensus_genes=consensus_genes,
        output_dir=seed_output_dir,
        device=device,
        logger=logger
    )
    
    # Process ORIEN
    orien_comparison = process_cohort(
        cohort='orien',
        expr_file=ORIEN_EXPR_FILE,
        surv_file=ORIEN_SURV_FILE,
        model_path=orien_model_path,
        params_path=orien_params_path,
        consensus_genes=consensus_genes,
        output_dir=seed_output_dir,
        device=device,
        logger=logger
    )
    
    # Cox gene overlap analysis
    logger.info("\n" + "="*70)
    logger.info("COX GENE OVERLAP ANALYSIS")
    logger.info("="*70)
    
    logger.info("\nTCGA:")
    tcga_cox_overlap = check_cox_gene_overlap(tcga_comparison, COX_GENES_FILE, logger)
    
    logger.info("\nORIEN:")
    orien_cox_overlap = check_cox_gene_overlap(orien_comparison, COX_GENES_FILE, logger)
    
    # Save Cox overlap results
    cox_overlap_df = pd.DataFrame({
        'k': tcga_cox_overlap['k'],
        'tcga_ig_overlap': tcga_cox_overlap['ig_overlap'],
        'tcga_l2_overlap': tcga_cox_overlap['l2_overlap'],
        'tcga_shap_overlap': tcga_cox_overlap['shap_overlap'],
        'orien_ig_overlap': orien_cox_overlap['ig_overlap'],
        'orien_l2_overlap': orien_cox_overlap['l2_overlap'],
        'orien_shap_overlap': orien_cox_overlap['shap_overlap']
    })
    
    comparison_dir = OUTPUT_DIR / "comparison_with_l2"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    cox_overlap_df.to_csv(comparison_dir / f'cox_gene_overlap_seed{args.seed}.csv', index=False)
    logger.info(f"\nSaved Cox overlap analysis to: {comparison_dir}")
    
    # Final summary
    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    
    logger.info("\nImportance Score Statistics:")
    logger.info("-" * 50)
    
    for cohort, df in [('TCGA', tcga_comparison), ('ORIEN', orien_comparison)]:
        ig_range = df['ig_importance'].max() - df['ig_importance'].min()
        ig_cv = df['ig_importance'].std() / df['ig_importance'].mean()
        l2_range = df['l2_importance'].max() - df['l2_importance'].min()
        l2_cv = df['l2_importance'].std() / df['l2_importance'].mean()
        
        logger.info(f"\n{cohort}:")
        logger.info(f"  IG:  Range={ig_range:.4f}, CV={ig_cv:.4f}")
        logger.info(f"  L2:  Range={l2_range:.4f}, CV={l2_cv:.4f}")
        logger.info(f"  Improvement: {ig_range/l2_range:.2f}x range, {ig_cv/l2_cv:.2f}x CV")
    
    logger.info("\n" + "="*70)
    logger.info("COMPLETE")
    logger.info("="*70)
    logger.info(f"Results saved to: {seed_output_dir}")
    

if __name__ == "__main__":
    main()
