"""
Compute Feature Importance using Integrated Gradients and SHAP
Version 2: Added convergence validation and per-sample attribution saving

This script computes gene importance scores using:
1. Integrated Gradients (Captum) - primary method
2. SHAP GradientExplainer - for validation

NEW IN V2:
- Convergence validation (completeness axiom check)
- Per-sample attribution saving for downstream analysis
- Sensitivity analysis for n_steps parameter

References:
- Sundararajan et al. (2017) "Axiomatic Attribution for Deep Networks" - ICML
- Lundberg & Lee (2017) "A Unified Approach to Interpreting Model Predictions" - NeurIPS
- Kokhlikyan et al. (2020) "Captum: A unified and generic model interpretability library" - arXiv

Usage:
    python compute_shap_ig_importance_v2.py --seed 42
    python compute_shap_ig_importance_v2.py --seed 42 --validate_convergence
    python compute_shap_ig_importance_v2.py --seed 42 --n_steps 100

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
        gene_names: List of gene names (filtered)
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
    
    # Verify model is in eval mode
    logger.info(f"  Model training mode: {model.training} (should be False)")
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Parameters: {n_params:,}")
    
    return model


# =============================================================================
# Convergence Validation (Completeness Check)
# =============================================================================

def validate_convergence(
    model: nn.Module,
    expr_tensor: torch.Tensor,
    attributions: np.ndarray,
    baseline: torch.Tensor,
    device: str,
    logger: logging.Logger,
    tolerance: float = 0.05
) -> dict:
    """
    Validate Integrated Gradients convergence using the completeness axiom.
    
    The completeness axiom states:
        sum(IG_i(x)) = F(x) - F(baseline)
    
    Args:
        model: Trained model
        expr_tensor: Input expression tensor (samples x genes)
        attributions: Computed IG attributions (samples x genes)
        baseline: Baseline tensor (1 x genes)
        device: 'cuda' or 'cpu'
        logger: Logger instance
        tolerance: Acceptable relative error threshold (default 5%)
        
    Returns:
        Dictionary with convergence statistics
    """
    logger.info("\n" + "="*60)
    logger.info("CONVERGENCE VALIDATION (Completeness Axiom)")
    logger.info("="*60)
    
    model = model.to(device)
    model.eval()
    
    n_samples = expr_tensor.shape[0]
    
    # Compute model outputs
    with torch.no_grad():
        # Output for all samples
        outputs = model(expr_tensor.to(device)).cpu().numpy().flatten()
        
        # Output for baseline
        baseline_output = model(baseline.to(device)).cpu().numpy().flatten()[0]
    
    logger.info(f"  Baseline output F(x'): {baseline_output:.6f}")
    logger.info(f"  Sample outputs F(x) range: [{outputs.min():.6f}, {outputs.max():.6f}]")
    
    # Compute expected difference: F(x) - F(baseline)
    expected_diff = outputs - baseline_output
    
    # Compute actual sum of attributions
    attribution_sums = attributions.sum(axis=1)
    
    # Compute approximation errors
    absolute_errors = np.abs(expected_diff - attribution_sums)
    
    # Compute relative errors (with small epsilon to avoid division by zero)
    epsilon = 1e-7
    relative_errors = absolute_errors / (np.abs(expected_diff) + epsilon)
    
    # Statistics
    mean_abs_error = absolute_errors.mean()
    max_abs_error = absolute_errors.max()
    mean_rel_error = relative_errors.mean()
    max_rel_error = relative_errors.max()
    
    # Count samples meeting tolerance
    samples_within_tolerance = (relative_errors < tolerance).sum()
    pct_within_tolerance = 100 * samples_within_tolerance / n_samples
    
    # Log results
    logger.info(f"\n  Convergence Statistics:")
    logger.info(f"    Mean absolute error: {mean_abs_error:.6f}")
    logger.info(f"    Max absolute error:  {max_abs_error:.6f}")
    logger.info(f"    Mean relative error: {mean_rel_error:.4f} ({100*mean_rel_error:.2f}%)")
    logger.info(f"    Max relative error:  {max_rel_error:.4f} ({100*max_rel_error:.2f}%)")
    logger.info(f"    Samples within {100*tolerance:.0f}% tolerance: "
                f"{samples_within_tolerance}/{n_samples} ({pct_within_tolerance:.1f}%)")
    
    # Determine if convergence is acceptable
    converged = (mean_rel_error < tolerance) and (pct_within_tolerance >= 95)
    
    if converged:
        logger.info(f"\n  ✓ CONVERGENCE VALIDATED")
        logger.info(f"    Mean relative error ({100*mean_rel_error:.2f}%) < {100*tolerance:.0f}%")
        logger.info(f"    {pct_within_tolerance:.1f}% samples within tolerance (>= 95% required)")
    else:
        logger.warning(f"\n  ⚠ CONVERGENCE WARNING")
        if mean_rel_error >= tolerance:
            logger.warning(f"    Mean relative error ({100*mean_rel_error:.2f}%) >= {100*tolerance:.0f}%")
        if pct_within_tolerance < 95:
            logger.warning(f"    Only {pct_within_tolerance:.1f}% samples within tolerance (< 95%)")
        logger.warning(f"    Consider increasing n_steps for better approximation")
    
    # Compile results
    results = {
        'converged': converged,
        'tolerance': tolerance,
        'mean_absolute_error': float(mean_abs_error),
        'max_absolute_error': float(max_abs_error),
        'mean_relative_error': float(mean_rel_error),
        'max_relative_error': float(max_rel_error),
        'samples_within_tolerance': int(samples_within_tolerance),
        'pct_within_tolerance': float(pct_within_tolerance),
        'n_samples': n_samples,
        'baseline_output': float(baseline_output),
        'expected_diff': expected_diff.tolist(),
        'attribution_sums': attribution_sums.tolist(),
        'absolute_errors': absolute_errors.tolist(),
        'relative_errors': relative_errors.tolist()
    }
    
    return results


def run_convergence_sensitivity_analysis(
    model: nn.Module,
    expr_tensor: torch.Tensor,
    baseline: torch.Tensor,
    gene_names: list,
    device: str,
    logger: logging.Logger,
    n_steps_list: list = [20, 50, 100, 200]
) -> pd.DataFrame:
    """
    Run sensitivity analysis for different n_steps values.
    
    This helps determine the minimum n_steps needed for convergence.
    
    Args:
        model: Trained model
        expr_tensor: Input expression tensor
        baseline: Baseline tensor
        gene_names: List of gene names
        device: 'cuda' or 'cpu'
        logger: Logger instance
        n_steps_list: List of n_steps values to test
        
    Returns:
        DataFrame with convergence statistics for each n_steps
    """
    logger.info("\n" + "="*60)
    logger.info("CONVERGENCE SENSITIVITY ANALYSIS")
    logger.info("="*60)
    logger.info(f"Testing n_steps: {n_steps_list}")
    
    model = model.to(device)
    model.eval()
    
    # Use subset of samples for efficiency
    n_samples_test = min(100, expr_tensor.shape[0])
    test_indices = np.random.choice(expr_tensor.shape[0], n_samples_test, replace=False)
    test_tensor = expr_tensor[test_indices]
    test_baseline = baseline.expand(n_samples_test, -1)
    
    logger.info(f"Using {n_samples_test} samples for sensitivity analysis")
    
    results = []
    
    for n_steps in n_steps_list:
        logger.info(f"\n  Testing n_steps = {n_steps}...")
        
        # Create IG with this n_steps
        ig = IntegratedGradients(model)
        
        # Compute attributions
        attributions = ig.attribute(
            test_tensor.to(device),
            baselines=test_baseline.to(device),
            n_steps=n_steps,
            return_convergence_delta=False
        ).cpu().numpy()
        
        # Validate convergence
        conv_results = validate_convergence(
            model, test_tensor, attributions, baseline,
            device, logger, tolerance=0.05
        )
        
        results.append({
            'n_steps': n_steps,
            'mean_relative_error': conv_results['mean_relative_error'],
            'max_relative_error': conv_results['max_relative_error'],
            'pct_within_5pct': conv_results['pct_within_tolerance'],
            'converged': conv_results['converged']
        })
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(results)
    
    logger.info("\n" + "="*60)
    logger.info("SENSITIVITY ANALYSIS SUMMARY")
    logger.info("="*60)
    logger.info("\n" + summary_df.to_string(index=False))
    
    # Recommend optimal n_steps
    converged_steps = summary_df[summary_df['converged']]['n_steps'].tolist()
    if converged_steps:
        recommended = min(converged_steps)
        logger.info(f"\n  Recommended n_steps: {recommended} (minimum for convergence)")
    else:
        logger.warning(f"\n  ⚠ No n_steps value achieved convergence. Consider n_steps > {max(n_steps_list)}")
    
    return summary_df


# =============================================================================
# Integrated Gradients Computation (Enhanced)
# =============================================================================

def compute_integrated_gradients(
    model: nn.Module,
    expr_tensor: torch.Tensor,
    gene_names: list,
    sample_ids: list,
    device: str,
    logger: logging.Logger,
    n_steps: int = 50,
    validate: bool = True
) -> dict:
    """
    Compute Integrated Gradients attributions with convergence validation.
    
    Args:
        model: Trained ElasticDeepSurv model
        expr_tensor: Expression tensor (samples x genes)
        gene_names: List of gene names
        sample_ids: List of sample IDs (for per-sample output)
        device: 'cuda' or 'cpu'
        logger: Logger instance
        n_steps: Number of integration steps (default 50)
        validate: Whether to run convergence validation (default True)
        
    Returns:
        Dictionary with attribution results including convergence info
    """
    logger.info("\n" + "="*60)
    logger.info("COMPUTING INTEGRATED GRADIENTS")
    logger.info("="*60)
    logger.info(f"  Samples: {expr_tensor.shape[0]}")
    logger.info(f"  Genes: {expr_tensor.shape[1]}")
    logger.info(f"  Integration steps: {n_steps}")
    logger.info(f"  Convergence validation: {validate}")
    
    # Move model and data to device
    model = model.to(device)
    model.eval()
    expr_tensor = expr_tensor.to(device)
    
    # Compute mean baseline (average patient)
    baseline = expr_tensor.mean(dim=0, keepdim=True)  # Shape: (1, n_genes)
    logger.info(f"  Baseline shape: {baseline.shape}")
    logger.info(f"  Baseline mean: {baseline.mean():.6f}")
    logger.info(f"  Baseline std: {baseline.std():.6f}")
    
    # Note: Since data is z-scored, baseline should be approximately zero
    logger.info(f"  Baseline L2 norm: {torch.norm(baseline):.6f} (should be small for z-scored data)")
    
    # Expand baseline to match input size for batch processing
    baseline_expanded = baseline.expand(expr_tensor.shape[0], -1)
    
    # Create Integrated Gradients attributor
    ig = IntegratedGradients(model)
    
    # Compute attributions
    logger.info("  Computing attributions (this may take a few minutes)...")
    
    # Process in batches to avoid memory issues
    batch_size = 100
    n_samples = expr_tensor.shape[0]
    all_attributions = []
    all_deltas = []  # For convergence check
    
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch_inputs = expr_tensor[i:end_idx]
        batch_baselines = baseline_expanded[i:end_idx]
        
        # Compute attributions with convergence delta
        attributions, delta = ig.attribute(
            batch_inputs,
            baselines=batch_baselines,
            n_steps=n_steps,
            return_convergence_delta=True
        )
        
        all_attributions.append(attributions.cpu())
        all_deltas.append(delta.cpu())
        
        if (i // batch_size + 1) % 5 == 0 or end_idx == n_samples:
            logger.info(f"    Processed {end_idx}/{n_samples} samples")
    
    # Concatenate all attributions
    attributions = torch.cat(all_attributions, dim=0)
    convergence_deltas = torch.cat(all_deltas, dim=0)
    
    logger.info(f"  Attributions shape: {attributions.shape}")
    
    # Convert to numpy
    attributions_np = attributions.numpy()
    deltas_np = convergence_deltas.numpy().flatten()
    
    # Log convergence delta statistics (Captum's built-in check)
    logger.info(f"\n  Captum Convergence Deltas:")
    logger.info(f"    Mean |delta|: {np.abs(deltas_np).mean():.6f}")
    logger.info(f"    Max |delta|:  {np.abs(deltas_np).max():.6f}")
    logger.info(f"    Std delta:    {deltas_np.std():.6f}")
    
    # Compute aggregated importance scores
    # 1. Mean absolute attribution (magnitude)
    importance_magnitude = np.abs(attributions_np).mean(axis=0)
    
    # 2. Signed mean attribution (direction)
    importance_signed = attributions_np.mean(axis=0)
    
    # 3. Standard deviation (variability across samples)
    importance_std = attributions_np.std(axis=0)
    
    # 4. Median absolute attribution (robust to outliers)
    importance_median = np.median(np.abs(attributions_np), axis=0)
    
    # Log statistics
    logger.info(f"\n  Importance Statistics (Magnitude - Mean |IG|):")
    logger.info(f"    Range: [{importance_magnitude.min():.6f}, {importance_magnitude.max():.6f}]")
    logger.info(f"    Mean: {importance_magnitude.mean():.6f}")
    logger.info(f"    Std: {importance_magnitude.std():.6f}")
    logger.info(f"    CV: {importance_magnitude.std() / importance_magnitude.mean():.4f}")
    
    logger.info(f"\n  Importance Statistics (Signed - Mean IG):")
    logger.info(f"    Range: [{importance_signed.min():.6f}, {importance_signed.max():.6f}]")
    logger.info(f"    Mean: {importance_signed.mean():.6f}")
    logger.info(f"    Positive (risk): {(importance_signed > 0).sum()} genes")
    logger.info(f"    Negative (protective): {(importance_signed < 0).sum()} genes")
    
    # Run full convergence validation if requested
    convergence_results = None
    if validate:
        convergence_results = validate_convergence(
            model, expr_tensor.cpu(), attributions_np, baseline.cpu(),
            device, logger, tolerance=0.05
        )
    
    # Create results dictionary
    results = {
        # Per-sample attributions (full matrix)
        'attributions_per_sample': attributions_np,  # (n_samples, n_genes)
        'sample_ids': sample_ids,  # For mapping back to patients
        
        # Aggregated importance scores
        'importance_magnitude': importance_magnitude,  # (n_genes,)
        'importance_signed': importance_signed,  # (n_genes,)
        'importance_std': importance_std,  # (n_genes,)
        'importance_median': importance_median,  # (n_genes,)
        
        # Gene names
        'gene_names': gene_names,
        
        # Metadata
        'n_samples': n_samples,
        'n_genes': len(gene_names),
        'n_steps': n_steps,
        'baseline_type': 'mean',
        'baseline_values': baseline.cpu().numpy().flatten(),
        
        # Convergence information
        'convergence_deltas': deltas_np,  # Captum's convergence delta per sample
        'convergence_results': convergence_results  # Full validation results
    }
    
    return results


# =============================================================================
# Save Results (Enhanced)
# =============================================================================

def save_per_sample_attributions(
    cohort: str,
    ig_results: dict,
    output_dir: Path,
    logger: logging.Logger
):
    """
    Save per-sample attributions to files.
    
    Saves:
    1. Full attribution matrix (samples x genes) as CSV
    2. Full attribution matrix as NPY for efficient loading
    3. Sample metadata file
    
    Args:
        cohort: 'tcga' or 'orien'
        ig_results: Dictionary from compute_integrated_gradients
        output_dir: Output directory
        logger: Logger instance
    """
    logger.info(f"\n  Saving per-sample attributions for {cohort.upper()}...")
    
    # Create per-sample directory
    per_sample_dir = output_dir / "per_sample_attributions"
    per_sample_dir.mkdir(parents=True, exist_ok=True)
    
    attributions = ig_results['attributions_per_sample']
    sample_ids = ig_results['sample_ids']
    gene_names = ig_results['gene_names']
    
    n_samples, n_genes = attributions.shape
    logger.info(f"    Attribution matrix shape: {n_samples} samples x {n_genes} genes")
    
    # 1. Save as CSV (human-readable, larger file)
    # Create DataFrame with sample IDs as index and gene names as columns
    attr_df = pd.DataFrame(
        attributions,
        index=sample_ids,
        columns=gene_names
    )
    attr_df.index.name = 'sample_id'
    
    csv_path = per_sample_dir / f'{cohort}_attributions_per_sample.csv'
    attr_df.to_csv(csv_path)
    logger.info(f"    Saved CSV: {csv_path.name} ({csv_path.stat().st_size / 1e6:.1f} MB)")
    
    # 2. Save as NPY (efficient for loading in Python)
    npy_path = per_sample_dir / f'{cohort}_attributions_per_sample.npy'
    np.save(npy_path, attributions)
    logger.info(f"    Saved NPY: {npy_path.name} ({npy_path.stat().st_size / 1e6:.1f} MB)")
    
    # 3. Save sample metadata
    metadata = {
        'sample_ids': sample_ids,
        'gene_names': gene_names,
        'n_samples': n_samples,
        'n_genes': n_genes,
        'cohort': cohort
    }
    
    metadata_path = per_sample_dir / f'{cohort}_attribution_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"    Saved metadata: {metadata_path.name}")
    
    # 4. Save per-sample summary statistics
    sample_summary = pd.DataFrame({
        'sample_id': sample_ids,
        'attribution_sum': attributions.sum(axis=1),
        'attribution_mean': attributions.mean(axis=1),
        'attribution_std': attributions.std(axis=1),
        'n_positive': (attributions > 0).sum(axis=1),
        'n_negative': (attributions < 0).sum(axis=1),
        'max_attribution': attributions.max(axis=1),
        'min_attribution': attributions.min(axis=1)
    })
    
    summary_path = per_sample_dir / f'{cohort}_per_sample_summary.csv'
    sample_summary.to_csv(summary_path, index=False)
    logger.info(f"    Saved per-sample summary: {summary_path.name}")
    
    return per_sample_dir


def save_convergence_results(
    cohort: str,
    ig_results: dict,
    output_dir: Path,
    logger: logging.Logger
):
    """
    Save convergence validation results.
    
    Args:
        cohort: 'tcga' or 'orien'
        ig_results: Dictionary from compute_integrated_gradients
        output_dir: Output directory
        logger: Logger instance
    """
    logger.info(f"\n  Saving convergence results for {cohort.upper()}...")
    
    conv_dir = output_dir / "convergence_validation"
    conv_dir.mkdir(parents=True, exist_ok=True)
    
    # Save convergence deltas
    deltas_df = pd.DataFrame({
        'sample_id': ig_results['sample_ids'],
        'convergence_delta': ig_results['convergence_deltas']
    })
    deltas_df.to_csv(conv_dir / f'{cohort}_convergence_deltas.csv', index=False)
    
    # Save full convergence results if available
    if ig_results['convergence_results'] is not None:
        conv_results = ig_results['convergence_results']
        
        # Summary JSON
        summary = {
            'cohort': cohort,
            'converged': conv_results['converged'],
            'tolerance': conv_results['tolerance'],
            'mean_relative_error': conv_results['mean_relative_error'],
            'max_relative_error': conv_results['max_relative_error'],
            'pct_within_tolerance': conv_results['pct_within_tolerance'],
            'n_samples': conv_results['n_samples'],
            'n_steps': ig_results['n_steps']
        }
        
        with open(conv_dir / f'{cohort}_convergence_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Detailed per-sample errors
        errors_df = pd.DataFrame({
            'sample_id': ig_results['sample_ids'],
            'expected_diff': conv_results['expected_diff'],
            'attribution_sum': conv_results['attribution_sums'],
            'absolute_error': conv_results['absolute_errors'],
            'relative_error': conv_results['relative_errors']
        })
        errors_df.to_csv(conv_dir / f'{cohort}_convergence_errors.csv', index=False)
        
        logger.info(f"    Saved convergence summary and errors to {conv_dir}")
    
    return conv_dir


def save_results(
    cohort: str,
    ig_results: dict,
    shap_results: dict,
    l2_results: dict,
    comparison_df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
    save_per_sample: bool = True
):
    """
    Save all results to files.
    
    Args:
        cohort: 'tcga' or 'orien'
        ig_results: IG results dictionary
        shap_results: SHAP results dictionary (can be None)
        l2_results: L2 results dictionary
        comparison_df: Method comparison DataFrame
        output_dir: Output directory
        logger: Logger instance
        save_per_sample: Whether to save per-sample attributions
    """
    logger.info(f"\nSaving results to {output_dir}")
    
    # Ensure arrays are flattened
    for results_dict in [ig_results, shap_results]:
        if results_dict is not None:
            for key in ['importance_magnitude', 'importance_signed', 'importance_std', 'importance_median']:
                if key in results_dict and hasattr(results_dict[key], 'shape'):
                    if len(results_dict[key].shape) > 1:
                        results_dict[key] = results_dict[key].flatten()
    
    # Save IG importance (aggregated)
    ig_df = pd.DataFrame({
        'gene': ig_results['gene_names'],
        'importance_magnitude': ig_results['importance_magnitude'],
        'importance_signed': ig_results['importance_signed'],
        'importance_std': ig_results['importance_std'],
        'importance_median': ig_results.get('importance_median', ig_results['importance_magnitude'])
    }).sort_values('importance_magnitude', ascending=False)
    
    ig_df.to_csv(output_dir / f'{cohort}_ig_importance.csv', index=False)
    logger.info(f"  Saved: {cohort}_ig_importance.csv")
    
    # Save per-sample attributions
    if save_per_sample:
        save_per_sample_attributions(cohort, ig_results, output_dir, logger)
    
    # Save convergence results
    save_convergence_results(cohort, ig_results, output_dir, logger)
    
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
# L2 Importance (for comparison)
# =============================================================================

def compute_l2_importance(
    model: nn.Module,
    gene_names: list,
    logger: logging.Logger
) -> dict:
    """
    Compute L2 norm importance from first layer weights.
    This is the baseline method - included for comparison.
    """
    logger.info("Computing L2 norm importance (baseline method)...")
    
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
# SHAP Computation (unchanged from original)
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
    """
    logger.info("Computing SHAP GradientExplainer...")
    logger.info(f"  Background samples: {n_background}")
    
    model = model.to(device)
    model.eval()
    
    n_samples = expr_tensor.shape[0]
    if n_background >= n_samples:
        background_idx = np.arange(n_samples)
    else:
        np.random.seed(42)
        background_idx = np.random.choice(n_samples, n_background, replace=False)
    
    background = expr_tensor[background_idx].to(device)
    
    try:
        explainer = shap.GradientExplainer(model, background)
        logger.info("  GradientExplainer created successfully")
    except Exception as e:
        logger.error(f"  Failed to create GradientExplainer: {e}")
        return None
    
    logger.info("  Computing SHAP values...")
    expr_device = expr_tensor.to(device)
    
    try:
        shap_values = explainer.shap_values(expr_device)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        if torch.is_tensor(shap_values):
            shap_values = shap_values.cpu().numpy()
            
    except Exception as e:
        logger.error(f"  Failed to compute SHAP values: {e}")
        return None
    
    importance_magnitude = np.abs(shap_values).mean(axis=0)
    importance_signed = shap_values.mean(axis=0)
    importance_std = shap_values.std(axis=0)
    
    logger.info(f"  SHAP values shape: {shap_values.shape}")
    logger.info(f"  Importance range: [{importance_magnitude.min():.6f}, {importance_magnitude.max():.6f}]")
    
    return {
        'shap_values_per_sample': shap_values,
        'importance_magnitude': importance_magnitude,
        'importance_signed': importance_signed,
        'importance_std': importance_std,
        'gene_names': gene_names,
        'n_samples': n_samples,
        'n_background': n_background
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
    
    ig_importance = ig_results['importance_magnitude']
    l2_importance = l2_results['importance']
    
    if shap_results is not None:
        shap_importance = shap_results['importance_magnitude']
    else:
        shap_importance = np.full(len(gene_names), np.nan)
    
    # Create rankings
    ig_ranks = stats.rankdata(-ig_importance)
    l2_ranks = stats.rankdata(-l2_importance)
    
    if shap_results is not None:
        shap_ranks = stats.rankdata(-shap_importance)
    else:
        shap_ranks = np.full(len(gene_names), np.nan)
    
    # Compute correlations
    logger.info("\nSpearman Rank Correlations:")
    
    corr_ig_l2, p_ig_l2 = stats.spearmanr(ig_importance, l2_importance)
    logger.info(f"  IG vs L2:   rho = {corr_ig_l2:.4f}, p = {p_ig_l2:.2e}")
    
    if shap_results is not None:
        corr_ig_shap, p_ig_shap = stats.spearmanr(ig_importance, shap_importance)
        logger.info(f"  IG vs SHAP: rho = {corr_ig_shap:.4f}, p = {p_ig_shap:.2e}")
        
        corr_shap_l2, p_shap_l2 = stats.spearmanr(shap_importance, l2_importance)
        logger.info(f"  SHAP vs L2: rho = {corr_shap_l2:.4f}, p = {p_shap_l2:.2e}")
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'gene': gene_names,
        'ig_importance': ig_importance,
        'ig_signed': ig_results['importance_signed'],
        'ig_rank': ig_ranks,
        'l2_importance': l2_importance,
        'l2_rank': l2_ranks,
        'shap_importance': shap_importance,
        'shap_rank': shap_ranks,
        'rank_diff_ig_l2': np.abs(ig_ranks - l2_ranks)
    }).sort_values('ig_importance', ascending=False)
    
    return comparison_df


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
    logger: logging.Logger,
    n_steps: int = 50,
    validate_convergence: bool = True,
    save_per_sample: bool = True
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
    
    # Compute L2 importance (baseline method)
    l2_results = compute_l2_importance(model, gene_names, logger)
    
    # Compute Integrated Gradients (with convergence validation)
    ig_results = compute_integrated_gradients(
        model, expr_tensor, gene_names, sample_ids,
        device, logger, n_steps=n_steps, validate=validate_convergence
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
        comparison_df, output_dir, logger, save_per_sample=save_per_sample
    )
    
    return comparison_df


def main():
    parser = argparse.ArgumentParser(
        description='Compute feature importance using Integrated Gradients and SHAP (v2)'
    )
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for model checkpoint (default: 42)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: cuda, cpu, or auto (default: auto)')
    parser.add_argument('--n_steps', type=int, default=50,
                        help='Number of IG integration steps (default: 50)')
    parser.add_argument('--validate_convergence', action='store_true',
                        help='Run full convergence validation')
    parser.add_argument('--sensitivity_analysis', action='store_true',
                        help='Run n_steps sensitivity analysis')
    parser.add_argument('--no_per_sample', action='store_true',
                        help='Skip saving per-sample attributions')
    
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
    logger.info("FEATURE IMPORTANCE COMPUTATION (v2)")
    logger.info("Integrated Gradients + SHAP GradientExplainer")
    logger.info("With Convergence Validation and Per-Sample Attribution")
    logger.info("="*70)
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Device: {device}")
    logger.info(f"Integration steps: {args.n_steps}")
    logger.info(f"Convergence validation: {args.validate_convergence}")
    logger.info(f"Save per-sample: {not args.no_per_sample}")
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
        (CONSENSUS_GENES_FILE, "Consensus genes")
    ]:
        if not path.exists():
            logger.error(f"{name} not found: {path}")
            sys.exit(1)
        logger.info(f"Found: {name}")
    
    # Run sensitivity analysis if requested
    if args.sensitivity_analysis:
        logger.info("\n" + "="*70)
        logger.info("RUNNING N_STEPS SENSITIVITY ANALYSIS")
        logger.info("="*70)
        
        # Load TCGA for sensitivity analysis
        _, expr_tensor, _, gene_names = load_expression_data(
            TCGA_EXPR_FILE, TCGA_SURV_FILE, consensus_genes, logger
        )
        model = load_model(tcga_model_path, tcga_params_path, len(gene_names), logger)
        baseline = expr_tensor.mean(dim=0, keepdim=True)
        
        sensitivity_df = run_convergence_sensitivity_analysis(
            model, expr_tensor, baseline, gene_names,
            device, logger, n_steps_list=[20, 50, 100, 200]
        )
        
        sensitivity_df.to_csv(seed_output_dir / 'convergence_sensitivity_analysis.csv', index=False)
        logger.info(f"Saved sensitivity analysis to {seed_output_dir}")
    
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
        logger=logger,
        n_steps=args.n_steps,
        validate_convergence=args.validate_convergence,
        save_per_sample=not args.no_per_sample
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
        logger=logger,
        n_steps=args.n_steps,
        validate_convergence=args.validate_convergence,
        save_per_sample=not args.no_per_sample
    )
    
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
    logger.info(f"\nOutput files:")
    logger.info(f"  - {cohort}_ig_importance.csv (aggregated importance)")
    logger.info(f"  - per_sample_attributions/{cohort}_attributions_per_sample.csv")
    logger.info(f"  - per_sample_attributions/{cohort}_attributions_per_sample.npy")
    logger.info(f"  - convergence_validation/{cohort}_convergence_summary.json")


if __name__ == "__main__":
    main()
