"""
Consensus-Based Biomarker Validation with Risk Score Calculation
================================================================

Method 1: Strict Consensus Approach
1. Identify consensus genes (TCGA_top_k ∩ ORIEN_top_k)
2. Retrain models using ONLY consensus genes
3. Bidirectional validation with same gene set
4. Calculate risk scores for Kaplan-Meier analysis

This matches the methodology of Cox elastic net (Chapter 2).
"""

import sys
sys.path.append('.')

import torch
import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Set
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

from src.data.dataset import SurvivalDataset
from torch.utils.data import DataLoader
from src.models.elastic_deepsurv import ElasticDeepSurv, ElasticDeepSurvTrainer
from src.utils.batch_samplers import StratifiedBatchSampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

# Dense sampling in the region where overlap becomes useful
K_VALUES = [40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110, 120, 130, 140, 150]

CONSENSUS_GENES_FILE = "data/raw/consensus_genes_308.txt"
COX_BASELINE_FILE = "data/raw/cox_consensus_genes_20.txt"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_consensus_genes(filepath: str) -> List[str]:
    """Load gene list from file."""
    with open(filepath, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]
    return genes


def compute_l2_feature_importance(model: ElasticDeepSurv) -> np.ndarray:
    """Compute L2 norm of first layer weights as feature importance."""
    first_layer = model.network[0]
    weights = first_layer.weight.data.cpu().numpy()
    importance = np.linalg.norm(weights, axis=0)
    return importance


def get_top_k_genes(
    importance: np.ndarray,
    gene_names: List[str],
    k: int
) -> List[str]:
    """Get top k genes by importance."""
    top_k_indices = np.argsort(importance)[::-1][:k]
    return [gene_names[i] for i in top_k_indices]


def train_consensus_model(
    expr_standardized: pd.DataFrame,
    surv: pd.DataFrame,
    consensus_genes: List[str],
    best_params: dict,
    n_epochs: int = 50,
    device: str = 'cuda',
    verbose: bool = False
) -> ElasticDeepSurv:
    """
    Train a model using ONLY consensus genes.
    
    Uses simple architecture appropriate for small gene sets.
    
    Args:
        expr_standardized: Full expression data (genes × samples)
        surv: Survival data
        consensus_genes: List of consensus genes to use
        best_params: Hyperparameters
        n_epochs: Training epochs
        device: Device to use
        verbose: Print training progress
        
    Returns:
        Trained model
    """
    n_consensus = len(consensus_genes)
    
    # Filter to consensus genes only
    expr_consensus = expr_standardized.loc[consensus_genes, :]
    
    # Create dataset
    dataset = SurvivalDataset(expr_consensus, surv)
    
    batch_size = min(best_params.get('batch_size', 32), len(dataset) // 4)
    sampler = StratifiedBatchSampler(
        events=surv['event'].values,
        batch_size=batch_size,
        shuffle=True
    )
    loader = DataLoader(dataset, batch_sampler=sampler)
    
    # Simple architecture for consensus genes
    # Rule: hidden layer size = min(64, 2 × n_consensus)
    hidden_size = min(64, max(16, n_consensus * 2))
    
    logger.info(f"  Training architecture: {n_consensus} → {hidden_size} → 1")
    
    model = ElasticDeepSurv(
        n_features=n_consensus,
        hidden_sizes=[hidden_size],
        dropout=best_params.get('dropout', 0.3),
        activation=best_params.get('activation', 'relu'),
        batch_norm=best_params.get('batch_norm', False),
        weight_init=best_params.get('weight_init', 'xavier_normal'),
        l1_ratio=best_params.get('l1_ratio', 0.9),
        alpha=best_params.get('alpha', 0.001)
    )
    
    trainer = ElasticDeepSurvTrainer(
        model=model,
        learning_rate=best_params.get('learning_rate', 1e-4),
        weight_decay=0.0,
        device=device
    )
    
    # Train without validation (cross-cohort is our validation)
    trainer.fit(
        train_loader=loader,
        valid_loader=None,
        n_epochs=n_epochs,
        early_stopping_patience=None,
        verbose=verbose
    )
    
    return model


def evaluate_consensus_model_with_risks(
    model: ElasticDeepSurv,
    expr_test: pd.DataFrame,
    surv_test: pd.DataFrame,
    consensus_genes: List[str],
    device: str = 'cuda'
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate consensus model and return risk scores for KM analysis.
    
    Args:
        model: Trained model
        expr_test: Test expression data (ALL genes × samples)
        surv_test: Test survival data
        consensus_genes: Consensus genes used by model
        device: Device
        
    Returns:
        Tuple of (C-index, risk_scores, times, events)
    """
    model.to(device)
    model.eval()
    
    # Filter test data to consensus genes only
    expr_consensus = expr_test.loc[consensus_genes, :]
    
    dataset = SurvivalDataset(expr_consensus, surv_test)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    all_risks = []
    all_times = []
    all_events = []
    
    with torch.no_grad():
        for batch in loader:
            expr_batch = batch['expr'].to(device)
            time_batch = batch['time'].cpu().numpy()
            event_batch = batch['event'].cpu().numpy()
            
            # Get risk predictions (higher = worse prognosis)
            risk = model(expr_batch).cpu().numpy().flatten()
            
            all_risks.extend(risk)
            all_times.extend(time_batch)
            all_events.extend(event_batch)
    
    all_risks = np.array(all_risks)
    all_times = np.array(all_times)
    all_events = np.array(all_events)
    
    # Compute C-index
    cindex = concordance_index(all_times, -all_risks, all_events)
    
    return cindex, all_risks, all_times, all_events


def perform_km_analysis(
    risk_scores: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    cohort_name: str,
    output_dir: Path,
    k: int
):
    """
    Perform Kaplan-Meier analysis with risk stratification.
    
    Splits patients into high/low risk groups and performs log-rank test.
    """
    # Stratify by median risk
    median_risk = np.median(risk_scores)
    high_risk = risk_scores >= median_risk
    low_risk = risk_scores < median_risk
    
    # Kaplan-Meier analysis
    kmf = KaplanMeierFitter()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # High risk group
    kmf.fit(times[high_risk], events[high_risk], label='High Risk')
    kmf.plot_survival_function(ax=ax, ci_show=True, color='red', linewidth=2)
    
    # Low risk group
    kmf.fit(times[low_risk], events[low_risk], label='Low Risk')
    kmf.plot_survival_function(ax=ax, ci_show=True, color='blue', linewidth=2)
    
    # Log-rank test
    results = logrank_test(
        times[high_risk], times[low_risk],
        events[high_risk], events[low_risk]
    )
    
    p_value = results.p_value
    
    ax.set_xlabel('Time (months)', fontsize=12)
    ax.set_ylabel('Survival Probability', fontsize=12)
    ax.set_title(f'{cohort_name} - Kaplan-Meier Analysis (k={k})\n' + 
                 f'Log-rank p-value: {p_value:.4f}',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'km_curve_{cohort_name.lower()}_k{k}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'p_value': p_value,
        'high_risk_n': int(np.sum(high_risk)),
        'low_risk_n': int(np.sum(low_risk)),
        'high_risk_events': int(np.sum(events[high_risk])),
        'low_risk_events': int(np.sum(events[low_risk]))
    }


def test_consensus_at_k(
    k: int,
    tcga_expr: pd.DataFrame,
    tcga_surv: pd.DataFrame,
    orien_expr: pd.DataFrame,
    orien_surv: pd.DataFrame,
    tcga_importance: np.ndarray,
    orien_importance: np.ndarray,
    gene_names: List[str],
    tcga_params: dict,
    orien_params: dict,
    tcga_mean: np.ndarray,
    tcga_std: np.ndarray,
    orien_mean: np.ndarray,
    orien_std: np.ndarray,
    output_dir: Path,
    device: str = 'cuda'
) -> Dict:
    """
    Test consensus validation at a specific k value.
    
    Returns comprehensive results including risk scores for KM analysis.
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"TESTING k = {k}")
    logger.info(f"{'='*70}")
    
    # STEP 1: IDENTIFY CONSENSUS GENES
    logger.info("Step 1: Identifying consensus genes...")
    tcga_top_k = get_top_k_genes(tcga_importance, gene_names, k)
    orien_top_k = get_top_k_genes(orien_importance, gene_names, k)
    
    consensus_genes = sorted(list(set(tcga_top_k) & set(orien_top_k)))
    n_consensus = len(consensus_genes)
    overlap_pct = (n_consensus / k) * 100
    
    logger.info(f"  TCGA top {k}: {len(tcga_top_k)} genes")
    logger.info(f"  ORIEN top {k}: {len(orien_top_k)} genes")
    logger.info(f"  Consensus: {n_consensus} genes ({overlap_pct:.1f}% overlap)")
    
    if n_consensus < 3:
        logger.warning(f"  ⚠️  Only {n_consensus} consensus genes - skipping k={k}")
        return None
    
    # STEP 2: PREPARE STANDARDIZED DATA (consensus genes only)
    logger.info("Step 2: Preparing data...")
    
    # Filter to consensus genes
    tcga_expr_consensus = tcga_expr.loc[consensus_genes, :]
    orien_expr_consensus = orien_expr.loc[consensus_genes, :]
    
    # Compute statistics for consensus genes only
    tcga_mean_consensus = tcga_expr_consensus.mean(axis=1).values
    tcga_std_consensus = tcga_expr_consensus.std(axis=1).values
    orien_mean_consensus = orien_expr_consensus.mean(axis=1).values
    orien_std_consensus = orien_expr_consensus.std(axis=1).values
    
    # Standardize
    tcga_standardized = pd.DataFrame(
        (tcga_expr_consensus.values - tcga_mean_consensus.reshape(-1, 1)) / 
        (tcga_std_consensus.reshape(-1, 1) + 1e-8),
        index=consensus_genes,
        columns=tcga_expr_consensus.columns
    )
    
    orien_standardized = pd.DataFrame(
        (orien_expr_consensus.values - orien_mean_consensus.reshape(-1, 1)) / 
        (orien_std_consensus.reshape(-1, 1) + 1e-8),
        index=consensus_genes,
        columns=orien_expr_consensus.columns
    )
    
    # STEP 3: TRAIN TCGA MODEL (consensus genes only)
    logger.info("Step 3: Training TCGA model on consensus genes...")
    tcga_model_consensus = train_consensus_model(
        tcga_standardized, tcga_surv, consensus_genes,
        tcga_params, n_epochs=50, device=device, verbose=False
    )
    
    # STEP 4: TEST TCGA → ORIEN
    logger.info("Step 4: Testing TCGA model on ORIEN...")
    
    # Standardize ORIEN using TCGA statistics (CRITICAL!)
    orien_by_tcga_stats = pd.DataFrame(
        (orien_expr_consensus.values - tcga_mean_consensus.reshape(-1, 1)) / 
        (tcga_std_consensus.reshape(-1, 1) + 1e-8),
        index=consensus_genes,
        columns=orien_expr_consensus.columns
    )
    
    cindex_tcga_on_orien, risks_orien, times_orien, events_orien = \
        evaluate_consensus_model_with_risks(
            tcga_model_consensus, orien_by_tcga_stats, orien_surv,
            consensus_genes, device
        )
    
    logger.info(f"  TCGA→ORIEN C-index: {cindex_tcga_on_orien:.4f}")
    
    # KM analysis for ORIEN
    km_orien = perform_km_analysis(
        risks_orien, times_orien, events_orien,
        'ORIEN', output_dir, k
    )
    logger.info(f"  ORIEN KM log-rank p: {km_orien['p_value']:.4f}")
    
    # STEP 5: TRAIN ORIEN MODEL (consensus genes only)
    logger.info("Step 5: Training ORIEN model on consensus genes...")
    orien_model_consensus = train_consensus_model(
        orien_standardized, orien_surv, consensus_genes,
        orien_params, n_epochs=50, device=device, verbose=False
    )
    
    # STEP 6: TEST ORIEN → TCGA
    logger.info("Step 6: Testing ORIEN model on TCGA...")
    
    # Standardize TCGA using ORIEN statistics
    tcga_by_orien_stats = pd.DataFrame(
        (tcga_expr_consensus.values - orien_mean_consensus.reshape(-1, 1)) / 
        (orien_std_consensus.reshape(-1, 1) + 1e-8),
        index=consensus_genes,
        columns=tcga_expr_consensus.columns
    )
    
    cindex_orien_on_tcga, risks_tcga, times_tcga, events_tcga = \
        evaluate_consensus_model_with_risks(
            orien_model_consensus, tcga_by_orien_stats, tcga_surv,
            consensus_genes, device
        )
    
    logger.info(f"  ORIEN→TCGA C-index: {cindex_orien_on_tcga:.4f}")
    
    # KM analysis for TCGA
    km_tcga = perform_km_analysis(
        risks_tcga, times_tcga, events_tcga,
        'TCGA', output_dir, k
    )
    logger.info(f"  TCGA KM log-rank p: {km_tcga['p_value']:.4f}")
    
    # STEP 7: COMPUTE SUMMARY METRICS
    avg_cindex = (cindex_tcga_on_orien + cindex_orien_on_tcga) / 2
    geom_cindex = np.sqrt(cindex_tcga_on_orien * cindex_orien_on_tcga)
    min_cindex = min(cindex_tcga_on_orien, cindex_orien_on_tcga)
    
    logger.info(f"\n  Summary:")
    logger.info(f"    Average C-index: {avg_cindex:.4f}")
    logger.info(f"    Geometric mean C-index: {geom_cindex:.4f}")
    logger.info(f"    Minimum C-index: {min_cindex:.4f}")
    
    # Save risk scores
    risk_dir = output_dir / f'risk_scores_k{k}'
    risk_dir.mkdir(exist_ok=True)
    
    # TCGA risk scores
    pd.DataFrame({
        'sample_id': tcga_surv.index,
        'risk_score': risks_tcga,
        'time': times_tcga,
        'event': events_tcga
    }).to_csv(risk_dir / 'tcga_risk_scores.csv', index=False)
    
    # ORIEN risk scores
    pd.DataFrame({
        'sample_id': orien_surv.index,
        'risk_score': risks_orien,
        'time': times_orien,
        'event': events_orien
    }).to_csv(risk_dir / 'orien_risk_scores.csv', index=False)
    
    return {
        'k': k,
        'n_consensus': n_consensus,
        'overlap_pct': overlap_pct,
        'consensus_genes': consensus_genes,
        'tcga_on_orien_cindex': cindex_tcga_on_orien,
        'orien_on_tcga_cindex': cindex_orien_on_tcga,
        'avg_cindex': avg_cindex,
        'geom_cindex': geom_cindex,
        'min_cindex': min_cindex,
        'orien_km_pvalue': km_orien['p_value'],
        'tcga_km_pvalue': km_tcga['p_value'],
        'orien_km_stats': km_orien,
        'tcga_km_stats': km_tcga
    }


def plot_results(results_df: pd.DataFrame, output_dir: Path):
    """Generate comprehensive visualization of results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: C-index vs k
    ax1 = axes[0, 0]
    ax1.plot(results_df['k'], results_df['tcga_on_orien_cindex'], 
             'o-', label='TCGA→ORIEN', linewidth=2, markersize=8)
    ax1.plot(results_df['k'], results_df['orien_on_tcga_cindex'], 
             's-', label='ORIEN→TCGA', linewidth=2, markersize=8)
    ax1.plot(results_df['k'], results_df['avg_cindex'], 
             '^-', label='Average', linewidth=2, markersize=8, color='black')
    
    # Mark optimal
    optimal_idx = results_df['avg_cindex'].idxmax()
    optimal_k = results_df.loc[optimal_idx, 'k']
    optimal_c = results_df.loc[optimal_idx, 'avg_cindex']
    ax1.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.5)
    ax1.text(optimal_k, optimal_c + 0.01, f'k={optimal_k}',
             ha='center', fontsize=9, color='red')
    
    ax1.axhline(y=0.6, color='green', linestyle='--', alpha=0.3)
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.3)
    ax1.set_xlabel('k (Top genes selected)', fontsize=11)
    ax1.set_ylabel('C-index', fontsize=11)
    ax1.set_title('Cross-Cohort C-index vs k', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Number of consensus genes vs k
    ax2 = axes[0, 1]
    ax2.plot(results_df['k'], results_df['n_consensus'], 
             'o-', linewidth=2, markersize=8, color='steelblue')
    ax2.set_xlabel('k (Top genes selected)', fontsize=11)
    ax2.set_ylabel('Number of Consensus Genes', fontsize=11)
    ax2.set_title('Consensus Gene Count vs k', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Overlap percentage vs k
    ax3 = axes[1, 0]
    ax3.plot(results_df['k'], results_df['overlap_pct'], 
             'o-', linewidth=2, markersize=8, color='orange')
    ax3.axhline(y=20, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('k (Top genes selected)', fontsize=11)
    ax3.set_ylabel('Overlap Percentage (%)', fontsize=11)
    ax3.set_title('Gene Overlap vs k', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: KM p-values vs k
    ax4 = axes[1, 1]
    ax4.plot(results_df['k'], results_df['tcga_km_pvalue'], 
             'o-', label='TCGA', linewidth=2, markersize=8)
    ax4.plot(results_df['k'], results_df['orien_km_pvalue'], 
             's-', label='ORIEN', linewidth=2, markersize=8)
    ax4.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='p=0.05')
    ax4.set_xlabel('k (Top genes selected)', fontsize=11)
    ax4.set_ylabel('Log-rank p-value', fontsize=11)
    ax4.set_title('Survival Stratification vs k', fontweight='bold')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'consensus_validation_summary.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved: {output_dir / 'consensus_validation_summary.png'}")


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Consensus-based biomarker validation'
    )
    parser.add_argument('--tcga_model', type=str, required=True)
    parser.add_argument('--orien_model', type=str, required=True)
    parser.add_argument('--tcga_params', type=str, required=True)
    parser.add_argument('--orien_params', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--k_values', type=int, nargs='+', default=K_VALUES)
    parser.add_argument('--cox_genes', type=str, default=None)
    
    args = parser.parse_args()
    
    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"results/consensus_validation_{timestamp}"
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n{'='*70}")
    logger.info("CONSENSUS-BASED BIOMARKER VALIDATION")
    logger.info(f"{'='*70}")
    logger.info(f"Testing k values: {args.k_values}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"{'='*70}\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}\n")
    
    # ============================================================
    # LOAD DATA
    # ============================================================
    
    logger.info("Loading data...")
    consensus_genes = load_consensus_genes(CONSENSUS_GENES_FILE)
    
    tcga_expr = pd.read_csv("data/raw/tcga_batch_corrected_2sv.csv", index_col=0)
    orien_expr = pd.read_csv("data/raw/orien_batch_corrected.csv", index_col=0)
    
    surv_tcga = pd.read_csv("data/processed/surv_tcga_harmonized.csv", index_col=0)
    surv_orien = pd.read_csv("data/processed/surv_orien_harmonized.csv", index_col=0)
    
    # Filter and compute global statistics (for original 308 genes)
    tcga_genes_available = [g for g in consensus_genes if g in tcga_expr.index]
    tcga_expr_full = tcga_expr.loc[tcga_genes_available, :]
    tcga_mean_full = tcga_expr_full.mean(axis=1).values
    tcga_std_full = tcga_expr_full.std(axis=1).values
    
    orien_genes_available = [g for g in consensus_genes if g in orien_expr.index]
    orien_expr_full = orien_expr.loc[orien_genes_available, :]
    orien_mean_full = orien_expr_full.mean(axis=1).values
    orien_std_full = orien_expr_full.std(axis=1).values
    
    # Standardize full data (for importance extraction)
    tcga_standardized_full = pd.DataFrame(
        (tcga_expr_full.values - tcga_mean_full.reshape(-1, 1)) / 
        (tcga_std_full.reshape(-1, 1) + 1e-8),
        index=tcga_expr_full.index,
        columns=tcga_expr_full.columns
    )
    
    orien_standardized_full = pd.DataFrame(
        (orien_expr_full.values - orien_mean_full.reshape(-1, 1)) / 
        (orien_std_full.reshape(-1, 1) + 1e-8),
        index=orien_expr_full.index,
        columns=orien_expr_full.columns
    )
    
    gene_names = tcga_standardized_full.index.tolist()
    
    logger.info(f"Loaded {len(gene_names)} genes")
    logger.info(f"TCGA: {tcga_expr_full.shape[1]} samples")
    logger.info(f"ORIEN: {orien_expr_full.shape[1]} samples\n")
    
    # ============================================================
    # LOAD MODELS & EXTRACT IMPORTANCE
    # ============================================================
    
    logger.info("Loading trained models for importance extraction...")
    
    with open(args.tcga_params, 'r') as f:
        tcga_params = json.load(f)
    with open(args.orien_params, 'r') as f:
        orien_params = json.load(f)
    
    # TCGA model
    tcga_hidden = [tcga_params.get('layer1_size', 256)]
    tcga_model_full = ElasticDeepSurv(
        n_features=len(gene_names),
        hidden_sizes=tcga_hidden,
        dropout=tcga_params.get('dropout', 0.3),
        activation=tcga_params.get('activation', 'relu'),
        batch_norm=tcga_params.get('batch_norm', False),
        weight_init=tcga_params.get('weight_init', 'xavier_normal'),
        l1_ratio=tcga_params.get('l1_ratio', 0.9),
        alpha=tcga_params.get('alpha', 0.001)
    )
    tcga_model_full.load_state_dict(torch.load(args.tcga_model, weights_only=False))
    tcga_importance = compute_l2_feature_importance(tcga_model_full)
    
    # ORIEN model
    orien_hidden = [orien_params.get('layer1_size', 256)]
    orien_model_full = ElasticDeepSurv(
        n_features=len(gene_names),
        hidden_sizes=orien_hidden,
        dropout=orien_params.get('dropout', 0.3),
        activation=orien_params.get('activation', 'relu'),
        batch_norm=orien_params.get('batch_norm', False),
        weight_init=orien_params.get('weight_init', 'xavier_normal'),
        l1_ratio=orien_params.get('l1_ratio', 0.7),
        alpha=orien_params.get('alpha', 0.001)
    )
    orien_model_full.load_state_dict(torch.load(args.orien_model, weights_only=False))
    orien_importance = compute_l2_feature_importance(orien_model_full)
    
    logger.info("Importance extraction complete.\n")
    
    # ============================================================
    # TEST EACH K VALUE
    # ============================================================
    
    all_results = []
    
    for k in args.k_values:
        result = test_consensus_at_k(
            k,
            tcga_standardized_full, surv_tcga,
            orien_standardized_full, surv_orien,
            tcga_importance, orien_importance,
            gene_names,
            tcga_params, orien_params,
            tcga_mean_full, tcga_std_full,
            orien_mean_full, orien_std_full,
            output_dir,
            device
        )
        
        if result is not None:
            all_results.append(result)
    
    # ============================================================
    # ANALYZE RESULTS
    # ============================================================
    
    logger.info(f"\n{'='*70}")
    logger.info("RESULTS SUMMARY")
    logger.info(f"{'='*70}\n")
    
    # Create results DataFrame
    results_df = pd.DataFrame([{
        'k': r['k'],
        'n_consensus': r['n_consensus'],
        'overlap_pct': r['overlap_pct'],
        'tcga_on_orien_cindex': r['tcga_on_orien_cindex'],
        'orien_on_tcga_cindex': r['orien_on_tcga_cindex'],
        'avg_cindex': r['avg_cindex'],
        'geom_cindex': r['geom_cindex'],
        'min_cindex': r['min_cindex'],
        'tcga_km_pvalue': r['tcga_km_pvalue'],
        'orien_km_pvalue': r['orien_km_pvalue']
    } for r in all_results])
    
    logger.info(results_df.to_string(index=False))
    
    # Find optimal k
    optimal_idx = results_df['avg_cindex'].idxmax()
    optimal_result = all_results[optimal_idx]
    
    logger.info(f"\n{'='*70}")
    logger.info(f"OPTIMAL CONFIGURATION")
    logger.info(f"{'='*70}")
    logger.info(f"k = {optimal_result['k']}")
    logger.info(f"Consensus genes: {optimal_result['n_consensus']} "
                f"({optimal_result['overlap_pct']:.1f}% overlap)")
    logger.info(f"Average C-index: {optimal_result['avg_cindex']:.4f}")
    logger.info(f"  TCGA→ORIEN: {optimal_result['tcga_on_orien_cindex']:.4f}")
    logger.info(f"  ORIEN→TCGA: {optimal_result['orien_on_tcga_cindex']:.4f}")
    logger.info(f"KM stratification:")
    logger.info(f"  TCGA p-value: {optimal_result['tcga_km_pvalue']:.4f}")
    logger.info(f"  ORIEN p-value: {optimal_result['orien_km_pvalue']:.4f}")
    logger.info(f"{'='*70}\n")
    
    # ============================================================
    # SAVE RESULTS
    # ============================================================
    
    # Summary table
    results_df.to_csv(output_dir / 'consensus_validation_results.csv', index=False)
    
    # Optimal consensus genes
    with open(output_dir / f'optimal_consensus_genes_k{optimal_result["k"]}.txt', 'w') as f:
        f.write('\n'.join(optimal_result['consensus_genes']))
    
    # Full results JSON
    with open(output_dir / 'full_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Plots
    plot_results(results_df, output_dir)
    
    # Compare with Cox if available
    if args.cox_genes and Path(args.cox_genes).exists():
        logger.info("Comparison with Cox Elastic Net (Chapter 2):")
        cox_genes = load_consensus_genes(args.cox_genes)
        logger.info(f"  Cox baseline: {len(cox_genes)} genes")
        logger.info(f"  DeepSurv optimal: {optimal_result['n_consensus']} genes")
        
        # Check overlap
        cox_set = set(cox_genes)
        deepsurv_set = set(optimal_result['consensus_genes'])
        overlap = cox_set & deepsurv_set
        logger.info(f"  Methods overlap: {len(overlap)} genes ({len(overlap)/len(cox_genes)*100:.1f}%)")
        
        if len(overlap) > 0:
            logger.info(f"  Shared genes: {', '.join(sorted(list(overlap)))}")
    
    logger.info(f"\n{'='*70}")
    logger.info("ANALYSIS COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"{'='*70}\n")
    
    return results_df, optimal_result


if __name__ == "__main__":
    results_df, optimal_result = main()