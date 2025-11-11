"""
Cross-Cohort Validation and Biomarker Overlap Analysis
======================================================

This script performs comprehensive cross-cohort validation:
1. Evaluates models trained on one cohort, tested on another
2. Analyzes gene overlap at multiple k values
3. Compares with Chapter 2 Cox elastic net baseline
4. Generates publication-ready tables and figures

NO RETRAINING - uses existing trained models.
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

from src.data.dataset import SurvivalDataset
from torch.utils.data import DataLoader
from src.models.elastic_deepsurv import ElasticDeepSurv
from src.utils.batch_samplers import StratifiedBatchSampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

K_VALUES = [10, 15, 20, 25, 30, 40, 50, 75, 100]  # Range of k to test
CONSENSUS_GENES_FILE = "data/raw/consensus_genes_308.txt"
COX_BASELINE_FILE = "data/raw/cox_consensus_genes_20.txt"  # Your Chapter 2 results


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_consensus_genes(filepath: str) -> List[str]:
    """Load gene list from file."""
    with open(filepath, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]
    return genes


def load_model_and_genes(
    model_path: str,
    n_features: int,
    hidden_sizes: List[int],
    best_params: dict
) -> ElasticDeepSurv:
    """Load a trained model."""
    model = ElasticDeepSurv(
        n_features=n_features,
        hidden_sizes=hidden_sizes,
        dropout=best_params.get('dropout', 0.3),
        activation=best_params.get('activation', 'relu'),
        batch_norm=best_params.get('batch_norm', False),
        weight_init=best_params.get('weight_init', 'xavier_normal'),
        l1_ratio=best_params.get('l1_ratio', 0.9),
        alpha=best_params.get('alpha', 0.001)
    )
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    return model


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


def evaluate_model_cross_cohort(
    model: ElasticDeepSurv,
    expr_standardized: pd.DataFrame,
    surv: pd.DataFrame,
    device: str = 'cuda'
) -> float:
    """
    Evaluate model on a different cohort.
    
    Args:
        model: Trained model
        expr_standardized: Standardized expression data (genes × samples)
        surv: Survival data
        device: Device to use
        
    Returns:
        C-index on the test cohort
    """
    model.to(device)
    model.eval()
    
    # Create dataset
    dataset = SurvivalDataset(expr_standardized, surv)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    all_risks = []
    all_times = []
    all_events = []
    
    with torch.no_grad():
        for batch in loader:
            expr_batch = batch['feature'].to(device)
            time_batch = batch['time'].cpu().numpy()
            event_batch = batch['event'].cpu().numpy()
            
            # Get risk predictions
            risk = model(expr_batch).cpu().numpy().flatten()
            
            all_risks.extend(risk)
            all_times.extend(time_batch)
            all_events.extend(event_batch)
    
    # Compute C-index
    cindex = concordance_index(all_times, -np.array(all_risks), all_events)
    
    return cindex


def compute_overlap_analysis(
    tcga_importance: np.ndarray,
    orien_importance: np.ndarray,
    gene_names: List[str],
    k_values: List[int]
) -> pd.DataFrame:
    """
    Compute gene overlap at different k values.
    
    Returns:
        DataFrame with overlap statistics
    """
    results = []
    
    for k in k_values:
        tcga_top_k = set(get_top_k_genes(tcga_importance, gene_names, k))
        orien_top_k = set(get_top_k_genes(orien_importance, gene_names, k))
        
        overlap = tcga_top_k & orien_top_k
        overlap_count = len(overlap)
        overlap_pct = (overlap_count / k) * 100
        
        results.append({
            'k': k,
            'overlap_count': overlap_count,
            'overlap_pct': overlap_pct,
            'tcga_genes': sorted(list(tcga_top_k)),
            'orien_genes': sorted(list(orien_top_k)),
            'consensus_genes': sorted(list(overlap))
        })
    
    return pd.DataFrame(results)


def compare_with_cox_baseline(
    overlap_df: pd.DataFrame,
    cox_genes: List[str],
    gene_names: List[str]
) -> pd.DataFrame:
    """
    Compare DeepSurv gene selections with Cox baseline.
    
    Returns:
        DataFrame with comparison statistics
    """
    cox_genes_set = set(cox_genes)
    k_cox = len(cox_genes)
    
    # Find the row with k closest to Cox baseline
    cox_row = overlap_df.iloc[(overlap_df['k'] - k_cox).abs().argmin()]
    
    tcga_genes = set(cox_row['tcga_genes'])
    orien_genes = set(cox_row['orien_genes'])
    consensus_genes = set(cox_row['consensus_genes'])
    
    comparison = {
        'method': ['Cox EN (Chapter 2)', 'DeepSurv TCGA', 'DeepSurv ORIEN', 'DeepSurv Consensus'],
        'n_genes': [k_cox, k_cox, k_cox, len(consensus_genes)],
        'overlap_with_cox': [
            k_cox,  # Cox overlaps 100% with itself
            len(tcga_genes & cox_genes_set),
            len(orien_genes & cox_genes_set),
            len(consensus_genes & cox_genes_set)
        ]
    }
    
    comparison_df = pd.DataFrame(comparison)
    comparison_df['overlap_pct'] = (comparison_df['overlap_with_cox'] / k_cox * 100).round(1)
    
    return comparison_df


# ============================================================
# PLOTTING FUNCTIONS
# ============================================================

def plot_overlap_curve(overlap_df: pd.DataFrame, output_dir: Path):
    """Plot overlap percentage vs k."""
    plt.figure(figsize=(10, 6))
    
    plt.plot(overlap_df['k'], overlap_df['overlap_pct'], 'o-', linewidth=2, markersize=8)
    plt.axhline(y=50, color='r', linestyle='--', alpha=0.3, label='50% overlap')
    plt.axhline(y=30, color='orange', linestyle='--', alpha=0.3, label='30% overlap')
    
    plt.xlabel('Number of Top Genes (k)', fontsize=12)
    plt.ylabel('Overlap Percentage (%)', fontsize=12)
    plt.title('Gene Selection Stability: TCGA ∩ ORIEN', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'overlap_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved: {output_dir / 'overlap_curve.png'}")


def plot_importance_comparison(
    tcga_importance: np.ndarray,
    orien_importance: np.ndarray,
    gene_names: List[str],
    output_dir: Path,
    top_k: int = 50
):
    """Plot importance score comparison for top genes."""
    # Get top 50 genes from each cohort
    tcga_top_idx = np.argsort(tcga_importance)[::-1][:top_k]
    orien_top_idx = np.argsort(orien_importance)[::-1][:top_k]
    
    # Union of top genes
    top_genes_idx = sorted(set(tcga_top_idx) | set(orien_top_idx))
    
    # Create DataFrame
    df = pd.DataFrame({
        'gene': [gene_names[i] for i in top_genes_idx],
        'tcga_importance': tcga_importance[top_genes_idx],
        'orien_importance': orien_importance[top_genes_idx]
    })
    
    plt.figure(figsize=(10, 10))
    plt.scatter(df['tcga_importance'], df['orien_importance'], alpha=0.6, s=50)
    
    # Add diagonal line
    max_val = max(df['tcga_importance'].max(), df['orien_importance'].max())
    plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Perfect agreement')
    
    plt.xlabel('TCGA Importance Score', fontsize=12)
    plt.ylabel('ORIEN Importance Score', fontsize=12)
    plt.title(f'Feature Importance Correlation (Top {top_k} genes from each cohort)', 
              fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add correlation
    corr = np.corrcoef(df['tcga_importance'], df['orien_importance'])[0, 1]
    plt.text(0.05, 0.95, f'Pearson r = {corr:.3f}', 
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'importance_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved: {output_dir / 'importance_correlation.png'}")


def plot_cross_cohort_summary(
    tcga_on_orien: float,
    orien_on_tcga: float,
    tcga_val: float,
    orien_val: float,
    output_dir: Path
):
    """Plot cross-cohort validation summary."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['TCGA→ORIEN', 'ORIEN→TCGA']
    cross_cohort = [tcga_on_orien, orien_on_tcga]
    validation = [tcga_val, orien_val]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax.bar(x - width/2, cross_cohort, width, label='Cross-Cohort Test', color='steelblue')
    ax.bar(x + width/2, validation, width, label='Within-Cohort Validation', color='coral')
    
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random (0.5)')
    ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Good (0.7)')
    
    ax.set_ylabel('C-index', fontsize=12)
    ax.set_title('Cross-Cohort Validation Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.set_ylim([0.4, 0.8])
    
    # Add values on bars
    for i, (cc, val) in enumerate(zip(cross_cohort, validation)):
        ax.text(i - width/2, cc + 0.01, f'{cc:.3f}', ha='center', va='bottom', fontsize=10)
        ax.text(i + width/2, val + 0.01, f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cross_cohort_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved: {output_dir / 'cross_cohort_summary.png'}")


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Cross-cohort validation and biomarker overlap analysis'
    )
    parser.add_argument('--tcga_model', type=str, required=True,
                       help='Path to trained TCGA model (.pth)')
    parser.add_argument('--orien_model', type=str, required=True,
                       help='Path to trained ORIEN model (.pth)')
    parser.add_argument('--tcga_params', type=str, required=True,
                       help='Path to TCGA hyperparameters (best_params.json)')
    parser.add_argument('--orien_params', type=str, required=True,
                       help='Path to ORIEN hyperparameters (best_params.json)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory')
    parser.add_argument('--cox_genes', type=str, default=None,
                       help='Path to Cox baseline genes (optional)')
    
    args = parser.parse_args()
    
    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"results/cross_cohort_analysis_{timestamp}"
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n{'='*70}")
    logger.info("CROSS-COHORT VALIDATION ANALYSIS")
    logger.info(f"{'='*70}\n")
    
    # ============================================================
    # LOAD DATA
    # ============================================================
    
    logger.info("Loading consensus genes...")
    consensus_genes = load_consensus_genes(CONSENSUS_GENES_FILE)
    logger.info(f"Loaded {len(consensus_genes)} consensus genes")
    
    logger.info("\nLoading expression data...")
    tcga_expr = pd.read_csv("data/raw/tcga_batch_corrected_2sv.csv", index_col=0)
    orien_expr = pd.read_csv("data/raw/orien_batch_corrected.csv", index_col=0)
    
    logger.info("Loading survival data...")
    surv_tcga = pd.read_csv("data/processed/surv_tcga_harmonized.csv", index_col=0)
    surv_orien = pd.read_csv("data/processed/surv_orien_harmonized.csv", index_col=0)
    
    # Filter and standardize
    logger.info("\nPreprocessing TCGA...")
    tcga_genes_available = [g for g in consensus_genes if g in tcga_expr.index]
    tcga_expr_filtered = tcga_expr.loc[tcga_genes_available, :]
    tcga_mean = tcga_expr_filtered.mean(axis=1).values.reshape(-1, 1)
    tcga_std = tcga_expr_filtered.std(axis=1).values.reshape(-1, 1)
    tcga_standardized = pd.DataFrame(
        (tcga_expr_filtered.values - tcga_mean) / (tcga_std + 1e-8),
        index=tcga_expr_filtered.index,
        columns=tcga_expr_filtered.columns
    )
    
    logger.info("Preprocessing ORIEN...")
    orien_genes_available = [g for g in consensus_genes if g in orien_expr.index]
    orien_expr_filtered = orien_expr.loc[orien_genes_available, :]
    orien_mean = orien_expr_filtered.mean(axis=1).values.reshape(-1, 1)
    orien_std = orien_expr_filtered.std(axis=1).values.reshape(-1, 1)
    orien_standardized = pd.DataFrame(
        (orien_expr_filtered.values - orien_mean) / (orien_std + 1e-8),
        index=orien_expr_filtered.index,
        columns=orien_expr_filtered.columns
    )
    
    gene_names = tcga_standardized.index.tolist()
    assert gene_names == orien_standardized.index.tolist(), "Gene order mismatch!"
    
    # ============================================================
    # LOAD MODELS
    # ============================================================
    
    logger.info("\nLoading trained models...")
    
    with open(args.tcga_params, 'r') as f:
        tcga_params = json.load(f)
    with open(args.orien_params, 'r') as f:
        orien_params = json.load(f)
    
    # TCGA model
    tcga_hidden_sizes = [tcga_params['layer1_size']] if 'layer1_size' in tcga_params else [256]
    tcga_model = load_model_and_genes(
        args.tcga_model,
        n_features=len(gene_names),
        hidden_sizes=tcga_hidden_sizes,
        best_params=tcga_params
    )
    logger.info(f"Loaded TCGA model: {len(gene_names)} → {tcga_hidden_sizes} → 1")
    
    # ORIEN model
    orien_hidden_sizes = [orien_params['layer1_size']] if 'layer1_size' in orien_params else [256]
    orien_model = load_model_and_genes(
        args.orien_model,
        n_features=len(gene_names),
        hidden_sizes=orien_hidden_sizes,
        best_params=orien_params
    )
    logger.info(f"Loaded ORIEN model: {len(gene_names)} → {orien_hidden_sizes} → 1")
    
    # ============================================================
    # EXTRACT FEATURE IMPORTANCE
    # ============================================================
    
    logger.info("\nComputing feature importance...")
    tcga_importance = compute_l2_feature_importance(tcga_model)
    orien_importance = compute_l2_feature_importance(orien_model)
    
    logger.info(f"TCGA importance range: [{tcga_importance.min():.4f}, {tcga_importance.max():.4f}]")
    logger.info(f"ORIEN importance range: [{orien_importance.min():.4f}, {orien_importance.max():.4f}]")
    
    # ============================================================
    # OVERLAP ANALYSIS
    # ============================================================
    
    logger.info(f"\n{'='*70}")
    logger.info("GENE OVERLAP ANALYSIS")
    logger.info(f"{'='*70}\n")
    
    overlap_df = compute_overlap_analysis(
        tcga_importance, orien_importance, gene_names, K_VALUES
    )
    
    logger.info("Overlap statistics:")
    logger.info("\n" + overlap_df[['k', 'overlap_count', 'overlap_pct']].to_string(index=False))
    
    # Save detailed results
    overlap_df[['k', 'overlap_count', 'overlap_pct']].to_csv(
        output_dir / 'overlap_summary.csv', index=False
    )
    
    # Save gene lists for key k values
    for k in [10, 20, 30]:
        row = overlap_df[overlap_df['k'] == k].iloc[0]
        
        with open(output_dir / f'tcga_top{k}_genes.txt', 'w') as f:
            f.write('\n'.join(row['tcga_genes']))
        
        with open(output_dir / f'orien_top{k}_genes.txt', 'w') as f:
            f.write('\n'.join(row['orien_genes']))
        
        with open(output_dir / f'consensus_top{k}_genes.txt', 'w') as f:
            f.write('\n'.join(row['consensus_genes']))
    
    logger.info(f"\nSaved overlap analysis to: {output_dir}")
    
    # ============================================================
    # CROSS-COHORT EVALUATION
    # ============================================================
    
    logger.info(f"\n{'='*70}")
    logger.info("CROSS-COHORT VALIDATION")
    logger.info(f"{'='*70}\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    
    logger.info("\nEvaluating TCGA model on ORIEN...")
    tcga_on_orien = evaluate_model_cross_cohort(
        tcga_model, orien_standardized, surv_orien, device
    )
    logger.info(f"TCGA→ORIEN C-index: {tcga_on_orien:.4f}")
    
    logger.info("\nEvaluating ORIEN model on TCGA...")
    orien_on_tcga = evaluate_model_cross_cohort(
        orien_model, tcga_standardized, surv_tcga, device
    )
    logger.info(f"ORIEN→TCGA C-index: {orien_on_tcga:.4f}")
    
    # ============================================================
    # COMPARE WITH COX BASELINE
    # ============================================================
    
    if args.cox_genes and Path(args.cox_genes).exists():
        logger.info(f"\n{'='*70}")
        logger.info("COMPARISON WITH COX ELASTIC NET (CHAPTER 2)")
        logger.info(f"{'='*70}\n")
        
        cox_genes = load_consensus_genes(args.cox_genes)
        logger.info(f"Loaded {len(cox_genes)} Cox consensus genes")
        
        comparison_df = compare_with_cox_baseline(overlap_df, cox_genes, gene_names)
        logger.info("\n" + comparison_df.to_string(index=False))
        
        comparison_df.to_csv(output_dir / 'cox_comparison.csv', index=False)
    
    # ============================================================
    # GENERATE PLOTS
    # ============================================================
    
    logger.info(f"\n{'='*70}")
    logger.info("GENERATING VISUALIZATIONS")
    logger.info(f"{'='*70}\n")
    
    plot_overlap_curve(overlap_df, output_dir)
    plot_importance_comparison(tcga_importance, orien_importance, gene_names, output_dir)
    
    # For plotting, we need validation C-indices (from training logs or params)
    tcga_val_cindex = 0.6814  # From your logs
    orien_val_cindex = 0.5821  # From your logs
    
    plot_cross_cohort_summary(
        tcga_on_orien, orien_on_tcga,
        tcga_val_cindex, orien_val_cindex,
        output_dir
    )
    
    # ============================================================
    # SUMMARY REPORT
    # ============================================================
    
    logger.info(f"\n{'='*70}")
    logger.info("SUMMARY")
    logger.info(f"{'='*70}\n")
    
    summary = {
        'analysis_date': datetime.now().isoformat(),
        'n_genes': len(gene_names),
        'cross_cohort_validation': {
            'tcga_on_orien_cindex': float(tcga_on_orien),
            'orien_on_tcga_cindex': float(orien_on_tcga),
            'tcga_validation_cindex': float(tcga_val_cindex),
            'orien_validation_cindex': float(orien_val_cindex)
        },
        'overlap_analysis': {
            f'k{row["k"]}': {
                'overlap_count': int(row['overlap_count']),
                'overlap_pct': float(row['overlap_pct'])
            }
            for _, row in overlap_df.iterrows()
        },
        'interpretation': {
            'cross_cohort_performance': (
                'Good' if min(tcga_on_orien, orien_on_tcga) >= 0.60 else
                'Moderate' if min(tcga_on_orien, orien_on_tcga) >= 0.55 else
                'Poor'
            ),
            'biomarker_stability': (
                'High' if overlap_df[overlap_df['k']==20]['overlap_pct'].values[0] >= 40 else
                'Moderate' if overlap_df[overlap_df['k']==20]['overlap_pct'].values[0] >= 25 else
                'Low'
            )
        }
    }
    
    with open(output_dir / 'SUMMARY.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("Cross-Cohort Performance:")
    logger.info(f"  TCGA→ORIEN: C-index = {tcga_on_orien:.4f}")
    logger.info(f"  ORIEN→TCGA: C-index = {orien_on_tcga:.4f}")
    logger.info(f"  Average: {(tcga_on_orien + orien_on_tcga)/2:.4f}")
    
    logger.info(f"\nBiomarker Overlap (k=20):")
    k20_overlap = overlap_df[overlap_df['k']==20]['overlap_pct'].values[0]
    logger.info(f"  {k20_overlap:.1f}% ({int(overlap_df[overlap_df['k']==20]['overlap_count'].values[0])}/20 genes)")
    
    logger.info(f"\nOverall Assessment:")
    logger.info(f"  Cross-cohort performance: {summary['interpretation']['cross_cohort_performance']}")
    logger.info(f"  Biomarker stability: {summary['interpretation']['biomarker_stability']}")
    
    logger.info(f"\n{'='*70}")
    logger.info(f"ANALYSIS COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"{'='*70}\n")
    
    return summary


if __name__ == "__main__":
    summary = main()