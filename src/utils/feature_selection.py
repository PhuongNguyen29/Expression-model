"""
Feature selection for neural survival models with soft sparsity.

Based on:
- Scardapane et al. (2017), "Group Sparse Regularization for Deep Neural Networks"
- Yousefi et al. (2017), "Predicting clinical outcomes from large scale cancer genomic profiles"
- Katzman et al. (2018), "DeepSurv: personalized treatment recommender system"

Key insight: Neural networks with L1 produce soft sparsity (small weights, not exact zeros).
We use magnitude-based thresholding instead of exact zero counting.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


def extract_first_layer_weights(model: nn.Module) -> torch.Tensor:
    """
    Extract weights from first layer (input → first hidden).
    
    Args:
        model: Trained neural network
        
    Returns:
        weights: Tensor of shape [n_hidden, n_genes]
    """
    # Get first Linear layer
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            return module.weight.data
    
    raise ValueError("Model does not contain Linear layer")


def compute_gene_importance_l2(
    model: nn.Module,
    method: str = 'l2_norm'
) -> np.ndarray:
    """
    Compute importance score for each input gene.
    
    Args:
        model: Trained model
        method: How to aggregate across hidden units
            - 'l2_norm': ||weights||_2 across hidden units (default, most stable)
            - 'l1_norm': ||weights||_1 across hidden units (Yousefi 2017)
            - 'max_abs': max|weight| across hidden units (most activated path)
            - 'mean_abs': mean|weight| across hidden units
            
    Returns:
        importance: Array of shape [n_genes] with importance scores
        
    References:
        - Yousefi et al. (2017): Used sum of absolute weights
        - Katzman et al. (2018): Used connection weights method
    """
    weights = extract_first_layer_weights(model)  # [n_hidden, n_genes]
    weights_np = weights.cpu().numpy()
    
    if method == 'l2_norm':
        # L2 norm across hidden units (default)
        importance = np.linalg.norm(weights_np, axis=0)
        
    elif method == 'l1_norm':
        # L1 norm (sum of absolute values)
        importance = np.sum(np.abs(weights_np), axis=0)
        
    elif method == 'max_abs':
        # Maximum absolute weight
        importance = np.max(np.abs(weights_np), axis=0)
        
    elif method == 'mean_abs':
        # Mean absolute weight
        importance = np.mean(np.abs(weights_np), axis=0)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return importance


def select_features_percentile(
    importance_scores: np.ndarray,
    gene_names: List[str],
    percentile: float = 95.0
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Select features using percentile threshold.
    
    Args:
        importance_scores: Importance score per gene
        gene_names: List of gene names
        percentile: Keep genes above this percentile (default: 95 = top 5%)
        
    Returns:
        selected_indices: Indices of selected genes
        selected_scores: Importance scores of selected genes
        threshold: The percentile threshold value
        
    Example:
        percentile=95 → Keep top 5% genes
        percentile=90 → Keep top 10% genes
    """
    threshold = np.percentile(importance_scores, percentile)
    selected_mask = importance_scores >= threshold
    selected_indices = np.where(selected_mask)[0]
    selected_scores = importance_scores[selected_indices]
    
    n_selected = len(selected_indices)
    n_total = len(importance_scores)
    pct_selected = 100 * n_selected / n_total
    
    logger.info(f"Feature selection (percentile={percentile}):")
    logger.info(f"  Threshold: {threshold:.6f}")
    logger.info(f"  Selected: {n_selected}/{n_total} genes ({pct_selected:.1f}%)")
    logger.info(f"  Score range: [{selected_scores.min():.6f}, {selected_scores.max():.6f}]")
    
    return selected_indices, selected_scores, threshold


def select_features_top_k(
    importance_scores: np.ndarray,
    gene_names: List[str],
    top_k: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select top-k features by importance.
    
    Args:
        importance_scores: Importance score per gene
        gene_names: List of gene names
        top_k: Number of top genes to select
        
    Returns:
        selected_indices: Indices of top-k genes
        selected_scores: Importance scores of top-k genes
    """
    # Sort by importance (descending)
    sorted_indices = np.argsort(importance_scores)[::-1]
    
    # Take top-k
    selected_indices = sorted_indices[:top_k]
    selected_scores = importance_scores[selected_indices]
    
    logger.info(f"Feature selection (top-{top_k}):")
    logger.info(f"  Score range: [{selected_scores.min():.6f}, {selected_scores.max():.6f}]")
    
    return selected_indices, selected_scores


def get_selected_gene_names(
    selected_indices: np.ndarray,
    gene_names: List[str],
    importance_scores: np.ndarray
) -> pd.DataFrame:
    """
    Create DataFrame of selected genes with scores.
    
    Args:
        selected_indices: Indices of selected genes
        gene_names: All gene names
        importance_scores: All importance scores
        
    Returns:
        DataFrame with columns: gene_name, importance, rank
    """
    selected_genes = [gene_names[i] for i in selected_indices]
    selected_scores = importance_scores[selected_indices]
    
    # Sort by importance
    sort_order = np.argsort(selected_scores)[::-1]
    
    df = pd.DataFrame({
        'gene_name': [selected_genes[i] for i in sort_order],
        'importance': selected_scores[sort_order],
        'original_index': selected_indices[sort_order],
        'rank': np.arange(1, len(selected_genes) + 1)
    })
    
    return df


def compute_bidirectional_consensus(
    model1_importance: np.ndarray,
    model2_importance: np.ndarray,
    gene_names: List[str],
    selection_method: str = 'percentile',
    percentile: float = 95.0,
    top_k: int = 100
) -> Dict:
    """
    Find consensus genes between two bidirectional models.
    
    Args:
        model1_importance: Importance scores from model 1 (e.g., TCGA→ORIEN)
        model2_importance: Importance scores from model 2 (e.g., ORIEN→TCGA)
        gene_names: List of all gene names
        selection_method: 'percentile' or 'top_k'
        percentile: If using percentile method
        top_k: If using top_k method
        
    Returns:
        Dictionary with consensus analysis results
    """
    # Select genes from each model
    if selection_method == 'percentile':
        indices1, scores1, thresh1 = select_features_percentile(
            model1_importance, gene_names, percentile
        )
        indices2, scores2, thresh2 = select_features_percentile(
            model2_importance, gene_names, percentile
        )
    else:  # top_k
        indices1, scores1 = select_features_top_k(
            model1_importance, gene_names, top_k
        )
        indices2, scores2 = select_features_top_k(
            model2_importance, gene_names, top_k
        )
    
    # Get gene names
    genes1 = set([gene_names[i] for i in indices1])
    genes2 = set([gene_names[i] for i in indices2])
    
    # Compute consensus
    consensus_genes = genes1 & genes2
    union_genes = genes1 | genes2
    
    # Calculate overlap metrics
    overlap_rate = len(consensus_genes) / len(genes1) if len(genes1) > 0 else 0
    jaccard_index = len(consensus_genes) / len(union_genes) if len(union_genes) > 0 else 0
    
    # Get consensus gene details
    consensus_indices = [i for i, g in enumerate(gene_names) if g in consensus_genes]
    consensus_df = pd.DataFrame({
        'gene_name': [gene_names[i] for i in consensus_indices],
        'importance_model1': model1_importance[consensus_indices],
        'importance_model2': model2_importance[consensus_indices],
        'mean_importance': (model1_importance[consensus_indices] + 
                          model2_importance[consensus_indices]) / 2
    })
    consensus_df = consensus_df.sort_values('mean_importance', ascending=False)
    consensus_df['rank'] = np.arange(1, len(consensus_df) + 1)
    
    logger.info(f"\n{'='*60}")
    logger.info("BIDIRECTIONAL CONSENSUS ANALYSIS")
    logger.info(f"{'='*60}")
    logger.info(f"Model 1 selected: {len(genes1)} genes")
    logger.info(f"Model 2 selected: {len(genes2)} genes")
    logger.info(f"Consensus (intersection): {len(consensus_genes)} genes")
    logger.info(f"Union: {len(union_genes)} genes")
    logger.info(f"Overlap rate: {overlap_rate:.1%}")
    logger.info(f"Jaccard index: {jaccard_index:.3f}")
    logger.info(f"{'='*60}\n")
    
    return {
        'genes_model1': genes1,
        'genes_model2': genes2,
        'consensus_genes': consensus_genes,
        'union_genes': union_genes,
        'overlap_rate': overlap_rate,
        'jaccard_index': jaccard_index,
        'consensus_df': consensus_df,
        'n_model1': len(genes1),
        'n_model2': len(genes2),
        'n_consensus': len(consensus_genes)
    }


def compare_with_chapter2_biomarkers(
    neural_net_genes: set,
    chapter2_genes: List[str]
) -> Dict:
    """
    Compare neural network biomarkers with Chapter 2 linear model biomarkers.
    
    Args:
        neural_net_genes: Set of genes from neural network
        chapter2_genes: List of genes from Chapter 2 elastic net
        
    Returns:
        Dictionary with comparison metrics
    """
    chapter2_set = set(chapter2_genes)
    
    overlap = neural_net_genes & chapter2_set
    overlap_rate = len(overlap) / len(chapter2_set) if len(chapter2_set) > 0 else 0
    
    logger.info(f"\n{'='*60}")
    logger.info("COMPARISON WITH CHAPTER 2 BIOMARKERS")
    logger.info(f"{'='*60}")
    logger.info(f"Chapter 2 (linear elastic net): {len(chapter2_set)} genes")
    logger.info(f"Chapter 3 (neural network): {len(neural_net_genes)} genes")
    logger.info(f"Overlap: {len(overlap)} genes ({overlap_rate:.1%})")
    
    if len(overlap) > 0:
        logger.info(f"Overlapping genes: {sorted(list(overlap))}")
    logger.info(f"{'='*60}\n")
    
    return {
        'n_chapter2': len(chapter2_set),
        'n_chapter3': len(neural_net_genes),
        'n_overlap': len(overlap),
        'overlap_rate': overlap_rate,
        'overlapping_genes': sorted(list(overlap))
    }