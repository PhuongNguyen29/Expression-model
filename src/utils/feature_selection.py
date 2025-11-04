"""
Feature selection utilities for ElasticDeepSurv biomarker extraction.

This module provides functions for:
1. Computing feature importance from neural networks
2. Selecting top features based on importance scores
3. Computing consensus biomarkers across cohorts
4. Comparing with Chapter 2 Cox regression biomarkers

Evidence-based methods:
- L2 norm importance: Simonyan et al. (2014) "Deep Inside CNNs"
- Consensus approach: Haibe-Kains et al. (2012) "Comparison of prognostic signatures"
- Jaccard similarity: Standard set similarity measure
"""

import numpy as np
import pandas as pd
import torch
from typing import List, Tuple, Dict, Set
import logging

logger = logging.getLogger(__name__)


def compute_gene_importance_l2(model, method: str = 'l2_norm') -> np.ndarray:
    """
    Compute feature importance from ElasticDeepSurv model.
    
    Standard approach from deep learning literature:
    - Simonyan et al. (2014): "Deep Inside Convolutional Networks"
    - Uses L2 norm of first layer weights
    
    Args:
        model: Trained ElasticDeepSurv model
        method: Currently only 'l2_norm' supported
        
    Returns:
        Array of importance scores (one per input feature/gene)
    """
    if method != 'l2_norm':
        raise ValueError(f"Only 'l2_norm' method supported, got: {method}")
    
    # Get first layer weights
    # ElasticDeepSurv.network[0] is the first Linear layer
    first_layer = model.network[0]
    
    # Weights shape: (hidden_size, n_features)
    weights = first_layer.weight.data.cpu().numpy()
    
    # L2 norm across output dimension (axis=0)
    # Each gene gets one importance score
    importance = np.linalg.norm(weights, axis=0)
    
    return importance


def select_features_percentile(
    importance_scores: np.ndarray,
    gene_names: List[str],
    percentile: float = 95.0
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Select top features based on percentile threshold.
    
    Args:
        importance_scores: Array of importance values
        gene_names: List of gene names (for logging)
        percentile: Percentile threshold (e.g., 95.0 for top 5%)
        
    Returns:
        Tuple of (selected_indices, selected_scores, threshold)
    """
    threshold = np.percentile(importance_scores, percentile)
    selected_indices = np.where(importance_scores >= threshold)[0]
    selected_scores = importance_scores[selected_indices]
    
    # Sort by importance descending
    sort_idx = np.argsort(selected_scores)[::-1]
    selected_indices = selected_indices[sort_idx]
    selected_scores = selected_scores[sort_idx]
    
    logger.info(f"Selected {len(selected_indices)} genes at {percentile}th percentile")
    logger.info(f"Threshold: {threshold:.6f}")
    logger.info(f"Top 5 genes: {[gene_names[i] for i in selected_indices[:5]]}")
    
    return selected_indices, selected_scores, threshold


def select_features_top_n(
    importance_scores: np.ndarray,
    gene_names: List[str],
    top_n: int = 20
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Select exactly top N features by importance.
    
    Args:
        importance_scores: Array of importance values
        gene_names: List of gene names (for logging)
        top_n: Number of top features to select
        
    Returns:
        Tuple of (selected_indices, selected_scores, min_threshold)
    """
    # Get indices of top N genes
    selected_indices = np.argsort(importance_scores)[-top_n:][::-1]
    selected_scores = importance_scores[selected_indices]
    threshold = selected_scores[-1]  # Minimum score in top N
    
    logger.info(f"Selected top {top_n} genes")
    logger.info(f"Min importance in top {top_n}: {threshold:.6f}")
    logger.info(f"Top 5 genes: {[gene_names[i] for i in selected_indices[:5]]}")
    
    return selected_indices, selected_scores, threshold


def get_selected_gene_names(
    selected_indices: np.ndarray,
    gene_names: List[str],
    importance_scores: np.ndarray
) -> pd.DataFrame:
    """
    Create DataFrame of selected genes with their importance scores.
    
    Args:
        selected_indices: Array of selected gene indices
        gene_names: List of all gene names
        importance_scores: Array of all importance scores
        
    Returns:
        DataFrame with columns: gene_name, importance, rank
    """
    selected_genes_df = pd.DataFrame({
        'gene_name': [gene_names[i] for i in selected_indices],
        'importance': importance_scores[selected_indices],
        'rank': range(1, len(selected_indices) + 1)
    })
    
    return selected_genes_df


def compute_bidirectional_consensus(
    model1_importance: np.ndarray,
    model2_importance: np.ndarray,
    gene_names: List[str],
    selection_method: str = 'percentile',
    percentile: float = 95.0,
    top_n: int = 20
) -> Dict:
    """
    Compute consensus genes between two models (bidirectional selection).
    
    Following Haibe-Kains et al. (2012) for cross-cohort biomarker stability.
    
    Args:
        model1_importance: Importance scores from model 1 (e.g., TCGA)
        model2_importance: Importance scores from model 2 (e.g., ORIEN)
        gene_names: List of gene names (must match both models)
        selection_method: 'percentile' or 'top_n'
        percentile: Percentile threshold if using percentile method
        top_n: Number of top genes if using top_n method
        
    Returns:
        Dictionary with consensus analysis results
    """
    logger.info(f"\nComputing bidirectional consensus ({selection_method})")
    
    # Select genes from each model
    if selection_method == 'percentile':
        indices1, scores1, _ = select_features_percentile(
            model1_importance, gene_names, percentile
        )
        indices2, scores2, _ = select_features_percentile(
            model2_importance, gene_names, percentile
        )
    elif selection_method == 'top_n':
        indices1, scores1, _ = select_features_top_n(
            model1_importance, gene_names, top_n
        )
        indices2, scores2, _ = select_features_top_n(
            model2_importance, gene_names, top_n
        )
    else:
        raise ValueError(f"Unknown selection method: {selection_method}")
    
    # Get gene names
    genes1 = set([gene_names[i] for i in indices1])
    genes2 = set([gene_names[i] for i in indices2])
    
    # Compute consensus (intersection)
    consensus_genes = genes1 & genes2
    
    # Compute similarity metrics
    union = genes1 | genes2
    jaccard = len(consensus_genes) / len(union) if len(union) > 0 else 0.0
    overlap_rate = len(consensus_genes) / min(len(genes1), len(genes2)) if min(len(genes1), len(genes2)) > 0 else 0.0
    
    logger.info(f"Model 1 genes: {len(genes1)}")
    logger.info(f"Model 2 genes: {len(genes2)}")
    logger.info(f"Consensus genes: {len(consensus_genes)}")
    logger.info(f"Jaccard index: {jaccard:.3f}")
    logger.info(f"Overlap rate: {overlap_rate:.1%}")
    
    # Create consensus DataFrame with combined importance
    consensus_list = []
    for gene in consensus_genes:
        idx = gene_names.index(gene)
        consensus_list.append({
            'gene_name': gene,
            'model1_importance': model1_importance[idx],
            'model2_importance': model2_importance[idx],
            'mean_importance': (model1_importance[idx] + model2_importance[idx]) / 2
        })
    
    consensus_df = pd.DataFrame(consensus_list).sort_values(
        'mean_importance', ascending=False
    )
    
    return {
        'consensus_genes': sorted(list(consensus_genes)),
        'consensus_df': consensus_df,
        'n_consensus': len(consensus_genes),
        'n_model1': len(genes1),
        'n_model2': len(genes2),
        'jaccard_index': jaccard,
        'overlap_rate': overlap_rate,
        'model1_only_genes': sorted(list(genes1 - genes2)),
        'model2_only_genes': sorted(list(genes2 - genes1))
    }


def compare_with_chapter2_biomarkers(
    neural_net_genes: List[str],
    chapter2_genes: List[str]
) -> Dict:
    """
    Compare neural network consensus genes with Chapter 2 Cox regression genes.
    
    Args:
        neural_net_genes: List of consensus genes from neural network
        chapter2_genes: List of consensus genes from Chapter 2 Cox regression
        
    Returns:
        Dictionary with comparison metrics
    """
    neural_set = set(neural_net_genes)
    chapter2_set = set(chapter2_genes)
    
    overlap = neural_set & chapter2_set
    neural_only = neural_set - chapter2_set
    chapter2_only = chapter2_set - neural_set
    union = neural_set | chapter2_set
    
    jaccard = len(overlap) / len(union) if len(union) > 0 else 0.0
    overlap_pct = 100.0 * len(overlap) / len(chapter2_set) if len(chapter2_set) > 0 else 0.0
    
    logger.info(f"\nComparison with Chapter 2 biomarkers:")
    logger.info(f"Neural network genes: {len(neural_set)}")
    logger.info(f"Chapter 2 genes: {len(chapter2_set)}")
    logger.info(f"Overlap: {len(overlap)} genes")
    logger.info(f"Overlap percentage: {overlap_pct:.1f}%")
    logger.info(f"Jaccard index: {jaccard:.3f}")
    
    if len(overlap) > 0:
        logger.info(f"Overlapping genes: {sorted(list(overlap))}")
    
    return {
        'n_neural_genes': len(neural_set),
        'n_chapter2_genes': len(chapter2_set),
        'n_overlap': len(overlap),
        'overlap_percentage': overlap_pct,
        'jaccard_index': jaccard,
        'overlap_genes': sorted(list(overlap)),
        'neural_only_genes': sorted(list(neural_only)),
        'chapter2_only_genes': sorted(list(chapter2_only))
    }


def rank_based_consensus(
    model1_importance: np.ndarray,
    model2_importance: np.ndarray,
    gene_names: List[str],
    top_k: int = 50,
    consensus_threshold: int = 25
) -> Dict:
    """
    Alternative consensus method using rank-based selection.
    
    Instead of hard thresholding, use rank positions.
    A gene is "consensus" if it appears in top K of both models.
    
    Args:
        model1_importance: Importance from model 1
        model2_importance: Importance from model 2
        gene_names: List of gene names
        top_k: Consider top K genes from each model
        consensus_threshold: Minimum combined rank threshold
        
    Returns:
        Dictionary with consensus results
    """
    # Get ranks (higher importance = lower rank number)
    ranks1 = len(gene_names) - np.argsort(np.argsort(model1_importance))
    ranks2 = len(gene_names) - np.argsort(np.argsort(model2_importance))
    
    # Get top K genes from each
    top_k_idx1 = set(np.argsort(model1_importance)[-top_k:])
    top_k_idx2 = set(np.argsort(model2_importance)[-top_k:])
    
    # Consensus = genes in top K of both models
    consensus_idx = top_k_idx1 & top_k_idx2
    
    # Create results
    consensus_data = []
    for idx in consensus_idx:
        consensus_data.append({
            'gene_name': gene_names[idx],
            'model1_rank': int(ranks1[idx]),
            'model2_rank': int(ranks2[idx]),
            'combined_rank': int(ranks1[idx] + ranks2[idx]),
            'model1_importance': model1_importance[idx],
            'model2_importance': model2_importance[idx],
            'mean_importance': (model1_importance[idx] + model2_importance[idx]) / 2
        })
    
    consensus_df = pd.DataFrame(consensus_data).sort_values('combined_rank')
    
    return {
        'consensus_genes': sorted([gene_names[i] for i in consensus_idx]),
        'consensus_df': consensus_df,
        'n_consensus': len(consensus_idx),
        'top_k': top_k
    }


def stability_across_folds(
    fold_importance_scores: List[np.ndarray],
    gene_names: List[str],
    selection_method: str = 'percentile',
    percentile: float = 95.0
) -> Dict:
    """
    Assess feature stability across cross-validation folds.
    
    Useful for understanding if feature selection is stable.
    
    Args:
        fold_importance_scores: List of importance arrays (one per fold)
        gene_names: List of gene names
        selection_method: 'percentile' or 'frequency'
        percentile: Percentile threshold
        
    Returns:
        Dictionary with stability metrics
    """
    n_folds = len(fold_importance_scores)
    selected_genes_per_fold = []
    
    # Select genes in each fold
    for fold_scores in fold_importance_scores:
        if selection_method == 'percentile':
            threshold = np.percentile(fold_scores, percentile)
            selected = set([gene_names[i] for i in np.where(fold_scores >= threshold)[0]])
        else:
            raise ValueError(f"Unknown method: {selection_method}")
        
        selected_genes_per_fold.append(selected)
    
    # Compute frequency of selection
    gene_frequency = {}
    for gene in gene_names:
        count = sum(1 for selected in selected_genes_per_fold if gene in selected)
        if count > 0:
            gene_frequency[gene] = count / n_folds
    
    # Genes selected in all folds
    stable_genes = [g for g, freq in gene_frequency.items() if freq == 1.0]
    
    # Genes selected in majority of folds
    majority_genes = [g for g, freq in gene_frequency.items() if freq >= 0.5]
    
    return {
        'n_folds': n_folds,
        'stable_genes': stable_genes,
        'n_stable': len(stable_genes),
        'majority_genes': majority_genes,
        'n_majority': len(majority_genes),
        'gene_frequency': gene_frequency
    }
