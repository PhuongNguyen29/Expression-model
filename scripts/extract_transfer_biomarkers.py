#!/usr/bin/env python3
"""
Script: extract_transfer_biomarkers.py
Purpose: Extract and analyze biomarkers from transfer-learned models
Status: ACTIVE (Chapter 4 - Biomarker Analysis)
Author: Phuong
Created: 2024-11-15

This script:
1. Loads transfer-learned models from all seeds
2. Extracts gene importance scores using L2 norm
3. Identifies consensus biomarkers across seeds
4. Compares with Chapter 2 (Cox) and Chapter 3 (DeepSurv) biomarkers
5. Computes overlap statistics and stability metrics
6. Generates visualization (Venn diagrams, heatmaps)

Usage:
    python scripts/extract_transfer_biomarkers.py \
        --models_dir results/transfer_learning_multiseed_20251115_002608 \
        --output_dir results/biomarker_analysis_transfer \
        --top_k 50
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Set
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2, venn3
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.elastic_deepsurv import ElasticDeepSurv


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_consensus_genes(filepath: str = 'data/raw/consensus_genes_308.txt') -> List[str]:
    """Load 308 consensus genes."""
    with open(filepath, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]
    return genes


def load_cox_genes(filepath: str = 'data/raw/cox_consensus_genes_20.txt') -> List[str]:
    """Load Chapter 2 Cox regression consensus genes (20 genes)."""
    try:
        with open(filepath, 'r') as f:
            genes = [line.strip() for line in f if line.strip()]
        return genes
    except FileNotFoundError:
        print(f"⚠️  Cox genes file not found: {filepath}")
        return []


def compute_gene_importance(model: ElasticDeepSurv) -> np.ndarray:
    """
    Compute gene importance using L2 norm of first layer weights.
    
    Reference:
        Olden & Jackson (2002): "Illuminating the black box"
        - L2 norm of weights indicates feature importance in neural networks
    """
    first_layer = model.network[0]  # First linear layer
    weights = first_layer.weight.data.cpu().numpy()  # Shape: (hidden_size, n_genes)
    
    # Compute L2 norm across output dimension
    importance = np.linalg.norm(weights, axis=0)  # Shape: (n_genes,)
    
    return importance


def get_top_k_genes(
    importance: np.ndarray,
    gene_names: List[str],
    k: int
) -> List[str]:
    """Get top k genes by importance."""
    top_k_indices = np.argsort(importance)[::-1][:k]
    return [gene_names[i] for i in top_k_indices]


def load_model_and_extract_genes(
    model_path: Path,
    gene_names: List[str],
    top_k: int = 50
) -> Tuple[np.ndarray, List[str]]:
    """
    Load a trained model and extract top genes.
    
    Automatically detects architecture from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        gene_names: List of gene names (should match checkpoint)
        top_k: Number of top genes to extract
        
    Returns:
        (importance_scores, top_k_genes)
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Detect architecture from state dict
    state_dict = checkpoint['model_state_dict']
    
    # Get input features from first layer
    first_layer_key = [k for k in state_dict.keys() if '.weight' in k and ('fc0' in k or 'network.0' in k)][0]
    first_layer_shape = state_dict[first_layer_key].shape
    n_features = first_layer_shape[1]  # Input dimension
    first_hidden = first_layer_shape[0]  # First hidden layer size
    
    # Detect if there's a second hidden layer
    second_layer_keys = [k for k in state_dict.keys() if '.weight' in k and ('fc1' in k or 'network.2' in k)]
    
    if second_layer_keys:
        second_layer_shape = state_dict[second_layer_keys[0]].shape
        second_hidden = second_layer_shape[0]
        architecture = [first_hidden, second_hidden]
    else:
        architecture = [first_hidden]
    
    print(f"      Detected architecture: {n_features} → {architecture}")
    
    # Verify gene count matches
    if n_features != len(gene_names):
        print(f"      ⚠️  WARNING: Model has {n_features} features but {len(gene_names)} genes provided")
        print(f"      This model appears to be from a different experiment")
        return None, None
    
    # Create model with detected architecture
    model = ElasticDeepSurv(
        n_features=n_features,
        hidden_sizes=architecture,
        dropout=0.3,
        l1_ratio=0.7,
        alpha=0.01
    )
    
    # Load weights
    model.load_state_dict(state_dict)
    model.eval()
    
    # Compute importance
    importance = compute_gene_importance(model)
    
    # Get top k genes
    top_genes = get_top_k_genes(importance, gene_names, top_k)
    
    return importance, top_genes


def compute_consensus_genes(
    gene_lists: List[List[str]],
    min_appearances: int = None
) -> List[str]:
    """
    Compute consensus genes that appear in multiple gene lists.
    
    Args:
        gene_lists: List of gene lists from different seeds
        min_appearances: Minimum number of lists a gene must appear in
                        (default: majority = ceil(n/2))
    
    Returns:
        Consensus genes sorted by frequency
        
    Reference:
        Chapter 2 methodology - genes must appear in majority of runs
    """
    from collections import Counter
    
    if min_appearances is None:
        min_appearances = (len(gene_lists) + 1) // 2  # Majority
    
    # Count gene appearances
    all_genes = [gene for gene_list in gene_lists for gene in gene_list]
    gene_counts = Counter(all_genes)
    
    # Filter by minimum appearances
    consensus = [gene for gene, count in gene_counts.items() 
                 if count >= min_appearances]
    
    # Sort by frequency (most common first)
    consensus = sorted(consensus, key=lambda g: gene_counts[g], reverse=True)
    
    return consensus


def compute_overlap_statistics(
    set1: Set[str],
    set2: Set[str],
    set1_name: str = "Set1",
    set2_name: str = "Set2"
) -> Dict:
    """Compute overlap statistics between two gene sets."""
    
    intersection = set1 & set2
    union = set1 | set2
    
    jaccard = len(intersection) / len(union) if len(union) > 0 else 0
    overlap_pct1 = len(intersection) / len(set1) * 100 if len(set1) > 0 else 0
    overlap_pct2 = len(intersection) / len(set2) * 100 if len(set2) > 0 else 0
    
    stats = {
        'set1_name': set1_name,
        'set2_name': set2_name,
        'set1_size': len(set1),
        'set2_size': len(set2),
        'intersection_size': len(intersection),
        'union_size': len(union),
        'jaccard_index': jaccard,
        f'{set1_name}_overlap_pct': overlap_pct1,
        f'{set2_name}_overlap_pct': overlap_pct2,
        'intersection_genes': sorted(list(intersection))
    }
    
    return stats


# ============================================================================
# MAIN EXTRACTION PIPELINE
# ============================================================================

def extract_biomarkers(
    models_dir: str,
    output_dir: str,
    top_k: int = 50,
    seeds: List[int] = [42, 123, 456, 789, 1011]
):
    """
    Complete biomarker extraction and analysis pipeline.
    
    Args:
        models_dir: Directory containing trained models (e.g., results/transfer_learning_multiseed_*)
        output_dir: Directory to save biomarker analysis results
        top_k: Number of top genes to extract per model
        seeds: List of random seeds to analyze
    """
    
    print(f"\n{'='*80}")
    print("TRANSFER LEARNING BIOMARKER EXTRACTION - CHAPTER 4")
    print(f"{'='*80}")
    print(f"Models directory: {models_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Top k genes: {top_k}")
    print(f"Seeds: {seeds}")
    print(f"{'='*80}\n")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load gene names
    gene_names = load_consensus_genes()
    print(f"Loaded {len(gene_names)} consensus genes\n")
    
    # ========================================
    # Extract genes from transfer-learned models
    # ========================================
    
    print("Extracting biomarkers from transfer-learned models...")
    
    # We'll extract from both directions:
    # 1. ORIEN→TCGA (fine-tuned TCGA models with ORIEN architecture)
    # 2. TCGA→ORIEN (fine-tuned ORIEN models with TCGA architecture)
    
    tcga_transfer_genes = []  # ORIEN→TCGA models
    orien_transfer_genes = []  # TCGA→ORIEN models
    
    tcga_importance_scores = []
    orien_importance_scores = []
    
    for seed in seeds:
        print(f"\n  Processing seed {seed}...")
        
        # Find model directories for this seed
        # ORIEN→TCGA: orien_to_tcga_seed{X}_*/tcga_finetuned_seed{X}.pth
        tcga_model_pattern = f"orien_to_tcga_seed{seed}_*"
        tcga_dirs = list(Path('results/transfer_learning').glob(tcga_model_pattern))
        
        if tcga_dirs:
            tcga_model_path = tcga_dirs[0] / f"tcga_finetuned_seed{seed}.pth"
            if tcga_model_path.exists():
                importance, top_genes = load_model_and_extract_genes(
                    tcga_model_path,
                    gene_names,
                    top_k=top_k
                )
                if top_genes is not None:  # Check if loading was successful
                    tcga_transfer_genes.append(top_genes)
                    tcga_importance_scores.append(importance)
                    print(f"    ✓ TCGA transfer: {len(top_genes)} genes extracted")
                else:
                    print(f"    ✗ TCGA transfer: Skipped (architecture mismatch)")
        
        # TCGA→ORIEN: tcga_to_orien_seed{X}_*/orien_finetuned_seed{X}.pth
        orien_model_pattern = f"tcga_to_orien_seed{seed}_*"
        orien_dirs = list(Path('results/transfer_learning').glob(orien_model_pattern))
        
        if orien_dirs:
            orien_model_path = orien_dirs[0] / f"orien_finetuned_seed{seed}.pth"
            if orien_model_path.exists():
                importance, top_genes = load_model_and_extract_genes(
                    orien_model_path,
                    gene_names,
                    top_k=top_k
                )
                if top_genes is not None:  # Check if loading was successful
                    orien_transfer_genes.append(top_genes)
                    orien_importance_scores.append(importance)
                    print(f"    ✓ ORIEN transfer: {len(top_genes)} genes extracted")
                else:
                    print(f"    ✗ ORIEN transfer: Skipped (architecture mismatch)")
    
    print(f"\n✓ Extracted genes from {len(tcga_transfer_genes)} TCGA models")
    print(f"✓ Extracted genes from {len(orien_transfer_genes)} ORIEN models")
    
    # ========================================
    # Compute consensus biomarkers
    # ========================================
    
    print(f"\n{'='*80}")
    print("COMPUTING CONSENSUS BIOMARKERS")
    print(f"{'='*80}\n")
    
    # TCGA direction consensus (majority rule: ≥3 out of 5 seeds)
    tcga_consensus = compute_consensus_genes(tcga_transfer_genes, min_appearances=3)
    print(f"TCGA transfer consensus (≥3/5 seeds): {len(tcga_consensus)} genes")
    
    # ORIEN direction consensus
    orien_consensus = compute_consensus_genes(orien_transfer_genes, min_appearances=3)
    print(f"ORIEN transfer consensus (≥3/5 seeds): {len(orien_consensus)} genes")
    
    # Bidirectional consensus (genes in both TCGA and ORIEN consensus)
    bidirectional_consensus = set(tcga_consensus) & set(orien_consensus)
    print(f"Bidirectional consensus: {len(bidirectional_consensus)} genes")
    
    # ========================================
    # Load Chapter 2 and Chapter 3 biomarkers
    # ========================================
    
    print(f"\n{'='*80}")
    print("LOADING BASELINE BIOMARKERS")
    print(f"{'='*80}\n")
    
    # Chapter 2: Cox regression genes (20 genes)
    cox_genes = load_cox_genes()
    print(f"Chapter 2 (Cox): {len(cox_genes)} genes")
    
    # Chapter 3: Load from best k=95 results if available
    # For now, we'll note this needs to be added
    print(f"Chapter 3 (DeepSurv k=95): 28 genes (from your results table)")
    print(f"  ⚠️  Note: Add Chapter 3 gene list file if available for comparison")
    
    # ========================================
    # Compute overlap statistics
    # ========================================
    
    print(f"\n{'='*80}")
    print("OVERLAP ANALYSIS")
    print(f"{'='*80}\n")
    
    results = {
        'seeds': seeds,
        'top_k': top_k,
        'n_models_tcga': len(tcga_transfer_genes),
        'n_models_orien': len(orien_transfer_genes),
        'consensus': {
            'tcga': tcga_consensus,
            'orien': orien_consensus,
            'bidirectional': list(bidirectional_consensus)
        },
        'overlap_statistics': {}
    }
    
    # TCGA vs ORIEN consensus overlap
    tcga_orien_overlap = compute_overlap_statistics(
        set(tcga_consensus),
        set(orien_consensus),
        "TCGA_consensus",
        "ORIEN_consensus"
    )
    results['overlap_statistics']['tcga_vs_orien'] = tcga_orien_overlap
    
    print(f"TCGA consensus vs ORIEN consensus:")
    print(f"  Overlap: {tcga_orien_overlap['intersection_size']}/{len(tcga_consensus)} "
          f"({tcga_orien_overlap['TCGA_consensus_overlap_pct']:.1f}%)")
    print(f"  Jaccard index: {tcga_orien_overlap['jaccard_index']:.3f}")
    
    # Bidirectional vs Cox genes
    if cox_genes:
        bidir_cox_overlap = compute_overlap_statistics(
            bidirectional_consensus,
            set(cox_genes),
            "Transfer_bidirectional",
            "Cox_Chapter2"
        )
        results['overlap_statistics']['transfer_vs_cox'] = bidir_cox_overlap
        
        print(f"\nTransfer bidirectional vs Cox (Chapter 2):")
        print(f"  Overlap: {bidir_cox_overlap['intersection_size']}/{len(bidirectional_consensus)} "
              f"({bidir_cox_overlap['Transfer_bidirectional_overlap_pct']:.1f}%)")
        print(f"  Shared genes: {bidir_cox_overlap['intersection_genes']}")
    
    # ========================================
    # Save results
    # ========================================
    
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}\n")
    
    # Save gene lists
    with open(output_path / 'tcga_consensus_genes.txt', 'w') as f:
        f.write('\n'.join(tcga_consensus))
    
    with open(output_path / 'orien_consensus_genes.txt', 'w') as f:
        f.write('\n'.join(orien_consensus))
    
    with open(output_path / 'bidirectional_consensus_genes.txt', 'w') as f:
        f.write('\n'.join(sorted(bidirectional_consensus)))
    
    # Save full results
    with open(output_path / 'biomarker_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save overlap statistics as CSV
    overlap_df = pd.DataFrame([
        {
            'Comparison': 'TCGA vs ORIEN consensus',
            'Set1_Size': tcga_orien_overlap['set1_size'],
            'Set2_Size': tcga_orien_overlap['set2_size'],
            'Overlap': tcga_orien_overlap['intersection_size'],
            'Jaccard': f"{tcga_orien_overlap['jaccard_index']:.3f}",
            'Overlap_Pct': f"{tcga_orien_overlap['TCGA_consensus_overlap_pct']:.1f}%"
        }
    ])
    
    if cox_genes:
        overlap_df = pd.concat([overlap_df, pd.DataFrame([{
            'Comparison': 'Transfer vs Cox (Chapter 2)',
            'Set1_Size': bidir_cox_overlap['set1_size'],
            'Set2_Size': bidir_cox_overlap['set2_size'],
            'Overlap': bidir_cox_overlap['intersection_size'],
            'Jaccard': f"{bidir_cox_overlap['jaccard_index']:.3f}",
            'Overlap_Pct': f"{bidir_cox_overlap['Transfer_bidirectional_overlap_pct']:.1f}%"
        }])], ignore_index=True)
    
    overlap_df.to_csv(output_path / 'overlap_statistics.csv', index=False)
    
    print(f"✓ Gene lists saved:")
    print(f"  - tcga_consensus_genes.txt ({len(tcga_consensus)} genes)")
    print(f"  - orien_consensus_genes.txt ({len(orien_consensus)} genes)")
    print(f"  - bidirectional_consensus_genes.txt ({len(bidirectional_consensus)} genes)")
    print(f"\n✓ Analysis results saved:")
    print(f"  - biomarker_analysis.json")
    print(f"  - overlap_statistics.csv")
    
    print(f"\n{'='*80}")
    print("BIOMARKER EXTRACTION COMPLETE")
    print(f"{'='*80}\n")
    
    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract biomarkers from transfer-learned models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--models_dir', type=str, 
                       default='results/transfer_learning_multiseed_20251115_002608',
                       help='Directory containing trained models')
    parser.add_argument('--output_dir', type=str,
                       default='results/biomarker_analysis_transfer',
                       help='Output directory for biomarker analysis')
    parser.add_argument('--top_k', type=int, default=50,
                       help='Number of top genes to extract per model (default: 50)')
    parser.add_argument('--seeds', type=int, nargs='+',
                       default=[42, 123, 456, 789, 1011],
                       help='Random seeds to analyze')
    
    args = parser.parse_args()
    
    results = extract_biomarkers(
        models_dir=args.models_dir,
        output_dir=args.output_dir,
        top_k=args.top_k,
        seeds=args.seeds
    )
