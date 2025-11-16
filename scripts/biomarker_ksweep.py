#!/usr/bin/env python3
"""
Script: biomarker_ksweep.py
Purpose: Sweep k values to find optimal number of biomarkers from transfer learning
Status: ACTIVE (Chapter 4 - k-value optimization)
Author: Phuong
Created: 2024-11-15

This script:
1. Sweeps through k values [60, 70, 80, 90, 95, 100, 110, 120, 130, 140, 150]
2. For each k, extracts top k genes from all seeds
3. Computes consensus genes (≥3/5 seeds) for both directions
4. Tracks: n_consensus_tcga, n_consensus_orien, n_bidirectional, overlap%
5. Generates summary table similar to Chapter 3
6. Helps identify optimal k for final biomarker panel

Methodology:
- Consensus threshold: ≥3/5 seeds (60% agreement)
- Bidirectional stability: genes appearing in both TCGA→ORIEN and ORIEN→TCGA
- Target: ~20-30 consensus genes (matching Chapter 2 scale)

Reference:
- Chapter 3 methodology: k-sweep to find stable gene count
- Ein-Dor et al. 2005 (Nature Genetics): Consensus gene selection

Usage:
    python scripts/biomarker_ksweep.py \
        --output_dir results/biomarker_ksweep_transfer \
        --k_values 60 70 80 90 95 100 110 120 130 140 150
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Set
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.elastic_deepsurv import ElasticDeepSurv


# ============================================================================
# HELPER FUNCTIONS (Reused from extract_transfer_biomarkers.py)
# ============================================================================

def load_consensus_genes(filepath: str = 'data/raw/consensus_genes_308.txt') -> List[str]:
    """Load 308 consensus genes."""
    with open(filepath, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]
    return genes


def load_cox_genes(filepath: str = 'data/raw/cox_consensus_genes_20.txt') -> List[str]:
    """
    Load Chapter 2 Cox regression consensus genes (20 genes).
    
    Returns empty list if file not found (Cox comparison will be skipped).
    """
    try:
        with open(filepath, 'r') as f:
            genes = [line.strip() for line in f if line.strip()]
        return genes
    except FileNotFoundError:
        print(f"  ⚠️  Cox genes file not found: {filepath}")
        print(f"      Chapter 2 comparison will be skipped")
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
    top_k: int
) -> Tuple[np.ndarray, List[str]]:
    """
    Load a trained model and extract top k genes.
    
    Returns:
        (importance_scores, top_k_genes) or (None, None) if loading fails
    """
    try:
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
        
        # Verify gene count matches
        if n_features != len(gene_names):
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
        
    except Exception as e:
        print(f"        ⚠️  Error loading model: {e}")
        return None, None


def compute_consensus_genes(
    gene_lists: List[List[str]],
    min_appearances: int = 3
) -> List[str]:
    """
    Compute consensus genes that appear in multiple gene lists.
    
    Args:
        gene_lists: List of gene lists from different seeds
        min_appearances: Minimum number of lists a gene must appear in (default: 3/5)
    
    Returns:
        Consensus genes sorted by frequency (most stable first)
    """
    # Count gene appearances
    all_genes = [gene for gene_list in gene_lists for gene in gene_list]
    gene_counts = Counter(all_genes)
    
    # Filter by minimum appearances
    consensus = [gene for gene, count in gene_counts.items() 
                 if count >= min_appearances]
    
    # Sort by frequency (most common first)
    consensus = sorted(consensus, key=lambda g: gene_counts[g], reverse=True)
    
    return consensus


# ============================================================================
# K-SWEEP FUNCTIONS
# ============================================================================

def extract_genes_for_k(
    k: int,
    gene_names: List[str],
    seeds: List[int],
    models_dir: Path,
    verbose: bool = False
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Extract top k genes from all seeds for a given k value.
    
    Args:
        k: Number of top genes to extract
        gene_names: List of all gene names (308 genes)
        seeds: List of random seeds to process
        models_dir: Directory containing model files (e.g., results/transfer_learning)
        verbose: Print detailed progress
        
    Returns:
        (tcga_transfer_genes, orien_transfer_genes)
        - Each is a list of gene lists (one per seed)
    """
    tcga_transfer_genes = []
    orien_transfer_genes = []
    
    if verbose:
        print(f"\n  Extracting top {k} genes from each seed...")
        print(f"  Models directory: {models_dir}")
    
    for seed in seeds:
        # ORIEN→TCGA: orien_to_tcga_seed{X}_*/tcga_finetuned_seed{X}.pth
        tcga_model_pattern = f"orien_to_tcga_seed{seed}_*"
        tcga_dirs = list(models_dir.glob(tcga_model_pattern))
        
        if tcga_dirs:
            tcga_model_path = tcga_dirs[0] / f"tcga_finetuned_seed{seed}.pth"
            if tcga_model_path.exists():
                importance, top_genes = load_model_and_extract_genes(
                    tcga_model_path,
                    gene_names,
                    top_k=k
                )
                if top_genes is not None:
                    tcga_transfer_genes.append(top_genes)
                    if verbose:
                        print(f"    ✓ TCGA seed {seed}: {len(top_genes)} genes")
        else:
            if verbose:
                print(f"    ⚠️  TCGA seed {seed}: No model directory found (pattern: {tcga_model_pattern})")
        
        # TCGA→ORIEN: tcga_to_orien_seed{X}_*/orien_finetuned_seed{X}.pth
        orien_model_pattern = f"tcga_to_orien_seed{seed}_*"
        orien_dirs = list(models_dir.glob(orien_model_pattern))
        
        if orien_dirs:
            orien_model_path = orien_dirs[0] / f"orien_finetuned_seed{seed}.pth"
            if orien_model_path.exists():
                importance, top_genes = load_model_and_extract_genes(
                    orien_model_path,
                    gene_names,
                    top_k=k
                )
                if top_genes is not None:
                    orien_transfer_genes.append(top_genes)
                    if verbose:
                        print(f"    ✓ ORIEN seed {seed}: {len(top_genes)} genes")
        else:
            if verbose:
                print(f"    ⚠️  ORIEN seed {seed}: No model directory found (pattern: {orien_model_pattern})")
    
    return tcga_transfer_genes, orien_transfer_genes


def analyze_k_value(
    k: int,
    tcga_genes: List[List[str]],
    orien_genes: List[List[str]],
    cox_genes: List[str] = None,
    min_appearances: int = 3
) -> Dict:
    """
    Analyze consensus and overlap for a given k value.
    
    Args:
        k: The k value being analyzed
        tcga_genes: List of top-k gene lists from TCGA transfer models
        orien_genes: List of top-k gene lists from ORIEN transfer models
        cox_genes: Chapter 2 Cox regression genes for comparison (optional)
        min_appearances: Minimum appearances for consensus (default: 3/5 seeds)
        
    Returns:
        Dictionary with analysis results including Cox overlap
    """
    # Compute consensus for each direction
    tcga_consensus = compute_consensus_genes(tcga_genes, min_appearances)
    orien_consensus = compute_consensus_genes(orien_genes, min_appearances)
    
    # Bidirectional consensus
    bidirectional = set(tcga_consensus) & set(orien_consensus)
    
    # Compute overlap percentages
    # Overlap relative to smaller consensus set (more stringent)
    smaller_consensus_size = min(len(tcga_consensus), len(orien_consensus))
    if smaller_consensus_size > 0:
        overlap_pct = (len(bidirectional) / smaller_consensus_size) * 100
    else:
        overlap_pct = 0.0
    
    # Jaccard index
    union = set(tcga_consensus) | set(orien_consensus)
    jaccard = len(bidirectional) / len(union) if len(union) > 0 else 0.0
    
    # Chapter 2 Cox genes overlap (if provided)
    cox_overlap_bidir = 0
    cox_overlap_tcga = 0
    cox_overlap_orien = 0
    cox_overlap_pct_bidir = 0.0
    cox_overlap_pct_tcga = 0.0
    cox_overlap_pct_orien = 0.0
    shared_with_cox = []
    
    if cox_genes:
        cox_set = set(cox_genes)
        
        # Bidirectional vs Cox
        shared_with_cox = list(bidirectional & cox_set)
        cox_overlap_bidir = len(shared_with_cox)
        cox_overlap_pct_bidir = (cox_overlap_bidir / len(bidirectional) * 100) if len(bidirectional) > 0 else 0.0
        
        # TCGA consensus vs Cox
        tcga_cox_shared = set(tcga_consensus) & cox_set
        cox_overlap_tcga = len(tcga_cox_shared)
        cox_overlap_pct_tcga = (cox_overlap_tcga / len(tcga_consensus) * 100) if len(tcga_consensus) > 0 else 0.0
        
        # ORIEN consensus vs Cox
        orien_cox_shared = set(orien_consensus) & cox_set
        cox_overlap_orien = len(orien_cox_shared)
        cox_overlap_pct_orien = (cox_overlap_orien / len(orien_consensus) * 100) if len(orien_consensus) > 0 else 0.0
    
    return {
        'k': k,
        'n_tcga_consensus': len(tcga_consensus),
        'n_orien_consensus': len(orien_consensus),
        'n_bidirectional': len(bidirectional),
        'overlap_pct': overlap_pct,
        'jaccard_index': jaccard,
        'tcga_consensus_genes': tcga_consensus,
        'orien_consensus_genes': orien_consensus,
        'bidirectional_genes': list(bidirectional),
        # Cox overlap metrics
        'cox_overlap_bidir': cox_overlap_bidir,
        'cox_overlap_pct_bidir': cox_overlap_pct_bidir,
        'cox_overlap_tcga': cox_overlap_tcga,
        'cox_overlap_pct_tcga': cox_overlap_pct_tcga,
        'cox_overlap_orien': cox_overlap_orien,
        'cox_overlap_pct_orien': cox_overlap_pct_orien,
        'shared_genes_with_cox': shared_with_cox
    }


def run_ksweep(
    k_values: List[int],
    models_dir: str,
    output_dir: str,
    seeds: List[int] = [42, 123, 456, 789, 1011],
    min_appearances: int = 3
) -> pd.DataFrame:
    """
    Run k-value sweep analysis.
    
    Args:
        k_values: List of k values to sweep
        models_dir: Directory containing trained models
        output_dir: Output directory for results
        seeds: Random seeds to analyze
        min_appearances: Minimum seed appearances for consensus
        
    Returns:
        DataFrame with results for all k values
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Validate models directory
    models_path = Path(models_dir)
    if not models_path.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    
    # Load gene names
    print(f"{'='*80}")
    print("K-VALUE SWEEP ANALYSIS")
    print(f"{'='*80}\n")
    
    print("Loading 308 consensus genes...")
    gene_names = load_consensus_genes()
    print(f"✓ Loaded {len(gene_names)} genes\n")
    
    # Load Chapter 2 Cox genes for comparison
    print("Loading Chapter 2 Cox genes...")
    cox_genes = load_cox_genes()
    if cox_genes:
        print(f"✓ Loaded {len(cox_genes)} Cox genes for comparison\n")
    else:
        print(f"  (Chapter 2 comparison will be skipped)\n")
    
    print(f"Configuration:")
    print(f"  Models directory: {models_dir}")
    print(f"  K values: {k_values}")
    print(f"  Seeds: {seeds}")
    print(f"  Consensus threshold: ≥{min_appearances}/{len(seeds)} seeds")
    print(f"  Output directory: {output_dir}\n")
    
    # Run sweep
    results = []
    
    for i, k in enumerate(k_values, 1):
        print(f"\n{'='*80}")
        print(f"K-VALUE {i}/{len(k_values)}: k = {k}")
        print(f"{'='*80}")
        
        # Extract genes for this k value
        tcga_genes, orien_genes = extract_genes_for_k(
            k=k,
            gene_names=gene_names,
            seeds=seeds,
            models_dir=models_path,
            verbose=True
        )
        
        print(f"\n  Successfully extracted from:")
        print(f"    TCGA models: {len(tcga_genes)}/{len(seeds)}")
        print(f"    ORIEN models: {len(orien_genes)}/{len(seeds)}")
        
        # Analyze this k value
        analysis = analyze_k_value(
            k=k,
            tcga_genes=tcga_genes,
            orien_genes=orien_genes,
            cox_genes=cox_genes if cox_genes else None,
            min_appearances=min_appearances
        )
        
        results.append(analysis)
        
        # Print summary
        print(f"\n  Results for k={k}:")
        print(f"    TCGA consensus: {analysis['n_tcga_consensus']} genes")
        print(f"    ORIEN consensus: {analysis['n_orien_consensus']} genes")
        print(f"    Bidirectional: {analysis['n_bidirectional']} genes")
        print(f"    Overlap: {analysis['overlap_pct']:.1f}%")
        print(f"    Jaccard index: {analysis['jaccard_index']:.3f}")
        
        # Print Cox overlap if available
        if cox_genes and analysis['cox_overlap_bidir'] > 0:
            print(f"    Cox overlap: {analysis['cox_overlap_bidir']}/{len(cox_genes)} ({analysis['cox_overlap_pct_bidir']:.1f}%)")
            print(f"    Shared genes: {', '.join(analysis['shared_genes_with_cox'][:5])}{' ...' if len(analysis['shared_genes_with_cox']) > 5 else ''}")
    
    # ========================================
    # Create summary table
    # ========================================
    
    print(f"\n{'='*80}")
    print("GENERATING SUMMARY TABLE")
    print(f"{'='*80}\n")
    
    # Build summary table with optional Cox columns
    table_data = []
    for r in results:
        row = {
            'k': r['k'],
            'TCGA_consensus': r['n_tcga_consensus'],
            'ORIEN_consensus': r['n_orien_consensus'],
            'Bidirectional': r['n_bidirectional'],
            'Overlap_%': f"{r['overlap_pct']:.1f}",
            'Jaccard': f"{r['jaccard_index']:.3f}"
        }
        
        # Add Cox overlap if available
        if cox_genes:
            row['Cox_overlap'] = r['cox_overlap_bidir']
            row['Cox_overlap_%'] = f"{r['cox_overlap_pct_bidir']:.1f}"
        
        table_data.append(row)
    
    summary_df = pd.DataFrame(table_data)
    
    print(summary_df.to_string(index=False))
    
    # ========================================
    # Save results
    # ========================================
    
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}\n")
    
    # Save summary table
    summary_df.to_csv(output_path / 'ksweep_summary_table.csv', index=False)
    print(f"✓ Summary table: ksweep_summary_table.csv")
    
    # Save detailed results
    with open(output_path / 'ksweep_full_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Full results: ksweep_full_results.json")
    
    # Save gene lists for each k value
    genes_dir = output_path / 'gene_lists'
    genes_dir.mkdir(exist_ok=True)
    
    for r in results:
        k = r['k']
        
        # Save TCGA consensus
        with open(genes_dir / f'k{k}_tcga_consensus.txt', 'w') as f:
            f.write('\n'.join(r['tcga_consensus_genes']))
        
        # Save ORIEN consensus
        with open(genes_dir / f'k{k}_orien_consensus.txt', 'w') as f:
            f.write('\n'.join(r['orien_consensus_genes']))
        
        # Save bidirectional
        with open(genes_dir / f'k{k}_bidirectional.txt', 'w') as f:
            f.write('\n'.join(sorted(r['bidirectional_genes'])))
    
    print(f"✓ Gene lists saved in: gene_lists/")
    
    # ========================================
    # Print recommendations
    # ========================================
    
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}\n")
    
    # Find k values that yield ~20-30 consensus genes
    target_range = (20, 30)
    
    print(f"Looking for k values that yield ~{target_range[0]}-{target_range[1]} bidirectional consensus genes:\n")
    
    candidates = []
    for r in results:
        n_bidir = r['n_bidirectional']
        if target_range[0] <= n_bidir <= target_range[1]:
            candidates.append(r)
            print(f"  ✓ k={r['k']:3d}: {n_bidir:2d} genes, {r['overlap_pct']:5.1f}% overlap, Jaccard={r['jaccard_index']:.3f}")
    
    if candidates:
        # Recommend k with highest overlap %
        best_k = max(candidates, key=lambda r: r['overlap_pct'])
        print(f"\n🎯 RECOMMENDED: k={best_k['k']}")
        print(f"   - Bidirectional consensus: {best_k['n_bidirectional']} genes")
        print(f"   - Overlap: {best_k['overlap_pct']:.1f}%")
        print(f"   - Jaccard index: {best_k['jaccard_index']:.3f}")
        print(f"   - Rationale: Highest stability within target range")
    else:
        print(f"\n⚠️  No k value yielded {target_range[0]}-{target_range[1]} consensus genes")
        print(f"   Consider adjusting k range or consensus threshold")
    
    print(f"\n{'='*80}")
    print("K-SWEEP ANALYSIS COMPLETE")
    print(f"{'='*80}\n")
    
    return summary_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sweep k values to find optimal biomarker count",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python scripts/biomarker_ksweep.py \
      --models_dir results/transfer_learning \
      --output_dir results/biomarker_ksweep
  
  python scripts/biomarker_ksweep.py \
      --models_dir results/transfer_learning \
      --k_values 60 70 80 90 95 100 110 120 130 140 150 \
      --output_dir results/biomarker_ksweep \
      --seeds 42 123 456 789 1011
        """
    )
    
    parser.add_argument('--models_dir', type=str,
                       default='results/transfer_learning',
                       help='Directory containing trained models (default: results/transfer_learning)')
    parser.add_argument('--output_dir', type=str,
                       default='results/biomarker_ksweep_transfer',
                       help='Output directory for k-sweep analysis')
    parser.add_argument('--k_values', type=int, nargs='+',
                       default=[60, 70, 80, 90, 95, 100, 110, 120, 130, 140, 150],
                       help='K values to sweep (default: 60 to 150)')
    parser.add_argument('--seeds', type=int, nargs='+',
                       default=[42, 123, 456, 789, 1011],
                       help='Random seeds to analyze (default: 42 123 456 789 1011)')
    parser.add_argument('--min_appearances', type=int, default=3,
                       help='Minimum seed appearances for consensus (default: 3/5)')
    
    args = parser.parse_args()
    
    # Run k-sweep
    summary_df = run_ksweep(
        k_values=args.k_values,
        models_dir=args.models_dir,
        output_dir=args.output_dir,
        seeds=args.seeds,
        min_appearances=args.min_appearances
    )
    
    print("\n✅ K-sweep analysis completed successfully!")
    print(f"📁 Results saved in: {args.output_dir}/")
