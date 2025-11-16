#!/usr/bin/env python3
"""
Script: consensus_ksweep_wrapper.py
Purpose: Wrapper to evaluate consensus genes using existing transfer_learning_trainer.py
Status: ACTIVE (Chapter 4 - Consensus k-sweep evaluation)
Author: Phuong
Created: 2024-11-15

This wrapper:
1. Loops through k values from k-sweep results
2. For each k, creates temporary gene list file with consensus genes
3. Calls your existing transfer_learning_trainer.py (which already works!)
4. Collects results and generates comparison table

Strategy: Leverage existing working code instead of reimplementing from scratch

Usage:
    python scripts/consensus_ksweep_wrapper.py \
        --k_values 90 95 100 120 140 150 \
        --gene_lists_dir results/biomarker_ksweep_transfer/gene_lists \
        --output_dir results/consensus_ksweep_evaluation \
        --seed 42
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_consensus_genes(filepath: Path) -> List[str]:
    """Load consensus genes from text file."""
    with open(filepath, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]
    return genes


def create_temp_gene_file(genes: List[str], output_path: Path) -> Path:
    """
    Create temporary gene list file for transfer_learning_trainer.py
    
    Args:
        genes: List of gene names
        output_path: Where to save the temporary file
        
    Returns:
        Path to created file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(genes))
    
    return output_path


def run_transfer_learning(
    source_cohort: str,
    target_cohort: str,
    gene_file: Path,
    output_dir: Path,
    seed: int,
    source_params: str,
    target_params: str
) -> Dict:
    """
    Run transfer_learning_trainer.py via subprocess.
    
    Args:
        source_cohort: 'tcga' or 'orien'
        target_cohort: 'orien' or 'tcga'
        gene_file: Path to consensus gene list
        output_dir: Output directory for this run
        seed: Random seed
        source_params: Path to source hyperparameters JSON
        target_params: Path to target hyperparameters JSON
        
    Returns:
        Dictionary with results (parsed from output)
    """
    # Build command
    cmd = [
        'python',
        'scripts/transfer_learning_trainer.py',
        '--source_cohort', source_cohort,
        '--target_cohort', target_cohort,
        '--source_params', source_params,
        '--target_params', target_params,
        '--consensus_gene_file', str(gene_file),
        '--output_dir', str(output_dir),
        '--seed', str(seed),
        '--n_epochs_pretrain', '100',
        '--n_epochs_finetune', '50',
        '--lr_pretrain', '0.0001',
        '--lr_finetune', '0.00001'
    ]
    
    print(f"    Running: {' '.join(cmd)}")
    
    # Run subprocess
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=str(project_root)
        )
        
        # Parse results from output directory
        results_file = output_dir / 'transfer_results.json'
        
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
            return results
        else:
            # Try to extract C-index from stdout
            print(f"    Warning: No results file found, parsing stdout")
            return parse_stdout_for_cindex(result.stdout)
    
    except subprocess.CalledProcessError as e:
        print(f"    Error running transfer learning:")
        print(f"    {e.stderr}")
        raise


def parse_stdout_for_cindex(stdout: str) -> Dict:
    """
    Parse C-index from stdout as fallback.
    
    Looks for patterns like:
    "Final C-index: 0.7650"
    "Fine-tuning C-index: 0.8621"
    """
    results = {
        'pretrain_cindex': None,
        'finetune_cindex': None,
        'final_cindex': None
    }
    
    lines = stdout.split('\n')
    for line in lines:
        if 'C-index' in line or 'c-index' in line or 'c_index' in line:
            # Try to extract number
            import re
            match = re.search(r'(\d+\.\d+)', line)
            if match:
                cindex = float(match.group(1))
                if 'final' in line.lower():
                    results['final_cindex'] = cindex
                elif 'finetune' in line.lower() or 'fine-tun' in line.lower():
                    results['finetune_cindex'] = cindex
                elif 'pretrain' in line.lower() or 'pre-train' in line.lower():
                    results['pretrain_cindex'] = cindex
    
    return results


def evaluate_k_value(
    k: int,
    gene_lists_dir: Path,
    output_dir: Path,
    seed: int,
    source_params: str,
    target_params: str,
    verbose: bool = True
) -> Dict:
    """
    Evaluate one k value by running transfer learning in both directions.
    
    Args:
        k: K value (number of top genes extracted)
        gene_lists_dir: Directory with consensus gene lists
        output_dir: Output directory for results
        seed: Random seed
        source_params: Path to source hyperparameters
        target_params: Path to target hyperparameters
        verbose: Print progress
        
    Returns:
        Dictionary with results for this k value
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"EVALUATING k = {k}")
        print(f"{'='*80}")
    
    # Load consensus genes
    gene_file = gene_lists_dir / f'k{k}_bidirectional.txt'
    
    if not gene_file.exists():
        print(f"  ⚠️  Gene list not found: {gene_file}")
        return None
    
    genes = load_consensus_genes(gene_file)
    n_genes = len(genes)
    
    if verbose:
        print(f"  Loaded {n_genes} consensus genes")
    
    # Create k-specific output directory
    k_output_dir = output_dir / f'k{k}'
    k_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temporary gene file for this k
    temp_gene_file = k_output_dir / 'consensus_genes.txt'
    create_temp_gene_file(genes, temp_gene_file)
    
    # Run TCGA → ORIEN
    if verbose:
        print(f"\n  Direction 1: TCGA → ORIEN")
    
    tcga_to_orien_dir = k_output_dir / 'tcga_to_orien'
    tcga_to_orien_dir.mkdir(exist_ok=True)
    
    try:
        tcga_to_orien_results = run_transfer_learning(
            source_cohort='tcga',
            target_cohort='orien',
            gene_file=temp_gene_file,
            output_dir=tcga_to_orien_dir,
            seed=seed,
            source_params=source_params,
            target_params=target_params
        )
        
        tcga_to_orien_cindex = (
            tcga_to_orien_results.get('final_cindex') or
            tcga_to_orien_results.get('finetune_cindex') or
            tcga_to_orien_results.get('target_cindex', 0.0)
        )
        
        if verbose:
            print(f"  ✓ TCGA→ORIEN C-index: {tcga_to_orien_cindex:.4f}")
    
    except Exception as e:
        print(f"  ❌ Error in TCGA→ORIEN: {e}")
        tcga_to_orien_cindex = None
        tcga_to_orien_results = {}
    
    # Run ORIEN → TCGA
    if verbose:
        print(f"\n  Direction 2: ORIEN → TCGA")
    
    orien_to_tcga_dir = k_output_dir / 'orien_to_tcga'
    orien_to_tcga_dir.mkdir(exist_ok=True)
    
    try:
        orien_to_tcga_results = run_transfer_learning(
            source_cohort='orien',
            target_cohort='tcga',
            gene_file=temp_gene_file,
            output_dir=orien_to_tcga_dir,
            seed=seed,
            source_params=target_params,  # Note: swapped for direction
            target_params=source_params,
        )
        
        orien_to_tcga_cindex = (
            orien_to_tcga_results.get('final_cindex') or
            orien_to_tcga_results.get('finetune_cindex') or
            orien_to_tcga_results.get('target_cindex', 0.0)
        )
        
        if verbose:
            print(f"  ✓ ORIEN→TCGA C-index: {orien_to_tcga_cindex:.4f}")
    
    except Exception as e:
        print(f"  ❌ Error in ORIEN→TCGA: {e}")
        orien_to_tcga_cindex = None
        orien_to_tcga_results = {}
    
    # Compute average if both succeeded
    if tcga_to_orien_cindex and orien_to_tcga_cindex:
        avg_cindex = (tcga_to_orien_cindex + orien_to_tcga_cindex) / 2
    else:
        avg_cindex = None
    
    # Compile results
    result = {
        'k': k,
        'n_genes': n_genes,
        'tcga_to_orien_cindex': tcga_to_orien_cindex,
        'orien_to_tcga_cindex': orien_to_tcga_cindex,
        'average_cindex': avg_cindex,
        'tcga_to_orien_details': tcga_to_orien_results,
        'orien_to_tcga_details': orien_to_tcga_results,
        'gene_file': str(temp_gene_file)
    }
    
    # Save k-specific results
    with open(k_output_dir / 'results.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    if verbose:
        print(f"\n  Summary for k={k}:")
        print(f"    Genes: {n_genes}")
        print(f"    TCGA→ORIEN: {tcga_to_orien_cindex:.4f if tcga_to_orien_cindex else 'Failed'}")
        print(f"    ORIEN→TCGA: {orien_to_tcga_cindex:.4f if orien_to_tcga_cindex else 'Failed'}")
        print(f"    Average: {avg_cindex:.4f if avg_cindex else 'N/A'}")
    
    return result


def run_consensus_ksweep(
    k_values: List[int],
    gene_lists_dir: str,
    output_dir: str,
    seed: int = 42,
    source_params: str = 'results/hyperparam_FIXED_tcga_20251109_194909/best_params.json',
    target_params: str = 'results/hyperparam_FIXED_orien_20251109_195430/best_params.json'
):
    """
    Run consensus k-sweep evaluation using existing transfer_learning_trainer.py
    
    Args:
        k_values: List of k values to test
        gene_lists_dir: Directory containing consensus gene lists
        output_dir: Output directory for results
        seed: Random seed
        source_params: Path to TCGA hyperparameters
        target_params: Path to ORIEN hyperparameters
    """
    # Setup
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    gene_lists_path = Path(gene_lists_dir)
    
    print(f"{'='*80}")
    print("CONSENSUS K-SWEEP EVALUATION (Using Existing Infrastructure)")
    print(f"{'='*80}\n")
    
    print(f"Configuration:")
    print(f"  K values: {k_values}")
    print(f"  Gene lists: {gene_lists_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Seed: {seed}")
    print(f"  Source params (TCGA): {source_params}")
    print(f"  Target params (ORIEN): {target_params}")
    print(f"\n  Strategy: Using existing transfer_learning_trainer.py")
    print(f"  Benefits: Proven code, no compatibility issues, faster!")
    print()
    
    # Verify parameter files exist
    if not Path(source_params).exists():
        raise FileNotFoundError(f"Source params not found: {source_params}")
    if not Path(target_params).exists():
        raise FileNotFoundError(f"Target params not found: {target_params}")
    
    # Run evaluation for each k
    all_results = []
    
    for i, k in enumerate(k_values, 1):
        print(f"\n{'='*80}")
        print(f"K-VALUE {i}/{len(k_values)}: k = {k}")
        print(f"{'='*80}")
        
        try:
            result = evaluate_k_value(
                k=k,
                gene_lists_dir=gene_lists_path,
                output_dir=output_path,
                seed=seed,
                source_params=source_params,
                target_params=target_params,
                verbose=True
            )
            
            if result:
                all_results.append(result)
        
        except Exception as e:
            print(f"\n  ❌ Error processing k={k}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate summary
    if not all_results:
        print("\n❌ No results to summarize!")
        return
    
    print(f"\n{'='*80}")
    print("GENERATING COMPARISON TABLE")
    print(f"{'='*80}\n")
    
    # Create summary DataFrame
    summary_data = []
    for r in all_results:
        summary_data.append({
            'k': r['k'],
            'n_genes': r['n_genes'],
            'TCGA_to_ORIEN': f"{r['tcga_to_orien_cindex']:.4f}" if r['tcga_to_orien_cindex'] else 'Failed',
            'ORIEN_to_TCGA': f"{r['orien_to_tcga_cindex']:.4f}" if r['orien_to_tcga_cindex'] else 'Failed',
            'Average': f"{r['average_cindex']:.4f}" if r['average_cindex'] else 'N/A'
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('k')
    
    print(summary_df.to_string(index=False))
    
    # Find best k (if any completed successfully)
    valid_results = [r for r in all_results if r['average_cindex'] is not None]
    
    if valid_results:
        best_result = max(valid_results, key=lambda r: r['average_cindex'])
        
        print(f"\n{'='*80}")
        print("RECOMMENDATION")
        print(f"{'='*80}\n")
        
        print(f"🎯 OPTIMAL k = {best_result['k']}")
        print(f"   - Number of genes: {best_result['n_genes']}")
        print(f"   - TCGA→ORIEN C-index: {best_result['tcga_to_orien_cindex']:.4f}")
        print(f"   - ORIEN→TCGA C-index: {best_result['orien_to_tcga_cindex']:.4f}")
        print(f"   - Average C-index: {best_result['average_cindex']:.4f}")
    
    # Save results
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}\n")
    
    summary_df.to_csv(output_path / 'consensus_ksweep_summary.csv', index=False)
    print(f"✓ Summary table: consensus_ksweep_summary.csv")
    
    with open(output_path / 'consensus_ksweep_full_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✓ Full results: consensus_ksweep_full_results.json")
    
    if valid_results:
        with open(output_path / 'RECOMMENDATION.txt', 'w') as f:
            f.write(f"OPTIMAL K-VALUE SELECTION\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Recommended k: {best_result['k']}\n")
            f.write(f"Number of genes: {best_result['n_genes']}\n")
            f.write(f"TCGA→ORIEN C-index: {best_result['tcga_to_orien_cindex']:.4f}\n")
            f.write(f"ORIEN→TCGA C-index: {best_result['orien_to_tcga_cindex']:.4f}\n")
            f.write(f"Average C-index: {best_result['average_cindex']:.4f}\n\n")
            f.write(f"Gene list: {best_result['gene_file']}\n")
        
        print(f"✓ Recommendation: RECOMMENDATION.txt")
    
    print(f"\n{'='*80}")
    print("CONSENSUS K-SWEEP EVALUATION COMPLETE")
    print(f"{'='*80}\n")
    
    print(f"Results saved in: {output_dir}/")
    print(f"\nNext steps:")
    print(f"  1. Review: {output_dir}/consensus_ksweep_summary.csv")
    print(f"  2. Check: {output_dir}/RECOMMENDATION.txt")
    print(f"  3. Examine models in: {output_dir}/k*/")
    
    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Consensus k-sweep wrapper using existing transfer_learning_trainer.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Test representative k values (recommended)
  python scripts/consensus_ksweep_wrapper.py \\
      --k_values 90 95 100 120 140 150 \\
      --gene_lists_dir results/biomarker_ksweep_transfer/gene_lists \\
      --output_dir results/consensus_ksweep_evaluation
  
  # Specify custom hyperparameter files
  python scripts/consensus_ksweep_wrapper.py \\
      --k_values 90 100 120 \\
      --source_params results/hyperparam_FIXED_tcga_20251109_194909/best_params.json \\
      --target_params results/hyperparam_FIXED_orien_20251109_195430/best_params.json
        """
    )
    
    parser.add_argument('--k_values', type=int, nargs='+',
                       default=[90, 95, 100, 120, 140, 150],
                       help='K values to test')
    parser.add_argument('--gene_lists_dir', type=str,
                       default='results/biomarker_ksweep_transfer/gene_lists',
                       help='Directory with consensus gene lists')
    parser.add_argument('--output_dir', type=str,
                       default='results/consensus_ksweep_evaluation',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--source_params', type=str,
                       default='results/hyperparam_FIXED_tcga_20251109_194909/best_params.json',
                       help='TCGA hyperparameters JSON')
    parser.add_argument('--target_params', type=str,
                       default='results/hyperparam_FIXED_orien_20251109_195430/best_params.json',
                       help='ORIEN hyperparameters JSON')
    
    args = parser.parse_args()
    
    summary_df = run_consensus_ksweep(
        k_values=args.k_values,
        gene_lists_dir=args.gene_lists_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        source_params=args.source_params,
        target_params=args.target_params
    )
    
    print("\n✅ Consensus k-sweep evaluation completed successfully!")