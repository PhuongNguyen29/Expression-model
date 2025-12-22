"""
Sign Consistency Exploratory Analysis for Integrated Gradients Attributions

Purpose: Examine sign consistency of IG attributions across 5 seeds and 2 cohorts
to determine appropriate filtering before k-selection.

Methodology:
- Within-cohort sign consistency: proportion of samples with same sign as mean
- Across-seed agreement: majority vote (≥3/5 seeds)
- Cross-cohort sign match: consensus sign matches between TCGA and ORIEN

References:
- Sundararajan et al. (2017) - Axiomatic Attribution for Deep Networks
- Bernau et al. (2014) - Cross-study validation for genomic biomarkers
- Zou & Hastie (2005) - Elastic net regularization

Author: Phuong
Date: December 2024
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

SEEDS = [42, 123, 456, 789, 1011]
COHORTS = ['tcga', 'orien']

# Input paths
ATTRIBUTION_BASE_DIR = Path("results_v2/06_importance_methods")
COX_VALIDATION_FILE = Path("data/raw/biomarker_tables_pen_cox.csv")

# Output paths
OUTPUT_DIR = Path("results_v2/06_importance_methods/sign_consistency_analysis")
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_attribution_file(seed: int, cohort: str) -> pd.DataFrame:
    """
    Load per-sample IG attribution file for a given seed and cohort.
    
    Args:
        seed: Random seed (42, 123, 456, 789, 1011)
        cohort: 'tcga' or 'orien'
    
    Returns:
        DataFrame with samples as rows, genes as columns
    """
    filepath = ATTRIBUTION_BASE_DIR / f"seed_{seed}" / "per_sample_attributions" / f"{cohort}_attributions_per_sample.csv"
    
    if not filepath.exists():
        raise FileNotFoundError(f"Attribution file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Set sample_id as index if present
    if 'sample_id' in df.columns:
        df = df.set_index('sample_id')
    
    return df


def load_all_attributions() -> Dict[str, Dict[int, pd.DataFrame]]:
    """
    Load all attribution files for all seeds and cohorts.
    
    Returns:
        Nested dict: {cohort: {seed: DataFrame}}
    """
    print("Loading attribution files...")
    
    attributions = {cohort: {} for cohort in COHORTS}
    
    for cohort in COHORTS:
        for seed in SEEDS:
            print(f"  Loading {cohort} seed {seed}...")
            attributions[cohort][seed] = load_attribution_file(seed, cohort)
    
    # Verify consistency
    gene_sets = []
    for cohort in COHORTS:
        for seed in SEEDS:
            gene_sets.append(set(attributions[cohort][seed].columns))
    
    common_genes = set.intersection(*gene_sets)
    print(f"\nCommon genes across all files: {len(common_genes)}")
    
    if len(common_genes) != len(gene_sets[0]):
        print(f"  Warning: Some genes not present in all files")
    
    return attributions, sorted(list(common_genes))


def load_cox_validation() -> pd.DataFrame:
    """
    Load Cox coefficient validation data from Chapter 2.
    
    Returns:
        DataFrame with gene, TCGA_Coef, ORIEN_Coef columns
    """
    if not COX_VALIDATION_FILE.exists():
        print(f"Warning: Cox validation file not found: {COX_VALIDATION_FILE}")
        return None
    
    df = pd.read_csv(COX_VALIDATION_FILE)
    print(f"\nLoaded Cox validation: {len(df)} genes")
    
    return df


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def compute_within_seed_stats(attributions: Dict, genes: List[str]) -> pd.DataFrame:
    """
    Compute within-seed statistics for each gene.
    
    For each seed × cohort × gene:
    - mean_ig: Mean IG across samples
    - sign_consistency: Proportion of samples with same sign as mean
    
    Returns:
        DataFrame with genes as rows, multi-level columns for seed/cohort/metric
    """
    print("\nComputing within-seed statistics...")
    
    results = []
    
    for gene in genes:
        row = {'gene': gene}
        
        for cohort in COHORTS:
            for seed in SEEDS:
                df = attributions[cohort][seed]
                
                if gene not in df.columns:
                    continue
                
                values = df[gene].values
                mean_ig = np.mean(values)
                
                # Sign consistency: proportion agreeing with mean sign
                if mean_ig != 0:
                    mean_sign = np.sign(mean_ig)
                    sample_signs = np.sign(values)
                    # Handle zeros: count as agreeing if mean is positive, disagreeing if negative
                    agreement = np.sum(sample_signs == mean_sign) + np.sum(values == 0) * 0.5
                    sign_consistency = agreement / len(values)
                else:
                    sign_consistency = 0.5  # Undefined if mean is exactly 0
                
                col_prefix = f"{cohort}_seed{seed}"
                row[f"{col_prefix}_mean"] = mean_ig
                row[f"{col_prefix}_sign_consistency"] = sign_consistency
        
        results.append(row)
    
    return pd.DataFrame(results)


def compute_across_seed_stats(within_seed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute across-seed agreement using majority vote.
    
    For each gene × cohort:
    - n_positive_seeds: Number of seeds with positive mean IG
    - n_negative_seeds: Number of seeds with negative mean IG
    - seed_agreement: max(n_pos, n_neg) / 5
    - consensus_sign: +1 if majority positive, -1 if majority negative
    
    Returns:
        DataFrame with genes as rows
    """
    print("\nComputing across-seed agreement (majority vote)...")
    
    results = []
    
    for _, row in within_seed_df.iterrows():
        gene = row['gene']
        gene_result = {'gene': gene}
        
        for cohort in COHORTS:
            # Collect mean signs across seeds
            signs = []
            for seed in SEEDS:
                mean_col = f"{cohort}_seed{seed}_mean"
                if mean_col in row:
                    mean_val = row[mean_col]
                    if not np.isnan(mean_val):
                        signs.append(np.sign(mean_val))
            
            if len(signs) == 0:
                gene_result[f"{cohort}_n_positive_seeds"] = np.nan
                gene_result[f"{cohort}_n_negative_seeds"] = np.nan
                gene_result[f"{cohort}_seed_agreement"] = np.nan
                gene_result[f"{cohort}_consensus_sign"] = np.nan
                continue
            
            n_positive = sum(1 for s in signs if s > 0)
            n_negative = sum(1 for s in signs if s < 0)
            n_zero = sum(1 for s in signs if s == 0)
            
            n_seeds = len(signs)
            seed_agreement = max(n_positive, n_negative) / n_seeds
            
            # Majority vote for consensus sign
            if n_positive > n_negative:
                consensus_sign = 1
            elif n_negative > n_positive:
                consensus_sign = -1
            else:
                consensus_sign = 0  # Tie
            
            gene_result[f"{cohort}_n_positive_seeds"] = n_positive
            gene_result[f"{cohort}_n_negative_seeds"] = n_negative
            gene_result[f"{cohort}_seed_agreement"] = seed_agreement
            gene_result[f"{cohort}_consensus_sign"] = consensus_sign
        
        results.append(gene_result)
    
    return pd.DataFrame(results)


def compute_cross_cohort_agreement(across_seed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cross-cohort sign agreement.
    
    For each gene:
    - cross_cohort_match: True if TCGA and ORIEN have same consensus sign
    
    Returns:
        DataFrame with cross-cohort analysis
    """
    print("\nComputing cross-cohort sign agreement...")
    
    df = across_seed_df.copy()
    
    # Cross-cohort match
    df['cross_cohort_match'] = (
        (df['tcga_consensus_sign'] == df['orien_consensus_sign']) & 
        (df['tcga_consensus_sign'] != 0) & 
        (df['orien_consensus_sign'] != 0)
    )
    
    # Both positive
    df['both_positive'] = (df['tcga_consensus_sign'] == 1) & (df['orien_consensus_sign'] == 1)
    
    # Both negative
    df['both_negative'] = (df['tcga_consensus_sign'] == -1) & (df['orien_consensus_sign'] == -1)
    
    # Opposite signs
    df['opposite_signs'] = (
        ((df['tcga_consensus_sign'] == 1) & (df['orien_consensus_sign'] == -1)) |
        ((df['tcga_consensus_sign'] == -1) & (df['orien_consensus_sign'] == 1))
    )
    
    return df


def compute_mean_ig_across_seeds(attributions: Dict, genes: List[str]) -> pd.DataFrame:
    """
    Compute mean IG across all seeds for scatter plot.
    
    Returns:
        DataFrame with gene, tcga_mean_ig, orien_mean_ig
    """
    print("\nComputing mean IG across seeds...")
    
    results = []
    
    for gene in genes:
        row = {'gene': gene}
        
        for cohort in COHORTS:
            seed_means = []
            for seed in SEEDS:
                df = attributions[cohort][seed]
                if gene in df.columns:
                    seed_means.append(df[gene].mean())
            
            if seed_means:
                row[f"{cohort}_mean_ig"] = np.mean(seed_means)
            else:
                row[f"{cohort}_mean_ig"] = np.nan
        
        results.append(row)
    
    return pd.DataFrame(results)


def validate_against_cox(cross_cohort_df: pd.DataFrame, cox_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate IG signs against Cox coefficient signs for the 20 consensus genes.
    
    Returns:
        DataFrame with validation results
    """
    if cox_df is None:
        print("\nSkipping Cox validation (no Cox data)")
        return None
    
    print("\nValidating IG signs against Cox coefficients...")
    
    results = []
    
    for _, cox_row in cox_df.iterrows():
        gene = cox_row['Gene']
        
        # Find gene in cross-cohort results
        ig_row = cross_cohort_df[cross_cohort_df['gene'] == gene]
        
        if len(ig_row) == 0:
            print(f"  Warning: Gene {gene} not found in IG results")
            continue
        
        ig_row = ig_row.iloc[0]
        
        # Cox signs
        cox_tcga_sign = np.sign(cox_row['TCGA_Coef'])
        cox_orien_sign = np.sign(cox_row['ORIEN_Coef'])
        
        # IG consensus signs
        ig_tcga_sign = ig_row['tcga_consensus_sign']
        ig_orien_sign = ig_row['orien_consensus_sign']
        
        result = {
            'gene': gene,
            'cox_tcga_sign': int(cox_tcga_sign),
            'cox_orien_sign': int(cox_orien_sign),
            'ig_tcga_consensus_sign': int(ig_tcga_sign) if not np.isnan(ig_tcga_sign) else np.nan,
            'ig_orien_consensus_sign': int(ig_orien_sign) if not np.isnan(ig_orien_sign) else np.nan,
            'tcga_match': cox_tcga_sign == ig_tcga_sign,
            'orien_match': cox_orien_sign == ig_orien_sign,
            'both_match': (cox_tcga_sign == ig_tcga_sign) and (cox_orien_sign == ig_orien_sign)
        }
        
        results.append(result)
    
    return pd.DataFrame(results)


def compute_within_cohort_sign_consistency_avg(within_seed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average within-cohort sign consistency across seeds.
    
    Returns:
        DataFrame with gene, tcga_avg_sign_consistency, orien_avg_sign_consistency
    """
    results = []
    
    for _, row in within_seed_df.iterrows():
        gene = row['gene']
        gene_result = {'gene': gene}
        
        for cohort in COHORTS:
            consistencies = []
            for seed in SEEDS:
                col = f"{cohort}_seed{seed}_sign_consistency"
                if col in row and not np.isnan(row[col]):
                    consistencies.append(row[col])
            
            if consistencies:
                gene_result[f"{cohort}_avg_sign_consistency"] = np.mean(consistencies)
            else:
                gene_result[f"{cohort}_avg_sign_consistency"] = np.nan
        
        results.append(gene_result)
    
    return pd.DataFrame(results)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_within_cohort_sign_consistency(consistency_df: pd.DataFrame, output_path: Path):
    """
    Figure 1: Histogram of within-cohort sign consistency scores.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    tcga_vals = consistency_df['tcga_avg_sign_consistency'].dropna()
    orien_vals = consistency_df['orien_avg_sign_consistency'].dropna()
    
    bins = np.linspace(0.5, 1.0, 26)
    
    ax.hist(tcga_vals, bins=bins, alpha=0.6, label=f'TCGA (n={len(tcga_vals)})', color='steelblue', edgecolor='black')
    ax.hist(orien_vals, bins=bins, alpha=0.6, label=f'ORIEN (n={len(orien_vals)})', color='coral', edgecolor='black')
    
    ax.axvline(x=0.7, color='red', linestyle='--', linewidth=2, label='Threshold 0.7')
    ax.axvline(x=0.8, color='darkred', linestyle='--', linewidth=2, label='Threshold 0.8')
    
    ax.set_xlabel('Average Within-Cohort Sign Consistency', fontsize=12)
    ax.set_ylabel('Number of Genes', fontsize=12)
    ax.set_title('Figure 1: Within-Cohort Sign Consistency Distribution\n(Averaged across 5 seeds)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add summary stats
    tcga_above_07 = (tcga_vals >= 0.7).sum()
    orien_above_07 = (orien_vals >= 0.7).sum()
    textstr = f'Genes with consistency ≥0.7:\nTCGA: {tcga_above_07} ({100*tcga_above_07/len(tcga_vals):.1f}%)\nORIEN: {orien_above_07} ({100*orien_above_07/len(orien_vals):.1f}%)'
    ax.text(0.52, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_across_seed_agreement(across_seed_df: pd.DataFrame, output_path: Path):
    """
    Figure 2: Histogram of across-seed agreement scores.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    tcga_vals = across_seed_df['tcga_seed_agreement'].dropna()
    orien_vals = across_seed_df['orien_seed_agreement'].dropna()
    
    # Possible values: 0.6, 0.8, 1.0 (for 5 seeds: 3/5, 4/5, 5/5)
    bins = [0.55, 0.7, 0.9, 1.05]
    
    width = 0.12
    x_tcga = np.array([0.6, 0.8, 1.0]) - width/2
    x_orien = np.array([0.6, 0.8, 1.0]) + width/2
    
    tcga_counts = [(tcga_vals == 0.6).sum(), (tcga_vals == 0.8).sum(), (tcga_vals == 1.0).sum()]
    orien_counts = [(orien_vals == 0.6).sum(), (orien_vals == 0.8).sum(), (orien_vals == 1.0).sum()]
    
    ax.bar(x_tcga, tcga_counts, width=width, label=f'TCGA (n={len(tcga_vals)})', color='steelblue', edgecolor='black')
    ax.bar(x_orien, orien_counts, width=width, label=f'ORIEN (n={len(orien_vals)})', color='coral', edgecolor='black')
    
    ax.set_xlabel('Across-Seed Agreement (Majority Vote)', fontsize=12)
    ax.set_ylabel('Number of Genes', fontsize=12)
    ax.set_title('Figure 2: Across-Seed Sign Agreement Distribution\n(5 seeds: 3/5=0.6, 4/5=0.8, 5/5=1.0)', fontsize=13, fontweight='bold')
    ax.set_xticks([0.6, 0.8, 1.0])
    ax.set_xticklabels(['3/5 seeds\n(0.6)', '4/5 seeds\n(0.8)', '5/5 seeds\n(1.0)'])
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add counts on bars
    for i, (tc, oc) in enumerate(zip(tcga_counts, orien_counts)):
        ax.text(x_tcga[i], tc + 2, str(tc), ha='center', fontsize=9)
        ax.text(x_orien[i], oc + 2, str(oc), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_tcga_vs_orien_seed_agreement(across_seed_df: pd.DataFrame, output_path: Path):
    """
    Figure 3: Scatter plot of TCGA vs ORIEN seed agreement.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    tcga_vals = across_seed_df['tcga_seed_agreement'].values
    orien_vals = across_seed_df['orien_seed_agreement'].values
    
    # Add jitter for visibility (values are discrete)
    jitter = 0.02
    tcga_jitter = tcga_vals + np.random.uniform(-jitter, jitter, len(tcga_vals))
    orien_jitter = orien_vals + np.random.uniform(-jitter, jitter, len(orien_vals))
    
    # Color by whether both have high agreement
    colors = []
    for t, o in zip(tcga_vals, orien_vals):
        if t >= 0.8 and o >= 0.8:
            colors.append('green')
        elif t >= 0.6 and o >= 0.6:
            colors.append('orange')
        else:
            colors.append('red')
    
    ax.scatter(tcga_jitter, orien_jitter, c=colors, alpha=0.6, s=30, edgecolors='black', linewidths=0.5)
    
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0.8, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('TCGA Seed Agreement', fontsize=12)
    ax.set_ylabel('ORIEN Seed Agreement', fontsize=12)
    ax.set_title('Figure 3: TCGA vs ORIEN Across-Seed Agreement\n(308 genes)', fontsize=13, fontweight='bold')
    ax.set_xlim(0.55, 1.05)
    ax.set_ylim(0.55, 1.05)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label=f'Both ≥0.8: {sum(1 for c in colors if c == "green")}'),
        Patch(facecolor='orange', label=f'Both ≥0.6: {sum(1 for c in colors if c == "orange")}'),
        Patch(facecolor='red', label=f'At least one <0.6: {sum(1 for c in colors if c == "red")}')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_cross_cohort_contingency(cross_cohort_df: pd.DataFrame, output_path: Path):
    """
    Figure 4: 2x2 heatmap of cross-cohort sign contingency.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Count contingency
    both_pos = cross_cohort_df['both_positive'].sum()
    both_neg = cross_cohort_df['both_negative'].sum()
    
    tcga_pos_orien_neg = ((cross_cohort_df['tcga_consensus_sign'] == 1) & 
                          (cross_cohort_df['orien_consensus_sign'] == -1)).sum()
    tcga_neg_orien_pos = ((cross_cohort_df['tcga_consensus_sign'] == -1) & 
                          (cross_cohort_df['orien_consensus_sign'] == 1)).sum()
    
    contingency = np.array([
        [both_pos, tcga_pos_orien_neg],
        [tcga_neg_orien_pos, both_neg]
    ])
    
    sns.heatmap(contingency, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['ORIEN +', 'ORIEN -'],
                yticklabels=['TCGA +', 'TCGA -'],
                ax=ax, annot_kws={'size': 16}, cbar_kws={'label': 'Count'})
    
    ax.set_xlabel('ORIEN Consensus Sign', fontsize=12)
    ax.set_ylabel('TCGA Consensus Sign', fontsize=12)
    ax.set_title('Figure 4: Cross-Cohort Sign Contingency\n(Majority Vote Consensus)', fontsize=13, fontweight='bold')
    
    # Summary
    total = contingency.sum()
    matching = both_pos + both_neg
    opposite = tcga_pos_orien_neg + tcga_neg_orien_pos
    
    textstr = f'Sign Match: {matching} ({100*matching/total:.1f}%)\nOpposite: {opposite} ({100*opposite/total:.1f}%)'
    ax.text(1.35, 0.5, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_mean_ig_scatter(mean_ig_df: pd.DataFrame, cross_cohort_df: pd.DataFrame, output_path: Path):
    """
    Figure 5: Scatter plot of mean IG (TCGA vs ORIEN) with quadrant coloring.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    tcga_vals = mean_ig_df['tcga_mean_ig'].values
    orien_vals = mean_ig_df['orien_mean_ig'].values
    
    # Color by quadrant
    colors = []
    for t, o in zip(tcga_vals, orien_vals):
        if t > 0 and o > 0:
            colors.append('green')  # Both positive
        elif t < 0 and o < 0:
            colors.append('blue')   # Both negative
        else:
            colors.append('red')    # Opposite signs
    
    ax.scatter(tcga_vals, orien_vals, c=colors, alpha=0.6, s=40, edgecolors='black', linewidths=0.5)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    
    ax.set_xlabel('Mean IG - TCGA (averaged across 5 seeds)', fontsize=12)
    ax.set_ylabel('Mean IG - ORIEN (averaged across 5 seeds)', fontsize=12)
    ax.set_title('Figure 5: Cross-Cohort Mean IG Attribution\n(308 genes, quadrant coloring)', fontsize=13, fontweight='bold')
    
    # Count quadrants
    q1 = sum(1 for c in colors if c == 'green')  # Both +
    q3 = sum(1 for c in colors if c == 'blue')   # Both -
    opposite = sum(1 for c in colors if c == 'red')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label=f'Both + (Q1): {q1}'),
        Patch(facecolor='blue', label=f'Both - (Q3): {q3}'),
        Patch(facecolor='red', label=f'Opposite (Q2/Q4): {opposite}')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    # Add correlation
    valid_mask = ~(np.isnan(tcga_vals) | np.isnan(orien_vals))
    if valid_mask.sum() > 2:
        from scipy.stats import spearmanr
        rho, pval = spearmanr(tcga_vals[valid_mask], orien_vals[valid_mask])
        ax.text(0.05, 0.95, f'Spearman ρ = {rho:.3f}\np = {pval:.2e}', 
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_cox_validation(cox_validation_df: pd.DataFrame, output_path: Path):
    """
    Figure 6: Bar chart of Cox validation results.
    """
    if cox_validation_df is None or len(cox_validation_df) == 0:
        print("  Skipping Cox validation plot (no data)")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    # TCGA match
    tcga_match = cox_validation_df['tcga_match'].sum()
    tcga_total = len(cox_validation_df)
    axes[0].bar(['Match', 'Mismatch'], [tcga_match, tcga_total - tcga_match], 
                color=['green', 'red'], edgecolor='black')
    axes[0].set_ylabel('Number of Genes', fontsize=11)
    axes[0].set_title(f'TCGA: IG vs Cox Sign\n({tcga_match}/{tcga_total} = {100*tcga_match/tcga_total:.1f}% match)', 
                      fontsize=12, fontweight='bold')
    axes[0].set_ylim(0, tcga_total + 2)
    
    # ORIEN match
    orien_match = cox_validation_df['orien_match'].sum()
    axes[1].bar(['Match', 'Mismatch'], [orien_match, tcga_total - orien_match], 
                color=['green', 'red'], edgecolor='black')
    axes[1].set_ylabel('Number of Genes', fontsize=11)
    axes[1].set_title(f'ORIEN: IG vs Cox Sign\n({orien_match}/{tcga_total} = {100*orien_match/tcga_total:.1f}% match)', 
                      fontsize=12, fontweight='bold')
    axes[1].set_ylim(0, tcga_total + 2)
    
    # Both match
    both_match = cox_validation_df['both_match'].sum()
    axes[2].bar(['Both Match', 'At Least One Mismatch'], [both_match, tcga_total - both_match], 
                color=['green', 'red'], edgecolor='black')
    axes[2].set_ylabel('Number of Genes', fontsize=11)
    axes[2].set_title(f'Both Cohorts: IG vs Cox Sign\n({both_match}/{tcga_total} = {100*both_match/tcga_total:.1f}% match)', 
                      fontsize=12, fontweight='bold')
    axes[2].set_ylim(0, tcga_total + 2)
    
    plt.suptitle('Figure 6: IG Sign Validation Against Cox Coefficients\n(20 Chapter 2 Consensus Genes)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# SUMMARY FUNCTIONS
# =============================================================================

def compute_summary_statistics(
    within_seed_df: pd.DataFrame,
    across_seed_df: pd.DataFrame,
    cross_cohort_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    cox_validation_df: pd.DataFrame
) -> Dict:
    """
    Compute summary statistics for the analysis.
    """
    summary = {
        'total_genes': len(cross_cohort_df),
        'seeds_used': SEEDS,
        'n_seeds': len(SEEDS)
    }
    
    # Within-cohort sign consistency
    for cohort in COHORTS:
        col = f"{cohort}_avg_sign_consistency"
        vals = consistency_df[col].dropna()
        summary[f'{cohort}_within_cohort_consistency'] = {
            'mean': float(vals.mean()),
            'std': float(vals.std()),
            'median': float(vals.median()),
            'genes_above_0.7': int((vals >= 0.7).sum()),
            'genes_above_0.8': int((vals >= 0.8).sum())
        }
    
    # Across-seed agreement
    for cohort in COHORTS:
        col = f"{cohort}_seed_agreement"
        vals = across_seed_df[col].dropna()
        summary[f'{cohort}_seed_agreement'] = {
            'genes_5_of_5': int((vals == 1.0).sum()),
            'genes_4_of_5': int((vals == 0.8).sum()),
            'genes_3_of_5': int((vals == 0.6).sum())
        }
    
    # Cross-cohort sign agreement
    summary['cross_cohort_sign_agreement'] = {
        'both_positive': int(cross_cohort_df['both_positive'].sum()),
        'both_negative': int(cross_cohort_df['both_negative'].sum()),
        'matching_signs': int(cross_cohort_df['cross_cohort_match'].sum()),
        'opposite_signs': int(cross_cohort_df['opposite_signs'].sum())
    }
    
    # Combined filters
    tcga_stable = across_seed_df['tcga_seed_agreement'] >= 0.8
    orien_stable = across_seed_df['orien_seed_agreement'] >= 0.8
    both_stable = tcga_stable & orien_stable
    sign_match = cross_cohort_df['cross_cohort_match']
    
    summary['combined_filters'] = {
        'both_seed_agreement_0.8': int(both_stable.sum()),
        'cross_cohort_sign_match': int(sign_match.sum()),
        'both_stable_and_sign_match': int((both_stable & sign_match).sum())
    }
    
    # Cox validation
    if cox_validation_df is not None and len(cox_validation_df) > 0:
        summary['cox_validation'] = {
            'n_genes': len(cox_validation_df),
            'tcga_match': int(cox_validation_df['tcga_match'].sum()),
            'orien_match': int(cox_validation_df['orien_match'].sum()),
            'both_match': int(cox_validation_df['both_match'].sum()),
            'tcga_match_rate': float(cox_validation_df['tcga_match'].mean()),
            'orien_match_rate': float(cox_validation_df['orien_match'].mean()),
            'both_match_rate': float(cox_validation_df['both_match'].mean())
        }
    
    return summary


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution function.
    """
    print("="*70)
    print("SIGN CONSISTENCY EXPLORATORY ANALYSIS")
    print("Integrated Gradients Attributions - 5 Seeds × 2 Cohorts")
    print("="*70)
    
    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    attributions, genes = load_all_attributions()
    cox_df = load_cox_validation()
    
    print(f"\nAnalyzing {len(genes)} genes across {len(SEEDS)} seeds × {len(COHORTS)} cohorts")
    
    # Compute statistics
    print("\n" + "="*70)
    print("COMPUTING STATISTICS")
    print("="*70)
    
    within_seed_df = compute_within_seed_stats(attributions, genes)
    across_seed_df = compute_across_seed_stats(within_seed_df)
    cross_cohort_df = compute_cross_cohort_agreement(across_seed_df)
    consistency_df = compute_within_cohort_sign_consistency_avg(within_seed_df)
    mean_ig_df = compute_mean_ig_across_seeds(attributions, genes)
    cox_validation_df = validate_against_cox(cross_cohort_df, cox_df)
    
    # Merge for full table
    full_table = within_seed_df.merge(across_seed_df, on='gene')
    full_table = full_table.merge(cross_cohort_df[['gene', 'cross_cohort_match', 'both_positive', 'both_negative', 'opposite_signs']], on='gene')
    full_table = full_table.merge(consistency_df, on='gene')
    full_table = full_table.merge(mean_ig_df, on='gene')
    
    # Save tables
    print("\n" + "="*70)
    print("SAVING TABLES")
    print("="*70)
    
    full_table.to_csv(TABLES_DIR / "sign_analysis_full_table.csv", index=False)
    print(f"  Saved: {TABLES_DIR / 'sign_analysis_full_table.csv'}")
    
    cross_cohort_summary = cross_cohort_df[['gene', 'tcga_consensus_sign', 'orien_consensus_sign', 
                                             'tcga_seed_agreement', 'orien_seed_agreement',
                                             'cross_cohort_match', 'both_positive', 'both_negative', 'opposite_signs']]
    cross_cohort_summary.to_csv(TABLES_DIR / "cross_cohort_sign_summary.csv", index=False)
    print(f"  Saved: {TABLES_DIR / 'cross_cohort_sign_summary.csv'}")
    
    if cox_validation_df is not None:
        cox_validation_df.to_csv(TABLES_DIR / "cox_validation_table.csv", index=False)
        print(f"  Saved: {TABLES_DIR / 'cox_validation_table.csv'}")
    
    # Generate figures
    print("\n" + "="*70)
    print("GENERATING FIGURES")
    print("="*70)
    
    plot_within_cohort_sign_consistency(consistency_df, FIGURES_DIR / "fig1_within_cohort_sign_consistency.png")
    plot_across_seed_agreement(across_seed_df, FIGURES_DIR / "fig2_across_seed_agreement.png")
    plot_tcga_vs_orien_seed_agreement(across_seed_df, FIGURES_DIR / "fig3_tcga_vs_orien_seed_agreement.png")
    plot_cross_cohort_contingency(cross_cohort_df, FIGURES_DIR / "fig4_cross_cohort_contingency.png")
    plot_mean_ig_scatter(mean_ig_df, cross_cohort_df, FIGURES_DIR / "fig5_mean_ig_scatter.png")
    plot_cox_validation(cox_validation_df, FIGURES_DIR / "fig6_cox_validation.png")
    
    # Compute and save summary
    print("\n" + "="*70)
    print("COMPUTING SUMMARY")
    print("="*70)
    
    summary = compute_summary_statistics(
        within_seed_df, across_seed_df, cross_cohort_df, 
        consistency_df, cox_validation_df
    )
    
    with open(OUTPUT_DIR / "sign_analysis_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {OUTPUT_DIR / 'sign_analysis_summary.json'}")
    
    # Print key findings
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    
    print(f"\n1. WITHIN-COHORT SIGN CONSISTENCY (averaged across 5 seeds):")
    for cohort in COHORTS:
        stats = summary[f'{cohort}_within_cohort_consistency']
        print(f"   {cohort.upper()}: mean={stats['mean']:.3f}, median={stats['median']:.3f}")
        print(f"          Genes ≥0.7: {stats['genes_above_0.7']}, Genes ≥0.8: {stats['genes_above_0.8']}")
    
    print(f"\n2. ACROSS-SEED AGREEMENT (majority vote):")
    for cohort in COHORTS:
        stats = summary[f'{cohort}_seed_agreement']
        print(f"   {cohort.upper()}: 5/5 seeds={stats['genes_5_of_5']}, 4/5={stats['genes_4_of_5']}, 3/5={stats['genes_3_of_5']}")
    
    print(f"\n3. CROSS-COHORT SIGN AGREEMENT:")
    cc_stats = summary['cross_cohort_sign_agreement']
    total = summary['total_genes']
    print(f"   Both positive: {cc_stats['both_positive']} ({100*cc_stats['both_positive']/total:.1f}%)")
    print(f"   Both negative: {cc_stats['both_negative']} ({100*cc_stats['both_negative']/total:.1f}%)")
    print(f"   MATCHING SIGNS: {cc_stats['matching_signs']} ({100*cc_stats['matching_signs']/total:.1f}%)")
    print(f"   Opposite signs: {cc_stats['opposite_signs']} ({100*cc_stats['opposite_signs']/total:.1f}%)")
    
    print(f"\n4. COMBINED FILTERS:")
    cf_stats = summary['combined_filters']
    print(f"   Both cohorts seed agreement ≥0.8: {cf_stats['both_seed_agreement_0.8']}")
    print(f"   Cross-cohort sign match: {cf_stats['cross_cohort_sign_match']}")
    print(f"   BOTH criteria: {cf_stats['both_stable_and_sign_match']}")
    
    if 'cox_validation' in summary:
        print(f"\n5. COX VALIDATION ({summary['cox_validation']['n_genes']} genes):")
        cv_stats = summary['cox_validation']
        print(f"   TCGA IG sign matches Cox: {cv_stats['tcga_match']}/{cv_stats['n_genes']} ({100*cv_stats['tcga_match_rate']:.1f}%)")
        print(f"   ORIEN IG sign matches Cox: {cv_stats['orien_match']}/{cv_stats['n_genes']} ({100*cv_stats['orien_match_rate']:.1f}%)")
        print(f"   BOTH match: {cv_stats['both_match']}/{cv_stats['n_genes']} ({100*cv_stats['both_match_rate']:.1f}%)")
    
    print(f"\n6. RECOMMENDED NEW GENE POOL FOR K-SELECTION:")
    print(f"   Sign-consistent genes (cross-cohort match): {cc_stats['matching_signs']}")
    print(f"   → New k search range: [10, {cc_stats['matching_signs']}]")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*70)
    
    return summary


if __name__ == "__main__":
    summary = main()
