"""
Step 3.4: Statistical Comparison & Analysis

Purpose: Compare all methods and establish transfer learning benefit.

Baselines:
1. Zero-shot (from Step 2.2B): Direct transfer without fine-tuning
2. Target-only (from Step 3.1): Train from scratch on target
3. Fine-tuned (from Step 3.3): Transfer learning with fine-tuning

Statistical Tests:
- Paired t-test: Fine-tuned vs Zero-shot
- Paired t-test: Fine-tuned vs Target-only
- Cohen's d effect size
- Relative improvement percentages

Output:
- Comparison tables
- Statistical test results
- Performance plots
- Effect size analysis
"""

import sys
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from scipy import stats


def setup_logging(output_dir):
    """Setup logging configuration"""
    log_file = output_dir / f"step3_4_analysis_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_baseline2_results(baseline_dir):
    """Load target-only baseline results from Step 3.1"""
    results = {'tcga': [], 'orien': []}
    
    for cohort in ['tcga', 'orien']:
        cohort_dir = baseline_dir / cohort
        for seed_file in sorted(cohort_dir.glob('seed*_results.json')):
            with open(seed_file, 'r') as f:
                data = json.load(f)
                results[cohort].append({
                    'seed': data['seed'],
                    'test_cindex': data['test_cindex']
                })
    
    return results


def load_finetuning_results(finetune_dir):
    """Load fine-tuning results from Step 3.3"""
    results = {'orien_to_tcga': [], 'tcga_to_orien': []}
    
    for direction in ['orien_to_tcga', 'tcga_to_orien']:
        direction_dir = finetune_dir / direction
        for seed_file in sorted(direction_dir.glob('seed*_results.json')):
            with open(seed_file, 'r') as f:
                data = json.load(f)
                results[direction].append({
                    'seed': data['seed'],
                    'test_cindex': data['finetune_test_cindex'],
                    'pretrain_cindex': data.get('pretrain_train_cindex', data.get('pretrain_valid_cindex', 0))
                })
    
    return results


def cohens_d(group1, group2):
    """
    Calculate Cohen's d effect size.
    
    Cohen's d interpretation:
    - 0.2: Small effect
    - 0.5: Medium effect
    - 0.8: Large effect
    
    Reference: Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences.
    """
    diff = np.mean(group1) - np.mean(group2)
    pooled_std = np.sqrt((np.var(group1, ddof=1) + np.var(group2, ddof=1)) / 2)
    return diff / pooled_std


def paired_ttest(group1, group2, alternative='two-sided'):
    """
    Perform paired t-test.
    
    Reference: Paired t-test is appropriate when comparing the same subjects
    under different conditions (same seeds, different methods).
    """
    t_stat, p_value = stats.ttest_rel(group1, group2, alternative=alternative)
    return t_stat, p_value


def calculate_statistics(baseline2, finetuned, logger):
    """Calculate comprehensive statistics comparing methods"""
    
    # Ensure same order of seeds
    baseline2_sorted = sorted(baseline2, key=lambda x: x['seed'])
    finetuned_sorted = sorted(finetuned, key=lambda x: x['seed'])
    
    baseline2_cindices = np.array([r['test_cindex'] for r in baseline2_sorted])
    finetuned_cindices = np.array([r['test_cindex'] for r in finetuned_sorted])
    
    # Basic statistics
    baseline2_mean = baseline2_cindices.mean()
    baseline2_std = baseline2_cindices.std(ddof=1)
    finetuned_mean = finetuned_cindices.mean()
    finetuned_std = finetuned_cindices.std(ddof=1)
    
    # Paired t-test (one-sided: fine-tuned > baseline)
    t_stat, p_value = paired_ttest(finetuned_cindices, baseline2_cindices, alternative='greater')
    
    # Cohen's d
    effect_size = cohens_d(finetuned_cindices, baseline2_cindices)
    
    # Relative improvement
    rel_improvement = (finetuned_mean - baseline2_mean) / baseline2_mean * 100
    
    # Absolute improvement
    abs_improvement = finetuned_mean - baseline2_mean
    
    logger.info(f"  Baseline2: {baseline2_mean:.4f} ± {baseline2_std:.4f}")
    logger.info(f"  Fine-tuned: {finetuned_mean:.4f} ± {finetuned_std:.4f}")
    logger.info(f"  Absolute improvement: {abs_improvement:.4f}")
    logger.info(f"  Relative improvement: {rel_improvement:.2f}%")
    logger.info(f"  Paired t-test: t={t_stat:.3f}, p={p_value:.6f}")
    logger.info(f"  Cohen's d: {effect_size:.3f}")
    
    return {
        'baseline2_mean': baseline2_mean,
        'baseline2_std': baseline2_std,
        'finetuned_mean': finetuned_mean,
        'finetuned_std': finetuned_std,
        'abs_improvement': abs_improvement,
        'rel_improvement': rel_improvement,
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': effect_size,
        'baseline2_cindices': baseline2_cindices,
        'finetuned_cindices': finetuned_cindices
    }


def create_comparison_table(stats_orien_to_tcga, stats_tcga_to_orien, 
                            zeroshot_results, output_dir, logger):
    """Create comprehensive comparison table"""
    
    logger.info("\nCreating comparison table...")
    
    # Note: Zero-shot results should come from Step 2.2B
    # For now, we'll create placeholders
    # User should update these values from their Step 2.2B results
    
    comparison_data = []
    
    # ORIEN→TCGA direction
    comparison_data.append({
        'Direction': 'ORIEN→TCGA',
        'Method': 'Zero-shot*',
        'C-index': zeroshot_results.get('orien_to_tcga_mean', 0.6256),
        'Std': zeroshot_results.get('orien_to_tcga_std', 0.0131),
        'vs_Zeroshot': '—',
        'vs_TargetOnly': '—',
        'P-value': '—'
    })
    
    comparison_data.append({
        'Direction': 'ORIEN→TCGA',
        'Method': 'Target-only',
        'C-index': stats_orien_to_tcga['baseline2_mean'],
        'Std': stats_orien_to_tcga['baseline2_std'],
        'vs_Zeroshot': f"+{((stats_orien_to_tcga['baseline2_mean'] - zeroshot_results.get('orien_to_tcga_mean', 0.6256)) / zeroshot_results.get('orien_to_tcga_mean', 0.6256) * 100):.1f}%",
        'vs_TargetOnly': '—',
        'P-value': '—'
    })
    
    comparison_data.append({
        'Direction': 'ORIEN→TCGA',
        'Method': 'Fine-tuned',
        'C-index': stats_orien_to_tcga['finetuned_mean'],
        'Std': stats_orien_to_tcga['finetuned_std'],
        'vs_Zeroshot': f"+{((stats_orien_to_tcga['finetuned_mean'] - zeroshot_results.get('orien_to_tcga_mean', 0.6256)) / zeroshot_results.get('orien_to_tcga_mean', 0.6256) * 100):.1f}%",
        'vs_TargetOnly': f"+{stats_orien_to_tcga['rel_improvement']:.1f}%",
        'P-value': f"{stats_orien_to_tcga['p_value']:.4f}"
    })
    
    # TCGA→ORIEN direction
    comparison_data.append({
        'Direction': 'TCGA→ORIEN',
        'Method': 'Zero-shot*',
        'C-index': zeroshot_results.get('tcga_to_orien_mean', 0.6093),
        'Std': zeroshot_results.get('tcga_to_orien_std', 0.0068),
        'vs_Zeroshot': '—',
        'vs_TargetOnly': '—',
        'P-value': '—'
    })
    
    comparison_data.append({
        'Direction': 'TCGA→ORIEN',
        'Method': 'Target-only',
        'C-index': stats_tcga_to_orien['baseline2_mean'],
        'Std': stats_tcga_to_orien['baseline2_std'],
        'vs_Zeroshot': f"+{((stats_tcga_to_orien['baseline2_mean'] - zeroshot_results.get('tcga_to_orien_mean', 0.6093)) / zeroshot_results.get('tcga_to_orien_mean', 0.6093) * 100):.1f}%",
        'vs_TargetOnly': '—',
        'P-value': '—'
    })
    
    comparison_data.append({
        'Direction': 'TCGA→ORIEN',
        'Method': 'Fine-tuned',
        'C-index': stats_tcga_to_orien['finetuned_mean'],
        'Std': stats_tcga_to_orien['finetuned_std'],
        'vs_Zeroshot': f"+{((stats_tcga_to_orien['finetuned_mean'] - zeroshot_results.get('tcga_to_orien_mean', 0.6093)) / zeroshot_results.get('tcga_to_orien_mean', 0.6093) * 100):.1f}%",
        'vs_TargetOnly': f"+{stats_tcga_to_orien['rel_improvement']:.1f}%",
        'P-value': f"{stats_tcga_to_orien['p_value']:.4f}"
    })
    
    df = pd.DataFrame(comparison_data)
    
    # Format C-index column
    df['C-index ± SD'] = df.apply(
        lambda row: f"{row['C-index']:.4f} ± {row['Std']:.4f}", axis=1
    )
    
    # Select final columns
    final_df = df[['Direction', 'Method', 'C-index ± SD', 'vs_Zeroshot', 'vs_TargetOnly', 'P-value']]
    
    # Save
    final_df.to_csv(output_dir / 'comparison_table.csv', index=False)
    
    logger.info("\nComparison Table:")
    logger.info(f"\n{final_df.to_string(index=False)}")
    logger.info("\n* Zero-shot results from Step 2.2B (placeholder values - update with actual results)")
    
    return final_df


def create_effect_size_report(stats_orien_to_tcga, stats_tcga_to_orien, output_dir, logger):
    """Create detailed effect size report"""
    
    logger.info("\nCreating effect size report...")
    
    effect_sizes = []
    
    for direction, stats in [('ORIEN→TCGA', stats_orien_to_tcga), 
                             ('TCGA→ORIEN', stats_tcga_to_orien)]:
        cohens_d = stats['cohens_d']
        
        # Interpret effect size
        if abs(cohens_d) < 0.2:
            interpretation = "Negligible"
        elif abs(cohens_d) < 0.5:
            interpretation = "Small"
        elif abs(cohens_d) < 0.8:
            interpretation = "Medium"
        else:
            interpretation = "Large"
        
        effect_sizes.append({
            'Direction': direction,
            'Cohens_d': cohens_d,
            'Interpretation': interpretation,
            't_statistic': stats['t_statistic'],
            'p_value': stats['p_value'],
            'abs_improvement': stats['abs_improvement'],
            'rel_improvement': stats['rel_improvement']
        })
    
    df = pd.DataFrame(effect_sizes)
    df.to_csv(output_dir / 'effect_sizes.csv', index=False)
    
    # Save detailed text report
    with open(output_dir / 'statistical_tests.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("Transfer Learning Statistical Analysis\n")
        f.write("="*60 + "\n\n")
        
        for direction, stats in [('ORIEN→TCGA', stats_orien_to_tcga), 
                                 ('TCGA→ORIEN', stats_tcga_to_orien)]:
            f.write(f"\n{direction}:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Target-only:  {stats['baseline2_mean']:.4f} ± {stats['baseline2_std']:.4f}\n")
            f.write(f"Fine-tuned:   {stats['finetuned_mean']:.4f} ± {stats['finetuned_std']:.4f}\n\n")
            
            f.write(f"Improvement:\n")
            f.write(f"  Absolute: +{stats['abs_improvement']:.4f}\n")
            f.write(f"  Relative: +{stats['rel_improvement']:.2f}%\n\n")
            
            f.write(f"Statistical Tests:\n")
            f.write(f"  Paired t-test: t = {stats['t_statistic']:.3f}\n")
            f.write(f"  P-value: {stats['p_value']:.6f}\n")
            f.write(f"  Significance: {'***' if stats['p_value'] < 0.001 else '**' if stats['p_value'] < 0.01 else '*' if stats['p_value'] < 0.05 else 'ns'}\n\n")
            
            f.write(f"Effect Size:\n")
            f.write(f"  Cohen's d: {stats['cohens_d']:.3f}\n")
            f.write(f"  Interpretation: {df[df['Direction']==direction]['Interpretation'].values[0]}\n\n")
    
    logger.info(f"\nEffect size report saved to {output_dir / 'effect_sizes.csv'}")
    logger.info(f"Statistical tests saved to {output_dir / 'statistical_tests.txt'}")
    
    return df


def create_performance_plot(stats_orien_to_tcga, stats_tcga_to_orien, output_dir, logger):
    """Create performance comparison plot"""
    
    logger.info("\nCreating performance comparison plot...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ORIEN→TCGA
    ax = axes[0]
    methods = ['Target-only', 'Fine-tuned']
    means = [stats_orien_to_tcga['baseline2_mean'], stats_orien_to_tcga['finetuned_mean']]
    stds = [stats_orien_to_tcga['baseline2_std'], stats_orien_to_tcga['finetuned_std']]
    colors = ['#3498db', '#e74c3c']
    
    bars = ax.bar(methods, means, yerr=stds, capsize=5, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('C-index', fontsize=12)
    ax.set_title('ORIEN→TCGA Transfer Learning', fontsize=14, fontweight='bold')
    ax.set_ylim([0.5, 0.8])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{mean:.4f}±{std:.4f}',
                ha='center', va='bottom', fontsize=10)
    
    # Add significance annotation
    if stats_orien_to_tcga['p_value'] < 0.001:
        sig = '***'
    elif stats_orien_to_tcga['p_value'] < 0.01:
        sig = '**'
    elif stats_orien_to_tcga['p_value'] < 0.05:
        sig = '*'
    else:
        sig = 'ns'
    
    y_max = max(means) + max(stds) + 0.05
    ax.plot([0, 1], [y_max, y_max], 'k-', linewidth=1)
    ax.text(0.5, y_max + 0.01, sig, ha='center', va='bottom', fontsize=14)
    
    # TCGA→ORIEN
    ax = axes[1]
    means = [stats_tcga_to_orien['baseline2_mean'], stats_tcga_to_orien['finetuned_mean']]
    stds = [stats_tcga_to_orien['baseline2_std'], stats_tcga_to_orien['finetuned_std']]
    
    bars = ax.bar(methods, means, yerr=stds, capsize=5, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('C-index', fontsize=12)
    ax.set_title('TCGA→ORIEN Transfer Learning', fontsize=14, fontweight='bold')
    ax.set_ylim([0.5, 0.8])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{mean:.4f}±{std:.4f}',
                ha='center', va='bottom', fontsize=10)
    
    # Add significance annotation
    if stats_tcga_to_orien['p_value'] < 0.001:
        sig = '***'
    elif stats_tcga_to_orien['p_value'] < 0.01:
        sig = '**'
    elif stats_tcga_to_orien['p_value'] < 0.05:
        sig = '*'
    else:
        sig = 'ns'
    
    y_max = max(means) + max(stds) + 0.05
    ax.plot([0, 1], [y_max, y_max], 'k-', linewidth=1)
    ax.text(0.5, y_max + 0.01, sig, ha='center', va='bottom', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Performance plot saved to {output_dir / 'performance_comparison_plot.png'}")


def create_improvement_analysis(stats_orien_to_tcga, stats_tcga_to_orien, output_dir, logger):
    """Create detailed improvement analysis"""
    
    logger.info("\nCreating improvement analysis...")
    
    improvements = []
    
    for direction, stats in [('ORIEN→TCGA', stats_orien_to_tcga), 
                             ('TCGA→ORIEN', stats_tcga_to_orien)]:
        
        # Per-seed improvements
        for i, (baseline, finetuned) in enumerate(zip(stats['baseline2_cindices'], 
                                                       stats['finetuned_cindices'])):
            improvements.append({
                'Direction': direction,
                'Seed': [42, 123, 456, 789, 1011][i],
                'Target_only': baseline,
                'Fine_tuned': finetuned,
                'Absolute_improvement': finetuned - baseline,
                'Relative_improvement_%': (finetuned - baseline) / baseline * 100
            })
    
    df = pd.DataFrame(improvements)
    df.to_csv(output_dir / 'improvement_analysis.csv', index=False)
    
    logger.info(f"Improvement analysis saved to {output_dir / 'improvement_analysis.csv'}")
    
    # Print summary
    logger.info("\nPer-seed improvements:")
    for direction in ['ORIEN→TCGA', 'TCGA→ORIEN']:
        logger.info(f"\n{direction}:")
        direction_df = df[df['Direction'] == direction]
        for _, row in direction_df.iterrows():
            logger.info(f"  Seed {row['Seed']}: {row['Target_only']:.4f} → {row['Fine_tuned']:.4f} "
                       f"(+{row['Relative_improvement_%']:.2f}%)")
    
    return df


def main():
    # Configuration
    BASELINE2_DIR = Path("results_v2/03_transfer_learning/k155/baseline_target_only")
    FINETUNE_DIR = Path("results_v2/03_transfer_learning/k155/finetuned")
    OUTPUT_DIR = Path("results_v2/03_transfer_learning/k155/analysis")
    
    # Zero-shot results from Step 2.2B (PLACEHOLDER - update with actual values)
    ZEROSHOT_RESULTS = {
        'orien_to_tcga_mean': 0.6236,  # From your fine-tune output
        'orien_to_tcga_std': 0.0590,
        'tcga_to_orien_mean': 0.6172,
        'tcga_to_orien_std': 0.0347
    }
    
    # Setup
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    logger = setup_logging(OUTPUT_DIR)
    
    logger.info("="*60)
    logger.info("Step 3.4: Statistical Comparison & Analysis")
    logger.info("="*60)
    logger.info(f"Output: {OUTPUT_DIR}")
    
    # Load results
    logger.info("\nLoading results...")
    baseline2_results = load_baseline2_results(BASELINE2_DIR)
    finetuning_results = load_finetuning_results(FINETUNE_DIR)
    
    logger.info(f"  Baseline2 (Target-only):")
    logger.info(f"    TCGA: {len(baseline2_results['tcga'])} seeds")
    logger.info(f"    ORIEN: {len(baseline2_results['orien'])} seeds")
    logger.info(f"  Fine-tuning:")
    logger.info(f"    ORIEN→TCGA: {len(finetuning_results['orien_to_tcga'])} seeds")
    logger.info(f"    TCGA→ORIEN: {len(finetuning_results['tcga_to_orien'])} seeds")
    
    # Calculate statistics for each direction
    logger.info("\n" + "="*60)
    logger.info("ORIEN→TCGA Analysis")
    logger.info("="*60)
    stats_orien_to_tcga = calculate_statistics(
        baseline2_results['tcga'],
        finetuning_results['orien_to_tcga'],
        logger
    )
    
    logger.info("\n" + "="*60)
    logger.info("TCGA→ORIEN Analysis")
    logger.info("="*60)
    stats_tcga_to_orien = calculate_statistics(
        baseline2_results['orien'],
        finetuning_results['tcga_to_orien'],
        logger
    )
    
    # Create outputs
    comparison_table = create_comparison_table(
        stats_orien_to_tcga, stats_tcga_to_orien,
        ZEROSHOT_RESULTS, OUTPUT_DIR, logger
    )
    
    effect_sizes = create_effect_size_report(
        stats_orien_to_tcga, stats_tcga_to_orien,
        OUTPUT_DIR, logger
    )
    
    create_performance_plot(
        stats_orien_to_tcga, stats_tcga_to_orien,
        OUTPUT_DIR, logger
    )
    
    improvement_analysis = create_improvement_analysis(
        stats_orien_to_tcga, stats_tcga_to_orien,
        OUTPUT_DIR, logger
    )
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("Summary")
    logger.info("="*60)
    logger.info("\nKey Findings:")
    
    for direction, stats in [('ORIEN→TCGA', stats_orien_to_tcga), 
                             ('TCGA→ORIEN', stats_tcga_to_orien)]:
        logger.info(f"\n{direction}:")
        logger.info(f"  ✓ Fine-tuned: {stats['finetuned_mean']:.4f} ± {stats['finetuned_std']:.4f}")
        logger.info(f"  ✓ Improvement: +{stats['rel_improvement']:.2f}% over target-only")
        logger.info(f"  ✓ P-value: {stats['p_value']:.6f} {'***' if stats['p_value'] < 0.001 else '**' if stats['p_value'] < 0.01 else '*' if stats['p_value'] < 0.05 else 'ns'}")
        logger.info(f"  ✓ Cohen's d: {stats['cohens_d']:.3f} ({effect_sizes[effect_sizes['Direction']==direction]['Interpretation'].values[0]} effect)")
    
    logger.info(f"\n{'='*60}")
    logger.info("Step 3.4 Complete!")
    logger.info(f"{'='*60}")
    logger.info(f"\nAll results saved to: {OUTPUT_DIR}")
    logger.info("\nGenerated files:")
    logger.info("  - comparison_table.csv")
    logger.info("  - statistical_tests.txt")
    logger.info("  - effect_sizes.csv")
    logger.info("  - improvement_analysis.csv")
    logger.info("  - performance_comparison_plot.png")
    
    logger.info("\n⚠️  NOTE: Update ZEROSHOT_RESULTS in the script with actual values from Step 2.2B")


if __name__ == "__main__":
    main()
