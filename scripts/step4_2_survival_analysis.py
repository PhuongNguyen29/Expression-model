#!/usr/bin/env python3
"""
Step 4.2: Survival Analysis

Purpose: Perform survival analysis on risk scores from transfer learning models.

Analysis:
- Stratify patients into High/Low risk groups (median split)
- Generate Kaplan-Meier survival curves
- Calculate log-rank p-values
- Calculate hazard ratios with 95% CI
- Compare TCGA and ORIEN cohorts

Configuration: k=155 (87 consensus genes)
"""

import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Plot settings
plt.style.use('seaborn-v0_8-whitegrid')

# ============================================================
# Configuration for k=155
# ============================================================

K_VALUE = 155
INPUT_DIR = Path(f'results_v2/04_final_models/k{K_VALUE}')
OUTPUT_DIR = Path(f'results_v2/04_final_models/k{K_VALUE}/survival_analysis')


# ============================================================
# Helper Functions
# ============================================================

def stratify_patients(risk_scores, method='median'):
    """
    Stratify patients into risk groups.
    
    Args:
        risk_scores: array of risk scores
        method: 'median' or 'tertile'
    
    Returns:
        array of group labels
    """
    if method == 'median':
        threshold = np.median(risk_scores)
        groups = np.where(risk_scores > threshold, 'High Risk', 'Low Risk')
    elif method == 'tertile':
        thresholds = np.percentile(risk_scores, [33.33, 66.67])
        groups = np.where(
            risk_scores <= thresholds[0], 'Low Risk',
            np.where(risk_scores <= thresholds[1], 'Medium Risk', 'High Risk')
        )
    else:
        raise ValueError(f"Unknown stratification method: {method}")
    
    return groups


def calculate_hazard_ratio(durations, events, groups, reference='Low Risk'):
    """
    Calculate hazard ratio between risk groups using Cox regression.
    
    Args:
        durations: survival times
        events: event indicators
        groups: risk group labels
        reference: reference group for HR calculation
    
    Returns:
        dict with HR, CI, and p-value
    """
    # Prepare data
    df = pd.DataFrame({
        'duration': durations,
        'event': events,
        'high_risk': (groups == 'High Risk').astype(int)
    })
    
    # Fit Cox model
    cph = CoxPHFitter()
    try:
        cph.fit(df, duration_col='duration', event_col='event')
        
        hr = np.exp(cph.params_['high_risk'])
        ci = np.exp(cph.confidence_intervals_.loc['high_risk'].values)
        p_value = cph.summary['p']['high_risk']
        
        return {
            'HR': float(hr),
            'CI_lower': float(ci[0]),
            'CI_upper': float(ci[1]),
            'p_value': float(p_value)
        }
    except Exception as e:
        logger.warning(f"Cox regression failed: {e}")
        return {
            'HR': np.nan,
            'CI_lower': np.nan,
            'CI_upper': np.nan,
            'p_value': np.nan
        }


def plot_km_curve(durations, events, groups, title, output_file, cohort_info=None):
    """
    Plot Kaplan-Meier survival curves with professional styling.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    kmf = KaplanMeierFitter()
    
    # Colors for risk groups
    colors = {'High Risk': '#e74c3c', 'Low Risk': '#3498db'}
    
    # Calculate log-rank test
    high_mask = (groups == 'High Risk')
    low_mask = (groups == 'Low Risk')
    
    logrank_result = logrank_test(
        durations[high_mask], durations[low_mask],
        events[high_mask], events[low_mask]
    )
    
    # Plot each group
    for group in ['Low Risk', 'High Risk']:
        mask = (groups == group)
        kmf.fit(durations[mask], events[mask], label=group)
        kmf.plot_survival_function(
            ax=ax, 
            ci_show=True, 
            color=colors[group],
            linewidth=2
        )
    
    # Styling
    ax.set_xlabel('Time (months)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Survival Probability', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower left', fontsize=11, frameon=True)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    
    # Add p-value annotation
    p_text = f'Log-rank p = {logrank_result.p_value:.4f}'
    if logrank_result.p_value < 0.001:
        p_text = 'Log-rank p < 0.001'
    ax.text(0.95, 0.95, p_text, transform=ax.transAxes, 
            fontsize=11, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add cohort info if provided
    if cohort_info:
        info_text = f"N={cohort_info['n']}, Events={cohort_info['events']}"
        ax.text(0.95, 0.85, info_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  Saved KM curve: {output_file.name}")
    
    return logrank_result.p_value


def analyze_cohort(risk_scores_file, model_name, cohort_name):
    """
    Perform complete survival analysis for one cohort.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"{model_name} - {cohort_name}")
    logger.info(f"{'='*60}")
    
    # Load risk scores
    df = pd.read_csv(risk_scores_file)
    risk_scores = df['risk_score'].values
    durations = df['time'].values
    events = df['event'].values.astype(bool)
    
    n_patients = len(risk_scores)
    n_events = events.sum()
    
    logger.info(f"  Patients: {n_patients}")
    logger.info(f"  Events: {n_events} ({100*n_events/n_patients:.1f}%)")
    
    # Stratify patients
    groups = stratify_patients(risk_scores, method='median')
    
    # Group statistics
    high_mask = (groups == 'High Risk')
    low_mask = (groups == 'Low Risk')
    
    logger.info(f"\n  Risk Stratification (Median Split):")
    logger.info(f"    High Risk: n={high_mask.sum()}, events={events[high_mask].sum()}")
    logger.info(f"    Low Risk: n={low_mask.sum()}, events={events[low_mask].sum()}")
    
    # Log-rank test
    logrank_result = logrank_test(
        durations[high_mask], durations[low_mask],
        events[high_mask], events[low_mask]
    )
    
    logger.info(f"\n  Log-rank Test:")
    logger.info(f"    Test statistic: {logrank_result.test_statistic:.3f}")
    logger.info(f"    P-value: {logrank_result.p_value:.6f}")
    logger.info(f"    Significant (p<0.05): {'Yes ***' if logrank_result.p_value < 0.05 else 'No'}")
    
    # Hazard ratio
    hr_result = calculate_hazard_ratio(durations, events, groups)
    
    logger.info(f"\n  Hazard Ratio (High vs Low Risk):")
    logger.info(f"    HR: {hr_result['HR']:.3f} (95% CI: {hr_result['CI_lower']:.3f}-{hr_result['CI_upper']:.3f})")
    logger.info(f"    P-value: {hr_result['p_value']:.6f}")
    
    # Median survival times
    kmf = KaplanMeierFitter()
    medians = {}
    
    logger.info(f"\n  Median Survival Time:")
    for group in ['High Risk', 'Low Risk']:
        mask = (groups == group)
        kmf.fit(durations[mask], events[mask])
        median_surv = kmf.median_survival_time_
        medians[group] = median_surv
        if np.isinf(median_surv):
            logger.info(f"    {group}: Not reached")
        else:
            logger.info(f"    {group}: {median_surv:.1f} months")
    
    # Plot KM curve
    output_file = OUTPUT_DIR / f'{cohort_name}_{model_name.replace(" ", "_").replace("→", "to")}_KM.png'
    cohort_info = {'n': n_patients, 'events': n_events}
    
    plot_km_curve(
        durations, events, groups,
        f'{cohort_name} - {model_name}\nTransfer Learning Survival Analysis',
        output_file,
        cohort_info
    )
    
    # Return results
    return {
        'model': model_name,
        'cohort': cohort_name,
        'n_patients': int(n_patients),
        'n_events': int(n_events),
        'n_high_risk': int(high_mask.sum()),
        'n_low_risk': int(low_mask.sum()),
        'events_high_risk': int(events[high_mask].sum()),
        'events_low_risk': int(events[low_mask].sum()),
        'logrank_statistic': float(logrank_result.test_statistic),
        'logrank_p': float(logrank_result.p_value),
        'logrank_significant': logrank_result.p_value < 0.05,
        'hazard_ratio': hr_result['HR'],
        'hr_ci_lower': hr_result['CI_lower'],
        'hr_ci_upper': hr_result['CI_upper'],
        'hr_p_value': hr_result['p_value'],
        'median_survival_high': float(medians['High Risk']) if not np.isinf(medians['High Risk']) else None,
        'median_survival_low': float(medians['Low Risk']) if not np.isinf(medians['Low Risk']) else None
    }


def create_combined_km_plot(all_results):
    """
    Create a combined KM plot for both cohorts.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    cohort_files = {
        'TCGA': INPUT_DIR / 'TCGA_ORIENtoTCGA_risk_scores.csv',
        'ORIEN': INPUT_DIR / 'ORIEN_TCGAtoORIEN_risk_scores.csv'
    }
    
    titles = {
        'TCGA': 'TCGA Cohort (ORIEN→TCGA Transfer)',
        'ORIEN': 'ORIEN Cohort (TCGA→ORIEN Transfer)'
    }
    
    colors = {'High Risk': '#e74c3c', 'Low Risk': '#3498db'}
    
    for ax, (cohort, risk_file) in zip(axes, cohort_files.items()):
        if not risk_file.exists():
            logger.warning(f"Risk file not found: {risk_file}")
            continue
        
        df = pd.read_csv(risk_file)
        risk_scores = df['risk_score'].values
        durations = df['time'].values
        events = df['event'].values.astype(bool)
        
        groups = stratify_patients(risk_scores)
        
        # Log-rank test
        high_mask = (groups == 'High Risk')
        low_mask = (groups == 'Low Risk')
        logrank_result = logrank_test(
            durations[high_mask], durations[low_mask],
            events[high_mask], events[low_mask]
        )
        
        # Plot
        kmf = KaplanMeierFitter()
        for group in ['Low Risk', 'High Risk']:
            mask = (groups == group)
            kmf.fit(durations[mask], events[mask], label=group)
            kmf.plot_survival_function(ax=ax, ci_show=True, color=colors[group], linewidth=2)
        
        ax.set_xlabel('Time (months)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Survival Probability', fontsize=12, fontweight='bold')
        ax.set_title(titles[cohort], fontsize=13, fontweight='bold')
        ax.legend(loc='lower left', fontsize=11)
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        
        # P-value annotation
        p_text = f'p = {logrank_result.p_value:.4f}' if logrank_result.p_value >= 0.001 else 'p < 0.001'
        ax.text(0.95, 0.95, p_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle(f'Transfer Learning Survival Analysis (k={K_VALUE}, {87} genes)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'combined_KM_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"\nSaved combined KM plot: combined_KM_plot.png")


def create_summary_table(all_results):
    """
    Create formatted summary table for dissertation.
    """
    summary_data = []
    
    for result in all_results:
        summary_data.append({
            'Cohort': result['cohort'],
            'Model': result['model'],
            'N': result['n_patients'],
            'Events': result['n_events'],
            'Event Rate': f"{100*result['n_events']/result['n_patients']:.1f}%",
            'HR (95% CI)': f"{result['hazard_ratio']:.2f} ({result['hr_ci_lower']:.2f}-{result['hr_ci_upper']:.2f})",
            'Log-rank p': f"{result['logrank_p']:.4f}" if result['logrank_p'] >= 0.0001 else '<0.0001',
            'Significant': '***' if result['logrank_p'] < 0.001 else '**' if result['logrank_p'] < 0.01 else '*' if result['logrank_p'] < 0.05 else 'ns'
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(OUTPUT_DIR / 'survival_summary_table.csv', index=False)
    
    logger.info("\n" + "="*60)
    logger.info("Survival Analysis Summary Table")
    logger.info("="*60)
    logger.info("\n" + summary_df.to_string(index=False))
    
    return summary_df


# ============================================================
# Main Analysis Function
# ============================================================

def run_survival_analysis():
    """
    Run complete survival analysis for transfer learning models.
    """
    logger.info("=" * 60)
    logger.info(f"Step 4.2: Survival Analysis (k={K_VALUE})")
    logger.info("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Define models to analyze
    models_to_analyze = [
        {
            'file': INPUT_DIR / 'TCGA_ORIENtoTCGA_risk_scores.csv',
            'model_name': 'ORIEN→TCGA Transfer',
            'cohort': 'TCGA'
        },
        {
            'file': INPUT_DIR / 'ORIEN_TCGAtoORIEN_risk_scores.csv',
            'model_name': 'TCGA→ORIEN Transfer',
            'cohort': 'ORIEN'
        }
    ]
    
    # Check for required files
    logger.info("\nChecking for risk score files...")
    for model in models_to_analyze:
        if model['file'].exists():
            logger.info(f"  ✓ Found: {model['file'].name}")
        else:
            logger.error(f"  ✗ Missing: {model['file'].name}")
            logger.error("Please run Step 4.1 first to generate risk scores.")
            return None
    
    # Analyze each model
    all_results = []
    
    for model in models_to_analyze:
        result = analyze_cohort(
            model['file'],
            model['model_name'],
            model['cohort']
        )
        all_results.append(result)
    
    # Save detailed results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / 'survival_analysis_detailed.csv', index=False)
    
    # Create combined KM plot
    logger.info("\n" + "="*60)
    logger.info("Creating Combined Plots")
    logger.info("="*60)
    create_combined_km_plot(all_results)
    
    # Create summary table
    summary_df = create_summary_table(all_results)
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("Key Findings")
    logger.info("="*60)
    
    for result in all_results:
        sig_marker = '***' if result['logrank_p'] < 0.001 else '**' if result['logrank_p'] < 0.01 else '*' if result['logrank_p'] < 0.05 else ''
        logger.info(f"\n{result['cohort']} ({result['model']}):")
        logger.info(f"  Log-rank p = {result['logrank_p']:.4f} {sig_marker}")
        logger.info(f"  HR = {result['hazard_ratio']:.2f} (95% CI: {result['hr_ci_lower']:.2f}-{result['hr_ci_upper']:.2f})")
        if result['logrank_significant']:
            logger.info(f"  → Significant risk stratification achieved")
        else:
            logger.info(f"  → Risk stratification not significant at α=0.05")
    
    logger.info("\n" + "="*60)
    logger.info("Step 4.2 Complete!")
    logger.info("="*60)
    logger.info(f"\nAll results saved to: {OUTPUT_DIR}")
    logger.info("\nGenerated files:")
    logger.info("  - survival_analysis_detailed.csv")
    logger.info("  - survival_summary_table.csv")
    logger.info("  - TCGA_ORIEN_to_TCGA_Transfer_KM.png")
    logger.info("  - ORIEN_TCGA_to_ORIEN_Transfer_KM.png")
    logger.info("  - combined_KM_plot.png")
    
    return all_results


if __name__ == '__main__':
    run_survival_analysis()
