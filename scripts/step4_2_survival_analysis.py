#!/usr/bin/env python3
"""
Step 4.2: Survival Analysis
- Load risk scores from Step 4.1
- Stratify patients into risk groups
- Generate Kaplan-Meier survival curves
- Calculate log-rank p-values and hazard ratios
- Compare all methods
"""

import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.utils import median_survival_times

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Plot settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


INPUT_DIR = Path('results_v2/04_final_models')
OUTPUT_DIR = Path('results_v2/04_final_models/survival_analysis')
CV_RESULTS_FILE = Path('results_v2/03_transfer_learning/analysis/performance_summary.csv')

# ============================================================
# Helper Functions
# ============================================================

def stratify_patients(risk_scores, method='median'):
    """
    Stratify patients into risk groups
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


def plot_km_curve(durations, events, groups, title, output_file):
    """
    Plot Kaplan-Meier survival curves
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    kmf = KaplanMeierFitter()
    
    unique_groups = np.unique(groups)
    colors = ['#e74c3c', '#3498db', '#2ecc71']  # Red, Blue, Green
    
    for i, group in enumerate(unique_groups):
        mask = (groups == group)
        kmf.fit(
            durations[mask],
            events[mask],
            label=group
        )
        kmf.plot_survival_function(ax=ax, ci_show=True, color=colors[i])
    
    ax.set_xlabel('Time (months)', fontsize=12)
    ax.set_ylabel('Survival Probability', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  Saved KM curve to {output_file.name}")


def calculate_hazard_ratio(durations, events, groups, reference='Low Risk'):
    """
    Calculate hazard ratio between risk groups
    """
    from lifelines import CoxPHFitter
    
    # Prepare data
    df = pd.DataFrame({
        'duration': durations,
        'event': events,
        'group': groups
    })
    
    # Encode groups (reference = 0)
    group_encoding = {group: i for i, group in enumerate(df['group'].unique())}
    if reference in group_encoding:
        # Make reference group = 0
        ref_value = group_encoding[reference]
        group_encoding = {
            k: (v - ref_value) % len(group_encoding) 
            for k, v in group_encoding.items()
        }
    
    df['group_encoded'] = df['group'].map(group_encoding)
    
    # Fit Cox model
    cph = CoxPHFitter()
    cph.fit(df[['duration', 'event', 'group_encoded']], 
            duration_col='duration', event_col='event')
    
    # Extract HR and CI
    hr = np.exp(cph.params_['group_encoded'])
    ci = np.exp(cph.confidence_intervals_['group_encoded'].values)
    
    return {
        'HR': float(hr),
        'CI_lower': float(ci[0]),
        'CI_upper': float(ci[1]),
        'p_value': float(cph.summary['p']['group_encoded'])
    }


def analyze_model(risk_scores_file, model_name, cohort_name):
    """
    Perform survival analysis for a single model
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"{model_name} on {cohort_name}")
    logger.info(f"{'='*60}")
    
    # Load risk scores
    df = pd.read_csv(risk_scores_file)
    risk_scores = df['risk_score'].values
    durations = df['time'].values
    events = df['event'].values.astype(bool)
    
    logger.info(f"  Loaded {len(risk_scores)} patients")
    logger.info(f"  Events: {events.sum()} ({100*events.mean():.1f}%)")
    
    # Stratify patients
    groups = stratify_patients(risk_scores, method='median')
    
    # Log-rank test
    high_mask = (groups == 'High Risk')
    low_mask = (groups == 'Low Risk')
    
    logrank_result = logrank_test(
        durations[high_mask], durations[low_mask],
        events[high_mask], events[low_mask]
    )
    
    logger.info(f"\n  Log-rank test:")
    logger.info(f"    High risk: n={high_mask.sum()}, events={events[high_mask].sum()}")
    logger.info(f"    Low risk: n={low_mask.sum()}, events={events[low_mask].sum()}")
    logger.info(f"    p-value: {logrank_result.p_value:.6f}")
    logger.info(f"    Significant: {'Yes' if logrank_result.p_value < 0.05 else 'No'}")
    
    # Hazard ratio
    hr_result = calculate_hazard_ratio(durations, events, groups, reference='Low Risk')
    logger.info(f"\n  Hazard Ratio (High vs Low):")
    logger.info(f"    HR: {hr_result['HR']:.3f} ({hr_result['CI_lower']:.3f}-{hr_result['CI_upper']:.3f})")
    logger.info(f"    p-value: {hr_result['p_value']:.6f}")
    
    # Median survival times
    kmf = KaplanMeierFitter()
    medians = {}
    for group in ['High Risk', 'Low Risk']:
        mask = (groups == group)
        kmf.fit(durations[mask], events[mask])
        median_surv = kmf.median_survival_time_
        medians[group] = median_surv
        logger.info(f"  {group} median survival: {median_surv:.1f} months")
    
    # Plot KM curve
    output_file = OUTPUT_DIR / f'{cohort_name}_{model_name.replace(" ", "_")}_KM.png'
    plot_km_curve(
        durations, events, groups,
        f'{model_name} - {cohort_name}\nLog-rank p={logrank_result.p_value:.4f}',
        output_file
    )
    
    # Return results
    return {
        'model': model_name,
        'cohort': cohort_name,
        'n_patients': int(len(risk_scores)),
        'n_events': int(events.sum()),
        'logrank_p': float(logrank_result.p_value),
        'logrank_significant': logrank_result.p_value < 0.05,
        'hazard_ratio': hr_result['HR'],
        'hr_ci_lower': hr_result['CI_lower'],
        'hr_ci_upper': hr_result['CI_upper'],
        'hr_p_value': hr_result['p_value'],
        'median_survival_high': float(medians['High Risk']),
        'median_survival_low': float(medians['Low Risk'])
    }


def create_comparison_plot(all_results):
    """
    Create comparison plot of all methods
    """
    # Separate by cohort
    tcga_results = [r for r in all_results if r['cohort'] == 'TCGA']
    orien_results = [r for r in all_results if r['cohort'] == 'ORIEN']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for ax, results, cohort in zip(axes, [tcga_results, orien_results], ['TCGA', 'ORIEN']):
        # Load all risk scores for this cohort
        for result in results:
            model_name = result['model']
            file_pattern = f"{cohort}_{model_name.replace(' ', '').replace('→', 'to')}_risk_scores.csv"
            risk_file = INPUT_DIR / file_pattern
            
            if not risk_file.exists():
                continue
            
            df = pd.read_csv(risk_file)
            groups = stratify_patients(df['risk_score'].values)
            
            # Plot KM for this model
            kmf = KaplanMeierFitter()
            for group in ['High Risk', 'Low Risk']:
                mask = (groups == group)
                kmf.fit(df['time'][mask], df['event'][mask], 
                       label=f"{model_name} - {group}")
                kmf.plot_survival_function(ax=ax, ci_show=False)
        
        ax.set_xlabel('Time (months)', fontsize=12)
        ax.set_ylabel('Survival Probability', fontsize=12)
        ax.set_title(f'{cohort} - All Methods Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'all_methods_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"\nSaved comparison plot: all_methods_comparison.png")


def create_final_comparison_table(bootstrap_results, survival_results, cv_results=None):
    """
    Create comprehensive comparison table
    """
    comparison_data = []
    
    # Load bootstrap results
    with open(bootstrap_results, 'r') as f:
        bootstrap_data = json.load(f)
    
    # Load CV results if available
    cv_data = {}
    if cv_results and cv_results.exists():
        cv_df = pd.read_csv(cv_results)
        for _, row in cv_df.iterrows():
            key = f"{row['Cohort']}_{row['Method'].replace(' ', '')}"
            cv_data[key] = {
                'cv_cindex': row.get('C-index', None),
                'cv_std': row.get('SD', None)
            }
    
    # Combine all results
    for key, boot_result in bootstrap_data.items():
        # Find matching survival result
        surv_result = next(
            (r for r in survival_results 
             if r['cohort'] == boot_result['cohort'] and 
             r['model'] == boot_result['method']),
            None
        )
        
        row = {
            'Method': boot_result['method'],
            'Cohort': boot_result['cohort'],
            'N': boot_result['n_samples'],
            'Events': boot_result['n_events'],
        }
        
        # Add CV results if available
        cv_key = f"{boot_result['cohort']}_{boot_result['method'].replace(' ', '')}"
        if cv_key in cv_data:
            row['CV_Cindex'] = f"{cv_data[cv_key]['cv_cindex']:.4f}"
            row['CV_SD'] = f"±{cv_data[cv_key]['cv_std']:.4f}"
        
        # Add bootstrap results
        boot = boot_result['bootstrap_results']
        row['Apparent_Cindex'] = f"{boot['apparent']:.4f}"
        row['Corrected_Cindex'] = f"{boot['corrected']:.4f}"
        row['Corrected_CI'] = f"({boot['corrected_ci_95'][0]:.4f}-{boot['corrected_ci_95'][1]:.4f})"
        
        # Add survival results
        if surv_result:
            row['Logrank_P'] = f"{surv_result['logrank_p']:.4f}"
            row['HR'] = f"{surv_result['hazard_ratio']:.2f}"
            row['HR_CI'] = f"({surv_result['hr_ci_lower']:.2f}-{surv_result['hr_ci_upper']:.2f})"
        
        comparison_data.append(row)
    
    # Create DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save
    comparison_df.to_csv(OUTPUT_DIR / 'comprehensive_comparison.csv', index=False)
    
    logger.info("\n" + "="*60)
    logger.info("Comprehensive Comparison Table")
    logger.info("="*60)
    logger.info("\n" + comparison_df.to_string(index=False))
    
    return comparison_df


# ============================================================
# Main Analysis Function
# ============================================================

def run_survival_analysis():
    """
    Run survival analysis for all models
    """
    logger.info("=" * 60)
    logger.info("Step 4.2: Survival Analysis")
    logger.info("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find all risk score files
    risk_score_files = list(INPUT_DIR.glob('*_risk_scores.csv'))
    logger.info(f"\nFound {len(risk_score_files)} risk score files")
    
    all_results = []
    
    # Analyze each model
    for risk_file in sorted(risk_score_files):
        # Parse filename: {COHORT}_{METHOD}_risk_scores.csv
        filename = risk_file.stem  # Remove .csv
        filename = filename.replace('_risk_scores', '')
        
        # Split into cohort and method
        if filename.startswith('TCGA'):
            cohort = 'TCGA'
            method = filename[5:]  # Remove 'TCGA_'
        elif filename.startswith('ORIEN'):
            cohort = 'ORIEN'
            method = filename[6:]  # Remove 'ORIEN_'
        else:
            logger.warning(f"Could not parse filename: {risk_file.name}")
            continue
        
        # Clean method name
        method = method.replace('_', ' ')
        method = method.replace('ORIENtoTCGA', 'ORIEN→TCGA')
        method = method.replace('TCGAtoORIEN', 'TCGA→ORIEN')
        
        # Analyze this model
        result = analyze_model(risk_file, method, cohort)
        all_results.append(result)
    
    # Save survival analysis results
    survival_df = pd.DataFrame(all_results)
    survival_df.to_csv(OUTPUT_DIR / 'survival_analysis_results.csv', index=False)
    
    logger.info("\n" + "="*60)
    logger.info("Survival Analysis Summary")
    logger.info("="*60)
    logger.info("\n" + survival_df.to_string(index=False))
    
    # Create comparison plot
    logger.info("\n" + "="*60)
    logger.info("Creating Comparison Plots")
    logger.info("="*60)
    create_comparison_plot(all_results)
    
    # Create comprehensive comparison table
    logger.info("\n" + "="*60)
    logger.info("Creating Comprehensive Comparison")
    logger.info("="*60)
    bootstrap_file = INPUT_DIR / 'bootstrap_results.json'
    create_final_comparison_table(bootstrap_file, all_results, CV_RESULTS_FILE)
    
    logger.info("\n" + "="*60)
    logger.info("Step 4.2 Complete!")
    logger.info("="*60)
    logger.info(f"\nAll results saved to: {OUTPUT_DIR}")
    logger.info("\nGenerated files:")
    logger.info("  - survival_analysis_results.csv")
    logger.info("  - comprehensive_comparison.csv")
    logger.info("  - *_KM.png (8 Kaplan-Meier plots)")
    logger.info("  - all_methods_comparison.png")


if __name__ == '__main__':
    run_survival_analysis()