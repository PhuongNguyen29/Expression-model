#!/usr/bin/env python3
"""
Step 4.2: Survival Analysis with 5-Year AUC

Purpose: Perform survival analysis on risk scores from transfer learning models.

Analysis:
- Stratify patients into High/Low risk groups (median split)
- Generate Kaplan-Meier survival curves (truncated at 5 years)
- Calculate time-dependent AUC at 5 years
- Calculate log-rank p-values and hazard ratios
- Create publication-quality 4-panel figure

Configuration: k=155 (87 consensus genes)
"""

import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# Configuration for k=155
# ============================================================

K_VALUE = 155
INPUT_DIR = Path(f'results_v2/04_final_models/k{K_VALUE}')
OUTPUT_DIR = Path(f'results_v2/04_final_models/k{K_VALUE}/survival_analysis')

# Time settings (in months)
MAX_TIME = 60  # 5 years
AUC_TIMEPOINT = 60  # 5-year AUC

# Color scheme (matching your previous figures)
COLORS = {
    'Low Risk': '#3498db',   # Blue
    'High Risk': '#f39c12'   # Yellow/Orange
}


# ============================================================
# Helper Functions
# ============================================================

def stratify_patients(risk_scores, method='median'):
    """
    Stratify patients into risk groups.
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


def calculate_hazard_ratio(durations, events, groups):
    """
    Calculate hazard ratio between risk groups using Cox regression.
    """
    df = pd.DataFrame({
        'duration': durations,
        'event': events,
        'high_risk': (groups == 'High Risk').astype(int)
    })
    
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


def calculate_time_dependent_auc(durations, events, risk_scores, timepoint=60):
    """
    Calculate time-dependent AUC at a specific timepoint using scikit-survival.
    
    Args:
        durations: survival times (months)
        events: event indicators (boolean)
        risk_scores: predicted risk scores
        timepoint: time at which to evaluate AUC (months)
    
    Returns:
        AUC value at the specified timepoint
    """
    # Create structured array for scikit-survival
    y_surv = Surv.from_arrays(event=events.astype(bool), time=durations)
    
    # Filter to patients with follow-up >= timepoint or event before timepoint
    # This ensures we have valid comparisons at the timepoint
    valid_mask = (durations >= timepoint) | (events & (durations <= timepoint))
    
    if valid_mask.sum() < 10:
        logger.warning(f"Too few valid samples ({valid_mask.sum()}) for AUC calculation at {timepoint} months")
        return np.nan
    
    try:
        # Calculate cumulative/dynamic AUC
        times = np.array([timepoint])
        auc, mean_auc = cumulative_dynamic_auc(
            y_surv, y_surv, risk_scores, times
        )
        return float(auc[0])
    except Exception as e:
        logger.warning(f"AUC calculation failed: {e}")
        return np.nan


def plot_km_curve_single(ax, durations, events, groups, cohort_name, max_time=60):
    """
    Plot Kaplan-Meier curve on a given axis with 5-year truncation.
    """
    kmf = KaplanMeierFitter()
    
    # Truncate data at max_time
    durations_truncated = np.minimum(durations, max_time)
    events_truncated = events.copy()
    events_truncated[durations > max_time] = False
    
    # Log-rank test (on original data)
    high_mask = (groups == 'High Risk')
    low_mask = (groups == 'Low Risk')
    
    logrank_result = logrank_test(
        durations[high_mask], durations[low_mask],
        events[high_mask], events[low_mask]
    )
    
    # Plot each group
    for group in ['Low Risk', 'High Risk']:
        mask = (groups == group)
        kmf.fit(
            durations_truncated[mask], 
            events_truncated[mask], 
            label=group
        )
        
        # Plot with confidence interval
        kmf.plot_survival_function(
            ax=ax,
            ci_show=True,
            color=COLORS[group],
            linewidth=2,
            ci_alpha=0.2
        )
        
        # Add censoring marks
        kmf.plot_survival_function(
            ax=ax,
            ci_show=False,
            color=COLORS[group],
            linewidth=0,
            show_censors=True,
            censor_styles={'marker': '+', 'ms': 6, 'mew': 1}
        )
    
    # Styling
    ax.set_xlabel('Time (months)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Overall Survival Probability', fontsize=11, fontweight='bold')
    ax.set_xlim([0, max_time])
    ax.set_ylim([0, 1.05])
    ax.set_xticks([0, 12, 24, 36, 48, 60])
    
    # Legend
    low_patch = mpatches.Patch(color=COLORS['Low Risk'], label='Low Risk', alpha=0.7)
    high_patch = mpatches.Patch(color=COLORS['High Risk'], label='High Risk', alpha=0.7)
    ax.legend(handles=[low_patch, high_patch], loc='upper right', fontsize=10, 
              title='Risk Group', title_fontsize=10)
    
    # P-value annotation
    if logrank_result.p_value < 0.0001:
        p_text = 'Log-rank\np < 0.0001'
    else:
        p_text = f'Log-rank\np = {logrank_result.p_value:.4f}'
    
    ax.text(0.05, 0.15, p_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    ax.grid(True, alpha=0.3, linestyle='--')
    
    return logrank_result.p_value


def plot_roc_curve_single(ax, durations, events, risk_scores, cohort_name, timepoint=60):
    """
    Plot time-dependent ROC curve at a specific timepoint.
    """
    # Create structured array for scikit-survival
    y_surv = Surv.from_arrays(event=events.astype(bool), time=durations)
    
    try:
        # Calculate ROC curve points using cumulative sensitivity/specificity
        from sksurv.metrics import cumulative_dynamic_auc
        
        # Calculate AUC
        times = np.array([timepoint])
        auc_values, mean_auc = cumulative_dynamic_auc(y_surv, y_surv, risk_scores, times)
        auc = auc_values[0]
        
        # Generate ROC curve manually using thresholds
        thresholds = np.percentile(risk_scores, np.linspace(0, 100, 101))
        tpr_list = []
        fpr_list = []
        
        for thresh in thresholds:
            predicted_high = risk_scores >= thresh
            
            # At timepoint: cases = events before timepoint, controls = alive at timepoint
            cases_mask = events & (durations <= timepoint)
            controls_mask = durations > timepoint
            
            if cases_mask.sum() > 0 and controls_mask.sum() > 0:
                tpr = predicted_high[cases_mask].mean()  # Sensitivity
                fpr = predicted_high[controls_mask].mean()  # 1 - Specificity
                tpr_list.append(tpr)
                fpr_list.append(fpr)
        
        # Sort by FPR for proper ROC curve
        if len(fpr_list) > 0:
            sorted_idx = np.argsort(fpr_list)
            fpr_sorted = np.array(fpr_list)[sorted_idx]
            tpr_sorted = np.array(tpr_list)[sorted_idx]
            
            # Add (0,0) and (1,1) points
            fpr_sorted = np.concatenate([[0], fpr_sorted, [1]])
            tpr_sorted = np.concatenate([[0], tpr_sorted, [1]])
            
            # Plot ROC curve
            ax.plot(fpr_sorted, tpr_sorted, color='#3498db', linewidth=2)
        
        # Diagonal reference line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        
        # Fill under curve
        if len(fpr_list) > 0:
            ax.fill_between(fpr_sorted, 0, tpr_sorted, alpha=0.1, color='#3498db')
        
        # Styling
        ax.set_xlabel('1 - Specificity', fontsize=11, fontweight='bold')
        ax.set_ylabel('Sensitivity', fontsize=11, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.set_xticks([0, 0.25, 0.50, 0.75, 1.00])
        ax.set_yticks([0, 0.25, 0.50, 0.75, 1.00])
        
        # AUC annotation
        ax.text(0.95, 0.05, f'AUC = {auc:.3f}', transform=ax.transAxes,
                fontsize=11, verticalalignment='bottom', horizontalalignment='right',
                fontweight='bold')
        
        ax.grid(True, alpha=0.3, linestyle='--')
        
        return auc
        
    except Exception as e:
        logger.warning(f"ROC curve generation failed: {e}")
        ax.text(0.5, 0.5, 'ROC curve\nnot available', transform=ax.transAxes,
                ha='center', va='center', fontsize=12)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        return np.nan


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
    sig_status = '***' if logrank_result.p_value < 0.001 else '**' if logrank_result.p_value < 0.01 else '*' if logrank_result.p_value < 0.05 else 'ns'
    logger.info(f"    Significant: {sig_status}")
    
    # Hazard ratio
    hr_result = calculate_hazard_ratio(durations, events, groups)
    
    logger.info(f"\n  Hazard Ratio (High vs Low Risk):")
    logger.info(f"    HR: {hr_result['HR']:.3f} (95% CI: {hr_result['CI_lower']:.3f}-{hr_result['CI_upper']:.3f})")
    logger.info(f"    P-value: {hr_result['p_value']:.6f}")
    
    # 5-year AUC
    auc_5year = calculate_time_dependent_auc(durations, events, risk_scores, timepoint=AUC_TIMEPOINT)
    logger.info(f"\n  5-Year Time-Dependent AUC: {auc_5year:.3f}")
    
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
        'auc_5year': auc_5year,
        'median_survival_high': float(medians['High Risk']) if not np.isinf(medians['High Risk']) else None,
        'median_survival_low': float(medians['Low Risk']) if not np.isinf(medians['Low Risk']) else None,
        'risk_scores': risk_scores,
        'durations': durations,
        'events': events,
        'groups': groups
    }


def create_publication_figure(results_tcga, results_orien):
    """
    Create publication-quality 4-panel figure (2 KM + 2 ROC).
    """
    logger.info("\nCreating publication figure...")
    
    # Create figure with 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel A: TCGA KM curve
    ax_km_tcga = axes[0, 0]
    p_tcga = plot_km_curve_single(
        ax_km_tcga,
        results_tcga['durations'],
        results_tcga['events'],
        results_tcga['groups'],
        'TCGA',
        max_time=MAX_TIME
    )
    ax_km_tcga.set_title(f"TCGA – {results_tcga['n_patients']} Patients ({results_tcga['n_events']} Events)", 
                         fontsize=12, fontweight='bold', pad=10)
    ax_km_tcga.text(-0.15, 1.05, 'A', transform=ax_km_tcga.transAxes, 
                    fontsize=16, fontweight='bold', va='bottom')
    
    # Panel B: ORIEN KM curve
    ax_km_orien = axes[0, 1]
    p_orien = plot_km_curve_single(
        ax_km_orien,
        results_orien['durations'],
        results_orien['events'],
        results_orien['groups'],
        'ORIEN',
        max_time=MAX_TIME
    )
    ax_km_orien.set_title(f"ORIEN – {results_orien['n_patients']} Patients ({results_orien['n_events']} Events)", 
                          fontsize=12, fontweight='bold', pad=10)
    ax_km_orien.text(-0.15, 1.05, 'B', transform=ax_km_orien.transAxes, 
                     fontsize=16, fontweight='bold', va='bottom')
    
    # Panel C: TCGA 5-year ROC
    ax_roc_tcga = axes[1, 0]
    auc_tcga = plot_roc_curve_single(
        ax_roc_tcga,
        results_tcga['durations'],
        results_tcga['events'],
        results_tcga['risk_scores'],
        'TCGA',
        timepoint=AUC_TIMEPOINT
    )
    ax_roc_tcga.set_title(f"TCGA – 5 Year", fontsize=12, fontweight='bold', pad=10)
    ax_roc_tcga.text(-0.15, 1.05, 'C', transform=ax_roc_tcga.transAxes, 
                     fontsize=16, fontweight='bold', va='bottom')
    
    # Panel D: ORIEN 5-year ROC
    ax_roc_orien = axes[1, 1]
    auc_orien = plot_roc_curve_single(
        ax_roc_orien,
        results_orien['durations'],
        results_orien['events'],
        results_orien['risk_scores'],
        'ORIEN',
        timepoint=AUC_TIMEPOINT
    )
    ax_roc_orien.set_title(f"ORIEN – 5 Year", fontsize=12, fontweight='bold', pad=10)
    ax_roc_orien.text(-0.15, 1.05, 'D', transform=ax_roc_orien.transAxes, 
                      fontsize=16, fontweight='bold', va='bottom')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_file = OUTPUT_DIR / 'publication_figure_KM_ROC.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"  Saved: {output_file.name}")
    
    # Also save as PDF for publication
    output_pdf = OUTPUT_DIR / 'publication_figure_KM_ROC.pdf'
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Recreate for PDF
    plot_km_curve_single(axes[0, 0], results_tcga['durations'], results_tcga['events'], 
                         results_tcga['groups'], 'TCGA', max_time=MAX_TIME)
    axes[0, 0].set_title(f"TCGA – {results_tcga['n_patients']} Patients ({results_tcga['n_events']} Events)", 
                         fontsize=12, fontweight='bold', pad=10)
    axes[0, 0].text(-0.15, 1.05, 'A', transform=axes[0, 0].transAxes, fontsize=16, fontweight='bold', va='bottom')
    
    plot_km_curve_single(axes[0, 1], results_orien['durations'], results_orien['events'], 
                         results_orien['groups'], 'ORIEN', max_time=MAX_TIME)
    axes[0, 1].set_title(f"ORIEN – {results_orien['n_patients']} Patients ({results_orien['n_events']} Events)", 
                         fontsize=12, fontweight='bold', pad=10)
    axes[0, 1].text(-0.15, 1.05, 'B', transform=axes[0, 1].transAxes, fontsize=16, fontweight='bold', va='bottom')
    
    plot_roc_curve_single(axes[1, 0], results_tcga['durations'], results_tcga['events'], 
                          results_tcga['risk_scores'], 'TCGA', timepoint=AUC_TIMEPOINT)
    axes[1, 0].set_title(f"TCGA – 5 Year", fontsize=12, fontweight='bold', pad=10)
    axes[1, 0].text(-0.15, 1.05, 'C', transform=axes[1, 0].transAxes, fontsize=16, fontweight='bold', va='bottom')
    
    plot_roc_curve_single(axes[1, 1], results_orien['durations'], results_orien['events'], 
                          results_orien['risk_scores'], 'ORIEN', timepoint=AUC_TIMEPOINT)
    axes[1, 1].set_title(f"ORIEN – 5 Year", fontsize=12, fontweight='bold', pad=10)
    axes[1, 1].text(-0.15, 1.05, 'D', transform=axes[1, 1].transAxes, fontsize=16, fontweight='bold', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"  Saved: {output_pdf.name}")
    
    return {'tcga_auc': auc_tcga, 'orien_auc': auc_orien}


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
            '5-Year AUC': f"{result['auc_5year']:.3f}",
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
    Run complete survival analysis with 5-year AUC.
    """
    logger.info("=" * 60)
    logger.info(f"Step 4.2: Survival Analysis (k={K_VALUE})")
    logger.info("=" * 60)
    logger.info(f"  5-Year Analysis (truncated at {MAX_TIME} months)")
    
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
    results_dict = {}
    
    for model in models_to_analyze:
        result = analyze_cohort(
            model['file'],
            model['model_name'],
            model['cohort']
        )
        all_results.append(result)
        results_dict[model['cohort']] = result
    
    # Save detailed results (without arrays)
    results_for_save = []
    for r in all_results:
        r_save = {k: v for k, v in r.items() if k not in ['risk_scores', 'durations', 'events', 'groups']}
        results_for_save.append(r_save)
    
    results_df = pd.DataFrame(results_for_save)
    results_df.to_csv(OUTPUT_DIR / 'survival_analysis_detailed.csv', index=False)
    
    # Create publication figure
    logger.info("\n" + "="*60)
    logger.info("Creating Publication Figure")
    logger.info("="*60)
    
    auc_results = create_publication_figure(results_dict['TCGA'], results_dict['ORIEN'])
    
    # Create summary table
    summary_df = create_summary_table(all_results)
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("Key Findings")
    logger.info("="*60)
    
    for result in all_results:
        sig_marker = '***' if result['logrank_p'] < 0.001 else '**' if result['logrank_p'] < 0.01 else '*' if result['logrank_p'] < 0.05 else ''
        logger.info(f"\n{result['cohort']} ({result['model']}):")
        logger.info(f"  5-Year AUC = {result['auc_5year']:.3f}")
        logger.info(f"  Log-rank p = {result['logrank_p']:.6f} {sig_marker}")
        logger.info(f"  HR = {result['hazard_ratio']:.2f} (95% CI: {result['hr_ci_lower']:.2f}-{result['hr_ci_upper']:.2f})")
        if result['logrank_significant']:
            logger.info(f"  → Significant risk stratification achieved ✓")
        else:
            logger.info(f"  → Risk stratification not significant at α=0.05")
    
    logger.info("\n" + "="*60)
    logger.info("Step 4.2 Complete!")
    logger.info("="*60)
    logger.info(f"\nAll results saved to: {OUTPUT_DIR}")
    logger.info("\nGenerated files:")
    logger.info("  - survival_analysis_detailed.csv")
    logger.info("  - survival_summary_table.csv")
    logger.info("  - publication_figure_KM_ROC.png")
    logger.info("  - publication_figure_KM_ROC.pdf")
    
    return all_results


if __name__ == '__main__':
    run_survival_analysis()
