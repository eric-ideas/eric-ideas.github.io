"""
Visualization functions for Difference-in-Differences analysis
Based on best practices for econometric visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import warnings

def plot_did_results(results: Dict, title: str = "Difference-in-Differences Results") -> None:
    """
    Plot DID estimation results with confidence intervals
    
    Parameters:
    -----------
    results : dict
        DID estimation results
    title : str
        Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Treatment effect with confidence interval
    coef = results['did_coefficient']
    se = results['did_std_error']
    ci_lower = coef - 1.96 * se
    ci_upper = coef + 1.96 * se
    
    ax1.bar(['DID Estimate'], [coef], yerr=[[coef - ci_lower], [ci_upper - coef]], 
            capsize=10, color='steelblue', alpha=0.7)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Treatment Effect')
    ax1.set_title('DID Treatment Effect')
    ax1.grid(True, alpha=0.3)
    
    # Add text with coefficient and p-value
    ax1.text(0, coef + se + 0.1, f'Coeff: {coef:.3f}\nP-value: {results["did_p_value"]:.3f}', 
             ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Sample composition
    n_treated = results['n_treated']
    n_control = results['n_control']
    n_periods = results['n_periods']
    
    categories = ['Treated', 'Control', 'Periods']
    values = [n_treated, n_control, n_periods]
    colors = ['lightcoral', 'lightblue', 'lightgreen']
    
    ax2.bar(categories, values, color=colors, alpha=0.7)
    ax2.set_ylabel('Count')
    ax2.set_title('Sample Composition')
    
    # Add value labels on bars
    for i, v in enumerate(values):
        ax2.text(i, v + max(values) * 0.01, str(v), ha='center', va='bottom')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_parallel_trends(data: pd.DataFrame, y_var: str, treatment_var: str,
                        time_var: str, entity_var: str, 
                        treatment_time: Optional[int] = None,
                        title: str = "Parallel Trends Test") -> None:
    """
    Plot parallel trends for treatment and control groups
    
    Parameters:
    -----------
    data : pd.DataFrame
        Panel dataset
    y_var : str
        Outcome variable
    treatment_var : str
        Treatment indicator
    time_var : str
        Time variable
    entity_var : str
        Entity identifier
    treatment_time : int, optional
        Treatment start time
    title : str
        Plot title
    """
    # Calculate group means over time
    group_means = data.groupby([time_var, treatment_var])[y_var].mean().reset_index()
    
    # Separate treated and control groups
    treated_means = group_means[group_means[treatment_var] == 1]
    control_means = group_means[group_means[treatment_var] == 0]
    
    plt.figure(figsize=(12, 6))
    
    # Plot trends
    plt.plot(treated_means[time_var], treated_means[y_var], 'o-', 
             label='Treated Group', linewidth=2, markersize=6)
    plt.plot(control_means[time_var], control_means[y_var], 's-', 
             label='Control Group', linewidth=2, markersize=6)
    
    # Add treatment line if specified
    if treatment_time is not None:
        plt.axvline(x=treatment_time, color='red', linestyle='--', alpha=0.7, 
                   label=f'Treatment Start (t={treatment_time})')
    
    plt.xlabel('Time')
    plt.ylabel(f'Mean {y_var}')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_event_study(event_results: Dict, title: str = "Event Study Results") -> None:
    """
    Plot event study results with confidence intervals
    
    Parameters:
    -----------
    event_results : dict
        Event study results
    title : str
        Plot title
    """
    relative_times = event_results['relative_times']
    coefficients = [event_results['event_coefficients'][t] for t in relative_times]
    std_errors = [event_results['event_std_errors'][t] for t in relative_times]
    
    # Create confidence intervals
    ci_lower = [c - 1.96 * se for c, se in zip(coefficients, std_errors)]
    ci_upper = [c + 1.96 * se for c, se in zip(coefficients, std_errors)]
    
    plt.figure(figsize=(12, 6))
    
    # Plot coefficients
    plt.plot(relative_times, coefficients, 'o-', color='steelblue', 
             linewidth=2, markersize=6, label='Treatment Effect')
    
    # Plot confidence intervals
    plt.fill_between(relative_times, ci_lower, ci_upper, alpha=0.3, 
                     color='steelblue', label='95% Confidence Interval')
    
    # Add reference lines
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, 
                label='Treatment Start')
    
    # Add reference period line
    ref_period = event_results.get('reference_period', -1)
    if ref_period in relative_times:
        plt.axvline(x=ref_period, color='green', linestyle=':', alpha=0.7, 
                   label=f'Reference Period (t={ref_period})')
    
    plt.xlabel('Relative Time')
    plt.ylabel('Treatment Effect')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_placebo_test(placebo_results: Dict, title: str = "Placebo Test Results") -> None:
    """
    Plot placebo test results
    
    Parameters:
    -----------
    placebo_results : dict
        Placebo test results
    title : str
        Plot title
    """
    periods = list(placebo_results.keys())
    coefficients = [placebo_results[p]['coefficient'] for p in periods]
    std_errors = [placebo_results[p]['std_error'] for p in periods]
    p_values = [placebo_results[p]['p_value'] for p in periods]
    
    # Create confidence intervals
    ci_lower = [c - 1.96 * se for c, se in zip(coefficients, std_errors)]
    ci_upper = [c + 1.96 * se for c, se in zip(coefficients, std_errors)]
    
    plt.figure(figsize=(12, 6))
    
    # Plot coefficients
    colors = ['red' if p < 0.05 else 'steelblue' for p in p_values]
    plt.scatter(periods, coefficients, c=colors, s=100, alpha=0.7)
    
    # Plot confidence intervals
    for period, coef, ci_l, ci_u in zip(periods, coefficients, ci_lower, ci_upper):
        plt.plot([period, period], [ci_l, ci_u], color='gray', alpha=0.5)
    
    # Add reference line
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    plt.xlabel('Placebo Treatment Period')
    plt.ylabel('Placebo Treatment Effect')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Add legend for significance
    plt.scatter([], [], c='red', s=100, alpha=0.7, label='Significant (p<0.05)')
    plt.scatter([], [], c='steelblue', s=100, alpha=0.7, label='Not Significant (p≥0.05)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_balance_table(data: pd.DataFrame, treatment_var: str, 
                      covariates: List[str], title: str = "Balance Table") -> None:
    """
    Plot balance table for treatment and control groups
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset
    treatment_var : str
        Treatment indicator
    covariates : list
        Covariates to check balance for
    title : str
        Plot title
    """
    # Calculate means by treatment status
    treated_means = data[data[treatment_var] == 1][covariates].mean()
    control_means = data[data[treatment_var] == 0][covariates].mean()
    
    # Calculate standardized differences
    treated_std = data[data[treatment_var] == 1][covariates].std()
    control_std = data[data[treatment_var] == 0][covariates].std()
    
    std_diff = (treated_means - control_means) / np.sqrt((treated_std**2 + control_std**2) / 2)
    
    # Create balance table
    balance_df = pd.DataFrame({
        'Variable': covariates,
        'Treated Mean': treated_means.values,
        'Control Mean': control_means.values,
        'Std. Diff.': std_diff.values
    })
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Means comparison
    x = np.arange(len(covariates))
    width = 0.35
    
    ax1.bar(x - width/2, treated_means.values, width, label='Treated', alpha=0.7)
    ax1.bar(x + width/2, control_means.values, width, label='Control', alpha=0.7)
    
    ax1.set_xlabel('Variables')
    ax1.set_ylabel('Mean Values')
    ax1.set_title('Mean Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(covariates, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Standardized differences
    colors = ['red' if abs(diff) > 0.25 else 'steelblue' for diff in std_diff.values]
    ax2.bar(x, std_diff.values, color=colors, alpha=0.7)
    ax2.axhline(y=0.25, color='red', linestyle='--', alpha=0.5, label='±0.25 threshold')
    ax2.axhline(y=-0.25, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    ax2.set_xlabel('Variables')
    ax2.set_ylabel('Standardized Difference')
    ax2.set_title('Standardized Differences')
    ax2.set_xticks(x)
    ax2.set_xticklabels(covariates, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return balance_df

def plot_dynamic_effects(did_results: Dict, event_results: Dict, 
                        title: str = "Dynamic Treatment Effects") -> None:
    """
    Plot dynamic treatment effects from DID and event study
    
    Parameters:
    -----------
    did_results : dict
        DID estimation results
    event_results : dict
        Event study results
    title : str
        Plot title
    """
    # Extract event study coefficients
    relative_times = event_results['relative_times']
    coefficients = [event_results['event_coefficients'][t] for t in relative_times]
    std_errors = [event_results['event_std_errors'][t] for t in relative_times]
    
    # Create confidence intervals
    ci_lower = [c - 1.96 * se for c, se in zip(coefficients, std_errors)]
    ci_upper = [c + 1.96 * se for c, se in zip(coefficients, std_errors)]
    
    plt.figure(figsize=(14, 8))
    
    # Plot event study coefficients
    plt.plot(relative_times, coefficients, 'o-', color='steelblue', 
             linewidth=2, markersize=6, label='Event Study Coefficients')
    
    # Plot confidence intervals
    plt.fill_between(relative_times, ci_lower, ci_upper, alpha=0.3, 
                     color='steelblue', label='95% Confidence Interval')
    
    # Add DID coefficient as horizontal line
    did_coef = did_results['did_coefficient']
    plt.axhline(y=did_coef, color='red', linestyle='--', linewidth=2, 
               label=f'DID Estimate: {did_coef:.3f}')
    
    # Add reference lines
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.axvline(x=0, color='red', linestyle=':', alpha=0.7, 
                label='Treatment Start')
    
    plt.xlabel('Relative Time')
    plt.ylabel('Treatment Effect')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_robustness_checks(results_dict: Dict, title: str = "Robustness Checks") -> None:
    """
    Plot robustness check results
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary of robustness check results
    title : str
        Plot title
    """
    methods = list(results_dict.keys())
    coefficients = [results_dict[method]['coefficient'] for method in methods]
    std_errors = [results_dict[method]['std_error'] for method in methods]
    
    # Create confidence intervals
    ci_lower = [c - 1.96 * se for c, se in zip(coefficients, std_errors)]
    ci_upper = [c + 1.96 * se for c, se in zip(coefficients, std_errors)]
    
    plt.figure(figsize=(12, 6))
    
    # Plot coefficients
    x = np.arange(len(methods))
    plt.bar(x, coefficients, yerr=[[c - ci_l for c, ci_l in zip(coefficients, ci_lower)],
                                   [ci_u - c for c, ci_u in zip(coefficients, ci_upper)]], 
            capsize=10, alpha=0.7, color='steelblue')
    
    # Add reference line
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.xlabel('Specification')
    plt.ylabel('Treatment Effect')
    plt.title(title)
    plt.xticks(x, methods, rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()