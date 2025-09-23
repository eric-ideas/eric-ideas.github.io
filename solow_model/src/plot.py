"""
Visualization functions for Solow Growth Model
Based on Barro & Sala-i-Martin (2004) and Romer (2019)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import warnings

def plot_steady_state(model, figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Plot steady state analysis
    
    Parameters:
    -----------
    model : SolowModel
        Solow model instance
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # Create k vector
    k = np.linspace(0, 2 * model.k_star, 100)
    
    # Plot 1: Production function
    y = model.production_function(k)
    axes[0, 0].plot(k, y, 'b-', linewidth=2, label='y = k^α')
    axes[0, 0].axvline(x=model.k_star, color='r', linestyle='--', alpha=0.7, label='k*')
    axes[0, 0].set_xlabel('Capital per effective worker (k)')
    axes[0, 0].set_ylabel('Output per effective worker (y)')
    axes[0, 0].set_title('Production Function')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Investment and depreciation
    investment = model.s * y
    depreciation = (model.n + model.g + model.delta) * k
    
    axes[0, 1].plot(k, investment, 'g-', linewidth=2, label='sy')
    axes[0, 1].plot(k, depreciation, 'r-', linewidth=2, label='(n+g+δ)k')
    axes[0, 1].axvline(x=model.k_star, color='k', linestyle='--', alpha=0.7, label='k*')
    axes[0, 1].set_xlabel('Capital per effective worker (k)')
    axes[0, 1].set_ylabel('Investment/Depreciation')
    axes[0, 1].set_title('Investment vs Depreciation')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Capital accumulation
    dk_dt = model.capital_accumulation(k)
    axes[0, 2].plot(k, dk_dt, 'b-', linewidth=2, label='dk/dt')
    axes[0, 2].axhline(y=0, color='k', linestyle='-', alpha=0.5)
    axes[0, 2].axvline(x=model.k_star, color='r', linestyle='--', alpha=0.7, label='k*')
    axes[0, 2].set_xlabel('Capital per effective worker (k)')
    axes[0, 2].set_ylabel('Change in k (dk/dt)')
    axes[0, 2].set_title('Capital Accumulation')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Consumption function
    consumption = (1 - model.s) * y
    axes[1, 0].plot(k, consumption, 'purple', linewidth=2, label='c = (1-s)y')
    axes[1, 0].axvline(x=model.k_star, color='r', linestyle='--', alpha=0.7, label='k*')
    axes[1, 0].set_xlabel('Capital per effective worker (k)')
    axes[1, 0].set_ylabel('Consumption per effective worker (c)')
    axes[1, 0].set_title('Consumption Function')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Growth rates
    k_growth = model.capital_accumulation(k) / k
    axes[1, 1].plot(k, k_growth, 'orange', linewidth=2, label='k growth rate')
    axes[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.5)
    axes[1, 1].axvline(x=model.k_star, color='r', linestyle='--', alpha=0.7, label='k*')
    axes[1, 1].set_xlabel('Capital per effective worker (k)')
    axes[1, 1].set_ylabel('Growth rate')
    axes[1, 1].set_title('Capital Growth Rate')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Golden rule
    s_golden = model.golden_rule_savings_rate()
    k_golden = (s_golden / (model.n + model.g + model.delta))**(1 / (1 - model.alpha))
    y_golden = k_golden**model.alpha
    c_golden = (1 - s_golden) * y_golden
    
    axes[1, 2].plot(k, consumption, 'purple', linewidth=2, label='c = (1-s)y')
    axes[1, 2].axvline(x=model.k_star, color='r', linestyle='--', alpha=0.7, label='k* (current)')
    axes[1, 2].axvline(x=k_golden, color='g', linestyle='--', alpha=0.7, label='k* (golden rule)')
    axes[1, 2].set_xlabel('Capital per effective worker (k)')
    axes[1, 2].set_ylabel('Consumption per effective worker (c)')
    axes[1, 2].set_title('Golden Rule')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_transition_dynamics(transition: pd.DataFrame, 
                           figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Plot transition dynamics
    
    Parameters:
    -----------
    transition : pd.DataFrame
        Transition path data
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # Plot 1: Capital per effective worker
    axes[0, 0].plot(transition['time'], transition['k'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Capital per effective worker (k)')
    axes[0, 0].set_title('Capital Transition')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Output per effective worker
    axes[0, 1].plot(transition['time'], transition['y'], 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Output per effective worker (y)')
    axes[0, 1].set_title('Output Transition')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Consumption per effective worker
    axes[0, 2].plot(transition['time'], transition['c'], 'purple', linewidth=2)
    axes[0, 2].set_xlabel('Time')
    axes[0, 2].set_ylabel('Consumption per effective worker (c)')
    axes[0, 2].set_title('Consumption Transition')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Investment per effective worker
    axes[1, 0].plot(transition['time'], transition['i'], 'orange', linewidth=2)
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Investment per effective worker (i)')
    axes[1, 0].set_title('Investment Transition')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Growth rates
    axes[1, 1].plot(transition['time'], transition['k_growth'], 'b-', linewidth=2, label='k growth')
    axes[1, 1].plot(transition['time'], transition['y_growth'], 'g-', linewidth=2, label='y growth')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Growth rate')
    axes[1, 1].set_title('Growth Rates')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Phase diagram
    axes[1, 2].plot(transition['k'], transition['y'], 'b-', linewidth=2)
    axes[1, 2].set_xlabel('Capital per effective worker (k)')
    axes[1, 2].set_ylabel('Output per effective worker (y)')
    axes[1, 2].set_title('Phase Diagram')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_shock_analysis(shock_results: Dict, shock_type: str,
                      figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Plot shock analysis results
    
    Parameters:
    -----------
    shock_results : dict
        Shock analysis results
    shock_type : str
        Type of shock
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    if 'shock_results' in shock_results:
        data = shock_results['shock_results']
    else:
        data = shock_results['transition_path']
    
    # Plot 1: Capital per effective worker
    axes[0, 0].plot(data['time'], data['k'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Capital per effective worker (k)')
    axes[0, 0].set_title(f'{shock_type.title()} Shock - Capital')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Output per effective worker
    axes[0, 1].plot(data['time'], data['y'], 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Output per effective worker (y)')
    axes[0, 1].set_title(f'{shock_type.title()} Shock - Output')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Consumption per effective worker
    axes[0, 2].plot(data['time'], data['c'], 'purple', linewidth=2)
    axes[0, 2].set_xlabel('Time')
    axes[0, 2].set_ylabel('Consumption per effective worker (c)')
    axes[0, 2].set_title(f'{shock_type.title()} Shock - Consumption')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Technology level (if available)
    if 'A' in data.columns:
        axes[1, 0].plot(data['time'], data['A'], 'orange', linewidth=2)
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Technology level (A)')
        axes[1, 0].set_title(f'{shock_type.title()} Shock - Technology')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Shock magnitude (if available)
    if 'shock' in data.columns:
        axes[1, 1].plot(data['time'], data['shock'], 'red', linewidth=2)
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Shock magnitude')
        axes[1, 1].set_title(f'{shock_type.title()} Shock - Magnitude')
        axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Growth rates
    if 'k_growth' in data.columns and 'y_growth' in data.columns:
        axes[1, 2].plot(data['time'], data['k_growth'], 'b-', linewidth=2, label='k growth')
        axes[1, 2].plot(data['time'], data['y_growth'], 'g-', linewidth=2, label='y growth')
        axes[1, 2].set_xlabel('Time')
        axes[1, 2].set_ylabel('Growth rate')
        axes[1, 2].set_title(f'{shock_type.title()} Shock - Growth Rates')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_impulse_response(impulse_response: pd.DataFrame, 
                        shock_type: str,
                        figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Plot impulse response function
    
    Parameters:
    -----------
    impulse_response : pd.DataFrame
        Impulse response data
    shock_type : str
        Type of shock
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Capital response
    axes[0, 0].plot(impulse_response['time'], impulse_response['k'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Capital per effective worker (k)')
    axes[0, 0].set_title(f'{shock_type.title()} Shock - Capital Response')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Output response
    axes[0, 1].plot(impulse_response['time'], impulse_response['y'], 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Output per effective worker (y)')
    axes[0, 1].set_title(f'{shock_type.title()} Shock - Output Response')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Consumption response
    axes[1, 0].plot(impulse_response['time'], impulse_response['c'], 'purple', linewidth=2)
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Consumption per effective worker (c)')
    axes[1, 0].set_title(f'{shock_type.title()} Shock - Consumption Response')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Growth rate response
    if 'k_growth' in impulse_response.columns:
        axes[1, 1].plot(impulse_response['time'], impulse_response['k_growth'], 'b-', linewidth=2, label='k growth')
        axes[1, 1].plot(impulse_response['time'], impulse_response['y_growth'], 'g-', linewidth=2, label='y growth')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Growth rate')
        axes[1, 1].set_title(f'{shock_type.title()} Shock - Growth Rate Response')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_parameter_sensitivity(model, parameter: str, 
                              parameter_values: List[float],
                              figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Plot parameter sensitivity analysis
    
    Parameters:
    -----------
    model : SolowModel
        Solow model instance
    parameter : str
        Parameter to vary
    parameter_values : list
        Values of parameter to test
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    steady_states = []
    
    for value in parameter_values:
        # Set parameter value
        if parameter == 's':
            model.s = value
        elif parameter == 'n':
            model.n = value
        elif parameter == 'g':
            model.g = value
        elif parameter == 'delta':
            model.delta = value
        elif parameter == 'alpha':
            model.alpha = value
        
        # Recalculate steady state
        model._calculate_steady_state()
        
        steady_states.append({
            'parameter_value': value,
            'k_star': model.k_star,
            'y_star': model.y_star,
            'c_star': model.c_star
        })
    
    steady_states_df = pd.DataFrame(steady_states)
    
    # Plot 1: Capital per effective worker
    axes[0, 0].plot(steady_states_df['parameter_value'], steady_states_df['k_star'], 'b-o', linewidth=2)
    axes[0, 0].set_xlabel(f'{parameter.title()}')
    axes[0, 0].set_ylabel('Capital per effective worker (k*)')
    axes[0, 0].set_title(f'Steady State Capital vs {parameter.title()}')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Output per effective worker
    axes[0, 1].plot(steady_states_df['parameter_value'], steady_states_df['y_star'], 'g-o', linewidth=2)
    axes[0, 1].set_xlabel(f'{parameter.title()}')
    axes[0, 1].set_ylabel('Output per effective worker (y*)')
    axes[0, 1].set_title(f'Steady State Output vs {parameter.title()}')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Consumption per effective worker
    axes[1, 0].plot(steady_states_df['parameter_value'], steady_states_df['c_star'], 'purple', marker='o', linewidth=2)
    axes[1, 0].set_xlabel(f'{parameter.title()}')
    axes[1, 0].set_ylabel('Consumption per effective worker (c*)')
    axes[1, 0].set_title(f'Steady State Consumption vs {parameter.title()}')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Golden rule comparison
    if parameter == 's':
        golden_rule_s = model.golden_rule_savings_rate()
        axes[1, 1].axvline(x=golden_rule_s, color='r', linestyle='--', alpha=0.7, label='Golden Rule')
        axes[1, 1].plot(steady_states_df['parameter_value'], steady_states_df['c_star'], 'purple', marker='o', linewidth=2)
        axes[1, 1].set_xlabel(f'{parameter.title()}')
        axes[1, 1].set_ylabel('Consumption per effective worker (c*)')
        axes[1, 1].set_title(f'Golden Rule Comparison')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_growth_accounting(growth_accounting: Dict, 
                         figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot growth accounting decomposition
    
    Parameters:
    -----------
    growth_accounting : dict
        Growth accounting results
    figsize : tuple
        Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Growth contributions
    contributions = [
        growth_accounting['capital_contribution'],
        growth_accounting['labor_contribution'],
        growth_accounting['technology_contribution']
    ]
    labels = ['Capital', 'Labor', 'Technology']
    colors = ['steelblue', 'green', 'orange']
    
    ax1.bar(labels, contributions, color=colors, alpha=0.7)
    ax1.set_ylabel('Growth contribution')
    ax1.set_title('Growth Accounting - Contributions')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Relative shares
    shares = [
        growth_accounting['capital_share'],
        growth_accounting['labor_share'],
        growth_accounting['technology_share']
    ]
    
    ax2.bar(labels, shares, color=colors, alpha=0.7)
    ax2.set_ylabel('Relative share')
    ax2.set_title('Growth Accounting - Relative Shares')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()