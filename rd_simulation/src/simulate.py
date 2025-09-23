"""
Simulation functions for Regression Discontinuity
Based on Imbens & Lemieux (2008) and Calonico et al. (2014)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Union
from scipy import stats
import warnings

def simulate_rd_data(n_obs: int = 1000, cutoff: float = 0.0,
                    treatment_effect: float = 2.0, 
                    running_var_effect: float = 1.0,
                    noise_level: float = 1.0,
                    manipulation: float = 0.0,
                    seed: int = 42) -> pd.DataFrame:
    """
    Simulate regression discontinuity data
    
    Based on Imbens & Lemieux (2008) simulation design
    
    Parameters:
    -----------
    n_obs : int
        Number of observations
    cutoff : float
        Discontinuity cutoff
    treatment_effect : float
        True treatment effect
    running_var_effect : float
        Effect of running variable on outcome
    noise_level : float
        Noise level in outcome
    manipulation : float
        Degree of manipulation (0 = no manipulation)
    seed : int
        Random seed
        
    Returns:
    --------
    pd.DataFrame
        Simulated RD data
    """
    np.random.seed(seed)
    
    # Generate running variable
    # Add manipulation if specified
    if manipulation > 0:
        # Manipulation creates bunching at cutoff
        manipulation_prob = 1 / (1 + np.exp(-manipulation * (np.random.uniform(-2, 2, n_obs))))
        running_var = np.random.normal(0, 1, n_obs)
        
        # Add bunching at cutoff
        bunching_mask = np.random.binomial(1, manipulation_prob, n_obs).astype(bool)
        running_var[bunching_mask] = cutoff + np.random.normal(0, 0.1, np.sum(bunching_mask))
    else:
        running_var = np.random.normal(0, 1, n_obs)
    
    # Create treatment indicator
    treatment = (running_var >= cutoff).astype(int)
    
    # Generate outcome
    # Y = α + β₁*X + β₂*T + β₃*X*T + ε
    alpha = 0.0
    beta_1 = running_var_effect
    beta_2 = treatment_effect
    beta_3 = 0.1  # Interaction effect
    
    epsilon = np.random.normal(0, noise_level, n_obs)
    
    outcome = (alpha + 
              beta_1 * running_var + 
              beta_2 * treatment + 
              beta_3 * running_var * treatment + 
              epsilon)
    
    # Create dataset
    data = pd.DataFrame({
        'id': range(n_obs),
        'running_var': running_var,
        'outcome': outcome,
        'treatment': treatment,
        'cutoff': cutoff
    })
    
    return data

def simulate_heterogeneous_rd(n_obs: int = 1000, cutoff: float = 0.0,
                            base_effect: float = 2.0,
                            heterogeneity: float = 1.0,
                            seed: int = 42) -> pd.DataFrame:
    """
    Simulate RD data with heterogeneous treatment effects
    
    Parameters:
    -----------
    n_obs : int
        Number of observations
    cutoff : float
        Discontinuity cutoff
    base_effect : float
        Base treatment effect
    heterogeneity : float
        Degree of effect heterogeneity
    seed : int
        Random seed
        
    Returns:
    --------
    pd.DataFrame
        Simulated RD data with heterogeneous effects
    """
    np.random.seed(seed)
    
    # Generate running variable
    running_var = np.random.normal(0, 1, n_obs)
    
    # Create treatment indicator
    treatment = (running_var >= cutoff).astype(int)
    
    # Generate heterogeneous treatment effects
    # τ(X) = τ₀ + δ*X
    tau_0 = base_effect
    delta = heterogeneity
    
    treatment_effects = tau_0 + delta * running_var
    
    # Generate outcome
    # Y = α + β₁*X + τ(X)*T + ε
    alpha = 0.0
    beta_1 = 1.0
    
    epsilon = np.random.normal(0, 1, n_obs)
    
    outcome = (alpha + 
              beta_1 * running_var + 
              treatment_effects * treatment + 
              epsilon)
    
    # Create dataset
    data = pd.DataFrame({
        'id': range(n_obs),
        'running_var': running_var,
        'outcome': outcome,
        'treatment': treatment,
        'cutoff': cutoff,
        'true_effect': treatment_effects
    })
    
    return data

def simulate_fuzzy_rd(n_obs: int = 1000, cutoff: float = 0.0,
                     first_stage_effect: float = 0.8,
                     treatment_effect: float = 2.0,
                     seed: int = 42) -> pd.DataFrame:
    """
    Simulate fuzzy regression discontinuity data
    
    Parameters:
    -----------
    n_obs : int
        Number of observations
    cutoff : float
        Discontinuity cutoff
    first_stage_effect : float
        First stage effect (compliance rate)
    treatment_effect : float
        Treatment effect on outcome
    seed : int
        Random seed
        
    Returns:
    --------
    pd.DataFrame
        Simulated fuzzy RD data
    """
    np.random.seed(seed)
    
    # Generate running variable
    running_var = np.random.normal(0, 1, n_obs)
    
    # Create eligibility indicator
    eligibility = (running_var >= cutoff).astype(int)
    
    # Generate treatment assignment (fuzzy)
    # P(T=1|X) = α + β*Eligibility + γ*X
    alpha = 0.1  # Base treatment probability
    beta = first_stage_effect  # Effect of eligibility
    gamma = 0.2  # Effect of running variable
    
    treatment_prob = alpha + beta * eligibility + gamma * running_var
    treatment_prob = np.clip(treatment_prob, 0, 1)  # Ensure valid probabilities
    
    treatment = np.random.binomial(1, treatment_prob)
    
    # Generate outcome
    # Y = α + β₁*X + β₂*T + ε
    alpha_outcome = 0.0
    beta_1 = 1.0
    beta_2 = treatment_effect
    
    epsilon = np.random.normal(0, 1, n_obs)
    
    outcome = (alpha_outcome + 
              beta_1 * running_var + 
              beta_2 * treatment + 
              epsilon)
    
    # Create dataset
    data = pd.DataFrame({
        'id': range(n_obs),
        'running_var': running_var,
        'outcome': outcome,
        'eligibility': eligibility,
        'treatment': treatment,
        'cutoff': cutoff,
        'treatment_prob': treatment_prob
    })
    
    return data

def simulate_rd_with_covariates(n_obs: int = 1000, cutoff: float = 0.0,
                               treatment_effect: float = 2.0,
                               n_covariates: int = 3,
                               covariate_effects: List[float] = None,
                               seed: int = 42) -> pd.DataFrame:
    """
    Simulate RD data with additional covariates
    
    Parameters:
    -----------
    n_obs : int
        Number of observations
    cutoff : float
        Discontinuity cutoff
    treatment_effect : float
        Treatment effect
    n_covariates : int
        Number of covariates
    covariate_effects : list, optional
        Effects of covariates on outcome
    seed : int
        Random seed
        
    Returns:
    --------
    pd.DataFrame
        Simulated RD data with covariates
    """
    np.random.seed(seed)
    
    # Generate running variable
    running_var = np.random.normal(0, 1, n_obs)
    
    # Create treatment indicator
    treatment = (running_var >= cutoff).astype(int)
    
    # Generate covariates
    covariates = np.random.normal(0, 1, (n_obs, n_covariates))
    
    # Set covariate effects
    if covariate_effects is None:
        covariate_effects = np.random.normal(0, 0.5, n_covariates)
    
    # Generate outcome
    # Y = α + β₁*X + β₂*T + Σᵢ γᵢ*Zᵢ + ε
    alpha = 0.0
    beta_1 = 1.0
    beta_2 = treatment_effect
    
    epsilon = np.random.normal(0, 1, n_obs)
    
    outcome = (alpha + 
              beta_1 * running_var + 
              beta_2 * treatment + 
              covariates @ covariate_effects + 
              epsilon)
    
    # Create dataset
    data = pd.DataFrame({
        'id': range(n_obs),
        'running_var': running_var,
        'outcome': outcome,
        'treatment': treatment,
        'cutoff': cutoff
    })
    
    # Add covariates
    for i in range(n_covariates):
        data[f'covariate_{i+1}'] = covariates[:, i]
    
    return data

def monte_carlo_rd(n_simulations: int = 100, n_obs: int = 500,
                  true_effect: float = 2.0, **kwargs) -> Dict:
    """
    Monte Carlo simulation for RD estimation
    
    Parameters:
    -----------
    n_simulations : int
        Number of simulations
    n_obs : int
        Number of observations per simulation
    true_effect : float
        True treatment effect
    **kwargs
        Additional arguments for data simulation
        
    Returns:
    --------
    dict
        Simulation results
    """
    estimates = []
    standard_errors = []
    p_values = []
    
    for i in range(n_simulations):
        # Generate data
        data = simulate_rd_data(n_obs=n_obs, treatment_effect=true_effect, **kwargs)
        
        # Estimate treatment effect (simplified)
        treated_outcomes = data[data['treatment'] == 1]['outcome']
        control_outcomes = data[data['treatment'] == 0]['outcome']
        
        if len(treated_outcomes) > 0 and len(control_outcomes) > 0:
            estimate = treated_outcomes.mean() - control_outcomes.mean()
            se = np.sqrt(treated_outcomes.var() / len(treated_outcomes) + 
                        control_outcomes.var() / len(control_outcomes))
            
            t_stat = estimate / se if se > 0 else np.nan
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(data) - 2)) if not np.isnan(t_stat) else np.nan
            
            estimates.append(estimate)
            standard_errors.append(se)
            p_values.append(p_value)
    
    # Calculate simulation statistics
    bias = np.mean(estimates) - true_effect
    rmse = np.sqrt(np.mean((np.array(estimates) - true_effect)**2))
    coverage = np.mean([ci_lower <= true_effect <= ci_upper 
                       for ci_lower, ci_upper in 
                       zip(np.array(estimates) - 1.96 * np.array(standard_errors),
                           np.array(estimates) + 1.96 * np.array(standard_errors))])
    
    return {
        'n_simulations': n_simulations,
        'true_effect': true_effect,
        'mean_estimate': np.mean(estimates),
        'bias': bias,
        'rmse': rmse,
        'coverage': coverage,
        'estimates': estimates,
        'standard_errors': standard_errors,
        'p_values': p_values
    }

def test_rd_robustness(data: pd.DataFrame, cutoffs: List[float]) -> Dict:
    """
    Test robustness of RD estimates to different cutoffs
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset
    cutoffs : list
        List of cutoffs to test
        
    Returns:
    --------
    dict
        Robustness test results
    """
    results = {}
    
    for cutoff in cutoffs:
        # Create fake treatment indicator
        fake_treatment = (data['running_var'] >= cutoff).astype(int)
        
        # Estimate fake RD
        treated_outcomes = data[fake_treatment == 1]['outcome']
        control_outcomes = data[fake_treatment == 0]['outcome']
        
        if len(treated_outcomes) > 0 and len(control_outcomes) > 0:
            estimate = treated_outcomes.mean() - control_outcomes.mean()
            se = np.sqrt(treated_outcomes.var() / len(treated_outcomes) + 
                        control_outcomes.var() / len(control_outcomes))
            
            t_stat = estimate / se if se > 0 else np.nan
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(data) - 2)) if not np.isnan(t_stat) else np.nan
        else:
            estimate = np.nan
            se = np.nan
            t_stat = np.nan
            p_value = np.nan
        
        results[cutoff] = {
            'rd_estimate': estimate,
            'se': se,
            't_statistic': t_stat,
            'p_value': p_value
        }
    
    return results