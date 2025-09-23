"""
Simulation functions for propensity score matching
Based on Imbens & Wooldridge (2009) and Rubin (1974)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Union
from scipy import stats
import warnings

def simulate_treatment_data(n_obs: int = 1000, n_covariates: int = 5,
                          treatment_effect: float = 2.0, 
                          selection_bias: float = 0.5,
                          seed: int = 42) -> pd.DataFrame:
    """
    Simulate data with known treatment effects for testing matching methods
    
    Based on Imbens & Wooldridge (2009) simulation design
    
    Parameters:
    -----------
    n_obs : int
        Number of observations
    n_covariates : int
        Number of covariates
    treatment_effect : float
        True treatment effect
    selection_bias : float
        Degree of selection bias
    seed : int
        Random seed
        
    Returns:
    --------
    pd.DataFrame
        Simulated dataset
    """
    np.random.seed(seed)
    
    # Generate covariates
    X = np.random.normal(0, 1, (n_obs, n_covariates))
    
    # Generate propensity scores (logistic model)
    # True propensity score model: P(T=1|X) = expit(α + β'X)
    alpha = -0.5  # Intercept
    beta = np.random.normal(0, 0.5, n_covariates)  # Coefficients
    
    linear_predictor = alpha + X @ beta
    propensity_scores = 1 / (1 + np.exp(-linear_predictor))
    
    # Generate treatment assignment
    treatment = np.random.binomial(1, propensity_scores)
    
    # Generate potential outcomes
    # Y(0) = X'γ + ε₀
    # Y(1) = Y(0) + τ + X'δ
    gamma = np.random.normal(0, 0.3, n_covariates)
    delta = np.random.normal(0, 0.1, n_covariates)  # Heterogeneous effects
    
    epsilon_0 = np.random.normal(0, 1, n_obs)
    epsilon_1 = np.random.normal(0, 1, n_obs)
    
    y_0 = X @ gamma + epsilon_0
    y_1 = y_0 + treatment_effect + X @ delta + epsilon_1
    
    # Observed outcomes
    outcome = treatment * y_1 + (1 - treatment) * y_0
    
    # Create dataset
    data = pd.DataFrame({
        'id': range(n_obs),
        'outcome': outcome,
        'treatment': treatment,
        'propensity_score': propensity_scores
    })
    
    # Add covariates
    for i in range(n_covariates):
        data[f'X{i+1}'] = X[:, i]
    
    return data

def simulate_heterogeneous_effects(n_obs: int = 1000, n_covariates: int = 3,
                                   base_effect: float = 2.0,
                                   heterogeneity: float = 1.0,
                                   seed: int = 42) -> pd.DataFrame:
    """
    Simulate data with heterogeneous treatment effects
    
    Parameters:
    -----------
    n_obs : int
        Number of observations
    n_covariates : int
        Number of covariates
    base_effect : float
        Base treatment effect
    heterogeneity : float
        Degree of effect heterogeneity
    seed : int
        Random seed
        
    Returns:
    --------
    pd.DataFrame
        Simulated dataset
    """
    np.random.seed(seed)
    
    # Generate covariates
    X = np.random.normal(0, 1, (n_obs, n_covariates))
    
    # Generate propensity scores
    alpha = -0.5
    beta = np.random.normal(0, 0.5, n_covariates)
    
    linear_predictor = alpha + X @ beta
    propensity_scores = 1 / (1 + np.exp(-linear_predictor))
    
    # Generate treatment assignment
    treatment = np.random.binomial(1, propensity_scores)
    
    # Generate heterogeneous treatment effects
    # τ(X) = τ₀ + X'δ
    tau_0 = base_effect
    delta = np.random.normal(0, heterogeneity, n_covariates)
    
    treatment_effects = tau_0 + X @ delta
    
    # Generate potential outcomes
    gamma = np.random.normal(0, 0.3, n_covariates)
    epsilon_0 = np.random.normal(0, 1, n_obs)
    epsilon_1 = np.random.normal(0, 1, n_obs)
    
    y_0 = X @ gamma + epsilon_0
    y_1 = y_0 + treatment_effects + epsilon_1
    
    # Observed outcomes
    outcome = treatment * y_1 + (1 - treatment) * y_0
    
    # Create dataset
    data = pd.DataFrame({
        'id': range(n_obs),
        'outcome': outcome,
        'treatment': treatment,
        'propensity_score': propensity_scores,
        'true_effect': treatment_effects
    })
    
    # Add covariates
    for i in range(n_covariates):
        data[f'X{i+1}'] = X[:, i]
    
    return data

def simulate_missing_data(data: pd.DataFrame, missing_rate: float = 0.1,
                         missing_mechanism: str = 'MCAR') -> pd.DataFrame:
    """
    Simulate missing data for robustness testing
    
    Parameters:
    -----------
    data : pd.DataFrame
        Original dataset
    missing_rate : float
        Proportion of missing values
    missing_mechanism : str
        Missing data mechanism ('MCAR', 'MAR', 'MNAR')
        
    Returns:
    --------
    pd.DataFrame
        Dataset with missing values
    """
    data_missing = data.copy()
    
    if missing_mechanism == 'MCAR':
        # Missing Completely at Random
        n_missing = int(len(data) * missing_rate)
        missing_indices = np.random.choice(len(data), n_missing, replace=False)
        data_missing.loc[missing_indices, 'outcome'] = np.nan
        
    elif missing_mechanism == 'MAR':
        # Missing at Random (depends on observed variables)
        # Higher probability of missing for lower propensity scores
        missing_prob = 1 / (1 + np.exp(-(data['propensity_score'] - 0.5) * 2))
        missing_prob = missing_prob * missing_rate * 2  # Scale to desired rate
        
        missing_indices = np.random.binomial(1, missing_prob, len(data)).astype(bool)
        data_missing.loc[missing_indices, 'outcome'] = np.nan
        
    elif missing_mechanism == 'MNAR':
        # Missing Not at Random (depends on unobserved variables)
        # Higher probability of missing for higher outcomes
        missing_prob = 1 / (1 + np.exp(-(data['outcome'] - data['outcome'].mean()) / data['outcome'].std()))
        missing_prob = missing_prob * missing_rate * 2
        
        missing_indices = np.random.binomial(1, missing_prob, len(data)).astype(bool)
        data_missing.loc[missing_indices, 'outcome'] = np.nan
    
    return data_missing

def simulate_measurement_error(data: pd.DataFrame, error_variance: float = 0.1,
                              error_covariates: List[str] = None) -> pd.DataFrame:
    """
    Simulate measurement error in covariates
    
    Parameters:
    -----------
    data : pd.DataFrame
        Original dataset
    error_variance : float
        Variance of measurement error
    error_covariates : list, optional
        Covariates to add error to
        
    Returns:
    --------
    pd.DataFrame
        Dataset with measurement error
    """
    data_error = data.copy()
    
    if error_covariates is None:
        error_covariates = [col for col in data.columns if col.startswith('X')]
    
    for covariate in error_covariates:
        if covariate in data.columns:
            # Add measurement error
            error = np.random.normal(0, np.sqrt(error_variance), len(data))
            data_error[covariate] = data[covariate] + error
    
    return data_error

def bootstrap_confidence_interval(estimator_func, data: pd.DataFrame, 
                                n_bootstrap: int = 1000,
                                confidence_level: float = 0.95,
                                **kwargs) -> Dict:
    """
    Bootstrap confidence intervals for treatment effect estimates
    
    Parameters:
    -----------
    estimator_func : callable
        Function that estimates treatment effect
    data : pd.DataFrame
        Dataset
    n_bootstrap : int
        Number of bootstrap samples
    confidence_level : float
        Confidence level
    **kwargs
        Additional arguments for estimator function
        
    Returns:
    --------
    dict
        Bootstrap results
    """
    bootstrap_estimates = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        bootstrap_indices = np.random.choice(len(data), len(data), replace=True)
        bootstrap_data = data.iloc[bootstrap_indices]
        
        try:
            # Estimate treatment effect
            estimate = estimator_func(bootstrap_data, **kwargs)
            bootstrap_estimates.append(estimate)
        except:
            continue
    
    if len(bootstrap_estimates) == 0:
        return {'estimate': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan}
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_estimates, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))
    
    return {
        'estimate': np.mean(bootstrap_estimates),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'bootstrap_estimates': bootstrap_estimates,
        'n_bootstrap': len(bootstrap_estimates)
    }

def monte_carlo_simulation(n_simulations: int = 100, n_obs: int = 500,
                          true_effect: float = 2.0, **kwargs) -> Dict:
    """
    Monte Carlo simulation for evaluating matching methods
    
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
        data = simulate_treatment_data(n_obs=n_obs, **kwargs)
        
        # Estimate treatment effect (placeholder - would use actual matching)
        # This is a simplified version
        treated_outcomes = data[data['treatment'] == 1]['outcome']
        control_outcomes = data[data['treatment'] == 0]['outcome']
        
        estimate = treated_outcomes.mean() - control_outcomes.mean()
        se = np.sqrt(treated_outcomes.var() / len(treated_outcomes) + 
                    control_outcomes.var() / len(control_outcomes))
        
        t_stat = estimate / se
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(data) - 2))
        
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