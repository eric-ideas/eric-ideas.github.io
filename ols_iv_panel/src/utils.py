"""
Utility functions for econometric analysis
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import warnings

def generate_sample_data(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic panel data for econometric analysis
    
    Parameters:
    -----------
    n : int
        Number of observations
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        Generated panel data
    """
    np.random.seed(seed)
    
    # Generate individual and time identifiers
    n_individuals = n // 10  # 10 observations per individual
    individual_ids = np.repeat(range(n_individuals), 10)
    time_periods = np.tile(range(10), n_individuals)
    
    # Generate covariates
    X1 = np.random.normal(0, 1, n)
    X2 = np.random.normal(0, 1, n)
    X3 = np.random.normal(0, 1, n)
    
    # Generate individual fixed effects
    alpha_i = np.repeat(np.random.normal(0, 1, n_individuals), 10)
    
    # Generate time fixed effects
    gamma_t = np.tile(np.random.normal(0, 0.5, 10), n_individuals)
    
    # Generate error term
    epsilon = np.random.normal(0, 0.5, n)
    
    # Generate dependent variable
    y = 2 + 0.5 * X1 + 0.3 * X2 - 0.2 * X3 + alpha_i + gamma_t + epsilon
    
    return pd.DataFrame({
        'individual_id': individual_ids,
        'time': time_periods,
        'y': y,
        'X1': X1,
        'X2': X2,
        'X3': X3
    })

def check_stationarity(series: pd.Series, test: str = 'adf') -> dict:
    """
    Check stationarity of a time series
    
    Parameters:
    -----------
    series : pd.Series
        Time series to test
    test : str
        Type of test ('adf', 'kpss', 'pp')
        
    Returns:
    --------
    dict
        Test results
    """
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.stats.diagnostic import unitroot_adf
    
    results = {}
    
    if test == 'adf':
        result = adfuller(series.dropna())
        results = {
            'test_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'stationary': result[1] < 0.05
        }
    elif test == 'kpss':
        result = kpss(series.dropna())
        results = {
            'test_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[3],
            'stationary': result[1] > 0.05
        }
    
    return results

def calculate_robust_se(residuals: np.ndarray, X: np.ndarray, 
                       cluster_var: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculate robust standard errors
    
    Parameters:
    -----------
    residuals : np.ndarray
        Regression residuals
    X : np.ndarray
        Design matrix
    cluster_var : np.ndarray, optional
        Clustering variable
        
    Returns:
    --------
    np.ndarray
        Robust standard errors
    """
    n, k = X.shape
    
    if cluster_var is not None:
        # Clustered standard errors
        clusters = np.unique(cluster_var)
        meat = np.zeros((k, k))
        
        for cluster in clusters:
            cluster_mask = cluster_var == cluster
            X_cluster = X[cluster_mask]
            e_cluster = residuals[cluster_mask]
            
            if len(e_cluster) > 0:
                meat += np.outer(X_cluster.T @ e_cluster, X_cluster.T @ e_cluster)
        
        bread = np.linalg.inv(X.T @ X)
        vcov = bread @ meat @ bread
    else:
        # White/Huber-White standard errors
        meat = X.T @ np.diag(residuals**2) @ X
        bread = np.linalg.inv(X.T @ X)
        vcov = bread @ meat @ bread
    
    return np.sqrt(np.diag(vcov))

def format_results(coefficients: np.ndarray, se: np.ndarray, 
                  p_values: np.ndarray, variable_names: list) -> pd.DataFrame:
    """
    Format regression results into a nice table
    
    Parameters:
    -----------
    coefficients : np.ndarray
        Coefficient estimates
    se : np.ndarray
        Standard errors
    p_values : np.ndarray
        P-values
    variable_names : list
        Names of variables
        
    Returns:
    --------
    pd.DataFrame
        Formatted results table
    """
    results_df = pd.DataFrame({
        'Variable': variable_names,
        'Coefficient': coefficients,
        'Std Error': se,
        'P-value': p_values,
        'Significant': p_values < 0.05
    })
    
    return results_df