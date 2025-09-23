"""
Ordinary Least Squares (OLS) estimation with various standard error options
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from scipy import stats
import warnings

class OLS:
    """
    OLS regression class with robust standard errors
    """
    
    def __init__(self, y: np.ndarray, X: np.ndarray, 
                 variable_names: Optional[List[str]] = None):
        """
        Initialize OLS regression
        
        Parameters:
        -----------
        y : np.ndarray
            Dependent variable
        X : np.ndarray
            Independent variables (design matrix)
        variable_names : list, optional
            Names of variables
        """
        self.y = np.array(y)
        self.X = np.array(X)
        self.n, self.k = X.shape
        
        if variable_names is None:
            self.variable_names = [f'X{i}' for i in range(self.k)]
        else:
            self.variable_names = variable_names
            
        # Add constant if not present
        if not np.allclose(self.X[:, 0], 1):
            self.X = np.column_stack([np.ones(self.n), self.X])
            self.variable_names = ['const'] + self.variable_names
            self.k += 1
    
    def fit(self, robust: bool = False, cluster_var: Optional[np.ndarray] = None) -> 'OLS':
        """
        Fit OLS regression
        
        Parameters:
        -----------
        robust : bool
            Whether to use robust standard errors
        cluster_var : np.ndarray, optional
            Clustering variable for clustered standard errors
            
        Returns:
        --------
        self
        """
        # Calculate coefficients
        try:
            self.coefficients = np.linalg.solve(self.X.T @ self.X, self.X.T @ self.y)
        except np.linalg.LinAlgError:
            warnings.warn("Singular matrix, using pseudo-inverse")
            self.coefficients = np.linalg.pinv(self.X.T @ self.X) @ self.X.T @ self.y
        
        # Calculate fitted values and residuals
        self.fitted_values = self.X @ self.coefficients
        self.residuals = self.y - self.fitted_values
        
        # Calculate R-squared
        self.ssr = np.sum(self.residuals**2)
        self.sst = np.sum((self.y - np.mean(self.y))**2)
        self.r_squared = 1 - self.ssr / self.sst
        
        # Calculate standard errors
        if robust or cluster_var is not None:
            self.se = self._calculate_robust_se(cluster_var)
        else:
            self.se = self._calculate_standard_se()
        
        # Calculate t-statistics and p-values
        self.t_stats = self.coefficients / self.se
        self.p_values = 2 * (1 - stats.t.cdf(np.abs(self.t_stats), self.n - self.k))
        
        return self
    
    def _calculate_standard_se(self) -> np.ndarray:
        """Calculate standard OLS standard errors"""
        mse = self.ssr / (self.n - self.k)
        vcov = mse * np.linalg.inv(self.X.T @ self.X)
        return np.sqrt(np.diag(vcov))
    
    def _calculate_robust_se(self, cluster_var: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate robust standard errors"""
        if cluster_var is not None:
            # Clustered standard errors
            clusters = np.unique(cluster_var)
            meat = np.zeros((self.k, self.k))
            
            for cluster in clusters:
                cluster_mask = cluster_var == cluster
                X_cluster = self.X[cluster_mask]
                e_cluster = self.residuals[cluster_mask]
                
                if len(e_cluster) > 0:
                    meat += np.outer(X_cluster.T @ e_cluster, X_cluster.T @ e_cluster)
            
            bread = np.linalg.inv(self.X.T @ self.X)
            vcov = bread @ meat @ bread
        else:
            # White/Huber-White standard errors
            meat = self.X.T @ np.diag(self.residuals**2) @ self.X
            bread = np.linalg.inv(self.X.T @ self.X)
            vcov = bread @ meat @ bread
        
        return np.sqrt(np.diag(vcov))
    
    def summary(self) -> pd.DataFrame:
        """Return regression summary"""
        return pd.DataFrame({
            'Variable': self.variable_names,
            'Coefficient': self.coefficients,
            'Std Error': self.se,
            't-statistic': self.t_stats,
            'P-value': self.p_values,
            'Significant': self.p_values < 0.05
        })
    
    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """Make predictions for new data"""
        if X_new.shape[1] == self.k - 1:  # No constant
            X_new = np.column_stack([np.ones(X_new.shape[0]), X_new])
        return X_new @ self.coefficients
    
    def get_confidence_intervals(self, alpha: float = 0.05) -> np.ndarray:
        """Get confidence intervals for coefficients"""
        t_critical = stats.t.ppf(1 - alpha/2, self.n - self.k)
        margin_error = t_critical * self.se
        
        return np.column_stack([
            self.coefficients - margin_error,
            self.coefficients + margin_error
        ])

def pooled_ols(data: pd.DataFrame, y_var: str, x_vars: List[str], 
               robust: bool = True) -> OLS:
    """
    Run pooled OLS regression on panel data
    
    Parameters:
    -----------
    data : pd.DataFrame
        Panel dataset
    y_var : str
        Name of dependent variable
    x_vars : list
        Names of independent variables
    robust : bool
        Whether to use robust standard errors
        
    Returns:
    --------
    OLS
        Fitted OLS model
    """
    y = data[y_var].values
    X = data[x_vars].values
    
    model = OLS(y, X, x_vars)
    return model.fit(robust=robust)

def test_linear_hypothesis(model: OLS, R: np.ndarray, r: np.ndarray) -> dict:
    """
    Test linear hypothesis H0: R*beta = r
    
    Parameters:
    -----------
    model : OLS
        Fitted OLS model
    R : np.ndarray
        Restriction matrix
    r : np.ndarray
        Restriction vector
        
    Returns:
    --------
    dict
        Test results
    """
    # Calculate Wald statistic
    R_beta = R @ model.coefficients
    diff = R_beta - r
    
    # Calculate variance of R*beta
    vcov = np.linalg.inv(model.X.T @ model.X)
    if hasattr(model, 'se') and len(model.se) == len(model.coefficients):
        # Use robust standard errors if available
        vcov_robust = np.outer(model.se, model.se) * np.eye(len(model.se))
        vcov = vcov_robust
    
    var_R_beta = R @ vcov @ R.T
    wald_stat = diff.T @ np.linalg.inv(var_R_beta) @ diff
    
    # Degrees of freedom
    df = R.shape[0]
    
    # P-value
    p_value = 1 - stats.chi2.cdf(wald_stat, df)
    
    return {
        'wald_statistic': wald_stat,
        'p_value': p_value,
        'degrees_of_freedom': df,
        'reject_null': p_value < 0.05
    }