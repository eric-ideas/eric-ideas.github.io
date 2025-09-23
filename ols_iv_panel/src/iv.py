"""
Instrumental Variables (IV) estimation with numerical stability analysis
Based on Wooldridge (2010) "Econometric Analysis of Cross Section and Panel Data"
and Angrist & Pischke (2009) "Mostly Harmless Econometrics"
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict
from scipy import stats, linalg
import warnings
from .ols import OLS

class IV:
    """
    Two-Stage Least Squares (2SLS) instrumental variables estimation
    
    Implements the standard 2SLS estimator with numerical stability checks
    and diagnostic tests for instrument validity and relevance.
    """
    
    def __init__(self, y: np.ndarray, X: np.ndarray, Z: np.ndarray,
                 variable_names: Optional[List[str]] = None,
                 instrument_names: Optional[List[str]] = None):
        """
        Initialize IV regression
        
        Parameters:
        -----------
        y : np.ndarray
            Dependent variable (n x 1)
        X : np.ndarray
            Endogenous variables (n x k)
        Z : np.ndarray
            Instruments (n x l)
        variable_names : list, optional
            Names of endogenous variables
        instrument_names : list, optional
            Names of instruments
        """
        self.y = np.array(y).flatten()
        self.X = np.array(X)
        self.Z = np.array(Z)
        self.n, self.k = X.shape
        self.l = Z.shape[1]
        
        if variable_names is None:
            self.variable_names = [f'X{i}' for i in range(self.k)]
        else:
            self.variable_names = variable_names
            
        if instrument_names is None:
            self.instrument_names = [f'Z{i}' for i in range(self.l)]
        else:
            self.instrument_names = instrument_names
        
        # Add constant if not present
        if not np.allclose(self.X[:, 0], 1):
            self.X = np.column_stack([np.ones(self.n), self.X])
            self.variable_names = ['const'] + self.variable_names
            self.k += 1
        
        if not np.allclose(self.Z[:, 0], 1):
            self.Z = np.column_stack([np.ones(self.n), self.Z])
            self.instrument_names = ['const'] + self.instrument_names
            self.l += 1
    
    def fit(self, robust: bool = True, cluster_var: Optional[np.ndarray] = None) -> 'IV':
        """
        Fit 2SLS regression with numerical stability analysis
        
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
        # Check identification condition
        if self.l < self.k:
            raise ValueError(f"Underidentified: {self.l} instruments < {self.k} endogenous variables")
        
        # First stage: X = Z * Pi + v
        self._first_stage()
        
        # Second stage: y = X_hat * beta + u
        self._second_stage()
        
        # Calculate standard errors
        if robust or cluster_var is not None:
            self.se = self._calculate_robust_se(cluster_var)
        else:
            self.se = self._calculate_standard_se()
        
        # Calculate t-statistics and p-values
        self.t_stats = self.coefficients / self.se
        self.p_values = 2 * (1 - stats.t.cdf(np.abs(self.t_stats), self.n - self.k))
        
        # Diagnostic tests
        self._run_diagnostics()
        
        return self
    
    def _first_stage(self):
        """First stage regression: X = Z * Pi + v"""
        try:
            # Use SVD for numerical stability (Golub & Van Loan, 2013)
            U, s, Vt = linalg.svd(self.Z, full_matrices=False)
            
            # Check condition number
            self.condition_number = s[0] / s[-1]
            if self.condition_number > 1e12:
                warnings.warn(f"High condition number: {self.condition_number:.2e}")
            
            # Calculate Pi using pseudo-inverse for numerical stability
            self.Pi = Vt.T @ np.diag(1/s) @ U.T @ self.X
            
            # Predicted values
            self.X_hat = self.Z @ self.Pi
            
            # First stage residuals
            self.v = self.X - self.X_hat
            
            # First stage R-squared for each endogenous variable
            self.first_stage_r2 = []
            for i in range(self.k):
                if i == 0:  # Skip constant
                    self.first_stage_r2.append(np.nan)
                else:
                    ssr = np.sum(self.v[:, i]**2)
                    sst = np.sum((self.X[:, i] - np.mean(self.X[:, i]))**2)
                    r2 = 1 - ssr / sst
                    self.first_stage_r2.append(r2)
            
        except np.linalg.LinAlgError as e:
            raise ValueError(f"First stage regression failed: {e}")
    
    def _second_stage(self):
        """Second stage regression: y = X_hat * beta + u"""
        try:
            # Use QR decomposition for numerical stability
            Q, R = linalg.qr(self.X_hat)
            
            # Check rank
            rank = np.sum(np.abs(np.diag(R)) > 1e-10)
            if rank < self.k:
                warnings.warn(f"Second stage matrix is rank deficient: {rank} < {self.k}")
            
            # Calculate coefficients using back substitution
            y_proj = Q.T @ self.y
            self.coefficients = linalg.solve_triangular(R, y_proj)
            
            # Fitted values and residuals
            self.fitted_values = self.X_hat @ self.coefficients
            self.residuals = self.y - self.fitted_values
            
            # R-squared
            self.ssr = np.sum(self.residuals**2)
            self.sst = np.sum((self.y - np.mean(self.y))**2)
            self.r_squared = 1 - self.ssr / self.sst
            
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Second stage regression failed: {e}")
    
    def _calculate_standard_se(self) -> np.ndarray:
        """Calculate standard 2SLS standard errors"""
        # Variance of 2SLS estimator (Wooldridge, 2010, p. 99)
        try:
            # Sigma squared
            sigma2 = self.ssr / (self.n - self.k)
            
            # (X_hat'X_hat)^(-1)
            X_hat_inv = np.linalg.inv(self.X_hat.T @ self.X_hat)
            
            # Standard variance formula
            vcov = sigma2 * X_hat_inv
            
            return np.sqrt(np.diag(vcov))
        except np.linalg.LinAlgError:
            warnings.warn("Singular matrix in standard error calculation")
            return np.full(self.k, np.nan)
    
    def _calculate_robust_se(self, cluster_var: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate robust standard errors for 2SLS"""
        try:
            # Robust variance estimator (Wooldridge, 2010, p. 100)
            X_hat_inv = np.linalg.inv(self.X_hat.T @ self.X_hat)
            
            if cluster_var is not None:
                # Clustered standard errors
                clusters = np.unique(cluster_var)
                meat = np.zeros((self.k, self.k))
                
                for cluster in clusters:
                    cluster_mask = cluster_var == cluster
                    X_hat_cluster = self.X_hat[cluster_mask]
                    e_cluster = self.residuals[cluster_mask]
                    
                    if len(e_cluster) > 0:
                        meat += np.outer(X_hat_cluster.T @ e_cluster, 
                                       X_hat_cluster.T @ e_cluster)
            else:
                # White/Huber-White standard errors
                meat = self.X_hat.T @ np.diag(self.residuals**2) @ self.X_hat
            
            vcov = X_hat_inv @ meat @ X_hat_inv
            return np.sqrt(np.diag(vcov))
            
        except np.linalg.LinAlgError:
            warnings.warn("Singular matrix in robust standard error calculation")
            return np.full(self.k, np.nan)
    
    def _run_diagnostics(self):
        """Run diagnostic tests for instrument validity and relevance"""
        # Weak instruments test (Stock & Yogo, 2005)
        self._weak_instruments_test()
        
        # Overidentification test (Sargan, 1958; Hansen, 1982)
        if self.l > self.k:
            self._overidentification_test()
        
        # Endogeneity test (Hausman, 1978)
        self._endogeneity_test()
    
    def _weak_instruments_test(self):
        """Test for weak instruments using F-statistic"""
        # First stage F-statistic for each endogenous variable (excluding constant)
        self.first_stage_f = []
        
        for i in range(1, self.k):  # Skip constant
            # Reduced form: X_i = Z * gamma + error
            X_i = self.X[:, i]
            Z_reduced = self.Z[:, 1:]  # Exclude constant
            
            try:
                # OLS regression
                gamma = linalg.solve(Z_reduced.T @ Z_reduced, Z_reduced.T @ X_i)
                fitted = Z_reduced @ gamma
                residuals = X_i - fitted
                
                # F-statistic
                ssr = np.sum(residuals**2)
                sse = np.sum((X_i - np.mean(X_i))**2)
                r2 = 1 - ssr / sse
                
                f_stat = (r2 / (1 - r2)) * (self.n - self.l) / (self.l - 1)
                self.first_stage_f.append(f_stat)
                
            except np.linalg.LinAlgError:
                self.first_stage_f.append(np.nan)
        
        # Rule of thumb: F > 10 suggests strong instruments (Stock & Yogo, 2005)
        self.weak_instruments = any(f < 10 for f in self.first_stage_f if not np.isnan(f))
    
    def _overidentification_test(self):
        """Hansen J-test for overidentifying restrictions"""
        if self.l <= self.k:
            self.overid_test = None
            return
        
        try:
            # Residuals from second stage
            u = self.residuals
        
            # Project residuals on instruments
            Z_proj = self.Z @ np.linalg.inv(self.Z.T @ self.Z) @ self.Z.T
            u_proj = Z_proj @ u
            
            # J-statistic
            j_stat = u.T @ u_proj / (u.T @ u / self.n)
            
            # Degrees of freedom
            df = self.l - self.k
            
            # P-value
            p_value = 1 - stats.chi2.cdf(j_stat, df)
            
            self.overid_test = {
                'j_statistic': j_stat,
                'p_value': p_value,
                'degrees_of_freedom': df,
                'reject_overid': p_value < 0.05
            }
            
        except np.linalg.LinAlgError:
            self.overid_test = None
    
    def _endogeneity_test(self):
        """Hausman test for endogeneity"""
        try:
            # OLS regression
            ols_model = OLS(self.y, self.X[:, 1:], self.variable_names[1:])
            ols_model.fit()
            
            # Difference in coefficients
            diff = self.coefficients[1:] - ols_model.coefficients[1:]
            
            # Variance of difference
            var_diff = np.diag(self.se[1:]**2) - np.diag(ols_model.se[1:]**2)
            
            # Hausman statistic
            hausman_stat = diff.T @ np.linalg.inv(np.diag(var_diff)) @ diff
            
            # P-value
            p_value = 1 - stats.chi2.cdf(hausman_stat, len(diff))
            
            self.endogeneity_test = {
                'hausman_statistic': hausman_stat,
                'p_value': p_value,
                'degrees_of_freedom': len(diff),
                'reject_exogeneity': p_value < 0.05
            }
            
        except np.linalg.LinAlgError:
            self.endogeneity_test = None
    
    def summary(self) -> pd.DataFrame:
        """Return IV regression summary"""
        summary_df = pd.DataFrame({
            'Variable': self.variable_names,
            'Coefficient': self.coefficients,
            'Std Error': self.se,
            't-statistic': self.t_stats,
            'P-value': self.p_values,
            'Significant': self.p_values < 0.05
        })
        
        return summary_df
    
    def diagnostics_summary(self) -> Dict:
        """Return diagnostic test results"""
        diagnostics = {
            'first_stage_r2': self.first_stage_r2,
            'first_stage_f': self.first_stage_f,
            'weak_instruments': self.weak_instruments,
            'condition_number': self.condition_number,
            'overidentification_test': self.overid_test,
            'endogeneity_test': self.endogeneity_test
        }
        
        return diagnostics

def two_stage_least_squares(data: pd.DataFrame, y_var: str, x_vars: List[str], 
                           z_vars: List[str], robust: bool = True) -> IV:
    """
    Convenience function for 2SLS estimation
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset
    y_var : str
        Dependent variable
    x_vars : list
        Endogenous variables
    z_vars : list
        Instruments
    robust : bool
        Whether to use robust standard errors
        
    Returns:
    --------
    IV
        Fitted IV model
    """
    y = data[y_var].values
    X = data[x_vars].values
    Z = data[z_vars].values
    
    model = IV(y, X, Z, x_vars, z_vars)
    return model.fit(robust=robust)