"""
Kernel Methods for Regression Discontinuity
Based on Imbens & Kalyanaraman (2012) "Optimal Bandwidth Choice for the Regression Discontinuity Estimator"
and Calonico, Cattaneo, and Titiunik (2014) "Robust Nonparametric Confidence Intervals"
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Union
from scipy import stats, linalg, optimize
import warnings

class KernelRD:
    """
    Kernel-based regression discontinuity estimation
    
    Implements various kernel methods and bandwidth selection procedures
    for RD estimation with optimal theoretical properties.
    """
    
    def __init__(self, data: pd.DataFrame, running_var: str, outcome_var: str,
                 cutoff: float, treatment_var: Optional[str] = None):
        """
        Initialize kernel RD analysis
        
        Parameters:
        -----------
        data : pd.DataFrame
            Dataset
        running_var : str
            Running variable name
        outcome_var : str
            Outcome variable name
        cutoff : float
            Discontinuity cutoff
        treatment_var : str, optional
            Treatment indicator
        """
        self.data = data.copy()
        self.running_var = running_var
        self.outcome_var = outcome_var
        self.cutoff = cutoff
        self.treatment_var = treatment_var
        
        # Create treatment indicator if not provided
        if treatment_var is None:
            self.data['treatment'] = (self.data[running_var] >= cutoff).astype(int)
            self.treatment_var = 'treatment'
        
        # Center running variable at cutoff
        self.data['running_var_centered'] = self.data[running_var] - cutoff
        
        # Get treatment and control groups
        self.treated = self.data[self.data[running_var] >= cutoff]
        self.control = self.data[self.data[running_var] < cutoff]
    
    def ik_bandwidth(self, kernel: str = 'triangular') -> float:
        """
        Imbens-Kalyanaraman optimal bandwidth
        
        Based on Imbens & Kalyanaraman (2012)
        
        Parameters:
        -----------
        kernel : str
            Kernel function
            
        Returns:
        --------
        float
            Optimal bandwidth
        """
        # Get data
        y = self.data[self.outcome_var].values
        x = self.data['running_var_centered'].values
        treatment = self.data[self.treatment_var].values
        
        n = len(y)
        
        # Step 1: Pilot bandwidth for density estimation
        h_pilot = self._pilot_bandwidth(x, kernel)
        
        # Step 2: Estimate density at cutoff
        density_est = self._estimate_density(x, h_pilot, kernel)
        
        # Step 3: Estimate second derivatives
        h_2nd = self._second_derivative_bandwidth(x, y, treatment, kernel)
        
        # Step 4: Calculate optimal bandwidth
        if density_est > 0:
            # Calculate variance term
            var_term = self._calculate_variance_term(x, y, treatment, h_2nd, kernel)
            
            # Calculate bias term
            bias_term = self._calculate_bias_term(x, y, treatment, h_2nd, kernel)
            
            # Optimal bandwidth
            h_opt = (var_term / (4 * bias_term**2))**(1/5) * n**(-1/5)
        else:
            h_opt = np.inf
        
        return h_opt
    
    def cct_bandwidth(self, kernel: str = 'triangular') -> float:
        """
        Calonico-Cattaneo-Titiunik optimal bandwidth
        
        Based on Calonico et al. (2014)
        
        Parameters:
        -----------
        kernel : str
            Kernel function
            
        Returns:
        --------
        float
            Optimal bandwidth
        """
        # Get data
        y = self.data[self.outcome_var].values
        x = self.data['running_var_centered'].values
        treatment = self.data[self.treatment_var].values
        
        n = len(y)
        
        # Use IK bandwidth as starting point
        h_ik = self.ik_bandwidth(kernel)
        
        # Refine using CCT procedure
        h_cct = self._cct_refinement(x, y, treatment, h_ik, kernel)
        
        return h_cct
    
    def _pilot_bandwidth(self, x: np.ndarray, kernel: str) -> float:
        """Calculate pilot bandwidth for density estimation"""
        n = len(x)
        
        # Rule of thumb bandwidth
        if kernel == 'triangular':
            C = 2.34
        elif kernel == 'rectangular':
            C = 1.84
        elif kernel == 'epanechnikov':
            C = 2.34
        else:
            C = 2.34
        
        h_pilot = C * np.std(x) * n**(-1/5)
        
        return h_pilot
    
    def _estimate_density(self, x: np.ndarray, bandwidth: float, kernel: str) -> float:
        """Estimate density at cutoff (x=0)"""
        # Kernel density estimation at x=0
        u = x / bandwidth
        
        if kernel == 'triangular':
            weights = np.maximum(0, 1 - np.abs(u))
        elif kernel == 'rectangular':
            weights = (np.abs(u) <= 1).astype(float)
        elif kernel == 'epanechnikov':
            weights = np.maximum(0, 0.75 * (1 - u**2))
        else:
            weights = np.maximum(0, 1 - np.abs(u))
        
        density = np.sum(weights) / (len(x) * bandwidth)
        
        return density
    
    def _second_derivative_bandwidth(self, x: np.ndarray, y: np.ndarray, 
                                   treatment: np.ndarray, kernel: str) -> float:
        """Calculate bandwidth for second derivative estimation"""
        n = len(x)
        
        # Use rule of thumb for second derivative
        if kernel == 'triangular':
            C = 3.42
        elif kernel == 'rectangular':
            C = 2.70
        elif kernel == 'epanechnikov':
            C = 3.42
        else:
            C = 3.42
        
        h_2nd = C * np.std(x) * n**(-1/7)
        
        return h_2nd
    
    def _calculate_variance_term(self, x: np.ndarray, y: np.ndarray, 
                               treatment: np.ndarray, bandwidth: float, kernel: str) -> float:
        """Calculate variance term for bandwidth selection"""
        # Get data within bandwidth
        within_bandwidth = np.abs(x) <= bandwidth
        x_local = x[within_bandwidth]
        y_local = y[within_bandwidth]
        treatment_local = treatment[within_bandwidth]
        
        if len(x_local) < 4:
            return np.inf
        
        # Calculate kernel weights
        u = x_local / bandwidth
        if kernel == 'triangular':
            weights = np.maximum(0, 1 - np.abs(u))
        elif kernel == 'rectangular':
            weights = (np.abs(u) <= 1).astype(float)
        elif kernel == 'epanechnikov':
            weights = np.maximum(0, 0.75 * (1 - u**2))
        else:
            weights = np.maximum(0, 1 - np.abs(u))
        
        # Calculate variance
        try:
            # Local linear regression
            X = np.column_stack([np.ones(len(x_local)), x_local, treatment_local, x_local * treatment_local])
            W = np.diag(weights)
            
            X_weighted = W @ X
            y_weighted = W @ y_local
            
            beta = linalg.solve(X_weighted.T @ X_weighted, X_weighted.T @ y_weighted)
            residuals = y_local - X @ beta
            
            # Variance term
            var_term = np.sum(weights * residuals**2) / np.sum(weights)
            
        except np.linalg.LinAlgError:
            var_term = np.inf
        
        return var_term
    
    def _calculate_bias_term(self, x: np.ndarray, y: np.ndarray, 
                           treatment: np.ndarray, bandwidth: float, kernel: str) -> float:
        """Calculate bias term for bandwidth selection"""
        # Get data within bandwidth
        within_bandwidth = np.abs(x) <= bandwidth
        x_local = x[within_bandwidth]
        y_local = y[within_bandwidth]
        treatment_local = treatment[within_bandwidth]
        
        if len(x_local) < 4:
            return np.inf
        
        # Calculate kernel weights
        u = x_local / bandwidth
        if kernel == 'triangular':
            weights = np.maximum(0, 1 - np.abs(u))
        elif kernel == 'rectangular':
            weights = (np.abs(u) <= 1).astype(float)
        elif kernel == 'epanechnikov':
            weights = np.maximum(0, 0.75 * (1 - u**2))
        else:
            weights = np.maximum(0, 1 - np.abs(u))
        
        # Calculate bias term (simplified)
        try:
            # Local quadratic regression for bias estimation
            X = np.column_stack([np.ones(len(x_local)), x_local, x_local**2, 
                               treatment_local, x_local * treatment_local, 
                               x_local**2 * treatment_local])
            W = np.diag(weights)
            
            X_weighted = W @ X
            y_weighted = W @ y_local
            
            beta = linalg.solve(X_weighted.T @ X_weighted, X_weighted.T @ y_weighted)
            
            # Bias term (coefficient on x^2)
            bias_term = abs(beta[2])
            
        except np.linalg.LinAlgError:
            bias_term = np.inf
        
        return bias_term
    
    def _cct_refinement(self, x: np.ndarray, y: np.ndarray, treatment: np.ndarray,
                       h_ik: float, kernel: str) -> float:
        """CCT refinement of IK bandwidth"""
        # CCT uses a different approach - simplified version
        # In practice, this would involve more sophisticated optimization
        
        # Use IK bandwidth as base
        h_cct = h_ik
        
        # Apply CCT-specific adjustments
        n = len(x)
        
        # CCT uses different constants
        if kernel == 'triangular':
            C_cct = 2.34
        elif kernel == 'rectangular':
            C_cct = 1.84
        elif kernel == 'epanechnikov':
            C_cct = 2.34
        else:
            C_cct = 2.34
        
        # Refined bandwidth
        h_refined = C_cct * np.std(x) * n**(-1/5)
        
        return h_refined
    
    def kernel_regression(self, bandwidth: float, kernel: str = 'triangular',
                         robust: bool = True) -> Dict:
        """
        Kernel regression RD estimation
        
        Parameters:
        -----------
        bandwidth : float
            Bandwidth for kernel regression
        kernel : str
            Kernel function
        robust : bool
            Whether to use robust standard errors
            
        Returns:
        --------
        dict
            Kernel regression results
        """
        # Get data
        y = self.data[self.outcome_var].values
        x = self.data['running_var_centered'].values
        treatment = self.data[self.treatment_var].values
        
        # Get data within bandwidth
        within_bandwidth = np.abs(x) <= bandwidth
        x_local = x[within_bandwidth]
        y_local = y[within_bandwidth]
        treatment_local = treatment[within_bandwidth]
        
        if len(x_local) < 4:
            return {'rd_estimate': np.nan, 'se': np.nan, 't_stat': np.nan, 'p_value': np.nan}
        
        # Calculate kernel weights
        u = x_local / bandwidth
        if kernel == 'triangular':
            weights = np.maximum(0, 1 - np.abs(u))
        elif kernel == 'rectangular':
            weights = (np.abs(u) <= 1).astype(float)
        elif kernel == 'epanechnikov':
            weights = np.maximum(0, 0.75 * (1 - u**2))
        else:
            weights = np.maximum(0, 1 - np.abs(u))
        
        # Local linear regression
        X = np.column_stack([np.ones(len(x_local)), x_local, treatment_local, x_local * treatment_local])
        
        try:
            # Weighted regression
            W = np.diag(weights)
            X_weighted = W @ X
            y_weighted = W @ y_local
            
            beta = linalg.solve(X_weighted.T @ X_weighted, X_weighted.T @ y_weighted)
            
            # RD estimate
            rd_estimate = beta[2]
            
            # Standard errors
            if robust:
                se = self._calculate_robust_se_kernel(X, y_local, beta, weights)
            else:
                se = self._calculate_standard_se_kernel(X, y_local, beta, weights)
            
            # T-statistic and p-value
            t_stat = rd_estimate / se if se > 0 else np.nan
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(y_local) - 4)) if not np.isnan(t_stat) else np.nan
            
        except np.linalg.LinAlgError:
            rd_estimate = np.nan
            se = np.nan
            t_stat = np.nan
            p_value = np.nan
        
        return {
            'rd_estimate': rd_estimate,
            'se': se,
            't_statistic': t_stat,
            'p_value': p_value,
            'bandwidth': bandwidth,
            'n_obs': len(x_local),
            'kernel': kernel
        }
    
    def _calculate_standard_se_kernel(self, X: np.ndarray, y: np.ndarray, 
                                    beta: np.ndarray, weights: np.ndarray) -> float:
        """Calculate standard standard errors for kernel regression"""
        residuals = y - X @ beta
        mse = np.sum(weights * residuals**2) / (np.sum(weights) - X.shape[1])
        
        try:
            XTX_inv = linalg.inv(X.T @ X)
            se = np.sqrt(mse * XTX_inv[2, 2])
        except np.linalg.LinAlgError:
            se = np.nan
        
        return se
    
    def _calculate_robust_se_kernel(self, X: np.ndarray, y: np.ndarray, 
                                  beta: np.ndarray, weights: np.ndarray) -> float:
        """Calculate robust standard errors for kernel regression"""
        residuals = y - X @ beta
        
        try:
            XTX_inv = linalg.inv(X.T @ X)
            meat = X.T @ np.diag(weights * residuals**2) @ X
            vcov = XTX_inv @ meat @ XTX_inv
            se = np.sqrt(vcov[2, 2])
        except np.linalg.LinAlgError:
            se = np.nan
        
        return se
    
    def bandwidth_sensitivity(self, bandwidths: List[float], 
                            kernel: str = 'triangular') -> pd.DataFrame:
        """
        Test sensitivity to bandwidth choice
        
        Parameters:
        -----------
        bandwidths : list
            List of bandwidths to test
        kernel : str
            Kernel function
            
        Returns:
        --------
        pd.DataFrame
            Sensitivity analysis results
        """
        results = []
        
        for h in bandwidths:
            result = self.kernel_regression(h, kernel)
            
            results.append({
                'Bandwidth': h,
                'RD Estimate': result['rd_estimate'],
                'Standard Error': result['se'],
                'T-statistic': result['t_statistic'],
                'P-value': result['p_value'],
                'N Observations': result['n_obs']
            })
        
        return pd.DataFrame(results)
    
    def kernel_density_test(self, bandwidth: float, kernel: str = 'triangular') -> Dict:
        """
        Test for manipulation using kernel density estimation
        
        Parameters:
        -----------
        bandwidth : float
            Bandwidth for density estimation
        kernel : str
            Kernel function
            
        Returns:
        --------
        dict
            Density test results
        """
        # Get running variable
        x = self.data['running_var_centered'].values
        
        # Estimate density on each side of cutoff
        x_left = x[x < 0]
        x_right = x[x >= 0]
        
        if len(x_left) < 2 or len(x_right) < 2:
            return {'test_statistic': np.nan, 'p_value': np.nan}
        
        # Estimate density at cutoff from each side
        density_left = self._estimate_density(x_left, bandwidth, kernel)
        density_right = self._estimate_density(x_right, bandwidth, kernel)
        
        # Test for discontinuity in density
        if density_left > 0 and density_right > 0:
            # Log difference test
            log_diff = np.log(density_right) - np.log(density_left)
            
            # Standard error of log difference
            se_log_diff = np.sqrt(1/len(x_left) + 1/len(x_right))
            
            # Test statistic
            test_stat = log_diff / se_log_diff
            p_value = 2 * (1 - stats.norm.cdf(abs(test_stat)))
        else:
            test_stat = np.nan
            p_value = np.nan
        
        return {
            'test_statistic': test_stat,
            'p_value': p_value,
            'density_left': density_left,
            'density_right': density_right,
            'log_difference': log_diff if not np.isnan(test_stat) else np.nan
        }