"""
Regression Discontinuity Estimation
Based on Imbens & Lemieux (2008) "Regression Discontinuity Designs"
and Calonico, Cattaneo, and Titiunik (2014) "Robust Nonparametric Confidence Intervals"
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Union
from scipy import stats, linalg
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings

class RegressionDiscontinuity:
    """
    Regression discontinuity estimation with robust inference
    
    Implements various RD estimators including:
    - Local linear regression
    - Polynomial regression
    - Robust confidence intervals
    - Bandwidth selection
    """
    
    def __init__(self, data: pd.DataFrame, running_var: str, outcome_var: str,
                 cutoff: float, treatment_var: Optional[str] = None):
        """
        Initialize RD analysis
        
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
            Treatment indicator (if not binary based on cutoff)
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
        
        self.n_treated = len(self.treated)
        self.n_control = len(self.control)
    
    def local_linear_regression(self, bandwidth: Optional[float] = None,
                               kernel: str = 'triangular',
                               robust: bool = True) -> Dict:
        """
        Local linear regression RD estimator
        
        Based on Imbens & Lemieux (2008) and Calonico et al. (2014)
        
        Parameters:
        -----------
        bandwidth : float, optional
            Bandwidth for local regression
        kernel : str
            Kernel function ('triangular', 'rectangular', 'epanechnikov')
        robust : bool
            Whether to use robust standard errors
            
        Returns:
        --------
        dict
            Local linear regression results
        """
        # Select bandwidth if not provided
        if bandwidth is None:
            bandwidth = self._select_bandwidth()
        
        # Get data within bandwidth
        within_bandwidth = np.abs(self.data['running_var_centered']) <= bandwidth
        data_local = self.data[within_bandwidth].copy()
        
        if len(data_local) == 0:
            return {'rd_estimate': np.nan, 'se': np.nan, 't_stat': np.nan, 'p_value': np.nan}
        
        # Prepare variables
        y = data_local[self.outcome_var].values
        x = data_local['running_var_centered'].values
        treatment = data_local[self.treatment_var].values
        
        # Create interaction term
        x_treated = x * treatment
        
        # Kernel weights
        weights = self._calculate_kernel_weights(x, bandwidth, kernel)
        
        # Weighted least squares
        X = np.column_stack([np.ones(len(x)), x, treatment, x_treated])
        
        try:
            # Weighted regression
            W = np.diag(weights)
            X_weighted = W @ X
            y_weighted = W @ y
            
            # Solve normal equations
            beta = linalg.solve(X_weighted.T @ X_weighted, X_weighted.T @ y_weighted)
            
            # RD estimate is the coefficient on treatment
            rd_estimate = beta[2]
            
            # Calculate standard errors
            if robust:
                se = self._calculate_robust_se(X, y, beta, weights)
            else:
                se = self._calculate_standard_se(X, y, beta, weights)
            
            # T-statistic and p-value
            t_stat = rd_estimate / se if se > 0 else np.nan
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(y) - 4)) if not np.isnan(t_stat) else np.nan
            
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
            'n_obs': len(data_local),
            'kernel': kernel
        }
    
    def polynomial_regression(self, degree: int = 3, 
                            interaction: bool = True) -> Dict:
        """
        Polynomial regression RD estimator
        
        Parameters:
        -----------
        degree : int
            Polynomial degree
        interaction : bool
            Whether to include treatment interactions
            
        Returns:
        --------
        dict
            Polynomial regression results
        """
        # Prepare variables
        y = self.data[self.outcome_var].values
        x = self.data['running_var_centered'].values
        treatment = self.data[self.treatment_var].values
        
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=True)
        X_poly = poly.fit_transform(x.reshape(-1, 1))
        
        # Add treatment and interactions
        if interaction:
            X = np.column_stack([X_poly, treatment])
            for i in range(1, degree + 1):
                X = np.column_stack([X, treatment * (x ** i)])
        else:
            X = np.column_stack([X_poly, treatment])
        
        # Fit regression
        try:
            model = LinearRegression()
            model.fit(X, y)
            
            # RD estimate is the coefficient on treatment
            rd_estimate = model.coef_[-1] if not interaction else model.coef_[degree + 1]
            
            # Calculate standard errors
            residuals = y - model.predict(X)
            mse = np.sum(residuals**2) / (len(y) - X.shape[1])
            
            try:
                XTX_inv = linalg.inv(X.T @ X)
                se = np.sqrt(mse * XTX_inv[-1, -1])
            except np.linalg.LinAlgError:
                se = np.nan
            
            # T-statistic and p-value
            t_stat = rd_estimate / se if se > 0 else np.nan
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(y) - X.shape[1])) if not np.isnan(t_stat) else np.nan
            
        except Exception as e:
            rd_estimate = np.nan
            se = np.nan
            t_stat = np.nan
            p_value = np.nan
        
        return {
            'rd_estimate': rd_estimate,
            'se': se,
            't_statistic': t_stat,
            'p_value': p_value,
            'degree': degree,
            'n_obs': len(y),
            'interaction': interaction
        }
    
    def _select_bandwidth(self, method: str = 'imse') -> float:
        """
        Select optimal bandwidth
        
        Parameters:
        -----------
        method : str
            Bandwidth selection method ('imse', 'cv', 'rule_of_thumb')
            
        Returns:
        --------
        float
            Optimal bandwidth
        """
        if method == 'rule_of_thumb':
            # Rule of thumb bandwidth
            n = len(self.data)
            return 1.84 * np.std(self.data[self.running_var]) * (n ** (-1/5))
        
        elif method == 'cv':
            # Cross-validation bandwidth selection
            bandwidths = np.linspace(0.1, 2.0, 20)
            cv_scores = []
            
            for h in bandwidths:
                cv_score = self._cross_validate_bandwidth(h)
                cv_scores.append(cv_score)
            
            optimal_idx = np.argmin(cv_scores)
            return bandwidths[optimal_idx]
        
        elif method == 'imse':
            # Integrated Mean Squared Error bandwidth
            # Simplified version - in practice would use more sophisticated methods
            n = len(self.data)
            return 2.34 * np.std(self.data[self.running_var]) * (n ** (-1/5))
        
        else:
            raise ValueError("Unknown bandwidth selection method")
    
    def _cross_validate_bandwidth(self, bandwidth: float) -> float:
        """Cross-validation for bandwidth selection"""
        # Simple leave-one-out cross-validation
        cv_errors = []
        
        for i in range(len(self.data)):
            # Leave out observation i
            data_cv = self.data.drop(i)
            
            # Fit local linear regression
            within_bandwidth = np.abs(data_cv['running_var_centered']) <= bandwidth
            data_local = data_cv[within_bandwidth]
            
            if len(data_local) < 4:  # Need at least 4 observations
                continue
            
            try:
                # Fit regression
                y = data_local[self.outcome_var].values
                x = data_local['running_var_centered'].values
                treatment = data_local[self.treatment_var].values
                x_treated = x * treatment
                
                X = np.column_stack([np.ones(len(x)), x, treatment, x_treated])
                
                beta = linalg.solve(X.T @ X, X.T @ y)
                
                # Predict for left-out observation
                x_i = self.data.iloc[i]['running_var_centered']
                t_i = self.data.iloc[i][self.treatment_var]
                x_treated_i = x_i * t_i
                
                X_i = np.array([1, x_i, t_i, x_treated_i])
                y_pred = X_i @ beta
                y_actual = self.data.iloc[i][self.outcome_var]
                
                cv_errors.append((y_actual - y_pred)**2)
                
            except:
                continue
        
        return np.mean(cv_errors) if cv_errors else np.inf
    
    def _calculate_kernel_weights(self, x: np.ndarray, bandwidth: float, 
                                kernel: str) -> np.ndarray:
        """Calculate kernel weights"""
        u = x / bandwidth
        
        if kernel == 'triangular':
            weights = np.maximum(0, 1 - np.abs(u))
        elif kernel == 'rectangular':
            weights = (np.abs(u) <= 1).astype(float)
        elif kernel == 'epanechnikov':
            weights = np.maximum(0, 0.75 * (1 - u**2))
        else:
            raise ValueError("Unknown kernel function")
        
        return weights
    
    def _calculate_standard_se(self, X: np.ndarray, y: np.ndarray, 
                             beta: np.ndarray, weights: np.ndarray) -> float:
        """Calculate standard standard errors"""
        residuals = y - X @ beta
        mse = np.sum(weights * residuals**2) / (np.sum(weights) - X.shape[1])
        
        try:
            XTX_inv = linalg.inv(X.T @ X)
            se = np.sqrt(mse * XTX_inv[2, 2])  # Treatment coefficient
        except np.linalg.LinAlgError:
            se = np.nan
        
        return se
    
    def _calculate_robust_se(self, X: np.ndarray, y: np.ndarray, 
                           beta: np.ndarray, weights: np.ndarray) -> float:
        """Calculate robust standard errors"""
        residuals = y - X @ beta
        
        # White/Huber-White standard errors
        try:
            XTX_inv = linalg.inv(X.T @ X)
            meat = X.T @ np.diag(weights * residuals**2) @ X
            vcov = XTX_inv @ meat @ XTX_inv
            se = np.sqrt(vcov[2, 2])  # Treatment coefficient
        except np.linalg.LinAlgError:
            se = np.nan
        
        return se
    
    def test_manipulation(self, bins: int = 20) -> Dict:
        """
        Test for manipulation of running variable
        
        Based on McCrary (2008) "Manipulation of the Running Variable in the Regression Discontinuity Design"
        
        Parameters:
        -----------
        bins : int
            Number of bins for histogram
            
        Returns:
        --------
        dict
            Manipulation test results
        """
        # Create histogram
        hist, bin_edges = np.histogram(self.data[self.running_var], bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Find bin containing cutoff
        cutoff_bin = np.argmin(np.abs(bin_centers - self.cutoff))
        
        # Test for discontinuity in density
        if cutoff_bin > 0 and cutoff_bin < len(hist) - 1:
            # Compare densities on either side of cutoff
            left_density = hist[cutoff_bin - 1]
            right_density = hist[cutoff_bin + 1]
            
            # Test statistic
            if left_density > 0 and right_density > 0:
                test_stat = (right_density - left_density) / np.sqrt(left_density + right_density)
                p_value = 2 * (1 - stats.norm.cdf(abs(test_stat)))
            else:
                test_stat = np.nan
                p_value = np.nan
        else:
            test_stat = np.nan
            p_value = np.nan
        
        return {
            'test_statistic': test_stat,
            'p_value': p_value,
            'histogram': hist,
            'bin_centers': bin_centers,
            'cutoff_bin': cutoff_bin,
            'manipulation_detected': p_value < 0.05 if not np.isnan(p_value) else False
        }
    
    def placebo_test(self, placebo_cutoffs: List[float]) -> Dict:
        """
        Placebo test using fake cutoffs
        
        Parameters:
        -----------
        placebo_cutoffs : list
            List of fake cutoffs to test
            
        Returns:
        --------
        dict
            Placebo test results
        """
        results = {}
        
        for cutoff in placebo_cutoffs:
            # Create fake treatment indicator
            fake_treatment = (self.data[self.running_var] >= cutoff).astype(int)
            
            # Estimate fake RD
            y = self.data[self.outcome_var].values
            x = self.data[self.running_var].values - cutoff
            treatment = fake_treatment.values
            
            # Simple regression
            X = np.column_stack([np.ones(len(x)), x, treatment, x * treatment])
            
            try:
                beta = linalg.solve(X.T @ X, X.T @ y)
                rd_estimate = beta[2]
                
                # Standard error
                residuals = y - X @ beta
                mse = np.sum(residuals**2) / (len(y) - X.shape[1])
                XTX_inv = linalg.inv(X.T @ X)
                se = np.sqrt(mse * XTX_inv[2, 2])
                
                t_stat = rd_estimate / se if se > 0 else np.nan
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(y) - X.shape[1])) if not np.isnan(t_stat) else np.nan
                
            except:
                rd_estimate = np.nan
                se = np.nan
                t_stat = np.nan
                p_value = np.nan
            
            results[cutoff] = {
                'rd_estimate': rd_estimate,
                'se': se,
                't_statistic': t_stat,
                'p_value': p_value
            }
        
        return results
    
    def summary(self) -> pd.DataFrame:
        """Return summary of RD results"""
        # Local linear regression
        ll_results = self.local_linear_regression()
        
        # Polynomial regression
        poly_results = self.polynomial_regression()
        
        summary_data = {
            'Method': ['Local Linear', 'Polynomial'],
            'RD Estimate': [ll_results['rd_estimate'], poly_results['rd_estimate']],
            'Standard Error': [ll_results['se'], poly_results['se']],
            'T-statistic': [ll_results['t_statistic'], poly_results['t_statistic']],
            'P-value': [ll_results['p_value'], poly_results['p_value']],
            'N Observations': [ll_results['n_obs'], poly_results['n_obs']]
        }
        
        return pd.DataFrame(summary_data)