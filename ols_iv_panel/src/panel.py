"""
Panel Data Econometrics with Fixed Effects and Random Effects
Based on Wooldridge (2010) "Econometric Analysis of Cross Section and Panel Data"
and Baltagi (2013) "Econometric Analysis of Panel Data"
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Union
from scipy import stats, linalg
import warnings
from .ols import OLS

class PanelData:
    """
    Panel data analysis with fixed effects, random effects, and first differences
    """
    
    def __init__(self, data: pd.DataFrame, y_var: str, x_vars: List[str],
                 entity_var: str, time_var: str):
        """
        Initialize panel data analysis
        
        Parameters:
        -----------
        data : pd.DataFrame
            Panel dataset
        y_var : str
            Dependent variable name
        x_vars : list
            Independent variable names
        entity_var : str
            Entity identifier (e.g., 'firm_id', 'country_id')
        time_var : str
            Time identifier (e.g., 'year', 'quarter')
        """
        self.data = data.copy()
        self.y_var = y_var
        self.x_vars = x_vars
        self.entity_var = entity_var
        self.time_var = time_var
        
        # Sort data by entity and time
        self.data = self.data.sort_values([entity_var, time_var]).reset_index(drop=True)
        
        # Get dimensions
        self.n_entities = self.data[entity_var].nunique()
        self.n_periods = self.data[time_var].nunique()
        self.n_obs = len(self.data)
        
        # Check for balanced panel
        self.balanced = self._check_balanced_panel()
        
        # Create entity and time dummies
        self._create_dummies()
    
    def _check_balanced_panel(self) -> bool:
        """Check if panel is balanced"""
        entity_counts = self.data.groupby(self.entity_var).size()
        return entity_counts.nunique() == 1
    
    def _create_dummies(self):
        """Create entity and time dummy variables"""
        # Entity dummies
        entity_dummies = pd.get_dummies(self.data[self.entity_var], prefix='entity')
        self.entity_dummies = entity_dummies.iloc[:, :-1]  # Drop last to avoid multicollinearity
        
        # Time dummies
        time_dummies = pd.get_dummies(self.data[self.time_var], prefix='time')
        self.time_dummies = time_dummies.iloc[:, :-1]  # Drop last to avoid multicollinearity
    
    def within_estimator(self, robust: bool = True) -> Dict:
        """
        Within (Fixed Effects) estimator
        
        Implements the within transformation to eliminate entity fixed effects
        (Wooldridge, 2010, p. 308-312)
        """
        # Within transformation: subtract entity means
        y_within = self._within_transform(self.data[self.y_var])
        X_within = self._within_transform(self.data[self.x_vars])
        
        # Fit OLS on transformed data
        model = OLS(y_within, X_within, self.x_vars)
        model.fit(robust=robust)
        
        # Calculate R-squared within
        r2_within = 1 - model.ssr / np.sum((y_within - np.mean(y_within))**2)
        
        return {
            'model': model,
            'r2_within': r2_within,
            'n_entities': self.n_entities,
            'n_obs': self.n_obs
        }
    
    def between_estimator(self, robust: bool = True) -> Dict:
        """
        Between estimator using entity means
        
        (Wooldridge, 2010, p. 312-314)
        """
        # Calculate entity means
        entity_means = self.data.groupby(self.entity_var)[[self.y_var] + self.x_vars].mean()
        
        y_between = entity_means[self.y_var].values
        X_between = entity_means[self.x_vars].values
        
        # Fit OLS on entity means
        model = OLS(y_between, X_between, self.x_vars)
        model.fit(robust=robust)
        
        return {
            'model': model,
            'n_entities': self.n_entities
        }
    
    def random_effects(self, robust: bool = True) -> Dict:
        """
        Random Effects (GLS) estimator
        
        Assumes entity effects are random and uncorrelated with regressors
        (Wooldridge, 2010, p. 314-320)
        """
        # Calculate theta (transformation parameter)
        sigma_u = self._estimate_sigma_u()
        sigma_e = self._estimate_sigma_e()
        
        if sigma_u + sigma_e == 0:
            warnings.warn("Both sigma_u and sigma_e are zero")
            theta = 0
        else:
            theta = 1 - np.sqrt(sigma_e / (sigma_e + self.n_periods * sigma_u))
        
        # Transform data
        y_re, X_re = self._random_effects_transform(theta)
        
        # Fit OLS on transformed data
        model = OLS(y_re, X_re, self.x_vars)
        model.fit(robust=robust)
        
        return {
            'model': model,
            'theta': theta,
            'sigma_u': sigma_u,
            'sigma_e': sigma_e
        }
    
    def first_differences(self, robust: bool = True) -> Dict:
        """
        First Differences estimator
        
        Eliminates entity fixed effects by taking first differences
        (Wooldridge, 2010, p. 320-325)
        """
        # Calculate first differences
        y_fd = self._first_differences(self.data[self.y_var])
        X_fd = self._first_differences(self.data[self.x_vars])
        
        # Remove observations where we can't calculate differences
        valid_obs = ~(y_fd.isna().any(axis=1) | X_fd.isna().any(axis=1))
        y_fd = y_fd[valid_obs].values
        X_fd = X_fd[valid_obs].values
        
        # Fit OLS on first differences
        model = OLS(y_fd, X_fd, self.x_vars)
        model.fit(robust=robust)
        
        return {
            'model': model,
            'n_obs_fd': len(y_fd)
        }
    
    def hausman_test(self) -> Dict:
        """
        Hausman test for fixed effects vs random effects
        
        Tests H0: Random effects are consistent (no correlation between
        entity effects and regressors)
        (Hausman, 1978; Wooldridge, 2010, p. 325-327)
        """
        # Get fixed effects and random effects estimates
        fe_results = self.within_estimator()
        re_results = self.random_effects()
        
        # Extract coefficients (excluding constant)
        beta_fe = fe_results['model'].coefficients[1:]  # Skip constant
        beta_re = re_results['model'].coefficients[1:]
        
        # Calculate difference
        diff = beta_fe - beta_re
        
        # Calculate variance of difference
        var_fe = np.diag(fe_results['model'].se[1:]**2)
        var_re = np.diag(re_results['model'].se[1:]**2)
        var_diff = var_fe - var_re
        
        # Check for negative variances (can happen with small samples)
        if np.any(var_diff <= 0):
            warnings.warn("Negative variance in Hausman test - may indicate specification problems")
            return {'test_statistic': np.nan, 'p_value': np.nan, 'reject_random_effects': False}
        
        # Hausman statistic
        hausman_stat = diff.T @ np.linalg.inv(np.diag(var_diff)) @ diff
        
        # P-value
        p_value = 1 - stats.chi2.cdf(hausman_stat, len(diff))
        
        return {
            'test_statistic': hausman_stat,
            'p_value': p_value,
            'degrees_of_freedom': len(diff),
            'reject_random_effects': p_value < 0.05
        }
    
    def _within_transform(self, data: Union[pd.Series, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """Apply within transformation (subtract entity means)"""
        if isinstance(data, pd.Series):
            entity_means = data.groupby(self.data[self.entity_var]).transform('mean')
            return data - entity_means
        else:
            return data.groupby(self.data[self.entity_var]).transform(lambda x: x - x.mean())
    
    def _first_differences(self, data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """Calculate first differences"""
        if isinstance(data, pd.Series):
            return data.groupby(self.data[self.entity_var]).diff()
        else:
            return data.groupby(self.data[self.entity_var]).diff()
    
    def _random_effects_transform(self, theta: float) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random effects transformation"""
        # Get entity means
        y_means = self.data.groupby(self.entity_var)[self.y_var].transform('mean')
        X_means = self.data.groupby(self.entity_var)[self.x_vars].transform('mean')
        
        # Transform
        y_re = self.data[self.y_var] - theta * y_means
        X_re = self.data[self.x_vars] - theta * X_means
        
        return y_re.values, X_re.values
    
    def _estimate_sigma_u(self) -> float:
        """Estimate sigma_u (variance of entity effects)"""
        # Use between estimator residuals
        between_results = self.between_estimator()
        y_means = self.data.groupby(self.entity_var)[self.y_var].mean()
        X_means = self.data.groupby(self.entity_var)[self.x_vars].mean()
        
        y_pred = X_means @ between_results['model'].coefficients[1:] + between_results['model'].coefficients[0]
        residuals = y_means - y_pred
        
        # Adjust for degrees of freedom
        sigma_u = np.var(residuals) - self._estimate_sigma_e() / self.n_periods
        return max(0, sigma_u)  # Ensure non-negative
    
    def _estimate_sigma_e(self) -> float:
        """Estimate sigma_e (variance of idiosyncratic error)"""
        # Use within estimator residuals
        within_results = self.within_estimator()
        return within_results['model'].ssr / (self.n_obs - self.n_entities - len(self.x_vars))
    
    def breusch_pagan_test(self) -> Dict:
        """
        Breusch-Pagan test for random effects
        
        Tests H0: sigma_u = 0 (no random effects)
        (Breusch & Pagan, 1980)
        """
        # Pooled OLS residuals
        y_pooled = self.data[self.y_var].values
        X_pooled = self.data[self.x_vars].values
        
        pooled_model = OLS(y_pooled, X_pooled, self.x_vars)
        pooled_model.fit()
        
        # Calculate entity means of squared residuals
        residuals_sq = pooled_model.residuals**2
        entity_means_sq = pd.Series(residuals_sq).groupby(self.data[self.entity_var]).mean()
        
        # Regression of squared residuals on entity means
        n_entities = len(entity_means_sq)
        entity_means_sq = entity_means_sq.values
        
        # Test statistic
        lm_stat = (self.n_obs**2 / (2 * (self.n_obs - 1))) * \
                  (np.sum(entity_means_sq) / np.sum(residuals_sq) - 1)**2
        
        # P-value
        p_value = 1 - stats.chi2.cdf(lm_stat, 1)
        
        return {
            'test_statistic': lm_stat,
            'p_value': p_value,
            'reject_no_random_effects': p_value < 0.05
        }
    
    def wooldridge_test(self) -> Dict:
        """
        Wooldridge test for serial correlation in first differences
        
        Tests H0: No serial correlation in first differences
        (Wooldridge, 2002)
        """
        # First differences
        y_fd = self._first_differences(self.data[self.y_var])
        X_fd = self._first_differences(self.data[self.x_vars])
        
        # Remove invalid observations
        valid_obs = ~(y_fd.isna() | X_fd.isna().any(axis=1))
        y_fd = y_fd[valid_obs]
        X_fd = X_fd[valid_obs]
        
        # Fit first differences model
        fd_model = OLS(y_fd.values, X_fd.values, self.x_vars)
        fd_model.fit()
        
        # Lagged residuals
        residuals_fd = pd.Series(fd_model.residuals)
        residuals_fd_lag = residuals_fd.shift(1)
        
        # Remove first observation of each entity
        entity_groups = self.data[self.entity_var][valid_obs]
        first_obs = entity_groups != entity_groups.shift(1)
        residuals_fd = residuals_fd[~first_obs]
        residuals_fd_lag = residuals_fd_lag[~first_obs]
        
        # Regression of residuals on lagged residuals
        if len(residuals_fd) > 1:
            rho = np.corrcoef(residuals_fd, residuals_fd_lag)[0, 1]
            t_stat = rho * np.sqrt(len(residuals_fd) - 1)
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(residuals_fd) - 1))
        else:
            t_stat = np.nan
            p_value = np.nan
        
        return {
            'test_statistic': t_stat,
            'p_value': p_value,
            'reject_no_serial_correlation': p_value < 0.05 if not np.isnan(p_value) else False
        }