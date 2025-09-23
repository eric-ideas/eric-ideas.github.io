"""
Hodrick-Prescott Filter Implementation
Based on Hodrick & Prescott (1997) "Postwar U.S. Business Cycles: An Empirical Investigation"
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Union
from scipy import linalg
import warnings

class HPFilter:
    """
    Hodrick-Prescott filter for business cycle analysis
    
    Implements the HP filter with:
    - Optimal lambda selection
    - Trend and cycle decomposition
    - Statistical properties analysis
    """
    
    def __init__(self, lambda_param: float = 1600):
        """
        Initialize HP filter
        
        Parameters:
        -----------
        lambda_param : float
            Smoothing parameter (1600 for quarterly data, 100 for annual)
        """
        self.lambda_param = lambda_param
    
    def filter(self, data: Union[np.ndarray, pd.Series], 
               lambda_param: Optional[float] = None) -> Dict:
        """
        Apply HP filter to data
        
        Parameters:
        -----------
        data : np.ndarray or pd.Series
            Time series data
        lambda_param : float, optional
            Smoothing parameter
            
        Returns:
        --------
        dict
            Filter results
        """
        if lambda_param is None:
            lambda_param = self.lambda_param
        
        # Convert to numpy array
        if isinstance(data, pd.Series):
            y = data.values
            index = data.index
        else:
            y = np.array(data)
            index = np.arange(len(y))
        
        n = len(y)
        
        # Create HP filter matrix
        # The HP filter solves: min Σ(y_t - τ_t)² + λΣ(τ_{t+1} - 2τ_t + τ_{t-1})²
        # This can be written as: min (y - τ)'I(y - τ) + λτ'K'Kτ
        # where K is the second difference matrix
        
        # Second difference matrix
        K = self._create_second_diff_matrix(n)
        
        # HP filter solution: τ = (I + λK'K)^(-1)y
        I = np.eye(n)
        A = I + lambda_param * K.T @ K
        
        try:
            # Solve for trend
            trend = linalg.solve(A, y)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            trend = linalg.pinv(A) @ y
        
        # Calculate cycle
        cycle = y - trend
        
        # Create results DataFrame
        results = pd.DataFrame({
            'original': y,
            'trend': trend,
            'cycle': cycle
        }, index=index)
        
        return {
            'data': results,
            'lambda_param': lambda_param,
            'n_obs': n,
            'trend': trend,
            'cycle': cycle
        }
    
    def _create_second_diff_matrix(self, n: int) -> np.ndarray:
        """Create second difference matrix"""
        K = np.zeros((n-2, n))
        
        for i in range(n-2):
            K[i, i] = 1
            K[i, i+1] = -2
            K[i, i+2] = 1
        
        return K
    
    def optimal_lambda(self, data: Union[np.ndarray, pd.Series],
                      lambda_range: Tuple[float, float] = (1, 10000),
                      n_points: int = 100) -> float:
        """
        Find optimal lambda parameter using cross-validation
        
        Parameters:
        -----------
        data : np.ndarray or pd.Series
            Time series data
        lambda_range : tuple
            Range of lambda values to test
        n_points : int
            Number of lambda values to test
            
        Returns:
        --------
        float
            Optimal lambda parameter
        """
        # Convert to numpy array
        if isinstance(data, pd.Series):
            y = data.values
        else:
            y = np.array(data)
        
        n = len(y)
        lambda_values = np.logspace(np.log10(lambda_range[0]), 
                                  np.log10(lambda_range[1]), n_points)
        
        cv_scores = []
        
        for lam in lambda_values:
            # Leave-one-out cross-validation
            cv_errors = []
            
            for i in range(n):
                # Remove observation i
                y_cv = np.concatenate([y[:i], y[i+1:]])
                
                if len(y_cv) < 3:  # Need at least 3 observations
                    continue
                
                # Apply HP filter
                try:
                    K_cv = self._create_second_diff_matrix(len(y_cv))
                    I_cv = np.eye(len(y_cv))
                    A_cv = I_cv + lam * K_cv.T @ K_cv
                    trend_cv = linalg.solve(A_cv, y_cv)
                    
                    # Predict removed observation
                    if i == 0:
                        y_pred = trend_cv[0]
                    elif i == n-1:
                        y_pred = trend_cv[-1]
                    else:
                        # Interpolate
                        y_pred = (trend_cv[i-1] + trend_cv[i]) / 2
                    
                    cv_errors.append((y[i] - y_pred)**2)
                    
                except:
                    continue
            
            if cv_errors:
                cv_scores.append(np.mean(cv_errors))
            else:
                cv_scores.append(np.inf)
        
        # Find optimal lambda
        optimal_idx = np.argmin(cv_scores)
        optimal_lambda = lambda_values[optimal_idx]
        
        return optimal_lambda
    
    def frequency_response(self, lambda_param: Optional[float] = None,
                          frequencies: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate frequency response of HP filter
        
        Parameters:
        -----------
        lambda_param : float, optional
            Smoothing parameter
        frequencies : np.ndarray, optional
            Frequencies to evaluate
            
        Returns:
        --------
        dict
            Frequency response results
        """
        if lambda_param is None:
            lambda_param = self.lambda_param
        
        if frequencies is None:
            frequencies = np.linspace(0, 0.5, 1000)
        
        # Frequency response of HP filter
        # H(ω) = 1 / (1 + λ(2 - 2cos(2πω))²)
        omega = 2 * np.pi * frequencies
        response = 1 / (1 + lambda_param * (2 - 2 * np.cos(omega))**2)
        
        return {
            'frequencies': frequencies,
            'response': response,
            'lambda_param': lambda_param
        }
    
    def statistical_properties(self, cycle: np.ndarray) -> Dict:
        """
        Calculate statistical properties of cycle
        
        Parameters:
        -----------
        cycle : np.ndarray
            Cyclical component
            
        Returns:
        --------
        dict
            Statistical properties
        """
        # Basic statistics
        mean_cycle = np.mean(cycle)
        std_cycle = np.std(cycle)
        skewness = np.mean((cycle - mean_cycle)**3) / (std_cycle**3)
        kurtosis = np.mean((cycle - mean_cycle)**4) / (std_cycle**4)
        
        # Autocorrelation
        autocorr = np.corrcoef(cycle[:-1], cycle[1:])[0, 1]
        
        # Persistence (AR(1) coefficient)
        if len(cycle) > 1:
            X = cycle[:-1].reshape(-1, 1)
            y = cycle[1:]
            try:
                persistence = linalg.lstsq(X, y)[0][0]
            except:
                persistence = np.nan
        else:
            persistence = np.nan
        
        return {
            'mean': mean_cycle,
            'std': std_cycle,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'autocorrelation': autocorr,
            'persistence': persistence
        }
    
    def compare_lambda(self, data: Union[np.ndarray, pd.Series],
                      lambda_values: List[float]) -> pd.DataFrame:
        """
        Compare HP filter results for different lambda values
        
        Parameters:
        -----------
        data : np.ndarray or pd.Series
            Time series data
        lambda_values : list
            List of lambda values to compare
            
        Returns:
        --------
        pd.DataFrame
            Comparison results
        """
        results = []
        
        for lam in lambda_values:
            # Apply HP filter
            filter_results = self.filter(data, lam)
            cycle = filter_results['cycle']
            
            # Calculate properties
            properties = self.statistical_properties(cycle)
            
            results.append({
                'lambda': lam,
                'cycle_std': properties['std'],
                'cycle_skewness': properties['skewness'],
                'cycle_kurtosis': properties['kurtosis'],
                'cycle_autocorr': properties['autocorrelation'],
                'cycle_persistence': properties['persistence']
            })
        
        return pd.DataFrame(results)
    
    def summary(self, data: Union[np.ndarray, pd.Series],
                lambda_param: Optional[float] = None) -> Dict:
        """
        Return summary of HP filter results
        
        Parameters:
        -----------
        data : np.ndarray or pd.Series
            Time series data
        lambda_param : float, optional
            Smoothing parameter
            
        Returns:
        --------
        dict
            Summary results
        """
        # Apply filter
        filter_results = self.filter(data, lambda_param)
        cycle = filter_results['cycle']
        
        # Calculate properties
        properties = self.statistical_properties(cycle)
        
        return {
            'lambda_param': filter_results['lambda_param'],
            'n_obs': filter_results['n_obs'],
            'cycle_properties': properties,
            'trend_properties': {
                'mean': np.mean(filter_results['trend']),
                'std': np.std(filter_results['trend']),
                'trend': np.polyfit(range(len(filter_results['trend'])), 
                                  filter_results['trend'], 1)[0]
            }
        }