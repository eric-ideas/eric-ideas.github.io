"""
Propensity Score Matching Algorithms
Based on Rosenbaum & Rubin (1983) and Imbens & Wooldridge (2009)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Union
from scipy import stats
from sklearn.neighbors import NearestNeighbors
import warnings

class PropensityScoreMatching:
    """
    Propensity score matching with various algorithms
    
    Implements:
    - Nearest neighbor matching
    - Caliper matching
    - Kernel matching
    - Radius matching
    """
    
    def __init__(self, data: pd.DataFrame, treatment_var: str, 
                 propensity_scores: np.ndarray, outcome_var: str):
        """
        Initialize propensity score matching
        
        Parameters:
        -----------
        data : pd.DataFrame
            Dataset
        treatment_var : str
            Treatment indicator
        propensity_scores : np.ndarray
            Estimated propensity scores
        outcome_var : str
            Outcome variable
        """
        self.data = data.copy()
        self.treatment_var = treatment_var
        self.propensity_scores = propensity_scores
        self.outcome_var = outcome_var
        
        # Get treatment and control groups
        self.treated = self.data[self.data[treatment_var] == 1]
        self.control = self.data[self.data[treatment_var] == 0]
        self.treated_scores = propensity_scores[self.data[treatment_var] == 1]
        self.control_scores = propensity_scores[self.data[treatment_var] == 0]
        
        self.n_treated = len(self.treated)
        self.n_control = len(self.control)
    
    def nearest_neighbor_matching(self, n_neighbors: int = 1, 
                                with_replacement: bool = False) -> Dict:
        """
        Nearest neighbor propensity score matching
        
        Based on Rosenbaum & Rubin (1983) and Imbens & Wooldridge (2009, p. 28-30)
        
        Parameters:
        -----------
        n_neighbors : int
            Number of neighbors to match
        with_replacement : bool
            Whether to allow replacement of control units
            
        Returns:
        --------
        dict
            Matching results
        """
        # Find nearest neighbors
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(self.control_scores.reshape(-1, 1))
        
        distances, indices = nn.kneighbors(self.treated_scores.reshape(-1, 1))
        
        # Create matched dataset
        matched_control_indices = []
        matched_treated_indices = []
        
        for i, (dist, idx) in enumerate(zip(distances, indices)):
            if not with_replacement:
                # Remove already matched control units
                available_controls = [j for j in idx if j not in matched_control_indices]
                if len(available_controls) > 0:
                    matched_control_indices.extend(available_controls[:n_neighbors])
                    matched_treated_indices.extend([i] * len(available_controls[:n_neighbors]))
            else:
                matched_control_indices.extend(idx)
                matched_treated_indices.extend([i] * n_neighbors)
        
        # Calculate treatment effect
        treated_outcomes = self.treated[self.outcome_var].iloc[matched_treated_indices].values
        control_outcomes = self.control[self.outcome_var].iloc[matched_control_indices].values
        
        # Average treatment effect on the treated (ATT)
        att = np.mean(treated_outcomes - control_outcomes)
        
        # Calculate standard error using Abadie & Imbens (2006) formula
        n_matched = len(matched_treated_indices)
        if n_matched > 1:
            se_att = np.std(treated_outcomes - control_outcomes) / np.sqrt(n_matched)
        else:
            se_att = np.nan
        
        # Calculate t-statistic and p-value
        t_stat = att / se_att if se_att > 0 else np.nan
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n_matched - 1)) if not np.isnan(t_stat) else np.nan
        
        return {
            'att': att,
            'se_att': se_att,
            't_statistic': t_stat,
            'p_value': p_value,
            'n_matched': n_matched,
            'n_treated': self.n_treated,
            'n_control': self.n_control,
            'matched_treated_indices': matched_treated_indices,
            'matched_control_indices': matched_control_indices,
            'with_replacement': with_replacement
        }
    
    def caliper_matching(self, caliper: float = 0.25) -> Dict:
        """
        Caliper matching with propensity scores
        
        Based on Rosenbaum & Rubin (1983) and Imbens & Wooldridge (2009, p. 30-31)
        
        Parameters:
        -----------
        caliper : float
            Maximum distance for matching (in standard deviations)
            
        Returns:
        --------
        dict
            Caliper matching results
        """
        # Calculate caliper in propensity score units
        propensity_std = np.std(self.propensity_scores)
        caliper_ps = caliper * propensity_std
        
        matched_treated_indices = []
        matched_control_indices = []
        
        for i, treated_score in enumerate(self.treated_scores):
            # Find control units within caliper
            distances = np.abs(self.control_scores - treated_score)
            within_caliper = distances <= caliper_ps
            
            if np.any(within_caliper):
                # Select closest match
                control_indices = np.where(within_caliper)[0]
                closest_idx = control_indices[np.argmin(distances[within_caliper])]
                
                matched_treated_indices.append(i)
                matched_control_indices.append(closest_idx)
        
        if len(matched_treated_indices) == 0:
            return {
                'att': np.nan,
                'se_att': np.nan,
                't_statistic': np.nan,
                'p_value': np.nan,
                'n_matched': 0,
                'n_treated': self.n_treated,
                'n_control': self.n_control,
                'caliper': caliper,
                'caliper_ps': caliper_ps
            }
        
        # Calculate treatment effect
        treated_outcomes = self.treated[self.outcome_var].iloc[matched_treated_indices].values
        control_outcomes = self.control[self.outcome_var].iloc[matched_control_indices].values
        
        att = np.mean(treated_outcomes - control_outcomes)
        
        # Standard error
        n_matched = len(matched_treated_indices)
        se_att = np.std(treated_outcomes - control_outcomes) / np.sqrt(n_matched)
        
        # T-statistic and p-value
        t_stat = att / se_att if se_att > 0 else np.nan
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n_matched - 1)) if not np.isnan(t_stat) else np.nan
        
        return {
            'att': att,
            'se_att': se_att,
            't_statistic': t_stat,
            'p_value': p_value,
            'n_matched': n_matched,
            'n_treated': self.n_treated,
            'n_control': self.n_control,
            'caliper': caliper,
            'caliper_ps': caliper_ps,
            'matched_treated_indices': matched_treated_indices,
            'matched_control_indices': matched_control_indices
        }
    
    def kernel_matching(self, bandwidth: float = 0.06) -> Dict:
        """
        Kernel matching with propensity scores
        
        Based on Heckman, Ichimura, and Todd (1997) and Imbens & Wooldridge (2009, p. 31-32)
        
        Parameters:
        -----------
        bandwidth : float
            Bandwidth for kernel matching
            
        Returns:
        --------
        dict
            Kernel matching results
        """
        # Calculate weights for each treated unit
        weights = np.zeros((self.n_treated, self.n_control))
        
        for i, treated_score in enumerate(self.treated_scores):
            # Calculate distances
            distances = np.abs(self.control_scores - treated_score)
            
            # Epanechnikov kernel weights
            kernel_weights = np.maximum(0, 1 - (distances / bandwidth) ** 2)
            kernel_weights = kernel_weights / np.sum(kernel_weights) if np.sum(kernel_weights) > 0 else kernel_weights
            
            weights[i, :] = kernel_weights
        
        # Calculate treatment effect
        treated_outcomes = self.treated[self.outcome_var].values
        control_outcomes = self.control[self.outcome_var].values
        
        # Weighted average of control outcomes for each treated unit
        weighted_control_outcomes = np.sum(weights * control_outcomes, axis=1)
        
        # Average treatment effect on the treated
        att = np.mean(treated_outcomes - weighted_control_outcomes)
        
        # Calculate standard error using bootstrap
        n_bootstrap = 1000
        bootstrap_effects = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            bootstrap_indices = np.random.choice(self.n_treated, size=self.n_treated, replace=True)
            bootstrap_treated = treated_outcomes[bootstrap_indices]
            bootstrap_weights = weights[bootstrap_indices, :]
            bootstrap_control = np.sum(bootstrap_weights * control_outcomes, axis=1)
            bootstrap_att = np.mean(bootstrap_treated - bootstrap_control)
            bootstrap_effects.append(bootstrap_att)
        
        se_att = np.std(bootstrap_effects)
        
        # T-statistic and p-value
        t_stat = att / se_att if se_att > 0 else np.nan
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), self.n_treated - 1)) if not np.isnan(t_stat) else np.nan
        
        return {
            'att': att,
            'se_att': se_att,
            't_statistic': t_stat,
            'p_value': p_value,
            'n_treated': self.n_treated,
            'n_control': self.n_control,
            'bandwidth': bandwidth,
            'weights': weights
        }
    
    def radius_matching(self, radius: float = 0.1) -> Dict:
        """
        Radius matching with propensity scores
        
        Based on Dehejia & Wahba (2002) and Imbens & Wooldridge (2009, p. 32)
        
        Parameters:
        -----------
        radius : float
            Radius for matching
            
        Returns:
        --------
        dict
            Radius matching results
        """
        matched_treated_indices = []
        matched_control_indices = []
        
        for i, treated_score in enumerate(self.treated_scores):
            # Find control units within radius
            distances = np.abs(self.control_scores - treated_score)
            within_radius = distances <= radius
            
            if np.any(within_radius):
                control_indices = np.where(within_radius)[0]
                matched_treated_indices.append(i)
                matched_control_indices.append(control_indices)
        
        if len(matched_treated_indices) == 0:
            return {
                'att': np.nan,
                'se_att': np.nan,
                't_statistic': np.nan,
                'p_value': np.nan,
                'n_matched': 0,
                'n_treated': self.n_treated,
                'n_control': self.n_control,
                'radius': radius
            }
        
        # Calculate treatment effect
        treated_outcomes = self.treated[self.outcome_var].iloc[matched_treated_indices].values
        
        # Average control outcomes for each treated unit
        control_outcomes = []
        for i, control_indices in enumerate(matched_control_indices):
            control_outcomes.append(self.control[self.outcome_var].iloc[control_indices].mean())
        
        control_outcomes = np.array(control_outcomes)
        att = np.mean(treated_outcomes - control_outcomes)
        
        # Standard error
        n_matched = len(matched_treated_indices)
        se_att = np.std(treated_outcomes - control_outcomes) / np.sqrt(n_matched)
        
        # T-statistic and p-value
        t_stat = att / se_att if se_att > 0 else np.nan
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n_matched - 1)) if not np.isnan(t_stat) else np.nan
        
        return {
            'att': att,
            'se_att': se_att,
            't_statistic': t_stat,
            'p_value': p_value,
            'n_matched': n_matched,
            'n_treated': self.n_treated,
            'n_control': self.n_control,
            'radius': radius,
            'matched_treated_indices': matched_treated_indices,
            'matched_control_indices': matched_control_indices
        }
    
    def compare_methods(self, methods: List[str] = None) -> pd.DataFrame:
        """
        Compare different matching methods
        
        Parameters:
        -----------
        methods : list, optional
            List of methods to compare
            
        Returns:
        --------
        pd.DataFrame
            Comparison results
        """
        if methods is None:
            methods = ['nearest_neighbor', 'caliper', 'kernel', 'radius']
        
        results = []
        
        for method in methods:
            if method == 'nearest_neighbor':
                result = self.nearest_neighbor_matching()
            elif method == 'caliper':
                result = self.caliper_matching()
            elif method == 'kernel':
                result = self.kernel_matching()
            elif method == 'radius':
                result = self.radius_matching()
            else:
                continue
            
            results.append({
                'Method': method,
                'ATT': result['att'],
                'SE': result['se_att'],
                'T-statistic': result['t_statistic'],
                'P-value': result['p_value'],
                'N Matched': result['n_matched']
            })
        
        return pd.DataFrame(results)
    
    def plot_matching_quality(self, matched_indices: List[int], 
                             method_name: str = "Matching") -> None:
        """
        Plot matching quality assessment
        
        Parameters:
        -----------
        matched_indices : list
            Indices of matched units
        method_name : str
            Name of matching method
        """
        import matplotlib.pyplot as plt
        
        # Get matched propensity scores
        matched_treated_scores = self.treated_scores[matched_indices]
        matched_control_scores = self.control_scores[matched_indices]
        
        plt.figure(figsize=(12, 6))
        
        # Plot 1: Propensity score distributions
        plt.subplot(1, 2, 1)
        plt.hist(matched_control_scores, bins=20, alpha=0.7, label='Matched Control', 
                color='steelblue', density=True)
        plt.hist(matched_treated_scores, bins=20, alpha=0.7, label='Treated', 
                color='red', density=True)
        plt.xlabel('Propensity Score')
        plt.ylabel('Density')
        plt.title(f'{method_name} - Propensity Score Distributions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Q-Q plot
        plt.subplot(1, 2, 2)
        sorted_treated = np.sort(matched_treated_scores)
        sorted_control = np.sort(matched_control_scores)
        
        plt.scatter(sorted_control, sorted_treated, alpha=0.6)
        plt.plot([0, 1], [0, 1], 'r--', alpha=0.7)
        plt.xlabel('Matched Control Propensity Scores')
        plt.ylabel('Treated Propensity Scores')
        plt.title(f'{method_name} - Q-Q Plot')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()