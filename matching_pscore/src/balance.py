"""
Balance Testing for Propensity Score Matching
Based on Rosenbaum & Rubin (1985) and Imbens & Wooldridge (2009)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Union
from scipy import stats
import warnings

class BalanceTest:
    """
    Balance testing for propensity score matching
    
    Implements various balance tests including:
    - Standardized differences
    - T-tests for equality of means
    - Kolmogorov-Smirnov tests
    - Variance ratio tests
    """
    
    def __init__(self, data: pd.DataFrame, treatment_var: str, 
                 covariates: List[str], matched_indices: Optional[List[int]] = None):
        """
        Initialize balance testing
        
        Parameters:
        -----------
        data : pd.DataFrame
            Dataset
        treatment_var : str
            Treatment indicator
        covariates : list
            Covariates to test balance for
        matched_indices : list, optional
            Indices of matched units
        """
        self.data = data.copy()
        self.treatment_var = treatment_var
        self.covariates = covariates
        self.matched_indices = matched_indices
        
        # Get treatment and control groups
        self.treated = self.data[self.data[treatment_var] == 1]
        self.control = self.data[self.data[treatment_var] == 0]
        
        # Get matched data if indices provided
        if matched_indices is not None:
            self.matched_treated = self.treated.iloc[matched_indices]
            self.matched_control = self.control.iloc[matched_indices]
        else:
            self.matched_treated = self.treated
            self.matched_control = self.control
    
    def standardized_differences(self, matched: bool = True) -> pd.DataFrame:
        """
        Calculate standardized differences
        
        Based on Rosenbaum & Rubin (1985) and Imbens & Wooldridge (2009, p. 33-34)
        
        Parameters:
        -----------
        matched : bool
            Whether to use matched or unmatched data
            
        Returns:
        --------
        pd.DataFrame
            Standardized differences
        """
        if matched:
            treated_data = self.matched_treated
            control_data = self.matched_control
        else:
            treated_data = self.treated
            control_data = self.control
        
        results = []
        
        for covariate in self.covariates:
            # Calculate means
            treated_mean = treated_data[covariate].mean()
            control_mean = control_data[covariate].mean()
            
            # Calculate standard deviations
            treated_std = treated_data[covariate].std()
            control_std = control_data[covariate].std()
            
            # Standardized difference
            # Following Imbens & Wooldridge (2009, p. 33)
            pooled_std = np.sqrt((treated_std**2 + control_std**2) / 2)
            std_diff = (treated_mean - control_mean) / pooled_std
            
            # Variance ratio
            var_ratio = treated_std**2 / control_std**2 if control_std**2 > 0 else np.inf
            
            results.append({
                'Covariate': covariate,
                'Treated_Mean': treated_mean,
                'Control_Mean': control_mean,
                'Treated_Std': treated_std,
                'Control_Std': control_std,
                'Std_Diff': std_diff,
                'Var_Ratio': var_ratio,
                'Balanced': abs(std_diff) < 0.25  # Rule of thumb
            })
        
        return pd.DataFrame(results)
    
    def t_tests(self, matched: bool = True) -> pd.DataFrame:
        """
        T-tests for equality of means
        
        Parameters:
        -----------
        matched : bool
            Whether to use matched or unmatched data
            
        Returns:
        --------
        pd.DataFrame
            T-test results
        """
        if matched:
            treated_data = self.matched_treated
            control_data = self.matched_control
        else:
            treated_data = self.treated
            control_data = self.control
        
        results = []
        
        for covariate in self.covariates:
            treated_values = treated_data[covariate].values
            control_values = control_data[covariate].values
            
            # T-test for equality of means
            t_stat, p_value = stats.ttest_ind(treated_values, control_values)
            
            # Levene's test for equality of variances
            levene_stat, levene_p = stats.levene(treated_values, control_values)
            
            results.append({
                'Covariate': covariate,
                'T_statistic': t_stat,
                'P_value': p_value,
                'Levene_statistic': levene_stat,
                'Levene_p_value': levene_p,
                'Means_equal': p_value > 0.05,
                'Variances_equal': levene_p > 0.05
            })
        
        return pd.DataFrame(results)
    
    def kolmogorov_smirnov_tests(self, matched: bool = True) -> pd.DataFrame:
        """
        Kolmogorov-Smirnov tests for distributional equality
        
        Parameters:
        -----------
        matched : bool
            Whether to use matched or unmatched data
            
        Returns:
        --------
        pd.DataFrame
            KS test results
        """
        if matched:
            treated_data = self.matched_treated
            control_data = self.matched_control
        else:
            treated_data = self.treated
            control_data = self.control
        
        results = []
        
        for covariate in self.covariates:
            treated_values = treated_data[covariate].values
            control_values = control_data[covariate].values
            
            # KS test
            ks_stat, ks_p = stats.ks_2samp(treated_values, control_values)
            
            # Anderson-Darling test (more sensitive to tail differences)
            try:
                from scipy.stats import anderson_ksamp
                ad_stat, ad_critical, ad_p = anderson_ksamp([treated_values, control_values])
            except:
                ad_stat, ad_p = np.nan, np.nan
            
            results.append({
                'Covariate': covariate,
                'KS_statistic': ks_stat,
                'KS_p_value': ks_p,
                'AD_statistic': ad_stat,
                'AD_p_value': ad_p,
                'Distributions_equal': ks_p > 0.05
            })
        
        return pd.DataFrame(results)
    
    def overall_balance_test(self, matched: bool = True) -> Dict:
        """
        Overall balance test using Hotelling's T-squared test
        
        Parameters:
        -----------
        matched : bool
            Whether to use matched or unmatched data
            
        Returns:
        --------
        dict
            Overall balance test results
        """
        if matched:
            treated_data = self.matched_treated
            control_data = self.matched_control
        else:
            treated_data = self.treated
            control_data = self.control
        
        # Prepare data matrices
        X_treated = treated_data[self.covariates].values
        X_control = control_data[self.covariates].values
        
        # Calculate means
        mean_treated = np.mean(X_treated, axis=0)
        mean_control = np.mean(X_control, axis=0)
        
        # Calculate pooled covariance matrix
        n_treated = len(X_treated)
        n_control = len(X_control)
        
        cov_treated = np.cov(X_treated.T)
        cov_control = np.cov(X_control.T)
        
        pooled_cov = ((n_treated - 1) * cov_treated + (n_control - 1) * cov_control) / (n_treated + n_control - 2)
        
        # Hotelling's T-squared test
        try:
            pooled_cov_inv = np.linalg.inv(pooled_cov)
            diff = mean_treated - mean_control
            
            t_squared = (n_treated * n_control / (n_treated + n_control)) * diff.T @ pooled_cov_inv @ diff
            
            # Convert to F-statistic
            p = len(self.covariates)
            n = n_treated + n_control
            f_stat = (n - p - 1) / (p * (n - 2)) * t_squared
            
            # P-value
            p_value = 1 - stats.f.cdf(f_stat, p, n - p - 1)
            
        except np.linalg.LinAlgError:
            t_squared = np.nan
            f_stat = np.nan
            p_value = np.nan
        
        return {
            't_squared': t_squared,
            'f_statistic': f_stat,
            'p_value': p_value,
            'n_treated': n_treated,
            'n_control': n_control,
            'n_covariates': len(self.covariates),
            'balanced': p_value > 0.05 if not np.isnan(p_value) else False
        }
    
    def balance_improvement(self) -> pd.DataFrame:
        """
        Compare balance before and after matching
        
        Returns:
        --------
        pd.DataFrame
            Balance improvement results
        """
        if self.matched_indices is None:
            raise ValueError("Matched indices required for balance improvement analysis")
        
        # Before matching
        before_balance = self.standardized_differences(matched=False)
        before_balance['Period'] = 'Before'
        
        # After matching
        after_balance = self.standardized_differences(matched=True)
        after_balance['Period'] = 'After'
        
        # Combine results
        combined = pd.concat([before_balance, after_balance], ignore_index=True)
        
        # Calculate improvement
        improvement = []
        for covariate in self.covariates:
            before_std_diff = before_balance[before_balance['Covariate'] == covariate]['Std_Diff'].iloc[0]
            after_std_diff = after_balance[after_balance['Covariate'] == covariate]['Std_Diff'].iloc[0]
            
            improvement.append({
                'Covariate': covariate,
                'Before_Std_Diff': before_std_diff,
                'After_Std_Diff': after_std_diff,
                'Improvement': abs(before_std_diff) - abs(after_std_diff),
                'Percent_Improvement': (abs(before_std_diff) - abs(after_std_diff)) / abs(before_std_diff) * 100 if abs(before_std_diff) > 0 else 0
            })
        
        return pd.DataFrame(improvement)
    
    def plot_balance(self, matched: bool = True, title: str = "Balance Assessment") -> None:
        """
        Plot balance assessment
        
        Parameters:
        -----------
        matched : bool
            Whether to use matched or unmatched data
        title : str
            Plot title
        """
        import matplotlib.pyplot as plt
        
        # Get balance results
        balance_results = self.standardized_differences(matched=matched)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Standardized differences
        covariates = balance_results['Covariate']
        std_diffs = balance_results['Std_Diff']
        
        colors = ['red' if abs(diff) > 0.25 else 'steelblue' for diff in std_diffs]
        
        ax1.barh(covariates, std_diffs, color=colors, alpha=0.7)
        ax1.axvline(x=0.25, color='red', linestyle='--', alpha=0.5, label='±0.25 threshold')
        ax1.axvline(x=-0.25, color='red', linestyle='--', alpha=0.5)
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax1.set_xlabel('Standardized Difference')
        ax1.set_title('Standardized Differences')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Variance ratios
        var_ratios = balance_results['Var_Ratio']
        var_ratios = np.minimum(var_ratios, 4)  # Cap at 4 for visualization
        
        ax2.barh(covariates, var_ratios, color='green', alpha=0.7)
        ax2.axvline(x=1, color='black', linestyle='-', alpha=0.5, label='Perfect balance')
        ax2.axvline(x=0.8, color='red', linestyle='--', alpha=0.5, label='±0.2 threshold')
        ax2.axvline(x=1.2, color='red', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Variance Ratio')
        ax2.set_title('Variance Ratios')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def summary(self, matched: bool = True) -> Dict:
        """
        Summary of balance assessment
        
        Parameters:
        -----------
        matched : bool
            Whether to use matched or unmatched data
            
        Returns:
        --------
        dict
            Balance summary
        """
        # Standardized differences
        std_diffs = self.standardized_differences(matched=matched)
        
        # Overall balance test
        overall_test = self.overall_balance_test(matched=matched)
        
        # Count balanced covariates
        n_balanced = sum(std_diffs['Balanced'])
        n_covariates = len(self.covariates)
        
        # Average standardized difference
        avg_std_diff = np.mean(np.abs(std_diffs['Std_Diff']))
        
        return {
            'n_covariates': n_covariates,
            'n_balanced': n_balanced,
            'balance_ratio': n_balanced / n_covariates,
            'avg_std_diff': avg_std_diff,
            'overall_balanced': overall_test['balanced'],
            'overall_p_value': overall_test['p_value'],
            'n_treated': overall_test['n_treated'],
            'n_control': overall_test['n_control']
        }