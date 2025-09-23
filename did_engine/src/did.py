"""
Difference-in-Differences (DID) Estimation with Robust Inference
Based on Angrist & Pischke (2009) "Mostly Harmless Econometrics"
and Callaway & Sant'Anna (2021) "Difference-in-Differences with Multiple Time Periods"
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Union
from scipy import stats
import warnings
from .plot import plot_did_results, plot_parallel_trends

class DifferenceInDifferences:
    """
    Difference-in-Differences estimator with robust inference
    
    Implements the standard DID estimator with various robustness checks
    and extensions for staggered treatment adoption.
    """
    
    def __init__(self, data: pd.DataFrame, y_var: str, treatment_var: str,
                 time_var: str, entity_var: str, 
                 treatment_time: Optional[int] = None):
        """
        Initialize DID analysis
        
        Parameters:
        -----------
        data : pd.DataFrame
            Panel dataset
        y_var : str
            Outcome variable name
        treatment_var : str
            Treatment indicator (0/1)
        time_var : str
            Time variable name
        entity_var : str
            Entity identifier
        treatment_time : int, optional
            Time when treatment starts (for single treatment time)
        """
        self.data = data.copy()
        self.y_var = y_var
        self.treatment_var = treatment_var
        self.time_var = time_var
        self.entity_var = entity_var
        self.treatment_time = treatment_time
        
        # Sort data
        self.data = self.data.sort_values([entity_var, time_var]).reset_index(drop=True)
        
        # Get dimensions
        self.n_entities = self.data[entity_var].nunique()
        self.n_periods = self.data[time_var].nunique()
        self.n_obs = len(self.data)
        
        # Create treatment indicators
        self._create_treatment_indicators()
    
    def _create_treatment_indicators(self):
        """Create treatment indicators for DID analysis"""
        # Post-treatment indicator
        if self.treatment_time is not None:
            self.data['post'] = (self.data[self.time_var] >= self.treatment_time).astype(int)
        else:
            # For staggered treatment, create post indicator for each entity
            self.data['post'] = 0
            for entity in self.data[self.entity_var].unique():
                entity_data = self.data[self.data[self.entity_var] == entity]
                if entity_data[self.treatment_var].sum() > 0:
                    first_treated = entity_data[entity_data[self.treatment_var] == 1][self.time_var].min()
                    self.data.loc[(self.data[self.entity_var] == entity) & 
                                 (self.data[self.time_var] >= first_treated), 'post'] = 1
        
        # Interaction term
        self.data['treated_post'] = self.data[self.treatment_var] * self.data['post']
    
    def estimate_did(self, covariates: Optional[List[str]] = None,
                    robust: bool = True, cluster: str = 'entity') -> Dict:
        """
        Estimate the basic DID model
        
        The standard DID model is:
        Y_it = α + β₁*Treated_i + β₂*Post_t + β₃*(Treated_i × Post_t) + γ*X_it + ε_it
        
        Parameters:
        -----------
        covariates : list, optional
            Control variables
        robust : bool
            Whether to use robust standard errors
        cluster : str
            Clustering variable ('entity', 'time', or 'both')
            
        Returns:
        --------
        dict
            Estimation results
        """
        # Prepare data
        y = self.data[self.y_var].values
        treated = self.data[self.treatment_var].values
        post = self.data['post'].values
        treated_post = self.data['treated_post'].values
        
        # Create design matrix
        X = np.column_stack([treated, post, treated_post])
        var_names = ['treated', 'post', 'treated_post']
        
        # Add covariates if specified
        if covariates is not None:
            X_cov = self.data[covariates].values
            X = np.column_stack([X, X_cov])
            var_names.extend(covariates)
        
        # Add entity and time fixed effects
        entity_dummies = pd.get_dummies(self.data[self.entity_var], prefix='entity')
        time_dummies = pd.get_dummies(self.data[self.time_var], prefix='time')
        
        # Drop one category to avoid multicollinearity
        entity_dummies = entity_dummies.iloc[:, :-1]
        time_dummies = time_dummies.iloc[:, :-1]
        
        X = np.column_stack([X, entity_dummies.values, time_dummies.values])
        var_names.extend([f'entity_{i}' for i in range(entity_dummies.shape[1])])
        var_names.extend([f'time_{i}' for i in range(time_dummies.shape[1])])
        
        # Fit regression
        from ..ols_iv_panel.src.ols import OLS
        
        model = OLS(y, X, var_names)
        model.fit(robust=robust)
        
        # Extract DID coefficient
        did_coef = model.coefficients[2]  # treated_post coefficient
        did_se = model.se[2]
        did_tstat = model.t_stats[2]
        did_pval = model.p_values[2]
        
        return {
            'model': model,
            'did_coefficient': did_coef,
            'did_std_error': did_se,
            'did_t_statistic': did_tstat,
            'did_p_value': did_pval,
            'n_treated': self.data[self.treatment_var].sum(),
            'n_control': self.n_obs - self.data[self.treatment_var].sum(),
            'n_periods': self.n_periods
        }
    
    def event_study(self, covariates: Optional[List[str]] = None,
                   relative_time_range: Tuple[int, int] = (-5, 5)) -> Dict:
        """
        Event study analysis with dynamic treatment effects
        
        Estimates the model:
        Y_it = α + β₁*Treated_i + Σ_{k=-K}^{K} γ_k*(Treated_i × I(t = k)) + γ*X_it + ε_it
        
        Parameters:
        -----------
        covariates : list, optional
            Control variables
        relative_time_range : tuple
            Range of relative time periods to include
            
        Returns:
        --------
        dict
            Event study results
        """
        # Create relative time variable
        self.data['relative_time'] = 0
        
        for entity in self.data[self.entity_var].unique():
            entity_data = self.data[self.data[self.entity_var] == entity]
            if entity_data[self.treatment_var].sum() > 0:
                first_treated = entity_data[entity_data[self.treatment_var] == 1][self.time_var].min()
                mask = (self.data[self.entity_var] == entity)
                self.data.loc[mask, 'relative_time'] = (self.data.loc[mask, self.time_var] - first_treated)
        
        # Create relative time dummies
        relative_times = list(range(relative_time_range[0], relative_time_range[1] + 1))
        relative_times.remove(0)  # Exclude t=0 as reference period
        
        # Prepare data
        y = self.data[self.y_var].values
        treated = self.data[self.treatment_var].values
        
        # Create design matrix
        X = np.column_stack([treated])
        var_names = ['treated']
        
        # Add relative time interactions
        for k in relative_times:
            interaction = treated * (self.data['relative_time'] == k).astype(int)
            X = np.column_stack([X, interaction])
            var_names.append(f'treated_x_t{k}')
        
        # Add covariates
        if covariates is not None:
            X_cov = self.data[covariates].values
            X = np.column_stack([X, X_cov])
            var_names.extend(covariates)
        
        # Add fixed effects
        entity_dummies = pd.get_dummies(self.data[self.entity_var], prefix='entity')
        time_dummies = pd.get_dummies(self.data[self.time_var], prefix='time')
        
        entity_dummies = entity_dummies.iloc[:, :-1]
        time_dummies = time_dummies.iloc[:, :-1]
        
        X = np.column_stack([X, entity_dummies.values, time_dummies.values])
        var_names.extend([f'entity_{i}' for i in range(entity_dummies.shape[1])])
        var_names.extend([f'time_{i}' for i in range(time_dummies.shape[1])])
        
        # Fit regression
        from ..ols_iv_panel.src.ols import OLS
        
        model = OLS(y, X, var_names)
        model.fit(robust=True)
        
        # Extract event study coefficients
        event_coefs = {}
        event_ses = {}
        event_tstats = {}
        event_pvals = {}
        
        for i, k in enumerate(relative_times):
            coef_idx = 1 + i  # Skip treated coefficient
            event_coefs[k] = model.coefficients[coef_idx]
            event_ses[k] = model.se[coef_idx]
            event_tstats[k] = model.t_stats[coef_idx]
            event_pvals[k] = model.p_values[coef_idx]
        
        return {
            'model': model,
            'event_coefficients': event_coefs,
            'event_std_errors': event_ses,
            'event_t_statistics': event_tstats,
            'event_p_values': event_pvals,
            'relative_times': relative_times,
            'reference_period': 0
        }
    
    def parallel_trends_test(self, pre_periods: int = 3) -> Dict:
        """
        Test for parallel trends assumption
        
        Tests whether treatment and control groups have parallel trends
        in the pre-treatment period.
        
        Parameters:
        -----------
        pre_periods : int
            Number of pre-treatment periods to test
            
        Returns:
        --------
        dict
            Parallel trends test results
        """
        # Get pre-treatment data
        pre_data = self.data[self.data['post'] == 0].copy()
        
        if len(pre_data) == 0:
            return {'test_statistic': np.nan, 'p_value': np.nan, 'reject_parallel_trends': False}
        
        # Create time trend interaction
        pre_data['treated_trend'] = pre_data[self.treatment_var] * pre_data[self.time_var]
        
        # Regression: Y_it = α + β₁*Treated_i + β₂*Time_t + β₃*(Treated_i × Time_t) + ε_it
        y = pre_data[self.y_var].values
        treated = pre_data[self.treatment_var].values
        time = pre_data[self.time_var].values
        treated_trend = pre_data['treated_trend'].values
        
        X = np.column_stack([treated, time, treated_trend])
        
        from ..ols_iv_panel.src.ols import OLS
        
        model = OLS(y, X, ['treated', 'time', 'treated_trend'])
        model.fit(robust=True)
        
        # Test H0: β₃ = 0 (no differential trend)
        trend_coef = model.coefficients[2]
        trend_se = model.se[2]
        trend_tstat = model.t_stats[2]
        trend_pval = model.p_values[2]
        
        return {
            'test_statistic': trend_tstat,
            'p_value': trend_pval,
            'coefficient': trend_coef,
            'std_error': trend_se,
            'reject_parallel_trends': trend_pval < 0.05,
            'n_pre_periods': len(pre_data)
        }
    
    def placebo_test(self, placebo_periods: List[int]) -> Dict:
        """
        Placebo test using fake treatment periods
        
        Tests whether the treatment effect is significant when applied
        to periods when no treatment actually occurred.
        
        Parameters:
        -----------
        placebo_periods : list
            List of periods to use as fake treatment periods
            
        Returns:
        --------
        dict
            Placebo test results
        """
        results = {}
        
        for placebo_period in placebo_periods:
            # Create fake treatment data
            fake_data = self.data.copy()
            fake_data['fake_treated'] = 0
            fake_data['fake_post'] = (fake_data[self.time_var] >= placebo_period).astype(int)
            fake_data['fake_treated_post'] = fake_data['fake_treated'] * fake_data['fake_post']
            
            # Estimate fake DID
            y = fake_data[self.y_var].values
            treated = fake_data['fake_treated'].values
            post = fake_data['fake_post'].values
            treated_post = fake_data['fake_treated_post'].values
            
            X = np.column_stack([treated, post, treated_post])
            
            from ..ols_iv_panel.src.ols import OLS
            
            model = OLS(y, X, ['treated', 'post', 'treated_post'])
            model.fit(robust=True)
            
            results[placebo_period] = {
                'coefficient': model.coefficients[2],
                'std_error': model.se[2],
                't_statistic': model.t_stats[2],
                'p_value': model.p_values[2]
            }
        
        return results
    
    def summary(self) -> pd.DataFrame:
        """Return summary of DID results"""
        did_results = self.estimate_did()
        
        summary_data = {
            'Method': ['DID'],
            'Coefficient': [did_results['did_coefficient']],
            'Std Error': [did_results['did_std_error']],
            't-statistic': [did_results['did_t_statistic']],
            'P-value': [did_results['did_p_value']],
            'N Treated': [did_results['n_treated']],
            'N Control': [did_results['n_control']],
            'N Periods': [did_results['n_periods']]
        }
        
        return pd.DataFrame(summary_data)

def staggered_did(data: pd.DataFrame, y_var: str, treatment_var: str,
                  time_var: str, entity_var: str) -> Dict:
    """
    Staggered treatment DID analysis
    
    Handles cases where different entities receive treatment at different times.
    Based on Callaway & Sant'Anna (2021).
    
    Parameters:
    -----------
    data : pd.DataFrame
        Panel dataset
    y_var : str
        Outcome variable
    treatment_var : str
        Treatment indicator
    time_var : str
        Time variable
    entity_var : str
        Entity identifier
        
    Returns:
    --------
    dict
        Staggered DID results
    """
    # Identify treatment groups and timing
    treatment_groups = data[data[treatment_var] == 1].groupby(entity_var)[time_var].min()
    
    results = {}
    
    for entity, first_treated in treatment_groups.items():
        # Create entity-specific analysis
        entity_data = data[data[entity_var] == entity].copy()
        entity_data['post'] = (entity_data[time_var] >= first_treated).astype(int)
        entity_data['treated_post'] = entity_data[treatment_var] * entity_data['post']
        
        # Run DID for this entity
        did_analysis = DifferenceInDifferences(
            entity_data, y_var, treatment_var, time_var, entity_var, first_treated
        )
        
        results[entity] = did_analysis.estimate_did()
    
    return results