"""
Event Study Analysis with Dynamic Treatment Effects
Based on Jacobson, LaLonde, and Sullivan (1993) and Autor (2003)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict
from scipy import stats
import warnings

class EventStudy:
    """
    Event study analysis for treatment effects with dynamic specifications
    
    Implements various event study methodologies including:
    - Standard event study with relative time indicators
    - Sun and Abraham (2021) estimator for staggered treatment
    - Callaway and Sant'Anna (2021) estimator
    """
    
    def __init__(self, data: pd.DataFrame, y_var: str, treatment_var: str,
                 time_var: str, entity_var: str, 
                 treatment_time: Optional[int] = None):
        """
        Initialize event study analysis
        
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
            Time when treatment starts
        """
        self.data = data.copy()
        self.y_var = y_var
        self.treatment_var = treatment_var
        self.time_var = time_var
        self.entity_var = entity_var
        self.treatment_time = treatment_time
        
        # Sort data
        self.data = self.data.sort_values([entity_var, time_var]).reset_index(drop=True)
        
        # Create relative time variable
        self._create_relative_time()
    
    def _create_relative_time(self):
        """Create relative time variable for event study"""
        self.data['relative_time'] = 0
        
        for entity in self.data[self.entity_var].unique():
            entity_data = self.data[self.data[self.entity_var] == entity]
            if entity_data[self.treatment_var].sum() > 0:
                first_treated = entity_data[entity_data[self.treatment_var] == 1][self.time_var].min()
                mask = (self.data[self.entity_var] == entity)
                self.data.loc[mask, 'relative_time'] = (self.data.loc[mask, self.time_var] - first_treated)
    
    def standard_event_study(self, relative_time_range: Tuple[int, int] = (-5, 5),
                            covariates: Optional[List[str]] = None,
                            reference_period: int = -1) -> Dict:
        """
        Standard event study with relative time indicators
        
        Estimates the model:
        Y_it = α + β₁*Treated_i + Σ_{k≠ref} γ_k*(Treated_i × I(t = k)) + γ*X_it + ε_it
        
        Parameters:
        -----------
        relative_time_range : tuple
            Range of relative time periods
        covariates : list, optional
            Control variables
        reference_period : int
            Reference period (typically -1 or 0)
            
        Returns:
        --------
        dict
            Event study results
        """
        # Create relative time dummies
        relative_times = list(range(relative_time_range[0], relative_time_range[1] + 1))
        if reference_period in relative_times:
            relative_times.remove(reference_period)
        
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
            'reference_period': reference_period,
            'n_obs': len(y)
        }
    
    def sun_abraham_estimator(self, relative_time_range: Tuple[int, int] = (-5, 5),
                            covariates: Optional[List[str]] = None) -> Dict:
        """
        Sun and Abraham (2021) estimator for staggered treatment
        
        Addresses the problem of negative weighting in standard event studies
        with staggered treatment adoption.
        
        Parameters:
        -----------
        relative_time_range : tuple
            Range of relative time periods
        covariates : list, optional
            Control variables
            
        Returns:
        --------
        dict
            Sun-Abraham estimator results
        """
        # Identify treatment cohorts
        treatment_cohorts = self.data[self.data[self.treatment_var] == 1].groupby(self.entity_var)[self.time_var].min()
        
        # Create cohort-specific relative time variables
        for cohort_time in treatment_cohorts.unique():
            cohort_entities = treatment_cohorts[treatment_cohorts == cohort_time].index
            mask = self.data[self.entity_var].isin(cohort_entities)
            self.data.loc[mask, 'cohort_relative_time'] = self.data.loc[mask, self.time_var] - cohort_time
        
        # Create cohort dummies
        cohort_dummies = pd.get_dummies(self.data['cohort_relative_time'], prefix='cohort_t')
        
        # Prepare data
        y = self.data[self.y_var].values
        treated = self.data[self.treatment_var].values
        
        # Create design matrix
        X = np.column_stack([treated])
        var_names = ['treated']
        
        # Add cohort interactions
        for col in cohort_dummies.columns:
            if col != 'cohort_t_0':  # Exclude reference period
                X = np.column_stack([X, cohort_dummies[col].values])
                var_names.append(col)
        
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
        
        return {
            'model': model,
            'cohort_effects': treatment_cohorts,
            'n_cohorts': len(treatment_cohorts.unique()),
            'n_obs': len(y)
        }
    
    def callaway_santanna_estimator(self, relative_time_range: Tuple[int, int] = (-5, 5),
                                   never_treated: bool = True) -> Dict:
        """
        Callaway and Sant'Anna (2021) estimator
        
        Uses never-treated or not-yet-treated units as control groups
        to address the negative weighting problem.
        
        Parameters:
        -----------
        relative_time_range : tuple
            Range of relative time periods
        never_treated : bool
            Whether to use never-treated units as controls
            
        Returns:
        --------
        dict
            Callaway-Sant'Anna estimator results
        """
        # Identify treatment groups
        treatment_groups = self.data[self.data[self.treatment_var] == 1].groupby(self.entity_var)[self.time_var].min()
        
        # Create control groups
        if never_treated:
            control_entities = self.data[self.data[self.treatment_var] == 0][self.entity_var].unique()
        else:
            # Not-yet-treated units
            control_entities = []
            for entity in self.data[self.entity_var].unique():
                entity_data = self.data[self.data[self.entity_var] == entity]
                if entity_data[self.treatment_var].sum() == 0:
                    control_entities.append(entity)
        
        # Calculate group-time average treatment effects
        group_time_effects = {}
        
        for entity, first_treated in treatment_groups.items():
            for t in range(first_treated + relative_time_range[0], 
                          first_treated + relative_time_range[1] + 1):
                if t in self.data[self.time_var].values:
                    # Treatment group
                    treated_data = self.data[
                        (self.data[self.entity_var] == entity) & 
                        (self.data[self.time_var] == t)
                    ]
                    
                    # Control group
                    control_data = self.data[
                        (self.data[self.entity_var].isin(control_entities)) & 
                        (self.data[self.time_var] == t)
                    ]
                    
                    if len(treated_data) > 0 and len(control_data) > 0:
                        # Calculate treatment effect
                        treated_outcome = treated_data[self.y_var].mean()
                        control_outcome = control_data[self.y_var].mean()
                        
                        # Baseline comparison (pre-treatment)
                        baseline_treated = self.data[
                            (self.data[self.entity_var] == entity) & 
                            (self.data[self.time_var] == first_treated - 1)
                        ][self.y_var].mean()
                        
                        baseline_control = self.data[
                            (self.data[self.entity_var].isin(control_entities)) & 
                            (self.data[self.time_var] == first_treated - 1)
                        ][self.y_var].mean()
                        
                        # DID estimate
                        did_estimate = (treated_outcome - control_outcome) - (baseline_treated - baseline_control)
                        
                        group_time_effects[(entity, t)] = {
                            'effect': did_estimate,
                            'treated_outcome': treated_outcome,
                            'control_outcome': control_outcome,
                            'relative_time': t - first_treated
                        }
        
        return {
            'group_time_effects': group_time_effects,
            'n_treatment_groups': len(treatment_groups),
            'n_control_entities': len(control_entities)
        }
    
    def plot_event_study(self, results: Dict, title: str = "Event Study Results") -> None:
        """
        Plot event study results
        
        Parameters:
        -----------
        results : dict
            Event study results
        title : str
            Plot title
        """
        import matplotlib.pyplot as plt
        
        relative_times = results['relative_times']
        coefficients = [results['event_coefficients'][t] for t in relative_times]
        std_errors = [results['event_std_errors'][t] for t in relative_times]
        
        # Create confidence intervals
        ci_lower = [c - 1.96 * se for c, se in zip(coefficients, std_errors)]
        ci_upper = [c + 1.96 * se for c, se in zip(coefficients, std_errors)]
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(relative_times, coefficients, 'o-', label='Treatment Effect')
        plt.fill_between(relative_times, ci_lower, ci_upper, alpha=0.3, label='95% CI')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Treatment Start')
        
        plt.xlabel('Relative Time')
        plt.ylabel('Treatment Effect')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def test_pre_trends(self, pre_periods: int = 3) -> Dict:
        """
        Test for pre-treatment trends
        
        Tests whether there are significant treatment effects in the
        pre-treatment period (should be zero under parallel trends).
        
        Parameters:
        -----------
        pre_periods : int
            Number of pre-treatment periods to test
            
        Returns:
        --------
        dict
            Pre-trends test results
        """
        # Get pre-treatment data
        pre_data = self.data[self.data['relative_time'] < 0].copy()
        
        if len(pre_data) == 0:
            return {'test_statistic': np.nan, 'p_value': np.nan, 'reject_no_pre_trends': False}
        
        # Test for joint significance of pre-treatment effects
        pre_periods_list = list(range(-pre_periods, 0))
        
        # Create dummy variables for pre-treatment periods
        pre_dummies = {}
        for period in pre_periods_list:
            pre_dummies[f'pre_{period}'] = (pre_data['relative_time'] == period).astype(int)
        
        # Regression with pre-treatment dummies
        y = pre_data[self.y_var].values
        treated = pre_data[self.treatment_var].values
        
        X = np.column_stack([treated])
        var_names = ['treated']
        
        for period in pre_periods_list:
            interaction = treated * pre_dummies[f'pre_{period}']
            X = np.column_stack([X, interaction])
            var_names.append(f'treated_x_pre{period}')
        
        # Add fixed effects
        entity_dummies = pd.get_dummies(pre_data[self.entity_var], prefix='entity')
        time_dummies = pd.get_dummies(pre_data[self.time_var], prefix='time')
        
        entity_dummies = entity_dummies.iloc[:, :-1]
        time_dummies = time_dummies.iloc[:, :-1]
        
        X = np.column_stack([X, entity_dummies.values, time_dummies.values])
        var_names.extend([f'entity_{i}' for i in range(entity_dummies.shape[1])])
        var_names.extend([f'time_{i}' for i in range(time_dummies.shape[1])])
        
        # Fit regression
        from ..ols_iv_panel.src.ols import OLS
        
        model = OLS(y, X, var_names)
        model.fit(robust=True)
        
        # Test joint significance of pre-treatment effects
        pre_coef_indices = [i for i, name in enumerate(var_names) if 'treated_x_pre' in name]
        pre_coefficients = model.coefficients[pre_coef_indices]
        pre_cov_matrix = np.array([[model.se[i] * model.se[j] for j in pre_coef_indices] 
                                  for i in pre_coef_indices])
        
        # F-test for joint significance
        if len(pre_coefficients) > 0:
            f_stat = (pre_coefficients.T @ np.linalg.inv(pre_cov_matrix) @ pre_coefficients) / len(pre_coefficients)
            p_value = 1 - stats.f.cdf(f_stat, len(pre_coefficients), len(y) - len(var_names))
        else:
            f_stat = np.nan
            p_value = np.nan
        
        return {
            'test_statistic': f_stat,
            'p_value': p_value,
            'pre_periods_tested': pre_periods_list,
            'reject_no_pre_trends': p_value < 0.05 if not np.isnan(p_value) else False
        }