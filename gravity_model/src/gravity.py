"""
Gravity Model for International Trade
Based on Tinbergen (1962) "Shaping the World Economy" and Anderson & van Wincoop (2003)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Union
from scipy import optimize, stats
import warnings

class GravityModel:
    """
    Gravity model for international trade analysis
    
    Implements:
    - Basic gravity model
    - Anderson-van Wincoop model
    - Trade cost analysis
    - Counterfactual analysis
    """
    
    def __init__(self, data: pd.DataFrame, 
                 trade_var: str = 'trade',
                 gdp_i_var: str = 'gdp_i',
                 gdp_j_var: str = 'gdp_j',
                 distance_var: str = 'distance',
                 border_var: str = 'border',
                 language_var: str = 'language'):
        """
        Initialize gravity model
        
        Parameters:
        -----------
        data : pd.DataFrame
            Trade data
        trade_var : str
            Trade flow variable
        gdp_i_var : str
            GDP of origin country
        gdp_j_var : str
            GDP of destination country
        distance_var : str
            Distance between countries
        border_var : str
            Border dummy variable
        language_var : str
            Common language dummy variable
        """
        self.data = data.copy()
        self.trade_var = trade_var
        self.gdp_i_var = gdp_i_var
        self.gdp_j_var = gdp_j_var
        self.distance_var = distance_var
        self.border_var = border_var
        self.language_var = language_var
        
        # Prepare data
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare data for gravity model estimation"""
        # Create log variables
        self.data['log_trade'] = np.log(self.data[self.trade_var] + 1)
        self.data['log_gdp_i'] = np.log(self.data[self.gdp_i_var])
        self.data['log_gdp_j'] = np.log(self.data[self.gdp_j_var])
        self.data['log_distance'] = np.log(self.data[self.distance_var])
        
        # Create interaction terms
        self.data['log_gdp_product'] = self.data['log_gdp_i'] + self.data['log_gdp_j']
        
        # Create country dummies
        self.countries = sorted(list(set(self.data['country_i'].unique()) | 
                                   set(self.data['country_j'].unique())))
        
        # Create country dummy variables
        for country in self.countries:
            self.data[f'dummy_i_{country}'] = (self.data['country_i'] == country).astype(int)
            self.data[f'dummy_j_{country}'] = (self.data['country_j'] == country).astype(int)
    
    def basic_gravity(self, robust: bool = True) -> Dict:
        """
        Estimate basic gravity model
        
        log(T_ij) = α + β₁log(GDP_i) + β₂log(GDP_j) + β₃log(distance_ij) + 
                    β₄border_ij + β₅language_ij + ε_ij
        
        Parameters:
        -----------
        robust : bool
            Whether to use robust standard errors
            
        Returns:
        --------
        dict
            Estimation results
        """
        # Prepare variables
        y = self.data['log_trade'].values
        X = self.data[['log_gdp_i', 'log_gdp_j', 'log_distance', 
                      self.border_var, self.language_var]].values
        
        # Add constant
        X = np.column_stack([np.ones(len(X)), X])
        
        # Estimate OLS
        try:
            beta = np.linalg.solve(X.T @ X, X.T @ y)
            residuals = y - X @ beta
            
            # Calculate standard errors
            if robust:
                # White standard errors
                meat = X.T @ np.diag(residuals**2) @ X
                bread = np.linalg.inv(X.T @ X)
                vcov = bread @ meat @ bread
                se = np.sqrt(np.diag(vcov))
            else:
                # Standard OLS standard errors
                mse = np.sum(residuals**2) / (len(y) - X.shape[1])
                vcov = mse * np.linalg.inv(X.T @ X)
                se = np.sqrt(np.diag(vcov))
            
            # Calculate t-statistics and p-values
            t_stats = beta / se
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), len(y) - X.shape[1]))
            
            # R-squared
            ssr = np.sum(residuals**2)
            sst = np.sum((y - np.mean(y))**2)
            r_squared = 1 - ssr / sst
            
        except np.linalg.LinAlgError:
            beta = np.full(X.shape[1], np.nan)
            se = np.full(X.shape[1], np.nan)
            t_stats = np.full(X.shape[1], np.nan)
            p_values = np.full(X.shape[1], np.nan)
            r_squared = np.nan
        
        return {
            'coefficients': beta,
            'standard_errors': se,
            't_statistics': t_stats,
            'p_values': p_values,
            'r_squared': r_squared,
            'n_obs': len(y),
            'variable_names': ['constant', 'log_gdp_i', 'log_gdp_j', 'log_distance', 
                              self.border_var, self.language_var]
        }
    
    def anderson_van_wincoop(self, max_iter: int = 1000, 
                           tolerance: float = 1e-6) -> Dict:
        """
        Estimate Anderson-van Wincoop model
        
        log(T_ij) = α + β₁log(GDP_i) + β₂log(GDP_j) + β₃log(distance_ij) + 
                    β₄border_ij + β₅language_ij + γ_i + γ_j + ε_ij
        
        Parameters:
        -----------
        max_iter : int
            Maximum iterations
        tolerance : float
            Convergence tolerance
            
        Returns:
        --------
        dict
            Estimation results
        """
        # Prepare variables
        y = self.data['log_trade'].values
        
        # Create design matrix with country dummies
        X_vars = ['log_gdp_i', 'log_gdp_j', 'log_distance', self.border_var, self.language_var]
        X = self.data[X_vars].values
        
        # Add country dummies
        dummy_cols = [col for col in self.data.columns if col.startswith('dummy_i_') or col.startswith('dummy_j_')]
        X_dummies = self.data[dummy_cols].values
        
        # Combine variables
        X = np.column_stack([X, X_dummies])
        
        # Add constant
        X = np.column_stack([np.ones(len(X)), X])
        
        # Estimate OLS
        try:
            beta = np.linalg.solve(X.T @ X, X.T @ y)
            residuals = y - X @ beta
            
            # Calculate standard errors
            meat = X.T @ np.diag(residuals**2) @ X
            bread = np.linalg.inv(X.T @ X)
            vcov = bread @ meat @ bread
            se = np.sqrt(np.diag(vcov))
            
            # Calculate t-statistics and p-values
            t_stats = beta / se
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), len(y) - X.shape[1]))
            
            # R-squared
            ssr = np.sum(residuals**2)
            sst = np.sum((y - np.mean(y))**2)
            r_squared = 1 - ssr / sst
            
        except np.linalg.LinAlgError:
            beta = np.full(X.shape[1], np.nan)
            se = np.full(X.shape[1], np.nan)
            t_stats = np.full(X.shape[1], np.nan)
            p_values = np.full(X.shape[1], np.nan)
            r_squared = np.nan
        
        return {
            'coefficients': beta,
            'standard_errors': se,
            't_statistics': t_stats,
            'p_values': p_values,
            'r_squared': r_squared,
            'n_obs': len(y),
            'variable_names': ['constant'] + X_vars + dummy_cols
        }
    
    def trade_cost_analysis(self, distance_elasticity: float = -1.0) -> Dict:
        """
        Analyze trade costs
        
        Parameters:
        -----------
        distance_elasticity : float
            Distance elasticity of trade
            
        Returns:
        --------
        dict
            Trade cost analysis results
        """
        # Calculate trade costs
        self.data['trade_cost'] = np.exp(-distance_elasticity * self.data['log_distance'])
        
        # Calculate trade cost index
        trade_cost_index = self.data.groupby(['country_i', 'country_j'])['trade_cost'].mean()
        
        # Calculate average trade costs by country
        avg_trade_costs = {}
        for country in self.countries:
            country_data = self.data[(self.data['country_i'] == country) | 
                                   (self.data['country_j'] == country)]
            avg_trade_costs[country] = country_data['trade_cost'].mean()
        
        return {
            'trade_cost_index': trade_cost_index,
            'average_trade_costs': avg_trade_costs,
            'distance_elasticity': distance_elasticity
        }
    
    def counterfactual_analysis(self, scenario: str, 
                               parameter_change: float) -> Dict:
        """
        Counterfactual analysis
        
        Parameters:
        -----------
        scenario : str
            Scenario type ('distance', 'border', 'language')
        parameter_change : float
            Parameter change
            
        Returns:
        --------
        dict
            Counterfactual analysis results
        """
        # Store original data
        original_data = self.data.copy()
        
        # Apply scenario
        if scenario == 'distance':
            self.data['log_distance'] += parameter_change
        elif scenario == 'border':
            self.data[self.border_var] = (self.data[self.border_var] + parameter_change).clip(0, 1)
        elif scenario == 'language':
            self.data[self.language_var] = (self.data[self.language_var] + parameter_change).clip(0, 1)
        
        # Recalculate trade flows
        self.data['log_trade_counterfactual'] = (
            self.data['log_gdp_i'] + self.data['log_gdp_j'] + 
            self.data['log_distance'] + 
            self.data[self.border_var] + 
            self.data[self.language_var]
        )
        
        # Calculate changes
        trade_change = self.data['log_trade_counterfactual'] - self.data['log_trade']
        total_trade_change = np.sum(np.exp(self.data['log_trade_counterfactual']) - 
                                  np.exp(self.data['log_trade']))
        
        # Restore original data
        self.data = original_data
        
        return {
            'scenario': scenario,
            'parameter_change': parameter_change,
            'trade_change': trade_change,
            'total_trade_change': total_trade_change,
            'percentage_change': (total_trade_change / np.sum(np.exp(self.data['log_trade']))) * 100
        }
    
    def trade_intensity_index(self) -> pd.DataFrame:
        """
        Calculate trade intensity index
        
        Returns:
        --------
        pd.DataFrame
            Trade intensity index
        """
        # Calculate bilateral trade intensity
        self.data['trade_intensity'] = (
            self.data[self.trade_var] / 
            (self.data[self.gdp_i_var] * self.data[self.gdp_j_var])
        )
        
        # Calculate average trade intensity by country pair
        intensity_by_pair = self.data.groupby(['country_i', 'country_j'])['trade_intensity'].mean()
        
        # Calculate average trade intensity by country
        intensity_by_country = {}
        for country in self.countries:
            country_data = self.data[(self.data['country_i'] == country) | 
                                   (self.data['country_j'] == country)]
            intensity_by_country[country] = country_data['trade_intensity'].mean()
        
        return pd.DataFrame({
            'country': list(intensity_by_country.keys()),
            'trade_intensity': list(intensity_by_country.values())
        })
    
    def summary(self) -> Dict:
        """Return model summary"""
        return {
            'n_obs': len(self.data),
            'n_countries': len(self.countries),
            'countries': self.countries,
            'variables': {
                'trade': self.trade_var,
                'gdp_i': self.gdp_i_var,
                'gdp_j': self.gdp_j_var,
                'distance': self.distance_var,
                'border': self.border_var,
                'language': self.language_var
            }
        }