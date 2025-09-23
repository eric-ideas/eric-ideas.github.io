"""
Shock Analysis for Solow Growth Model
Based on Romer (2019) "Advanced Macroeconomics" and Barro & Sala-i-Martin (2004)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Union
from scipy import optimize, integrate
import warnings

class SolowShocks:
    """
    Shock analysis for Solow growth model
    
    Implements various shock scenarios including:
    - Technology shocks
    - Savings rate shocks
    - Population growth shocks
    - Depreciation shocks
    """
    
    def __init__(self, solow_model):
        """
        Initialize shock analysis
        
        Parameters:
        -----------
        solow_model : SolowModel
            Solow model instance
        """
        self.model = solow_model
        self.original_params = {
            's': solow_model.s,
            'n': solow_model.n,
            'g': solow_model.g,
            'delta': solow_model.delta,
            'alpha': solow_model.alpha
        }
    
    def technology_shock(self, shock_magnitude: float, 
                        shock_duration: int = 10,
                        shock_type: str = 'permanent',
                        T: int = 100) -> Dict:
        """
        Analyze technology shock
        
        Parameters:
        -----------
        shock_magnitude : float
            Magnitude of technology shock
        shock_duration : int
            Duration of shock (for temporary shocks)
        shock_type : str
            Type of shock ('permanent', 'temporary', 'transitory')
        T : int
            Number of time periods for simulation
            
        Returns:
        --------
        dict
            Shock analysis results
        """
        # Store original parameters
        original_A0 = self.model.A0
        
        # Create shock path
        if shock_type == 'permanent':
            shock_path = np.ones(T) * shock_magnitude
        elif shock_type == 'temporary':
            shock_path = np.zeros(T)
            shock_path[:shock_duration] = shock_magnitude
        elif shock_type == 'transitory':
            # Exponential decay
            shock_path = shock_magnitude * np.exp(-0.1 * np.arange(T))
        else:
            raise ValueError("Unknown shock type")
        
        # Simulate with shock
        results = []
        
        for t in range(T):
            # Apply shock
            self.model.A0 = original_A0 * (1 + shock_path[t])
            
            # Recalculate steady state
            self.model._calculate_steady_state()
            
            # Simulate one period
            k_current = self.model.K0 / (self.model.A0 * self.model.L0)
            k_next = k_current + self.model.capital_accumulation(k_current)
            
            y_current = self.model.production_function(k_current)
            c_current = (1 - self.model.s) * y_current
            
            results.append({
                'time': t,
                'k': k_current,
                'y': y_current,
                'c': c_current,
                'A': self.model.A0,
                'shock': shock_path[t]
            })
            
            # Update capital for next period
            self.model.K0 = k_next * self.model.A0 * self.model.L0
        
        # Restore original parameters
        self.model.A0 = original_A0
        self.model._calculate_steady_state()
        
        return {
            'shock_results': pd.DataFrame(results),
            'shock_magnitude': shock_magnitude,
            'shock_duration': shock_duration,
            'shock_type': shock_type
        }
    
    def savings_rate_shock(self, new_s: float, T: int = 100) -> Dict:
        """
        Analyze savings rate shock
        
        Parameters:
        -----------
        new_s : float
            New savings rate
        T : int
            Number of time periods for simulation
            
        Returns:
        --------
        dict
            Shock analysis results
        """
        # Store original savings rate
        original_s = self.model.s
        
        # Calculate new steady state
        self.model.s = new_s
        self.model._calculate_steady_state()
        new_steady_state = {
            'k_star': self.model.k_star,
            'y_star': self.model.y_star,
            'c_star': self.model.c_star
        }
        
        # Simulate transition
        transition = self.model.simulate_transition(T)
        
        # Calculate welfare effects
        welfare_change = self._calculate_welfare_change(transition, new_steady_state)
        
        # Restore original savings rate
        self.model.s = original_s
        self.model._calculate_steady_state()
        
        return {
            'new_steady_state': new_steady_state,
            'transition_path': transition,
            'welfare_change': welfare_change,
            'savings_rate_change': new_s - original_s
        }
    
    def population_growth_shock(self, new_n: float, T: int = 100) -> Dict:
        """
        Analyze population growth shock
        
        Parameters:
        -----------
        new_n : float
            New population growth rate
        T : int
            Number of time periods for simulation
            
        Returns:
        --------
        dict
            Shock analysis results
        """
        # Store original population growth rate
        original_n = self.model.n
        
        # Calculate new steady state
        self.model.n = new_n
        self.model._calculate_steady_state()
        new_steady_state = {
            'k_star': self.model.k_star,
            'y_star': self.model.y_star,
            'c_star': self.model.c_star
        }
        
        # Simulate transition
        transition = self.model.simulate_transition(T)
        
        # Restore original population growth rate
        self.model.n = original_n
        self.model._calculate_steady_state()
        
        return {
            'new_steady_state': new_steady_state,
            'transition_path': transition,
            'population_growth_change': new_n - original_n
        }
    
    def depreciation_shock(self, new_delta: float, T: int = 100) -> Dict:
        """
        Analyze depreciation shock
        
        Parameters:
        -----------
        new_delta : float
            New depreciation rate
        T : int
            Number of time periods for simulation
            
        Returns:
        --------
        dict
            Shock analysis results
        """
        # Store original depreciation rate
        original_delta = self.model.delta
        
        # Calculate new steady state
        self.model.delta = new_delta
        self.model._calculate_steady_state()
        new_steady_state = {
            'k_star': self.model.k_star,
            'y_star': self.model.y_star,
            'c_star': self.model.c_star
        }
        
        # Simulate transition
        transition = self.model.simulate_transition(T)
        
        # Restore original depreciation rate
        self.model.delta = original_delta
        self.model._calculate_steady_state()
        
        return {
            'new_steady_state': new_steady_state,
            'transition_path': transition,
            'depreciation_change': new_delta - original_delta
        }
    
    def multiple_shocks(self, shock_sequence: List[Dict], T: int = 100) -> Dict:
        """
        Analyze sequence of multiple shocks
        
        Parameters:
        -----------
        shock_sequence : list
            List of shock dictionaries
        T : int
            Number of time periods for simulation
            
        Returns:
        --------
        dict
            Multiple shocks analysis results
        """
        results = []
        
        for t in range(T):
            # Apply shocks in sequence
            for shock in shock_sequence:
                if shock['start_time'] <= t <= shock['end_time']:
                    if shock['type'] == 'technology':
                        self.model.A0 *= (1 + shock['magnitude'])
                    elif shock['type'] == 'savings':
                        self.model.s = shock['magnitude']
                    elif shock['type'] == 'population':
                        self.model.n = shock['magnitude']
                    elif shock['type'] == 'depreciation':
                        self.model.delta = shock['magnitude']
            
            # Recalculate steady state
            self.model._calculate_steady_state()
            
            # Simulate one period
            k_current = self.model.K0 / (self.model.A0 * self.model.L0)
            k_next = k_current + self.model.capital_accumulation(k_current)
            
            y_current = self.model.production_function(k_current)
            c_current = (1 - self.model.s) * y_current
            
            results.append({
                'time': t,
                'k': k_current,
                'y': y_current,
                'c': c_current,
                'A': self.model.A0,
                's': self.model.s,
                'n': self.model.n,
                'delta': self.model.delta
            })
            
            # Update capital for next period
            self.model.K0 = k_next * self.model.A0 * self.model.L0
        
        # Restore original parameters
        self._restore_original_parameters()
        
        return {
            'shock_results': pd.DataFrame(results),
            'shock_sequence': shock_sequence
        }
    
    def _calculate_welfare_change(self, transition: pd.DataFrame, 
                                new_steady_state: Dict) -> float:
        """
        Calculate welfare change from policy shock
        
        Parameters:
        -----------
        transition : pd.DataFrame
            Transition path
        new_steady_state : dict
            New steady state values
            
        Returns:
        --------
        float
            Welfare change
        """
        # Calculate present value of consumption
        rho = 0.02  # Discount rate
        discount_factor = np.exp(-rho * transition['time'])
        
        # Welfare change
        welfare_change = np.sum(transition['c'] * discount_factor) / len(transition)
        
        return welfare_change
    
    def _restore_original_parameters(self):
        """Restore original model parameters"""
        self.model.s = self.original_params['s']
        self.model.n = self.original_params['n']
        self.model.g = self.original_params['g']
        self.model.delta = self.original_params['delta']
        self.model.alpha = self.original_params['alpha']
        self.model._calculate_steady_state()
    
    def impulse_response(self, shock_type: str, shock_magnitude: float,
                        T: int = 100) -> pd.DataFrame:
        """
        Calculate impulse response function
        
        Parameters:
        -----------
        shock_type : str
            Type of shock
        shock_magnitude : float
            Magnitude of shock
        T : int
            Number of time periods
            
        Returns:
        --------
        pd.DataFrame
            Impulse response function
        """
        if shock_type == 'technology':
            return self.technology_shock(shock_magnitude, T=T)['shock_results']
        elif shock_type == 'savings':
            return self.savings_rate_shock(shock_magnitude, T=T)['transition_path']
        elif shock_type == 'population':
            return self.population_growth_shock(shock_magnitude, T=T)['transition_path']
        elif shock_type == 'depreciation':
            return self.depreciation_shock(shock_magnitude, T=T)['transition_path']
        else:
            raise ValueError("Unknown shock type")
    
    def variance_decomposition(self, shock_variances: Dict, T: int = 100) -> Dict:
        """
        Variance decomposition of output growth
        
        Parameters:
        -----------
        shock_variances : dict
            Variances of different shocks
        T : int
            Number of time periods
            
        Returns:
        --------
        dict
            Variance decomposition results
        """
        # Simulate with random shocks
        np.random.seed(42)
        
        results = []
        
        for t in range(T):
            # Generate random shocks
            tech_shock = np.random.normal(0, np.sqrt(shock_variances.get('technology', 0)))
            savings_shock = np.random.normal(0, np.sqrt(shock_variances.get('savings', 0)))
            pop_shock = np.random.normal(0, np.sqrt(shock_variances.get('population', 0)))
            dep_shock = np.random.normal(0, np.sqrt(shock_variances.get('depreciation', 0)))
            
            # Apply shocks
            self.model.A0 *= (1 + tech_shock)
            self.model.s += savings_shock
            self.model.n += pop_shock
            self.model.delta += dep_shock
            
            # Recalculate steady state
            self.model._calculate_steady_state()
            
            # Simulate one period
            k_current = self.model.K0 / (self.model.A0 * self.model.L0)
            k_next = k_current + self.model.capital_accumulation(k_current)
            
            y_current = self.production_function(k_current)
            
            results.append({
                'time': t,
                'y': y_current,
                'tech_shock': tech_shock,
                'savings_shock': savings_shock,
                'pop_shock': pop_shock,
                'dep_shock': dep_shock
            })
            
            # Update capital for next period
            self.model.K0 = k_next * self.model.A0 * self.model.L0
        
        # Restore original parameters
        self._restore_original_parameters()
        
        # Calculate variance decomposition
        results_df = pd.DataFrame(results)
        
        # Regress output on shocks
        from sklearn.linear_model import LinearRegression
        
        X = results_df[['tech_shock', 'savings_shock', 'pop_shock', 'dep_shock']]
        y = results_df['y']
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate contributions
        contributions = {}
        for i, shock in enumerate(['technology', 'savings', 'population', 'depreciation']):
            contributions[shock] = model.coef_[i]**2 * np.var(X.iloc[:, i])
        
        total_variance = np.var(y)
        
        return {
            'total_variance': total_variance,
            'contributions': contributions,
            'relative_contributions': {k: v/total_variance for k, v in contributions.items()},
            'simulation_results': results_df
        }