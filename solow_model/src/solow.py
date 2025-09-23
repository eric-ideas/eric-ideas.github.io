"""
Solow Growth Model Implementation
Based on Solow (1956) "A Contribution to the Theory of Economic Growth"
and Barro & Sala-i-Martin (2004) "Economic Growth"
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Union
from scipy import optimize, integrate
import warnings

class SolowModel:
    """
    Solow growth model with numerical solution methods
    
    Implements the standard Solow model with:
    - Cobb-Douglas production function
    - Capital accumulation equation
    - Steady state analysis
    - Transition dynamics
    """
    
    def __init__(self, alpha: float = 0.3, delta: float = 0.05, 
                 s: float = 0.2, n: float = 0.01, g: float = 0.02,
                 A0: float = 1.0, K0: float = 1.0, L0: float = 1.0):
        """
        Initialize Solow model
        
        Parameters:
        -----------
        alpha : float
            Capital share in production function
        delta : float
            Depreciation rate
        s : float
            Savings rate
        n : float
            Population growth rate
        g : float
            Technology growth rate
        A0 : float
            Initial technology level
        K0 : float
            Initial capital stock
        L0 : float
            Initial labor force
        """
        self.alpha = alpha
        self.delta = delta
        self.s = s
        self.n = n
        self.g = g
        self.A0 = A0
        self.K0 = K0
        self.L0 = L0
        
        # Calculate steady state
        self._calculate_steady_state()
    
    def _calculate_steady_state(self):
        """Calculate steady state values"""
        # Steady state capital per effective worker
        self.k_star = (self.s / (self.n + self.g + self.delta))**(1 / (1 - self.alpha))
        
        # Steady state output per effective worker
        self.y_star = self.k_star**self.alpha
        
        # Steady state consumption per effective worker
        self.c_star = (1 - self.s) * self.y_star
        
        # Steady state investment per effective worker
        self.i_star = self.s * self.y_star
    
    def production_function(self, k: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Cobb-Douglas production function
        
        Y = A * K^α * L^(1-α)
        y = k^α (in per effective worker terms)
        
        Parameters:
        -----------
        k : float or np.ndarray
            Capital per effective worker
            
        Returns:
        --------
        float or np.ndarray
            Output per effective worker
        """
        return k**self.alpha
    
    def capital_accumulation(self, k: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Capital accumulation equation
        
        dk/dt = s * y - (n + g + δ) * k
        
        Parameters:
        -----------
        k : float or np.ndarray
            Capital per effective worker
            
        Returns:
        --------
        float or np.ndarray
            Change in capital per effective worker
        """
        y = self.production_function(k)
        return self.s * y - (self.n + self.g + self.delta) * k
    
    def simulate_transition(self, T: int = 100, dt: float = 0.1) -> pd.DataFrame:
        """
        Simulate transition dynamics using numerical integration
        
        Parameters:
        -----------
        T : int
            Number of time periods
        dt : float
            Time step size
            
        Returns:
        --------
        pd.DataFrame
            Time series of model variables
        """
        # Time vector
        t = np.arange(0, T, dt)
        
        # Initial conditions
        k0 = self.K0 / (self.A0 * self.L0)
        
        # Solve differential equation
        def dkdt(t, k):
            return self.capital_accumulation(k)
        
        try:
            sol = integrate.solve_ivp(dkdt, [0, T], [k0], t_eval=t, method='RK45')
            k_path = sol.y[0]
        except:
            # Fallback to Euler method
            k_path = np.zeros(len(t))
            k_path[0] = k0
            
            for i in range(1, len(t)):
                k_path[i] = k_path[i-1] + dt * self.capital_accumulation(k_path[i-1])
        
        # Calculate other variables
        y_path = self.production_function(k_path)
        c_path = (1 - self.s) * y_path
        i_path = self.s * y_path
        
        # Calculate growth rates
        k_growth = np.gradient(k_path, dt) / k_path
        y_growth = np.gradient(y_path, dt) / y_path
        
        # Create DataFrame
        results = pd.DataFrame({
            'time': t,
            'k': k_path,
            'y': y_path,
            'c': c_path,
            'i': i_path,
            'k_growth': k_growth,
            'y_growth': y_growth
        })
        
        return results
    
    def golden_rule_savings_rate(self) -> float:
        """
        Calculate golden rule savings rate
        
        The savings rate that maximizes steady state consumption
        
        Returns:
        --------
        float
            Golden rule savings rate
        """
        # Golden rule savings rate
        s_golden = self.alpha
        
        return s_golden
    
    def golden_rule_steady_state(self) -> Dict:
        """
        Calculate golden rule steady state
        
        Returns:
        --------
        dict
            Golden rule steady state values
        """
        s_golden = self.golden_rule_savings_rate()
        
        # Golden rule steady state
        k_golden = (s_golden / (self.n + self.g + self.delta))**(1 / (1 - self.alpha))
        y_golden = k_golden**self.alpha
        c_golden = (1 - s_golden) * y_golden
        
        return {
            'savings_rate': s_golden,
            'k_star': k_golden,
            'y_star': y_golden,
            'c_star': c_golden
        }
    
    def analyze_policy_change(self, new_s: float, T: int = 100) -> Dict:
        """
        Analyze the effects of a change in savings rate
        
        Parameters:
        -----------
        new_s : float
            New savings rate
        T : int
            Number of time periods for simulation
            
        Returns:
        --------
        dict
            Analysis results
        """
        # Store original savings rate
        original_s = self.s
        
        # Calculate new steady state
        self.s = new_s
        self._calculate_steady_state()
        new_steady_state = {
            'k_star': self.k_star,
            'y_star': self.y_star,
            'c_star': self.c_star
        }
        
        # Simulate transition
        transition = self.simulate_transition(T)
        
        # Restore original savings rate
        self.s = original_s
        self._calculate_steady_state()
        
        return {
            'new_steady_state': new_steady_state,
            'transition_path': transition,
            'savings_rate_change': new_s - original_s
        }
    
    def analyze_technology_shock(self, shock_magnitude: float, 
                               shock_duration: int = 10, T: int = 100) -> Dict:
        """
        Analyze the effects of a technology shock
        
        Parameters:
        -----------
        shock_magnitude : float
            Magnitude of technology shock
        shock_duration : int
            Duration of shock
        T : int
            Number of time periods for simulation
            
        Returns:
        --------
        dict
            Analysis results
        """
        # Store original technology level
        original_A0 = self.A0
        
        # Simulate with shock
        results = []
        
        for t in range(T):
            if t < shock_duration:
                # Shock period
                self.A0 = original_A0 * (1 + shock_magnitude)
            else:
                # Post-shock period
                self.A0 = original_A0
            
            # Recalculate steady state
            self._calculate_steady_state()
            
            # Simulate one period
            k_current = self.K0 / (self.A0 * self.L0)
            k_next = k_current + self.capital_accumulation(k_current)
            
            y_current = self.production_function(k_current)
            c_current = (1 - self.s) * y_current
            
            results.append({
                'time': t,
                'k': k_current,
                'y': y_current,
                'c': c_current,
                'A': self.A0
            })
            
            # Update capital for next period
            self.K0 = k_next * self.A0 * self.L0
        
        # Restore original technology level
        self.A0 = original_A0
        self._calculate_steady_state()
        
        return {
            'shock_results': pd.DataFrame(results),
            'shock_magnitude': shock_magnitude,
            'shock_duration': shock_duration
        }
    
    def convergence_speed(self) -> float:
        """
        Calculate convergence speed to steady state
        
        Returns:
        --------
        float
            Convergence speed (λ)
        """
        # Convergence speed
        lambda_convergence = (1 - self.alpha) * (self.n + self.g + self.delta)
        
        return lambda_convergence
    
    def half_life(self) -> float:
        """
        Calculate half-life of convergence
        
        Returns:
        --------
        float
            Half-life in years
        """
        lambda_convergence = self.convergence_speed()
        half_life = np.log(2) / lambda_convergence
        
        return half_life
    
    def growth_accounting(self, k_growth: float, l_growth: float, 
                         a_growth: float) -> Dict:
        """
        Growth accounting decomposition
        
        Parameters:
        -----------
        k_growth : float
            Capital growth rate
        l_growth : float
            Labor growth rate
        a_growth : float
            Technology growth rate
            
        Returns:
        --------
        dict
            Growth accounting results
        """
        # Total output growth
        y_growth = self.alpha * k_growth + (1 - self.alpha) * l_growth + a_growth
        
        # Contributions
        capital_contribution = self.alpha * k_growth
        labor_contribution = (1 - self.alpha) * l_growth
        technology_contribution = a_growth
        
        return {
            'total_growth': y_growth,
            'capital_contribution': capital_contribution,
            'labor_contribution': labor_contribution,
            'technology_contribution': technology_contribution,
            'capital_share': capital_contribution / y_growth if y_growth != 0 else 0,
            'labor_share': labor_contribution / y_growth if y_growth != 0 else 0,
            'technology_share': technology_contribution / y_growth if y_growth != 0 else 0
        }
    
    def summary(self) -> Dict:
        """Return model summary"""
        return {
            'parameters': {
                'alpha': self.alpha,
                'delta': self.delta,
                's': self.s,
                'n': self.n,
                'g': self.g
            },
            'steady_state': {
                'k_star': self.k_star,
                'y_star': self.y_star,
                'c_star': self.c_star,
                'i_star': self.i_star
            },
            'convergence': {
                'speed': self.convergence_speed(),
                'half_life': self.half_life()
            },
            'golden_rule': self.golden_rule_steady_state()
        }