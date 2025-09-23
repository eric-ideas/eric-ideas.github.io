"""
Cournot Competition Model
Based on Cournot (1838) "Recherches sur les principes mathématiques de la théorie des richesses"
and Tirole (1988) "The Theory of Industrial Organization"
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Union
from scipy import optimize
import warnings

class CournotCompetition:
    """
    Cournot competition model with numerical solution methods
    
    Implements:
    - Static Cournot equilibrium
    - Dynamic Cournot competition
    - Entry and exit analysis
    - Welfare analysis
    """
    
    def __init__(self, n_firms: int = 2, a: float = 100, b: float = 1,
                 c: float = 10, fixed_cost: float = 0):
        """
        Initialize Cournot competition model
        
        Parameters:
        -----------
        n_firms : int
            Number of firms
        a : float
            Demand intercept
        b : float
            Demand slope
        c : float
            Marginal cost
        fixed_cost : float
            Fixed cost
        """
        self.n_firms = n_firms
        self.a = a
        self.b = b
        self.c = c
        self.fixed_cost = fixed_cost
        
        # Calculate equilibrium
        self._calculate_equilibrium()
    
    def _calculate_equilibrium(self):
        """Calculate Cournot equilibrium"""
        # Inverse demand: P = a - bQ
        # Total quantity: Q = Σq_i
        # Firm i's profit: π_i = (P - c)q_i - F
        # FOC: ∂π_i/∂q_i = a - bQ - bq_i - c = 0
        # Solving: q_i = (a - c - bQ)/b
        # In equilibrium: q_i = (a - c)/(b(n+1))
        
        self.q_star = (self.a - self.c) / (self.b * (self.n_firms + 1))
        self.Q_star = self.n_firms * self.q_star
        self.P_star = self.a - self.b * self.Q_star
        self.profit_star = (self.P_star - self.c) * self.q_star - self.fixed_cost
        self.consumer_surplus = 0.5 * self.b * self.Q_star**2
        self.producer_surplus = self.n_firms * self.profit_star
        self.total_surplus = self.consumer_surplus + self.producer_surplus
    
    def demand_function(self, Q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Inverse demand function
        
        Parameters:
        -----------
        Q : float or np.ndarray
            Total quantity
            
        Returns:
        --------
        float or np.ndarray
            Price
        """
        return self.a - self.b * Q
    
    def marginal_revenue(self, q_i: float, Q_others: float) -> float:
        """
        Marginal revenue for firm i
        
        Parameters:
        -----------
        q_i : float
            Firm i's quantity
        Q_others : float
            Other firms' total quantity
            
        Returns:
        --------
        float
            Marginal revenue
        """
        return self.a - self.b * (q_i + Q_others) - self.b * q_i
    
    def profit_function(self, q_i: float, Q_others: float) -> float:
        """
        Profit function for firm i
        
        Parameters:
        -----------
        q_i : float
            Firm i's quantity
        Q_others : float
            Other firms' total quantity
            
        Returns:
        --------
        float
            Profit
        """
        P = self.demand_function(q_i + Q_others)
        return (P - self.c) * q_i - self.fixed_cost
    
    def best_response(self, Q_others: float) -> float:
        """
        Best response function for firm i
        
        Parameters:
        -----------
        Q_others : float
            Other firms' total quantity
            
        Returns:
        --------
        float
            Best response quantity
        """
        # FOC: a - b(q_i + Q_others) - bq_i - c = 0
        # Solving: q_i = (a - c - bQ_others)/(2b)
        return max(0, (self.a - self.c - self.b * Q_others) / (2 * self.b))
    
    def solve_nash_equilibrium(self, max_iter: int = 1000, 
                              tolerance: float = 1e-6) -> Dict:
        """
        Solve Nash equilibrium using iterative best response
        
        Parameters:
        -----------
        max_iter : int
            Maximum iterations
        tolerance : float
            Convergence tolerance
            
        Returns:
        --------
        dict
            Nash equilibrium results
        """
        # Initialize quantities
        q = np.ones(self.n_firms) * self.q_star
        
        for iteration in range(max_iter):
            q_old = q.copy()
            
            # Update each firm's quantity
            for i in range(self.n_firms):
                Q_others = np.sum(q) - q[i]
                q[i] = self.best_response(Q_others)
            
            # Check convergence
            if np.max(np.abs(q - q_old)) < tolerance:
                break
        
        # Calculate equilibrium values
        Q_total = np.sum(q)
        P = self.demand_function(Q_total)
        profits = [(P - self.c) * q[i] - self.fixed_cost for i in range(self.n_firms)]
        
        return {
            'quantities': q,
            'total_quantity': Q_total,
            'price': P,
            'profits': profits,
            'iterations': iteration + 1,
            'converged': iteration < max_iter - 1
        }
    
    def analyze_entry(self, new_firms: int = 1) -> Dict:
        """
        Analyze entry of new firms
        
        Parameters:
        -----------
        new_firms : int
            Number of new firms entering
            
        Returns:
        --------
        dict
            Entry analysis results
        """
        # Store original number of firms
        original_n = self.n_firms
        
        # Add new firms
        self.n_firms += new_firms
        self._calculate_equilibrium()
        
        # Calculate changes
        delta_Q = self.Q_star - (original_n * (self.a - self.c) / (self.b * (original_n + 1)))
        delta_P = self.P_star - (self.a - self.b * (original_n * (self.a - self.c) / (self.b * (original_n + 1))))
        delta_profit = self.profit_star - ((self.a - self.c) / (self.b * (original_n + 1)))**2 - self.fixed_cost
        
        # Restore original number of firms
        self.n_firms = original_n
        self._calculate_equilibrium()
        
        return {
            'new_n_firms': self.n_firms + new_firms,
            'delta_quantity': delta_Q,
            'delta_price': delta_P,
            'delta_profit': delta_profit,
            'new_equilibrium': {
                'quantity': self.q_star,
                'total_quantity': self.Q_star,
                'price': self.P_star,
                'profit': self.profit_star
            }
        }
    
    def analyze_cost_shock(self, delta_c: float) -> Dict:
        """
        Analyze cost shock
        
        Parameters:
        -----------
        delta_c : float
            Change in marginal cost
            
        Returns:
        --------
        dict
            Cost shock analysis results
        """
        # Store original marginal cost
        original_c = self.c
        
        # Apply shock
        self.c += delta_c
        self._calculate_equilibrium()
        
        # Calculate changes
        delta_Q = self.Q_star - (self.n_firms * (self.a - original_c) / (self.b * (self.n_firms + 1)))
        delta_P = self.P_star - (self.a - self.b * (self.n_firms * (self.a - original_c) / (self.b * (self.n_firms + 1))))
        delta_profit = self.profit_star - ((self.a - original_c) / (self.b * (self.n_firms + 1)))**2 - self.fixed_cost
        
        # Restore original marginal cost
        self.c = original_c
        self._calculate_equilibrium()
        
        return {
            'delta_cost': delta_c,
            'delta_quantity': delta_Q,
            'delta_price': delta_P,
            'delta_profit': delta_profit,
            'new_equilibrium': {
                'quantity': self.q_star,
                'total_quantity': self.Q_star,
                'price': self.P_star,
                'profit': self.profit_star
            }
        }
    
    def welfare_analysis(self) -> Dict:
        """
        Welfare analysis of Cournot equilibrium
        
        Returns:
        --------
        dict
            Welfare analysis results
        """
        # Perfect competition benchmark
        pc_quantity = (self.a - self.c) / self.b
        pc_price = self.c
        pc_consumer_surplus = 0.5 * self.b * pc_quantity**2
        pc_producer_surplus = 0  # Zero profit in perfect competition
        pc_total_surplus = pc_consumer_surplus
        
        # Deadweight loss
        deadweight_loss = pc_total_surplus - self.total_surplus
        
        # Market power
        market_power = (self.P_star - self.c) / self.P_star
        
        return {
            'cournot_equilibrium': {
                'quantity': self.Q_star,
                'price': self.P_star,
                'consumer_surplus': self.consumer_surplus,
                'producer_surplus': self.producer_surplus,
                'total_surplus': self.total_surplus
            },
            'perfect_competition': {
                'quantity': pc_quantity,
                'price': pc_price,
                'consumer_surplus': pc_consumer_surplus,
                'producer_surplus': pc_producer_surplus,
                'total_surplus': pc_total_surplus
            },
            'deadweight_loss': deadweight_loss,
            'market_power': market_power
        }
    
    def dynamic_competition(self, T: int = 100, 
                          adjustment_speed: float = 0.1) -> pd.DataFrame:
        """
        Simulate dynamic Cournot competition
        
        Parameters:
        -----------
        T : int
            Number of periods
        adjustment_speed : float
            Speed of quantity adjustment
            
        Returns:
        --------
        pd.DataFrame
            Dynamic competition results
        """
        # Initialize quantities
        q = np.ones(self.n_firms) * self.q_star
        
        results = []
        
        for t in range(T):
            # Calculate current equilibrium
            Q_total = np.sum(q)
            P = self.demand_function(Q_total)
            profits = [(P - self.c) * q[i] - self.fixed_cost for i in range(self.n_firms)]
            
            # Store results
            results.append({
                'period': t,
                'total_quantity': Q_total,
                'price': P,
                'quantities': q.copy(),
                'profits': profits.copy(),
                'total_profit': np.sum(profits)
            })
            
            # Update quantities (partial adjustment)
            for i in range(self.n_firms):
                Q_others = Q_total - q[i]
                q_target = self.best_response(Q_others)
                q[i] = q[i] + adjustment_speed * (q_target - q[i])
        
        return pd.DataFrame(results)
    
    def summary(self) -> Dict:
        """Return model summary"""
        return {
            'parameters': {
                'n_firms': self.n_firms,
                'a': self.a,
                'b': self.b,
                'c': self.c,
                'fixed_cost': self.fixed_cost
            },
            'equilibrium': {
                'quantity_per_firm': self.q_star,
                'total_quantity': self.Q_star,
                'price': self.P_star,
                'profit_per_firm': self.profit_star,
                'consumer_surplus': self.consumer_surplus,
                'producer_surplus': self.producer_surplus,
                'total_surplus': self.total_surplus
            }
        }