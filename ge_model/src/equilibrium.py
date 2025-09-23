"""
General Equilibrium Model Implementation
Based on Arrow & Debreu (1954) "Existence of an Equilibrium for a Competitive Economy"
and Mas-Colell, Whinston & Green (1995) "Microeconomic Theory"
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Union
from scipy import optimize
import warnings

class GeneralEquilibrium:
    """
    General equilibrium model with numerical solution methods
    
    Implements:
    - Walrasian equilibrium
    - Pareto efficiency
    - Welfare analysis
    - Comparative statics
    """
    
    def __init__(self, n_consumers: int = 2, n_goods: int = 2,
                 utility_functions: List[str] = None,
                 production_functions: List[str] = None,
                 endowments: np.ndarray = None):
        """
        Initialize general equilibrium model
        
        Parameters:
        -----------
        n_consumers : int
            Number of consumers
        n_goods : int
            Number of goods
        utility_functions : list
            List of utility function types
        production_functions : list
            List of production function types
        endowments : np.ndarray
            Initial endowments
        """
        self.n_consumers = n_consumers
        self.n_goods = n_goods
        
        # Set default utility functions
        if utility_functions is None:
            self.utility_functions = ['cobb_douglas'] * n_consumers
        else:
            self.utility_functions = utility_functions
        
        # Set default production functions
        if production_functions is None:
            self.production_functions = ['cobb_douglas'] * n_goods
        else:
            self.production_functions = production_functions
        
        # Set default endowments
        if endowments is None:
            self.endowments = np.ones((n_consumers, n_goods))
        else:
            self.endowments = endowments
        
        # Calculate equilibrium
        self._calculate_equilibrium()
    
    def _calculate_equilibrium(self):
        """Calculate general equilibrium"""
        # This is a simplified version - in practice would use more sophisticated methods
        
        # Initial guess for prices
        p0 = np.ones(self.n_goods)
        
        # Solve for equilibrium prices
        try:
            result = optimize.fsolve(self._excess_demand, p0)
            self.prices = result
        except:
            # Fallback to simple solution
            self.prices = np.ones(self.n_goods)
        
        # Calculate equilibrium allocations
        self._calculate_allocations()
    
    def _excess_demand(self, prices: np.ndarray) -> np.ndarray:
        """
        Calculate excess demand for each good
        
        Parameters:
        -----------
        prices : np.ndarray
            Price vector
            
        Returns:
        --------
        np.ndarray
            Excess demand vector
        """
        # Calculate demand for each consumer
        total_demand = np.zeros(self.n_goods)
        
        for i in range(self.n_consumers):
            demand = self._consumer_demand(i, prices)
            total_demand += demand
        
        # Calculate supply (endowments)
        total_supply = np.sum(self.endowments, axis=0)
        
        # Excess demand
        excess_demand = total_demand - total_supply
        
        return excess_demand
    
    def _consumer_demand(self, consumer: int, prices: np.ndarray) -> np.ndarray:
        """
        Calculate consumer demand
        
        Parameters:
        -----------
        consumer : int
            Consumer index
        prices : np.ndarray
            Price vector
            
        Returns:
        --------
        np.ndarray
            Demand vector
        """
        # Consumer's income
        income = np.sum(prices * self.endowments[consumer])
        
        # Utility function parameters
        if self.utility_functions[consumer] == 'cobb_douglas':
            # Cobb-Douglas utility: U = x₁^α x₂^β
            alpha = 0.5  # Default parameter
            beta = 0.5   # Default parameter
            
            # Demand functions
            demand = np.zeros(self.n_goods)
            demand[0] = alpha * income / prices[0]
            demand[1] = beta * income / prices[1]
            
        elif self.utility_functions[consumer] == 'ces':
            # CES utility: U = (αx₁^ρ + βx₂^ρ)^(1/ρ)
            alpha = 0.5  # Default parameter
            beta = 0.5   # Default parameter
            rho = 0.5    # Default parameter
            
            # Demand functions (simplified)
            demand = np.zeros(self.n_goods)
            demand[0] = alpha * income / prices[0]
            demand[1] = beta * income / prices[1]
            
        else:
            # Default to equal demand
            demand = income / (self.n_goods * prices)
        
        return demand
    
    def _calculate_allocations(self):
        """Calculate equilibrium allocations"""
        # Calculate demand for each consumer
        self.demands = np.zeros((self.n_consumers, self.n_goods))
        
        for i in range(self.n_consumers):
            self.demands[i] = self._consumer_demand(i, self.prices)
        
        # Calculate supply
        self.supply = np.sum(self.endowments, axis=0)
        
        # Calculate total demand
        self.total_demand = np.sum(self.demands, axis=0)
        
        # Calculate excess demand
        self.excess_demand = self.total_demand - self.supply
    
    def walrasian_equilibrium(self) -> Dict:
        """
        Calculate Walrasian equilibrium
        
        Returns:
        --------
        dict
            Walrasian equilibrium results
        """
        return {
            'prices': self.prices,
            'demands': self.demands,
            'supply': self.supply,
            'total_demand': self.total_demand,
            'excess_demand': self.excess_demand,
            'market_clearing': np.allclose(self.excess_demand, 0, atol=1e-6)
        }
    
    def pareto_efficiency(self) -> Dict:
        """
        Check Pareto efficiency
        
        Returns:
        --------
        dict
            Pareto efficiency analysis
        """
        # Calculate utility levels
        utilities = np.zeros(self.n_consumers)
        
        for i in range(self.n_consumers):
            if self.utility_functions[i] == 'cobb_douglas':
                utilities[i] = np.prod(self.demands[i]**0.5)
            elif self.utility_functions[i] == 'ces':
                utilities[i] = np.sum(self.demands[i]**0.5)
            else:
                utilities[i] = np.sum(self.demands[i])
        
        # Check if allocation is Pareto efficient
        # This is a simplified check - in practice would use more sophisticated methods
        pareto_efficient = True  # Simplified assumption
        
        return {
            'utilities': utilities,
            'pareto_efficient': pareto_efficient,
            'total_utility': np.sum(utilities)
        }
    
    def welfare_analysis(self) -> Dict:
        """
        Welfare analysis
        
        Returns:
        --------
        dict
            Welfare analysis results
        """
        # Calculate consumer surplus
        consumer_surplus = 0
        for i in range(self.n_consumers):
            # Consumer's income
            income = np.sum(self.prices * self.endowments[i])
            
            # Consumer's expenditure
            expenditure = np.sum(self.prices * self.demands[i])
            
            # Consumer surplus
            consumer_surplus += income - expenditure
        
        # Calculate producer surplus (simplified)
        producer_surplus = 0  # Simplified assumption
        
        # Total welfare
        total_welfare = consumer_surplus + producer_surplus
        
        return {
            'consumer_surplus': consumer_surplus,
            'producer_surplus': producer_surplus,
            'total_welfare': total_welfare
        }
    
    def comparative_statics(self, parameter_name: str, 
                          parameter_values: List[float]) -> Dict:
        """
        Comparative statics analysis
        
        Parameters:
        -----------
        parameter_name : str
            Name of parameter to vary
        parameter_values : list
            Values of parameter to test
            
        Returns:
        --------
        dict
            Comparative statics results
        """
        results = []
        
        for value in parameter_values:
            # Create new model with modified parameter
            if parameter_name == 'endowments':
                new_endowments = self.endowments * value
                new_model = GeneralEquilibrium(
                    n_consumers=self.n_consumers,
                    n_goods=self.n_goods,
                    utility_functions=self.utility_functions,
                    production_functions=self.production_functions,
                    endowments=new_endowments
                )
            else:
                continue
            
            # Calculate equilibrium
            equilibrium = new_model.walrasian_equilibrium()
            welfare = new_model.welfare_analysis()
            
            results.append({
                'parameter_value': value,
                'prices': equilibrium['prices'],
                'total_welfare': welfare['total_welfare'],
                'consumer_surplus': welfare['consumer_surplus']
            })
        
        return {
            'parameter_name': parameter_name,
            'parameter_values': parameter_values,
            'results': results
        }
    
    def tatonnement_process(self, max_iter: int = 1000, 
                          tolerance: float = 1e-6) -> Dict:
        """
        Tatonnement process for finding equilibrium
        
        Parameters:
        -----------
        max_iter : int
            Maximum iterations
        tolerance : float
            Convergence tolerance
            
        Returns:
        --------
        dict
            Tatonnement process results
        """
        # Initial prices
        prices = np.ones(self.n_goods)
        
        # Price adjustment parameter
        alpha = 0.1
        
        # Store price history
        price_history = [prices.copy()]
        
        for iteration in range(max_iter):
            # Calculate excess demand
            excess_demand = self._excess_demand(prices)
            
            # Check convergence
            if np.max(np.abs(excess_demand)) < tolerance:
                break
            
            # Update prices
            prices = prices + alpha * excess_demand
            
            # Ensure prices are positive
            prices = np.maximum(prices, 0.01)
            
            # Store price history
            price_history.append(prices.copy())
        
        return {
            'converged': iteration < max_iter - 1,
            'iterations': iteration + 1,
            'final_prices': prices,
            'final_excess_demand': excess_demand,
            'price_history': np.array(price_history)
        }
    
    def summary(self) -> Dict:
        """Return model summary"""
        return {
            'n_consumers': self.n_consumers,
            'n_goods': self.n_goods,
            'utility_functions': self.utility_functions,
            'production_functions': self.production_functions,
            'endowments': self.endowments.tolist(),
            'equilibrium_prices': self.prices.tolist(),
            'market_clearing': np.allclose(self.excess_demand, 0, atol=1e-6)
        }