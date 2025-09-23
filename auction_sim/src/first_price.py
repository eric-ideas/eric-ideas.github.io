"""
First-Price Sealed-Bid Auction Simulation
Based on Vickrey (1961) "Counterspeculation, Auctions, and Competitive Sealed Tenders"
and Myerson (1981) "Optimal Auction Design"
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Union
from scipy import stats, optimize
import warnings

class FirstPriceAuction:
    """
    First-price sealed-bid auction simulation
    
    Implements:
    - Nash equilibrium bidding strategies
    - Revenue analysis
    - Efficiency analysis
    - Comparative statics
    """
    
    def __init__(self, n_bidders: int = 2, value_distribution: str = 'uniform',
                 value_params: Tuple[float, float] = (0, 1),
                 reserve_price: float = 0.0):
        """
        Initialize first-price auction
        
        Parameters:
        -----------
        n_bidders : int
            Number of bidders
        value_distribution : str
            Distribution of values ('uniform', 'normal', 'exponential')
        value_params : tuple
            Parameters for value distribution
        reserve_price : float
            Reserve price
        """
        self.n_bidders = n_bidders
        self.value_distribution = value_distribution
        self.value_params = value_params
        self.reserve_price = reserve_price
        
        # Calculate equilibrium bidding strategy
        self._calculate_equilibrium_strategy()
    
    def _calculate_equilibrium_strategy(self):
        """Calculate Nash equilibrium bidding strategy"""
        if self.value_distribution == 'uniform':
            # For uniform distribution on [0, 1], equilibrium bid is (n-1)/n * value
            self.bidding_function = lambda v: (self.n_bidders - 1) / self.n_bidders * v
        elif self.value_distribution == 'normal':
            # For normal distribution, use numerical solution
            self.bidding_function = self._numerical_bidding_function
        elif self.value_distribution == 'exponential':
            # For exponential distribution, use analytical solution
            self.bidding_function = lambda v: v - 1 / (self.n_bidders - 1)
        else:
            raise ValueError("Unknown value distribution")
    
    def _numerical_bidding_function(self, v: float) -> float:
        """Numerical solution for bidding function"""
        # This is a simplified version - in practice would use more sophisticated methods
        return v * (self.n_bidders - 1) / self.n_bidders
    
    def generate_values(self, n_auctions: int = 1000) -> np.ndarray:
        """
        Generate bidder values
        
        Parameters:
        -----------
        n_auctions : int
            Number of auctions to simulate
            
        Returns:
        --------
        np.ndarray
            Array of bidder values
        """
        if self.value_distribution == 'uniform':
            values = np.random.uniform(self.value_params[0], self.value_params[1], 
                                     (n_auctions, self.n_bidders))
        elif self.value_distribution == 'normal':
            values = np.random.normal(self.value_params[0], self.value_params[1], 
                                    (n_auctions, self.n_bidders))
        elif self.value_distribution == 'exponential':
            values = np.random.exponential(self.value_params[0], 
                                         (n_auctions, self.n_bidders))
        else:
            raise ValueError("Unknown value distribution")
        
        # Ensure values are non-negative
        values = np.maximum(values, 0)
        
        return values
    
    def simulate_auctions(self, n_auctions: int = 1000) -> Dict:
        """
        Simulate first-price auctions
        
        Parameters:
        -----------
        n_auctions : int
            Number of auctions to simulate
            
        Returns:
        --------
        dict
            Simulation results
        """
        # Generate values
        values = self.generate_values(n_auctions)
        
        # Calculate bids
        bids = np.zeros_like(values)
        for i in range(n_auctions):
            for j in range(self.n_bidders):
                bids[i, j] = self.bidding_function(values[i, j])
        
        # Apply reserve price
        bids = np.maximum(bids, self.reserve_price)
        
        # Determine winners and payments
        winners = np.argmax(bids, axis=1)
        winning_bids = np.max(bids, axis=1)
        winning_values = values[np.arange(n_auctions), winners]
        
        # Calculate outcomes
        revenue = np.sum(winning_bids)
        efficiency = np.sum(winning_values) / np.sum(np.max(values, axis=1))
        
        # Calculate bidder profits
        profits = winning_values - winning_bids
        
        return {
            'n_auctions': n_auctions,
            'values': values,
            'bids': bids,
            'winners': winners,
            'winning_bids': winning_bids,
            'winning_values': winning_values,
            'revenue': revenue,
            'efficiency': efficiency,
            'profits': profits,
            'average_revenue': revenue / n_auctions,
            'average_efficiency': efficiency,
            'average_profit': np.mean(profits)
        }
    
    def analyze_bidding_behavior(self, n_auctions: int = 1000) -> Dict:
        """
        Analyze bidding behavior
        
        Parameters:
        -----------
        n_auctions : int
            Number of auctions to simulate
            
        Returns:
        --------
        dict
            Bidding behavior analysis
        """
        # Generate values
        values = self.generate_values(n_auctions)
        
        # Calculate bids
        bids = np.zeros_like(values)
        for i in range(n_auctions):
            for j in range(self.n_bidders):
                bids[i, j] = self.bidding_function(values[i, j])
        
        # Calculate bid-to-value ratios
        bid_value_ratios = bids / values
        
        # Calculate overbidding (bids > values)
        overbidding = np.sum(bids > values) / (n_auctions * self.n_bidders)
        
        # Calculate bid dispersion
        bid_dispersion = np.std(bids, axis=1)
        
        # Calculate bid correlation with values
        bid_value_correlation = np.corrcoef(bids.flatten(), values.flatten())[0, 1]
        
        return {
            'n_auctions': n_auctions,
            'bid_value_ratios': bid_value_ratios,
            'overbidding_rate': overbidding,
            'bid_dispersion': bid_dispersion,
            'bid_value_correlation': bid_value_correlation,
            'average_bid_value_ratio': np.mean(bid_value_ratios),
            'std_bid_value_ratio': np.std(bid_value_ratios)
        }
    
    def revenue_analysis(self, n_auctions: int = 1000) -> Dict:
        """
        Analyze auction revenue
        
        Parameters:
        -----------
        n_auctions : int
            Number of auctions to simulate
            
        Returns:
        --------
        dict
            Revenue analysis results
        """
        # Simulate auctions
        results = self.simulate_auctions(n_auctions)
        
        # Calculate revenue statistics
        revenue_stats = {
            'total_revenue': results['revenue'],
            'average_revenue': results['average_revenue'],
            'revenue_std': np.std(results['winning_bids']),
            'revenue_min': np.min(results['winning_bids']),
            'revenue_max': np.max(results['winning_bids']),
            'revenue_median': np.median(results['winning_bids'])
        }
        
        # Calculate revenue efficiency
        max_possible_revenue = np.sum(np.max(results['values'], axis=1))
        revenue_efficiency = results['revenue'] / max_possible_revenue
        
        return {
            'revenue_stats': revenue_stats,
            'revenue_efficiency': revenue_efficiency,
            'max_possible_revenue': max_possible_revenue
        }
    
    def efficiency_analysis(self, n_auctions: int = 1000) -> Dict:
        """
        Analyze auction efficiency
        
        Parameters:
        -----------
        n_auctions : int
            Number of auctions to simulate
            
        Returns:
        --------
        dict
            Efficiency analysis results
        """
        # Simulate auctions
        results = self.simulate_auctions(n_auctions)
        
        # Calculate efficiency statistics
        efficiency_stats = {
            'average_efficiency': results['average_efficiency'],
            'efficiency_std': np.std(results['winning_values'] / np.max(results['values'], axis=1)),
            'efficiency_min': np.min(results['winning_values'] / np.max(results['values'], axis=1)),
            'efficiency_max': np.max(results['winning_values'] / np.max(results['values'], axis=1)),
            'efficiency_median': np.median(results['winning_values'] / np.max(results['values'], axis=1))
        }
        
        # Calculate allocative efficiency
        allocative_efficiency = np.sum(results['winning_values']) / np.sum(np.max(results['values'], axis=1))
        
        return {
            'efficiency_stats': efficiency_stats,
            'allocative_efficiency': allocative_efficiency
        }
    
    def comparative_statics(self, parameter_values: List[float], 
                          parameter_name: str) -> Dict:
        """
        Comparative statics analysis
        
        Parameters:
        -----------
        parameter_values : list
            Values of parameter to test
        parameter_name : str
            Name of parameter to vary
            
        Returns:
        --------
        dict
            Comparative statics results
        """
        results = []
        
        for value in parameter_values:
            # Create new auction with modified parameter
            if parameter_name == 'n_bidders':
                new_auction = FirstPriceAuction(n_bidders=int(value), 
                                               value_distribution=self.value_distribution,
                                               value_params=self.value_params,
                                               reserve_price=self.reserve_price)
            elif parameter_name == 'reserve_price':
                new_auction = FirstPriceAuction(n_bidders=self.n_bidders,
                                               value_distribution=self.value_distribution,
                                               value_params=self.value_params,
                                               reserve_price=value)
            else:
                continue
            
            # Simulate auctions
            sim_results = new_auction.simulate_auctions(1000)
            
            results.append({
                'parameter_value': value,
                'average_revenue': sim_results['average_revenue'],
                'average_efficiency': sim_results['average_efficiency'],
                'average_profit': sim_results['average_profit']
            })
        
        return {
            'parameter_name': parameter_name,
            'parameter_values': parameter_values,
            'results': results
        }
    
    def summary(self) -> Dict:
        """Return model summary"""
        return {
            'n_bidders': self.n_bidders,
            'value_distribution': self.value_distribution,
            'value_params': self.value_params,
            'reserve_price': self.reserve_price,
            'bidding_function': self.bidding_function.__name__ if hasattr(self.bidding_function, '__name__') else 'custom'
        }