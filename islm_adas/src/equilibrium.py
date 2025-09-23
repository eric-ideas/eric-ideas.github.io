"""
IS-LM and AD-AS Equilibrium Models
Based on Hicks (1937) "Mr. Keynes and the Classics" and Blanchard (2017) "Macroeconomics"
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Union
from scipy import optimize
import warnings

class ISLMModel:
    """
    IS-LM model with numerical solution methods
    
    Implements the standard IS-LM model with:
    - IS curve: Y = C(Y-T) + I(r) + G
    - LM curve: M/P = L(Y, r)
    - Equilibrium solution
    - Policy analysis
    """
    
    def __init__(self, C0: float = 100, c: float = 0.8, I0: float = 50, 
                 b: float = 10, G: float = 200, T: float = 150,
                 M: float = 1000, P: float = 1.0, k: float = 0.5, 
                 h: float = 20, r0: float = 0.05):
        """
        Initialize IS-LM model
        
        Parameters:
        -----------
        C0 : float
            Autonomous consumption
        c : float
            Marginal propensity to consume
        I0 : float
            Autonomous investment
        b : float
            Investment sensitivity to interest rate
        G : float
            Government spending
        T : float
            Taxes
        M : float
            Money supply
        P : float
            Price level
        k : float
            Money demand sensitivity to income
        h : float
            Money demand sensitivity to interest rate
        r0 : float
            Initial interest rate
        """
        self.C0 = C0
        self.c = c
        self.I0 = I0
        self.b = b
        self.G = G
        self.T = T
        self.M = M
        self.P = P
        self.k = k
        self.h = h
        self.r0 = r0
        
        # Calculate equilibrium
        self._calculate_equilibrium()
    
    def _calculate_equilibrium(self):
        """Calculate IS-LM equilibrium"""
        # IS curve: Y = C0 + c(Y-T) + I0 - br + G
        # Solving for Y: Y = (C0 + I0 + G - cT - br) / (1-c)
        
        # LM curve: M/P = kY - hr
        # Solving for r: r = (kY - M/P) / h
        
        # Substitute LM into IS
        # Y = (C0 + I0 + G - cT - b(kY - M/P)/h) / (1-c)
        # Y = (C0 + I0 + G - cT + bM/(Ph)) / (1-c + bk/h)
        
        denominator = 1 - self.c + self.b * self.k / self.h
        numerator = self.C0 + self.I0 + self.G - self.c * self.T + self.b * self.M / (self.P * self.h)
        
        self.Y_star = numerator / denominator
        self.r_star = (self.k * self.Y_star - self.M / self.P) / self.h
        
        # Calculate other variables
        self.C_star = self.C0 + self.c * (self.Y_star - self.T)
        self.I_star = self.I0 - self.b * self.r_star
        self.L_star = self.k * self.Y_star - self.h * self.r_star
    
    def is_curve(self, r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        IS curve: Y = C(Y-T) + I(r) + G
        
        Parameters:
        -----------
        r : float or np.ndarray
            Interest rate
            
        Returns:
        --------
        float or np.ndarray
            Output level
        """
        return (self.C0 + self.I0 + self.G - self.c * self.T - self.b * r) / (1 - self.c)
    
    def lm_curve(self, Y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        LM curve: M/P = L(Y, r)
        
        Parameters:
        -----------
        Y : float or np.ndarray
            Output level
            
        Returns:
        --------
        float or np.ndarray
            Interest rate
        """
        return (self.k * Y - self.M / self.P) / self.h
    
    def fiscal_policy_shock(self, delta_G: float) -> Dict:
        """
        Analyze fiscal policy shock
        
        Parameters:
        -----------
        delta_G : float
            Change in government spending
            
        Returns:
        --------
        dict
            Fiscal policy analysis results
        """
        # Store original government spending
        original_G = self.G
        
        # Apply shock
        self.G += delta_G
        self._calculate_equilibrium()
        
        # Calculate multipliers
        delta_Y = self.Y_star - ((self.C0 + self.I0 + original_G - self.c * self.T - self.b * self.r_star) / (1 - self.c))
        fiscal_multiplier = delta_Y / delta_G
        
        # Restore original government spending
        self.G = original_G
        self._calculate_equilibrium()
        
        return {
            'delta_G': delta_G,
            'delta_Y': delta_Y,
            'fiscal_multiplier': fiscal_multiplier,
            'new_equilibrium': {
                'Y': self.Y_star,
                'r': self.r_star,
                'C': self.C_star,
                'I': self.I_star
            }
        }
    
    def monetary_policy_shock(self, delta_M: float) -> Dict:
        """
        Analyze monetary policy shock
        
        Parameters:
        -----------
        delta_M : float
            Change in money supply
            
        Returns:
        --------
        dict
            Monetary policy analysis results
        """
        # Store original money supply
        original_M = self.M
        
        # Apply shock
        self.M += delta_M
        self._calculate_equilibrium()
        
        # Calculate multipliers
        delta_Y = self.Y_star - ((self.C0 + self.I0 + self.G - self.c * self.T - self.b * self.r_star) / (1 - self.c))
        monetary_multiplier = delta_Y / delta_M
        
        # Restore original money supply
        self.M = original_M
        self._calculate_equilibrium()
        
        return {
            'delta_M': delta_M,
            'delta_Y': delta_Y,
            'monetary_multiplier': monetary_multiplier,
            'new_equilibrium': {
                'Y': self.Y_star,
                'r': self.r_star,
                'C': self.C_star,
                'I': self.I_star
            }
        }
    
    def tax_policy_shock(self, delta_T: float) -> Dict:
        """
        Analyze tax policy shock
        
        Parameters:
        -----------
        delta_T : float
            Change in taxes
            
        Returns:
        --------
        dict
            Tax policy analysis results
        """
        # Store original taxes
        original_T = self.T
        
        # Apply shock
        self.T += delta_T
        self._calculate_equilibrium()
        
        # Calculate multipliers
        delta_Y = self.Y_star - ((self.C0 + self.I0 + self.G - self.c * original_T - self.b * self.r_star) / (1 - self.c))
        tax_multiplier = delta_Y / delta_T
        
        # Restore original taxes
        self.T = original_T
        self._calculate_equilibrium()
        
        return {
            'delta_T': delta_T,
            'delta_Y': delta_Y,
            'tax_multiplier': tax_multiplier,
            'new_equilibrium': {
                'Y': self.Y_star,
                'r': self.r_star,
                'C': self.C_star,
                'I': self.I_star
            }
        }
    
    def summary(self) -> Dict:
        """Return model summary"""
        return {
            'parameters': {
                'C0': self.C0,
                'c': self.c,
                'I0': self.I0,
                'b': self.b,
                'G': self.G,
                'T': self.T,
                'M': self.M,
                'P': self.P,
                'k': self.k,
                'h': self.h
            },
            'equilibrium': {
                'Y': self.Y_star,
                'r': self.r_star,
                'C': self.C_star,
                'I': self.I_star,
                'L': self.L_star
            }
        }

class ADASModel:
    """
    AD-AS model with numerical solution methods
    
    Implements the standard AD-AS model with:
    - AD curve: Y = C(Y-T) + I(r) + G
    - AS curve: P = P_e + α(Y - Y_n)
    - Equilibrium solution
    - Policy analysis
    """
    
    def __init__(self, C0: float = 100, c: float = 0.8, I0: float = 50,
                 b: float = 10, G: float = 200, T: float = 150,
                 M: float = 1000, k: float = 0.5, h: float = 20,
                 P_e: float = 1.0, alpha: float = 0.1, Y_n: float = 1000):
        """
        Initialize AD-AS model
        
        Parameters:
        -----------
        C0 : float
            Autonomous consumption
        c : float
            Marginal propensity to consume
        I0 : float
            Autonomous investment
        b : float
            Investment sensitivity to interest rate
        G : float
            Government spending
        T : float
            Taxes
        M : float
            Money supply
        k : float
            Money demand sensitivity to income
        h : float
            Money demand sensitivity to interest rate
        P_e : float
            Expected price level
        alpha : float
            Price adjustment parameter
        Y_n : float
            Natural output level
        """
        self.C0 = C0
        self.c = c
        self.I0 = I0
        self.b = b
        self.G = G
        self.T = T
        self.M = M
        self.k = k
        self.h = h
        self.P_e = P_e
        self.alpha = alpha
        self.Y_n = Y_n
        
        # Calculate equilibrium
        self._calculate_equilibrium()
    
    def _calculate_equilibrium(self):
        """Calculate AD-AS equilibrium"""
        # AD curve: Y = (C0 + I0 + G - cT + bM/(Ph)) / (1-c + bk/h)
        # AS curve: P = P_e + α(Y - Y_n)
        
        # Solve for equilibrium
        # Y = (C0 + I0 + G - cT + bM/(P_e + α(Y - Y_n))/h) / (1-c + bk/h)
        
        # This is a nonlinear equation, solve numerically
        def equilibrium_equation(Y):
            P = self.P_e + self.alpha * (Y - self.Y_n)
            if P <= 0:
                return np.inf
            
            # IS curve with price level P
            Y_is = (self.C0 + self.I0 + self.G - self.c * self.T + self.b * self.M / (P * self.h)) / (1 - self.c + self.b * self.k / self.h)
            
            return Y - Y_is
        
        try:
            # Find equilibrium output
            self.Y_star = optimize.fsolve(equilibrium_equation, self.Y_n)[0]
            self.P_star = self.P_e + self.alpha * (self.Y_star - self.Y_n)
            
            # Calculate other variables
            self.r_star = (self.k * self.Y_star - self.M / self.P_star) / self.h
            self.C_star = self.C0 + self.c * (self.Y_star - self.T)
            self.I_star = self.I0 - self.b * self.r_star
            
        except:
            # Fallback to simple solution
            self.Y_star = self.Y_n
            self.P_star = self.P_e
            self.r_star = 0.05
            self.C_star = self.C0 + self.c * (self.Y_star - self.T)
            self.I_star = self.I0 - self.b * self.r_star
    
    def ad_curve(self, P: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        AD curve: Y = C(Y-T) + I(r) + G
        
        Parameters:
        -----------
        P : float or np.ndarray
            Price level
            
        Returns:
        --------
        float or np.ndarray
            Output level
        """
        return (self.C0 + self.I0 + self.G - self.c * self.T + self.b * self.M / (P * self.h)) / (1 - self.c + self.b * self.k / self.h)
    
    def as_curve(self, Y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        AS curve: P = P_e + α(Y - Y_n)
        
        Parameters:
        -----------
        Y : float or np.ndarray
            Output level
            
        Returns:
        --------
        float or np.ndarray
            Price level
        """
        return self.P_e + self.alpha * (Y - self.Y_n)
    
    def demand_shock(self, delta_C0: float) -> Dict:
        """
        Analyze demand shock
        
        Parameters:
        -----------
        delta_C0 : float
            Change in autonomous consumption
            
        Returns:
        --------
        dict
            Demand shock analysis results
        """
        # Store original autonomous consumption
        original_C0 = self.C0
        
        # Apply shock
        self.C0 += delta_C0
        self._calculate_equilibrium()
        
        # Calculate multipliers
        delta_Y = self.Y_star - ((original_C0 + self.I0 + self.G - self.c * self.T + self.b * self.M / (self.P_star * self.h)) / (1 - self.c + self.b * self.k / self.h))
        demand_multiplier = delta_Y / delta_C0
        
        # Restore original autonomous consumption
        self.C0 = original_C0
        self._calculate_equilibrium()
        
        return {
            'delta_C0': delta_C0,
            'delta_Y': delta_Y,
            'delta_P': self.P_star - (self.P_e + self.alpha * (self.Y_star - self.Y_n)),
            'demand_multiplier': demand_multiplier,
            'new_equilibrium': {
                'Y': self.Y_star,
                'P': self.P_star,
                'r': self.r_star,
                'C': self.C_star,
                'I': self.I_star
            }
        }
    
    def supply_shock(self, delta_P_e: float) -> Dict:
        """
        Analyze supply shock
        
        Parameters:
        -----------
        delta_P_e : float
            Change in expected price level
            
        Returns:
        --------
        dict
            Supply shock analysis results
        """
        # Store original expected price level
        original_P_e = self.P_e
        
        # Apply shock
        self.P_e += delta_P_e
        self._calculate_equilibrium()
        
        # Calculate multipliers
        delta_Y = self.Y_star - self.Y_n
        delta_P = self.P_star - original_P_e
        
        # Restore original expected price level
        self.P_e = original_P_e
        self._calculate_equilibrium()
        
        return {
            'delta_P_e': delta_P_e,
            'delta_Y': delta_Y,
            'delta_P': delta_P,
            'new_equilibrium': {
                'Y': self.Y_star,
                'P': self.P_star,
                'r': self.r_star,
                'C': self.C_star,
                'I': self.I_star
            }
        }
    
    def monetary_policy_shock(self, delta_M: float) -> Dict:
        """
        Analyze monetary policy shock
        
        Parameters:
        -----------
        delta_M : float
            Change in money supply
            
        Returns:
        --------
        dict
            Monetary policy analysis results
        """
        # Store original money supply
        original_M = self.M
        
        # Apply shock
        self.M += delta_M
        self._calculate_equilibrium()
        
        # Calculate multipliers
        delta_Y = self.Y_star - self.Y_n
        delta_P = self.P_star - self.P_e
        
        # Restore original money supply
        self.M = original_M
        self._calculate_equilibrium()
        
        return {
            'delta_M': delta_M,
            'delta_Y': delta_Y,
            'delta_P': delta_P,
            'new_equilibrium': {
                'Y': self.Y_star,
                'P': self.P_star,
                'r': self.r_star,
                'C': self.C_star,
                'I': self.I_star
            }
        }
    
    def summary(self) -> Dict:
        """Return model summary"""
        return {
            'parameters': {
                'C0': self.C0,
                'c': self.c,
                'I0': self.I0,
                'b': self.b,
                'G': self.G,
                'T': self.T,
                'M': self.M,
                'k': self.k,
                'h': self.h,
                'P_e': self.P_e,
                'alpha': self.alpha,
                'Y_n': self.Y_n
            },
            'equilibrium': {
                'Y': self.Y_star,
                'P': self.P_star,
                'r': self.r_star,
                'C': self.C_star,
                'I': self.I_star
            }
        }