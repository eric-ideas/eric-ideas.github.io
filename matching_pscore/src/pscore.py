"""
Propensity Score Estimation and Analysis
Based on Rosenbaum & Rubin (1983) "The Central Role of the Propensity Score"
and Imbens & Wooldridge (2009) "Recent Developments in the Econometrics of Program Evaluation"
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Union
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import warnings

class PropensityScore:
    """
    Propensity score estimation and analysis
    
    Implements various methods for estimating propensity scores including:
    - Logistic regression
    - Random forest
    - Cross-validation for model selection
    """
    
    def __init__(self, data: pd.DataFrame, treatment_var: str, 
                 covariates: List[str], outcome_var: Optional[str] = None):
        """
        Initialize propensity score analysis
        
        Parameters:
        -----------
        data : pd.DataFrame
            Dataset
        treatment_var : str
            Treatment indicator variable
        covariates : list
            Covariates for propensity score estimation
        outcome_var : str, optional
            Outcome variable for analysis
        """
        self.data = data.copy()
        self.treatment_var = treatment_var
        self.covariates = covariates
        self.outcome_var = outcome_var
        
        # Get treatment and control groups
        self.treated = self.data[self.data[treatment_var] == 1]
        self.control = self.data[self.data[treatment_var] == 0]
        self.n_treated = len(self.treated)
        self.n_control = len(self.control)
        
        # Prepare data for estimation
        self.X = self.data[covariates].values
        self.y = self.data[treatment_var].values
        
    def estimate_logistic(self, regularization: float = 1.0, 
                         max_iter: int = 1000) -> Dict:
        """
        Estimate propensity score using logistic regression
        
        Based on Rosenbaum & Rubin (1983) and Imbens & Wooldridge (2009)
        
        Parameters:
        -----------
        regularization : float
            Regularization parameter (C = 1/regularization)
        max_iter : int
            Maximum iterations for convergence
            
        Returns:
        --------
        dict
            Estimation results
        """
        # Fit logistic regression
        model = LogisticRegression(C=1/regularization, max_iter=max_iter, 
                                  random_state=42)
        model.fit(self.X, self.y)
        
        # Calculate propensity scores
        propensity_scores = model.predict_proba(self.X)[:, 1]
        
        # Calculate standard errors using delta method
        # Following Imbens & Wooldridge (2009, p. 25)
        X_with_const = np.column_stack([np.ones(len(self.X)), self.X])
        n = len(self.X)
        k = X_with_const.shape[1]
        
        # Calculate Hessian
        p = propensity_scores
        W = np.diag(p * (1 - p))
        H = X_with_const.T @ W @ X_with_const / n
        
        # Calculate standard errors
        try:
            H_inv = np.linalg.inv(H)
            se = np.sqrt(np.diag(H_inv / n))
        except np.linalg.LinAlgError:
            warnings.warn("Singular Hessian matrix - using pseudo-inverse")
            H_inv = np.linalg.pinv(H)
            se = np.sqrt(np.diag(H_inv / n))
        
        # Calculate pseudo-R-squared
        y_pred = model.predict(self.X)
        y_true = self.y
        
        # McFadden's R-squared
        ll_null = np.sum(y_true * np.log(np.mean(y_true)) + 
                        (1 - y_true) * np.log(1 - np.mean(y_true)))
        ll_model = np.sum(y_true * np.log(propensity_scores + 1e-10) + 
                        (1 - y_true) * np.log(1 - propensity_scores + 1e-10))
        pseudo_r2 = 1 - ll_model / ll_null
        
        # Calculate AIC and BIC
        aic = 2 * k - 2 * ll_model
        bic = k * np.log(n) - 2 * ll_model
        
        return {
            'model': model,
            'propensity_scores': propensity_scores,
            'coefficients': model.coef_[0],
            'intercept': model.intercept_[0],
            'standard_errors': se[1:],  # Exclude intercept
            'intercept_se': se[0],
            'pseudo_r2': pseudo_r2,
            'aic': aic,
            'bic': bic,
            'n_obs': n,
            'n_covariates': len(self.covariates)
        }
    
    def estimate_random_forest(self, n_estimators: int = 100,
                              max_depth: Optional[int] = None,
                              min_samples_split: int = 2,
                              min_samples_leaf: int = 1) -> Dict:
        """
        Estimate propensity score using random forest
        
        Based on McCaffrey et al. (2004) "Propensity Score Estimation with Boosted Regression"
        
        Parameters:
        -----------
        n_estimators : int
            Number of trees in the forest
        max_depth : int, optional
            Maximum depth of trees
        min_samples_split : int
            Minimum samples to split a node
        min_samples_leaf : int
            Minimum samples in a leaf
            
        Returns:
        --------
        dict
            Estimation results
        """
        # Fit random forest
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        model.fit(self.X, self.y)
        
        # Calculate propensity scores
        propensity_scores = model.predict_proba(self.X)[:, 1]
        
        # Calculate feature importance
        feature_importance = model.feature_importances_
        
        # Cross-validation for model evaluation
        cv_scores = cross_val_score(model, self.X, self.y, cv=5, scoring='roc_auc')
        
        return {
            'model': model,
            'propensity_scores': propensity_scores,
            'feature_importance': feature_importance,
            'cv_scores': cv_scores,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'n_obs': len(self.X),
            'n_covariates': len(self.covariates)
        }
    
    def cross_validate_model(self, method: str = 'logistic', 
                           cv_folds: int = 5) -> Dict:
        """
        Cross-validate propensity score model
        
        Parameters:
        -----------
        method : str
            Estimation method ('logistic' or 'random_forest')
        cv_folds : int
            Number of cross-validation folds
            
        Returns:
        --------
        dict
            Cross-validation results
        """
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        cv_scores = []
        cv_auc = []
        
        for train_idx, val_idx in kf.split(self.X):
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]
            
            if method == 'logistic':
                model = LogisticRegression(random_state=42)
            elif method == 'random_forest':
                model = RandomForestClassifier(random_state=42)
            else:
                raise ValueError("Method must be 'logistic' or 'random_forest'")
            
            model.fit(X_train, y_train)
            
            # Calculate scores
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            # Accuracy
            accuracy = np.mean(y_pred == y_val)
            cv_scores.append(accuracy)
            
            # AUC
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_val, y_pred_proba)
            cv_auc.append(auc)
        
        return {
            'accuracy_mean': np.mean(cv_scores),
            'accuracy_std': np.std(cv_scores),
            'auc_mean': np.mean(cv_auc),
            'auc_std': np.std(cv_auc),
            'cv_folds': cv_folds
        }
    
    def check_common_support(self, propensity_scores: np.ndarray,
                           min_propensity: float = 0.1,
                           max_propensity: float = 0.9) -> Dict:
        """
        Check common support assumption
        
        Based on Imbens & Wooldridge (2009, p. 26-27)
        
        Parameters:
        -----------
        propensity_scores : np.ndarray
            Estimated propensity scores
        min_propensity : float
            Minimum propensity score threshold
        max_propensity : float
            Maximum propensity score threshold
            
        Returns:
        --------
        dict
            Common support analysis
        """
        # Get propensity scores by treatment status
        treated_scores = propensity_scores[self.y == 1]
        control_scores = propensity_scores[self.y == 0]
        
        # Calculate overlap
        min_treated = np.min(treated_scores)
        max_treated = np.max(treated_scores)
        min_control = np.min(control_scores)
        max_control = np.max(control_scores)
        
        # Common support region
        common_min = max(min_treated, min_control)
        common_max = min(max_treated, max_control)
        
        # Count observations in common support
        treated_in_support = np.sum((treated_scores >= common_min) & 
                                   (treated_scores <= common_max))
        control_in_support = np.sum((control_scores >= common_min) & 
                                   (control_scores <= common_max))
        
        # Calculate overlap statistics
        overlap_ratio = (treated_in_support + control_in_support) / len(propensity_scores)
        
        return {
            'treated_min': min_treated,
            'treated_max': max_treated,
            'control_min': min_control,
            'control_max': max_control,
            'common_min': common_min,
            'common_max': common_max,
            'treated_in_support': treated_in_support,
            'control_in_support': control_in_support,
            'overlap_ratio': overlap_ratio,
            'sufficient_overlap': overlap_ratio > 0.5
        }
    
    def plot_propensity_distribution(self, propensity_scores: np.ndarray,
                                   bins: int = 50) -> None:
        """
        Plot propensity score distributions by treatment status
        
        Parameters:
        -----------
        propensity_scores : np.ndarray
            Estimated propensity scores
        bins : int
            Number of histogram bins
        """
        import matplotlib.pyplot as plt
        
        treated_scores = propensity_scores[self.y == 1]
        control_scores = propensity_scores[self.y == 0]
        
        plt.figure(figsize=(12, 6))
        
        # Plot histograms
        plt.hist(control_scores, bins=bins, alpha=0.7, label='Control', 
                color='steelblue', density=True)
        plt.hist(treated_scores, bins=bins, alpha=0.7, label='Treated', 
                color='red', density=True)
        
        plt.xlabel('Propensity Score')
        plt.ylabel('Density')
        plt.title('Propensity Score Distributions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def summary(self, propensity_scores: np.ndarray) -> pd.DataFrame:
        """
        Summary statistics of propensity scores
        
        Parameters:
        -----------
        propensity_scores : np.ndarray
            Estimated propensity scores
            
        Returns:
        --------
        pd.DataFrame
            Summary statistics
        """
        treated_scores = propensity_scores[self.y == 1]
        control_scores = propensity_scores[self.y == 0]
        
        summary_data = {
            'Group': ['Treated', 'Control', 'Overall'],
            'N': [len(treated_scores), len(control_scores), len(propensity_scores)],
            'Mean': [np.mean(treated_scores), np.mean(control_scores), np.mean(propensity_scores)],
            'Std': [np.std(treated_scores), np.std(control_scores), np.std(propensity_scores)],
            'Min': [np.min(treated_scores), np.min(control_scores), np.min(propensity_scores)],
            'Max': [np.max(treated_scores), np.max(control_scores), np.max(propensity_scores)],
            'Median': [np.median(treated_scores), np.median(control_scores), np.median(propensity_scores)]
        }
        
        return pd.DataFrame(summary_data)