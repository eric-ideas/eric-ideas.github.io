#!/usr/bin/env python3
"""
Comprehensive test script for propensity score matching
Tests all components with academic rigor
"""

import sys
import os
sys.path.append('src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pscore import PropensityScore
from match import PropensityScoreMatching
from balance import BalanceTest
from simulate import simulate_treatment_data, monte_carlo_simulation

def test_propensity_score_estimation():
    """Test propensity score estimation methods"""
    print("Testing Propensity Score Estimation...")
    
    # Load data
    data = pd.read_csv('data/treatment_data.csv')
    covariates = ['age', 'education', 'income', 'health_score']
    
    # Test logistic regression
    ps_logistic = PropensityScore(data, 'treatment', covariates, 'outcome')
    results_logistic = ps_logistic.estimate_logistic()
    
    print(f"Logistic Regression Results:")
    print(f"  Pseudo R²: {results_logistic['pseudo_r2']:.4f}")
    print(f"  AIC: {results_logistic['aic']:.2f}")
    print(f"  BIC: {results_logistic['bic']:.2f}")
    
    # Test random forest
    ps_rf = PropensityScore(data, 'treatment', covariates, 'outcome')
    results_rf = ps_rf.estimate_random_forest()
    
    print(f"Random Forest Results:")
    print(f"  CV AUC: {results_rf['cv_mean']:.4f} ± {results_rf['cv_std']:.4f}")
    
    # Test common support
    support_check = ps_logistic.check_common_support(results_logistic['propensity_scores'])
    print(f"Common Support:")
    print(f"  Overlap Ratio: {support_check['overlap_ratio']:.4f}")
    print(f"  Sufficient Overlap: {support_check['sufficient_overlap']}")
    
    return results_logistic, results_rf

def test_matching_algorithms():
    """Test different matching algorithms"""
    print("\nTesting Matching Algorithms...")
    
    # Load data
    data = pd.read_csv('data/treatment_data.csv')
    covariates = ['age', 'education', 'income', 'health_score']
    
    # Estimate propensity scores
    ps = PropensityScore(data, 'treatment', covariates, 'outcome')
    ps_results = ps.estimate_logistic()
    propensity_scores = ps_results['propensity_scores']
    
    # Test matching
    matcher = PropensityScoreMatching(data, 'treatment', propensity_scores, 'outcome')
    
    # Test different methods
    methods = {
        'Nearest Neighbor': matcher.nearest_neighbor_matching(),
        'Caliper': matcher.caliper_matching(),
        'Kernel': matcher.kernel_matching(),
        'Radius': matcher.radius_matching()
    }
    
    print("Matching Results:")
    for method_name, results in methods.items():
        print(f"  {method_name}:")
        print(f"    ATT: {results['att']:.4f}")
        print(f"    SE: {results['se_att']:.4f}")
        print(f"    T-stat: {results['t_statistic']:.4f}")
        print(f"    P-value: {results['p_value']:.4f}")
        print(f"    N Matched: {results['n_matched']}")
    
    return methods

def test_balance_assessment():
    """Test balance assessment methods"""
    print("\nTesting Balance Assessment...")
    
    # Load data
    data = pd.read_csv('data/treatment_data.csv')
    covariates = ['age', 'education', 'income', 'health_score']
    
    # Estimate propensity scores
    ps = PropensityScore(data, 'treatment', covariates, 'outcome')
    ps_results = ps.estimate_logistic()
    propensity_scores = ps_results['propensity_scores']
    
    # Test matching
    matcher = PropensityScoreMatching(data, 'treatment', propensity_scores, 'outcome')
    match_results = matcher.nearest_neighbor_matching()
    
    # Test balance
    balance_tester = BalanceTest(data, 'treatment', covariates, 
                                match_results['matched_treated_indices'])
    
    # Before matching balance
    before_balance = balance_tester.standardized_differences(matched=False)
    print("Before Matching Balance:")
    print(f"  Average Std. Diff: {np.mean(np.abs(before_balance['Std_Diff'])):.4f}")
    print(f"  Balanced Covariates: {sum(before_balance['Balanced'])}/{len(covariates)}")
    
    # After matching balance
    after_balance = balance_tester.standardized_differences(matched=True)
    print("After Matching Balance:")
    print(f"  Average Std. Diff: {np.mean(np.abs(after_balance['Std_Diff'])):.4f}")
    print(f"  Balanced Covariates: {sum(after_balance['Balanced'])}/{len(covariates)}")
    
    # Overall balance test
    overall_test = balance_tester.overall_balance_test(matched=True)
    print(f"Overall Balance Test:")
    print(f"  F-statistic: {overall_test['f_statistic']:.4f}")
    print(f"  P-value: {overall_test['p_value']:.4f}")
    print(f"  Balanced: {overall_test['balanced']}")
    
    return before_balance, after_balance, overall_test

def test_simulation():
    """Test simulation functions"""
    print("\nTesting Simulation Functions...")
    
    # Generate simulated data
    sim_data = simulate_treatment_data(n_obs=500, treatment_effect=2.0)
    print(f"Simulated Data:")
    print(f"  N: {len(sim_data)}")
    print(f"  Treatment Rate: {sim_data['treatment'].mean():.4f}")
    print(f"  True Effect: 2.0")
    
    # Estimate treatment effect
    treated_outcomes = sim_data[sim_data['treatment'] == 1]['outcome']
    control_outcomes = sim_data[sim_data['treatment'] == 0]['outcome']
    naive_estimate = treated_outcomes.mean() - control_outcomes.mean()
    print(f"  Naive Estimate: {naive_estimate:.4f}")
    print(f"  Bias: {naive_estimate - 2.0:.4f}")
    
    # Monte Carlo simulation
    mc_results = monte_carlo_simulation(n_simulations=50, n_obs=200, true_effect=2.0)
    print(f"Monte Carlo Results:")
    print(f"  Mean Estimate: {mc_results['mean_estimate']:.4f}")
    print(f"  Bias: {mc_results['bias']:.4f}")
    print(f"  RMSE: {mc_results['rmse']:.4f}")
    print(f"  Coverage: {mc_results['coverage']:.4f}")
    
    return sim_data, mc_results

def test_robustness():
    """Test robustness of matching methods"""
    print("\nTesting Robustness...")
    
    # Generate data with different scenarios
    scenarios = {
        'Baseline': {'n_obs': 500, 'treatment_effect': 2.0},
        'Small Sample': {'n_obs': 100, 'treatment_effect': 2.0},
        'Large Effect': {'n_obs': 500, 'treatment_effect': 5.0},
        'Heterogeneous': {'n_obs': 500, 'treatment_effect': 2.0, 'heterogeneity': 1.0}
    }
    
    results = {}
    
    for scenario_name, params in scenarios.items():
        print(f"  {scenario_name}:")
        
        # Generate data
        if 'heterogeneity' in params:
            from simulate import simulate_heterogeneous_effects
            data = simulate_heterogeneous_effects(**params)
        else:
            data = simulate_treatment_data(**params)
        
        # Estimate propensity scores
        covariates = [col for col in data.columns if col.startswith('X')]
        ps = PropensityScore(data, 'treatment', covariates, 'outcome')
        ps_results = ps.estimate_logistic()
        
        # Test matching
        matcher = PropensityScoreMatching(data, 'treatment', 
                                         ps_results['propensity_scores'], 'outcome')
        match_results = matcher.nearest_neighbor_matching()
        
        print(f"    ATT: {match_results['att']:.4f}")
        print(f"    SE: {match_results['se_att']:.4f}")
        print(f"    N Matched: {match_results['n_matched']}")
        
        results[scenario_name] = match_results
    
    return results

def main():
    """Run all tests"""
    print("=" * 60)
    print("PROPENSITY SCORE MATCHING - COMPREHENSIVE TEST")
    print("=" * 60)
    
    try:
        # Test propensity score estimation
        ps_logistic, ps_rf = test_propensity_score_estimation()
        
        # Test matching algorithms
        match_results = test_matching_algorithms()
        
        # Test balance assessment
        before_balance, after_balance, overall_test = test_balance_assessment()
        
        # Test simulation
        sim_data, mc_results = test_simulation()
        
        # Test robustness
        robustness_results = test_robustness()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
        # Summary statistics
        print("\nSUMMARY:")
        print(f"  Propensity Score Estimation: ✓")
        print(f"  Matching Algorithms: ✓")
        print(f"  Balance Assessment: ✓")
        print(f"  Simulation Functions: ✓")
        print(f"  Robustness Tests: ✓")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)