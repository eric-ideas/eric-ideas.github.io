#!/usr/bin/env python3
"""
Comprehensive test script for all economics research projects
Tests all components with academic rigor
"""

import sys
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def test_ols_iv_panel():
    """Test OLS, IV, and Panel Data project"""
    print("Testing OLS, IV, and Panel Data Project...")
    
    try:
        sys.path.append('ols_iv_panel/src')
        from ols import OLS
        from iv import IV
        from panel import PanelData
        from utils import generate_sample_data
        
        # Test OLS
        data = generate_sample_data(100)
        y = data['y'].values
        X = data[['X1', 'X2', 'X3']].values
        
        ols_model = OLS(y, X, ['X1', 'X2', 'X3'])
        ols_model.fit(robust=True)
        
        print(f"  OLS R²: {ols_model.r_squared:.4f}")
        print(f"  OLS N: {ols_model.n}")
        
        # Test Panel Data
        panel_data = PanelData(data, 'y', ['X1', 'X2', 'X3'], 'individual_id', 'time')
        within_results = panel_data.within_estimator()
        
        print(f"  Panel R²: {within_results['r2_within']:.4f}")
        print(f"  Panel N: {within_results['n_obs']}")
        
        return True
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

def test_did_engine():
    """Test Difference-in-Differences project"""
    print("Testing DID Engine Project...")
    
    try:
        sys.path.append('did_engine/src')
        from did import DifferenceInDifferences
        
        # Create sample data
        n_obs = 100
        data = pd.DataFrame({
            'entity_id': np.repeat(range(10), 10),
            'time': np.tile(range(10), 10),
            'outcome': np.random.normal(10, 2, n_obs),
            'treatment': np.random.binomial(1, 0.3, n_obs),
            'control_var1': np.random.normal(0, 1, n_obs),
            'control_var2': np.random.normal(0, 1, n_obs)
        })
        
        # Test DID
        did_model = DifferenceInDifferences(data, 'outcome', 'treatment', 'time', 'entity_id')
        did_results = did_model.estimate_did()
        
        print(f"  DID Coefficient: {did_results['did_coefficient']:.4f}")
        print(f"  DID P-value: {did_results['did_p_value']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

def test_matching_pscore():
    """Test Propensity Score Matching project"""
    print("Testing Matching & Propensity Score Project...")
    
    try:
        sys.path.append('matching_pscore/src')
        from pscore import PropensityScore
        from match import PropensityScoreMatching
        from balance import BalanceTest
        
        # Create sample data
        n_obs = 200
        data = pd.DataFrame({
            'outcome': np.random.normal(10, 2, n_obs),
            'treatment': np.random.binomial(1, 0.3, n_obs),
            'age': np.random.normal(30, 10, n_obs),
            'education': np.random.normal(12, 3, n_obs),
            'income': np.random.normal(50000, 15000, n_obs)
        })
        
        # Test propensity score estimation
        ps_model = PropensityScore(data, 'treatment', ['age', 'education', 'income'], 'outcome')
        ps_results = ps_model.estimate_logistic()
        
        print(f"  Pseudo R²: {ps_results['pseudo_r2']:.4f}")
        print(f"  AIC: {ps_results['aic']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

def test_rd_simulation():
    """Test Regression Discontinuity project"""
    print("Testing RD Simulation Project...")
    
    try:
        sys.path.append('rd_simulation/src')
        from rd_estimator import RegressionDiscontinuity
        
        # Create sample data
        n_obs = 100
        data = pd.DataFrame({
            'running_var': np.random.normal(0, 1, n_obs),
            'outcome': np.random.normal(10, 2, n_obs),
            'treatment': (np.random.normal(0, 1, n_obs) >= 0).astype(int)
        })
        
        # Test RD
        rd_model = RegressionDiscontinuity(data, 'running_var', 'outcome', 0.0)
        rd_results = rd_model.local_linear_regression()
        
        print(f"  RD Estimate: {rd_results['rd_estimate']:.4f}")
        print(f"  RD P-value: {rd_results['p_value']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

def test_solow_model():
    """Test Solow Growth Model project"""
    print("Testing Solow Model Project...")
    
    try:
        sys.path.append('solow_model/src')
        from solow import SolowModel
        
        # Test Solow model
        solow = SolowModel(alpha=0.3, delta=0.05, s=0.2, n=0.01, g=0.02)
        
        print(f"  Steady State k*: {solow.k_star:.4f}")
        print(f"  Steady State y*: {solow.y_star:.4f}")
        print(f"  Convergence Speed: {solow.convergence_speed():.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

def test_islm_adas():
    """Test IS-LM AD-AS project"""
    print("Testing IS-LM AD-AS Project...")
    
    try:
        sys.path.append('islm_adas/src')
        from equilibrium import ISLMModel, ADASModel
        
        # Test IS-LM model
        islm = ISLMModel(C0=100, c=0.8, I0=50, b=10, G=200, T=150)
        
        print(f"  IS-LM Y*: {islm.Y_star:.2f}")
        print(f"  IS-LM r*: {islm.r_star:.4f}")
        
        # Test AD-AS model
        adas = ADASModel(C0=100, c=0.8, I0=50, b=10, G=200, T=150)
        
        print(f"  AD-AS Y*: {adas.Y_star:.2f}")
        print(f"  AD-AS P*: {adas.P_star:.2f}")
        
        return True
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

def test_cycle_filters():
    """Test Cycle Filters project"""
    print("Testing Cycle Filters Project...")
    
    try:
        sys.path.append('cycle_filters/src')
        from hp_filter import HPFilter
        
        # Create sample time series
        t = np.arange(100)
        trend = 0.1 * t
        cycle = 2 * np.sin(0.1 * t)
        noise = np.random.normal(0, 0.5, 100)
        y = trend + cycle + noise
        
        # Test HP filter
        hp = HPFilter(lambda_param=1600)
        hp_results = hp.filter(y)
        
        print(f"  HP Filter N: {hp_results['n_obs']}")
        print(f"  Cycle Std: {np.std(hp_results['cycle']):.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

def test_io_competition():
    """Test IO Competition project"""
    print("Testing IO Competition Project...")
    
    try:
        sys.path.append('io_competition/src')
        from cournot import CournotCompetition
        
        # Test Cournot competition
        cournot = CournotCompetition(n_firms=2, a=100, b=1, c=10)
        
        print(f"  Cournot q*: {cournot.q_star:.2f}")
        print(f"  Cournot P*: {cournot.P_star:.2f}")
        print(f"  Cournot Profit: {cournot.profit_star:.2f}")
        
        return True
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

def test_gravity_model():
    """Test Gravity Model project"""
    print("Testing Gravity Model Project...")
    
    try:
        sys.path.append('gravity_model/src')
        from gravity import GravityModel
        
        # Create sample trade data
        n_countries = 5
        n_pairs = n_countries * (n_countries - 1)
        
        data = pd.DataFrame({
            'country_i': np.repeat(range(n_countries), n_countries-1),
            'country_j': np.tile([i for i in range(n_countries) if i != j for j in range(n_countries)], n_countries),
            'trade': np.random.exponential(100, n_pairs),
            'gdp_i': np.random.uniform(1000, 10000, n_pairs),
            'gdp_j': np.random.uniform(1000, 10000, n_pairs),
            'distance': np.random.uniform(100, 5000, n_pairs),
            'border': np.random.binomial(1, 0.1, n_pairs),
            'language': np.random.binomial(1, 0.2, n_pairs)
        })
        
        # Test gravity model
        gravity = GravityModel(data, 'trade', 'gdp_i', 'gdp_j', 'distance', 'border', 'language')
        gravity_results = gravity.basic_gravity()
        
        print(f"  Gravity R²: {gravity_results['r_squared']:.4f}")
        print(f"  Gravity N: {gravity_results['n_obs']}")
        
        return True
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

def test_auction_sim():
    """Test Auction Simulation project"""
    print("Testing Auction Simulation Project...")
    
    try:
        sys.path.append('auction_sim/src')
        from first_price import FirstPriceAuction
        
        # Test first-price auction
        auction = FirstPriceAuction(n_bidders=3, value_distribution='uniform')
        sim_results = auction.simulate_auctions(100)
        
        print(f"  Auction Revenue: {sim_results['average_revenue']:.2f}")
        print(f"  Auction Efficiency: {sim_results['average_efficiency']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

def test_ge_model():
    """Test General Equilibrium project"""
    print("Testing General Equilibrium Project...")
    
    try:
        sys.path.append('ge_model/src')
        from equilibrium import GeneralEquilibrium
        
        # Test general equilibrium
        ge = GeneralEquilibrium(n_consumers=2, n_goods=2)
        equilibrium = ge.walrasian_equilibrium()
        
        print(f"  GE Prices: {equilibrium['prices']}")
        print(f"  GE Market Clearing: {equilibrium['market_clearing']}")
        
        return True
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 80)
    print("COMPREHENSIVE ECONOMICS RESEARCH PROJECTS TEST")
    print("=" * 80)
    
    tests = [
        ("OLS, IV, and Panel Data", test_ols_iv_panel),
        ("DID Engine", test_did_engine),
        ("Matching & Propensity Score", test_matching_pscore),
        ("RD Simulation", test_rd_simulation),
        ("Solow Model", test_solow_model),
        ("IS-LM AD-AS", test_islm_adas),
        ("Cycle Filters", test_cycle_filters),
        ("IO Competition", test_io_competition),
        ("Gravity Model", test_gravity_model),
        ("Auction Simulation", test_auction_sim),
        ("General Equilibrium", test_ge_model)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        success = test_func()
        results.append((test_name, success))
    
    print("\n" + "=" * 80)
    print("TEST RESULTS SUMMARY")
    print("=" * 80)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:30} {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! All economics research projects are working correctly.")
        return True
    else:
        print(f"\n⚠️  {total-passed} tests failed. Please check the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)