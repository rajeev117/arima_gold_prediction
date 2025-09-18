"""
Test Suite for Gold Price Prediction
===================================

Simple test script to validate the ARIMA implementation.
"""

import unittest
import sys
import pandas as pd
import numpy as np
from gold_price_prediction import GoldPricePredictor


class TestGoldPricePredictor(unittest.TestCase):
    """Test cases for the GoldPricePredictor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.predictor = GoldPricePredictor(symbol='GC=F', period='1y')
    
    def test_data_loading(self):
        """Test data loading functionality."""
        print("\n🧪 Testing data loading...")
        data = self.predictor.load_data()
        
        # Verify data is loaded
        self.assertIsNotNone(data)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)
        
        # Verify required columns exist
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            self.assertIn(col, data.columns)
        
        # Verify data types and values
        self.assertTrue(data['Close'].dtype in [np.float64, np.float32])
        self.assertTrue(all(data['Close'] > 0))
        
        print(f"✅ Data loading test passed - {len(data)} data points loaded")
    
    def test_stationarity_check(self):
        """Test stationarity testing functionality."""
        print("\n🧪 Testing stationarity check...")
        self.predictor.load_data()
        
        # Test stationarity check
        result = self.predictor.check_stationarity()
        
        # Verify result structure
        self.assertIsInstance(result, dict)
        required_keys = ['adf_statistic', 'p_value', 'critical_values', 'is_stationary']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Verify data types
        self.assertIsInstance(result['adf_statistic'], float)
        self.assertIsInstance(result['p_value'], float)
        self.assertIsInstance(bool(result['is_stationary']), bool)
        
        print("✅ Stationarity check test passed")
    
    def test_make_stationary(self):
        """Test differencing functionality."""
        print("\n🧪 Testing make stationary...")
        self.predictor.load_data()
        
        # Test making series stationary
        stationary_series = self.predictor.make_stationary()
        
        # Verify result
        self.assertIsNotNone(stationary_series)
        self.assertIsInstance(stationary_series, pd.Series)
        self.assertGreater(len(stationary_series), 0)
        
        # Verify no infinite or NaN values  
        self.assertFalse(np.isinf(stationary_series).any())
        self.assertFalse(stationary_series.isna().all())
        
        print("✅ Make stationary test passed")
    
    def test_arima_fitting(self):
        """Test ARIMA model fitting."""
        print("\n🧪 Testing ARIMA model fitting...")
        self.predictor.load_data()
        
        # Test model fitting with simple order
        fitted_model = self.predictor.fit_arima_model(order=(1, 1, 1))
        
        # Verify model was fitted
        self.assertIsNotNone(fitted_model)
        
        # Verify diagnostics were stored
        self.assertIn('fitted_model', self.predictor.diagnostics)
        self.assertIn('aic', self.predictor.diagnostics)
        self.assertIn('bic', self.predictor.diagnostics)
        
        # Verify AIC and BIC are reasonable
        self.assertIsInstance(self.predictor.diagnostics['aic'], float)
        self.assertIsInstance(self.predictor.diagnostics['bic'], float)
        self.assertGreater(self.predictor.diagnostics['aic'], 0)
        self.assertGreater(self.predictor.diagnostics['bic'], 0)
        
        print("✅ ARIMA fitting test passed")
    
    def test_parameter_search(self):
        """Test automated parameter search."""
        print("\n🧪 Testing parameter search...")
        self.predictor.load_data()
        
        # Test parameter search with limited range for speed
        best_order = self.predictor.find_best_arima_order(max_p=2, max_d=1, max_q=2)
        
        # Verify result
        self.assertIsNotNone(best_order)
        self.assertIsInstance(best_order, tuple)
        self.assertEqual(len(best_order), 3)
        
        # Verify parameters are within expected range
        p, d, q = best_order
        self.assertGreaterEqual(p, 0)
        self.assertLessEqual(p, 2)
        self.assertGreaterEqual(d, 0)
        self.assertLessEqual(d, 1)
        self.assertGreaterEqual(q, 0)
        self.assertLessEqual(q, 2)
        
        print(f"✅ Parameter search test passed - Best order: {best_order}")
    
    def test_forecast_generation(self):
        """Test forecast generation."""
        print("\n🧪 Testing forecast generation...")
        self.predictor.load_data()
        
        # Fit a simple model first
        self.predictor.fit_arima_model(order=(1, 1, 1))
        
        # Generate forecast
        forecast = self.predictor.generate_forecast(steps=7)
        
        # Verify forecast structure
        self.assertIsNotNone(forecast)
        self.assertIsInstance(forecast, dict)
        
        required_keys = ['forecast', 'lower_ci', 'upper_ci', 'forecast_dates']
        for key in required_keys:
            self.assertIn(key, forecast)
        
        # Verify forecast data
        self.assertEqual(len(forecast['forecast']), 7)
        self.assertEqual(len(forecast['lower_ci']), 7)
        self.assertEqual(len(forecast['upper_ci']), 7)
        self.assertEqual(len(forecast['forecast_dates']), 7)
        
        # Verify all forecasts are positive (gold prices should be positive)
        self.assertTrue(all(forecast['forecast'] > 0))
        self.assertTrue(all(forecast['lower_ci'] > 0))
        self.assertTrue(all(forecast['upper_ci'] > 0))
        
        # Verify confidence intervals are reasonable
        self.assertTrue(all(forecast['lower_ci'] <= forecast['forecast']))
        self.assertTrue(all(forecast['forecast'] <= forecast['upper_ci']))
        
        print("✅ Forecast generation test passed")
    
    def test_performance_metrics(self):
        """Test performance evaluation."""
        print("\n🧪 Testing performance metrics...")
        self.predictor.load_data()
        
        # Fit model and calculate metrics
        self.predictor.fit_arima_model(order=(1, 1, 1))
        metrics = self.predictor.calculate_performance_metrics(test_size=0.3)
        
        # Verify metrics structure
        self.assertIsNotNone(metrics)
        self.assertIsInstance(metrics, dict)
        
        required_keys = ['mse', 'rmse', 'mae', 'mape']
        for key in required_keys:
            self.assertIn(key, metrics)
            self.assertIsInstance(metrics[key], float)
            self.assertGreaterEqual(metrics[key], 0)
        
        # Verify MAPE is a reasonable percentage
        self.assertLess(metrics['mape'], 100)  # Should be less than 100%
        
        print(f"✅ Performance metrics test passed - MAPE: {metrics['mape']:.2f}%")


def run_tests():
    """Run all tests and display results."""
    print("🚀 Starting Gold Price Prediction Tests")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestGoldPricePredictor)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=0, stream=open('/dev/null', 'w'))
    result = runner.run(suite)
    
    # Display summary
    print(f"\n📊 TEST SUMMARY:")
    print(f"   Tests Run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"   - {test}: {traceback.split(chr(10))[-2]}")
    
    if result.errors:
        print(f"\n💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"   - {test}: {traceback.split(chr(10))[-2]}")
    
    if not result.failures and not result.errors:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"   The ARIMA gold price prediction system is working correctly.")
    else:
        print(f"\n⚠️ SOME TESTS FAILED")
        print(f"   Please check the implementation for issues.")
    
    print("=" * 50)
    return len(result.failures) + len(result.errors) == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)