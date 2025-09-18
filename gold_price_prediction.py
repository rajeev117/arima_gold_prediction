"""
Gold Price Prediction using ARIMA Model
=====================================

This script implements a complete time-series forecasting pipeline for gold price prediction
using ARIMA (AutoRegressive Integrated Moving Average) model.

Features:
- Statistical tests for stationarity (ADF test)
- ACF and PACF analysis for parameter determination
- ARIMA model implementation and optimization
- Model diagnostics and residual analysis
- Performance evaluation with error metrics
- Visualization of predictions and forecasts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime, timedelta
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')


class GoldPricePredictor:
    """
    A comprehensive class for gold price prediction using ARIMA model.
    
    This class handles data collection, preprocessing, statistical analysis,
    model fitting, and prediction visualization.
    """
    
    def __init__(self, symbol='GC=F', period='5y'):
        """
        Initialize the GoldPricePredictor.
        
        Args:
            symbol (str): Yahoo Finance symbol for gold (default: 'GC=F')
            period (str): Data period to fetch (default: '5y')
        """
        self.symbol = symbol
        self.period = period
        self.data = None
        self.model = None
        self.forecast = None
        self.diagnostics = {}
        
    def load_data(self):
        """
        Load historical gold price data from Yahoo Finance.
        
        Returns:
            pd.DataFrame: Historical gold price data
        """
        print(f"Loading gold price data for symbol: {self.symbol}")
        print(f"Period: {self.period}")
        
        try:
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(period=self.period)
            
            if self.data.empty:
                print("No data found. Using sample data instead.")
                self._create_sample_data()
            else:
                print(f"Successfully loaded {len(self.data)} data points")
                print(f"Date range: {self.data.index.min().date()} to {self.data.index.max().date()}")
                
        except Exception as e:
            print(f"Error loading data from Yahoo Finance: {e}")
            print("Using sample data instead.")
            self._create_sample_data()
            
        return self.data
    
    def _create_sample_data(self):
        """Create sample gold price data for demonstration purposes."""
        print("Creating sample gold price data...")
        
        # Create date range for last 3 years
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3*365)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate realistic gold price data with trend and seasonality
        np.random.seed(42)
        n = len(dates)
        
        # Base price around $1800
        base_price = 1800
        
        # Add trend (slight upward)
        trend = np.linspace(0, 200, n)
        
        # Add seasonality (annual cycle)
        seasonality = 50 * np.sin(2 * np.pi * np.arange(n) / 365.25)
        
        # Add random walk component
        random_walk = np.cumsum(np.random.normal(0, 5, n))
        
        # Add daily volatility
        daily_volatility = np.random.normal(0, 10, n)
        
        # Combine all components
        close_prices = base_price + trend + seasonality + random_walk + daily_volatility
        
        # Create OHLC data
        high_prices = close_prices + np.abs(np.random.normal(0, 5, n))
        low_prices = close_prices - np.abs(np.random.normal(0, 5, n))
        open_prices = close_prices + np.random.normal(0, 2, n)
        volume = np.random.lognormal(10, 0.5, n)
        
        self.data = pd.DataFrame({
            'Open': open_prices,
            'High': high_prices,
            'Low': low_prices,
            'Close': close_prices,
            'Volume': volume
        }, index=dates)
        
        print(f"Created sample data with {len(self.data)} data points")
    
    def explore_data(self):
        """
        Perform exploratory data analysis on the gold price data.
        """
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Basic statistics
        print("\nBasic Statistics:")
        print(self.data['Close'].describe())
        
        # Missing values
        print(f"\nMissing values: {self.data['Close'].isnull().sum()}")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Gold Price Exploratory Data Analysis', fontsize=16)
        
        # Time series plot
        axes[0, 0].plot(self.data.index, self.data['Close'], color='gold', linewidth=1.5)
        axes[0, 0].set_title('Gold Price Over Time')
        axes[0, 0].set_ylabel('Price ($)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Distribution
        axes[0, 1].hist(self.data['Close'], bins=50, alpha=0.7, color='gold', edgecolor='black')
        axes[0, 1].set_title('Gold Price Distribution')
        axes[0, 1].set_xlabel('Price ($)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Returns distribution
        returns = self.data['Close'].pct_change().dropna()
        axes[1, 0].hist(returns, bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].set_title('Daily Returns Distribution')
        axes[1, 0].set_xlabel('Returns')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Rolling statistics
        rolling_mean = self.data['Close'].rolling(window=30).mean()
        rolling_std = self.data['Close'].rolling(window=30).std()
        
        axes[1, 1].plot(self.data.index, self.data['Close'], label='Original', alpha=0.7)
        axes[1, 1].plot(rolling_mean.index, rolling_mean, label='30-day Mean', color='red', linewidth=2)
        axes[1, 1].fill_between(rolling_mean.index, 
                                rolling_mean - rolling_std, 
                                rolling_mean + rolling_std, 
                                alpha=0.2, color='red', label='±1 Std Dev')
        axes[1, 1].set_title('Rolling Statistics (30-day window)')
        axes[1, 1].set_ylabel('Price ($)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print(f"\nPrice Range: ${self.data['Close'].min():.2f} - ${self.data['Close'].max():.2f}")
        print(f"Average Daily Return: {returns.mean():.4f} ({returns.mean()*100:.2f}%)")
        print(f"Daily Volatility: {returns.std():.4f} ({returns.std()*100:.2f}%)")
        print(f"Annualized Volatility: {returns.std() * np.sqrt(252):.4f} ({returns.std() * np.sqrt(252)*100:.2f}%)")
    
    def check_stationarity(self, series=None, title="Gold Price"):
        """
        Check stationarity of time series using Augmented Dickey-Fuller test.
        
        Args:
            series (pd.Series): Time series data (default: self.data['Close'])
            title (str): Title for the analysis
            
        Returns:
            dict: ADF test results
        """
        if series is None:
            series = self.data['Close']
            
        print(f"\n{'='*50}")
        print(f"STATIONARITY TEST: {title}")
        print(f"{'='*50}")
        
        # Augmented Dickey-Fuller test
        adf_result = adfuller(series.dropna())
        
        print(f"\nAugmented Dickey-Fuller Test Results:")
        print(f"ADF Statistic: {adf_result[0]:.6f}")
        print(f"p-value: {adf_result[1]:.6f}")
        print(f"Critical Values:")
        for key, value in adf_result[4].items():
            print(f"\t{key}: {value:.6f}")
        
        # Interpretation
        if adf_result[1] <= 0.05:
            print(f"\n✅ Result: STATIONARY (p-value ≤ 0.05)")
            print("The time series is stationary - ready for ARIMA modeling")
        else:
            print(f"\n❌ Result: NON-STATIONARY (p-value > 0.05)")
            print("The time series is non-stationary - differencing may be needed")
        
        return {
            'adf_statistic': adf_result[0],
            'p_value': adf_result[1],
            'critical_values': adf_result[4],
            'is_stationary': adf_result[1] <= 0.05
        }
    
    def make_stationary(self, max_diff=2):
        """
        Make the time series stationary through differencing.
        
        Args:
            max_diff (int): Maximum number of differencing operations
            
        Returns:
            pd.Series: Stationary time series
        """
        print(f"\n{'='*50}")
        print("MAKING TIME SERIES STATIONARY")
        print(f"{'='*50}")
        
        series = self.data['Close'].copy()
        
        for d in range(max_diff + 1):
            if d == 0:
                current_series = series
                title = "Original Series"
            else:
                current_series = series.diff(d).dropna()
                title = f"After {d} Differencing"
            
            stationarity_result = self.check_stationarity(current_series, title)
            
            if stationarity_result['is_stationary']:
                print(f"\n✅ Series became stationary after {d} differencing operations")
                self.diagnostics['differencing_order'] = d
                self.diagnostics['stationary_series'] = current_series
                return current_series
        
        print(f"\n⚠️ Warning: Series did not become stationary after {max_diff} differencing operations")
        self.diagnostics['differencing_order'] = max_diff
        self.diagnostics['stationary_series'] = series.diff(max_diff).dropna()
        return self.diagnostics['stationary_series']
    
    def analyze_acf_pacf(self, series=None, lags=40):
        """
        Analyze ACF and PACF plots to determine ARIMA parameters.
        
        Args:
            series (pd.Series): Time series data
            lags (int): Number of lags to analyze
        """
        if series is None:
            series = self.diagnostics.get('stationary_series', self.data['Close'])
        
        print(f"\n{'='*50}")
        print("ACF AND PACF ANALYSIS")
        print(f"{'='*50}")
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # ACF plot
        plot_acf(series.dropna(), lags=lags, ax=axes[0], alpha=0.05)
        axes[0].set_title('Autocorrelation Function (ACF)')
        axes[0].grid(True, alpha=0.3)
        
        # PACF plot
        plot_pacf(series.dropna(), lags=lags, ax=axes[1], alpha=0.05)
        axes[1].set_title('Partial Autocorrelation Function (PACF)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("\nParameter Selection Guidelines:")
        print("- For AR(p): Look at PACF plot - significant lags indicate p value")
        print("- For MA(q): Look at ACF plot - significant lags indicate q value")
        print("- For d: Use the differencing order that made series stationary")
    
    def fit_arima_model(self, order=(1, 1, 1), seasonal_order=None):
        """
        Fit ARIMA model to the gold price data.
        
        Args:
            order (tuple): ARIMA order (p, d, q)
            seasonal_order (tuple): Seasonal ARIMA order (P, D, Q, s)
            
        Returns:
            statsmodels ARIMA model
        """
        print(f"\n{'='*50}")
        print(f"FITTING ARIMA{order} MODEL")
        print(f"{'='*50}")
        
        try:
            # Fit the model
            self.model = ARIMA(self.data['Close'], order=order, seasonal_order=seasonal_order)
            fitted_model = self.model.fit()
            
            print(f"✅ Successfully fitted ARIMA{order} model")
            print(f"\nModel Summary:")
            print(f"AIC: {fitted_model.aic:.2f}")
            print(f"BIC: {fitted_model.bic:.2f}")
            print(f"Log-Likelihood: {fitted_model.llf:.2f}")
            
            # Store model information
            self.diagnostics['model_order'] = order
            self.diagnostics['aic'] = fitted_model.aic
            self.diagnostics['bic'] = fitted_model.bic
            self.diagnostics['log_likelihood'] = fitted_model.llf
            self.diagnostics['fitted_model'] = fitted_model
            
            return fitted_model
            
        except Exception as e:
            print(f"❌ Error fitting ARIMA{order} model: {e}")
            return None
    
    def find_best_arima_order(self, max_p=3, max_d=2, max_q=3):
        """
        Find the best ARIMA order using AIC criterion.
        
        Args:
            max_p (int): Maximum AR order
            max_d (int): Maximum differencing order  
            max_q (int): Maximum MA order
            
        Returns:
            tuple: Best ARIMA order
        """
        print(f"\n{'='*50}")
        print("SEARCHING FOR OPTIMAL ARIMA PARAMETERS")
        print(f"{'='*50}")
        
        best_aic = float('inf')
        best_order = None
        results = []
        
        print(f"Testing combinations: p ∈ [0,{max_p}], d ∈ [0,{max_d}], q ∈ [0,{max_q}]")
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(self.data['Close'], order=(p, d, q))
                        fitted_model = model.fit()
                        
                        aic = fitted_model.aic
                        results.append((p, d, q, aic))
                        
                        if aic < best_aic:
                            best_aic = aic
                            best_order = (p, d, q)
                        
                        print(f"ARIMA({p},{d},{q}): AIC = {aic:.2f}")
                        
                    except Exception as e:
                        print(f"ARIMA({p},{d},{q}): Failed - {str(e)[:50]}...")
                        continue
        
        print(f"\n✅ Best ARIMA order: {best_order} (AIC = {best_aic:.2f})")
        
        # Store results
        self.diagnostics['parameter_search'] = results
        self.diagnostics['best_order'] = best_order
        self.diagnostics['best_aic'] = best_aic
        
        return best_order
    
    def model_diagnostics(self):
        """
        Perform comprehensive model diagnostics.
        """
        if 'fitted_model' not in self.diagnostics:
            print("❌ No fitted model found. Please fit a model first.")
            return
        
        fitted_model = self.diagnostics['fitted_model']
        
        print(f"\n{'='*50}")
        print("MODEL DIAGNOSTICS")
        print(f"{'='*50}")
        
        # Get residuals
        residuals = fitted_model.resid
        
        # Create diagnostic plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('ARIMA Model Diagnostics', fontsize=16)
        
        # 1. Residuals plot
        axes[0, 0].plot(residuals, color='blue', alpha=0.7)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[0, 0].set_title('Residuals over Time')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residuals histogram
        axes[0, 1].hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].axvline(residuals.mean(), color='red', linestyle='--', 
                          label=f'Mean: {residuals.mean():.4f}')
        axes[0, 1].set_title('Residuals Distribution')
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. ACF of residuals
        plot_acf(residuals, lags=20, ax=axes[1, 1], alpha=0.05)
        axes[1, 1].set_title('ACF of Residuals')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Statistical tests
        print(f"\nResidual Analysis:")
        print(f"Mean of residuals: {residuals.mean():.6f}")
        print(f"Std of residuals: {residuals.std():.6f}")
        print(f"Skewness: {residuals.skew():.6f}")
        print(f"Kurtosis: {residuals.kurtosis():.6f}")
        
        # Ljung-Box test for residual autocorrelation
        try:
            lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
            print(f"\nLjung-Box Test (H0: residuals are independently distributed):")
            print(f"Test statistic: {lb_test['lb_stat'].iloc[-1]:.4f}")
            print(f"p-value: {lb_test['lb_pvalue'].iloc[-1]:.4f}")
            
            if lb_test['lb_pvalue'].iloc[-1] > 0.05:
                print("✅ Residuals appear to be independently distributed")
            else:
                print("❌ Residuals show signs of autocorrelation")
                
        except Exception as e:
            print(f"Could not perform Ljung-Box test: {e}")
        
        # Store diagnostic results
        self.diagnostics['residuals'] = residuals
        self.diagnostics['residual_mean'] = residuals.mean()
        self.diagnostics['residual_std'] = residuals.std()
    
    def generate_forecast(self, steps=30):
        """
        Generate forecasts using the fitted ARIMA model.
        
        Args:
            steps (int): Number of steps to forecast
            
        Returns:
            dict: Forecast results including predictions and confidence intervals
        """
        if 'fitted_model' not in self.diagnostics:
            print("❌ No fitted model found. Please fit a model first.")
            return None
        
        fitted_model = self.diagnostics['fitted_model']
        
        print(f"\n{'='*50}")
        print(f"GENERATING {steps}-STEP FORECAST")
        print(f"{'='*50}")
        
        try:
            # Generate forecast
            forecast_result = fitted_model.forecast(steps=steps)
            conf_int = fitted_model.get_forecast(steps=steps).conf_int()
            
            # Create forecast dates
            last_date = self.data.index[-1]
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                         periods=steps, freq='D')
            
            # Store forecast results
            self.forecast = {
                'forecast': pd.Series(forecast_result, index=forecast_dates),
                'lower_ci': pd.Series(conf_int.iloc[:, 0], index=forecast_dates),
                'upper_ci': pd.Series(conf_int.iloc[:, 1], index=forecast_dates),
                'last_actual': self.data['Close'].iloc[-1],
                'forecast_dates': forecast_dates
            }
            
            print(f"✅ Successfully generated {steps}-day forecast")
            print(f"Forecast range: {forecast_dates[0].date()} to {forecast_dates[-1].date()}")
            print(f"\nForecast Summary:")
            print(f"First forecast: ${forecast_result.iloc[0]:.2f}")
            print(f"Last forecast: ${forecast_result.iloc[-1]:.2f}")
            print(f"Average forecast: ${forecast_result.mean():.2f}")
            
            return self.forecast
            
        except Exception as e:
            print(f"❌ Error generating forecast: {e}")
            return None
    
    def calculate_performance_metrics(self, test_size=0.2):
        """
        Calculate performance metrics using train-test split.
        
        Args:
            test_size (float): Proportion of data to use for testing
            
        Returns:
            dict: Performance metrics
        """
        print(f"\n{'='*50}")
        print("PERFORMANCE EVALUATION")
        print(f"{'='*50}")
        
        # Split data
        n = len(self.data)
        train_size = int(n * (1 - test_size))
        
        train_data = self.data['Close'][:train_size]
        test_data = self.data['Close'][train_size:]
        
        print(f"Training data: {len(train_data)} points ({train_data.index[0].date()} to {train_data.index[-1].date()})")
        print(f"Testing data: {len(test_data)} points ({test_data.index[0].date()} to {test_data.index[-1].date()})")
        
        # Fit model on training data
        if 'best_order' in self.diagnostics:
            order = self.diagnostics['best_order']
        else:
            order = (1, 1, 1)
        
        try:
            train_model = ARIMA(train_data, order=order).fit()
            
            # Generate predictions for test period
            predictions = train_model.forecast(steps=len(test_data))
            
            # Calculate metrics
            mse = mean_squared_error(test_data, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(test_data, predictions)
            mape = mean_absolute_percentage_error(test_data, predictions) * 100
            
            # Calculate baseline metrics (naive forecast - last value)
            naive_forecast = [train_data.iloc[-1]] * len(test_data)
            baseline_mse = mean_squared_error(test_data, naive_forecast)
            baseline_rmse = np.sqrt(baseline_mse)
            baseline_mae = mean_absolute_error(test_data, naive_forecast)
            baseline_mape = mean_absolute_percentage_error(test_data, naive_forecast) * 100
            
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'baseline_mse': baseline_mse,
                'baseline_rmse': baseline_rmse,
                'baseline_mae': baseline_mae,
                'baseline_mape': baseline_mape,
                'predictions': predictions,
                'test_data': test_data,
                'improvement_rmse': ((baseline_rmse - rmse) / baseline_rmse) * 100,
                'improvement_mae': ((baseline_mae - mae) / baseline_mae) * 100,
                'improvement_mape': ((baseline_mape - mape) / baseline_mape) * 100
            }
            
            print(f"\n📊 ARIMA Model Performance:")
            print(f"MSE: {mse:.4f}")
            print(f"RMSE: ${rmse:.2f}")
            print(f"MAE: ${mae:.2f}")
            print(f"MAPE: {mape:.2f}%")
            
            print(f"\n📊 Baseline Model (Naive) Performance:")
            print(f"RMSE: ${baseline_rmse:.2f}")
            print(f"MAE: ${baseline_mae:.2f}")
            print(f"MAPE: {baseline_mape:.2f}%")
            
            print(f"\n🎯 Model Improvement over Baseline:")
            print(f"RMSE improvement: {metrics['improvement_rmse']:.1f}%")
            print(f"MAE improvement: {metrics['improvement_mae']:.1f}%")
            print(f"MAPE improvement: {metrics['improvement_mape']:.1f}%")
            
            if metrics['improvement_rmse'] > 0:
                print("✅ ARIMA model outperforms baseline model!")
            else:
                print("❌ ARIMA model underperforms baseline model")
            
            self.diagnostics['performance_metrics'] = metrics
            return metrics
            
        except Exception as e:
            print(f"❌ Error calculating performance metrics: {e}")
            return None
    
    def visualize_results(self, show_forecast=True, last_n_days=180):
        """
        Create comprehensive visualizations of the results.
        
        Args:
            show_forecast (bool): Whether to include forecast in visualization
            last_n_days (int): Number of recent days to show in detail
        """
        print(f"\n{'='*50}")
        print("CREATING VISUALIZATIONS")
        print(f"{'='*50}")
        
        # Create interactive plot using Plotly
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Gold Price Prediction', 'Recent Price Action (Last 180 days)', 
                          'Residuals Analysis', 'Performance Comparison'),
            specs=[[{"colspan": 2}, None],
                   [{"type": "scatter"}, {"type": "bar"}]],
            row_heights=[0.6, 0.4]
        )
        
        # Main price chart
        fig.add_trace(
            go.Scatter(x=self.data.index, y=self.data['Close'], 
                      name='Historical Prices', line=dict(color='gold', width=2)),
            row=1, col=1
        )
        
        # Add forecast if available
        if show_forecast and self.forecast is not None:
            fig.add_trace(
                go.Scatter(x=self.forecast['forecast_dates'], y=self.forecast['forecast'],
                          name='Forecast', line=dict(color='red', width=2, dash='dash')),
                row=1, col=1
            )
            
            # Add confidence interval
            fig.add_trace(
                go.Scatter(x=self.forecast['forecast_dates'], y=self.forecast['upper_ci'],
                          fill=None, mode='lines', line_color='rgba(255,0,0,0)',
                          showlegend=False),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=self.forecast['forecast_dates'], y=self.forecast['lower_ci'],
                          fill='tonexty', mode='lines', line_color='rgba(255,0,0,0)',
                          name='95% Confidence Interval', fillcolor='rgba(255,0,0,0.2)'),
                row=1, col=1
            )
        
        # Recent price action
        recent_data = self.data.tail(last_n_days)
        fig.add_trace(
            go.Scatter(x=recent_data.index, y=recent_data['Close'],
                      name='Recent Prices', line=dict(color='blue', width=2)),
            row=2, col=1
        )
        
        # Residuals (if available)
        if 'residuals' in self.diagnostics:
            residuals = self.diagnostics['residuals']
            fig.add_trace(
                go.Scatter(x=residuals.index, y=residuals,
                          name='Residuals', mode='lines', line=dict(color='green')),
                row=2, col=1
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)
        
        # Performance comparison (if available)
        if 'performance_metrics' in self.diagnostics:
            metrics = self.diagnostics['performance_metrics']
            
            categories = ['RMSE', 'MAE', 'MAPE']
            arima_values = [metrics['rmse'], metrics['mae'], metrics['mape']]
            baseline_values = [metrics['baseline_rmse'], metrics['baseline_mae'], metrics['baseline_mape']]
            
            fig.add_trace(
                go.Bar(x=categories, y=arima_values, name='ARIMA Model', marker_color='lightblue'),
                row=2, col=2
            )
            fig.add_trace(
                go.Bar(x=categories, y=baseline_values, name='Baseline Model', marker_color='lightcoral'),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Gold Price Prediction Analysis Dashboard",
            title_x=0.5,
            showlegend=True
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=2, col=1)
        fig.update_xaxes(title_text="Metric", row=2, col=2)
        fig.update_yaxes(title_text="Value", row=2, col=2)
        
        fig.show()
        
        # Also create matplotlib plots for detailed analysis
        self._create_detailed_plots()
    
    def _create_detailed_plots(self):
        """Create detailed matplotlib visualizations."""
        
        fig, axes = plt.subplots(3, 2, figsize=(18, 15))
        fig.suptitle('Detailed Gold Price Analysis', fontsize=16)
        
        # 1. Price decomposition
        try:
            decomposition = seasonal_decompose(self.data['Close'], model='additive', period=252)
            
            axes[0, 0].plot(self.data.index, decomposition.trend, label='Trend', color='red', linewidth=2)
            axes[0, 0].set_title('Trend Component')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
            
            axes[0, 1].plot(self.data.index, decomposition.seasonal, label='Seasonal', color='green')
            axes[0, 1].set_title('Seasonal Component')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
        except:
            axes[0, 0].plot(self.data.index, self.data['Close'], color='gold')
            axes[0, 0].set_title('Gold Price Time Series')
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].hist(self.data['Close'], bins=50, alpha=0.7, color='gold')
            axes[0, 1].set_title('Price Distribution')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 2. Forecast visualization
        if self.forecast is not None:
            recent_data = self.data['Close'].tail(90)
            
            axes[1, 0].plot(recent_data.index, recent_data, label='Historical', color='gold', linewidth=2)
            axes[1, 0].plot(self.forecast['forecast_dates'], self.forecast['forecast'], 
                           label='Forecast', color='red', linewidth=2, linestyle='--')
            axes[1, 0].fill_between(self.forecast['forecast_dates'],
                                   self.forecast['lower_ci'],
                                   self.forecast['upper_ci'],
                                   alpha=0.3, color='red', label='95% CI')
            axes[1, 0].set_title('Recent Prices and Forecast')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Forecast distribution
            axes[1, 1].hist(self.forecast['forecast'], bins=20, alpha=0.7, color='red', edgecolor='black')
            axes[1, 1].axvline(self.forecast['forecast'].mean(), color='blue', linestyle='--', 
                              label=f'Mean: ${self.forecast["forecast"].mean():.2f}')
            axes[1, 1].set_title('Forecast Distribution')
            axes[1, 1].set_xlabel('Predicted Price ($)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # 3. Performance metrics visualization
        if 'performance_metrics' in self.diagnostics:
            metrics = self.diagnostics['performance_metrics']
            
            # Prediction vs Actual
            axes[2, 0].scatter(metrics['test_data'], metrics['predictions'], alpha=0.7)
            axes[2, 0].plot([metrics['test_data'].min(), metrics['test_data'].max()],
                           [metrics['test_data'].min(), metrics['test_data'].max()],
                           'r--', label='Perfect Prediction')
            axes[2, 0].set_xlabel('Actual Price ($)')
            axes[2, 0].set_ylabel('Predicted Price ($)')
            axes[2, 0].set_title('Prediction vs Actual')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)
            
            # Error metrics comparison
            metrics_names = ['RMSE', 'MAE', 'MAPE']
            arima_scores = [metrics['rmse'], metrics['mae'], metrics['mape']]
            baseline_scores = [metrics['baseline_rmse'], metrics['baseline_mae'], metrics['baseline_mape']]
            
            x = np.arange(len(metrics_names))
            width = 0.35
            
            axes[2, 1].bar(x - width/2, arima_scores, width, label='ARIMA Model', color='lightblue')
            axes[2, 1].bar(x + width/2, baseline_scores, width, label='Baseline Model', color='lightcoral')
            axes[2, 1].set_xlabel('Metrics')
            axes[2, 1].set_ylabel('Values')
            axes[2, 1].set_title('Model Performance Comparison')
            axes[2, 1].set_xticks(x)
            axes[2, 1].set_xticklabels(metrics_names)
            axes[2, 1].legend()
            axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self):
        """
        Generate a comprehensive analysis report.
        """
        print(f"\n{'='*80}")
        print("COMPREHENSIVE GOLD PRICE PREDICTION REPORT")
        print(f"{'='*80}")
        
        print(f"\n📊 DATA SUMMARY:")
        print(f"   • Dataset Period: {self.data.index.min().date()} to {self.data.index.max().date()}")
        print(f"   • Total Data Points: {len(self.data):,}")
        print(f"   • Price Range: ${self.data['Close'].min():.2f} - ${self.data['Close'].max():.2f}")
        print(f"   • Average Price: ${self.data['Close'].mean():.2f}")
        
        if 'best_order' in self.diagnostics:
            order = self.diagnostics['best_order']
            print(f"\n🎯 MODEL CONFIGURATION:")
            print(f"   • Optimal ARIMA Order: ARIMA{order}")
            print(f"   • AIC Score: {self.diagnostics.get('best_aic', 'N/A'):.2f}")
            
            if 'differencing_order' in self.diagnostics:
                print(f"   • Differencing Order: {self.diagnostics['differencing_order']}")
        
        if 'performance_metrics' in self.diagnostics:
            metrics = self.diagnostics['performance_metrics']
            print(f"\n📈 PERFORMANCE METRICS:")
            print(f"   • RMSE: ${metrics['rmse']:.2f}")
            print(f"   • MAE: ${metrics['mae']:.2f}")
            print(f"   • MAPE: {metrics['mape']:.2f}%")
            print(f"   • Improvement over Baseline: {metrics['improvement_rmse']:.1f}%")
        
        if self.forecast is not None:
            print(f"\n🔮 FORECAST SUMMARY:")
            print(f"   • Forecast Period: {len(self.forecast['forecast'])} days")
            print(f"   • Next Day Prediction: ${self.forecast['forecast'].iloc[0]:.2f}")
            print(f"   • 30-Day Average: ${self.forecast['forecast'].mean():.2f}")
            
            trend = "📈 Upward" if self.forecast['forecast'].iloc[-1] > self.forecast['forecast'].iloc[0] else "📉 Downward"
            print(f"   • Overall Trend: {trend}")
        
        print(f"\n💡 INVESTMENT INSIGHTS:")
        
        if self.forecast is not None:
            current_price = self.data['Close'].iloc[-1]
            next_price = self.forecast['forecast'].iloc[0]
            price_change = ((next_price - current_price) / current_price) * 100
            
            if price_change > 2:
                print(f"   • Strong Buy Signal: Expected price increase of {price_change:.1f}%")
            elif price_change > 0:
                print(f"   • Moderate Buy Signal: Expected price increase of {price_change:.1f}%")
            elif price_change > -2:
                print(f"   • Hold Signal: Expected price change of {price_change:.1f}%")
            else:
                print(f"   • Sell Signal: Expected price decrease of {price_change:.1f}%")
        
        # Volatility analysis
        returns = self.data['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100
        
        if volatility > 25:
            risk_level = "High"
        elif volatility > 15:
            risk_level = "Moderate"
        else:
            risk_level = "Low"
        
        print(f"   • Risk Level: {risk_level} (Annualized Volatility: {volatility:.1f}%)")
        
        print(f"\n⚠️  DISCLAIMER:")
        print(f"   This analysis is for educational purposes only.")
        print(f"   Past performance does not guarantee future results.")
        print(f"   Please conduct thorough research before making investment decisions.")
        
        print(f"\n{'='*80}")
    
    def run_complete_analysis(self):
        """
        Run the complete gold price prediction analysis pipeline.
        """
        print("🏃‍♂️ Starting Complete Gold Price Prediction Analysis...")
        print("="*80)
        
        # Step 1: Load and explore data
        self.load_data()
        self.explore_data()
        
        # Step 2: Check stationarity and make stationary if needed
        stationarity_result = self.check_stationarity()
        if not stationarity_result['is_stationary']:
            stationary_series = self.make_stationary()
        
        # Step 3: ACF/PACF analysis
        self.analyze_acf_pacf()
        
        # Step 4: Find optimal ARIMA parameters
        best_order = self.find_best_arima_order()
        
        # Step 5: Fit the best model
        self.fit_arima_model(order=best_order)
        
        # Step 6: Model diagnostics
        self.model_diagnostics()
        
        # Step 7: Performance evaluation
        self.calculate_performance_metrics()
        
        # Step 8: Generate forecasts
        self.generate_forecast(steps=30)
        
        # Step 9: Create visualizations
        self.visualize_results()
        
        # Step 10: Generate comprehensive report
        self.generate_report()
        
        print("\n🎉 Analysis Complete!")
        print("Check the visualizations and report above for detailed insights.")


def main():
    """
    Main function to demonstrate the Gold Price Prediction system.
    """
    print("🔮 Gold Price Prediction using ARIMA Model")
    print("=" * 50)
    
    # Initialize predictor
    predictor = GoldPricePredictor(symbol='GC=F', period='3y')
    
    # Run complete analysis
    predictor.run_complete_analysis()
    
    print("\n✨ Gold Price Prediction Analysis Complete!")
    print("This model demonstrates:")
    print("• Time-series data collection and preprocessing")
    print("• Statistical tests for stationarity (ADF test)")
    print("• ACF/PACF analysis for parameter selection")
    print("• ARIMA model optimization and fitting")
    print("• Comprehensive model diagnostics")
    print("• Performance evaluation with multiple metrics")
    print("• Interactive visualizations and forecasting")
    print("• Practical investment insights generation")


if __name__ == "__main__":
    main()