# Usage Guide: Gold Price Prediction with ARIMA

## Quick Start Guide

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/rajeev117/arima_gold_prediction.git
cd arima_gold_prediction

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Complete Analysis
```bash
python gold_price_prediction.py
```
This will:
- Load historical gold price data
- Perform statistical analysis (ADF test, stationarity)
- Find optimal ARIMA parameters
- Fit and validate the model
- Generate 30-day forecasts
- Create comprehensive visualizations
- Produce investment insights

### 3. Quick Demo
```bash
python quick_example.py
```
A simplified version with:
- 7-day forecast
- Basic visualizations
- Quick investment recommendation

### 4. Interactive Analysis
```bash
jupyter notebook gold_price_analysis.ipynb
```
Step-by-step notebook with detailed explanations.

### 5. Run Tests
```bash
python test_arima.py
```
Validates all core functionality.

## Key Features Demonstrated

### ✅ Statistical Analysis
- **Augmented Dickey-Fuller Test**: Tests for stationarity
- **ACF/PACF Analysis**: Determines optimal ARIMA parameters
- **Ljung-Box Test**: Validates residual independence
- **Q-Q Plots**: Checks residual normality

### ✅ ARIMA Implementation
- **Automated Parameter Search**: Finds optimal (p,d,q) using AIC
- **Model Diagnostics**: Comprehensive validation
- **Forecast Generation**: Multi-step predictions with confidence intervals
- **Performance Evaluation**: Multiple error metrics (RMSE, MAE, MAPE)

### ✅ Practical Applications
- **Investment Insights**: Buy/sell/hold recommendations
- **Risk Assessment**: Volatility-based risk metrics
- **Performance Comparison**: ARIMA vs baseline models
- **Interactive Dashboards**: Plotly-based visualizations

## Example Output

```
🎯 MODEL CONFIGURATION:
   • Optimal ARIMA Order: ARIMA(0,2,2)
   • AIC Score: 8696.40
   
📈 PERFORMANCE METRICS:
   • RMSE: $42.68 (vs $112.01 baseline)
   • Improvement: 61.9% over baseline
   • MAPE: 1.77%

🔮 7-DAY FORECAST:
   Day 1: $2176.36 ± $12.50
   Day 2: $2177.28 ± $15.20
   ...

💡 INVESTMENT INSIGHT:
   Next Day: +0.2% expected change
   Recommendation: HOLD
```

## API Usage

### Basic Usage
```python
from gold_price_prediction import GoldPricePredictor

# Initialize
predictor = GoldPricePredictor(symbol='GC=F', period='2y')

# Load data and run analysis
predictor.load_data()
predictor.run_complete_analysis()
```

### Custom Analysis
```python
# Step-by-step approach
predictor.load_data()
predictor.check_stationarity()
predictor.make_stationary()
predictor.analyze_acf_pacf()

# Find best parameters and fit model
best_order = predictor.find_best_arima_order()
predictor.fit_arima_model(order=best_order)

# Generate forecasts
forecast = predictor.generate_forecast(steps=14)
```

### Access Results
```python
# Model diagnostics
diagnostics = predictor.diagnostics
print(f"AIC: {diagnostics['aic']}")
print(f"Best Order: {diagnostics['best_order']}")

# Performance metrics
metrics = predictor.calculate_performance_metrics()
print(f"RMSE: {metrics['rmse']:.2f}")

# Forecast data
if predictor.forecast:
    print(f"Next price: ${predictor.forecast['forecast'][0]:.2f}")
```

## Customization Options

### Data Sources
- **Real Data**: Yahoo Finance (GC=F, GOLD, etc.)
- **Custom Data**: Use your own CSV files
- **Sample Data**: Built-in realistic gold price simulation

### Model Parameters
- **ARIMA Orders**: Customize (p,d,q) parameters
- **Seasonal ARIMA**: Add seasonal components
- **Forecast Horizon**: Any number of steps
- **Validation Split**: Adjust train/test ratios

### Visualizations
- **Static Plots**: Matplotlib/Seaborn
- **Interactive Charts**: Plotly dashboards
- **Custom Themes**: Modify styling
- **Export Options**: PNG, HTML, PDF

## Educational Value

This project demonstrates:

1. **Time-Series Analysis Fundamentals**
   - Stationarity concepts and testing
   - Autocorrelation and partial autocorrelation
   - Differencing and trend removal

2. **Statistical Modeling Best Practices**
   - Model selection criteria (AIC/BIC)
   - Cross-validation techniques
   - Residual analysis and diagnostics

3. **Financial Applications**
   - Risk assessment methodologies
   - Investment decision frameworks
   - Performance benchmarking

4. **Python Data Science Stack**
   - Pandas for data manipulation
   - Statsmodels for statistical analysis
   - Matplotlib/Plotly for visualization
   - Scikit-learn for metrics

## Next Steps

### Potential Enhancements
- **SARIMA Models**: Handle seasonal patterns
- **External Variables**: Include economic indicators
- **Ensemble Methods**: Combine multiple models
- **Real-time Updates**: Automated daily forecasts
- **Web Interface**: Flask/Streamlit dashboard

### Advanced Features
- **Monte Carlo Simulation**: Risk scenario analysis
- **Portfolio Optimization**: Multi-asset allocation
- **Alert Systems**: Price threshold notifications
- **API Integration**: Real-time data feeds

---

*This implementation serves as a comprehensive foundation for time-series forecasting and financial analysis applications.*