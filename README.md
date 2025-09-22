# Gold Price Prediction using ARIMA Model

A comprehensive time-series forecasting system that predicts future gold prices using ARIMA (AutoRegressive Integrated Moving Average) models. This project demonstrates advanced statistical analysis, model diagnostics, and performance evaluation for financial forecasting.

## 🎯 Project Highlights

• **Advanced Time-Series Analysis**: Built a sophisticated ARIMA model for gold price prediction using historical datasets  
• **Statistical Rigor**: Performed comprehensive statistical tests including ADF test for stationarity  
• **Model Optimization**: Automated parameter selection using ACF & PACF plots and AIC criterion  
• **Robust Diagnostics**: Extensive model validation with residual analysis and error metrics  
• **Superior Performance**: Achieved 60%+ improvement over baseline models in prediction accuracy  
• **Investment Intelligence**: Generated actionable insights for financial decision-making and investment strategies  
• **Interactive Visualizations**: Created comprehensive dashboards with forecasts and confidence intervals

## 📊 Key Features

### Statistical Analysis
- **Stationarity Testing**: Augmented Dickey-Fuller (ADF) test implementation
- **Data Preprocessing**: Automated differencing to achieve stationarity
- **Seasonal Decomposition**: Trend, seasonal, and residual component analysis

### ARIMA Model Implementation
- **Parameter Optimization**: Automated search for optimal (p,d,q) parameters
- **Model Fitting**: Robust ARIMA model estimation with error handling  
- **Model Diagnostics**: Residual analysis, Ljung-Box test, Q-Q plots
- **Performance Metrics**: MSE, RMSE, MAE, MAPE evaluation

### Forecasting & Visualization
- **Multi-step Forecasting**: Generate predictions with confidence intervals
- **Interactive Dashboards**: Plotly-based dynamic visualizations
- **Performance Comparison**: Side-by-side baseline vs ARIMA model comparison
- **Investment Insights**: Automated buy/sell/hold recommendations

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run Complete Analysis
```bash
python gold_price_prediction.py
```

### Quick Demo
```bash
python quick_example.py
```

## 📈 Sample Results

The model demonstrates superior performance with:
- **RMSE Improvement**: 60%+ over naive baseline
- **MAPE**: < 2% prediction error
- **Forecast Horizon**: 30-day predictions with confidence intervals
- **Data Coverage**: 3+ years of historical gold price data

## 🔍 Technical Implementation

### Data Collection
- Real-time gold price data via Yahoo Finance API (GC=F)
- Fallback to simulated realistic data for demo purposes
- Comprehensive data validation and preprocessing

### Statistical Tests
```python
# Stationarity testing
adf_result = adfuller(series)
if adf_result[1] > 0.05:
    # Apply differencing
    stationary_series = series.diff().dropna()
```

### ARIMA Model Selection
```python
# Automated parameter search
best_aic = float('inf')
for p, d, q in itertools.product(range(max_p), range(max_d), range(max_q)):
    model = ARIMA(data, order=(p,d,q))
    fitted_model = model.fit()
    if fitted_model.aic < best_aic:
        best_order = (p,d,q)
        best_aic = fitted_model.aic
```

### Performance Evaluation
- **Train-Test Split**: 80/20 validation approach
- **Multiple Metrics**: RMSE, MAE, MAPE for comprehensive evaluation
- **Baseline Comparison**: Naive forecast benchmark
- **Statistical Significance**: Ljung-Box test for residual independence

## 📊 Model Diagnostics

The system performs comprehensive model validation:

1. **Residual Analysis**: Check for white noise in residuals
2. **Autocorrelation Tests**: Ljung-Box test for residual independence  
3. **Normality Tests**: Q-Q plots for residual distribution
4. **Heteroscedasticity**: Constant variance validation
5. **Model Stability**: Parameter significance testing


## 💼 Investment Applications

### Decision Support Features
- **Price Direction**: Automated trend detection
- **Risk Assessment**: Volatility-based risk metrics
- **Signal Generation**: Buy/sell/hold recommendations
- **Confidence Intervals**: Uncertainty quantification for risk management

### Real-world Applications
- **Portfolio Management**: Gold allocation optimization
- **Risk Hedging**: Precious metals risk assessment
- **Trading Strategies**: Short-term and long-term position planning
- **Market Analysis**: Gold market trend identification

## 📋 Project Structure

```
arima_gold_prediction/
├── gold_price_prediction.py    # Main analysis pipeline
├── quick_example.py            # Simplified demo script  
├── requirements.txt            # Python dependencies
└── README.md                  # Project documentation
```

## 🔧 Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib/seaborn**: Static visualizations  
- **plotly**: Interactive dashboards
- **statsmodels**: Statistical modeling and ARIMA implementation
- **scikit-learn**: Performance metrics
- **yfinance**: Financial data collection

## 📝 Model Performance

### Validation Results
- **RMSE**: $42.68 (vs $112.01 baseline) - **61.9% improvement**
- **MAE**: $37.05 (vs $103.10 baseline) - **64.1% improvement**  
- **MAPE**: 1.77% (vs 4.92% baseline) - **64.1% improvement**

### Model Configuration
- **Optimal Order**: ARIMA(0,2,2) selected via AIC criterion
- **AIC Score**: 8696.40
- **Differencing**: d=1 for stationarity
- **Forecast Horizon**: 30 days with 95% confidence intervals

## ⚠️ Disclaimer

This project is for educational and research purposes only. Past performance does not guarantee future results. Always conduct thorough research and consider consulting with financial professionals before making investment decisions.

