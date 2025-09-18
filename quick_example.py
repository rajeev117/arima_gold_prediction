"""
Quick Example: Gold Price Prediction with ARIMA
==============================================

This script demonstrates a simplified version of the gold price prediction
for quick testing and understanding.
"""

from gold_price_prediction import GoldPricePredictor
import matplotlib.pyplot as plt

def main():
    """
    Run a quick gold price prediction example.
    """
    print("🚀 Quick Gold Price Prediction Example")
    print("=" * 50)
    
    # Initialize predictor with shorter period for faster demo
    predictor = GoldPricePredictor(symbol='GC=F', period='1y')
    
    # Load and explore data
    predictor.load_data()
    print(f"\nLoaded {len(predictor.data)} data points")
    print(f"Price range: ${predictor.data['Close'].min():.2f} - ${predictor.data['Close'].max():.2f}")
    
    # Quick stationarity check
    stationarity_result = predictor.check_stationarity()
    if not stationarity_result['is_stationary']:
        print("Making series stationary...")
        predictor.make_stationary()
    
    # Find and fit best ARIMA model
    print("\nSearching for optimal ARIMA parameters...")
    best_order = predictor.find_best_arima_order(max_p=2, max_d=2, max_q=2)
    
    print(f"\nFitting ARIMA{best_order} model...")
    fitted_model = predictor.fit_arima_model(order=best_order)
    
    # Generate forecast
    print("\nGenerating 7-day forecast...")
    forecast = predictor.generate_forecast(steps=7)
    
    if forecast:
        print(f"\n📈 7-Day Forecast:")
        for i, (date, price) in enumerate(zip(forecast['forecast_dates'][:7], forecast['forecast'][:7])):
            print(f"   Day {i+1} ({date.strftime('%Y-%m-%d')}): ${price:.2f}")
        
        # Simple plot
        plt.figure(figsize=(12, 6))
        
        # Plot recent data
        recent_data = predictor.data['Close'].tail(30)
        plt.plot(recent_data.index, recent_data, label='Historical', color='blue', linewidth=2)
        
        # Plot forecast
        plt.plot(forecast['forecast_dates'][:7], forecast['forecast'][:7], 
                label='7-Day Forecast', color='red', linewidth=2, marker='o')
        
        plt.fill_between(forecast['forecast_dates'][:7],
                        forecast['lower_ci'][:7],
                        forecast['upper_ci'][:7],
                        alpha=0.3, color='red', label='95% Confidence Interval')
        
        plt.title('Gold Price Prediction - Last 30 Days + 7-Day Forecast')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Investment insight
        current_price = predictor.data['Close'].iloc[-1]
        next_price = forecast['forecast'].iloc[0]
        price_change = ((next_price - current_price) / current_price) * 100
        
        print(f"\n💡 Investment Insight:")
        print(f"   Current Price: ${current_price:.2f}")
        print(f"   Next Day Prediction: ${next_price:.2f}")
        print(f"   Expected Change: {price_change:+.2f}%")
        
        if price_change > 1:
            print(f"   💰 Recommendation: BUY (Expecting upward trend)")
        elif price_change < -1:
            print(f"   🔻 Recommendation: SELL (Expecting downward trend)")
        else:
            print(f"   ⏸️ Recommendation: HOLD (Minimal expected change)")
    
    print(f"\n✅ Quick analysis complete!")
    print(f"For comprehensive analysis, run: python gold_price_prediction.py")

if __name__ == "__main__":
    main()