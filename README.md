# Advanced Market Predictor

A comprehensive Python-based tool for predicting stock and forex market movements using multiple data sources and machine learning models.

## Features

- **Multiple Data Sources**
  - Yahoo Finance
  - Alpha Vantage
  - Extensible for additional sources

- **Advanced Technical Indicators**
  - Momentum (RSI, Stochastic, MACD)
  - Trend (ADX, CCI, Ichimoku)
  - Volatility (Bollinger Bands, ATR)
  - Volume (MFI, ADI)
  - Custom indicators

- **Multiple Prediction Models**
  - Linear Regression
  - Ridge and Lasso Regression
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - Facebook Prophet

- **Advanced Analytics**
  - Time series cross-validation
  - Feature importance analysis
  - Model performance comparison
  - Confidence intervals

## Installation

1. **Clone the repository** (optional)
```bash
git clone https://github.com/jasonofua/python-currency-prediction.git
cd python-currency-prediction
```

2. **Create and activate a virtual environment**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

## Configuration

1. **Get API Keys**
   - Sign up for a free Alpha Vantage API key at: https://www.alphavantage.co/support/#api-key
   - Save your API key in a secure location

2. **Configure API Keys**
   - Create a `config.py` file in the project root:
   ```python
   ALPHA_VANTAGE_API_KEY = 'YOUR_API_KEY'
   ```

## Usage

### Basic Usage

```python
from market_predictor import AdvancedMarketPredictor

# Initialize predictor
predictor = AdvancedMarketPredictor(api_key='YOUR_ALPHA_VANTAGE_API_KEY')

# Download and predict forex data
predictor.download_data(
    symbol='EURUSD',
    start_date='2020-01-01',
    end_date='2024-01-01',
    market_type='forex',
    source='alpha_vantage'
)

# Preprocess data and train models
predictor.preprocess_data()
X, y, feature_names = predictor.prepare_features()
results = predictor.train_models(X, y)

# Train Prophet model and make predictions
prophet_model, forecast = predictor.train_prophet_model()

# Visualize results
predictor.plot_predictions(forecast)
```

### Advanced Usage

```python
# Custom technical indicators and multiple symbols
symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
sources = ['yahoo', 'alpha_vantage']

for symbol in symbols:
    for source in sources:
        predictor.download_data(
            symbol=symbol,
            start_date='2020-01-01',
            end_date='2024-01-01',
            market_type='forex',
            source=source
        )
        # ... (continue with analysis)
```

## Example Outputs

The predictor generates several visualizations:

1. **Price Predictions**
   - Actual vs. predicted prices
   - Confidence intervals
   - Multiple model comparisons

2. **Model Performance**
   - R² scores comparison
   - MSE metrics
   - Feature importance analysis

## Common Issues and Solutions

1. **Installation Errors**
   ```
   Error: Microsoft Visual C++ 14.0 or greater is required
   ```
   Solution: Install Microsoft Build Tools for Visual Studio

2. **API Rate Limits**
   ```
   Error: Alpha Vantage API rate limit exceeded
   ```
   Solution: Wait for rate limit reset or upgrade API key

3. **Memory Issues**
   ```
   MemoryError: Unable to allocate array
   ```
   Solution: Reduce data timeframe or increase system memory

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Yahoo Finance for market data
- Alpha Vantage for API support
- scikit-learn team for machine learning tools
- Facebook for Prophet forecasting tool

## Contact

For support or queries:
- Create an issue in the repository
- Email: your.email@example.com

## Roadmap

Future improvements planned:
- Additional data sources (OANDA, FXCM)
- Deep learning models (LSTM, Transformer)
- Portfolio optimization features
- Real-time prediction capabilities