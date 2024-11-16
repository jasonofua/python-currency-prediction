import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from alpha_vantage.foreignexchange import ForeignExchange
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots


import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from alpha_vantage.foreignexchange import ForeignExchange
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
import ta

class AdvancedMarketPredictor:
    def __init__(self, api_key=None):
        self.data = None
        self.scaler = MinMaxScaler()
        self.models = {}
        self.market_type = None
        self.api_key = api_key
        
    def download_data(self, symbol, start_date, end_date, market_type='stock', source='yahoo'):
        """
        Download market data with forex-specific handling
        """
        try:
            self.market_type = market_type
            
            if source == 'yahoo':
                if market_type == 'forex':
                    symbol = f"{symbol}=X"
                ticker = yf.Ticker(symbol)
                self.data = ticker.history(start=start_date, end=end_date)
                
                # Handle forex-specific column names
                if market_type == 'forex':
                    # Map forex column names to standard names
                    column_mapping = {
                        'Open': 'Open',
                        'High': 'High',
                        'Low': 'Low',
                        'Close': 'Close',
                        'Adj Close': 'Close',  # Some forex data uses Adj Close
                        'Volume': 'Volume'
                    }
                    
                    # Rename columns if they exist
                    self.data.rename(columns=column_mapping, inplace=True)
                    
                    # If 'Close' is missing but 'Adj Close' exists, use that
                    if 'Close' not in self.data.columns and 'Adj Close' in self.data.columns:
                        self.data['Close'] = self.data['Adj Close']
                
            elif source == 'alpha_vantage':
                if not self.api_key:
                    raise ValueError("Alpha Vantage API key required")
                    
                fx = ForeignExchange(key=self.api_key)
            if market_type == 'forex':
                base_currency = symbol[:3]
                quote_currency = symbol[3:]
                data, _ = fx.get_currency_exchange_daily(
                    from_symbol=base_currency,
                    to_symbol=quote_currency,
                    outputsize='full'
                )
                
                # Create empty lists to store the data
                dates = []
                opens = []
                highs = []
                lows = []
                closes = []
                
                # Parse the data correctly
                for date, values in data.items():
                    dates.append(pd.to_datetime(date))
                    opens.append(float(values['1. open']))
                    highs.append(float(values['2. high']))
                    lows.append(float(values['3. low']))
                    closes.append(float(values['4. close']))
                
                # Create DataFrame with correct structure
                self.data = pd.DataFrame({
                    'Open': opens,
                    'High': highs,
                    'Low': lows,
                    'Close': closes
                }, index=dates)
                
                # Sort index and filter date range
                self.data.sort_index(inplace=True)
                mask = (self.data.index >= pd.to_datetime(start_date)) & \
                       (self.data.index <= pd.to_datetime(end_date))
                self.data = self.data.loc[mask]
            
            if self.data is None or len(self.data) == 0:
                raise ValueError(f"No data retrieved for {symbol}")
                
            # Verify required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            if missing_columns:
                print(f"Available columns: {self.data.columns.tolist()}")
                raise ValueError(f"Missing required columns: {missing_columns}")
                
            print(f"Downloaded {len(self.data)} days of data for {symbol}")
            print(f"Date range: {self.data.index.min()} to {self.data.index.max()}")
            print(f"Columns: {self.data.columns.tolist()}")

              # Display sample of the data
            print("\nFirst few rows of data:")
            print(self.data.head())
            return True
            
        except Exception as e:
            print(f"Error downloading data: {e}")
            import traceback
            traceback.print_exc()
            return False

    def preprocess_data(self):
        """
        Enhanced data preprocessing with forex support
        """
        if self.data is None:
            raise ValueError("No data available. Please download data first.")
            
        print(f"Initial data shape: {self.data.shape}")
        
        # Handle missing values
        self.data = self.data.ffill().bfill()
        
        # Add basic technical indicators
        self.data['Returns'] = self.data['Close'].pct_change()
        self.data['Volatility'] = self.data['Returns'].rolling(window=20).std()
        
        # Add advanced indicators with error handling
        try:
            self.add_advanced_indicators()
        except Exception as e:
            print(f"Warning: Error adding advanced indicators: {e}")
            print("Continuing with basic indicators only...")
            
        # Remove remaining NaN values
        original_length = len(self.data)
        self.data = self.data.dropna()
        removed_rows = original_length - len(self.data)
        
        if removed_rows > 0:
            print(f"Removed {removed_rows} rows containing NaN values")
        
        print(f"Final data shape after preprocessing: {self.data.shape}")
        
        if len(self.data) == 0:
            raise ValueError("All data was removed during preprocessing")
            
        return self.data

    def add_advanced_indicators(self):
        """
        Add technical indicators with validation
        """
        if len(self.data) == 0:
            raise ValueError("Empty dataset")
            
        # Momentum indicators
        self.data['RSI'] = ta.momentum.rsi(self.data['Close'])
        self.data['MACD_diff'] = ta.trend.macd_diff(self.data['Close'])
        
        # Trend indicators
        self.data['ADX'] = ta.trend.adx(
            self.data['High'],
            self.data['Low'],
            self.data['Close']
        )
        
        # Volatility indicators
        self.data['BBH'] = ta.volatility.bollinger_hband(self.data['Close'])
        self.data['BBL'] = ta.volatility.bollinger_lband(self.data['Close'])
        self.data['ATR'] = ta.volatility.average_true_range(
            self.data['High'],
            self.data['Low'],
            self.data['Close']
        )

    def prepare_features(self, target_days_ahead=5):
        """
        Prepare features with simplified feature set
        """
        if self.data is None or len(self.data) == 0:
            raise ValueError("No data available for feature preparation")
            
        print("Preparing features...")
        print(f"Available columns: {self.data.columns.tolist()}")
        
        # Create target variable
        self.data['Target'] = self.data['Close'].shift(-target_days_ahead)
        
        # Use a simplified feature set for forex
        base_features = ['Close', 'Returns', 'Volatility']
        
        # Add technical indicators if available
        technical_features = ['RSI', 'MACD_diff', 'ADX', 'BBH', 'BBL', 'ATR']
        
        # Combine available features
        all_features = base_features + [f for f in technical_features if f in self.data.columns]
        available_features = [f for f in all_features if f in self.data.columns]
        
        print(f"Selected features: {available_features}")
        
        # Create feature matrix
        X = self.data[available_features].iloc[:-target_days_ahead]
        y = self.data['Target'].iloc[:-target_days_ahead]
        
        if len(X) == 0:
            raise ValueError("No valid features available after preparation")
            
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"Final feature matrix shape: {X_scaled.shape}")
        print(f"Target vector shape: {y.shape}")
        
        return X_scaled, y, available_features
    
    def train_models(self, X, y):
        """
        Train multiple models and evaluate their performance
        """
        print("\nTraining models...")
    
        # Initialize models
        models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0),
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gbm': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'xgb': XGBRegressor(n_estimators=100, random_state=42)
        }
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name} model...")
            mse_scores = []
            r2_scores = []
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                mse_scores.append(mean_squared_error(y_test, y_pred))
                r2_scores.append(r2_score(y_test, y_pred))
            
            results[name] = {
                'model': model,
                'mse_mean': np.mean(mse_scores),
                'mse_std': np.std(mse_scores),
                'r2_mean': np.mean(r2_scores),
                'r2_std': np.std(r2_scores)
            }
            
            print(f"{name.upper()} Performance:")
            print(f"Mean R²: {results[name]['r2_mean']:.4f} (±{results[name]['r2_std']:.4f})")
            print(f"Mean MSE: {results[name]['mse_mean']:.4f} (±{results[name]['mse_std']:.4f})")
        
        self.models = results
        return results

    def train_prophet_model(self):
        """
        Train Facebook Prophet model with additional seasonality features
        and proper timezone handling
        """
        print("\nTraining Prophet model...")
        
        # Convert index to datetime and remove timezone
        prophet_data = pd.DataFrame({
            'ds': self.data.index.tz_localize(None),  # Remove timezone
            'y': self.data['Close']
        })
        
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        
        # Add additional regressors if available
        if 'Volume' in self.data.columns:
            prophet_data['volume'] = self.data['Volume']
            model.add_regressor('volume')
        
        model.fit(prophet_data)
        
        future_dates = model.make_future_dataframe(periods=30)
        if 'volume' in prophet_data.columns:
            # Use the mean volume for future predictions
            future_dates['volume'] = prophet_data['volume'].mean()
        
        forecast = model.predict(future_dates)
        
        return model, forecast

    def plot_predictions(self, forecast):
        """
        Enhanced visualization using plotly instead of matplotlib
        """
        print("\nCreating visualizations...")
        
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Price Prediction with Confidence Intervals',
                'Model R² Comparison',
                'Model MSE Comparison',
                'Feature Importance'
            ),
            specs=[
                [{"colspan": 2}, None],
                [{"type": "bar"}, {"type": "bar"}]
            ]
        )
        
        # Add price predictions
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data['Close'],
                name="Actual",
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat'],
                name="Forecast",
                line=dict(color='red', dash='dash')
            ),
            row=1, col=1
        )
        
        # Add confidence intervals
        fig.add_trace(
            go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_upper'],
                name='Upper Bound',
                line=dict(color='gray', dash='dot'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_lower'],
                name='Lower Bound',
                line=dict(color='gray', dash='dot'),
                fill='tonexty',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Add model comparison bars
        model_names = list(self.models.keys())
        r2_scores = [m['r2_mean'] for m in self.models.values()]
        mse_scores = [m['mse_mean'] for m in self.models.values()]
        
        # R² comparison
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=r2_scores,
                name='R² Score'
            ),
            row=2, col=1
        )
        
        # MSE comparison
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=mse_scores,
                name='MSE'
            ),
            row=2, col=2
        )
        
        # Add feature importance if available
        if 'rf' in self.models:
            rf_model = self.models['rf']['model']
            importances = pd.Series(
                rf_model.feature_importances_,
                index=self.feature_names
            ).sort_values()[-10:]  # Top 10 features
            
            fig.add_trace(
                go.Bar(
                    x=importances.values,
                    y=importances.index,
                    orientation='h',
                    name='Feature Importance'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1000,
            width=1200,
            showlegend=True,
            title_text="Market Predictions Dashboard"
        )
        
        # Save the plot to HTML file
        fig.write_html("market_predictions.html")
        print("Visualizations saved to 'market_predictions.html'")

def main():
    # Initialize predictor
    predictor = AdvancedMarketPredictor(api_key='9UTMTORNUPZATX1G')
    
    # Set date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # Download and process data
    symbol = 'GBPUSD'
    if predictor.download_data(
        symbol,
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d'),
        'forex',
        'alpha_vantage'
    ):
        try:

            print("\nPreprocessing data...")
            predictor.preprocess_data()
            
            print("\nPreparing features...")
            X, y, feature_names = predictor.prepare_features()
            predictor.feature_names = feature_names
            print("Feature preparation successful!")
            
            # Train and evaluate models
            print("\nTraining models...")
            results = predictor.train_models(X, y)
            
            # Train Prophet model
            print("\nTraining Prophet model...")
            prophet_model, forecast = predictor.train_prophet_model()
            
            # Plot results
            print("\nGenerating visualizations...")
            predictor.plot_predictions(forecast)

            
        except Exception as e:
            print(f"Error during data processing: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Failed to download data")

if __name__ == "__main__":
    main()
