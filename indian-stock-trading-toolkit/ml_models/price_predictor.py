"""
Machine Learning Model for Stock Price Prediction
Using Random Forest and LSTM examples
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
sys.path.append('..')

class StockPricePredictor:
    """
    ML-based stock price predictor
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize predictor
        
        Args:
            model_type: 'random_forest' or 'lstm'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_columns = None
    
    def create_features(self, df, lookback_period=5):
        """
        Create features for ML model
        
        Args:
            df: DataFrame with OHLCV and indicators
            lookback_period: Number of days to look back
            
        Returns:
            DataFrame with features
        """
        df = df.copy()
        
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Lagged features
        for i in range(1, lookback_period + 1):
            df[f'close_lag_{i}'] = df['close'].shift(i)
            df[f'volume_lag_{i}'] = df['volume'].shift(i)
            df[f'returns_lag_{i}'] = df['returns'].shift(i)
        
        # Rolling statistics
        df['close_rolling_mean_5'] = df['close'].rolling(window=5).mean()
        df['close_rolling_std_5'] = df['close'].rolling(window=5).std()
        df['volume_rolling_mean_5'] = df['volume'].rolling(window=5).mean()
        
        # Price ratios
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Day of week and month (temporal features)
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        
        # Target variable (next day's close price)
        df['target'] = df['close'].shift(-1)
        
        # Target direction (1 for up, 0 for down)
        df['target_direction'] = (df['target'] > df['close']).astype(int)
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def prepare_data(self, df, test_size=0.2):
        """
        Prepare data for training
        
        Args:
            df: DataFrame with features
            test_size: Proportion of test data
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Select feature columns (exclude target and non-feature columns)
        exclude_cols = ['target', 'target_direction', 'open', 'high', 'low', 'close', 'volume']
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        X = df[self.feature_columns]
        y = df['target']
        
        # Split data (use temporal split for time series)
        split_idx = int(len(df) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_random_forest(self, X_train, y_train, n_estimators=100):
        """
        Train Random Forest model
        
        Args:
            X_train: Training features
            y_train: Training target
            n_estimators: Number of trees
        """
        print("Training Random Forest model...")
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        print("Training complete!")
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary with metrics
        """
        predictions = self.predict(X_test)
        
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # Calculate directional accuracy
        actual_direction = (y_test.values > y_test.shift(1).values).astype(int)
        predicted_direction = (predictions > y_test.shift(1).values).astype(int)
        directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Directional_Accuracy_%': directional_accuracy
        }
        
        return metrics, predictions
    
    def get_feature_importance(self):
        """
        Get feature importance for Random Forest
        
        Returns:
            DataFrame with feature importance
        """
        if self.model_type == 'random_forest' and self.model is not None:
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        return None
    
    def predict_next_day(self, latest_data):
        """
        Predict next day's price
        
        Args:
            latest_data: Latest data point with features
            
        Returns:
            Predicted price
        """
        latest_scaled = self.scaler.transform(latest_data[self.feature_columns])
        prediction = self.model.predict(latest_scaled)
        return prediction[0]


# Example usage
if __name__ == "__main__":
    from utils.data_fetcher import IndianStockDataFetcher
    from utils.indicators import TechnicalIndicators
    
    # Fetch data
    print("Fetching data for RELIANCE...")
    fetcher = IndianStockDataFetcher()
    df = fetcher.get_historical_data_yfinance(
        'RELIANCE',
        start_date='2022-01-01',
        end_date='2024-01-01'
    )
    
    if df is not None:
        print(f"Data shape: {df.shape}")
        
        # Add technical indicators
        print("\nAdding technical indicators...")
        df = TechnicalIndicators.add_all_indicators(df)
        
        # Initialize predictor
        predictor = StockPricePredictor(model_type='random_forest')
        
        # Create features
        print("\nCreating features...")
        df_features = predictor.create_features(df, lookback_period=5)
        print(f"Features shape: {df_features.shape}")
        
        # Prepare data
        print("\nPreparing train/test split...")
        X_train, X_test, y_train, y_test = predictor.prepare_data(df_features, test_size=0.2)
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Train model
        print("\n" + "="*60)
        predictor.train_random_forest(X_train, y_train, n_estimators=100)
        
        # Evaluate
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        metrics, predictions = predictor.evaluate(X_test, y_test)
        
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Feature importance
        print("\n" + "="*60)
        print("TOP 10 MOST IMPORTANT FEATURES")
        print("="*60)
        importance = predictor.get_feature_importance()
        print(importance.head(10))
        
        # Sample predictions
        print("\n" + "="*60)
        print("SAMPLE PREDICTIONS (Last 5 days)")
        print("="*60)
        
        results_df = pd.DataFrame({
            'Actual': y_test.tail(5).values,
            'Predicted': predictions[-5:],
            'Error': y_test.tail(5).values - predictions[-5:]
        })
        print(results_df)
        
        # Predict next day
        print("\n" + "="*60)
        print("NEXT DAY PREDICTION")
        print("="*60)
        latest_data = df_features.tail(1)
        next_day_price = predictor.predict_next_day(latest_data)
        current_price = df['close'].iloc[-1]
        
        print(f"Current Price: ₹{current_price:.2f}")
        print(f"Predicted Next Day Price: ₹{next_day_price:.2f}")
        print(f"Expected Change: ₹{next_day_price - current_price:.2f} ({((next_day_price - current_price) / current_price * 100):.2f}%)")
        
        print("\n⚠️  IMPORTANT DISCLAIMER:")
        print("This is a simple ML model for educational purposes only.")
        print("DO NOT use this for actual trading without:")
        print("  - Extensive backtesting")
        print("  - Proper risk management")
        print("  - Understanding of model limitations")
        print("  - Professional financial advice")
