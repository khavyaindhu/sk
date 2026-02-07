"""
Technical Indicators for Stock Analysis
Common indicators used in trading strategies
"""

import pandas as pd
import numpy as np

class TechnicalIndicators:
    """
    Calculate various technical indicators
    """
    
    @staticmethod
    def add_moving_averages(df, periods=[20, 50, 200]):
        """
        Add Simple Moving Averages (SMA)
        
        Args:
            df: DataFrame with 'close' column
            periods: List of periods for MA calculation
            
        Returns:
            DataFrame with MA columns added
        """
        df = df.copy()
        for period in periods:
            df[f'SMA_{period}'] = df['close'].rolling(window=period).mean()
        return df
    
    @staticmethod
    def add_ema(df, periods=[12, 26, 50]):
        """
        Add Exponential Moving Averages (EMA)
        
        Args:
            df: DataFrame with 'close' column
            periods: List of periods for EMA calculation
            
        Returns:
            DataFrame with EMA columns added
        """
        df = df.copy()
        for period in periods:
            df[f'EMA_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        return df
    
    @staticmethod
    def add_rsi(df, period=14):
        """
        Add Relative Strength Index (RSI)
        
        Args:
            df: DataFrame with 'close' column
            period: RSI period (default 14)
            
        Returns:
            DataFrame with RSI column added
        """
        df = df.copy()
        
        # Calculate price changes
        delta = df['close'].diff()
        
        # Separate gains and losses
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df
    
    @staticmethod
    def add_macd(df, fast=12, slow=26, signal=9):
        """
        Add MACD (Moving Average Convergence Divergence)
        
        Args:
            df: DataFrame with 'close' column
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns:
            DataFrame with MACD columns added
        """
        df = df.copy()
        
        # Calculate EMAs
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        
        # MACD line
        df['MACD'] = ema_fast - ema_slow
        
        # Signal line
        df['MACD_signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
        
        # MACD histogram
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        return df
    
    @staticmethod
    def add_bollinger_bands(df, period=20, std_dev=2):
        """
        Add Bollinger Bands
        
        Args:
            df: DataFrame with 'close' column
            period: Period for moving average
            std_dev: Number of standard deviations
            
        Returns:
            DataFrame with Bollinger Bands columns added
        """
        df = df.copy()
        
        # Calculate middle band (SMA)
        df['BB_middle'] = df['close'].rolling(window=period).mean()
        
        # Calculate standard deviation
        std = df['close'].rolling(window=period).std()
        
        # Calculate upper and lower bands
        df['BB_upper'] = df['BB_middle'] + (std_dev * std)
        df['BB_lower'] = df['BB_middle'] - (std_dev * std)
        
        # Calculate bandwidth
        df['BB_width'] = df['BB_upper'] - df['BB_lower']
        
        return df
    
    @staticmethod
    def add_stochastic(df, k_period=14, d_period=3):
        """
        Add Stochastic Oscillator
        
        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            k_period: %K period
            d_period: %D period
            
        Returns:
            DataFrame with Stochastic columns added
        """
        df = df.copy()
        
        # Calculate %K
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        
        df['Stoch_K'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
        
        # Calculate %D (moving average of %K)
        df['Stoch_D'] = df['Stoch_K'].rolling(window=d_period).mean()
        
        return df
    
    @staticmethod
    def add_atr(df, period=14):
        """
        Add Average True Range (ATR) - volatility indicator
        
        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            period: ATR period
            
        Returns:
            DataFrame with ATR column added
        """
        df = df.copy()
        
        # Calculate True Range
        df['H-L'] = df['high'] - df['low']
        df['H-PC'] = abs(df['high'] - df['close'].shift(1))
        df['L-PC'] = abs(df['low'] - df['close'].shift(1))
        
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        
        # Calculate ATR
        df['ATR'] = df['TR'].rolling(window=period).mean()
        
        # Clean up temporary columns
        df.drop(['H-L', 'H-PC', 'L-PC', 'TR'], axis=1, inplace=True)
        
        return df
    
    @staticmethod
    def add_volume_indicators(df):
        """
        Add volume-based indicators
        
        Args:
            df: DataFrame with 'close' and 'volume' columns
            
        Returns:
            DataFrame with volume indicators added
        """
        df = df.copy()
        
        # Volume Moving Average
        df['Volume_MA'] = df['volume'].rolling(window=20).mean()
        
        # On-Balance Volume (OBV)
        df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        # Volume Rate of Change
        df['Volume_ROC'] = df['volume'].pct_change(periods=14) * 100
        
        return df
    
    @staticmethod
    def add_all_indicators(df):
        """
        Add all common technical indicators
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all indicators added
        """
        df = TechnicalIndicators.add_moving_averages(df)
        df = TechnicalIndicators.add_ema(df)
        df = TechnicalIndicators.add_rsi(df)
        df = TechnicalIndicators.add_macd(df)
        df = TechnicalIndicators.add_bollinger_bands(df)
        df = TechnicalIndicators.add_stochastic(df)
        df = TechnicalIndicators.add_atr(df)
        df = TechnicalIndicators.add_volume_indicators(df)
        
        return df


# Example usage
if __name__ == "__main__":
    # Create sample data
    import sys
    sys.path.append('..')
    from utils.data_fetcher import IndianStockDataFetcher
    
    fetcher = IndianStockDataFetcher()
    df = fetcher.get_historical_data_yfinance(
        'RELIANCE',
        start_date='2023-01-01',
        end_date='2024-01-01'
    )
    
    if df is not None:
        print("Original data shape:", df.shape)
        print("\nFirst few rows:")
        print(df.head())
        
        # Add all indicators
        df_with_indicators = TechnicalIndicators.add_all_indicators(df)
        
        print("\n" + "="*50)
        print("Data with indicators shape:", df_with_indicators.shape)
        print("\nColumns:", df_with_indicators.columns.tolist())
        
        print("\n" + "="*50)
        print("Sample data with indicators:")
        print(df_with_indicators[['close', 'SMA_20', 'RSI', 'MACD']].tail())
