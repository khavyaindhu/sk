"""
Example Trading Strategies
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from utils.indicators import TechnicalIndicators

class TradingStrategy:
    """Base class for trading strategies"""
    
    def __init__(self, name):
        self.name = name
    
    def generate_signals(self, df):
        """
        Generate trading signals
        Should return a Series with 'BUY', 'SELL', or 'HOLD' signals
        """
        raise NotImplementedError("Subclass must implement generate_signals")


class MovingAverageCrossover(TradingStrategy):
    """
    Moving Average Crossover Strategy
    
    Buy when fast MA crosses above slow MA
    Sell when fast MA crosses below slow MA
    """
    
    def __init__(self, fast_period=20, slow_period=50):
        super().__init__("MA Crossover")
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def generate_signals(self, df):
        """Generate signals based on MA crossover"""
        df = df.copy()
        
        # Add moving averages if not present
        if f'SMA_{self.fast_period}' not in df.columns:
            df = TechnicalIndicators.add_moving_averages(
                df, 
                periods=[self.fast_period, self.slow_period]
            )
        
        signals = pd.Series('HOLD', index=df.index)
        
        fast_ma = df[f'SMA_{self.fast_period}']
        slow_ma = df[f'SMA_{self.slow_period}']
        
        for i in range(1, len(df)):
            if pd.notna(fast_ma.iloc[i]) and pd.notna(slow_ma.iloc[i]):
                # Buy signal: fast MA crosses above slow MA
                if (fast_ma.iloc[i] > slow_ma.iloc[i] and 
                    fast_ma.iloc[i-1] <= slow_ma.iloc[i-1]):
                    signals.iloc[i] = 'BUY'
                
                # Sell signal: fast MA crosses below slow MA
                elif (fast_ma.iloc[i] < slow_ma.iloc[i] and 
                      fast_ma.iloc[i-1] >= slow_ma.iloc[i-1]):
                    signals.iloc[i] = 'SELL'
        
        return signals


class RSIStrategy(TradingStrategy):
    """
    RSI-based Strategy
    
    Buy when RSI crosses below oversold level (default 30)
    Sell when RSI crosses above overbought level (default 70)
    """
    
    def __init__(self, period=14, oversold=30, overbought=70):
        super().__init__("RSI Strategy")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signals(self, df):
        """Generate signals based on RSI"""
        df = df.copy()
        
        # Add RSI if not present
        if 'RSI' not in df.columns:
            df = TechnicalIndicators.add_rsi(df, period=self.period)
        
        signals = pd.Series('HOLD', index=df.index)
        position = False  # Track if we're in a position
        
        for i in range(1, len(df)):
            if pd.notna(df['RSI'].iloc[i]):
                # Buy signal: RSI crosses below oversold and we're not in position
                if df['RSI'].iloc[i] < self.oversold and not position:
                    signals.iloc[i] = 'BUY'
                    position = True
                
                # Sell signal: RSI crosses above overbought and we're in position
                elif df['RSI'].iloc[i] > self.overbought and position:
                    signals.iloc[i] = 'SELL'
                    position = False
        
        return signals


class MACDStrategy(TradingStrategy):
    """
    MACD Strategy
    
    Buy when MACD crosses above signal line
    Sell when MACD crosses below signal line
    """
    
    def __init__(self, fast=12, slow=26, signal=9):
        super().__init__("MACD Strategy")
        self.fast = fast
        self.slow = slow
        self.signal = signal
    
    def generate_signals(self, df):
        """Generate signals based on MACD"""
        df = df.copy()
        
        # Add MACD if not present
        if 'MACD' not in df.columns:
            df = TechnicalIndicators.add_macd(df, self.fast, self.slow, self.signal)
        
        signals = pd.Series('HOLD', index=df.index)
        
        for i in range(1, len(df)):
            if pd.notna(df['MACD'].iloc[i]) and pd.notna(df['MACD_signal'].iloc[i]):
                # Buy signal: MACD crosses above signal line
                if (df['MACD'].iloc[i] > df['MACD_signal'].iloc[i] and 
                    df['MACD'].iloc[i-1] <= df['MACD_signal'].iloc[i-1]):
                    signals.iloc[i] = 'BUY'
                
                # Sell signal: MACD crosses below signal line
                elif (df['MACD'].iloc[i] < df['MACD_signal'].iloc[i] and 
                      df['MACD'].iloc[i-1] >= df['MACD_signal'].iloc[i-1]):
                    signals.iloc[i] = 'SELL'
        
        return signals


class BollingerBandsStrategy(TradingStrategy):
    """
    Bollinger Bands Mean Reversion Strategy
    
    Buy when price touches lower band
    Sell when price touches upper band
    """
    
    def __init__(self, period=20, std_dev=2):
        super().__init__("Bollinger Bands")
        self.period = period
        self.std_dev = std_dev
    
    def generate_signals(self, df):
        """Generate signals based on Bollinger Bands"""
        df = df.copy()
        
        # Add Bollinger Bands if not present
        if 'BB_upper' not in df.columns:
            df = TechnicalIndicators.add_bollinger_bands(df, self.period, self.std_dev)
        
        signals = pd.Series('HOLD', index=df.index)
        position = False
        
        for i in range(1, len(df)):
            if pd.notna(df['BB_upper'].iloc[i]) and pd.notna(df['BB_lower'].iloc[i]):
                # Buy signal: price touches or goes below lower band
                if df['close'].iloc[i] <= df['BB_lower'].iloc[i] and not position:
                    signals.iloc[i] = 'BUY'
                    position = True
                
                # Sell signal: price touches or goes above upper band
                elif df['close'].iloc[i] >= df['BB_upper'].iloc[i] and position:
                    signals.iloc[i] = 'SELL'
                    position = False
        
        return signals


class CombinedStrategy(TradingStrategy):
    """
    Combined Strategy using multiple indicators
    
    Buy when:
    - RSI is oversold (< 30)
    - Price is below lower Bollinger Band
    - MACD is positive
    
    Sell when:
    - RSI is overbought (> 70)
    - Price is above upper Bollinger Band
    """
    
    def __init__(self):
        super().__init__("Combined Strategy")
    
    def generate_signals(self, df):
        """Generate signals based on multiple indicators"""
        df = df.copy()
        
        # Add all required indicators
        df = TechnicalIndicators.add_rsi(df, period=14)
        df = TechnicalIndicators.add_bollinger_bands(df, period=20, std_dev=2)
        df = TechnicalIndicators.add_macd(df)
        
        signals = pd.Series('HOLD', index=df.index)
        position = False
        
        for i in range(1, len(df)):
            if (pd.notna(df['RSI'].iloc[i]) and 
                pd.notna(df['BB_lower'].iloc[i]) and 
                pd.notna(df['MACD'].iloc[i])):
                
                # Buy signal: multiple conditions
                if (df['RSI'].iloc[i] < 35 and 
                    df['close'].iloc[i] < df['BB_lower'].iloc[i] and 
                    df['MACD'].iloc[i] > df['MACD_signal'].iloc[i] and 
                    not position):
                    signals.iloc[i] = 'BUY'
                    position = True
                
                # Sell signal: multiple conditions
                elif (df['RSI'].iloc[i] > 65 and 
                      df['close'].iloc[i] > df['BB_upper'].iloc[i] and 
                      position):
                    signals.iloc[i] = 'SELL'
                    position = False
        
        return signals


# Example usage and comparison
if __name__ == "__main__":
    from utils.data_fetcher import IndianStockDataFetcher
    from backtesting.engine import BacktestEngine
    
    # Fetch data
    print("Fetching data for TCS...")
    fetcher = IndianStockDataFetcher()
    df = fetcher.get_historical_data_yfinance(
        'TCS',
        start_date='2023-01-01',
        end_date='2024-01-01'
    )
    
    if df is not None:
        # Test different strategies
        strategies = [
            MovingAverageCrossover(fast_period=20, slow_period=50),
            RSIStrategy(period=14, oversold=30, overbought=70),
            MACDStrategy(),
            BollingerBandsStrategy(),
            CombinedStrategy()
        ]
        
        results = []
        
        for strategy in strategies:
            print(f"\nTesting {strategy.name}...")
            
            # Generate signals
            signals = strategy.generate_signals(df.copy())
            
            # Run backtest
            engine = BacktestEngine(initial_capital=100000, commission=0.0003)
            metrics = engine.run_backtest(df, signals)
            
            results.append({
                'Strategy': strategy.name,
                'Return %': metrics['total_return_pct'],
                'Win Rate %': metrics['win_rate_pct'],
                'Total Trades': metrics['total_trades'],
                'Sharpe Ratio': metrics['sharpe_ratio'],
                'Max Drawdown %': metrics['max_drawdown_pct']
            })
        
        # Compare results
        print("\n" + "="*80)
        print("STRATEGY COMPARISON")
        print("="*80)
        
        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))
        
        print("\n" + "="*80)
        print(f"Best Strategy by Return: {results_df.loc[results_df['Return %'].idxmax(), 'Strategy']}")
        print(f"Best Strategy by Win Rate: {results_df.loc[results_df['Win Rate %'].idxmax(), 'Strategy']}")
        print(f"Best Strategy by Sharpe Ratio: {results_df.loc[results_df['Sharpe Ratio'].idxmax(), 'Strategy']}")
