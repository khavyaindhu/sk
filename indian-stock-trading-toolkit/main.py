"""
Main Starter Script for Indian Stock Trading
This script demonstrates the complete workflow
"""

import pandas as pd
import sys
from datetime import datetime, timedelta

# Import custom modules
from utils.data_fetcher import IndianStockDataFetcher
from utils.indicators import TechnicalIndicators
from strategies.example_strategies import (
    MovingAverageCrossover, 
    RSIStrategy, 
    MACDStrategy,
    CombinedStrategy
)
from backtesting.engine import BacktestEngine
from ml_models.price_predictor import StockPricePredictor

def analyze_stock(symbol, start_date, end_date):
    """
    Complete analysis workflow for a stock
    
    Args:
        symbol: Stock symbol (e.g., 'RELIANCE')
        start_date: Start date for analysis
        end_date: End date for analysis
    """
    print("="*80)
    print(f"ANALYZING {symbol}")
    print("="*80)
    
    # 1. Fetch Data
    print("\n[1/5] Fetching historical data...")
    fetcher = IndianStockDataFetcher()
    df = fetcher.get_historical_data_yfinance(symbol, start_date, end_date)
    
    if df is None:
        print(f"❌ Failed to fetch data for {symbol}")
        return
    
    print(f"✓ Fetched {len(df)} days of data")
    
    # 2. Add Technical Indicators
    print("\n[2/5] Computing technical indicators...")
    df = TechnicalIndicators.add_all_indicators(df)
    print(f"✓ Added technical indicators")
    
    # 3. Display Current Status
    print("\n[3/5] Current Market Status:")
    print("-" * 80)
    latest = df.iloc[-1]
    print(f"Last Close:        ₹{latest['close']:.2f}")
    print(f"SMA 20:            ₹{latest['SMA_20']:.2f}")
    print(f"SMA 50:            ₹{latest['SMA_50']:.2f}")
    print(f"RSI:               {latest['RSI']:.2f}")
    print(f"MACD:              {latest['MACD']:.2f}")
    print(f"MACD Signal:       {latest['MACD_signal']:.2f}")
    print(f"Bollinger Upper:   ₹{latest['BB_upper']:.2f}")
    print(f"Bollinger Lower:   ₹{latest['BB_lower']:.2f}")
    print(f"Volume:            {latest['volume']:,.0f}")
    
    # Technical signals
    print("\nTechnical Signals:")
    if latest['SMA_20'] > latest['SMA_50']:
        print("  • MA Trend: BULLISH (SMA 20 > SMA 50)")
    else:
        print("  • MA Trend: BEARISH (SMA 20 < SMA 50)")
    
    if latest['RSI'] < 30:
        print("  • RSI: OVERSOLD (< 30)")
    elif latest['RSI'] > 70:
        print("  • RSI: OVERBOUGHT (> 70)")
    else:
        print(f"  • RSI: NEUTRAL ({latest['RSI']:.2f})")
    
    if latest['MACD'] > latest['MACD_signal']:
        print("  • MACD: BULLISH (MACD > Signal)")
    else:
        print("  • MACD: BEARISH (MACD < Signal)")
    
    # 4. Backtest Strategies
    print("\n[4/5] Backtesting trading strategies...")
    print("-" * 80)
    
    strategies = [
        MovingAverageCrossover(fast_period=20, slow_period=50),
        RSIStrategy(period=14, oversold=30, overbought=70),
        MACDStrategy(),
    ]
    
    results = []
    for strategy in strategies:
        signals = strategy.generate_signals(df.copy())
        engine = BacktestEngine(initial_capital=100000, commission=0.0003)
        metrics = engine.run_backtest(df, signals)
        
        results.append({
            'Strategy': strategy.name,
            'Return %': f"{metrics['total_return_pct']:.2f}",
            'Win Rate %': f"{metrics['win_rate_pct']:.2f}",
            'Trades': metrics['total_trades'],
            'Sharpe': f"{metrics['sharpe_ratio']:.2f}",
            'Max DD %': f"{metrics['max_drawdown_pct']:.2f}"
        })
    
    results_df = pd.DataFrame(results)
    print("\nStrategy Performance:")
    print(results_df.to_string(index=False))
    
    # 5. ML Price Prediction
    print("\n[5/5] Machine Learning price prediction...")
    print("-" * 80)
    
    try:
        predictor = StockPricePredictor(model_type='random_forest')
        df_features = predictor.create_features(df, lookback_period=5)
        X_train, X_test, y_train, y_test = predictor.prepare_data(df_features, test_size=0.2)
        
        predictor.train_random_forest(X_train, y_train, n_estimators=50)
        metrics, predictions = predictor.evaluate(X_test, y_test)
        
        print(f"ML Model Performance:")
        print(f"  RMSE: ₹{metrics['RMSE']:.2f}")
        print(f"  MAE: ₹{metrics['MAE']:.2f}")
        print(f"  R²: {metrics['R2']:.4f}")
        print(f"  Directional Accuracy: {metrics['Directional_Accuracy_%']:.2f}%")
        
        # Next day prediction
        latest_data = df_features.tail(1)
        next_day_price = predictor.predict_next_day(latest_data)
        current_price = df['close'].iloc[-1]
        
        print(f"\nNext Day Prediction:")
        print(f"  Current Price: ₹{current_price:.2f}")
        print(f"  Predicted Price: ₹{next_day_price:.2f}")
        change_pct = ((next_day_price - current_price) / current_price * 100)
        print(f"  Expected Change: {change_pct:+.2f}%")
        
    except Exception as e:
        print(f"⚠️  ML prediction failed: {str(e)}")
    
    print("\n" + "="*80)


def compare_stocks(symbols, start_date, end_date):
    """
    Compare multiple stocks
    
    Args:
        symbols: List of stock symbols
        start_date: Start date
        end_date: End date
    """
    print("="*80)
    print("COMPARING MULTIPLE STOCKS")
    print("="*80)
    
    fetcher = IndianStockDataFetcher()
    comparison = []
    
    for symbol in symbols:
        print(f"\nFetching {symbol}...")
        df = fetcher.get_historical_data_yfinance(symbol, start_date, end_date)
        
        if df is not None:
            # Calculate metrics
            start_price = df['close'].iloc[0]
            end_price = df['close'].iloc[-1]
            returns = ((end_price - start_price) / start_price) * 100
            volatility = df['close'].pct_change().std() * np.sqrt(252) * 100
            
            comparison.append({
                'Symbol': symbol,
                'Start Price': f"₹{start_price:.2f}",
                'End Price': f"₹{end_price:.2f}",
                'Return %': f"{returns:.2f}",
                'Volatility %': f"{volatility:.2f}",
                'Avg Volume': f"{df['volume'].mean():,.0f}"
            })
    
    if comparison:
        comp_df = pd.DataFrame(comparison)
        print("\n" + "="*80)
        print("COMPARISON RESULTS")
        print("="*80)
        print(comp_df.to_string(index=False))


def main():
    """
    Main function - Choose what to run
    """
    print("\n🚀 INDIAN STOCK TRADING TOOLKIT")
    print("="*80)
    print("\nWhat would you like to do?")
    print("1. Analyze a single stock (detailed)")
    print("2. Compare multiple stocks")
    print("3. Test all strategies on a stock")
    print("4. Get live market data")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        symbol = input("Enter stock symbol (e.g., RELIANCE, TCS, INFY): ").strip().upper()
        
        # Default to last 2 years
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        
        analyze_stock(symbol, start_date, end_date)
    
    elif choice == '2':
        symbols_input = input("Enter stock symbols separated by commas (e.g., RELIANCE,TCS,INFY): ")
        symbols = [s.strip().upper() for s in symbols_input.split(',')]
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        compare_stocks(symbols, start_date, end_date)
    
    elif choice == '3':
        symbol = input("Enter stock symbol: ").strip().upper()
        
        from strategies.example_strategies import (
            MovingAverageCrossover, RSIStrategy, MACDStrategy,
            BollingerBandsStrategy, CombinedStrategy
        )
        
        fetcher = IndianStockDataFetcher()
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        
        df = fetcher.get_historical_data_yfinance(symbol, start_date, end_date)
        
        if df is not None:
            strategies = [
                MovingAverageCrossover(20, 50),
                RSIStrategy(14, 30, 70),
                MACDStrategy(),
                BollingerBandsStrategy(),
                CombinedStrategy()
            ]
            
            results = []
            for strategy in strategies:
                signals = strategy.generate_signals(df.copy())
                engine = BacktestEngine(initial_capital=100000)
                metrics = engine.run_backtest(df, signals)
                
                results.append({
                    'Strategy': strategy.name,
                    'Return %': metrics['total_return_pct'],
                    'Win Rate %': metrics['win_rate_pct'],
                    'Trades': metrics['total_trades'],
                    'Sharpe': metrics['sharpe_ratio']
                })
            
            results_df = pd.DataFrame(results)
            print("\n" + "="*80)
            print("ALL STRATEGIES COMPARISON")
            print("="*80)
            print(results_df.to_string(index=False))
    
    elif choice == '4':
        symbol = input("Enter stock symbol: ").strip().upper()
        
        fetcher = IndianStockDataFetcher()
        price = fetcher.get_live_price(symbol)
        info = fetcher.get_stock_info(symbol)
        
        if price:
            print(f"\n{symbol} - Live Price: ₹{price:.2f}")
        
        if info:
            print("\nCompany Information:")
            for key, value in info.items():
                print(f"  {key}: {value}")
    
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    import numpy as np
    
    # Example: Quick analysis
    print("\n📊 Running example analysis for RELIANCE...")
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    
    analyze_stock('RELIANCE', start_date, end_date)
    
    print("\n\n💡 To run interactively, uncomment the main() function call")
    # main()
