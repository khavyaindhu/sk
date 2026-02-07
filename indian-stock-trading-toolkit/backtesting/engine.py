"""
Backtesting Engine for Trading Strategies
"""

import pandas as pd
import numpy as np
from datetime import datetime

class BacktestEngine:
    """
    Simple backtesting engine for trading strategies
    """
    
    def __init__(self, initial_capital=100000, commission=0.0003):
        """
        Initialize backtesting engine
        
        Args:
            initial_capital: Starting capital in rupees
            commission: Commission rate (0.03% default for Indian brokers)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.reset()
    
    def reset(self):
        """Reset backtest state"""
        self.capital = self.initial_capital
        self.position = 0  # Number of shares held
        self.trades = []
        self.equity_curve = []
        self.positions_history = []
    
    def execute_trade(self, date, price, signal, quantity=None):
        """
        Execute a trade based on signal
        
        Args:
            date: Trade date
            price: Current price
            signal: 'BUY', 'SELL', or 'HOLD'
            quantity: Number of shares (if None, use all available capital for buy)
            
        Returns:
            Trade details dictionary
        """
        trade = None
        
        if signal == 'BUY' and self.position == 0:
            # Calculate quantity if not specified
            if quantity is None:
                quantity = int((self.capital * 0.95) / price)  # Use 95% of capital
            
            if quantity > 0:
                cost = quantity * price
                commission_cost = cost * self.commission
                total_cost = cost + commission_cost
                
                if total_cost <= self.capital:
                    self.capital -= total_cost
                    self.position = quantity
                    
                    trade = {
                        'date': date,
                        'type': 'BUY',
                        'price': price,
                        'quantity': quantity,
                        'cost': cost,
                        'commission': commission_cost,
                        'total_cost': total_cost,
                        'capital_remaining': self.capital
                    }
                    self.trades.append(trade)
        
        elif signal == 'SELL' and self.position > 0:
            revenue = self.position * price
            commission_cost = revenue * self.commission
            total_revenue = revenue - commission_cost
            
            self.capital += total_revenue
            
            trade = {
                'date': date,
                'type': 'SELL',
                'price': price,
                'quantity': self.position,
                'revenue': revenue,
                'commission': commission_cost,
                'total_revenue': total_revenue,
                'capital_after': self.capital
            }
            self.trades.append(trade)
            self.position = 0
        
        # Record equity curve
        portfolio_value = self.capital + (self.position * price)
        self.equity_curve.append({
            'date': date,
            'capital': self.capital,
            'position_value': self.position * price,
            'total_value': portfolio_value
        })
        
        return trade
    
    def run_backtest(self, df, strategy_signals):
        """
        Run backtest on historical data with strategy signals
        
        Args:
            df: DataFrame with OHLCV data
            strategy_signals: Series with 'BUY', 'SELL', or 'HOLD' signals
            
        Returns:
            Results dictionary
        """
        self.reset()
        
        for idx, row in df.iterrows():
            signal = strategy_signals.loc[idx]
            self.execute_trade(idx, row['close'], signal)
        
        # Close any open position at the end
        if self.position > 0:
            last_price = df.iloc[-1]['close']
            self.execute_trade(df.index[-1], last_price, 'SELL')
        
        return self.calculate_metrics()
    
    def calculate_metrics(self):
        """
        Calculate performance metrics
        
        Returns:
            Dictionary with performance metrics
        """
        if len(self.equity_curve) == 0:
            return {}
        
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('date', inplace=True)
        
        # Calculate returns
        total_return = (equity_df['total_value'].iloc[-1] - self.initial_capital) / self.initial_capital * 100
        
        # Calculate daily returns
        equity_df['daily_return'] = equity_df['total_value'].pct_change()
        
        # Sharpe ratio (annualized, assuming 252 trading days)
        daily_returns = equity_df['daily_return'].dropna()
        if len(daily_returns) > 0 and daily_returns.std() != 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        equity_df['cummax'] = equity_df['total_value'].cummax()
        equity_df['drawdown'] = (equity_df['total_value'] - equity_df['cummax']) / equity_df['cummax'] * 100
        max_drawdown = equity_df['drawdown'].min()
        
        # Win rate
        profitable_trades = 0
        total_trades = 0
        total_profit = 0
        total_loss = 0
        
        for i in range(0, len(self.trades), 2):
            if i + 1 < len(self.trades):
                buy_trade = self.trades[i]
                sell_trade = self.trades[i + 1]
                
                profit = (sell_trade['total_revenue'] - buy_trade['total_cost'])
                
                total_trades += 1
                if profit > 0:
                    profitable_trades += 1
                    total_profit += profit
                else:
                    total_loss += abs(profit)
        
        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Profit factor
        profit_factor = (total_profit / total_loss) if total_loss > 0 else 0
        
        metrics = {
            'initial_capital': self.initial_capital,
            'final_capital': equity_df['total_value'].iloc[-1],
            'total_return_pct': total_return,
            'total_return_amount': equity_df['total_value'].iloc[-1] - self.initial_capital,
            'total_trades': total_trades,
            'winning_trades': profitable_trades,
            'losing_trades': total_trades - profitable_trades,
            'win_rate_pct': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'total_commission_paid': sum([t.get('commission', 0) for t in self.trades])
        }
        
        return metrics
    
    def print_results(self, metrics):
        """
        Print backtest results in a formatted way
        
        Args:
            metrics: Dictionary with performance metrics
        """
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        
        print(f"\nCapital:")
        print(f"  Initial Capital:        ₹{metrics['initial_capital']:,.2f}")
        print(f"  Final Capital:          ₹{metrics['final_capital']:,.2f}")
        print(f"  Total Return:           ₹{metrics['total_return_amount']:,.2f} ({metrics['total_return_pct']:.2f}%)")
        
        print(f"\nTrades:")
        print(f"  Total Trades:           {metrics['total_trades']}")
        print(f"  Winning Trades:         {metrics['winning_trades']}")
        print(f"  Losing Trades:          {metrics['losing_trades']}")
        print(f"  Win Rate:               {metrics['win_rate_pct']:.2f}%")
        print(f"  Profit Factor:          {metrics['profit_factor']:.2f}")
        
        print(f"\nRisk Metrics:")
        print(f"  Sharpe Ratio:           {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:           {metrics['max_drawdown_pct']:.2f}%")
        
        print(f"\nCosts:")
        print(f"  Total Commission:       ₹{metrics['total_commission_paid']:,.2f}")
        
        print("\n" + "="*60)
    
    def get_equity_curve(self):
        """
        Get equity curve as DataFrame
        
        Returns:
            DataFrame with equity curve
        """
        return pd.DataFrame(self.equity_curve)
    
    def get_trades(self):
        """
        Get all trades as DataFrame
        
        Returns:
            DataFrame with all trades
        """
        return pd.DataFrame(self.trades)


# Example usage
if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from utils.data_fetcher import IndianStockDataFetcher
    from utils.indicators import TechnicalIndicators
    
    # Fetch data
    fetcher = IndianStockDataFetcher()
    df = fetcher.get_historical_data_yfinance(
        'RELIANCE',
        start_date='2023-01-01',
        end_date='2024-01-01'
    )
    
    if df is not None:
        # Add indicators
        df = TechnicalIndicators.add_moving_averages(df, periods=[20, 50])
        
        # Create simple moving average crossover strategy
        signals = pd.Series('HOLD', index=df.index)
        
        for i in range(1, len(df)):
            if pd.notna(df['SMA_20'].iloc[i]) and pd.notna(df['SMA_50'].iloc[i]):
                # Buy signal: SMA 20 crosses above SMA 50
                if (df['SMA_20'].iloc[i] > df['SMA_50'].iloc[i] and 
                    df['SMA_20'].iloc[i-1] <= df['SMA_50'].iloc[i-1]):
                    signals.iloc[i] = 'BUY'
                
                # Sell signal: SMA 20 crosses below SMA 50
                elif (df['SMA_20'].iloc[i] < df['SMA_50'].iloc[i] and 
                      df['SMA_20'].iloc[i-1] >= df['SMA_50'].iloc[i-1]):
                    signals.iloc[i] = 'SELL'
        
        # Run backtest
        engine = BacktestEngine(initial_capital=100000, commission=0.0003)
        metrics = engine.run_backtest(df, signals)
        
        # Print results
        engine.print_results(metrics)
        
        # Show sample trades
        print("\nSample Trades:")
        print(engine.get_trades().head(10))
