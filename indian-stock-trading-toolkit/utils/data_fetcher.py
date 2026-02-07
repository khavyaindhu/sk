"""
Data Fetcher for Indian Stocks
Supports multiple data sources: yfinance, NSE, and broker APIs
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import requests
import json

class IndianStockDataFetcher:
    """
    Fetch historical and live data for Indian stocks
    """
    
    def __init__(self):
        self.nse_base_url = "https://www.nseindia.com/api"
        self.headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json'
        }
    
    def get_historical_data_yfinance(self, symbol, start_date, end_date, interval='1d'):
        """
        Fetch historical data using yfinance
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE.NS' for NSE, 'RELIANCE.BO' for BSE)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval ('1d', '1h', '5m', etc.)
        
        Returns:
            pandas DataFrame with OHLCV data
        """
        try:
            # Add .NS for NSE or .BO for BSE if not present
            if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
                symbol = f"{symbol}.NS"
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            
            # Rename columns to standard format
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            return df[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def get_nifty50_stocks(self):
        """
        Get list of NIFTY 50 stocks
        
        Returns:
            List of stock symbols
        """
        # Common NIFTY 50 stocks (you should update this list periodically)
        nifty50 = [
            'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
            'HINDUNILVR', 'ITC', 'SBIN', 'BHARTIARTL', 'KOTAKBANK',
            'LT', 'BAJFINANCE', 'ASIANPAINT', 'MARUTI', 'HCLTECH',
            'AXISBANK', 'SUNPHARMA', 'ULTRACEMCO', 'TITAN', 'WIPRO',
            'NESTLEIND', 'TATASTEEL', 'BAJAJFINSV', 'TECHM', 'POWERGRID',
            'NTPC', 'ONGC', 'M&M', 'ADANIPORTS', 'INDUSINDBK'
        ]
        return nifty50
    
    def get_stock_info(self, symbol):
        """
        Get company information
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with company info
        """
        try:
            if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
                symbol = f"{symbol}.NS"
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'pe_ratio': info.get('trailingPE', 'N/A'),
                'dividend_yield': info.get('dividendYield', 'N/A'),
            }
        except Exception as e:
            print(f"Error fetching info for {symbol}: {str(e)}")
            return None
    
    def get_live_price(self, symbol):
        """
        Get current live price
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Current price as float
        """
        try:
            if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
                symbol = f"{symbol}.NS"
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d', interval='1m')
            
            if not data.empty:
                return data['Close'].iloc[-1]
            return None
            
        except Exception as e:
            print(f"Error fetching live price for {symbol}: {str(e)}")
            return None
    
    def save_to_csv(self, df, filename):
        """
        Save DataFrame to CSV
        
        Args:
            df: pandas DataFrame
            filename: Output filename
        """
        df.to_csv(filename)
        print(f"Data saved to {filename}")


# Example usage
if __name__ == "__main__":
    fetcher = IndianStockDataFetcher()
    
    # Example 1: Fetch historical data for Reliance
    print("Fetching historical data for RELIANCE...")
    df = fetcher.get_historical_data_yfinance(
        'RELIANCE',
        start_date='2023-01-01',
        end_date='2024-01-01'
    )
    
    if df is not None:
        print(df.head())
        print(f"\nTotal rows: {len(df)}")
        fetcher.save_to_csv(df, 'data/RELIANCE_historical.csv')
    
    # Example 2: Get stock info
    print("\n" + "="*50)
    print("Fetching company info for TCS...")
    info = fetcher.get_stock_info('TCS')
    if info:
        for key, value in info.items():
            print(f"{key}: {value}")
    
    # Example 3: Get live price
    print("\n" + "="*50)
    print("Fetching live price for INFY...")
    price = fetcher.get_live_price('INFY')
    if price:
        print(f"Current price: ₹{price:.2f}")
    
    # Example 4: Get NIFTY 50 stocks
    print("\n" + "="*50)
    print("NIFTY 50 stocks:")
    stocks = fetcher.get_nifty50_stocks()
    print(stocks[:10], "...")
