"""
Zerodha Kite Connect Integration
For live trading with Zerodha broker

Prerequisites:
1. Zerodha trading account
2. Kite Connect API subscription (₹2000/month)
3. Generate API key and secret from https://kite.trade/

Installation:
pip install kiteconnect

Documentation: https://kite.trade/docs/connect/v3/
"""

from kiteconnect import KiteConnect
import logging
import pandas as pd
from datetime import datetime, timedelta

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

class ZerodhaAPI:
    """
    Wrapper for Zerodha Kite Connect API
    """
    
    def __init__(self, api_key, api_secret=None, access_token=None):
        """
        Initialize Zerodha API connection
        
        Args:
            api_key: Your Kite Connect API key
            api_secret: Your API secret (for generating access token)
            access_token: Pre-generated access token (if available)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.kite = KiteConnect(api_key=api_key)
        self.access_token = access_token
        
        if access_token:
            self.kite.set_access_token(access_token)
    
    def generate_session(self, request_token):
        """
        Generate session using request token
        
        Args:
            request_token: Request token from login flow
            
        Returns:
            Access token (save this for future use)
        """
        try:
            data = self.kite.generate_session(
                request_token, 
                api_secret=self.api_secret
            )
            self.access_token = data["access_token"]
            self.kite.set_access_token(self.access_token)
            
            print(f"Access token generated: {self.access_token}")
            print("Save this token - it's valid until next trading day")
            
            return self.access_token
        except Exception as e:
            print(f"Error generating session: {str(e)}")
            return None
    
    def get_login_url(self):
        """
        Get login URL for manual authentication
        
        Returns:
            Login URL
        """
        return self.kite.login_url()
    
    def get_profile(self):
        """
        Get user profile
        
        Returns:
            User profile dictionary
        """
        try:
            profile = self.kite.profile()
            return profile
        except Exception as e:
            print(f"Error fetching profile: {str(e)}")
            return None
    
    def get_instruments(self, exchange="NSE"):
        """
        Get list of all instruments for an exchange
        
        Args:
            exchange: Exchange name (NSE, BSE, NFO, etc.)
            
        Returns:
            DataFrame with instrument details
        """
        try:
            instruments = self.kite.instruments(exchange)
            df = pd.DataFrame(instruments)
            return df
        except Exception as e:
            print(f"Error fetching instruments: {str(e)}")
            return None
    
    def get_quote(self, symbols):
        """
        Get live quotes for symbols
        
        Args:
            symbols: List of symbols (e.g., ['NSE:RELIANCE', 'NSE:TCS'])
            
        Returns:
            Dictionary with quote data
        """
        try:
            quotes = self.kite.quote(symbols)
            return quotes
        except Exception as e:
            print(f"Error fetching quotes: {str(e)}")
            return None
    
    def get_ltp(self, symbols):
        """
        Get last traded price for symbols
        
        Args:
            symbols: List of symbols
            
        Returns:
            Dictionary with LTP data
        """
        try:
            ltp = self.kite.ltp(symbols)
            return ltp
        except Exception as e:
            print(f"Error fetching LTP: {str(e)}")
            return None
    
    def get_historical_data(self, instrument_token, from_date, to_date, interval="day"):
        """
        Get historical candlestick data
        
        Args:
            instrument_token: Instrument token (get from instruments list)
            from_date: Start date (datetime object or string 'YYYY-MM-DD')
            to_date: End date
            interval: Candle interval (minute, day, 3minute, 5minute, etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Convert string dates to datetime if needed
            if isinstance(from_date, str):
                from_date = datetime.strptime(from_date, '%Y-%m-%d')
            if isinstance(to_date, str):
                to_date = datetime.strptime(to_date, '%Y-%m-%d')
            
            data = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=interval
            )
            
            df = pd.DataFrame(data)
            return df
        except Exception as e:
            print(f"Error fetching historical data: {str(e)}")
            return None
    
    def place_order(self, symbol, transaction_type, quantity, order_type="MARKET", 
                    price=None, product="CNC", variety="regular"):
        """
        Place an order
        
        Args:
            symbol: Trading symbol (e.g., 'RELIANCE')
            transaction_type: 'BUY' or 'SELL'
            quantity: Number of shares
            order_type: 'MARKET', 'LIMIT', 'SL', 'SL-M'
            price: Price (for LIMIT orders)
            product: 'CNC' (delivery), 'MIS' (intraday), 'NRML' (carry forward)
            variety: 'regular', 'amo', 'co', 'iceberg'
            
        Returns:
            Order ID
        """
        try:
            order_params = {
                'tradingsymbol': symbol,
                'exchange': 'NSE',
                'transaction_type': transaction_type,
                'quantity': quantity,
                'order_type': order_type,
                'product': product,
                'variety': variety
            }
            
            if price:
                order_params['price'] = price
            
            order_id = self.kite.place_order(**order_params)
            print(f"Order placed successfully. Order ID: {order_id}")
            return order_id
        except Exception as e:
            print(f"Error placing order: {str(e)}")
            return None
    
    def get_orders(self):
        """
        Get all orders for the day
        
        Returns:
            List of orders
        """
        try:
            orders = self.kite.orders()
            return orders
        except Exception as e:
            print(f"Error fetching orders: {str(e)}")
            return None
    
    def get_positions(self):
        """
        Get current positions
        
        Returns:
            Dictionary with net and day positions
        """
        try:
            positions = self.kite.positions()
            return positions
        except Exception as e:
            print(f"Error fetching positions: {str(e)}")
            return None
    
    def get_holdings(self):
        """
        Get holdings (long-term investments)
        
        Returns:
            List of holdings
        """
        try:
            holdings = self.kite.holdings()
            return holdings
        except Exception as e:
            print(f"Error fetching holdings: {str(e)}")
            return None
    
    def cancel_order(self, order_id, variety="regular"):
        """
        Cancel an order
        
        Args:
            order_id: Order ID to cancel
            variety: Order variety
            
        Returns:
            Order ID
        """
        try:
            result = self.kite.cancel_order(variety=variety, order_id=order_id)
            print(f"Order {order_id} cancelled successfully")
            return result
        except Exception as e:
            print(f"Error cancelling order: {str(e)}")
            return None


# Example usage
if __name__ == "__main__":
    """
    SETUP PROCESS:
    
    1. Get API credentials from https://kite.trade/
    2. First time login flow:
       - Run this code to get login URL
       - Login through browser
       - Copy request_token from redirect URL
       - Generate access token
       - Save access token for future use
    
    3. Subsequent logins:
       - Use saved access token directly
       - Token is valid until next trading day
    """
    
    # Replace with your API key
    API_KEY = "your_api_key_here"
    API_SECRET = "your_api_secret_here"
    
    # Initialize API
    zerodha = ZerodhaAPI(api_key=API_KEY, api_secret=API_SECRET)
    
    # FIRST TIME LOGIN
    print("\n=== FIRST TIME LOGIN FLOW ===")
    print("1. Visit this URL to login:")
    print(zerodha.get_login_url())
    print("\n2. After login, you'll be redirected to a URL like:")
    print("http://127.0.0.1/?request_token=XXXXXX&action=login&status=success")
    print("\n3. Copy the request_token from URL and use it below:")
    
    # Uncomment and use this after getting request token
    # request_token = "paste_your_request_token_here"
    # access_token = zerodha.generate_session(request_token)
    # print(f"Save this access token: {access_token}")
    
    # SUBSEQUENT LOGINS - Use saved access token
    # zerodha = ZerodhaAPI(api_key=API_KEY, access_token="your_saved_access_token")
    
    # Example API calls (uncomment after authentication)
    """
    # Get profile
    profile = zerodha.get_profile()
    print("\nProfile:", profile)
    
    # Get live quotes
    quotes = zerodha.get_quote(['NSE:RELIANCE', 'NSE:TCS'])
    print("\nQuotes:", quotes)
    
    # Get LTP
    ltp = zerodha.get_ltp(['NSE:INFY', 'NSE:HDFCBANK'])
    print("\nLTP:", ltp)
    
    # Get instruments (to find instrument tokens)
    instruments = zerodha.get_instruments("NSE")
    print("\nSample instruments:")
    print(instruments[instruments['tradingsymbol'] == 'RELIANCE'])
    
    # Get historical data (you need instrument token from instruments list)
    # instrument_token = 738561  # Example: RELIANCE
    # historical = zerodha.get_historical_data(
    #     instrument_token,
    #     from_date='2024-01-01',
    #     to_date='2024-01-31',
    #     interval='day'
    # )
    # print("\nHistorical data:", historical.head())
    
    # Place order (BE CAREFUL - THIS IS REAL TRADING)
    # order_id = zerodha.place_order(
    #     symbol='RELIANCE',
    #     transaction_type='BUY',
    #     quantity=1,
    #     order_type='MARKET',
    #     product='CNC'
    # )
    
    # Get positions
    # positions = zerodha.get_positions()
    # print("\nPositions:", positions)
    
    # Get holdings
    # holdings = zerodha.get_holdings()
    # print("\nHoldings:", holdings)
    """
    
    print("\n⚠️  WARNING: Always test in paper trading mode first!")
    print("⚠️  Start with small amounts when going live!")
    print("⚠️  Set stop losses for all trades!")
