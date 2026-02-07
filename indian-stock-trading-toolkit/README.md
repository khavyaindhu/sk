# 🚀 AI-Powered Indian Stock Trading Toolkit

A comprehensive Python toolkit for algorithmic trading in the Indian stock market (NSE/BSE) with ML-based predictions, technical analysis, backtesting, and broker integration.

## 📋 Features

- 📊 **Data Fetching**: Historical and live data from yfinance, NSE, and broker APIs
- 📈 **Technical Analysis**: 15+ indicators (SMA, EMA, RSI, MACD, Bollinger Bands, etc.)
- 🎯 **Trading Strategies**: Pre-built strategies (MA Crossover, RSI, MACD, Combined)
- 🔄 **Backtesting Engine**: Test strategies on historical data with metrics
- 🤖 **ML Predictions**: Random Forest based price prediction
- 🔌 **Broker Integration**: Connect to Zerodha, Upstox, Angel One
- 📉 **Performance Metrics**: Sharpe ratio, win rate, max drawdown, etc.

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- VS Code (recommended)
- Trading account with API access (for live trading)

### Step 1: Clone or Download

```bash
# Create project directory
mkdir stock-trading
cd stock-trading

# Copy all files from this repository
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

**Note on TA-Lib**: If TA-Lib installation fails:
```bash
# For Windows: Download and install from
# https://github.com/cgohlke/talib-build/releases
pip install TA_Lib-0.4.xx-cpxxx-cpxxx-win_amd64.whl

# For Mac:
brew install ta-lib
pip install ta-lib

# For Linux:
sudo apt-get install ta-lib
pip install ta-lib

# If still having issues, use pandas-ta as alternative
pip install pandas-ta
```

## 🚦 Quick Start

### 1. Basic Stock Analysis

```python
from utils.data_fetcher import IndianStockDataFetcher
from utils.indicators import TechnicalIndicators

# Fetch data
fetcher = IndianStockDataFetcher()
df = fetcher.get_historical_data_yfinance(
    'RELIANCE', 
    start_date='2023-01-01', 
    end_date='2024-01-01'
)

# Add technical indicators
df = TechnicalIndicators.add_all_indicators(df)

# View data
print(df[['close', 'SMA_20', 'RSI', 'MACD']].tail())
```

### 2. Backtest a Strategy

```python
from strategies.example_strategies import MovingAverageCrossover
from backtesting.engine import BacktestEngine

# Create strategy
strategy = MovingAverageCrossover(fast_period=20, slow_period=50)

# Generate signals
signals = strategy.generate_signals(df)

# Backtest
engine = BacktestEngine(initial_capital=100000)
metrics = engine.run_backtest(df, signals)

# View results
engine.print_results(metrics)
```

### 3. ML Price Prediction

```python
from ml_models.price_predictor import StockPricePredictor

# Initialize predictor
predictor = StockPricePredictor()

# Create features and train
df_features = predictor.create_features(df, lookback_period=5)
X_train, X_test, y_train, y_test = predictor.prepare_data(df_features)
predictor.train_random_forest(X_train, y_train)

# Evaluate
metrics, predictions = predictor.evaluate(X_test, y_test)
print(metrics)
```

### 4. Run Complete Analysis

```bash
# Run the main script
python main.py
```

## 📁 Project Structure

```
stock-trading/
│
├── utils/
│   ├── data_fetcher.py      # Fetch historical/live data
│   └── indicators.py         # Technical indicators
│
├── strategies/
│   └── example_strategies.py # Pre-built strategies
│
├── backtesting/
│   └── engine.py             # Backtesting framework
│
├── ml_models/
│   └── price_predictor.py    # ML prediction models
│
├── broker_integration/
│   └── zerodha_api.py        # Zerodha Kite Connect
│
├── data/                     # Store CSV files
│   ├── historical/
│   └── live/
│
├── main.py                   # Main starter script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🔐 Broker API Setup (Zerodha Example)

### Step 1: Get API Credentials

1. Go to https://kite.trade/
2. Create developer account (₹2000/month subscription)
3. Create an app to get API key and secret
4. Save these credentials securely

### Step 2: First-Time Login

```python
from broker_integration.zerodha_api import ZerodhaAPI

# Initialize with your credentials
zerodha = ZerodhaAPI(
    api_key="your_api_key",
    api_secret="your_api_secret"
)

# Get login URL
print(zerodha.get_login_url())
# Visit this URL and login

# After login, get request_token from redirect URL
request_token = "paste_token_here"

# Generate access token
access_token = zerodha.generate_session(request_token)
print(f"Save this token: {access_token}")
```

### Step 3: Use Saved Token

```python
# Use saved access token for subsequent sessions
zerodha = ZerodhaAPI(
    api_key="your_api_key",
    access_token="your_saved_access_token"
)

# Get live price
price = zerodha.get_ltp(['NSE:RELIANCE'])
print(price)
```

## 📊 Available Strategies

1. **Moving Average Crossover**
   - Buy: Fast MA crosses above slow MA
   - Sell: Fast MA crosses below slow MA

2. **RSI Strategy**
   - Buy: RSI < 30 (oversold)
   - Sell: RSI > 70 (overbought)

3. **MACD Strategy**
   - Buy: MACD crosses above signal line
   - Sell: MACD crosses below signal line

4. **Bollinger Bands**
   - Buy: Price touches lower band
   - Sell: Price touches upper band

5. **Combined Strategy**
   - Uses multiple indicators for confirmation

## ⚠️ IMPORTANT WARNINGS

### Risk Disclosure
- **NEVER trade with money you can't afford to lose**
- Start with paper trading (simulated)
- Test strategies thoroughly before going live
- Past performance doesn't guarantee future results
- Stock market trading involves significant risk

### Best Practices

1. **Testing Phase**
   - Backtest for at least 2-3 years of data
   - Test across different market conditions
   - Account for transaction costs and slippage
   - Use walk-forward optimization

2. **Risk Management**
   - Never risk more than 1-2% per trade
   - Always use stop losses
   - Diversify across multiple stocks
   - Don't over-leverage

3. **Paper Trading First**
   - Test with simulated money for 3-6 months
   - Track all trades as if real
   - Refine strategy based on results

4. **When Going Live**
   - Start with minimal capital
   - Gradually increase position sizes
   - Keep detailed logs
   - Review and adjust regularly

### SEBI Regulations
- Algorithmic trading requires broker approval
- Maintain audit trails of all trades
- Follow exchange risk controls
- Different rules for retail vs institutional

## 🎓 Learning Path

### Week 1-2: Foundation
- Learn data fetching and manipulation
- Understand technical indicators
- Practice with historical data

### Week 3-4: Strategy Development
- Implement simple strategies
- Learn backtesting
- Understand metrics

### Week 5-8: Machine Learning
- Feature engineering
- Model training
- Performance evaluation

### Week 9-12: Paper Trading
- Connect to broker API
- Test with real-time data
- Refine strategies

### Week 13+: Live Trading
- Start with small capital
- Monitor closely
- Continuously improve

## 📚 Resources

### Documentation
- [Kite Connect API Docs](https://kite.trade/docs/connect/v3/)
- [yfinance Documentation](https://pypi.org/project/yfinance/)
- [pandas Documentation](https://pandas.pydata.org/docs/)
- [scikit-learn Documentation](https://scikit-learn.org/)

### Indian Market Specific
- [NSE India](https://www.nseindia.com/)
- [BSE India](https://www.bseindia.com/)
- [SEBI Guidelines](https://www.sebi.gov.in/)

### Learning
- Quantitative Trading courses on Coursera/Udemy
- "Algorithmic Trading" by Ernest Chan
- NSE's certification courses

## 🐛 Troubleshooting

### Common Issues

**1. TA-Lib installation fails**
```bash
# Use pandas-ta as alternative
pip uninstall ta-lib
pip install pandas-ta

# Update indicators.py to use pandas-ta
```

**2. yfinance data incomplete**
```python
# Try different ticker format
# Instead of 'RELIANCE', try 'RELIANCE.NS' or 'RELIANCE.BO'
```

**3. Broker API connection issues**
```python
# Check if access token is valid (expires daily)
# Regenerate token if needed
```

**4. Memory issues with large datasets**
```python
# Process data in chunks
# Reduce date range
# Use only required columns
```

## 🤝 Contributing

Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Share your strategies (after backtesting!)

## 📄 License

This is educational software. Use at your own risk.

## ⚡ Next Steps

1. ✅ Install all dependencies
2. ✅ Run `main.py` to see example analysis
3. ✅ Modify strategies to suit your needs
4. ✅ Backtest thoroughly
5. ✅ Paper trade for several months
6. ⚠️ Only then consider live trading with small amounts

## 💬 Support

For questions or issues:
- Check existing documentation
- Review example code
- Test with small datasets first
- Seek professional financial advice before trading

---

**Remember**: This toolkit is for educational purposes. Always do your own research and never invest more than you can afford to lose.

**Disclaimer**: I am not a financial advisor. This is not financial advice. Trading involves significant risk.
