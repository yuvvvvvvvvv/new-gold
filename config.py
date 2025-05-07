# Trading Bot Configuration

# API Credentials
MT5_LOGIN = "your_login"
MT5_PASSWORD = "your_password"
MT5_SERVER = "your_broker_server"

# News API Configuration
NEWS_API_KEY = "6d982cfcfb2c44448987450cad3b404a"

# Trading Parameters
SYMBOL = "XAUUSD"  # Gold symbol
TIMEFRAME = "5M"    # 5-minute timeframe

# Technical Analysis Parameters
EMA_FAST = 50
EMA_SLOW = 200
ADX_PERIOD = 14
ADX_THRESHOLD = 25
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
BB_PERIOD = 20
BB_STD = 2
FIB_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786]

# Risk Management
MAX_RISK_PER_TRADE = 0.02  # 2% max risk per trade
ATR_PERIOD = 14
ATR_MULTIPLIER = 2
VOLATILITY_THRESHOLD = 0.03  # 3% threshold for circuit breaker

# AI Model Parameters
LSTM_SEQUENCE_LENGTH = 17  # Match input dimension with feature count
LSTM_FEATURES = ["close", "volume", "rsi", "macd", "adx"]
LSTM_UNITS = 50
LSTM_EPOCHS = 100
LSTM_BATCH_SIZE = 32

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FILE = "trading_bot.log"

# Telegram Alerts
TELEGRAM_TOKEN = "7928421988:AAGsDv7ciorTOW8u56JTx1OOgHdsGsN9odU"
TELEGRAM_CHAT_ID = "6868805694"