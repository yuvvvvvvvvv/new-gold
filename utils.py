import numpy as np
import pandas as pd
import logging
from typing import Optional

def engineer_features(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Engineer features for time series data including technical indicators,
    cyclical time features, and market regime indicators.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data and datetime index
        
    Returns:
        pd.DataFrame: DataFrame with engineered features
        
    Raises:
        Exception: If feature engineering fails
    """
    try:
        # Cyclical encoding of time features
        df['hour_sin'] = np.sin(2 * np.pi * pd.to_datetime(df.index).hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * pd.to_datetime(df.index).hour / 24)
        df['day_sin'] = np.sin(2 * np.pi * pd.to_datetime(df.index).dayofweek / 7)
        df['day_cos'] = np.cos(2 * np.pi * pd.to_datetime(df.index).dayofweek / 7)
        df['month_sin'] = np.sin(2 * np.pi * pd.to_datetime(df.index).month / 12)
        df['month_cos'] = np.cos(2 * np.pi * pd.to_datetime(df.index).month / 12)
        
        # Technical indicator ratios
        df['price_volatility'] = df['high'] / df['low'] - 1
        df['volume_price_trend'] = df['volume'] * (df['close'] - df['open'])
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Volatility regime features
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['volatility_regime'] = np.where(df['volatility'] > df['volatility'].mean(), 1, 0)
        
        # Price action patterns
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Additional technical indicators
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['rsi'] = calculate_rsi(df['close'])
        df['macd'] = calculate_macd(df['close'])
        
        return df.fillna(0)
        
    except Exception as e:
        logging.error(f"Error engineering features: {e}")
        return None

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI)
    
    Args:
        prices (pd.Series): Price series
        period (int): RSI period
        
    Returns:
        pd.Series: RSI values
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    """
    Calculate Moving Average Convergence Divergence (MACD)
    
    Args:
        prices (pd.Series): Price series
        fast (int): Fast EMA period
        slow (int): Slow EMA period
        
    Returns:
        pd.Series: MACD line
    """
    fast_ema = prices.ewm(span=fast, adjust=False).mean()
    slow_ema = prices.ewm(span=slow, adjust=False).mean()
    return fast_ema - slow_ema