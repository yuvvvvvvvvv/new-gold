import pandas as pd
import numpy as np
from ai_model import AIModel
from datetime import datetime, timedelta

def generate_sample_data(n_samples=1000):
    """Generate sample financial data for testing"""
    dates = pd.date_range(end=datetime.now(), periods=n_samples, freq='H')
    
    # Generate synthetic price data
    close = pd.Series(np.random.normal(100, 10, n_samples).cumsum() + 1000)
    volume = pd.Series(np.random.normal(1000000, 100000, n_samples))
    
    # Calculate RSI
    delta = close.diff()
    gain = (delta > 0) * delta
    loss = (delta < 0) * -delta
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    
    # Calculate ADX
    high = close + pd.Series(np.random.normal(2, 0.5, n_samples))
    low = close - pd.Series(np.random.normal(2, 0.5, n_samples))
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    dx = tr.rolling(window=14).mean()
    adx = dx.rolling(window=14).mean()
    
    # Create DataFrame
    df = pd.DataFrame({
        'close': close,
        'volume': volume,
        'high': high,
        'low': low,
        'open': close - pd.Series(np.random.normal(1, 0.2, n_samples)),
        'rsi': rsi.fillna(50),
        'macd': macd.fillna(0),
        'adx': adx.fillna(25)
    }, index=dates)
    
    return df

def generate_sample_news():
    """Generate sample financial news for testing"""
    return [
        "Company XYZ reports strong quarterly earnings, beating market expectations.",
        "Federal Reserve maintains current interest rates, signals potential future adjustments.",
        "Market analysts predict positive outlook for technology sector.",
        "Global economic indicators show signs of stability and growth.",
        "New regulatory framework announced for cryptocurrency trading."
    ]

def main():
    print("Initializing AI Model...")
    model = AIModel()
    
    # Generate sample data
    print("Generating sample data...")
    df = generate_sample_data()
    news_texts = generate_sample_news()
    
    # Train models
    print("Training models...")
    model.train_models(df)
    
    # Train PPO agent
    print("Training PPO agent...")
    model.train_ppo(df)
    
    # Generate predictions and signals
    print("\nGenerating predictions and signals...")
    predictions = model.predict_price(df)
    if predictions:
        print(f"Price Prediction: {predictions['prediction']:.2f}")
        print(f"Confidence Interval: {predictions['confidence_interval']['lower']:.2f} - "
              f"{predictions['confidence_interval']['upper']:.2f}")
    
    signals = model.generate_signals(df, news_texts)
    if signals:
        print("\nTrading Signals:")
        print(f"PPO Action: {signals['ppo_action']}")
        print(f"PPO Confidence: {signals['ppo_confidence']:.2f}")
        if 'sentiment' in signals:
            print(f"Sentiment Score: {signals['sentiment']:.2f}")
        if 'model_confidence' in signals:
            print(f"Model Confidence: {signals['model_confidence']:.2f}")

if __name__ == "__main__":
    main()