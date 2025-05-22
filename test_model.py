import pandas as pd
import numpy as np
from ai_model import AIModel
from datetime import datetime, timedelta
import unittest
import logging
import MetaTrader5 as mt5

def generate_sample_data(n_samples=1000):
    """Generate real historical gold data from MetaTrader 5"""
    try:
        # Initialize MT5 connection
        if not mt5.initialize():
            logging.error("Failed to initialize MetaTrader 5")
            return None
            
        # Get XAUUSD hourly data
        symbol = "XAUUSD"
        timeframe = mt5.TIMEFRAME_H1
        current_time = datetime.now()
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_samples)
        dates = pd.date_range(end=datetime.now(), periods=n_samples, freq='H')
    
        if rates is None:
            logging.error("Failed to get XAUUSD data from MetaTrader 5")
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Calculate technical indicators
        # RSI
        delta = df['close'].diff()
        gain = (delta > 0) * delta
        loss = (delta < 0) * -delta
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        
        # ADX
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        dx = tr.rolling(window=14).mean()
        df['adx'] = dx.rolling(window=14).mean()
        
        # Fill NaN values
        df.fillna(method='bfill', inplace=True)
        
        mt5.shutdown()
        return df
        
    except Exception as e:
        logging.error(f"Error generating sample data: {e}")
        if mt5.initialize():
            mt5.shutdown()
        return None
    
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

class TestAIModel(unittest.TestCase):
    """Unit tests for AIModel class"""
    
    @classmethod
    def setUpClass(cls):
        cls.model = AIModel()
        cls.test_data = generate_sample_data(1000)
        cls.test_news = generate_sample_news()
        
    def test_predict_price_normal(self):
        """Test price prediction with normal data"""
        predictions = self.model.predict_price(self.test_data)
        self.assertIsNotNone(predictions)
        self.assertIn('prediction', predictions)
        self.assertIn('confidence_interval', predictions)
        
    def test_predict_price_empty(self):
        """Test price prediction with empty data"""
        empty_df = pd.DataFrame()
        predictions = self.model.predict_price(empty_df)
        self.assertIsNone(predictions)
        
    def test_generate_signals_normal(self):
        """Test signal generation with normal data"""
        signals = self.model.generate_signals(self.test_data, self.test_news)
        self.assertIsNotNone(signals)
        self.assertIn('ppo_action', signals)
        self.assertIn('ppo_confidence', signals)
        
    def test_generate_signals_invalid_news(self):
        """Test signal generation with invalid news"""
        signals = self.model.generate_signals(self.test_data, [])
        self.assertIsNotNone(signals)
        self.assertNotIn('sentiment', signals)
        
    def test_train_ppo_normal(self):
        """Test PPO training with normal data"""
        result = self.model.train_ppo(self.test_data)
        self.assertTrue(result)
        self.assertGreater(len(self.model.reward_history), 0)
        
    def test_model_accuracy(self):
        """Test model prediction accuracy"""
        # Train models first
        self.model.train_models(self.test_data)
        
        # Get last 100 candles for testing
        test_data = self.test_data.iloc[-100:]
        actual_prices = test_data['close'].values
        predicted_prices = []
        
        # Generate predictions
        for i in range(len(test_data)-1):
            pred = self.model.predict_price(test_data.iloc[:i+1])
            if pred:
                predicted_prices.append(pred['prediction'])
        
        # Calculate accuracy metrics
        mse = np.mean((actual_prices[-len(predicted_prices):] - predicted_prices) ** 2)
        self.assertLess(mse, 1000)  # Adjust threshold as needed

def run_unit_tests():
    """Run all unit tests and log results"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAIModel)
    
    # Run tests and capture results
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Log test results
    logger.info(f"Tests run: {result.testsRun}")
    logger.info(f"Tests passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    logger.info(f"Tests failed: {len(result.failures)}")
    logger.info(f"Tests with errors: {len(result.errors)}")

def main():
    print("Initializing AI Model...")
    model = AIModel()
    
    # Generate sample data
    print("Generating sample data...")
    df = generate_sample_data()
    if df is None:
        print("Failed to generate sample data")
        return
        
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
    
    # Run unit tests
    print("\nRunning unit tests...")
    run_unit_tests()

if __name__ == "__main__":
    main()