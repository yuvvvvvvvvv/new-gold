import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from pathlib import Path
import MetaTrader5 as mt5
from fredapi import Fred
from newsapi import NewsApiClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from data_fetcher import DataFetcher
from backtesting import BacktestEngine
from utils import engineer_features
import h5py

class DataPreparation:
    def prepare_training_data(self, config: Dict) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Prepare training data for machine learning models.

        Args:
            config (Dict): Configuration dictionary from ConfigLoader.load_config()

        Returns:
            Dict[str, Dict[str, pd.DataFrame]]: Dictionary containing train, validation,
            and test DataFrames for each timeframe.

        Raises:
            ConnectionError: If MT5 connection fails
            ValueError: If data validation fails
            Exception: For other unexpected errors
        """
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename='trading_bot.log'
        )
        logger = logging.getLogger(__name__)
        
        try:
            # Initialize data fetcher and ensure MT5 connection
            data_fetcher = DataFetcher()
            if not mt5.initialize(
                login=config['mt5']['credentials']['login'],
                password=config['mt5']['credentials']['password'],
                server=config['mt5']['credentials']['server']
            ):
                raise ConnectionError("Failed to connect to MT5")
            
            logger.info("MT5 connection established successfully")
            
            # Define timeframes and date range
            timeframes = {
                'M5': mt5.TIMEFRAME_M5,
                'M15': mt5.TIMEFRAME_M15,
                'H1': mt5.TIMEFRAME_H1,
                'H4': mt5.TIMEFRAME_H4
            }
            start_date = datetime(2020, 1, 1)
            end_date = datetime(2025, 5, 1)
            
            # Initialize result dictionary
            result_data = {
                'train': {},
                'val': {},
                'test': {}
            }
            
            # Fetch and process data for each timeframe
            for tf_name, tf_value in timeframes.items():
                logger.info(f"Processing {tf_name} timeframe data")
                
                try:
                    # Fetch OHLCV data
                    ohlcv_data = data_fetcher.fetch_mt5_data(
                        'XAUUSD',
                        tf_value,
                        start_date,
                        end_date
                    )
                    
                    # Fetch macroeconomic data
                    fred = Fred(api_key=config['apis']['fred']['api_key'])
                    macro_data = self._fetch_macro_data(fred, start_date, end_date)
                    
                    # Fetch and analyze news sentiment
                    news_data = self._fetch_news_sentiment(
                        config['apis']['news']['api_key'],
                        start_date,
                        end_date
                    )
                    
                    # Generate placeholder economic calendar data
                    economic_data = self._generate_economic_calendar(start_date, end_date)
                    
                    # Engineer features
                    features_df = engineer_features(
                        ohlcv_data,
                        config['feature_engineering']['technical_indicators']
                    )
                    
                    # Combine all data sources
                    combined_df = self._combine_data(
                        features_df,
                        macro_data,
                        news_data,
                        economic_data
                    )
                    
                    # Handle missing values
                    combined_df = self._handle_missing_values(combined_df)
                    
                    # Split data
                    splits = BacktestEngine.split_data(combined_df)
                    
                    # Save to HDF5
                    self._save_to_hdf5(splits, tf_name)
                    
                    # Store in result dictionary
                    result_data['train'][tf_name] = splits['train']
                    result_data['val'][tf_name] = splits['val']
                    result_data['test'][tf_name] = splits['test']
                    
                except Exception as e:
                    logger.error(f"Error processing {tf_name} timeframe: {e}")
                    continue
            
            return result_data
            
        except Exception as e:
            logger.error(f"Critical error in data preparation: {e}")
            raise
        
        finally:
            mt5.shutdown()
    
    def _fetch_macro_data(self, fred: Fred, start_date: datetime,
                         end_date: datetime) -> pd.DataFrame:
        """Fetch macroeconomic data from FRED."""
        indicators = {
            'DFF': 'fed_funds_rate',
            'CPIAUCSL': 'cpi',
            'UNRATE': 'unemployment_rate',
            'DTWEXB': 'usd_index'
        }
        
        macro_df = pd.DataFrame()
        for series_id, name in indicators.items():
            try:
                series = fred.get_series(
                    series_id,
                    start_date,
                    end_date
                )
                macro_df[name] = series
            except Exception as e:
                logging.warning(f"Failed to fetch {series_id}: {e}")
                macro_df[name] = np.nan
        
        return macro_df.fillna(method='ffill')
    
    def _fetch_news_sentiment(self, api_key: str, start_date: datetime,
                            end_date: datetime) -> pd.DataFrame:
        """Fetch and analyze news sentiment using FinBERT."""
        newsapi = NewsApiClient(api_key=api_key)
        sentiment_model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
        tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
        
        sentiments = []
        current_date = start_date
        
        while current_date <= end_date:
            try:
                articles = newsapi.get_everything(
                    q='gold OR XAUUSD OR precious metals',
                    from_param=current_date.strftime('%Y-%m-%d'),
                    to=min(current_date + timedelta(days=7), end_date).strftime('%Y-%m-%d'),
                    language='en'
                )
                
                for article in articles['articles']:
                    text = f"{article['title']} {article['description']}"
                    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
                    
                    with torch.no_grad():
                        outputs = sentiment_model(**inputs)
                        scores = torch.softmax(outputs.logits, dim=1)
                        sentiment = scores.argmax().item() - 1  # -1 (negative) to 1 (positive)
                    
                    sentiments.append({
                        'date': article['publishedAt'],
                        'sentiment': sentiment
                    })
                
                current_date += timedelta(days=7)
                
            except Exception as e:
                logging.warning(f"Error fetching news for {current_date}: {e}")
                current_date += timedelta(days=1)
        
        sentiment_df = pd.DataFrame(sentiments)
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        return sentiment_df.set_index('date').resample('1D').mean()
    
    def _generate_economic_calendar(self, start_date: datetime,
                                  end_date: datetime) -> pd.DataFrame:
        """Generate placeholder economic calendar data."""
        dates = pd.date_range(start_date, end_date, freq='1D')
        events = pd.DataFrame(index=dates)
        
        # Simulate important economic events
        events['high_impact'] = np.random.choice([0, 1], size=len(dates), p=[0.9, 0.1])
        events['medium_impact'] = np.random.choice([0, 1], size=len(dates), p=[0.8, 0.2])
        
        return events
    
    def _combine_data(self, features_df: pd.DataFrame, macro_df: pd.DataFrame,
                     news_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
        """Combine all data sources into a single DataFrame."""
        # Ensure all DataFrames have datetime index
        dfs = [features_df, macro_df, news_df, events_df]
        for df in dfs:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
        
        # Combine all data sources
        combined = features_df.copy()
        for df in [macro_df, news_df, events_df]:
            # Reindex to match features_df frequency
            resampled = df.reindex(features_df.index, method='ffill')
            # Add prefix to avoid column name conflicts
            prefix = 'macro_' if df is macro_df else 'news_' if df is news_df else 'event_'
            resampled.columns = [f"{prefix}{col}" for col in resampled.columns]
            combined = combined.join(resampled)
        
        return combined
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the combined DataFrame."""
        # Forward fill price and indicator data
        price_cols = ['open', 'high', 'low', 'close', 'volume']
        df[price_cols] = df[price_cols].fillna(method='ffill')
        
        # Zero-fill sentiment and event data
        sentiment_cols = [col for col in df.columns if 'sentiment' in col or 'event' in col]
        df[sentiment_cols] = df[sentiment_cols].fillna(0)
        
        # Forward fill remaining columns (e.g., technical indicators)
        df = df.fillna(method='ffill')
        
        return df
    
    def _save_to_hdf5(self, splits: Dict[str, pd.DataFrame], timeframe: str) -> None:
        """Save data splits to HDF5 files."""
        data_dir = Path('data')
        data_dir.mkdir(exist_ok=True)
        
        for split_name, split_data in splits.items():
            file_path = data_dir / f"{split_name}_{timeframe}.h5"
            with h5py.File(file_path, 'w') as f:
                for column in split_data.columns:
                    f.create_dataset(
                        column,
                        data=split_data[column].values,
                        compression='gzip',
                        compression_opts=9
                    )
                
                # Save index as timestamps
                f.create_dataset(
                    'index',
                    data=split_data.index.astype(np.int64),
                    compression='gzip',
                    compression_opts=9
                )