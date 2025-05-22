
import pandas as pd
import numpy as np
from datetime import datetime
import asyncio
import aiohttp
from newsapi import NewsApiClient
import logging
from config import *
from typing import Dict, List, Optional, Tuple
from bs4 import BeautifulSoup
from textblob import TextBlob
import yfinance as yf
import MetaTrader5 as mt5
from fredapi import Fred

class DataFetcher:
    def __init__(self, config=None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.setup_mt5()
        self.news_api = NewsApiClient(api_key=self.config.get('apis', {}).get('news', {}).get('api_key'))
        self.fred = Fred(api_key=self.config.get('apis', {}).get('fred', {}).get('api_key'))
        self.economic_calendar_url = "https://www.forexfactory.com/calendar.php"
        self.sentiment_window = self.config.get('sentiment_window', 7)  # days for sentiment trend analysis
        self.liquidity_threshold = self.config.get('liquidity_threshold', 1000)  # minimum volume for liquidity cluster
        
        # FRED series IDs for key macro indicators
        self.macro_indicators = {
            'gdp': 'GDP',               # Real GDP
            'inflation': 'CPIAUCSL',    # Consumer Price Index
            'interest_rate': 'DFF',     # Federal Funds Rate
            'unemployment': 'UNRATE',    # Unemployment Rate
            'gold_price': 'GOLDAMGBD228NLBM'  # Gold Fixing Price
        }
        
        # Thresholds for market condition scoring
        self.market_thresholds = {
            'gdp_growth': {'bullish': 2.0, 'bearish': 0.0},  # GDP growth rate
            'inflation': {'bullish': (1.5, 2.5), 'bearish': (3.5, float('inf'))},  # CPI YoY
            'interest_rate': {'bullish': (0, 2.5), 'bearish': (4.0, float('inf'))},  # Fed Funds Rate
            'unemployment': {'bullish': (0, 5.0), 'bearish': (6.0, float('inf'))}  # Unemployment rate
        }
        
    
    async def get_tick_data(self):
        """Fetch real-time tick data from MT5 with exponential backoff retry"""
        max_retries = 3
        base_delay = 1  # seconds

        for attempt in range(max_retries):
            try:
                if not mt5.initialize():
                    delay = base_delay * (2 ** attempt)
                    self.logger.warning(f"MT5 connection failed, attempt {attempt + 1}/{max_retries}. Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                    continue

                ticks = mt5.copy_ticks_from(SYMBOL, datetime.now(), 1000, mt5.COPY_TICKS_ALL)
                if ticks is None:
                    raise ConnectionError("Failed to fetch tick data")

                df = pd.DataFrame(ticks)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                return df

            except Exception as e:
                delay = base_delay * (2 ** attempt)
                self.logger.error(f"Error fetching tick data (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
                else:
                    self.logger.error("Max retries reached, tick data fetch failed")
                    return None

        return None

    async def get_order_book(self):
        """Fetch and analyze order book data with advanced metrics"""
        try:
            book = mt5.market_book_get(SYMBOL)
            if book:
                asks = [{'price': order.price, 'volume': order.volume} 
                        for order in book if order.type == mt5.BOOK_TYPE_SELL]
                bids = [{'price': order.price, 'volume': order.volume} 
                        for order in book if order.type == mt5.BOOK_TYPE_BUY]
                
                # Calculate advanced order book metrics
                metrics = self.calculate_order_book_metrics(asks, bids)
                
                return {
                    'asks': asks,
                    'bids': bids,
                    'metrics': metrics
                }
            return None
        except Exception as e:
            self.logger.error(f"Error fetching order book: {e}")
            return None

    def calculate_order_book_metrics(self, asks: List[Dict], bids: List[Dict]) -> Dict:
        """Calculate advanced order book analysis metrics"""
        try:
            # Calculate order book imbalance
            total_ask_volume = sum(order['volume'] for order in asks)
            total_bid_volume = sum(order['volume'] for order in bids)
            imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
            
            # Detect liquidity clusters
            ask_clusters = self.detect_liquidity_clusters(asks)
            bid_clusters = self.detect_liquidity_clusters(bids)
            
            # Calculate depth and spread metrics
            spread = asks[0]['price'] - bids[0]['price'] if asks and bids else 0
            depth_ask = sum(order['volume'] * order['price'] for order in asks[:5])
            depth_bid = sum(order['volume'] * order['price'] for order in bids[:5])
            
            # Detect potential spoofing patterns
            spoofing_alerts = self.detect_spoofing(asks, bids)
            
            return {
                'imbalance': imbalance,
                'ask_clusters': ask_clusters,
                'bid_clusters': bid_clusters,
                'spread': spread,
                'depth_ratio': depth_bid / depth_ask if depth_ask > 0 else 1,
                'spoofing_alerts': spoofing_alerts
            }
        except Exception as e:
            self.logger.error(f"Error calculating order book metrics: {e}")
            return {}

    def detect_liquidity_clusters(self, orders: List[Dict]) -> List[Dict]:
        """Detect significant liquidity clusters in order book"""
        clusters = []
        current_cluster = {'price': 0, 'volume': 0}
        
        for order in orders:
            if current_cluster['volume'] == 0:
                current_cluster = {'price': order['price'], 'volume': order['volume']}
            elif abs(order['price'] - current_cluster['price']) <= 0.0001:  # Price level threshold
                current_cluster['volume'] += order['volume']
            else:
                if current_cluster['volume'] >= self.liquidity_threshold:
                    clusters.append(current_cluster)
                current_cluster = {'price': order['price'], 'volume': order['volume']}
        
        return clusters

    def detect_spoofing(self, asks: List[Dict], bids: List[Dict]) -> List[Dict]:
        """Detect potential spoofing patterns in order book"""
        alerts = []
        
        # Check for large orders far from market price
        mid_price = (asks[0]['price'] + bids[0]['price']) / 2 if asks and bids else 0
        threshold_distance = 0.001  # 10 pips
        
        for side, orders in [('ask', asks), ('bid', bids)]:
            for order in orders:
                price_distance = abs(order['price'] - mid_price) / mid_price
                if price_distance > threshold_distance and order['volume'] > self.liquidity_threshold:
                    alerts.append({
                        'side': side,
                        'price': order['price'],
                        'volume': order['volume'],
                        'distance': price_distance
                    })
        
        return alerts

    async def get_economic_calendar(self) -> Dict:
        """Fetch and parse economic calendar data"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.economic_calendar_url) as response:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Parse calendar events (implementation depends on website structure)
                    events = self.parse_economic_events(soup)
                    
                    # Categorize and analyze events
                    categorized_events = self.categorize_events(events)
                    
                    return categorized_events
        except Exception as e:
            self.logger.error(f"Error fetching economic calendar: {e}")
            return {}

    def parse_economic_events(self, soup: BeautifulSoup) -> List[Dict]:
        """Parse economic calendar events from Forex Factory HTML"""
        events = []
        try:
            calendar_rows = soup.find_all('tr', class_='calendar_row')
            for row in calendar_rows:
                # Extract event time
                time_cell = row.find('td', class_='calendar__time')
                if not time_cell:
                    continue

                # Extract event title and impact
                title_cell = row.find('td', class_='calendar__event')
                impact_cell = row.find('td', class_='calendar__impact')

                if not all([time_cell, title_cell, impact_cell]):
                    continue

                # Parse time (assuming format HH:MM)
                time_str = time_cell.get_text(strip=True)
                if not time_str:
                    continue

                try:
                    event_time = datetime.strptime(time_str, '%H:%M').time()
                    event_datetime = datetime.combine(datetime.now().date(), event_time)
                except ValueError:
                    continue

                # Get event title
                title = title_cell.get_text(strip=True)

                # Determine impact level
                impact = 'low'
                if impact_cell.find('span', class_='high'):
                    impact = 'high'
                elif impact_cell.find('span', class_='medium'):
                    impact = 'medium'

                events.append({
                    'time': event_datetime,
                    'title': title,
                    'impact': impact
                })

        except Exception as e:
            self.logger.error(f"Error parsing economic events: {e}")

        return events

    def categorize_events(self, events: List[Dict]) -> Dict:
        """Categorize and analyze economic events"""
        categorized = {
            'high_impact': [],
            'medium_impact': [],
            'low_impact': [],
            'next_significant': None,
            'time_to_next': None
        }
        
        now = datetime.now()
        
        for event in events:
            if 'gold' in event['title'].lower() or 'precious metals' in event['title'].lower():
                if event['impact'] == 'high':
                    categorized['high_impact'].append(event)
                elif event['impact'] == 'medium':
                    categorized['medium_impact'].append(event)
                else:
                    categorized['low_impact'].append(event)
                
                # Update next significant event
                if event['impact'] in ['high', 'medium'] and event['time'] > now:
                    if not categorized['next_significant'] or event['time'] < categorized['next_significant']['time']:
                        categorized['next_significant'] = event
                        categorized['time_to_next'] = (event['time'] - now).total_seconds() / 3600  # hours
        
        return categorized

    async def get_news(self):
        """Fetch and analyze financial news with enhanced sentiment analysis"""
        try:
            news = self.news_api.get_everything(
                q='gold OR forex OR "federal reserve" OR inflation',
                language='en',
                sort_by='publishedAt'
            )
            
            # Enhance news with sentiment analysis
            analyzed_news = self.analyze_news_sentiment(news['articles'])
            
            # Extract entities and relationships
            entities = self.extract_entities(analyzed_news)
            
            # Analyze temporal patterns
            temporal_analysis = self.analyze_sentiment_trends(analyzed_news)
            
            return {
                'articles': analyzed_news,
                'entities': entities,
                'temporal_analysis': temporal_analysis
            }
        except Exception as e:
            self.logger.error(f"Error fetching news: {e}")
            return None

    def analyze_news_sentiment(self, articles: List[Dict]) -> List[Dict]:
        """Analyze sentiment of news articles"""
        for article in articles:
            blob = TextBlob(article['title'] + ' ' + article['description'])
            article['sentiment'] = {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
        return articles

    def extract_entities(self, articles: List[Dict]) -> Dict:
        """Extract and analyze entities from news articles"""
        entities = {'gold': [], 'central_banks': [], 'economic_indicators': []}
        relationships = []
        
        # Entity extraction implementation
        # This would typically use NLP libraries like spaCy or NLTK
        
        return {
            'entities': entities,
            'relationships': relationships
        }

    def analyze_sentiment_trends(self, articles: List[Dict]) -> Dict:
        """Analyze temporal patterns in sentiment"""
        df = pd.DataFrame(articles)
        df['publishedAt'] = pd.to_datetime(df['publishedAt'])
        df['sentiment'] = df.apply(lambda x: x['sentiment']['polarity'], axis=1)
        
        # Calculate rolling sentiment metrics
        rolling_sentiment = df.set_index('publishedAt')['sentiment'].rolling(f'{self.sentiment_window}D').mean()
        
        return {
            'current_sentiment': df['sentiment'].mean(),
            'sentiment_trend': rolling_sentiment.iloc[-1] if len(rolling_sentiment) > 0 else 0,
            'sentiment_volatility': df['sentiment'].std()
        }

    async def get_alternative_data(self) -> Dict:
        """Fetch alternative data sources including macroeconomic indicators"""
        try:
            # Fetch COT data
            cot_data = await self.get_cot_data()
            
            # Fetch Gold ETF flows
            etf_flows = await self.get_etf_flows()
            
            # Fetch central bank reserves
            reserves = await self.get_central_bank_reserves()
            
            # Fetch macroeconomic data
            macro_data = await self.get_macro_data()
            market_condition = await self.score_market_condition(macro_data)
            
            return {
                'cot_data': cot_data,
                'etf_flows': etf_flows,
                'central_bank_reserves': reserves,
                'macro_data': macro_data,
                'market_condition': market_condition
            }
        except Exception as e:
            self.logger.error(f"Error fetching alternative data: {e}")
            return {}
            
    async def get_macro_data(self) -> Dict:
        """Fetch and preprocess macroeconomic data from FRED"""
        try:
            macro_data = {}
            for indicator, series_id in self.macro_indicators.items():
                series = self.fred.get_series(series_id)
                if series is not None:
                    # Convert to DataFrame and handle frequency
                    df = pd.DataFrame(series)
                    df.columns = ['value']
                    
                    # Calculate growth rates and changes
                    if indicator == 'gdp':
                        df['growth'] = df['value'].pct_change(4) * 100  # Annualized growth
                    elif indicator == 'inflation':
                        df['yoy'] = df['value'].pct_change(12) * 100  # Year-over-year change
                    
                    macro_data[indicator] = df.tail(1).to_dict('records')[0]
            
            # Generate fundamental signals
            signals = self.generate_fundamental_signals(macro_data)
            macro_data['signals'] = signals
            
            return macro_data
        except Exception as e:
            self.logger.error(f"Error fetching macro data: {e}")
            return {}

    def generate_fundamental_signals(self, macro_data: Dict) -> Dict:
        """Generate trading signals based on macroeconomic indicators"""
        try:
            signals = {
                'overall': 'neutral',
                'indicators': {}
            }

            if not macro_data:
                return signals

            # Analyze GDP
            if 'gdp' in macro_data and 'growth' in macro_data['gdp']:
                gdp_growth = macro_data['gdp']['growth']
                if gdp_growth >= self.market_thresholds['gdp_growth']['bullish']:
                    signals['indicators']['gdp'] = 'bullish'
                elif gdp_growth <= self.market_thresholds['gdp_growth']['bearish']:
                    signals['indicators']['gdp'] = 'bearish'
                else:
                    signals['indicators']['gdp'] = 'neutral'

            # Analyze Inflation
            if 'inflation' in macro_data and 'yoy' in macro_data['inflation']:
                inflation = macro_data['inflation']['yoy']
                if self.market_thresholds['inflation']['bullish'][0] <= inflation <= self.market_thresholds['inflation']['bullish'][1]:
                    signals['indicators']['inflation'] = 'bullish'
                elif inflation >= self.market_thresholds['inflation']['bearish'][0]:
                    signals['indicators']['inflation'] = 'bearish'
                else:
                    signals['indicators']['inflation'] = 'neutral'

            # Analyze Interest Rates
            if 'interest_rate' in macro_data:
                rate = macro_data['interest_rate']['value']
                if self.market_thresholds['interest_rate']['bullish'][0] <= rate <= self.market_thresholds['interest_rate']['bullish'][1]:
                    signals['indicators']['interest_rate'] = 'bullish'
                elif rate >= self.market_thresholds['interest_rate']['bearish'][0]:
                    signals['indicators']['interest_rate'] = 'bearish'
                else:
                    signals['indicators']['interest_rate'] = 'neutral'

            # Calculate overall signal
            bullish_count = sum(1 for signal in signals['indicators'].values() if signal == 'bullish')
            bearish_count = sum(1 for signal in signals['indicators'].values() if signal == 'bearish')
            total_indicators = len(signals['indicators'])

            if total_indicators > 0:
                if bullish_count > total_indicators / 2:
                    signals['overall'] = 'bullish'
                elif bearish_count > total_indicators / 2:
                    signals['overall'] = 'bearish'

            return signals

        except Exception as e:
            self.logger.error(f"Error generating fundamental signals: {e}")
            return {'overall': 'neutral', 'indicators': {}}
    
    async def score_market_condition(self, macro_data: Dict) -> Dict:
        """Score market conditions as bullish, neutral, or bearish based on macro indicators"""
        try:
            if not macro_data:
                return {'overall': 'neutral', 'confidence': 0.0}
            
            scores = []
            confidences = []
            
            # Score GDP growth
            if 'gdp' in macro_data and 'growth' in macro_data['gdp']:
                gdp_growth = macro_data['gdp']['growth']
                if gdp_growth >= self.market_thresholds['gdp_growth']['bullish']:
                    scores.append(1)  # Bullish
                elif gdp_growth <= self.market_thresholds['gdp_growth']['bearish']:
                    scores.append(-1)  # Bearish
                else:
                    scores.append(0)  # Neutral
                confidences.append(0.8)  # High confidence in GDP signal
            
            # Score inflation
            if 'inflation' in macro_data and 'yoy' in macro_data['inflation']:
                inflation = macro_data['inflation']['yoy']
                if self.market_thresholds['inflation']['bullish'][0] <= inflation <= self.market_thresholds['inflation']['bullish'][1]:
                    scores.append(1)  # Bullish
                elif inflation >= self.market_thresholds['inflation']['bearish'][0]:
                    scores.append(-1)  # Bearish
                else:
                    scores.append(0)  # Neutral
                confidences.append(0.7)  # Medium-high confidence
            
            # Score interest rates
            if 'interest_rate' in macro_data:
                rate = macro_data['interest_rate']['value']
                if self.market_thresholds['interest_rate']['bullish'][0] <= rate <= self.market_thresholds['interest_rate']['bullish'][1]:
                    scores.append(1)  # Bullish
                elif rate >= self.market_thresholds['interest_rate']['bearish'][0]:
                    scores.append(-1)  # Bearish
                else:
                    scores.append(0)  # Neutral
                confidences.append(0.9)  # High confidence
            
            # Score unemployment
            if 'unemployment' in macro_data:
                unemp = macro_data['unemployment']['value']
                if self.market_thresholds['unemployment']['bullish'][0] <= unemp <= self.market_thresholds['unemployment']['bullish'][1]:
                    scores.append(1)  # Bullish
                elif unemp >= self.market_thresholds['unemployment']['bearish'][0]:
                    scores.append(-1)  # Bearish
                else:
                    scores.append(0)  # Neutral
                confidences.append(0.6)  # Medium confidence
            
            # Calculate weighted average score
            if scores and confidences:
                weighted_score = np.average(scores, weights=confidences)
                overall_confidence = np.mean(confidences)
                
                # Determine overall market condition
                if weighted_score >= 0.5:
                    condition = 'bullish'
                elif weighted_score <= -0.5:
                    condition = 'bearish'
                else:
                    condition = 'neutral'
                
                return {
                    'overall': condition,
                    'confidence': overall_confidence,
                    'weighted_score': weighted_score,
                    'indicators': {
                        'gdp': macro_data.get('gdp', {}),
                        'inflation': macro_data.get('inflation', {}),
                        'interest_rate': macro_data.get('interest_rate', {}),
                        'unemployment': macro_data.get('unemployment', {})
                    }
                }
            
            return {'overall': 'neutral', 'confidence': 0.0}
            
        except Exception as e:
            self.logger.error(f"Error scoring market condition: {e}")
            return {'overall': 'neutral', 'confidence': 0.0}

    async def get_cot_data(self) -> Dict:
        """Fetch and analyze Commitment of Traders data"""
        raise NotImplementedError("COT data fetching is not implemented. Requires CFTC data source integration.")

    async def get_etf_flows(self) -> Dict:
        """Fetch and analyze Gold ETF flows"""
        try:
            # Fetch GLD data using yfinance
            gld = yf.Ticker("GLD")
            hist = gld.history(period="1mo")
            
            # Calculate flow metrics
            flows = {
                'volume': hist['Volume'].mean(),
                'volume_trend': hist['Volume'].pct_change().mean(),
                'price_trend': hist['Close'].pct_change().mean()
            }
            
            return flows
        except Exception as e:
            self.logger.error(f"Error fetching ETF flows: {e}")
            return {}

    async def get_central_bank_reserves(self) -> Dict:
        """Fetch central bank gold reserve changes"""
        raise NotImplementedError("Central bank reserves data fetching is not implemented. Requires World Gold Council API integration.")

    def fetch_historical_data(self, symbol, timeframe, start_date, end_date):
        """Fetch historical OHLCV data from MT5 between specified dates
        
        Args:
            symbol (str): Trading symbol (e.g. 'XAUUSD' or 'XAU')
            timeframe (str): Timeframe for the data (e.g. 'M5', 'M15')
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            pd.DataFrame: Historical OHLCV data
        """
        try:
            # Handle XAU symbol format
            if symbol == 'XAU':
                symbol = 'XAUUSD'
            
            # Convert timeframe string to MT5 timeframe constant
            tf_map = {'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15,
                     'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4}
            mt5_timeframe = tf_map.get(timeframe, mt5.TIMEFRAME_M5)
            
            # Convert dates to datetime
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # Fetch data
            rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_dt, end_dt)
            if rates is None:
                self.logger.error(f"No data returned for {symbol} {timeframe}")
                return pd.DataFrame()
                
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()

    async def stream_data(self, callback):
        """Stream real-time data and process it through callback"""
        while True:
            try:
                # Fetch core market data
                tick_data = await self.get_tick_data()
                order_book = await self.get_order_book()
                news = await self.get_news()
                
                # Fetch additional data sources
                economic_calendar = await self.get_economic_calendar()
                alternative_data = await self.get_alternative_data()
                
                data_package = {
                    'tick_data': tick_data,
                    'order_book': order_book,
                    'news': news,
                    'economic_calendar': economic_calendar,
                    'alternative_data': alternative_data
                }
                
                await callback(data_package)
                await asyncio.sleep(1)  # Rate limiting
                
            except Exception as e:
                self.logger.error(f"Error in data stream: {e}")
                await asyncio.sleep(5)  # Backoff on error

    def __del__(self):
        """Cleanup MT5 connection"""
        mt5.shutdown()

    def setup_mt5(self):
        """Initialize MetaTrader5 connection with error handling"""
        try:
            # Get MT5 credentials from config
            mt5_config = self.config.get('mt5', {})
            credentials = mt5_config.get('credentials', {})
            connection = mt5_config.get('connection', {})
            
            # Initialize with credentials
            if not mt5.initialize(
                login=int(credentials.get('login')),
                password=credentials.get('password'),
                server=credentials.get('server'),
                timeout=connection.get('timeout', 60000)
            ):
                self.logger.error("Failed to initialize MetaTrader5")
                return False
            
            # Enable market data
            symbol = mt5_config.get('symbol', {}).get('name', 'XAU')
            if not mt5.market_book_add(symbol):
                self.logger.error(f"Failed to subscribe to market data for {symbol}")
                mt5.shutdown()
                return False
                
            self.logger.info("MetaTrader5 initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting up MetaTrader5: {e}")
            if mt5.initialize():
                mt5.shutdown()
            return False