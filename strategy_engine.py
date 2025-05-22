import numpy as np
import pandas as pd
import talib as ta
from sklearn.preprocessing import StandardScaler
from config import *
import logging
from scipy.stats import linregress
from collections import defaultdict

class StrategyEngine:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.seasonal_stats = defaultdict(lambda: defaultdict(list))
        self.historical_atr = []
        self.bb_width_history = []
        
    def calculate_indicators(self, df):
        """Calculate technical indicators using shared engineer_features function"""
        try:
            # Calculate base technical indicators
            df['ema_fast'] = ta.EMA(df['close'], timeperiod=EMA_FAST)
            df['ema_slow'] = ta.EMA(df['close'], timeperiod=EMA_SLOW)
            df['macd'], df['macd_signal'], df['macd_hist'] = ta.MACD(df['close'])
            df['adx'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=ADX_PERIOD)
            df['rsi'] = ta.RSI(df['close'], timeperiod=RSI_PERIOD)
            upper, middle, lower = ta.BBANDS(df['close'], timeperiod=BB_PERIOD, nbdevup=BB_STD, nbdevdn=BB_STD)
            df['bb_upper'] = upper
            df['bb_middle'] = middle
            df['bb_lower'] = lower
            df['atr'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=ATR_PERIOD)
            
            # Add engineered features from AI model
            try:
                from ai_model import engineer_features
                df = engineer_features(df)
                self.logger.info("Successfully added AI engineered features")
            except Exception as e:
                self.logger.warning(f"Could not add AI engineered features: {e}")
            
            return df
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            return None

    def detect_trend_signals(self, df):
        """Detect trend following signals"""
        signals = []
        
        # EMA Crossover
        ema_cross = (df['ema_fast'] > df['ema_slow']) & (df['adx'] > ADX_THRESHOLD)
        
        # MACD Momentum
        macd_momentum = df['macd_hist'] > 0
        
        if ema_cross.iloc[-1] and macd_momentum.iloc[-1]:
            signals.append(('TREND', 'LONG'))
        elif (not ema_cross.iloc[-1]) and (not macd_momentum.iloc[-1]):
            signals.append(('TREND', 'SHORT'))
            
        return signals

    def detect_mean_reversion_signals(self, df):
        """Detect mean reversion signals"""
        signals = []
        
        # Bollinger Bands
        bb_touch_lower = df['close'].iloc[-1] <= df['bb_lower'].iloc[-1]
        bb_touch_upper = df['close'].iloc[-1] >= df['bb_upper'].iloc[-1]
        
        # RSI Divergence
        rsi_oversold = df['rsi'].iloc[-1] < RSI_OVERSOLD
        rsi_overbought = df['rsi'].iloc[-1] > RSI_OVERBOUGHT
        
        if bb_touch_lower and rsi_oversold:
            signals.append(('MEAN_REVERSION', 'LONG'))
        elif bb_touch_upper and rsi_overbought:
            signals.append(('MEAN_REVERSION', 'SHORT'))
            
        return signals

    def detect_breakout_signals(self, df):
        """Enhanced breakout detection using dynamic Fibonacci levels and harmonic patterns with volume confirmation"""
        try:
            signals = []
            
            # Calculate dynamic Fibonacci levels using recent swing high/low
            lookback = 20  # Period for recent price action
            recent_high = df['high'].tail(lookback).max()
            recent_low = df['low'].tail(lookback).min()
            diff = recent_high - recent_low
            
            # Dynamic Fibonacci levels
            fib_levels = {
                0: recent_low,
                0.236: recent_low + 0.236 * diff,
                0.382: recent_low + 0.382 * diff,
                0.5: recent_low + 0.5 * diff,
                0.618: recent_low + 0.618 * diff,
                0.786: recent_low + 0.786 * diff,
                1: recent_high
            }
            
            current_price = df['close'].iloc[-1]
            prev_price = df['close'].iloc[-2]
            
            # Volume confirmation threshold
            volume_threshold = df['volume'].rolling(20).mean() * 1.5
            strong_volume = df['volume'].iloc[-1] > volume_threshold.iloc[-1]
            
            # Detect Fibonacci pattern breakouts with volume confirmation
            for level, price in fib_levels.items():
                if strong_volume:
                    if prev_price < price < current_price:
                        signals.append(('BREAKOUT', 'LONG', f'FIB_{level}'))
                    elif prev_price > price > current_price:
                        signals.append(('BREAKOUT', 'SHORT', f'FIB_{level}'))
            
            # Detect harmonic patterns
            self.detect_harmonic_patterns(df, signals)
            
            return signals
        except Exception as e:
            self.logger.error(f"Error detecting breakout signals: {e}")
            return []
            
    def detect_harmonic_patterns(self, df, signals):
        """Detect harmonic patterns (Gartley, Butterfly, Bat)"""
        try:
            # Get recent swing points
            highs = df['high'].tail(50)
            lows = df['low'].tail(50)
            
            # Find potential XABCD points
            swings = self.find_swing_points(highs, lows)
            if len(swings) < 5:
                return
                
            # Check for Gartley pattern
            if self.is_gartley_pattern(swings):
                if swings[-1]['type'] == 'low':
                    signals.append(('HARMONIC', 'LONG', 'GARTLEY'))
                else:
                    signals.append(('HARMONIC', 'SHORT', 'GARTLEY'))
                    
            # Check for Butterfly pattern
            if self.is_butterfly_pattern(swings):
                if swings[-1]['type'] == 'low':
                    signals.append(('HARMONIC', 'LONG', 'BUTTERFLY'))
                else:
                    signals.append(('HARMONIC', 'SHORT', 'BUTTERFLY'))
                    
        except Exception as e:
            self.logger.error(f"Error detecting harmonic patterns: {e}")
            
    def find_swing_points(self, highs, lows):
        """Find swing high and low points"""
        swings = []
        for i in range(2, len(highs)-2):
            # Swing high
            if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i-2] and \
               highs.iloc[i] > highs.iloc[i+1] and highs.iloc[i] > highs.iloc[i+2]:
                swings.append({'price': highs.iloc[i], 'type': 'high'})
            # Swing low
            if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i-2] and \
               lows.iloc[i] < lows.iloc[i+1] and lows.iloc[i] < lows.iloc[i+2]:
                swings.append({'price': lows.iloc[i], 'type': 'low'})
        return sorted(swings, key=lambda x: x['price'])
        
    def is_gartley_pattern(self, swings, df):
        """Check if swing points form a Gartley pattern with trend context"""
        if len(swings) < 5:
            return False
            
        # Calculate ratios between swing points
        xab = abs(swings[1]['price'] - swings[0]['price'])
        abc = abs(swings[2]['price'] - swings[1]['price'])
        bcd = abs(swings[3]['price'] - swings[2]['price'])
        xad = abs(swings[4]['price'] - swings[0]['price'])
        
        # Check Gartley ratios
        ab_retracement = abc / xab
        bc_retracement = bcd / abc
        xd_retracement = xad / xab
        
        # Check trend context
        trend_bullish = df['ema_fast'].iloc[-1] > df['ema_slow'].iloc[-1]
        pattern_valid = (0.618 <= ab_retracement <= 0.618) and \
                       (0.382 <= bc_retracement <= 0.886) and \
                       (0.786 <= xd_retracement <= 0.786)
        
        # Return pattern validity with trend alignment
        if pattern_valid:
            if swings[-1]['type'] == 'low' and trend_bullish:
                return True
            elif swings[-1]['type'] == 'high' and not trend_bullish:
                return True
        return False
               
    def is_butterfly_pattern(self, swings, df):
        """Check if swing points form a Butterfly pattern with trend context"""
        if len(swings) < 5:
            return False
            
        # Calculate ratios between swing points
        xab = abs(swings[1]['price'] - swings[0]['price'])
        abc = abs(swings[2]['price'] - swings[1]['price'])
        bcd = abs(swings[3]['price'] - swings[2]['price'])
        xad = abs(swings[4]['price'] - swings[0]['price'])
        
        # Check Butterfly ratios
        ab_retracement = abc / xab
        bc_retracement = bcd / abc
        xd_retracement = xad / xab
        
        # Check trend context
        trend_bullish = df['ema_fast'].iloc[-1] > df['ema_slow'].iloc[-1]
        pattern_valid = (0.786 <= ab_retracement <= 0.786) and \
                       (0.382 <= bc_retracement <= 0.886) and \
                       (1.618 <= xd_retracement <= 2.618)
        
        # Return pattern validity with trend alignment
        if pattern_valid:
            if swings[-1]['type'] == 'low' and trend_bullish:
                return True
            elif swings[-1]['type'] == 'high' and not trend_bullish:
                return True
        return False

    def detect_seasonal_patterns(self, df):
        """Identify profitable day-of-week and hour-of-day patterns with transaction costs"""
        try:
            signals = []
            df['hour'] = pd.to_datetime(df.index).hour
            df['day_of_week'] = pd.to_datetime(df.index).dayofweek
            
            # Calculate returns with transaction costs
            spread_cost = 0.5  # 0.5 points spread cost
            df['returns'] = df['close'].pct_change() - (spread_cost / df['close'])
            
            # Update seasonal statistics
            for day in range(7):
                day_data = df[df['day_of_week'] == day]
                for hour in range(24):
                    hour_data = day_data[day_data['hour'] == hour]
                    if not hour_data.empty:
                        self.seasonal_stats[day][hour].append(hour_data['returns'].mean())
            
            # Generate signals based on historical patterns with higher threshold
            current_day = pd.to_datetime(df.index[-1]).dayofweek
            current_hour = pd.to_datetime(df.index[-1]).hour
            
            if len(self.seasonal_stats[current_day][current_hour]) > 30:
                avg_return = np.mean(self.seasonal_stats[current_day][current_hour])
                if avg_return > 0.003:  # 0.3% threshold
                    signals.append(('SEASONAL', 'LONG'))
                elif avg_return < -0.003:
                    signals.append(('SEASONAL', 'SHORT'))
            
            return signals
        except Exception as e:
            self.logger.error(f"Error detecting seasonal patterns: {e}")
            return []

    def detect_market_regime(self, df):
        """Determine if market is trending, ranging, or volatile"""
        try:
            # Calculate ADX for trend strength
            adx_value = df['adx'].iloc[-1]
            
            # Calculate ATR and normalize
            current_atr = df['atr'].iloc[-1]
            self.historical_atr.append(current_atr)
            if len(self.historical_atr) > 100:
                self.historical_atr.pop(0)
            atr_percentile = np.percentile(self.historical_atr, 75)
            
            # Calculate BB width
            bb_width = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            self.bb_width_history.append(bb_width.iloc[-1])
            if len(self.bb_width_history) > 100:
                self.bb_width_history.pop(0)
            bb_width_percentile = np.percentile(self.bb_width_history, 25)
            
            # Classify regime
            if adx_value > 30 and current_atr < atr_percentile:
                return 'TRENDING'
            elif current_atr > atr_percentile:
                return 'VOLATILE'
            elif bb_width.iloc[-1] < bb_width_percentile:
                return 'RANGING'
            else:
                return 'UNDEFINED'
                
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {e}")
            return 'UNDEFINED'

    def analyze_order_flow(self, order_book):
        """Enhanced order flow analysis with liquidity clusters and iceberg detection"""
        try:
            signals = []
            if not order_book:
                return signals
            
            # Group orders into price clusters
            bid_clusters = defaultdict(float)
            ask_clusters = defaultdict(float)
            for bid in order_book['bids']:
                price_level = round(bid['price'], 2)
                bid_clusters[price_level] += bid['volume']
            for ask in order_book['asks']:
                price_level = round(ask['price'], 2)
                ask_clusters[price_level] += ask['volume']
            
            # Detect liquidity clusters
            bid_liquidity = max(bid_clusters.values()) if bid_clusters else 0
            ask_liquidity = max(ask_clusters.values()) if ask_clusters else 0
            
            # Calculate volume imbalance
            total_bid_volume = sum(bid_clusters.values())
            total_ask_volume = sum(ask_clusters.values())
            volume_ratio = total_bid_volume / total_ask_volume if total_ask_volume > 0 else float('inf')
            
            # Detect potential iceberg orders (large individual orders)
            max_normal_order = np.mean(list(bid_clusters.values()) + list(ask_clusters.values())) * 3
            iceberg_detected = bid_liquidity > max_normal_order or ask_liquidity > max_normal_order
            
            # Generate signals
            if volume_ratio > 1.5 and not iceberg_detected:
                signals.append(('ORDER_FLOW', 'LONG', 'STRONG_BUYING'))
            elif volume_ratio < 0.67 and not iceberg_detected:
                signals.append(('ORDER_FLOW', 'SHORT', 'STRONG_SELLING'))
            
            if iceberg_detected:
                if bid_liquidity > ask_liquidity:
                    signals.append(('ORDER_FLOW', 'LONG', 'ICEBERG_BID'))
                else:
                    signals.append(('ORDER_FLOW', 'SHORT', 'ICEBERG_ASK'))
            
            return signals
        except Exception as e:
            self.logger.error(f"Error analyzing order flow: {e}")
            return []

    def combine_strategies_by_regime(self, df, technical_signals, order_flow_signals, ai_signals, macro_signals=None):
        """Combine strategies with weights based on market regime and macro signals"""
        try:
            # Detect current market regime
            regime = self.detect_market_regime(df)
            
            # Initialize signal weights based on regime
            weights = {
                'TRENDING': {
                    'TREND': 0.4,
                    'BREAKOUT': 0.3,
                    'MEAN_REVERSION': 0.1,
                    'ORDER_FLOW': 0.1,
                    'SEASONAL': 0.05,
                    'MACRO': 0.05
                },
                'RANGING': {
                    'TREND': 0.1,
                    'BREAKOUT': 0.2,
                    'MEAN_REVERSION': 0.4,
                    'ORDER_FLOW': 0.15,
                    'SEASONAL': 0.05,
                    'MACRO': 0.1
                },
                'VOLATILE': {
                    'TREND': 0.2,
                    'BREAKOUT': 0.2,
                    'MEAN_REVERSION': 0.2,
                    'ORDER_FLOW': 0.25,
                    'SEASONAL': 0.05,
                    'MACRO': 0.1
                },
                'UNDEFINED': {
                    'TREND': 0.2,
                    'BREAKOUT': 0.2,
                    'MEAN_REVERSION': 0.2,
                    'ORDER_FLOW': 0.15,
                    'SEASONAL': 0.15,
                    'MACRO': 0.1
                }
            }
            
            # Calculate weighted signals
            long_score = 0
            short_score = 0
            current_weights = weights[regime]
            
            # Process technical signals
            for signal_type, direction, *details in technical_signals:
                if direction == 'LONG':
                    long_score += current_weights.get(signal_type, 0.1)
                elif direction == 'SHORT':
                    short_score += current_weights.get(signal_type, 0.1)
            
            # Process order flow signals
            for signal_type, direction, *details in order_flow_signals:
                if direction == 'LONG':
                    long_score += current_weights.get(signal_type, 0.1)
                elif direction == 'SHORT':
                    short_score += current_weights.get(signal_type, 0.1)
            
            # Include AI signals with regime-specific weights
            if ai_signals:
                ai_weight = 0.2 if regime in ['TRENDING', 'VOLATILE'] else 0.1
                if ai_signals.get('prediction', 0.5) > 0.6:
                    long_score += ai_weight
                elif ai_signals.get('prediction', 0.5) < 0.4:
                    short_score += ai_weight
            
            # Make final decision with regime-specific thresholds
            threshold = 0.3 if regime == 'VOLATILE' else 0.2
            if long_score > short_score and long_score >= threshold:
                return 'LONG'
            elif short_score > long_score and short_score >= threshold:
                return 'SHORT'
            else:
                return 'NEUTRAL'
                
        except Exception as e:
            self.logger.error(f"Error combining strategies by regime: {e}")
            return 'NEUTRAL'

    def generate_signals(self, df, order_book, ai_signals=None):
        """Generate trading signals by combining all strategies with regime-based weighting"""
        try:
            # Calculate technical indicators
            df = self.calculate_indicators(df)
            if df is None:
                return 'NEUTRAL'
            
            # Collect signals from different strategies
            technical_signals = (
                self.detect_trend_signals(df) +
                self.detect_mean_reversion_signals(df) +
                self.detect_breakout_signals(df) +
                self.detect_seasonal_patterns(df)
            )
            
            # Get enhanced order flow signals
            order_flow_signals = self.analyze_order_flow(order_book)
            
            # Combine signals using regime-based strategy
            final_signal = self.combine_strategies_by_regime(
                df,
                technical_signals,
                order_flow_signals,
                ai_signals
            )
            
            return final_signal
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return 'NEUTRAL'