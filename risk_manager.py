import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from config import *
from typing import Dict, List, Optional, Tuple

class RiskManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.last_trade_time = None
        self.last_volatility_check = datetime.now()
        self.circuit_breaker_active = False
        self.circuit_breaker_phase = 0  # 0: inactive, 1-3: recovery phases
        self.initial_equity = None
        self.high_water_mark = None
        self.drawdown_levels = {
            'warning': DRAWDOWN_WARNING,
            'reduced': DRAWDOWN_REDUCED,
            'stopped': DRAWDOWN_STOPPED
        }
        self.timeframe_weights = {
            '5m': 0.2,
            '15m': 0.3,
            '1h': 0.3,
            '4h': 0.2
        }
        self.correlation_lookback = 20  # days for correlation calculation
        self.market_regime = 'normal'  # normal, high_volatility, low_volatility
        
    def calculate_position_size(self, account_info, entry_price, stop_loss, correlation_data=None):
        """Calculate position size using Kelly Criterion with correlation and drawdown adjustments"""
        try:
            # Initialize equity tracking
            if self.initial_equity is None:
                self.initial_equity = account_info.equity
                self.high_water_mark = self.initial_equity
            
            # Update high water mark
            self.high_water_mark = max(self.high_water_mark, account_info.equity)
            
            # Calculate drawdown
            current_drawdown = 1 - (account_info.equity / self.high_water_mark)
            
            # Apply drawdown restrictions
            if current_drawdown >= self.drawdown_levels['stopped']:
                self.logger.warning(f"Trading stopped - Maximum drawdown reached: {current_drawdown:.2%}")
                return 0.0
            elif current_drawdown >= self.drawdown_levels['reduced']:
                risk_multiplier = 0.25  # Reduce position size to 25%
            elif current_drawdown >= self.drawdown_levels['warning']:
                risk_multiplier = 0.5   # Reduce position size to 50%
            else:
                risk_multiplier = 1.0
            
            # Calculate base position size and check exposure limit
            equity = account_info.equity
            risk_amount = equity * MAX_RISK_PER_TRADE * risk_multiplier
            max_exposure = equity * MAX_EXPOSURE_PER_TRADE
            
            # Adjust for correlation if data available
            if correlation_data is not None:
                correlation = correlation_data.get('gold_usd_correlation', 0)
                # Reduce position size when correlation is high (> 0.7 or < -0.7)
                if abs(correlation) > 0.7:
                    risk_amount *= (1 - (abs(correlation) - 0.7))  # Gradually reduce size
            
            # Calculate final position size
            pip_value = mt5.symbol_info(SYMBOL).trade_tick_value
            stop_loss_pips = abs(entry_price - stop_loss) / mt5.symbol_info(SYMBOL).point
            position_size = risk_amount / (stop_loss_pips * pip_value)
            
            # Adjust for market regime
            if self.market_regime == 'high_volatility':
                position_size *= 0.7  # Reduce size in high volatility
            elif self.market_regime == 'low_volatility':
                position_size *= 1.2  # Increase size in low volatility
            
            # Apply exposure limit and round to standard lot size
            max_position_by_exposure = max_exposure / entry_price
            position_size = round(position_size / 0.01) * 0.01
            position_size = max(min(position_size, max_position_by_exposure, MAX_POSITION_SIZE), 0.01)
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.01  # Minimum position size

    def calculate_stop_loss(self, df, direction):
        """Calculate ATR-based stop loss level"""
        try:
            # Get latest ATR value
            atr = df['atr'].iloc[-1]
            current_price = df['close'].iloc[-1]
            
            # Calculate stop loss based on ATR
            if direction == 'LONG':
                stop_loss = current_price - (atr * ATR_MULTIPLIER)
            else:  # SHORT
                stop_loss = current_price + (atr * ATR_MULTIPLIER)
                
            return stop_loss
            
        except Exception as e:
            self.logger.error(f"Error calculating stop loss: {e}")
            return None

    def calculate_volatility_metrics(self, df) -> Dict:
        """Calculate volatility metrics across different timeframes"""
        try:
            metrics = {}
            # Calculate volatility for different timeframes
            for timeframe, weight in self.timeframe_weights.items():
                period = int(timeframe[:-1]) if timeframe.endswith('m') else int(timeframe[:-1]) * 60
                returns = df['close'].pct_change(period).dropna()
                metrics[timeframe] = {
                    'volatility': returns.std(),
                    'recent_volatility': returns.tail(6).std(),
                    'historical_volatility': returns.expanding().std().mean()
                }
            return metrics
        except Exception as e:
            self.logger.error(f"Error calculating volatility metrics: {e}")
            return {}
    
    def update_market_regime(self, volatility_metrics: Dict):
        """Update market regime based on volatility metrics"""
        try:
            if not volatility_metrics:
                return
                
            # Calculate weighted volatility score
            total_vol_score = 0
            for timeframe, metrics in volatility_metrics.items():
                weight = self.timeframe_weights[timeframe]
                vol_ratio = metrics['recent_volatility'] / metrics['historical_volatility']
                total_vol_score += vol_ratio * weight
                
            # Update market regime
            if total_vol_score > 1.5:
                self.market_regime = 'high_volatility'
            elif total_vol_score < 0.5:
                self.market_regime = 'low_volatility'
            else:
                self.market_regime = 'normal'
                
        except Exception as e:
            self.logger.error(f"Error updating market regime: {e}")
    
    def calculate_portfolio_metrics(self, portfolio: Dict) -> Dict:
        """Calculate portfolio risk metrics and performance statistics"""
        try:
            metrics = {
                'total_equity': portfolio['equity'],
                'open_positions': len(portfolio['positions']),
                'total_exposure': 0,
                'largest_position': 0,
                'avg_position_size': 0,
                'portfolio_beta': 0
            }
            
            if portfolio['positions']:
                position_sizes = [pos['size'] for pos in portfolio['positions']]
                metrics.update({
                    'total_exposure': sum(position_sizes),
                    'largest_position': max(position_sizes),
                    'avg_position_size': sum(position_sizes) / len(position_sizes)
                })
                
                # Calculate portfolio beta (market sensitivity)
                if self.market_regime == 'high_volatility':
                    metrics['portfolio_beta'] = 1.2
                elif self.market_regime == 'low_volatility':
                    metrics['portfolio_beta'] = 0.8
                else:
                    metrics['portfolio_beta'] = 1.0
            
            return metrics
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {e}")
            return None

    def check_volatility(self, df) -> bool:
        """Enhanced circuit breaker with multi-phase recovery"""
        try:
            # Calculate volatility metrics
            volatility_metrics = self.calculate_volatility_metrics(df)
            self.update_market_regime(volatility_metrics)
            
            if not volatility_metrics:
                return True  # Conservative approach
            
            # Calculate current volatility level
            current_vol = volatility_metrics['5m']['recent_volatility']
            historical_vol = volatility_metrics['5m']['historical_volatility']
            vol_ratio = current_vol / historical_vol if historical_vol > 0 else float('inf')
            
            # Dynamic threshold based on market regime
            base_threshold = VOLATILITY_THRESHOLD
            if self.market_regime == 'high_volatility':
                threshold = base_threshold * 1.5
            elif self.market_regime == 'low_volatility':
                threshold = base_threshold * 0.8
            else:
                threshold = base_threshold
            
            # Circuit breaker logic
            if vol_ratio > threshold:
                self.circuit_breaker_active = True
                self.circuit_breaker_phase = 3  # Start at highest phase
                self.last_volatility_check = datetime.now()
                self.logger.warning(f"Circuit breaker activated - Phase {self.circuit_breaker_phase}")
                return True
            
            # Recovery phase logic
            if self.circuit_breaker_active:
                time_elapsed = datetime.now() - self.last_volatility_check
                
                # Phase transition logic
                if time_elapsed > timedelta(minutes=30) and self.circuit_breaker_phase > 0:
                    self.circuit_breaker_phase -= 1
                    self.last_volatility_check = datetime.now()
                    if self.circuit_breaker_phase == 0:
                        self.circuit_breaker_active = False
                        self.logger.info("Circuit breaker deactivated - Normal trading resumed")
                    else:
                        self.logger.info(f"Circuit breaker phase reduced to {self.circuit_breaker_phase}")
            
            return self.circuit_breaker_active
            
        except Exception as e:
            self.logger.error(f"Error checking volatility: {e}")
            return True  # Conservative approach

    def optimize_volatility_threshold(self, df: pd.DataFrame) -> float:
        """Optimize volatility threshold by testing different values"""
        try:
            best_threshold = VOLATILITY_THRESHOLD
            min_drawdown = float('inf')
            test_thresholds = np.arange(1.0, 3.5, 0.5)
            
            for threshold in test_thresholds:
                equity_curve = [self.initial_equity]
                current_equity = self.initial_equity
                max_equity = current_equity
                
                for i in range(1, len(df)):
                    returns = df['close'].pct_change().iloc[i]
                    vol_ratio = df['close'].pct_change().rolling(5).std().iloc[i] / \
                               df['close'].pct_change().expanding().std().iloc[i]
                    
                    if vol_ratio > threshold:
                        # Simulate reduced position size
                        returns *= 0.5
                    
                    current_equity *= (1 + returns)
                    max_equity = max(max_equity, current_equity)
                    max_drawdown = 1 - (current_equity / max_equity)
                    
                    if max_drawdown < min_drawdown:
                        min_drawdown = max_drawdown
                        best_threshold = threshold
            
            self.logger.info(f"Optimized volatility threshold: {best_threshold:.2f} with max drawdown: {min_drawdown:.2%}")
            return best_threshold
            
        except Exception as e:
            self.logger.error(f"Error optimizing volatility threshold: {e}")
            return VOLATILITY_THRESHOLD
    
    def fetch_correlation_data(self) -> Dict:
        """Fetch and calculate intraday correlation between gold and USD index"""
        try:
            # Fetch USD index hourly data
            usd_rates = mt5.copy_rates_from_pos('USDX', mt5.TIMEFRAME_H1, 0, self.correlation_lookback * 24)
            if usd_rates is None:
                return {}
            
            # Fetch gold hourly data
            gold_rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 0, self.correlation_lookback * 24)
            if gold_rates is None:
                return {}
            
            # Calculate hourly correlation
            usd_returns = pd.DataFrame(usd_rates)['close'].pct_change().dropna()
            gold_returns = pd.DataFrame(gold_rates)['close'].pct_change().dropna()
            correlation = usd_returns.corr(gold_returns)
            
            return {'gold_usd_correlation': correlation}
            
        except Exception as e:
            self.logger.error(f"Error fetching correlation data: {e}")
            return {}
    
    def analyze_timeframe_risks(self, df) -> Dict:
        """Analyze risks across multiple timeframes"""
        try:
            risk_scores = {}
            
            # Calculate risk metrics for each timeframe
            for timeframe, weight in self.timeframe_weights.items():
                period = int(timeframe[:-1]) if timeframe.endswith('m') else int(timeframe[:-1]) * 60
                
                # Calculate trend strength
                ema_fast = df['close'].ewm(span=period//2).mean()
                ema_slow = df['close'].ewm(span=period).mean()
                trend_strength = abs(ema_fast - ema_slow) / df['close'].std()
                
                # Calculate momentum
                momentum = df['close'].diff(period).abs().mean() / df['close'].std()
                
                # Calculate volatility ratio
                volatility = df['close'].pct_change(period).std()
                historical_vol = df['close'].pct_change(period).expanding().std().mean()
                vol_ratio = volatility / historical_vol if historical_vol > 0 else 1
                
                risk_scores[timeframe] = {
                    'trend_strength': trend_strength.iloc[-1],
                    'momentum': momentum,
                    'volatility_ratio': vol_ratio,
                    'weight': weight
                }
            
            return risk_scores
            
        except Exception as e:
            self.logger.error(f"Error analyzing timeframe risks: {e}")
            return {}
    
    def calculate_risk_parity_adjustment(self, risk_scores: Dict) -> float:
        """Calculate position size adjustment based on risk parity across timeframes"""
        try:
            if not risk_scores:
                return 1.0
            
            # Calculate weighted risk score
            total_risk_score = 0
            for timeframe, metrics in risk_scores.items():
                timeframe_risk = (
                    metrics['trend_strength'] +
                    metrics['momentum'] +
                    metrics['volatility_ratio']
                ) / 3
                total_risk_score += timeframe_risk * metrics['weight']
            
            # Convert to adjustment multiplier (0.5 to 1.5)
            adjustment = 1 + (1 - min(max(total_risk_score, 0), 2) / 2)
            return adjustment
            
        except Exception as e:
            self.logger.error(f"Error calculating risk parity adjustment: {e}")
            return 1.0
    
    def check_risk_limits(self, position_size: float, entry_price: float) -> bool:
        """Check if the proposed trade is within risk limits"""
        try:
            # Calculate potential exposure
            current_exposure = position_size * entry_price
            if current_exposure > MAX_POSITION_SIZE * entry_price:
                self.logger.warning("Maximum position size limit exceeded")
                return False
            
            # Check maximum drawdown
            current_drawdown = 1 - (metrics['total_equity'] / self.high_water_mark if self.high_water_mark else 1)
            if current_drawdown >= self.drawdown_levels['stopped']:
                self.logger.warning(f"Maximum drawdown limit reached: {current_drawdown:.2%}")
                return False
            
            # Check position concentration
            if metrics['largest_position'] > MAX_POSITION_SIZE:
                self.logger.warning("Maximum position size limit exceeded")
                return False
            
            # Check total exposure
            max_exposure = metrics['total_equity'] * MAX_PORTFOLIO_EXPOSURE
            if metrics['total_exposure'] > max_exposure:
                self.logger.warning("Maximum portfolio exposure limit exceeded")
                return False
            
            # Check market regime restrictions
            if self.market_regime == 'high_volatility' and metrics['portfolio_beta'] > 1.2:
                self.logger.warning("Portfolio beta too high for current market regime")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
            return False

    def execute_trade(self, direction, df):
        """Execute trade with enhanced risk management"""
        try:
            # Check circuit breaker
            if self.check_volatility(df):
                phase_msg = f" (Phase {self.circuit_breaker_phase})" if self.circuit_breaker_active else ""
                self.logger.warning(f"Circuit breaker active{phase_msg} - no trading")
                return False
            
            # Get current market info
            symbol_info = mt5.symbol_info(SYMBOL)
            if symbol_info is None:
                return False
            
            # Calculate entry and stop loss
            entry_price = symbol_info.ask if direction == 'LONG' else symbol_info.bid
            stop_loss = self.calculate_stop_loss(df, direction)
            if stop_loss is None:
                return False
            
            # Fetch correlation data
            correlation_data = self.fetch_correlation_data()
            
            # Analyze multi-timeframe risks
            risk_scores = self.analyze_timeframe_risks(df)
            risk_parity_adj = self.calculate_risk_parity_adjustment(risk_scores)
            
            # Calculate position size with all adjustments
            account_info = mt5.account_info()
            base_position_size = self.calculate_position_size(
                account_info, entry_price, stop_loss, correlation_data
            )
            adjusted_position_size = base_position_size * risk_parity_adj
            
            # Apply circuit breaker phase restrictions
            if self.circuit_breaker_active:
                phase_multiplier = 1 - (self.circuit_breaker_phase * 0.25)  # Phase 3: 25%, Phase 2: 50%, Phase 1: 75%
                adjusted_position_size *= phase_multiplier
            
            # Prepare trade request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": SYMBOL,
                "volume": adjusted_position_size,
                "type": mt5.ORDER_TYPE_BUY if direction == 'LONG' else mt5.ORDER_TYPE_SELL,
                "price": entry_price,
                "sl": stop_loss,
                "deviation": 20,
                "magic": 234000,
                "comment": f"AI Bot - Regime: {self.market_regime}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send trade
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Trade execution failed: {result.comment}")
                return False
            
            # Log trade details
            self.logger.info(
                f"Trade executed: {direction} {adjusted_position_size:.2f} lots "
                f"(Base: {base_position_size:.2f}, Risk Adj: {risk_parity_adj:.2f})"
            )
            
            self.last_trade_time = datetime.now()
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return False

    def update_stops(self, df):
        """Update trailing stops for open positions"""
        try:
            positions = mt5.positions_get(symbol=SYMBOL)
            if positions is None:
                return
                
            for position in positions:
                # Calculate new stop loss
                direction = 'LONG' if position.type == mt5.POSITION_TYPE_BUY else 'SHORT'
                new_stop = self.calculate_stop_loss(df, direction)
                
                # Update if new stop is more favorable
                if new_stop is not None:
                    if (direction == 'LONG' and new_stop > position.sl) or \
                       (direction == 'SHORT' and new_stop < position.sl):
                        request = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "symbol": SYMBOL,
                            "position": position.ticket,
                            "sl": new_stop
                        }
                        mt5.order_send(request)
                        
        except Exception as e:
            self.logger.error(f"Error updating stops: {e}")

    def close_all_positions(self):
        """Close all open positions"""
        try:
            positions = mt5.positions_get(symbol=SYMBOL)
            if positions is None:
                return
                
            for position in positions:
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": SYMBOL,
                    "volume": position.volume,
                    "type": mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                    "position": position.ticket,
                    "price": mt5.symbol_info_tick(SYMBOL).bid if position.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(SYMBOL).ask,
                    "deviation": 20,
                    "magic": 234000,
                    "comment": "Close by Risk Manager",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                mt5.order_send(request)
                
        except Exception as e:
            self.logger.error(f"Error closing positions: {e}")