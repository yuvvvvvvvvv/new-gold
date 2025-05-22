import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ai_model import AIModel
from risk_manager import RiskManager
import ta
import logging
from tqdm import tqdm

class Position(Enum):
    LONG = 1
    SHORT = -1
    FLAT = 0

@dataclass
class Trade:
    entry_time: datetime
    exit_time: Optional[datetime]
    position: Position
    entry_price: float
    exit_price: Optional[float]
    size: float
    pnl: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]

class BacktestEngine:
    def __init__(self, 
                 data: pd.DataFrame,
                 initial_capital: float = 100000.0,
                 position_size: float = 0.02,
                 stop_loss: float = 0.02,
                 take_profit: float = 0.04,
                 max_positions: int = 5,
                 timeframe: str = '1H',
                 spread: float = 0.5,
                 commission_per_lot: float = 5.0):
        
        self.data = data
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_positions = max_positions
        self.timeframe = timeframe
        self.spread = spread
        self.commission_per_lot = commission_per_lot
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.ai_model = AIModel()
        self.risk_manager = RiskManager()
        
        # Trading state
        self.current_position = Position.FLAT
        self.open_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.equity_curve = [initial_capital]
        self.current_capital = initial_capital
        
        # Performance metrics
        self.metrics = {}
        
    def preprocess_data(self) -> None:
        """Preprocess and validate input data"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in self.data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
            
        # Ensure data is sorted by time
        self.data = self.data.sort_index()
        
        # Calculate basic indicators if not present
        if 'rsi' not in self.data.columns:
            self._calculate_rsi()
        if 'macd' not in self.data.columns:
            self._calculate_macd()
        if 'atr' not in self.data.columns:
            self.data['atr'] = ta.volatility.AverageTrueRange(high=self.data['high'], 
                                                            low=self.data['low'], 
                                                            close=self.data['close'], 
                                                            window=14).average_true_range()
            
    def _calculate_rsi(self, period: int = 14) -> None:
        """Calculate RSI indicator"""
        delta = self.data['close'].diff()
        gain = (delta > 0) * delta
        loss = (delta < 0) * -delta
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        self.data['rsi'] = 100 - (100 / (1 + rs))
        
    def _calculate_macd(self, fast: int = 12, slow: int = 26) -> None:
        """Calculate MACD indicator"""
        exp1 = self.data['close'].ewm(span=fast, adjust=False).mean()
        exp2 = self.data['close'].ewm(span=slow, adjust=False).mean()
        self.data['macd'] = exp1 - exp2
        
    def run_backtest(self) -> Dict:
        """Execute backtest over the entire dataset"""
        self.preprocess_data()
        
        # Split data into train/validate/test sets
        train_size = int(len(self.data) * 0.6)
        validate_size = int(len(self.data) * 0.2)
        
        training_data = self.data[:train_size]
        validation_data = self.data[train_size:train_size + validate_size]
        testing_data = self.data[train_size + validate_size:]
        
        # Train AI model on training data
        self.ai_model.train_models(training_data)
        self.ai_model.train_ppo(training_data)
        
        # Validate on validation set
        self.logger.info("Starting validation phase...")
        for i in tqdm(range(train_size, train_size + validate_size)):
            current_time = self.data.index[i]
            current_data = self.data.iloc[:i+1]
            self._process_timeframe(current_time, current_data)
            
        # Test on unseen data
        self.logger.info("Starting testing phase...")
        for i in tqdm(range(train_size + validate_size, len(self.data))):
            current_time = self.data.index[i]
            current_data = self.data.iloc[:i+1]
            self._process_timeframe(current_time, current_data)
            
            # Update risk parameters
            self.risk_manager.update_parameters(current_data)
            
            # Generate trading signals
            signals = self.ai_model.generate_signals(
                current_data.tail(100),  # Use recent data for signals
                []  # Add news data integration here if available
            )
            
            if signals:
                self._process_signals(signals, current_time)
                
            # Update open positions
            self._update_positions(current_time)
            
            # Update equity curve
            self.equity_curve.append(self.calculate_current_equity())
            
        # Calculate final performance metrics
        self._calculate_performance_metrics()
        return self.metrics
    
    def _process_signals(self, signals: Dict, current_time: datetime) -> None:
        """Process trading signals and execute trades"""
        current_price = self.data.loc[current_time, 'close']
        
        # Check if we can open new positions
        if len(self.open_trades) >= self.max_positions:
            return
            
        # Get position size from risk manager
        position_size = self.risk_manager.calculate_position_size(
            self.current_capital,
            current_price,
            self.position_size
        )
        
        # Execute trades based on signals
        if signals['ppo_action'] == 'BUY' and signals['ppo_confidence'] > 0.6:
            self._open_trade(Position.LONG, current_time, current_price, position_size)
        elif signals['ppo_action'] == 'SELL' and signals['ppo_confidence'] > 0.6:
            self._open_trade(Position.SHORT, current_time, current_price, position_size)
            
    def _open_trade(self, position: Position, time: datetime, price: float, size: float) -> None:
        """Open a new trade with ATR-based stop loss"""
        try:
            current_atr = self.data.loc[time, 'atr']
            atr_multiplier = 2.0
            
            # Calculate ATR-based stop loss
            if position == Position.LONG:
                stop_loss = price - (current_atr * atr_multiplier)
                take_profit = price + (current_atr * atr_multiplier * 2)
            else:  # SHORT position
                stop_loss = price + (current_atr * atr_multiplier)
                take_profit = price - (current_atr * atr_multiplier * 2)
            
            trade = Trade(
                entry_time=time,
                exit_time=None,
                position=position,
                entry_price=price,
                exit_price=None,
                size=size,
                pnl=None,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            self.open_trades.append(trade)
            
        except Exception as e:
            self.logger.error(f"Error opening trade: {e}")
        
    def _update_positions(self, current_time: datetime) -> None:
        """Update open positions and check for exits with trailing stop"""
        try:
            current_price = self.data.loc[current_time, 'close']
            current_atr = self.data.loc[current_time, 'atr']
            
            for trade in self.open_trades.copy():
                # Update trailing stop if price moves favorably
                if trade.position == Position.LONG:
                    potential_stop = current_price - (current_atr * 1.5)
                    if potential_stop > trade.stop_loss:
                        trade.stop_loss = potential_stop
                    
                    if current_price <= trade.stop_loss or current_price >= trade.take_profit:
                        self._close_trade(trade, current_time, current_price)
                else:  # SHORT position
                    potential_stop = current_price + (current_atr * 1.5)
                    if potential_stop < trade.stop_loss:
                        trade.stop_loss = potential_stop
                    
                    if current_price >= trade.stop_loss or current_price <= trade.take_profit:
                        self._close_trade(trade, current_time, current_price)
                        
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
                    
    def _close_trade(self, trade: Trade, time: datetime, price: float) -> None:
        """Close an open trade with transaction costs"""
        try:
            trade.exit_time = time
            trade.exit_price = price
            
            # Calculate raw PnL
            if trade.position == Position.LONG:
                raw_pnl = (price - trade.entry_price) * trade.size
            else:  # SHORT position
                raw_pnl = (trade.entry_price - price) * trade.size
            
            # Apply transaction costs
            spread_cost = self.spread * trade.size
            commission = self.commission_per_lot * (trade.size / 100000)  # Standard lot size
            total_costs = spread_cost + commission
            
            trade.pnl = raw_pnl - total_costs
            self.current_capital += trade.pnl
            self.closed_trades.append(trade)
            self.open_trades.remove(trade)
            
        except Exception as e:
            self.logger.error(f"Error closing trade: {e}")
        
    def calculate_current_equity(self) -> float:
        """Calculate current equity including open positions"""
        equity = self.current_capital
        current_price = self.data.iloc[-1]['close']
        
        for trade in self.open_trades:
            if trade.position == Position.LONG:
                equity += (current_price - trade.entry_price) * trade.size
            else:  # SHORT position
                equity += (trade.entry_price - current_price) * trade.size
                
        return equity
        
    def _calculate_performance_metrics(self) -> None:
        """Calculate comprehensive performance metrics with forex adjustments"""
        if not self.closed_trades:
            return
            
        # Basic metrics
        total_trades = len(self.closed_trades)
        profitable_trades = len([t for t in self.closed_trades if t.pnl > 0])
        win_rate = profitable_trades / total_trades
        
        # PnL metrics
        total_pnl = sum(t.pnl for t in self.closed_trades)
        avg_pnl = total_pnl / total_trades
        pnl_std = np.std([t.pnl for t in self.closed_trades])
        
        # Risk metrics for forex (260 trading days)
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        sharpe_ratio = np.sqrt(260) * returns.mean() / returns.std()
        max_drawdown = self._calculate_max_drawdown()
        monte_carlo_dd = self.run_monte_carlo()
        
        self.metrics = {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'pnl_std': pnl_std,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_equity': self.equity_curve[-1],
            'return': (self.equity_curve[-1] - self.initial_capital) / self.initial_capital
        }
        
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        equity_series = pd.Series(self.equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdowns = equity_series / rolling_max - 1
        return abs(drawdowns.min())
        
    def run_monte_carlo(self, simulations: int = 1000) -> float:
        """Run Monte Carlo simulation to estimate worst-case drawdown"""
        try:
            if len(self.closed_trades) < 2:
                return 0.0
                
            # Extract trade returns
            trade_returns = [(t.pnl / self.initial_capital) for t in self.closed_trades]
            
            # Run simulations
            max_drawdowns = []
            for _ in range(simulations):
                # Shuffle returns
                np.random.shuffle(trade_returns)
                
                # Calculate equity curve
                equity = [self.initial_capital]
                for ret in trade_returns:
                    equity.append(equity[-1] * (1 + ret))
                    
                # Calculate drawdown for this simulation
                equity_series = pd.Series(equity)
                rolling_max = equity_series.expanding().max()
                drawdowns = equity_series / rolling_max - 1
                max_drawdowns.append(abs(drawdowns.min()))
                
            # Return 95th percentile drawdown
            return np.percentile(max_drawdowns, 95)
            
        except Exception as e:
            self.logger.error(f"Error in Monte Carlo simulation: {e}")
            return 0.0
            
    def _process_timeframe(self, current_time: datetime, current_data: pd.DataFrame) -> None:
        """Process a single timeframe in the backtest"""
        try:
            # Update risk parameters
            self.risk_manager.update_parameters(current_data)
            
            # Generate trading signals
            signals = self.ai_model.generate_signals(
                current_data.tail(100),
                []
            )
            
            if signals:
                self._process_signals(signals, current_time)
                
            # Update open positions
            self._update_positions(current_time)
            
            # Update equity curve
            self.equity_curve.append(self.calculate_current_equity())
            
        except Exception as e:
            self.logger.error(f"Error processing timeframe: {e}")
        
    def plot_results(self) -> None:
        """Generate comprehensive performance visualization"""
        plt.style.use('seaborn')
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot equity curve
        equity_df = pd.DataFrame(self.equity_curve, index=self.data.index[-len(self.equity_curve):])
        equity_df.plot(ax=axes[0], title='Equity Curve')
        axes[0].set_ylabel('Equity')
        
        # Plot drawdown
        rolling_max = pd.Series(self.equity_curve).expanding().max()
        drawdown = pd.Series(self.equity_curve) / rolling_max - 1
        drawdown.plot(ax=axes[1], title='Drawdown')
        axes[1].set_ylabel('Drawdown')
        
        # Plot trade distribution
        pnls = [t.pnl for t in self.closed_trades]
        sns.histplot(pnls, kde=True, ax=axes[2])
        axes[2].set_title('Trade PnL Distribution')
        axes[2].set_xlabel('PnL')
        
        plt.tight_layout()
        plt.show()
        
    def generate_report(self) -> str:
        """Generate detailed performance report"""
        report = ["Backtest Performance Report"]
        report.append("=" * 30)
        
        # Basic metrics
        report.append(f"Total Trades: {self.metrics['total_trades']}")
        report.append(f"Win Rate: {self.metrics['win_rate']:.2%}")
        report.append(f"Total PnL: ${self.metrics['total_pnl']:.2f}")
        report.append(f"Average PnL: ${self.metrics['avg_pnl']:.2f}")
        
        # Risk metrics
        report.append("\nRisk Metrics:")
        report.append(f"Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}")
        report.append(f"Max Drawdown: {self.metrics['max_drawdown']:.2%}")
        report.append(f"Final Equity: ${self.metrics['final_equity']:.2f}")
        report.append(f"Total Return: {self.metrics['return']:.2%}")
        
        # Position analysis
        long_trades = len([t for t in self.closed_trades if t.position == Position.LONG])
        short_trades = len([t for t in self.closed_trades if t.position == Position.SHORT])
        report.append("\nPosition Analysis:")
        report.append(f"Long Trades: {long_trades}")
        report.append(f"Short Trades: {short_trades}")
        
        return '\n'.join(report)