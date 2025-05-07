import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Optional, Union
from dataclasses import dataclass

@dataclass
class Trade:
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    position_size: float
    direction: str  # 'long' or 'short'
    pnl: Optional[float] = None
    status: str = 'open'  # 'open' or 'closed'

class BacktestEngine:
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.trades: List[Trade] = []
        self.positions: Dict[str, Trade] = {}
        self.equity_curve: List[float] = []
        self.returns: List[float] = []
        self.trade_history: pd.DataFrame = pd.DataFrame()
        
    def process_signals(self, df: pd.DataFrame, signals: List[Dict]):
        """Process trading signals and simulate trades"""
        self.equity_curve = [self.initial_capital]
        self.returns = [0.0]
        
        for i, (timestamp, row) in enumerate(df.iterrows()):
            # Update portfolio value
            portfolio_value = self.current_capital
            for pos in self.positions.values():
                if pos.direction == 'long':
                    portfolio_value += pos.position_size * (row['close'] - pos.entry_price)
                else:  # short
                    portfolio_value += pos.position_size * (pos.entry_price - row['close'])
            
            self.equity_curve.append(portfolio_value)
            self.returns.append((portfolio_value - self.equity_curve[-2]) / self.equity_curve[-2])
            
            # Process signal for current timestamp
            if i < len(signals):
                signal = signals[i]
                if signal['action'] in ['BUY', 'SELL']:
                    self._execute_trade(timestamp, row['close'], signal)
    
    def _execute_trade(self, timestamp: datetime, price: float, signal: Dict):
        """Execute a trade based on the signal"""
        action = signal['action']
        confidence = signal.get('confidence', 1.0)
        position_size = self.current_capital * 0.1 * confidence  # Use 10% of capital * confidence
        
        # Close existing opposite positions
        for symbol, pos in list(self.positions.items()):
            if (action == 'BUY' and pos.direction == 'short') or \
               (action == 'SELL' and pos.direction == 'long'):
                self._close_position(symbol, timestamp, price)
        
        # Open new position
        if action == 'BUY':
            self.positions[timestamp.isoformat()] = Trade(
                entry_time=timestamp,
                exit_time=None,
                entry_price=price,
                exit_price=None,
                position_size=position_size,
                direction='long'
            )
        elif action == 'SELL':
            self.positions[timestamp.isoformat()] = Trade(
                entry_time=timestamp,
                exit_time=None,
                entry_price=price,
                exit_price=None,
                position_size=position_size,
                direction='short'
            )
    
    def _close_position(self, symbol: str, timestamp: datetime, price: float):
        """Close an open position"""
        position = self.positions[symbol]
        position.exit_time = timestamp
        position.exit_price = price
        position.status = 'closed'
        
        # Calculate PnL
        if position.direction == 'long':
            position.pnl = position.position_size * (price - position.entry_price)
        else:  # short
            position.pnl = position.position_size * (position.entry_price - price)
        
        self.current_capital += position.pnl
        self.trades.append(position)
        del self.positions[symbol]
    
    def calculate_metrics(self) -> Dict:
        """Calculate performance metrics"""
        # Convert trades to DataFrame
        if self.trades:
            trade_data = [
                {
                    'entry_time': t.entry_time,
                    'exit_time': t.exit_time,
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'position_size': t.position_size,
                    'direction': t.direction,
                    'pnl': t.pnl,
                    'status': t.status
                } for t in self.trades
            ]
            self.trade_history = pd.DataFrame(trade_data)
        
        # Calculate metrics
        returns = pd.Series(self.returns)
        equity_curve = pd.Series(self.equity_curve)
        
        metrics = {
            'total_return': (self.current_capital - self.initial_capital) / self.initial_capital,
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'max_drawdown': self._calculate_max_drawdown(equity_curve),
            'win_rate': len(self.trade_history[self.trade_history['pnl'] > 0]) / len(self.trade_history) if len(self.trade_history) > 0 else 0,
            'total_trades': len(self.trades),
            'avg_trade_pnl': self.trade_history['pnl'].mean() if len(self.trade_history) > 0 else 0,
            'final_capital': self.current_capital
        }
        
        return metrics
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.01) -> float:
        """Calculate annualized Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
        excess_returns = returns - risk_free_rate/252  # Assuming daily data
        if excess_returns.std() == 0:
            return 0.0
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown"""
        rolling_max = equity_curve.expanding().max()
        drawdowns = equity_curve / rolling_max - 1.0
        return abs(drawdowns.min())
    
    def plot_equity_curve(self):
        """Plot equity curve and drawdown"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.equity_curve, label='Portfolio Value')
        plt.title('Equity Curve')
        plt.xlabel('Time')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Plot drawdown
        equity_series = pd.Series(self.equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdowns = equity_series / rolling_max - 1.0
        
        plt.figure(figsize=(12, 6))
        plt.plot(drawdowns, label='Drawdown', color='red')
        plt.title('Drawdown')
        plt.xlabel('Time')
        plt.ylabel('Drawdown %')
        plt.legend()
        plt.grid(True)
        plt.show()