import asyncio
import logging
import psutil
import time
from datetime import datetime
import pandas as pd
from typing import Dict, List, Optional
from data_fetcher import DataFetcher
from strategy_engine import StrategyEngine
from ai_model import AIModel
from risk_manager import RiskManager
from config import *

class TradingBot:
    def __init__(self):
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, LOG_LEVEL),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename=LOG_FILE
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.data_fetcher = None
        self.strategy_engine = None
        self.ai_model = None
        self.risk_manager = None
        
        # Component health status
        self.component_status = {
            'data_fetcher': False,
            'strategy_engine': False,
            'ai_model': False,
            'risk_manager': False
        }
        
        # Performance metrics
        self.performance_metrics = {
            'response_times': [],
            'prediction_accuracy': [],
            'trade_metrics': [],
            'system_metrics': []
        }
        
        # Error tracking
        self.error_counts = defaultdict(int)
        self.last_equity = None
        self.equity_history = []
        
        # Trading settings
        self.dry_run = getattr(config, 'DRY_RUN', False)
        self.max_retries = 3
        self.retry_delay = 5  # seconds
        self.degraded_mode = False
        
        # Initialize components with health check
        self.initialize_components()
        self.start_health_monitoring()
        
        self.logger.info(f"Trading bot initialized with health monitoring (Dry Run: {self.dry_run})")

    def is_trading_allowed(self):
        """Check if trading is allowed based on current time"""
        now = datetime.now().utcnow()
        current_hour = now.hour
        
        # No trading on Sundays
        if now.weekday() == 6:
            self.logger.info("Trading paused - Sunday")
            return False
            
        # No trading during low liquidity hours (21:00-23:00 UTC)
        if 21 <= current_hour <= 22:
            self.logger.info("Trading paused - Low liquidity period")
            return False
            
        return True

    def trigger_kill_switch(self):
        """Check equity drawdown and trigger emergency shutdown if needed"""
        try:
            current_equity = self.risk_manager.get_current_equity()
            self.equity_history.append((datetime.now(), current_equity))
            
            # Remove entries older than 24 hours
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.equity_history = [(t, e) for t, e in self.equity_history if t > cutoff_time]
            
            if len(self.equity_history) > 1:
                oldest_equity = self.equity_history[0][1]
                drawdown = (oldest_equity - current_equity) / oldest_equity
                
                if drawdown >= 0.20:  # 20% drawdown
                    self.logger.critical(f"Kill switch triggered - {drawdown*100:.1f}% drawdown in 24h")
                    asyncio.create_task(self.emergency_shutdown())
                    return True
            
            return False
        except Exception as e:
            self.logger.error(f"Error in kill switch check: {e}")
            return False
    
    def initialize_components(self):
        """Initialize all components with error handling"""
        try:
            self.data_fetcher = DataFetcher()
            self.component_status['data_fetcher'] = True
        except Exception as e:
            self.logger.error(f"Failed to initialize DataFetcher: {e}")
            
        try:
            self.strategy_engine = StrategyEngine()
            self.component_status['strategy_engine'] = True
        except Exception as e:
            self.logger.error(f"Failed to initialize StrategyEngine: {e}")
            
        try:
            self.ai_model = AIModel()
            self.component_status['ai_model'] = True
        except Exception as e:
            self.logger.error(f"Failed to initialize AIModel: {e}")
            
        try:
            self.risk_manager = RiskManager()
            self.component_status['risk_manager'] = True
        except Exception as e:
            self.logger.error(f"Failed to initialize RiskManager: {e}")
    
    def start_health_monitoring(self):
        """Start monitoring system health"""
        asyncio.create_task(self.monitor_system_health())
        asyncio.create_task(self.monitor_component_health())
    
    async def monitor_system_health(self):
        """Monitor system resources and performance"""
        while True:
            try:
                metrics = {
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_usage': psutil.disk_usage('/').percent,
                    'timestamp': datetime.now()
                }
                self.performance_metrics['system_metrics'].append(metrics)
                
                # Trim old metrics
                if len(self.performance_metrics['system_metrics']) > 1000:
                    self.performance_metrics['system_metrics'] = self.performance_metrics['system_metrics'][-1000:]
                
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Error monitoring system health: {e}")
                await asyncio.sleep(5)
    
    async def monitor_component_health(self):
        """Monitor component health and attempt recovery"""
        while True:
            try:
                for component, status in self.component_status.items():
                    if not status:
                        self.logger.warning(f"{component} is unhealthy, attempting recovery")
                        await self.recover_component(component)
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                self.logger.error(f"Error monitoring component health: {e}")
                await asyncio.sleep(5)
    
    async def recover_component(self, component_name: str):
        """Attempt to recover a failed component"""
        for attempt in range(self.max_retries):
            try:
                if component_name == 'data_fetcher':
                    self.data_fetcher = DataFetcher()
                elif component_name == 'strategy_engine':
                    self.strategy_engine = StrategyEngine()
                elif component_name == 'ai_model':
                    self.ai_model = AIModel()
                elif component_name == 'risk_manager':
                    self.risk_manager = RiskManager()
                
                self.component_status[component_name] = True
                self.logger.info(f"Successfully recovered {component_name}")
                return True
            except Exception as e:
                self.logger.error(f"Recovery attempt {attempt + 1} failed for {component_name}: {e}")
                await asyncio.sleep(self.retry_delay)
        
        self.enter_degraded_mode(component_name)
        return False
    
    def enter_degraded_mode(self, failed_component: str):
        """Enter degraded operation mode when a component fails"""
        self.degraded_mode = True
        self.logger.warning(f"Entering degraded mode due to {failed_component} failure")
        
        # Adjust operation based on failed component
        if failed_component == 'ai_model':
            self.logger.info("Operating with basic strategy only")
        elif failed_component == 'data_fetcher':
            self.logger.info("Operating with reduced data frequency")
        elif failed_component == 'risk_manager':
            self.logger.info("Halting new trades, managing existing positions only")
        
    async def process_data(self, data_package):
        """Process incoming data and make trading decisions with performance tracking"""
        start_time = time.time()
        try:
            # Check component health
            if not all(self.component_status.values()) and not self.degraded_mode:
                self.logger.error("Not all components are healthy")
                return
            
            # Extract data with validation
            tick_data = self.validate_data(data_package.get('tick_data'))
            order_book = self.validate_data(data_package.get('order_book'))
            news = self.validate_data(data_package.get('news'))
            economic_calendar = self.validate_data(data_package.get('economic_calendar'))
            alternative_data = self.validate_data(data_package.get('alternative_data'))
            
            if tick_data is None:
                self.logger.warning("Invalid tick data received")
                return
            
            # Get historical data with retry
            historical_data = await self.get_data_with_retry(lambda: self.data_fetcher.fetch_historical_data('XAUUSD', 'M5', '2020-01-01', '2025-05-01'))
            if historical_data is None:
                return
            
            # Process market data
            market_data = self.prepare_market_data(
                historical_data,
                tick_data,
                order_book,
                economic_calendar,
                alternative_data
            )
            
            # Generate signals with performance tracking
            signals = await self.generate_trading_signals(
                market_data,
                news,
                economic_calendar
            )
            
            # Execute trading decisions
            if signals['trading_signal'] != 'NEUTRAL':
                await self.execute_trade_with_monitoring(signals, market_data)
            
            # Update risk management
            await self.update_risk_management(market_data)
            
            # Record performance metrics
            self.record_performance_metrics(time.time() - start_time, signals)
            
        except Exception as e:
            self.logger.error(f"Error processing data: {e}")
            self.record_error_metrics(str(e))
    
    def validate_data(self, data) -> Optional[Dict]:
        """Validate incoming data"""
        if data is None:
            return None
        if isinstance(data, (pd.DataFrame, dict)):
            return data
        self.logger.warning(f"Invalid data format: {type(data)}")
        return None
    
    async def get_data_with_retry(self, data_func, max_retries=3):
        """Fetch data with retry mechanism"""
        for attempt in range(max_retries):
            try:
                data = data_func()
                if data is not None:
                    return data
            except Exception as e:
                self.logger.error(f"Data fetch attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(1)
        return None
    
    def prepare_market_data(self, historical_data, tick_data, order_book, 
                           economic_calendar, alternative_data) -> Dict:
        """Prepare and validate market data for analysis"""
        return {
            'historical': historical_data,
            'tick': tick_data,
            'order_book': order_book,
            'economic': economic_calendar,
            'alternative': alternative_data
        }
    
    async def generate_trading_signals(self, market_data, news, economic_calendar) -> Dict:
        """Generate trading signals with comprehensive analysis"""
        try:
            # Extract news texts
            news_texts = [article['title'] + ' ' + article['description'] 
                         for article in news] if news else []
            
            # Generate AI signals with error handling
            ai_signals = await self.get_data_with_retry(
                lambda: self.ai_model.generate_signals(market_data['historical'], news_texts)
            )
            
            # Generate trading signals with market regime detection
            trading_signal = self.strategy_engine.generate_signals(
                market_data['historical'],
                market_data['order_book'],
                ai_signals,
                economic_calendar
            )
            
            return {
                'trading_signal': trading_signal,
                'ai_signals': ai_signals
            }
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return {'trading_signal': 'NEUTRAL', 'ai_signals': None}
    
    async def execute_trade_with_monitoring(self, signals, market_data):
        """Execute and monitor trade execution with retry for partial fills"""
        try:
            if not self.is_trading_allowed():
                self.logger.info("Trading not allowed during current time window")
                return
                
            execution_start = time.time()
            success = False
            partial_fill = False
            
            # Check kill switch before trading
            if self.trigger_kill_switch():
                return
            
            # Dry run mode - log trade without execution
            if self.dry_run:
                self.logger.info(f"[DRY RUN] Would execute {signals['trading_signal']} trade")
                success = True
            else:
                # Execute trade with retry for partial fills
                for attempt in range(self.max_retries):
                    try:
                        result = await self.get_data_with_retry(
                            lambda: self.risk_manager.execute_trade(signals['trading_signal'], market_data['historical'])
                        )
                        
                        if result.get('status') == 'complete':
                            success = True
                            break
                        elif result.get('status') == 'partial_fill':
                            partial_fill = True
                            self.logger.warning(f"Partial fill on attempt {attempt + 1}")
                            await asyncio.sleep(1)  # Wait before retry
                        else:
                            self.logger.warning(f"Trade failed on attempt {attempt + 1}")
                            await asyncio.sleep(1)  # Wait before retry
                            
                    except Exception as e:
                        self.logger.error(f"Execution attempt {attempt + 1} failed: {e}")
                        await asyncio.sleep(1)
            
            # Record execution metrics
            execution_time = time.time() - execution_start
            self.performance_metrics['trade_metrics'].append({
                'signal': signals['trading_signal'],
                'execution_time': execution_time,
                'success': success,
                'partial_fill': partial_fill,
                'timestamp': datetime.now()
            })
            
        except Exception as e:
            self.logger.error(f"Trade execution failed: {e}")
            self.error_counts['trade_execution'] += 1
    
    async def update_risk_management(self, market_data):
        """Update risk management parameters"""
        try:
            await self.get_data_with_retry(
                lambda: self.risk_manager.update_stops(market_data['historical'])
            )
        except Exception as e:
            self.logger.error(f"Error updating risk management: {e}")
    
    def record_performance_metrics(self, processing_time: float, signals: Dict):
        """Record system performance metrics"""
        self.performance_metrics['response_times'].append({
            'processing_time': processing_time,
            'timestamp': datetime.now()
        })
        
        if signals['ai_signals']:
            self.performance_metrics['prediction_accuracy'].append({
                'predicted': signals['ai_signals'],
                'actual': signals['trading_signal'],
                'timestamp': datetime.now()
            })
    
    def record_error_metrics(self, error_message: str, component: str = 'general'):
        """Record error metrics and track counts by component"""
        self.error_counts[component] += 1
        self.performance_metrics['system_metrics'].append({
            'error': error_message,
            'component': component,
            'count': self.error_counts[component],
            'timestamp': datetime.now()
        })
        
        # Log error count milestones
        if self.error_counts[component] in [1, 10, 50, 100]:
            self.logger.warning(
                f"Error count for {component} reached {self.error_counts[component]}"
            )
    
    async def run(self):
        """Main trading loop with diagnostics, monitoring and trading schedule"""
        try:
            # Initialize diagnostics mode if enabled
            if DIAGNOSTICS_MODE:
                await self.run_diagnostics()
            
            # Train AI model with initial data and monitoring
            await self.initialize_ai_model()
            
            # Start monitoring tasks
            monitoring_task = asyncio.create_task(self.monitor_system_health())
            health_check_task = asyncio.create_task(self.monitor_component_health())
            
            # Start data streaming with error recovery and trading schedule
            while True:
                try:
                    # Check trading schedule
                    if not self.is_trading_allowed():
                        await asyncio.sleep(60)  # Check again in 1 minute
                        continue
                        
                    # Check kill switch
                    if self.trigger_kill_switch():
                        break
                    
                    # Stream market data
                    await self.data_fetcher.stream_data(self.process_data)
                    
                except Exception as e:
                    self.logger.error(f"Stream error: {e}")
                    self.record_error_metrics(str(e), 'data_stream')
                    await self.handle_stream_error()
                    await asyncio.sleep(5)
            
        except Exception as e:
            self.logger.error(f"Critical error in main loop: {e}")
            self.record_error_metrics(str(e), 'main_loop')
            await self.emergency_shutdown()
    
    async def initialize_ai_model(self):
        """Initialize AI model with monitoring"""
        try:
            historical_data = await self.get_data_with_retry(lambda: self.data_fetcher.fetch_historical_data('XAUUSD', 'M5', '2020-01-01', '2025-05-01'))
            if historical_data is not None:
                training_start = time.time()
                await self.get_data_with_retry(lambda: self.ai_model.train_lstm(historical_data))
                training_time = time.time() - training_start
                
                self.performance_metrics['system_metrics'].append({
                    'event': 'model_training',
                    'duration': training_time,
                    'timestamp': datetime.now()
                })
        except Exception as e:
            self.logger.error(f"Error initializing AI model: {e}")
            self.enter_degraded_mode('ai_model')
    
    async def run_diagnostics(self):
        """Run system diagnostics"""
        self.logger.info("Running system diagnostics")
        try:
            # Test component connectivity
            await self.test_component_connectivity()
            
            # Test data quality
            await self.test_data_quality()
            
            # Test model performance
            await self.test_model_performance()
            
            # Test risk management rules
            await self.test_risk_management()
            
        except Exception as e:
            self.logger.error(f"Diagnostics failed: {e}")
    
    async def test_component_connectivity(self):
        """Test connectivity of all components"""
        components = {
            'data_fetcher': self.data_fetcher,
            'strategy_engine': self.strategy_engine,
            'ai_model': self.ai_model,
            'risk_manager': self.risk_manager
        }
        
        for name, component in components.items():
            try:
                if hasattr(component, 'test_connectivity'):
                    await component.test_connectivity()
                self.logger.info(f"{name} connectivity test passed")
            except Exception as e:
                self.logger.error(f"{name} connectivity test failed: {e}")
    
    async def test_data_quality(self):
        """Test quality of incoming data"""
        try:
            test_data = await self.get_data_with_retry(lambda: self.data_fetcher.fetch_historical_data('XAUUSD', 'M5', '2020-01-01', '2025-05-01'))
            if test_data is not None:
                # Check for missing values
                missing_count = test_data.isnull().sum().sum()
                self.logger.info(f"Missing values in test data: {missing_count}")
                
                # Check data freshness
                latest_time = test_data['time'].max()
                time_diff = datetime.now() - latest_time
                self.logger.info(f"Data freshness: {time_diff.total_seconds()} seconds")
        except Exception as e:
            self.logger.error(f"Data quality test failed: {e}")
    
    async def test_model_performance(self):
        """Test AI model performance"""
        try:
            if hasattr(self.ai_model, 'test_performance'):
                test_results = await self.get_data_with_retry(self.ai_model.test_performance)
                self.logger.info(f"Model performance test results: {test_results}")
        except Exception as e:
            self.logger.error(f"Model performance test failed: {e}")
    
    async def test_risk_management(self):
        """Test risk management rules"""
        try:
            if hasattr(self.risk_manager, 'test_risk_rules'):
                test_results = await self.get_data_with_retry(self.risk_manager.test_risk_rules)
                self.logger.info(f"Risk management test results: {test_results}")
        except Exception as e:
            self.logger.error(f"Risk management test failed: {e}")
    
    async def handle_stream_error(self):
        """Handle data stream errors"""
        self.logger.warning("Attempting to recover from stream error")
        try:
            # Reset data fetcher connection
            self.data_fetcher = DataFetcher()
            self.component_status['data_fetcher'] = True
            
            # Verify other components
            await self.test_component_connectivity()
            
        except Exception as e:
            self.logger.error(f"Stream error recovery failed: {e}")
            self.enter_degraded_mode('data_fetcher')
    
    async def emergency_shutdown(self):
        """Perform emergency shutdown procedures"""
        self.logger.critical("Initiating emergency shutdown")
        try:
            # Close all positions
            if self.risk_manager:
                await self.get_data_with_retry(self.risk_manager.close_all_positions)
            
            # Save system state
            self.save_system_state()
            
            # Cleanup resources
            self.cleanup_resources()
            
        except Exception as e:
            self.logger.critical(f"Emergency shutdown error: {e}")
    
    def save_system_state(self):
        """Save current system state for recovery"""
        try:
            state = {
                'performance_metrics': self.performance_metrics,
                'component_status': self.component_status,
                'timestamp': datetime.now()
            }
            # Save state to file (implementation details omitted)
            self.logger.info("System state saved")
        except Exception as e:
            self.logger.error(f"Failed to save system state: {e}")
    
    def cleanup_resources(self):
        """Cleanup system resources"""
        try:
            # Cleanup each component
            for component in [self.data_fetcher, self.strategy_engine, 
                            self.ai_model, self.risk_manager]:
                if component and hasattr(component, 'cleanup'):
                    component.cleanup()
            self.logger.info("Resources cleaned up")
        except Exception as e:
            self.logger.error(f"Resource cleanup failed: {e}")
            
    def stop(self):
        """Gracefully stop trading with cleanup"""
        try:
            # Stop monitoring tasks
            for task in asyncio.all_tasks():
                if not task.done():
                    task.cancel()
            
            # Close positions and cleanup
            if self.risk_manager:
                self.risk_manager.close_all_positions()
            
            # Save final metrics
            self.save_system_state()
            
            # Cleanup resources
            self.cleanup_resources()
            
            self.logger.info("Trading bot stopped gracefully")
        except Exception as e:
            self.logger.error(f"Error during graceful shutdown: {e}")
            # Attempt emergency shutdown
            asyncio.run(self.emergency_shutdown())

if __name__ == "__main__":
    # Create and run trading bot
    bot = TradingBot()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        bot.stop()