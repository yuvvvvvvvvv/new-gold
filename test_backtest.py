import pandas as pd
from ai_model import AIModel
from backtesting import BacktestEngine
from test_model import generate_sample_data, generate_sample_news

def main():
    # Initialize models and generate data
    print("Initializing AI Model and generating test data...")
    model = AIModel()
    df = generate_sample_data(n_samples=1000)
    news_texts = generate_sample_news()
    
    # Train the models
    print("Training models...")
    model.train_models(df)
    model.train_ppo(df)
    
    # Generate trading signals for each timestamp
    print("Generating trading signals...")
    signals = []
    for i in range(len(df)):
        current_data = df.iloc[:i+1]
        if len(current_data) > 50:  # Ensure enough data for feature engineering
            signal = model.generate_signals(current_data, news_texts)
            if signal:
                signals.append({
                    'action': signal['ppo_action'],
                    'confidence': signal['ppo_confidence']
                })
            else:
                signals.append({'action': 'HOLD', 'confidence': 0.0})
        else:
            signals.append({'action': 'HOLD', 'confidence': 0.0})
    
    # Initialize and run backtest
    print("Running backtest...")
    backtest = BacktestEngine(initial_capital=100000.0)
    backtest.process_signals(df, signals)
    
    # Calculate and display metrics
    metrics = backtest.calculate_metrics()
    print("\nBacktest Results:")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Average Trade PnL: ${metrics['avg_trade_pnl']:.2f}")
    print(f"Final Capital: ${metrics['final_capital']:.2f}")
    
    # Plot results
    print("\nGenerating performance plots...")
    backtest.plot_equity_curve()

if __name__ == "__main__":
    main()