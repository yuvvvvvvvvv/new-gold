import argparse
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from sklearn.metrics import mean_squared_error, accuracy_score
from gym import Env
import MetaTrader5 as mt5
from data_fetcher import DataFetcher
from backtesting import BacktestEngine
from config import ConfigLoader
from utils import engineer_features

# Set up logging
logging.basicConfig(filename='trading_bot.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Data Preparation
def prepare_training_data(config):
    """
    Prepare data for training by fetching, processing, and splitting.
    
    Args:
        config (dict): Configuration dictionary from config.yaml.
    
    Returns:
        dict: Dictionary with train_data, val_data, test_data (DataFrames per timeframe).
    """
    try:
        data_fetcher = DataFetcher(config)
        symbol = "XAUUSD"
        timeframes = config.get("TIMEFRAMES", ["M5", "M15", "H1", "H4"])
        start_date = "2020-01-01"
        end_date = "2025-05-01"
        
        # Fetch data
        historical_data = {}
        for timeframe in timeframes:
            data = data_fetcher.fetch_historical_data(symbol, timeframe, start_date, end_date)
            if data.empty:
                raise ValueError(f"No historical data for {symbol} {timeframe}")
            historical_data[timeframe] = data
            logger.info(f"Fetched historical data for {symbol} {timeframe}")
        
        macro_data = data_fetcher.fetch_macro_data()
        news_data = data_fetcher.fetch_news_data(symbol="gold")
        sentiment_data = data_fetcher.analyze_sentiment(news_data)
        econ_events = data_fetcher.parse_economic_events()  # Placeholder if unreliable
        
        # Engineer features
        feature_data = {}
        for timeframe, data in historical_data.items():
            features = engineer_features(data, config)
            feature_data[timeframe] = features
        
        # Combine data
        combined_data = {}
        for timeframe in timeframes:
            df = feature_data[timeframe]
            df = df.merge(macro_data, on="timestamp", how="left").fillna(method="ffill")
            df = df.merge(sentiment_data, on="timestamp", how="left").fillna(0)
            df = df.merge(econ_events, on="timestamp", how="left").fillna(0)
            combined_data[timeframe] = data_fetcher.validate_data(df)
        
        # Split data
        backtest_engine = BacktestEngine(config)
        train_data, val_data, test_data = {}, {}, {}
        for timeframe, data in combined_data.items():
            train, val, test = backtest_engine.split_data(data)
            train_data[timeframe] = train
            val_data[timeframe] = val
            test_data[timeframe] = test
            # Save to HDF5
            train.to_hdf(f"data/train_{symbol}_{timeframe}.h5", key="data")
            val.to_hdf(f"data/val_{symbol}_{timeframe}.h5", key="data")
            test.to_hdf(f"data/test_{symbol}_{timeframe}.h5", key="data")
            logger.info(f"Saved data for {symbol} {timeframe}")
        
        return {"train_data": train_data, "val_data": val_data, "test_data": test_data}
    
    except Exception as e:
        logger.error(f"Error in prepare_training_data: {str(e)}")
        raise

# LSTM Training
def train_lstm_model(X_train, y_train, X_val, y_val, config):
    """
    Train an LSTM model for price prediction or trading signals.
    
    Args:
        X_train, y_train, X_val, y_val (np.ndarray): Training and validation data.
        config (dict): Configuration dictionary.
    
    Returns:
        tf.keras.Model: Trained LSTM model.
    """
    try:
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(config["LSTM_UNITS"], return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(config["LSTM_UNITS"]),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1 if config.get("TASK") == "regression" else 3, activation="linear" if config.get("TASK") == "regression" else "softmax")
        ])
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config["LEARNING_RATE"]),
                     loss="mse" if config.get("TASK") == "regression" else "categorical_crossentropy",
                     metrics=["mae" if config.get("TASK") == "regression" else "accuracy"])
        
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        
        with tf.device("/GPU:0" if config["DEVICE"] == "cuda" and tf.test.is_gpu_available() else "/CPU:0"):
            history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                              epochs=config["EPOCHS"], batch_size=config["BATCH_SIZE"],
                              callbacks=[early_stopping], verbose=1)
        
        model.save(config["LSTM_MODEL_PATH"])
        logger.info(f"Saved LSTM model to {config['LSTM_MODEL_PATH']}")
        return model
    
    except Exception as e:
        logger.error(f"Error in train_lstm_model: {str(e)}")
        raise

# Transformer Training
def train_transformer_model(X_train, y_train, X_val, y_val, config):
    """
    Train a Transformer model for price prediction or trading signals.
    
    Args:
        X_train, y_train, X_val, y_val (np.ndarray): Training and validation data.
        config (dict): Configuration dictionary.
    
    Returns:
        torch.nn.Module: Trained Transformer model.
    """
    try:
        class TransformerModel(nn.Module):
            def __init__(self, input_dim, layers, heads, output_dim):
                super().__init__()
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=input_dim, nhead=heads, dim_feedforward=512),
                    num_layers=layers
                )
                self.fc = nn.Linear(input_dim, output_dim)
            
            def forward(self, x):
                x = self.transformer(x)
                x = x.mean(dim=1)  # Global average pooling
                return self.fc(x)
        
        device = torch.device("cuda" if config["DEVICE"] == "cuda" and torch.cuda.is_available() else "cpu")
        model = TransformerModel(X_train.shape[2], config["TRANSFORMER_LAYERS"], config["ATTENTION_HEADS"],
                               1 if config.get("TASK") == "regression" else 3).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])
        criterion = nn.MSELoss() if config.get("TASK") == "regression" else nn.CrossEntropyLoss()
        
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32 if config.get("TASK") == "regression" else torch.long).to(device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32 if config.get("TASK") == "regression" else torch.long).to(device)
        
        best_val_loss = float("inf")
        patience, trials = 10, 0
        
        for epoch in range(config["EPOCHS"]):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t)
            
            logger.info(f"Epoch {epoch+1}/{config['EPOCHS']}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trials = 0
                torch.save(model.state_dict(), config["TRANSFORMER_MODEL_PATH"])
            else:
                trials += 1
                if trials >= patience:
                    logger.info("Early stopping triggered")
                    break
        
        model.load_state_dict(torch.load(config["TRANSFORMER_MODEL_PATH"]))
        logger.info(f"Saved Transformer model to {config['TRANSFORMER_MODEL_PATH']}")
        return model
    
    except Exception as e:
        logger.error(f"Error in train_transformer_model: {str(e)}")
        raise

# Trading Environment for PPO
class TradingEnv(Env):
    """
    Gym-like environment for PPO trading.
    """
    def __init__(self, data, config):
        super().__init__()
        self.data = data
        self.config = config
        self.current_step = 0
        self.max_steps = len(data) - 1
        self.action_space = gym.spaces.Discrete(3)  # Buy, Sell, Hold
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(config["LSTM_FEATURES"]),))
        self.position = 0
        self.cash = 100000
        self.portfolio_value = self.cash
    
    def reset(self):
        self.current_step = 0
        self.position = 0
        self.cash = 100000
        self.portfolio_value = self.cash
        return self._get_observation()
    
    def step(self, action):
        price = self.data.iloc[self.current_step]["close"]
        if action == 0 and self.cash >= price:  # Buy
            self.position += 1
            self.cash -= price
        elif action == 1 and self.position > 0:  # Sell
            self.position -= 1
            self.cash += price
        
        self.current_step += 1
        self.portfolio_value = self.cash + self.position * price
        reward = self.portfolio_value - 100000  # Simple profit-based reward
        done = self.current_step >= self.max_steps
        return self._get_observation(), reward, done, {}
    
    def _get_observation(self):
        return self.data.iloc[self.current_step][self.config["LSTM_FEATURES"]].values

# PPO Training
def train_ppo_agent(trading_env, data, config):
    """
    Train a PPO agent for optimizing trading actions.
    
    Args:
        trading_env (TradingEnv): Trading environment.
        data (pd.DataFrame): Data for environment.
        config (dict): Configuration dictionary.
    
    Returns:
        PPOAgent: Trained PPO agent.
    """
    try:
        class PPOAgent(nn.Module):
            def __init__(self, state_dim, action_dim):
                super().__init__()
                self.actor = nn.Sequential(
                    nn.Linear(state_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, action_dim),
                    nn.Softmax(dim=-1)
                )
                self.critic = nn.Sequential(
                    nn.Linear(state_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1)
                )
            
            def forward(self, state):
                return self.actor(state), self.critic(state)
        
        device = torch.device("cuda" if config["DEVICE"] == "cuda" and torch.cuda.is_available() else "cpu")
        agent = PPOAgent(len(config["LSTM_FEATURES"]), 3).to(device)
        optimizer = torch.optim.Adam(agent.parameters(), lr=config["LEARNING_RATE"])
        
        for episode in range(config["PPO_EPISODES"]):
            state = trading_env.reset()
            episode_reward = 0
            done = False
            states, actions, rewards, log_probs = [], [], [], []
            
            while not done:
                state_t = torch.tensor(state, dtype=torch.float32).to(device)
                action_probs, value = agent(state_t)
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                
                next_state, reward, done, _ = trading_env.step(action.item())
                
                states.append(state_t)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                
                state = next_state
                episode_reward += reward
            
            # Update policy
            returns = []
            R = 0
            for r in rewards[::-1]:
                R = r + config["GAMMA"] * R
                returns.insert(0, R)
            returns = torch.tensor(returns).to(device)
            
            for t in range(len(states)):
                state_t, action_t, log_prob_t = states[t], actions[t], log_probs[t]
                value = agent.critic(state_t)
                advantage = returns[t] - value
                
                actor_loss = -log_prob_t * advantage.detach()
                critic_loss = advantage.pow(2)
                loss = actor_loss + 0.5 * critic_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            logger.info(f"Episode {episode+1}/{config['PPO_EPISODES']}, Reward: {episode_reward:.2f}")
        
        torch.save(agent.state_dict(), config["PPO_MODEL_PATH"])
        logger.info(f"Saved PPO agent to {config['PPO_MODEL_PATH']}")
        return agent
    
    except Exception as e:
        logger.error(f"Error in train_ppo_agent: {str(e)}")
        raise

# FinBERT Fine-Tuning
def finetune_finbert(news_dataset, config):
    """
    Fine-tune FinBERT for forex news sentiment analysis.
    
    Args:
        news_dataset (pd.DataFrame): Dataset with text and label columns.
        config (dict): Configuration dictionary.
    
    Returns:
        tuple: Fine-tuned model and tokenizer.
    """
    try:
        if news_dataset is None or news_dataset.empty:
            logger.warning("No news dataset provided, using pre-trained FinBERT")
            model = BertForSequenceClassification.from_pretrained("prosusfin/finbert")
            tokenizer = BertTokenizer.from_pretrained("prosusfin/finbert")
            return model, tokenizer
        
        class NewsDataset(Dataset):
            def __init__(self, texts, labels):
                self.texts = texts
                self.labels = labels
                self.tokenizer = BertTokenizer.from_pretrained("prosusfin/finbert")
            
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                encoding = self.tokenizer(self.texts[idx], return_tensors="pt", padding="max_length",
                                        truncation=True, max_length=128)
                return {
                    "input_ids": encoding["input_ids"].squeeze(),
                    "attention_mask": encoding["attention_mask"].squeeze(),
                    "labels": torch.tensor(self.labels[idx], dtype=torch.long)
                }
        
        model = BertForSequenceClassification.from_pretrained("prosusfin/finbert")
        tokenizer = BertTokenizer.from_pretrained("prosusfin/finbert")
        dataset = NewsDataset(news_dataset["text"].tolist(), news_dataset["label"].tolist())
        
        training_args = TrainingArguments(
            output_dir=config["FINBERT_MODEL_PATH"],
            num_train_epochs=3,
            per_device_train_batch_size=8,
            learning_rate=config["LEARNING_RATE"],
            evaluation_strategy="epoch",
            logging_dir="logs/finbert",
        )
        
        trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
        trainer.train()
        
        model.save_pretrained(config["FINBERT_MODEL_PATH"])
        tokenizer.save_pretrained(config["FINBERT_MODEL_PATH"])
        logger.info(f"Saved fine-tuned FinBERT to {config['FINBERT_MODEL_PATH']}")
        return model, tokenizer
    
    except Exception as e:
        logger.error(f"Error in finetune_finbert: {str(e)}")
        raise

# Ensemble Weights Optimization
def optimize_ensemble_weights(lstm_preds, transformer_preds, ppo_preds, sentiment_preds, y_true, config):
    """
    Optimize weights for ensemble of model predictions.
    
    Args:
        lstm_preds, transformer_preds, ppo_preds, sentiment_preds (np.ndarray): Model predictions.
        y_true (np.ndarray): True labels/values.
        config (dict): Configuration dictionary.
    
    Returns:
        dict: Optimized ensemble weights.
    """
    try:
        task = config.get("TASK", "regression")
        if task == "regression":
            errors = {
                "lstm": mean_squared_error(y_true, lstm_preds),
                "transformer": mean_squared_error(y_true, transformer_preds),
                "ppo": mean_squared_error(y_true, ppo_preds),
                "sentiment": mean_squared_error(y_true, sentiment_preds)
            }
            inverse_errors = {k: 1.0 / (v + 1e-10) for k, v in errors.items()}
        else:
            errors = {
                "lstm": accuracy_score(y_true, np.argmax(lstm_preds, axis=1)),
                "transformer": accuracy_score(y_true, np.argmax(transformer_preds, axis=1)),
                "ppo": accuracy_score(y_true, np.argmax(ppo_preds, axis=1)),
                "sentiment": accuracy_score(y_true, np.argmax(sentiment_preds, axis=1))
            }
            inverse_errors = errors
        
        total = sum(inverse_errors.values())
        weights = {k: v / total for k, v in inverse_errors.items()}
        
        config["ensemble_weights"] = weights
        ConfigLoader.save_config(config, "config.yaml")
        logger.info(f"Optimized ensemble weights: {weights}")
        return weights
    
    except Exception as e:
        logger.error(f"Error in optimize_ensemble_weights: {str(e)}")
        raise

# Main Training Function
def main(args):
    """
    Orchestrate training of all machine learning components.
    
    Args:
        args: Command-line arguments with timeframe.
    """
    try:
        # Load configuration
        config = ConfigLoader.load_config()
        logger.info("Loaded configuration")
        
        # Prepare data
        data = prepare_training_data(config)
        timeframe = args.timeframe
        train_data = data["train_data"][timeframe]
        val_data = data["val_data"][timeframe]
        test_data = data["test_data"][timeframe]
        
        # Prepare inputs
        X_train = train_data[config["LSTM_FEATURES"]].values.reshape(-1, config["TIMESTEPS"], len(config["LSTM_FEATURES"]))
        y_train = train_data["target"].values
        X_val = val_data[config["LSTM_FEATURES"]].values.reshape(-1, config["TIMESTEPS"], len(config["LSTM_FEATURES"]))
        y_val = val_data["target"].values
        
        # Train models
        lstm_model = train_lstm_model(X_train, y_train, X_val, y_val, config)
        transformer_model = train_transformer_model(X_train, y_train, X_val, y_val, config)
        
        env = TradingEnv(train_data, config)
        ppo_agent = train_ppo_agent(env, train_data, config)
        
        # Fine-tune FinBERT (optional)
        news_dataset = None  # Replace with actual dataset if available
        finbert_model, finbert_tokenizer = finetune_finbert(news_dataset, config)
        
        # Generate predictions
        lstm_preds = lstm_model.predict(X_val)
        transformer_preds = transformer_model(torch.tensor(X_val, dtype=torch.float32).to("cuda" if torch.cuda.is_available() else "cpu")).detach().cpu().numpy()
        ppo_preds = np.zeros_like(lstm_preds)  # Placeholder, replace with actual PPO predictions
        sentiment_preds = np.zeros_like(lstm_preds)  # Placeholder, replace with FinBERT predictions
        
        # Optimize ensemble weights
        weights = optimize_ensemble_weights(lstm_preds, transformer_preds, ppo_preds, sentiment_preds, y_val, config)
        
        # Run backtest
        backtest_engine = BacktestEngine(config)
        backtest_results = backtest_engine.run_backtest({timeframe: test_data}, config)
        metrics = backtest_engine.calculate_metrics(backtest_results)
        backtest_engine.visualize_results(metrics)
        logger.info(f"Backtest metrics: {metrics}")
    
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train trading bot models")
    parser.add_argument("--timeframe", default="M5", choices=["M5", "M15", "H1", "H4"], help="Timeframe for training")
    args = parser.parse_args()
    main(args)