import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, MultiHeadAttention, LayerNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn as nn
from torch.optim import Adam as TorchAdam
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from config import *
import logging
from datetime import datetime
import tensorflow as tf
from collections import deque
import random

class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 0.2
        self.actor_lr = 0.0003
        self.critic_lr = 0.001
        self.batch_size = 64
        
        # Actor (Policy) and Critic (Value) networks
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        
    def _build_actor(self):
        inputs = Input(shape=(self.state_dim,))
        x = Dense(256, activation='relu')(inputs)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(self.action_dim, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=self.actor_lr))
        return model
        
    def _build_critic(self):
        inputs = Input(shape=(self.state_dim,))
        x = Dense(256, activation='relu')(inputs)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(1, activation=None)(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=self.critic_lr), loss='mse')
        return model
        
    def get_action(self, state):
        state = np.array([state])
        action_probs = self.actor.predict(state, verbose=0)[0]
        action = np.random.choice(self.action_dim, p=action_probs)
        return action, action_probs
        
    def store_transition(self, state, action, reward, next_state, done, action_probs):
        self.memory.append([state, action, reward, next_state, done, action_probs])
        
    def train(self):
        if len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])
        old_action_probs = np.array([x[5] for x in batch])
        
        # Compute advantages
        values = self.critic.predict(states, verbose=0)
        next_values = self.critic.predict(next_states, verbose=0)
        advantages = rewards + self.gamma * next_values * (1 - dones) - values
        
        # Update critic
        self.critic.train_on_batch(states, rewards + self.gamma * next_values * (1 - dones))
        
        # Update actor
        with tf.GradientTape() as tape:
            current_action_probs = self.actor(states)
            ratio = tf.exp(tf.math.log(current_action_probs + 1e-10) - tf.math.log(old_action_probs + 1e-10))
            surr1 = ratio * advantages
            surr2 = tf.clip_by_value(ratio, 1-self.epsilon, 1+self.epsilon) * advantages
            actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
            
        grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

class AIModel:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.lstm_model = None
        self.transformer_model = None
        self.ensemble_weights = {'lstm': 0.3, 'transformer': 0.3, 'sentiment': 0.2, 'ppo': 0.2}
        self.feature_scaler = StandardScaler()
        self.price_scaler = MinMaxScaler()
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
        self.model_performance = {'lstm': [], 'transformer': [], 'sentiment': [], 'ppo': []}
        self.last_retrain = datetime.now()

    def optimize_ensemble_weights(self, lstm_preds, transformer_preds, ppo_preds, sentiment_preds, y_true, config):
        """Optimize weights for combining predictions from different models based on their performance.

        Args:
            lstm_preds (np.ndarray): LSTM model predictions
            transformer_preds (np.ndarray): Transformer model predictions
            ppo_preds (np.ndarray): PPO agent predictions
            sentiment_preds (np.ndarray): Sentiment model predictions
            y_true (np.ndarray): True labels/values
            config (dict): Configuration dictionary

        Returns:
            dict: Optimized weights for each model

        Raises:
            ValueError: If prediction arrays have different shapes or invalid values
            RuntimeError: If numerical issues occur during optimization
            Exception: For other unexpected errors
        """
        try:
            # Setup logging
            logger = logging.getLogger(__name__)
            logger.info("Starting ensemble weight optimization...")

            # Validate inputs
            predictions = {
                'lstm': lstm_preds,
                'transformer': transformer_preds,
                'ppo': ppo_preds,
                'sentiment': sentiment_preds
            }
            
            # Check shapes
            base_shape = y_true.shape
            for model_name, preds in predictions.items():
                if preds.shape != base_shape:
                    raise ValueError(f"Shape mismatch for {model_name}: {preds.shape} vs {base_shape}")

            # Calculate performance metrics for each model
            metrics = {}
            for model_name, preds in predictions.items():
                # Use MSE for regression, accuracy for classification
                if y_true.shape[1] == 1:  # Regression
                    mse = np.mean((preds - y_true) ** 2)
                    metrics[model_name] = 1 / (mse + 1e-10)  # Inverse MSE, avoid division by zero
                else:  # Classification
                    accuracy = np.mean(np.argmax(preds, axis=1) == np.argmax(y_true, axis=1))
                    metrics[model_name] = accuracy

            # Normalize metrics to get weights
            total_metric = sum(metrics.values())
            weights = {model: score/total_metric for model, score in metrics.items()}

            # Update ensemble weights
            self.ensemble_weights = weights
            logger.info(f"Optimized weights: {weights}")

            # Update config and save
            config['ENSEMBLE_WEIGHTS'] = weights
            try:
                from config import ConfigLoader
                ConfigLoader.save_config(config)
                logger.info("Updated weights saved to config file")
            except Exception as e:
                logger.error(f"Error saving config: {e}")

            # Log performance metrics
            for model_name, metric in metrics.items():
                self.model_performance[model_name].append(metric)
                logger.info(f"{model_name} performance metric: {metric:.4f}")

            return weights

        except ValueError as ve:
            logger.error(f"Validation error: {ve}")
            raise
        except RuntimeError as re:
            logger.error(f"Runtime error during optimization: {re}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during weight optimization: {e}")
            raise
        
        # Initialize PPO agent
        self.state_dim = len(LSTM_FEATURES) + 12  # Base features + engineered features
        self.action_dim = 3  # Buy, Sell, Hold
        self.ppo_agent = PPOAgent(self.state_dim, self.action_dim)
        self.reward_history = []
        
    def engineer_features(df):
        """Advanced feature engineering for time series data"""
        try:
            # Cyclical encoding of time features
            df['hour_sin'] = np.sin(2 * np.pi * pd.to_datetime(df.index).hour / 24)
            df['hour_cos'] = np.cos(2 * np.pi * pd.to_datetime(df.index).hour / 24)
            df['day_sin'] = np.sin(2 * np.pi * pd.to_datetime(df.index).dayofweek / 7)
            df['day_cos'] = np.cos(2 * np.pi * pd.to_datetime(df.index).dayofweek / 7)
            df['month_sin'] = np.sin(2 * np.pi * pd.to_datetime(df.index).month / 12)
            df['month_cos'] = np.cos(2 * np.pi * pd.to_datetime(df.index).month / 12)
            
            # Technical indicator ratios
            df['price_volatility'] = df['high'] / df['low'] - 1
            df['volume_price_trend'] = df['volume'] * (df['close'] - df['open'])
            df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
            
            # Volatility regime features
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=20).std()
            df['volatility_regime'] = np.where(df['volatility'] > df['volatility'].mean(), 1, 0)
            
            # Price action patterns
            df['body_size'] = abs(df['close'] - df['open'])
            df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
            df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
            
            return df.fillna(0)
            
        except Exception as e:
            logging.error(f"Error engineering features: {e}")
            return None
            
    def prepare_model_data(self, df):
        """Prepare data for both LSTM and Transformer models"""
        try:
            # Engineer features
            df = engineer_features(df)
            if df is None:
                return None, None
            
            # Select and scale features
            features = df[LSTM_FEATURES + [
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
                'price_volatility', 'volume_price_trend', 'volatility', 'body_size',
                'upper_shadow', 'lower_shadow'
            ]].values
            
            prices = df[['close']].values
            
            # Scale data
            scaled_features = self.feature_scaler.fit_transform(features)
            scaled_prices = self.price_scaler.fit_transform(prices)
            
            # Create sequences
            X, y = [], []
            for i in range(len(scaled_features) - LSTM_SEQUENCE_LENGTH):
                X.append(scaled_features[i:(i + LSTM_SEQUENCE_LENGTH)])
                y.append(scaled_prices[i + LSTM_SEQUENCE_LENGTH])
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            self.logger.error(f"Error preparing model data: {e}")
            return None, None

    def train_lstm_model(self, X_train, y_train, X_val, y_val, config: dict) -> Sequential:
        """Train an LSTM model for predicting XAUUSD price movements or trading signals.

        Args:
            X_train (np.ndarray): Training features array
            y_train (np.ndarray): Training target array
            X_val (np.ndarray): Validation features array
            y_val (np.ndarray): Validation target array
            config (dict): Configuration dictionary containing model parameters

        Returns:
            Sequential: Trained LSTM model

        Raises:
            ValueError: If input shapes are invalid
            RuntimeError: If GPU memory issues occur
            IOError: If model saving fails
            Exception: For other unexpected errors
        """

    def train_transformer_model(self, X_train, y_train, X_val, y_val, config: dict) -> nn.Module:
        """Train a Transformer model for predicting XAUUSD price movements or trading signals.

        Args:
            X_train (np.ndarray): Training features array of shape (samples, sequence_length, features)
            y_train (np.ndarray): Training target array of shape (samples, output_dim)
            X_val (np.ndarray): Validation features array of shape (samples, sequence_length, features)
            y_val (np.ndarray): Validation target array of shape (samples, output_dim)
            config (dict): Configuration dictionary containing model parameters

        Returns:
            nn.Module: Trained Transformer model

        Raises:
            ValueError: If input shapes are invalid
            RuntimeError: If GPU memory issues occur
            IOError: If model saving fails
            Exception: For other unexpected errors
        """
        try:
            # Setup logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                filename='trading_bot.log'
            )
            logger = logging.getLogger(__name__)
            logger.info("Starting Transformer model training...")

            # Validate input shapes
            if len(X_train.shape) != 3 or len(y_train.shape) != 2:
                raise ValueError(f"Invalid input shapes. Expected X: (samples, sequence_length, features), "
                               f"y: (samples, output_dim). Got X: {X_train.shape}, y: {y_train.shape}")

            # Set device
            device = torch.device(config['DEVICE'] if torch.cuda.is_available() and config['DEVICE'].lower() == 'cuda' else 'cpu')
            logger.info(f"Using device: {device}")

            # Convert numpy arrays to PyTorch tensors
            X_train = torch.FloatTensor(X_train).to(device)
            y_train = torch.FloatTensor(y_train).to(device)
            X_val = torch.FloatTensor(X_val).to(device)
            y_val = torch.FloatTensor(y_val).to(device)

            # Create data loaders
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
            train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'])

            # Build Transformer model
            class TransformerModel(nn.Module):
                def __init__(self, input_dim, d_model=512):
                    super().__init__()
                    self.embedding = nn.Linear(input_dim, d_model)
                    encoder_layer = nn.TransformerEncoderLayer(
                        d_model=d_model,
                        nhead=config['ATTENTION_HEADS'],
                        dim_feedforward=2048,
                        dropout=0.1,
                        batch_first=True
                    )
                    self.transformer = nn.TransformerEncoder(
                        encoder_layer,
                        num_layers=config['TRANSFORMER_LAYERS']
                    )
                    self.fc = nn.Linear(d_model, y_train.shape[1])

                def forward(self, x):
                    x = self.embedding(x)
                    x = self.transformer(x)
                    return self.fc(x[:, -1, :])

            model = TransformerModel(X_train.shape[2]).to(device)
            criterion = nn.MSELoss() if y_train.shape[1] == 1 else nn.CrossEntropyLoss()
            optimizer = TorchAdam(model.parameters(), lr=config['LEARNING_RATE'])

            # Training loop with early stopping
            best_val_loss = float('inf')
            patience = 10
            patience_counter = 0
            best_model_state = None

            logger.info("Training Transformer model...")
            for epoch in range(config['EPOCHS']):
                # Training phase
                model.train()
                train_loss = 0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                # Validation phase
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = model(batch_X)
                        val_loss += criterion(outputs, batch_y).item()

                train_loss /= len(train_loader)
                val_loss /= len(val_loader)
                logger.info(f"Epoch {epoch+1}/{config['EPOCHS']} - "
                           f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping triggered after {epoch+1} epochs")
                        break

            # Load best model and save
            model.load_state_dict(best_model_state)
            model_dir = Path(config['TRANSFORMER_MODEL_PATH']).parent
            model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), config['TRANSFORMER_MODEL_PATH'])
            logger.info(f"Model saved to {config['TRANSFORMER_MODEL_PATH']}")

            return model

        except ValueError as ve:
            logger.error(f"Invalid input error: {ve}")
            raise
        except RuntimeError as re:
            logger.error(f"Runtime error (possibly GPU-related): {re}")
            raise
        except IOError as io:
            logger.error(f"I/O error when saving model: {io}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during Transformer training: {e}")
            raise
        try:
            # Setup logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                filename='trading_bot.log'
            )
            logger = logging.getLogger(__name__)
            logger.info("Starting LSTM model training...")

            # Validate input shapes
            if len(X_train.shape) != 3 or len(y_train.shape) != 2:
                raise ValueError(f"Invalid input shapes. Expected X: (samples, timesteps, features), "
                               f"y: (samples, output_dim). Got X: {X_train.shape}, y: {y_train.shape}")

            # Set device strategy
            if config['DEVICE'].lower() == 'cuda' and tf.config.list_physical_devices('GPU'):
                device_strategy = tf.device('/GPU:0')
                logger.info("Using GPU for training")
            else:
                device_strategy = tf.device('/CPU:0')
                logger.info("Using CPU for training")

            with device_strategy:
                # Build LSTM model
                model = Sequential([
                    LSTM(config['LSTM_UNITS'], return_sequences=True,
                         input_shape=(X_train.shape[1], X_train.shape[2])),
                    Dropout(0.2),
                    LSTM(config['LSTM_UNITS']),
                    Dropout(0.2),
                    Dense(y_train.shape[1], activation='linear' if y_train.shape[1] == 1 else 'softmax')
                ])

                # Compile model
                model.compile(
                    optimizer=Adam(learning_rate=config['LEARNING_RATE']),
                    loss='mse' if y_train.shape[1] == 1 else 'categorical_crossentropy',
                    metrics=['mae'] if y_train.shape[1] == 1 else ['accuracy']
                )

                # Setup callbacks
                callbacks = [
                    EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True
                    ),
                    ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=5,
                        min_lr=1e-6
                    )
                ]

                # Create models directory if it doesn't exist
                model_dir = Path(config['LSTM_MODEL_PATH']).parent
                model_dir.mkdir(parents=True, exist_ok=True)

                # Add ModelCheckpoint callback
                callbacks.append(
                    ModelCheckpoint(
                        config['LSTM_MODEL_PATH'],
                        monitor='val_loss',
                        save_best_only=True,
                        save_weights_only=False
                    )
                )

                # Train model
                logger.info("Training LSTM model...")
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=config['EPOCHS'],
                    batch_size=config['BATCH_SIZE'],
                    callbacks=callbacks,
                    verbose=1
                )

                # Log training results
                final_loss = history.history['loss'][-1]
                final_val_loss = history.history['val_loss'][-1]
                logger.info(f"Training completed - Loss: {final_loss:.4f}, Val Loss: {final_val_loss:.4f}")

                # Save model
                logger.info(f"Saving model to {config['LSTM_MODEL_PATH']}")
                model.save(config['LSTM_MODEL_PATH'])

                return model

        except ValueError as ve:
            logger.error(f"Invalid input error: {ve}")
            raise
        except tf.errors.ResourceExhaustedError as re:
            logger.error(f"GPU memory error: {re}")
            logger.info("Attempting to continue on CPU...")
            return self.train_lstm_model(X_train, y_train, X_val, y_val, {**config, 'DEVICE': 'cpu'})
        except IOError as io:
            logger.error(f"Model saving error: {io}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during LSTM training: {e}")
            raise

    def build_transformer_model(self, input_shape):
        """Build Transformer model for time series prediction"""
        try:
            inputs = Input(shape=input_shape)
            
            # Add positional encoding
            pos_encoding = np.zeros(input_shape)
            for pos in range(input_shape[0]):
                for i in range(0, input_shape[1], 2):
                    pos_encoding[pos, i] = np.sin(pos / (10000 ** (2 * i / input_shape[1])))
                    pos_encoding[pos, i + 1] = np.cos(pos / (10000 ** (2 * i / input_shape[1])))
            
            x = inputs + pos_encoding
            
            # Multi-head attention layers
            x = MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
            x = LayerNormalization(epsilon=1e-6)(x)
            x = Dropout(0.1)(x)
            
            # Dense layers
            x = Dense(128, activation='relu')(x)
            x = Dropout(0.1)(x)
            x = Dense(64, activation='relu')(x)
            x = LayerNormalization(epsilon=1e-6)(x)
            
            # Output layer
            x = Dense(32, activation='relu')(x)
            x = Dense(1)(x[:, -1, :])
            
            return Model(inputs=inputs, outputs=x)
            
        except Exception as e:
            self.logger.error(f"Error building transformer model: {e}")
            return None
    
    def train_models(self, df):
        """Train both LSTM and Transformer models with proper validation"""
        try:
            # Prepare data
            X, y = self.prepare_model_data(df)
            if X is None or y is None:
                return False
            
            # Split data into train/validation/test (60/20/20)
            train_size = int(len(X) * 0.6)
            val_size = int(len(X) * 0.2)
            X_train = X[:train_size]
            y_train = y[:train_size]
            X_val = X[train_size:train_size+val_size]
            y_val = y[train_size:train_size+val_size]
            
            # Define callbacks with validation monitoring
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
                ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
            ]
            
            # Build and train LSTM model
            self.lstm_model = Sequential([
                LSTM(LSTM_UNITS, return_sequences=True, 
                     input_shape=(LSTM_SEQUENCE_LENGTH, X.shape[2])),
                Dropout(0.2),
                LSTM(LSTM_UNITS // 2),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(1)
            ])
            
            self.lstm_model.compile(optimizer=Adam(learning_rate=0.001),
                                  loss='mse',
                                  metrics=['mae'])
            
            lstm_history = self.lstm_model.fit(X_train, y_train,
                                              batch_size=LSTM_BATCH_SIZE,
                                              epochs=LSTM_EPOCHS,
                                              validation_data=(X_val, y_val),
                                              callbacks=callbacks,
                                              verbose=0)
            
            # Build and train Transformer model
            self.transformer_model = self.build_transformer_model((LSTM_SEQUENCE_LENGTH, X.shape[2]))
            if self.transformer_model is None:
                return False
            
            self.transformer_model.compile(optimizer=Adam(learning_rate=0.001),
                                         loss='mse',
                                         metrics=['mae'])
            
            transformer_history = self.transformer_model.fit(X_train, y_train,
                                                           batch_size=LSTM_BATCH_SIZE,
                                                           epochs=LSTM_EPOCHS,
                                                           validation_data=(X_val, y_val),
                                                           callbacks=callbacks,
                                                           verbose=0)
            
            # Update model performance metrics
            self.model_performance['lstm'].append(lstm_history.history['val_loss'][-1])
            self.model_performance['transformer'].append(transformer_history.history['val_loss'][-1])
            
            # Update ensemble weights based on recent performance
            total_performance = sum(1/x for x in [self.model_performance['lstm'][-1],
                                                self.model_performance['transformer'][-1]])
            self.ensemble_weights['lstm'] = (1/self.model_performance['lstm'][-1]) / total_performance * 0.8
            self.ensemble_weights['transformer'] = (1/self.model_performance['transformer'][-1]) / total_performance * 0.8
            self.ensemble_weights['sentiment'] = 0.2
            
            self.last_retrain = datetime.now()
            return True
            
        except Exception as e:
            self.logger.error(f"Error training models: {e}")
            return False

    def predict_price(self, df):
        """Make ensemble predictions with confidence intervals"""
        try:
            if self.lstm_model is None or self.transformer_model is None:
                return None
            
            # Check if retraining is needed (e.g., every 24 hours)
            if (datetime.now() - self.last_retrain).days >= 1:
                self.train_models(df)
            
            # Prepare latest data
            df = self.engineer_features(df)
            if df is None:
                return None
                
            features = df[LSTM_FEATURES + [
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
                'price_volatility', 'volume_price_trend', 'volatility', 'body_size',
                'upper_shadow', 'lower_shadow'
            ]].tail(LSTM_SEQUENCE_LENGTH).values
            
            scaled_features = self.feature_scaler.transform(features)
            X = np.array([scaled_features])
            
            # Get predictions from both models
            lstm_pred = self.lstm_model.predict(X, verbose=0)[0][0]
            transformer_pred = self.transformer_model.predict(X, verbose=0)[0][0]
            
            # Calculate ensemble prediction
            ensemble_pred = (
                lstm_pred * self.ensemble_weights['lstm'] +
                transformer_pred * self.ensemble_weights['transformer']
            )
            
            # Calculate confidence interval
            pred_std = np.std([lstm_pred, transformer_pred])
            confidence_interval = {
                'lower': ensemble_pred - 1.96 * pred_std,
                'upper': ensemble_pred + 1.96 * pred_std
            }
            
            return {
                'prediction': self.price_scaler.inverse_transform([[ensemble_pred]])[0][0],
                'confidence_interval': {
                    'lower': self.price_scaler.inverse_transform([[confidence_interval['lower']]])[0][0],
                    'upper': self.price_scaler.inverse_transform([[confidence_interval['upper']]])[0][0]
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error making ensemble prediction: {e}")
            return None

    def analyze_sentiment(self, news_texts):
        """Analyze sentiment of news articles using FinBERT"""
        try:
            if not news_texts:
                return None
                
            # Combine texts and tokenize
            combined_text = ' '.join(news_texts[:5])  # Analyze latest 5 news articles
            inputs = self.sentiment_tokenizer(combined_text, 
                                            return_tensors="pt",
                                            max_length=512,
                                            truncation=True)
            
            # Get sentiment prediction
            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)
                predictions = torch.softmax(outputs.logits, dim=1)
                
            # Convert to sentiment score (0-1)
            sentiment_score = predictions[0][0].item()  # Probability of positive sentiment
            return sentiment_score
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {e}")
            return None

    def calculate_reward(self, action, next_price, current_price, drawdown=0.0):
        """Calculate risk-adjusted reward for the PPO agent"""
        try:
            # Calculate price change percentage
            price_change = (next_price - current_price) / current_price
            
            # Define rewards based on action and price movement
            if action == 0:  # Buy
                reward = price_change
            elif action == 1:  # Sell
                reward = -price_change
            else:  # Hold
                reward = -abs(price_change) * 0.1  # Small penalty for holding
            
            # Apply risk adjustment based on drawdown
            risk_factor = 1 - min(drawdown / 0.1, 1)  # Reduce reward as drawdown approaches 10%
            reward = reward * risk_factor
            
            # Scale and clip final reward
            reward = np.clip(reward * 100, -1, 1)
            return reward
            
        except Exception as e:
            self.logger.error(f"Error calculating reward: {e}")
            return 0.0
    
    def train_ppo(self, df):
        """Train PPO agent on historical data"""
        try:
            # Prepare state features
            df = engineer_features(df)
            if df is None:
                return False
            
            features = df[LSTM_FEATURES + [
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
                'price_volatility', 'volume_price_trend', 'volatility', 'body_size',
                'upper_shadow', 'lower_shadow'
            ]].values
            
            scaled_features = self.feature_scaler.transform(features)
            prices = df['close'].values
            
            # Training loop
            total_reward = 0
            for i in range(len(scaled_features) - 1):
                state = scaled_features[i]
                next_state = scaled_features[i + 1]
                current_price = prices[i]
                next_price = prices[i + 1]
                
                # Get action from PPO agent
                action, action_probs = self.ppo_agent.get_action(state)
                
                # Calculate reward
                reward = self.calculate_reward(action, next_price, current_price)
                total_reward += reward
                
                # Store transition and train
                done = (i == len(scaled_features) - 2)
                self.ppo_agent.store_transition(state, action, reward, next_state, done, action_probs)
                self.ppo_agent.train()
            
            # Store performance metrics
            self.model_performance['ppo'].append(total_reward)
            self.reward_history.append(total_reward)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error training PPO agent: {e}")
            return False
    
    def optimize_ppo_threshold(self, df, thresholds=None):
        """Find optimal PPO confidence threshold using historical data"""
        try:
            if thresholds is None:
                thresholds = np.arange(0.5, 0.9, 0.1)
            
            best_threshold = 0.6  # Default threshold
            best_sharpe = float('-inf')
            
            # Split data for optimization
            train_size = int(len(df) * 0.8)
            optimization_data = df[:train_size]
            
            for threshold in thresholds:
                returns = []
                position = 0  # -1: short, 0: flat, 1: long
                
                for i in range(len(optimization_data) - 1):
                    current_data = optimization_data.iloc[:i+1]
                    signals = self.generate_signals(current_data, [])
                    
                    if signals and signals['ppo_confidence'] > threshold:
                        if signals['ppo_action'] == 'BUY':
                            position = 1
                        elif signals['ppo_action'] == 'SELL':
                            position = -1
                    
                    # Calculate returns
                    if position != 0 and i < len(optimization_data) - 1:
                        price_change = ((optimization_data.iloc[i+1]['close'] - 
                                        optimization_data.iloc[i]['close']) /
                                        optimization_data.iloc[i]['close'])
                        returns.append(position * price_change)
                    else:
                        returns.append(0)
                
                if returns:
                    returns = np.array(returns)
                    avg_return = np.mean(returns)
                    std_return = np.std(returns)
                    sharpe = np.sqrt(252) * avg_return / std_return if std_return > 0 else float('-inf')
                    
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_threshold = threshold
            
            self.logger.info(f"Optimal PPO confidence threshold: {best_threshold:.2f} (Sharpe: {best_sharpe:.2f})")
            return best_threshold
            
        except Exception as e:
            self.logger.error(f"Error optimizing PPO threshold: {e}")
            return 0.6  # Return default threshold on error

    def generate_signals(self, df, news_texts):
        """Generate trading signals using ensemble of AI models including PPO"""
        try:
            signals = {}
            
            # Get price predictions with confidence intervals
            price_pred = self.predict_price(df)
            if price_pred is not None:
                signals['prediction'] = price_pred['prediction']
                signals['confidence_interval'] = price_pred['confidence_interval']
            
            # Get sentiment analysis
            if news_texts:
                sentiment = self.analyze_sentiment(news_texts)
                if sentiment is not None:
                    signals['sentiment'] = sentiment
            
            # Get PPO agent's action
            current_state = self.feature_scaler.transform([df[LSTM_FEATURES + [
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
                'price_volatility', 'volume_price_trend', 'volatility', 'body_size',
                'upper_shadow', 'lower_shadow'
            ]].iloc[-1].values])
            
            action, action_probs = self.ppo_agent.get_action(current_state[0])
            signals['ppo_action'] = ['BUY', 'SELL', 'HOLD'][action]
            signals['ppo_confidence'] = float(max(action_probs))
            
            # Combine all signals
            if 'prediction' in signals:
                # Adjust prediction based on PPO and sentiment
                ppo_adjustment = (action - 1) * signals['ppo_confidence'] * self.ensemble_weights['ppo']
                sentiment_adjustment = ((sentiment - 0.5) * 2 * self.ensemble_weights['sentiment']) if 'sentiment' in signals else 0
                signals['prediction'] *= (1 + ppo_adjustment + sentiment_adjustment)
            
            # Add prediction confidence
            if len(self.model_performance['lstm']) > 0:
                signals['model_confidence'] = 1 / (1 + np.mean([
                    self.model_performance['lstm'][-1],
                    self.model_performance['transformer'][-1],
                    -self.model_performance['ppo'][-1]  # Higher rewards are better
                ]))
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating AI signals: {e}")
            return None

    def train_ppo_agent(self, trading_env, data, config: dict) -> PPOAgent:
        """Train a PPO agent for optimizing XAUUSD trading actions.

        Args:
            trading_env: Gym-like environment with market data
            data (pd.DataFrame): DataFrame for environment setup
            config (dict): Configuration dictionary containing model parameters

        Returns:
            PPOAgent: Trained PPO agent

        Raises:
            ValueError: If environment setup fails
            RuntimeError: If numerical instability occurs
            IOError: If model saving fails
            Exception: For other unexpected errors
        """
        try:
            # Setup logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                filename='trading_bot.log'
            )
            logger = logging.getLogger(__name__)
            logger.info("Starting PPO agent training...")

            # Initialize PPO agent
            state_dim = len(LSTM_FEATURES) + 12  # Base features + engineered features
            action_dim = 3  # Buy, Sell, Hold
            device = torch.device(config['DEVICE'] if torch.cuda.is_available() and config['DEVICE'].lower() == 'cuda' else 'cpu')
            
            # Create PPO networks
            class ActorCritic(nn.Module):
                def __init__(self, state_dim, action_dim):
                    super().__init__()
                    # Actor network
                    self.actor = nn.Sequential(
                        nn.Linear(state_dim, 128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.ReLU(),
                        nn.Linear(128, action_dim),
                        nn.Softmax(dim=-1)
                    )
                    # Critic network
                    self.critic = nn.Sequential(
                        nn.Linear(state_dim, 128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.ReLU(),
                        nn.Linear(128, 1)
                    )

                def forward(self, state):
                    return self.actor(state), self.critic(state)

            # Initialize networks and optimizer
            ppo_net = ActorCritic(state_dim, action_dim).to(device)
            optimizer = TorchAdam(ppo_net.parameters(), lr=config['LEARNING_RATE'])

            # Training loop
            episodes = config['PPO_EPISODES']
            gamma = config['GAMMA']
            epsilon_clip = config['EPSILON_CLIP']
            best_reward = float('-inf')
            rewards_history = []

            logger.info(f"Training for {episodes} episodes...")
            for episode in range(episodes):
                state = trading_env.reset()
                done = False
                total_reward = 0
                transitions = []

                while not done:
                    state_tensor = torch.FloatTensor(state).to(device)
                    action_probs, value = ppo_net(state_tensor)
                    action = torch.multinomial(action_probs, 1).item()
                    next_state, reward, done, _ = trading_env.step(action)

                    # Store transition
                    transitions.append({
                        'state': state,
                        'action': action,
                        'reward': reward,
                        'next_state': next_state,
                        'action_prob': action_probs[action].item(),
                        'value': value.item()
                    })

                    state = next_state
                    total_reward += reward

                # Process episode transitions
                states = torch.FloatTensor([t['state'] for t in transitions]).to(device)
                actions = torch.LongTensor([t['action'] for t in transitions]).to(device)
                rewards = torch.FloatTensor([t['reward'] for t in transitions]).to(device)
                old_probs = torch.FloatTensor([t['action_prob'] for t in transitions]).to(device)
                values = torch.FloatTensor([t['value'] for t in transitions]).to(device)

                # Compute returns and advantages
                returns = []
                advantages = []
                R = 0
                for r in reversed(rewards):
                    R = r + gamma * R
                    returns.insert(0, R)
                returns = torch.FloatTensor(returns).to(device)
                advantages = returns - values
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # PPO update
                for _ in range(10):  # Multiple epochs of updates
                    new_probs, new_values = ppo_net(states)
                    new_probs = new_probs.gather(1, actions.unsqueeze(1)).squeeze()

                    ratio = new_probs / old_probs
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1 - epsilon_clip, 1 + epsilon_clip) * advantages
                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = nn.MSELoss()(new_values.squeeze(), returns)
                    loss = actor_loss + 0.5 * critic_loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                rewards_history.append(total_reward)
                avg_reward = sum(rewards_history[-100:]) / min(len(rewards_history), 100)

                if episode % 100 == 0:
                    logger.info(f"Episode {episode}/{episodes}, Avg Reward: {avg_reward:.2f}")

                # Save best model
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    torch.save(ppo_net.state_dict(), config['PPO_MODEL_PATH'])
                    logger.info(f"New best model saved with average reward: {best_reward:.2f}")

            logger.info("PPO training completed successfully")
            return self.ppo_agent

        except ValueError as ve:
            logger.error(f"Environment setup error: {ve}")
            raise
        except RuntimeError as re:
            logger.error(f"Numerical instability error: {re}")
            raise
        except IOError as io:
            logger.error(f"Model saving error: {io}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during PPO training: {e}")
            raise

    def optimize_ppo_threshold(self, df, thresholds=None):
        """Find optimal PPO confidence threshold using historical data"""
        try:
            if thresholds is None:
                thresholds = np.arange(0.5, 0.9, 0.1)
            
            best_threshold = 0.6  # Default threshold
            best_sharpe = float('-inf')
            
            # Split data for optimization
            train_size = int(len(df) * 0.8)
            optimization_data = df[:train_size]
            
            for threshold in thresholds:
                returns = []
                position = 0  # -1: short, 0: flat, 1: long
                
                for i in range(len(optimization_data) - 1):
                    current_data = optimization_data.iloc[:i+1]
                    signals = self.generate_signals(current_data, [])
                    
                    if signals and signals['ppo_confidence'] > threshold:
                        if signals['ppo_action'] == 'BUY':
                            position = 1
                        elif signals['ppo_action'] == 'SELL':
                            position = -1
                    
                    # Calculate returns
                    if position != 0 and i < len(optimization_data) - 1:
                        price_change = ((optimization_data.iloc[i+1]['close'] - 
                                        optimization_data.iloc[i]['close']) /
                                        optimization_data.iloc[i]['close'])
                        returns.append(position * price_change)
                    else:
                        returns.append(0)
                
                if returns:
                    returns = np.array(returns)
                    avg_return = np.mean(returns)
                    std_return = np.std(returns)
                    sharpe = np.sqrt(252) * avg_return / std_return if std_return > 0 else float('-inf')
                    
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_threshold = threshold
            
            self.logger.info(f"Optimal PPO confidence threshold: {best_threshold:.2f} (Sharpe: {best_sharpe:.2f})")
            return best_threshold
            
        except Exception as e:
            self.logger.error(f"Error optimizing PPO threshold: {e}")
            return 0.6  # Return default threshold on error

    def generate_signals(self, df, news_texts):
        """Generate trading signals using ensemble of AI models including PPO"""
        try:
            signals = {}
            
            # Get price predictions with confidence intervals
            price_pred = self.predict_price(df)
            if price_pred is not None:
                signals['prediction'] = price_pred['prediction']
                signals['confidence_interval'] = price_pred['confidence_interval']
            
            # Get sentiment analysis
            if news_texts:
                sentiment = self.analyze_sentiment(news_texts)
                if sentiment is not None:
                    signals['sentiment'] = sentiment
            
            # Get PPO agent's action
            current_state = self.feature_scaler.transform([df[LSTM_FEATURES + [
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
                'price_volatility', 'volume_price_trend', 'volatility', 'body_size',
                'upper_shadow', 'lower_shadow'
            ]].iloc[-1].values])
            
            action, action_probs = self.ppo_agent.get_action(current_state[0])
            signals['ppo_action'] = ['BUY', 'SELL', 'HOLD'][action]
            signals['ppo_confidence'] = float(max(action_probs))
            
            # Combine all signals
            if 'prediction' in signals:
                # Adjust prediction based on PPO and sentiment
                ppo_adjustment = (action - 1) * signals['ppo_confidence'] * self.ensemble_weights['ppo']
                sentiment_adjustment = ((sentiment - 0.5) * 2 * self.ensemble_weights['sentiment']) if 'sentiment' in signals else 0
                signals['prediction'] *= (1 + ppo_adjustment + sentiment_adjustment)
            
            # Add prediction confidence
            if len(self.model_performance['lstm']) > 0:
                signals['model_confidence'] = 1 / (1 + np.mean([
                    self.model_performance['lstm'][-1],
                    self.model_performance['transformer'][-1],
                    -self.model_performance['ppo'][-1]  # Higher rewards are better
                ]))
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating AI signals: {e}")
            return None