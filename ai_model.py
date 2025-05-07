import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, MultiHeadAttention, LayerNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
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
        
        # Initialize PPO agent
        self.state_dim = len(LSTM_FEATURES) + 12  # Base features + engineered features
        self.action_dim = 3  # Buy, Sell, Hold
        self.ppo_agent = PPOAgent(self.state_dim, self.action_dim)
        self.reward_history = []
        
    def engineer_features(self, df):
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
            self.logger.error(f"Error engineering features: {e}")
            return None
            
    def prepare_model_data(self, df):
        """Prepare data for both LSTM and Transformer models"""
        try:
            # Engineer features
            df = self.engineer_features(df)
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
        """Train both LSTM and Transformer models with adaptive learning"""
        try:
            # Prepare data
            X, y = self.prepare_model_data(df)
            if X is None or y is None:
                return False
            
            # Define callbacks
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
            
            lstm_history = self.lstm_model.fit(X, y,
                                              batch_size=LSTM_BATCH_SIZE,
                                              epochs=LSTM_EPOCHS,
                                              validation_split=0.2,
                                              callbacks=callbacks,
                                              verbose=0)
            
            # Build and train Transformer model
            self.transformer_model = self.build_transformer_model((LSTM_SEQUENCE_LENGTH, X.shape[2]))
            if self.transformer_model is None:
                return False
            
            self.transformer_model.compile(optimizer=Adam(learning_rate=0.001),
                                         loss='mse',
                                         metrics=['mae'])
            
            transformer_history = self.transformer_model.fit(X, y,
                                                           batch_size=LSTM_BATCH_SIZE,
                                                           epochs=LSTM_EPOCHS,
                                                           validation_split=0.2,
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

    def calculate_reward(self, action, next_price, current_price):
        """Calculate reward for the PPO agent"""
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
            
            # Scale reward and add risk-adjusted component
            reward = np.clip(reward * 100, -1, 1)  # Scale and clip rewards
            return reward
            
        except Exception as e:
            self.logger.error(f"Error calculating reward: {e}")
            return 0.0
    
    def train_ppo(self, df):
        """Train PPO agent on historical data"""
        try:
            # Prepare state features
            df = self.engineer_features(df)
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