"""
Machine learning model training and prediction module.
"""

import pandas as pd
import numpy as np
import pickle
import os
import logging
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score
from config import STRATEGY_PARAMS, ENV_CONFIG

logger = logging.getLogger("BinanceTrading.MLModel")


class MLModel:
    """Machine learning model for trading predictions"""
    
    def __init__(self, strategy_params=None):
        self.strategy_params = strategy_params or STRATEGY_PARAMS
        self.model = None
        
    def load_ml_model(self):
        """Load trained ML model if it exists"""
        try:
            if os.path.exists(ENV_CONFIG['model_file']):
                with open(ENV_CONFIG['model_file'], 'rb') as f:
                    self.model = pickle.load(f)
                logger.info("ML model loaded successfully")
                return self.model
            else:
                logger.info("No ML model found, will train new one")
                return None
        except Exception as e:
            logger.error(f"Error loading ML model: {e}")
            return None
            
    def create_features(self, df):
        """Create feature set for ML model"""
        if df is None:
            return None
            
        try:
            features = pd.DataFrame(index=df.index)
            
            # Normalize RSI (0-100 to 0-1)
            features['rsi'] = df['RSI'] / 100
            
            # Normalize MACD using recent range
            macd_range = max(abs(df['MACD'].max()), abs(df['MACD'].min()))
            features['macd'] = df['MACD'] / macd_range if macd_range > 0 else df['MACD']
            
            # Bollinger band position (already normalized 0-1)
            features['bb_position'] = df['BB_position']
            
            # Normalize ATR by price
            features['atr'] = df['ATR'] / df['close']
            
            # Normalize ADX (0-100 to 0-1)
            features['adx'] = df['ADX'] / 100
            
            # Volume change
            features['volume_change'] = df['volume_ratio'] / 5  # Normalize by dividing by typical max
            
            # Price momentum
            features['price_change'] = df['daily_return']
            
            # Moving average crossover indicator
            features['ma_cross'] = np.where(df['EMA_fast'] > df['EMA_medium'], 1, -1)
            
            # Add trend features
            features['uptrend'] = df['uptrend'].astype(int)
            features['downtrend'] = df['downtrend'].astype(int)
            
            # Add volatility
            features['volatility'] = df['volatility'] / df['volatility'].rolling(window=100).max()
            
            return features
            
        except Exception as e:
            logger.error(f"Error creating features: {e}")
            return None
            
    def create_labels(self, df, forward_period=24):
        """Create labels for ML model - predict price movement over next period"""
        try:
            # Calculate forward returns with proper handling of forward_period
            df_copy = df.copy()
            df_copy['forward_return'] = df_copy['close'].shift(-forward_period) / df_copy['close'] - 1
            
            # Create classification labels
            threshold = df_copy['ATR'].rolling(window=24).mean() / df_copy['close']
            labels = np.where(df_copy['forward_return'] > threshold, 1,  # Bullish
                     np.where(df_copy['forward_return'] < -threshold, -1,  # Bearish
                     0))  # Neutral
            
            # Trim labels to match original dataframe length
            labels = labels[:len(df)]
            
            return labels
            
        except Exception as e:
            logger.error(f"Error creating labels: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
            
    def train_ml_model(self, data_fetcher, indicator_calculator, symbols=['BTCUSDT', 'ETHUSDT'], interval='1h'):
        """Train a machine learning model on historical data with time-series cross-validation"""
        try:
            all_features = []
            all_labels = []
            
            # Update last retrain time
            self.strategy_params['last_retrain_time'] = datetime.now()
            
            for symbol in symbols:
                logger.info(f"Training with {symbol} data")
                
                # Get extended historical data for training
                df = data_fetcher.get_historical_data(symbol, interval, lookback='90 days')
                if df is None or len(df) < 100:
                    continue
                    
                # Calculate indicators
                df = indicator_calculator.calculate_indicators(df)
                if df is None:
                    continue
                
                # Create features and labels
                features = self.create_features(df)
                labels = self.create_labels(df)
                
                if features is None or labels is None:
                    continue
                
                # Combine features and labels into a single DataFrame
                combined_df = pd.DataFrame(features)
                combined_df['target'] = labels
                
                # Remove NaN values from both features and labels together
                combined_df_valid = combined_df.dropna()
                
                features_valid = combined_df_valid.drop('target', axis=1)
                labels_valid = combined_df_valid['target'].values
                
                if len(features_valid) != len(labels_valid):
                    logger.warning(f"Size mismatch for {symbol}: features={len(features_valid)}, labels={len(labels_valid)}")
                    continue
                
                all_features.append(features_valid)
                all_labels.extend(labels_valid)
            
            if not all_features:
                logger.warning("No valid training data found")
                return False
                
            # Combine all features
            X = pd.concat(all_features)
            y = np.array(all_labels)
            
            logger.info(f"Final training data: features shape={X.shape}, labels shape={y.shape}")
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Train model with multiple folds
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            accuracies = []
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                self.model.fit(X_train, y_train)
                y_pred = self.model.predict(X_test)
                accuracies.append(accuracy_score(y_test, y_pred))
            
            # Final fit on all data
            self.model.fit(X, y)
            
            # Log cross-validation results
            avg_accuracy = np.mean(accuracies)
            logger.info(f"Model cross-validation accuracy: {avg_accuracy:.4f}")
            
            # Save model
            with open(ENV_CONFIG['model_file'], 'wb') as f:
                pickle.dump(self.model, f)
            
            return True, avg_accuracy
            
        except Exception as e:
            logger.error(f"Error training ML model: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False, 0.0
            
    def predict(self, features):
        """Make prediction using trained model"""
        if self.model is None:
            logger.warning("No model loaded for prediction")
            return None
            
        try:
            if features is None or len(features) == 0:
                return None
                
            # Create a copy to avoid modifying the original dataframe
            features_clean = features.copy()
            
            # Check for NaN values
            if features_clean.isna().any().any():
                # First try to fill with column means
                column_means = features_clean.mean()
                
                # For any columns where mean is NaN (all values are NaN), use 0 instead
                column_means = column_means.fillna(0)
                
                # Apply the filling
                features_clean = features_clean.fillna(column_means)
                
                # Double-check and replace any remaining NaNs with 0
                if features_clean.isna().any().any():
                    features_clean = features_clean.fillna(0)
                    
                # Also replace infinite values with large finite values
                features_clean = features_clean.replace([np.inf, -np.inf], [1e10, -1e10])
                
                logger.info("Preprocessed NaN values in features before prediction")
                
            # Verify no NaNs remain before prediction
            if features_clean.isna().any().any():
                logger.error("Failed to clean all NaN values from features")
                return None
                
            prediction = self.model.predict(features_clean.iloc[[-1]])
            return prediction[0]
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None