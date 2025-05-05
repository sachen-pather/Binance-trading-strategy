"""
Technical indicators calculation module.
"""

import pandas as pd
import numpy as np
import logging
from config import STRATEGY_PARAMS

logger = logging.getLogger("BinanceTrading.Indicators")


class TechnicalIndicators:
    """Calculate technical indicators for trading decisions"""
    
    def __init__(self, strategy_params=None):
        self.strategy_params = strategy_params or STRATEGY_PARAMS
        
    def calculate_indicators(self, df):
        """Calculate comprehensive technical indicators"""
        if df is None or len(df) < 50:
            logger.warning("Not enough data for indicator calculation")
            return None
            
        try:
            # Price-based indicators
            # Moving averages
            fast = self.strategy_params['ma_periods']['fast']
            medium = self.strategy_params['ma_periods']['medium']
            slow = self.strategy_params['ma_periods']['slow']
            
            df['SMA_fast'] = df['close'].rolling(window=fast).mean()
            df['SMA_medium'] = df['close'].rolling(window=medium).mean()
            df['SMA_slow'] = df['close'].rolling(window=slow).mean()
            df['EMA_fast'] = df['close'].ewm(span=fast).mean()
            df['EMA_medium'] = df['close'].ewm(span=medium).mean()
            df['EMA_slow'] = df['close'].ewm(span=slow).mean()
            
            # MACD
            df['MACD'] = df['EMA_fast'] - df['EMA_medium']
            df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_hist'] = df['MACD'] - df['MACD_signal']
            
            # Bollinger Bands
            bb_period = self.strategy_params['bb_period']
            bb_std = self.strategy_params['bb_std_dev']
            df['BB_middle'] = df['close'].rolling(window=bb_period).mean()
            df['BB_std'] = df['close'].rolling(window=bb_period).std()
            df['BB_upper'] = df['BB_middle'] + (bb_std * df['BB_std'])
            df['BB_lower'] = df['BB_middle'] - (bb_std * df['BB_std'])
            df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
            df['BB_position'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
            
            # RSI
            rsi_period = self.strategy_params['rsi_period']
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=rsi_period).mean()
            avg_loss = loss.rolling(window=rsi_period).mean()
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # ATR for volatility measurement
            atr_period = self.strategy_params['atr_period']
            df['tr1'] = abs(df['high'] - df['low'])
            df['tr2'] = abs(df['high'] - df['close'].shift())
            df['tr3'] = abs(df['low'] - df['close'].shift())
            df['TR'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            df['ATR'] = df['TR'].rolling(window=atr_period).mean()
            
            # ADX for trend strength
            adx_period = self.strategy_params['adx_period']
            df['plus_dm'] = np.where((df['high'] - df['high'].shift()) > (df['low'].shift() - df['low']),
                                    np.maximum(df['high'] - df['high'].shift(), 0), 0)
            df['minus_dm'] = np.where((df['low'].shift() - df['low']) > (df['high'] - df['high'].shift()),
                                     np.maximum(df['low'].shift() - df['low'], 0), 0)
            df['plus_di'] = 100 * (df['plus_dm'].rolling(window=adx_period).mean() / df['ATR'])
            df['minus_di'] = 100 * (df['minus_dm'].rolling(window=adx_period).mean() / df['ATR'])
            df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
            df['ADX'] = df['dx'].rolling(window=adx_period).mean()
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Price change features
            df['daily_return'] = df['close'].pct_change()
            df['weekly_return'] = df['close'].pct_change(5)
            df['monthly_return'] = df['close'].pct_change(20)
            
            # Volatility
            df['volatility'] = df['daily_return'].rolling(window=20).std()
            
            # Identify trends
            df['uptrend'] = ((df['EMA_fast'] > df['EMA_slow']) & (df['ADX'] > self.strategy_params['adx_threshold']))
            df['downtrend'] = ((df['EMA_fast'] < df['EMA_slow']) & (df['ADX'] > self.strategy_params['adx_threshold']))
            
            # Drop NaN values
            df.dropna(inplace=True)
            
            return df
        
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return None