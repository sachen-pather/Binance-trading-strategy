"""
Market state analysis and signal generation module.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from config import STRATEGY_PARAMS, RISK_PARAMS, CIRCUIT_BREAKER_CONFIG

logger = logging.getLogger("BinanceTrading.MarketAnalysis")


class MarketAnalyzer:
    """Market state analysis and signal generation"""
    
    def __init__(self, strategy_params=None, risk_params=None):
        self.strategy_params = strategy_params or STRATEGY_PARAMS
        self.risk_params = risk_params or RISK_PARAMS
        self.market_state = {
            'regime': 'UNKNOWN',
            'volatility': 0,
            'trend_strength': 0,
            'correlation_matrix': pd.DataFrame()
        }
        self.circuit_breaker = CIRCUIT_BREAKER_CONFIG.copy()
        
    def update_market_state(self, data_fetcher, indicator_calculator, supported_symbols):
        """Update market state analysis for adaptive strategy"""
        try:
            # Analyze major cryptocurrencies
            key_symbols = ['BTCUSDT', 'ETHUSDT']
            dfs = {}
            
            for symbol in key_symbols:
                df = data_fetcher.get_historical_data(symbol, interval='1h', lookback='14 days')
                if df is not None:
                    df = indicator_calculator.calculate_indicators(df)
                    if df is not None:
                        dfs[symbol] = df
            
            if not dfs:
                logger.warning("Could not update market state: no data available")
                return
            
            # Determine market regime
            btc_data = dfs.get('BTCUSDT')
            if btc_data is not None:
                # Check latest ADX and trend
                latest = btc_data.iloc[-1]
                adx = latest['ADX']
                plus_di = latest['plus_di'] 
                minus_di = latest['minus_di']
                bb_width = latest['BB_width']
                
                # Regime classification
                if adx > self.strategy_params['adx_threshold']:
                    if plus_di > minus_di:
                        self.market_state['regime'] = 'BULL_TREND'
                    else:
                        self.market_state['regime'] = 'BEAR_TREND'
                else:
                    if bb_width < btc_data['BB_width'].rolling(window=20).mean().iloc[-1]:
                        self.market_state['regime'] = 'RANGE_CONTRACTION'
                    else:
                        self.market_state['regime'] = 'RANGE_EXPANSION'
                
                # Market volatility
                self.market_state['volatility'] = latest['ATR'] / latest['close']
                self.market_state['trend_strength'] = latest['ADX'] / 100
            
            # Calculate correlation matrix
            if len(dfs) > 1:
                # Extract close prices
                prices = pd.DataFrame({
                    symbol: df['close'] for symbol, df in dfs.items()
                })
                
                # Calculate correlation matrix
                self.market_state['correlation_matrix'] = prices.pct_change().corr()
            
            # Update circuit breaker
            self.update_circuit_breaker(data_fetcher, supported_symbols)
            
            logger.info(f"Market state updated: {self.market_state['regime']}, "
                       f"Volatility: {self.market_state['volatility']:.4f}, "
                       f"Trend strength: {self.market_state['trend_strength']:.4f}")
            
        except Exception as e:
            logger.error(f"Error updating market state: {e}")
            
    def check_extreme_volatility(self, symbol, data_fetcher):
        """Check if a symbol is experiencing extreme volatility"""
        try:
            # Get recent data
            df = data_fetcher.get_historical_data(symbol, interval='1h', lookback='1 day')
            if df is None or len(df) < 2:
                return False
                
            # Calculate hourly price changes
            hourly_changes = abs(df['close'].pct_change())
            
            # Check if any hourly change exceeds threshold
            if hourly_changes.max() > self.risk_params['extreme_volatility_threshold']:
                logger.warning(f"Extreme volatility detected for {symbol}: {hourly_changes.max():.2%}")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error checking volatility: {e}")
            return False
            
    def update_circuit_breaker(self, data_fetcher, supported_symbols):
        """Update circuit breaker status"""
        try:
            if self.circuit_breaker.get('manually_disabled', False):
                return False
                
            # Check if circuit breaker is active
            if self.circuit_breaker['active']:
                # Check if cooldown period has passed
                cooldown_time = self.circuit_breaker['triggered_time'] + timedelta(
                    minutes=self.circuit_breaker['cooldown_minutes']
                )
                
                if datetime.now() > cooldown_time:
                    logger.info("Circuit breaker cooldown period ended")
                    self.circuit_breaker['active'] = False
                    self.circuit_breaker['triggered_symbols'] = set()
            
            # Check key symbols for extreme volatility
            for symbol_info in supported_symbols[:5]:  # Check top 5 by volume
                symbol = symbol_info['symbol']
                
                if self.check_extreme_volatility(symbol, data_fetcher):
                    # Trigger circuit breaker
                    if not self.circuit_breaker['active']:
                        self.circuit_breaker['active'] = True
                        self.circuit_breaker['triggered_time'] = datetime.now()
                    
                    self.circuit_breaker['triggered_symbols'].add(symbol)
                    logger.warning(f"Circuit breaker triggered for {symbol}")
                    
            return self.circuit_breaker['active']
            
        except Exception as e:
            logger.error(f"Error updating circuit breaker: {e}")
            return False
            
    def generate_trading_signals(self, df, ml_model=None, use_ml=True):
        """Generate trading signals based on multiple indicators and ML model"""
        if df is None or len(df) < 50:
            return None, 0
            
        try:
            latest = df.iloc[-1]
            
            # 1. Trend-following signals
            trend_signal = None
            ma_crossover = (df['EMA_fast'].iloc[-1] > df['EMA_medium'].iloc[-1] and 
                           df['EMA_fast'].iloc[-2] <= df['EMA_medium'].iloc[-2])
            ma_crossunder = (df['EMA_fast'].iloc[-1] < df['EMA_medium'].iloc[-1] and 
                            df['EMA_fast'].iloc[-2] >= df['EMA_medium'].iloc[-2])
            
            if ma_crossover:
                trend_signal = 'BUY'
            elif ma_crossunder:
                trend_signal = 'SELL'
            
            # 2. Mean-reversion signals
            reversion_signal = None
            rsi_oversold = latest['RSI'] < self.strategy_params['rsi_oversold']
            rsi_overbought = latest['RSI'] > self.strategy_params['rsi_overbought']
            bb_lower_touch = latest['close'] < latest['BB_lower']
            bb_upper_touch = latest['close'] > latest['BB_upper']
            
            if rsi_oversold and bb_lower_touch:
                reversion_signal = 'BUY'
            elif rsi_overbought and bb_upper_touch:
                reversion_signal = 'SELL'
            
            # 3. Volatility breakout signals
            volatility_signal = None
            atr = latest['ATR']
            price_change = abs(latest['close'] - df['close'].iloc[-2])
            
            if price_change > 1.5 * atr and latest['volume_ratio'] > self.strategy_params['volume_threshold']:
                volatility_signal = 'BUY' if latest['close'] > df['close'].iloc[-2] else 'SELL'
            
            # 4. ML model prediction if available
            ml_signal = None
            if use_ml and ml_model is not None and ml_model.model is not None:
                try:
                    features = ml_model.create_features(df)
                    if features is not None:
                        prediction = ml_model.predict(features)
                        if prediction == 1:
                            ml_signal = 'BUY'
                        elif prediction == -1:
                            ml_signal = 'SELL'
                except Exception as e:
                    logger.error(f"Error using ML model: {e}")
            
            # 5. Combine signals based on market regime
            signals = {
                'trend': trend_signal,
                'reversion': reversion_signal,
                'volatility': volatility_signal,
                'ml': ml_signal
            }
            
            # Weight signals differently based on market regime
            signal_weights = {
                'BULL_TREND': {'trend': 0.5, 'reversion': 0.1, 'volatility': 0.2, 'ml': 0.2},
                'BEAR_TREND': {'trend': 0.5, 'reversion': 0.1, 'volatility': 0.2, 'ml': 0.2},
                'RANGE_CONTRACTION': {'trend': 0.1, 'reversion': 0.5, 'volatility': 0.2, 'ml': 0.2},
                'RANGE_EXPANSION': {'trend': 0.2, 'reversion': 0.2, 'volatility': 0.4, 'ml': 0.2},
                'UNKNOWN': {'trend': 0.25, 'reversion': 0.25, 'volatility': 0.25, 'ml': 0.25}
            }
            
            # Get weights for current regime
            weights = signal_weights.get(self.market_state['regime'], signal_weights['UNKNOWN'])
            
            # Calculate signal scores
            buy_score = sum(
                weights[signal_type] 
                for signal_type, signal in signals.items() 
                if signal == 'BUY' and signal is not None
            )
            
            sell_score = sum(
                weights[signal_type] 
                for signal_type, signal in signals.items() 
                if signal == 'SELL' and signal is not None
            )
            
            # Signal strength for position sizing
            signal_strength = max(buy_score, sell_score)
            
            # Determine final signal
            threshold = 0.3  # Minimum confidence threshold
            
            if buy_score > threshold and buy_score > sell_score:
                logger.info(f"BUY signal generated with strength {buy_score:.2f}")
                return ('BUY', signal_strength)
            elif sell_score > threshold and sell_score > buy_score:
                logger.info(f"SELL signal generated with strength {sell_score:.2f}")
                return ('SELL', signal_strength)
            else:
                return (None, 0)
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return (None, 0)
            
    def find_best_trading_opportunities(self, data_fetcher, indicator_calculator, ml_model, 
                                        supported_symbols, current_positions, limit=5):
        """Find the best trading opportunities across multiple symbols with correlation filtering"""
        try:
            opportunities = []
            
            # Check if we're in a circuit breaker
            if self.circuit_breaker['active']:
                logger.warning("Circuit breaker active - no trading opportunities")
                return []
            
            # Get current positions symbols
            current_symbols = [pos['symbol'] for pos in current_positions.values()]
            
            # Check each tradable symbol
            for symbol_info in supported_symbols[:20]:  # Top 20 by volume
                symbol = symbol_info['symbol']
                
                # Skip if we already have a position in this symbol
                if symbol in current_symbols:
                    continue
                
                # Get data and indicators
                df = data_fetcher.get_historical_data(symbol, interval='1h', lookback='3 days')
                if df is None:
                    continue
                    
                df = indicator_calculator.calculate_indicators(df)
                if df is None:
                    continue
                
                # Generate signals
                signal, strength = self.generate_trading_signals(df, ml_model)
                
                if signal:
                    # Check correlation with existing positions
                    if current_symbols and self.market_state['correlation_matrix'] is not None:
                        # Check if this symbol is too correlated with existing positions
                        is_correlated = False
                        for existing_symbol in current_symbols:
                            # Skip if correlation data not available
                            if (symbol not in self.market_state['correlation_matrix'].index or 
                                existing_symbol not in self.market_state['correlation_matrix'].columns):
                                continue
                                
                            correlation = self.market_state['correlation_matrix'].loc[symbol, existing_symbol]
                            if abs(correlation) > self.risk_params['correlation_threshold']:
                                logger.info(f"Skipping {symbol} due to high correlation ({correlation:.2f}) with {existing_symbol}")
                                is_correlated = True
                                break
                                
                        if is_correlated:
                            continue
                    
                    # Calculate score based on multiple factors
                    score_factors = {
                        'signal_strength': strength,
                        'volume': df['volume_ratio'].iloc[-1],
                        'trend_alignment': 1 if (signal == 'BUY' and self.market_state['regime'] == 'BULL_TREND') or
                                           (signal == 'SELL' and self.market_state['regime'] == 'BEAR_TREND') else 0.5,
                        'volatility': df['ATR'].iloc[-1] / df['close'].iloc[-1]
                    }
                    
                    # Weighted score
                    score = (
                        score_factors['signal_strength'] * 0.4 +
                        score_factors['volume'] * 0.2 +
                        score_factors['trend_alignment'] * 0.3 +
                        score_factors['volatility'] * 0.1
                    )
                    
                    opportunities.append({
                        'symbol': symbol,
                        'signal': signal,
                        'strength': strength,
                        'score': score,
                        'price': df['close'].iloc[-1],
                        'volume_ratio': df['volume_ratio'].iloc[-1],
                        'atr': df['ATR'].iloc[-1]
                    })
            
            # Sort by score
            opportunities.sort(key=lambda x: x['score'], reverse=True)
            
            return opportunities[:limit]
            
        except Exception as e:
            logger.error(f"Error finding trading opportunities: {e}")
            return []