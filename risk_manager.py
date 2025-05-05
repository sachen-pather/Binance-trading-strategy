"""
Risk management and position sizing module.
"""

import logging
from config import RISK_PARAMS

logger = logging.getLogger("BinanceTrading.RiskManager")


class RiskManager:
    """Risk management and position sizing calculations"""
    
    def __init__(self, risk_params=None):
        self.risk_params = risk_params or RISK_PARAMS
        
    def calculate_position_size(self, symbol, signal_strength, current_equity, 
                              data_fetcher, indicator_calculator, market_state, positions):
        """Calculate dynamic position size based on multiple factors including transaction costs"""
        try:
            # Get symbol data for volatility
            df = data_fetcher.get_historical_data(symbol, interval='1h', lookback='3 days')
            if df is None:
                logger.warning(f"Using default position size for {symbol} (no data)")
                return self.risk_params['base_position_size'] * current_equity
            
            df = indicator_calculator.calculate_indicators(df)
            if df is None:
                logger.warning(f"Using default position size for {symbol} (calculation failed)")
                return self.risk_params['base_position_size'] * current_equity
            
            # 1. Kelly Criterion adjustment
            win_rate = self.risk_params['win_rate']
            win_loss_ratio = self.risk_params['win_loss_ratio']
            kelly_fraction = max(0, (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio)
            
            # Conservative Kelly (use fraction of Kelly)
            kelly_position = kelly_fraction * 0.5  # Half-Kelly for safety
            
            # 2. Volatility adjustment
            latest = df.iloc[-1]
            volatility = latest['ATR'] / latest['close']
            vol_adjustment = 1.0 / (volatility * self.risk_params['volatility_factor'])
            
            # 3. Market regime adjustment
            regime_factors = {
                'BULL_TREND': 1.2,
                'BEAR_TREND': 0.5,
                'RANGE_CONTRACTION': 0.8,
                'RANGE_EXPANSION': 0.7,
                'UNKNOWN': 0.5
            }
            regime_adjustment = regime_factors.get(market_state['regime'], 0.5)
            
            # 4. Current exposure adjustment
            open_positions_value = sum(pos['value'] for pos in positions.values())
            exposure_ratio = open_positions_value / current_equity
            exposure_adjustment = max(0, 1 - exposure_ratio)
            
            # 5. Account drawdown adjustment
            initial_equity = current_equity  # This should be provided as parameter
            drawdown = 1 - (current_equity / initial_equity)
            drawdown_adjustment = max(0.2, 1 - drawdown * 2)
            
            # 6. Signal strength adjustment
            signal_adjustment = min(1.5, max(0.2, signal_strength))
            
            # 7. Transaction cost adjustment
            tx_cost = self.risk_params['taker_fee'] + self.risk_params['slippage']
            cost_adjustment = 1 - (2 * tx_cost)  # Adjust for round-trip costs
            
            # Calculate final position size as percentage of equity
            position_pct = (
                self.risk_params['base_position_size'] * 
                kelly_position *
                vol_adjustment * 
                regime_adjustment * 
                exposure_adjustment * 
                drawdown_adjustment *
                signal_adjustment *
                cost_adjustment
            )
            
            # Apply min/max limits
            position_pct = min(self.risk_params['max_position_size'], 
                             max(self.risk_params['min_position_size'], position_pct))
            
            # Convert to USDT value
            position_size = position_pct * current_equity
            
            logger.info(f"Position size for {symbol}: {position_size:.2f} USDT "
                      f"({position_pct*100:.2f}% of equity)")
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return self.risk_params['min_position_size'] * current_equity
            
    def check_risk_limits(self, daily_pnl, current_equity, initial_equity, positions):
        """Check if risk limits are breached"""
        try:
            # Check daily loss limit
            if daily_pnl < -self.risk_params['max_daily_loss'] * initial_equity:
                logger.warning(f"Daily loss limit reached: {daily_pnl:.2f} USDT")
                return False, "Daily loss limit reached"
            
            # Check maximum open positions
            if len(positions) >= self.risk_params['max_open_positions']:
                logger.warning(f"Maximum open positions reached ({self.risk_params['max_open_positions']})")
                return False, "Maximum open positions reached"
            
            # Check drawdown
            drawdown = (initial_equity - current_equity) / initial_equity
            if drawdown > 0.2:  # 20% drawdown limit
                logger.warning(f"High drawdown: {drawdown*100:.2f}%")
                return False, "High drawdown limit reached"
                
            return True, "Risk checks passed"
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return False, "Error in risk checks"
            
    def calculate_stop_loss_take_profit(self, symbol, side, entry_price, market_state, df):
        """Calculate dynamic stop loss and take profit levels"""
        try:
            # For extremely small priced tokens, use percentage-based approach
            if entry_price < 0.001:  # Define a threshold for "micro-priced" tokens
                logger.info(f"Using percentage-based SL/TP for micro-priced token {symbol}")
                if side == 'BUY': 
                    stop_loss = entry_price * (1 - self.risk_params['stop_loss_pct'])
                    take_profit = entry_price * (1 + self.risk_params['take_profit_pct'])
                else:  # SELL
                    stop_loss = entry_price * (1 + self.risk_params['stop_loss_pct'])
                    take_profit = entry_price * (1 - self.risk_params['take_profit_pct'])
                return stop_loss, take_profit
            
            if df is None or len(df) < 1:
                # Fallback to percentage-based
                if side == 'BUY':
                    stop_loss = entry_price * (1 - self.risk_params['stop_loss_pct'])
                    take_profit = entry_price * (1 + self.risk_params['take_profit_pct'])
                else:  # SELL
                    stop_loss = entry_price * (1 + self.risk_params['stop_loss_pct'])
                    take_profit = entry_price * (1 - self.risk_params['take_profit_pct'])
                return stop_loss, take_profit
            
            atr = df['ATR'].iloc[-1]
            regime = market_state['regime']
            
            # Get stop multiplier based on market regime
            stop_multiplier = {
                'BULL_TREND': 2.0,
                'BEAR_TREND': 1.0,
                'RANGE_CONTRACTION': 1.0,
                'RANGE_EXPANSION': 1.2,
                'UNKNOWN': 1.5
            }.get(regime, 1.5)
            
            tp_multiplier = stop_multiplier * 1.2  # 1.2:1 reward-risk ratio
            
            if side == 'BUY':
                stop_loss = entry_price - (atr * stop_multiplier)
                take_profit = entry_price + (atr * tp_multiplier)
            else:  # SELL
                stop_loss = entry_price + (atr * stop_multiplier)
                take_profit = entry_price - (atr * tp_multiplier)
                
            return stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"Error calculating stop loss/take profit: {e}")
            # Fallback to percentage-based
            if side == 'BUY':
                return (entry_price * (1 - self.risk_params['stop_loss_pct']),
                        entry_price * (1 + self.risk_params['take_profit_pct']))
            else:
                return (entry_price * (1 + self.risk_params['stop_loss_pct']),
                        entry_price * (1 - self.risk_params['take_profit_pct']))