"""
Configuration module for trading strategy parameters and constants.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# BINANCE API CONFIGURATION - Load from environment variables
BINANCE_CONFIG = {
    'api_key': os.getenv('BINANCE_API_KEY'),
    'api_secret': os.getenv('BINANCE_SECRET_KEY'),
    'testnet': False,  # Set to False for live trading
    'base_url': 'https://api.binance.com'
}

# Validate that API keys are loaded
if not BINANCE_CONFIG['api_key'] or not BINANCE_CONFIG['api_secret']:
    raise ValueError("API keys not found! Make sure your .env file contains BINANCE_API_KEY and BINANCE_SECRET_KEY")

# Risk management parameters - EXTREMELY AGGRESSIVE BUYING
RISK_PARAMS = {
    'base_position_size': 0.05,      # Increase to 5% for larger orders
    'max_position_size': 0.2,        # 20% max
    'min_position_size': 0.01,       # 1% minimum 
    'min_order_value_usdt': 5,      # Minimum 10 USDT per order
    'force_minimum_orders': True,    # Always meet minimum requirements
    'stop_loss_pct': 0.03,      # Very wide stop loss to hold positions
    'take_profit_pct': 0.05,    # Wider take profit for bigger gains
    'max_daily_loss': 0.1,        # 80% maximum daily loss (almost disabled)
    'max_open_positions': 20,     # Allow many concurrent buy positions
    'win_rate': 0.4,
    'win_loss_ratio': 1.5,        
    'volatility_factor': 0.001,   # Extremely low for any conditions
    'correlation_threshold': 0.99,# Allow virtually all correlated trades
    'maker_fee': 0.001,
    'taker_fee': 0.001,
    'slippage': 0.001,
    'extreme_volatility_threshold': 0.99  # Effectively disabled
}

# Strategy parameters - EXTREMELY AGGRESSIVE BUYING
STRATEGY_PARAMS = {
    'lookback_periods': {
        'short': '3 days',
        'medium': '14 days',
        'long': '30 days'
    },
    'ma_periods': {
        'fast': 5,    # Short periods
        'medium': 10, 
        'slow': 20
    },
    'rsi_period': 7,                # Shorter RSI period
    'rsi_overbought': 90,           # Extremely high to allow buying in overbought zones
    'rsi_oversold': 65,             # Very high to trigger more buys
    'bb_period': 10,                # Shorter period
    'bb_std_dev': 3.0,              # Very wide bands to catch price at upper band as buy signal
    'volume_threshold': 0.1,        # Almost any volume qualifies
    'atr_period': 7,                # Shorter ATR period
    'adx_period': 7,                # Shorter ADX period
    'adx_threshold': 5,             # Extremely low - any trend qualifies
    'ml_features': [
        'rsi', 'macd', 'bb_position', 'atr', 'adx', 
        'volume_change', 'price_change', 'ma_cross'
    ],
    'ml_retrain_hours': 6,
    'last_retrain_time': None,
    'trailing_stop_factor': {
        'BULL_TREND': 0.2,          # Small trailing stops to maximize gains
        'BEAR_TREND': 0.3,
        'RANGE_CONTRACTION': 0.3,
        'RANGE_EXPANSION': 0.3,
        'UNKNOWN': 0.3
    }
}

# Circuit breaker configuration - COMPLETELY DISABLED
CIRCUIT_BREAKER_CONFIG = {
    'active': False,
    'triggered_time': None,
    'cooldown_minutes': 1,
    'triggered_symbols': set()
}

# Performance metrics configuration
PERFORMANCE_METRICS_TEMPLATE = {
    'trades': [],
    'daily_pnl': {},
    'win_rate': 0,
    'profit_factor': 0,
    'max_drawdown': 0,
    'sharpe_ratio': 0
}

# Environment configuration
ENV_CONFIG = {
    'testnet_url': 'https://testnet.binance.vision/api',
    'log_file': 'trading.log',
    'state_file': 'strategy_state.json',
    'model_file': 'trading_model.pkl'
}