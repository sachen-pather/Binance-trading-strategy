"""
Main trading strategy coordination module.
"""

import json
import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from binance.client import Client
from config import (RISK_PARAMS, STRATEGY_PARAMS, CIRCUIT_BREAKER_CONFIG, 
                    PERFORMANCE_METRICS_TEMPLATE, ENV_CONFIG)
from data_fetcher import DataFetcher
from indicators import TechnicalIndicators
from ml_model import MLModel
from market_analysis import MarketAnalyzer
from risk_manager import RiskManager
from trade_executor import TradeExecutor
from position_manager import PositionManager
from performance_tracker import PerformanceTracker
from exit_analytics import ExitAnalytics

logger = logging.getLogger("BinanceTrading.Strategy")


class TradingStrategy:
    """Main trading strategy coordinator"""
    
    def __init__(self, test_mode=True):
        """Initialize the trading strategy with improved parameters"""
        load_dotenv()
        
        # Initialize Binance client
        if test_mode:
            self.client = Client(
                os.getenv('TESTNET_API_KEY'),
                os.getenv('TESTNET_SECRET_KEY')
            )
            self.client.API_URL = ENV_CONFIG['testnet_url']
            logger.info("Running in TEST mode")
        else:
            self.client = Client(
                os.getenv('BINANCE_API_KEY'),
                os.getenv('BINANCE_SECRET_KEY')
            )
            logger.info("Running in PRODUCTION mode")
        
        # Initialize components
        self.data_fetcher = DataFetcher(self.client)
        self.indicator_calculator = TechnicalIndicators(STRATEGY_PARAMS)
        self.ml_model = MLModel(STRATEGY_PARAMS)
        self.market_analyzer = MarketAnalyzer(STRATEGY_PARAMS, RISK_PARAMS)
        self.risk_manager = RiskManager(RISK_PARAMS)
        # Fixed: Initialize TradeExecutor with data_fetcher
        self.trade_executor = TradeExecutor(self.client, self.data_fetcher)
        self.position_manager = PositionManager()
        self.exit_analytics = ExitAnalytics()
        
        # Initialize equity - get with paper_trade=True to use default if API fails
        self.total_equity = self.data_fetcher.get_account_equity(paper_trade=True, default_equity=10000)
        self.initial_equity = self.total_equity
        self.performance_tracker = PerformanceTracker(self.initial_equity)
        
        # Load ML model
        self.ml_model.load_ml_model()
        
        # Get supported symbols - public API should work without credentials
        self.supported_symbols = self.data_fetcher.get_tradable_symbols()
        
        # Update market state before loading previous state
        try:
            self.market_analyzer.update_market_state(
                self.data_fetcher, 
                self.indicator_calculator, 
                self.supported_symbols
            )
        except Exception as e:
            logger.warning(f"Error updating market state: {e}")
        
        # Load previous state if available
        self.load_strategy_state()
        
        logger.info(f"Strategy initialized with equity: {self.total_equity} USDT")
        
    def run_strategy(self, automatic=False, paper_trade=True):
        """Run the complete trading strategy"""
        try:
            # Update market state
            self.market_analyzer.update_market_state(
                self.data_fetcher, 
                self.indicator_calculator, 
                self.supported_symbols
            )
            
            # Update open positions with paper_trade flag
            closed_trades, updated_positions = self.position_manager.update_positions(
                self.client, 
                RISK_PARAMS,
                paper_trade=paper_trade,
                trade_executor=self.trade_executor
            )
            
            # If paper trading, also update paper positions with proper dependencies
            if paper_trade:
                paper_closed_trades, paper_updated_positions = self.position_manager.update_paper_positions(
                    RISK_PARAMS,
                    trade_executor=self.trade_executor,
                )
                
            # Add paper closed trades to updated positions
            closed_trades += paper_closed_trades
            updated_positions.extend(paper_updated_positions)
            
            # Add closed trades to performance tracker and record exits
            for trade in updated_positions:
                self.performance_tracker.add_trade(trade)
                self.exit_analytics.record_exit(trade)  
                
            # Find best trading opportunities
            opportunities = self.market_analyzer.find_best_trading_opportunities(
                self.data_fetcher,
                self.indicator_calculator,
                self.ml_model,
                self.supported_symbols,
                self.position_manager.get_all_positions(paper_trade=paper_trade),
                limit=3
            )
            
            if not opportunities:
                logger.info("No trading opportunities found")
                return None
            
            # Execute trades if automatic mode is on
            executed_trades = []
            
            if automatic:
                for opp in opportunities:
                    # Check risk limits
                    risk_check, message = self.risk_manager.check_risk_limits(
                        self.performance_tracker.daily_pnl,
                        self.total_equity,
                        self.initial_equity,
                        self.position_manager.get_all_positions(paper_trade=paper_trade)
                    )
                    
                    if not risk_check:
                        logger.info(f"Risk check failed: {message}")
                        break
                    
                    # Calculate position size
                    position_size = self.risk_manager.calculate_position_size(
                        opp['symbol'],
                        opp['strength'],
                        self.total_equity,
                        self.data_fetcher,
                        self.indicator_calculator,
                        self.market_analyzer.market_state,
                        self.position_manager.get_all_positions(paper_trade=paper_trade)
                    )
                    
                    # Execute trade with paper_trade flag
                    trade = self.trade_executor.execute_trade(
                        symbol=opp['symbol'],
                        side=opp['signal'],
                        quantity=position_size / opp['price'],
                        risk_manager=self.risk_manager,
                        data_fetcher=self.data_fetcher,
                        indicator_calculator=self.indicator_calculator,
                        market_state=self.market_analyzer.market_state,
                        paper_trade=paper_trade
                    )
                    
                    if trade:
                        # Add to position manager with paper_trade flag
                        trade_id = self.position_manager.add_position(trade, paper_trade=paper_trade)
                        executed_trades.append(trade)
                        logger.info(f"Trade executed: {opp['symbol']} {opp['signal']}")
            
            # Save strategy state
            self.save_strategy_state()
            
            # Update equity with paper_trade flag
            self.total_equity = self.data_fetcher.get_account_equity(paper_trade=paper_trade, default_equity=self.total_equity)
            
            return {
                'opportunities': opportunities,
                'executed_trades': executed_trades,
                'market_regime': self.market_analyzer.market_state['regime'],
                'open_positions': len(self.position_manager.get_all_positions(paper_trade=paper_trade)),
                'equity': self.total_equity
            }
            
        except Exception as e:
            logger.error(f"Error in strategy execution: {e}")
            return None
            
    def save_strategy_state(self):
        """Save strategy state to file for recovery"""
        try:
            state = {
                'timestamp': datetime.now().isoformat(),
                'risk_params': RISK_PARAMS,
                'strategy_params': {
                    **STRATEGY_PARAMS,
                    'last_retrain_time': STRATEGY_PARAMS['last_retrain_time'].isoformat() 
                        if isinstance(STRATEGY_PARAMS['last_retrain_time'], datetime) 
                        else STRATEGY_PARAMS['last_retrain_time']
                    
                },
                'exit_analytics': self.exit_analytics.save_to_dict(),
                'performance_metrics': self.performance_tracker.get_performance_summary(),
                'positions': self.position_manager.get_all_positions(paper_trade=True),  # Save paper positions
                'paper_positions': self.position_manager.get_all_positions(paper_trade=True),  # Save paper positions separately
                'total_equity': self.total_equity,
                'initial_equity': self.initial_equity,
                'daily_pnl': self.performance_tracker.daily_pnl,
                'market_state': {
                    'regime': self.market_analyzer.market_state['regime'],
                    'volatility': self.market_analyzer.market_state['volatility'],
                    'trend_strength': self.market_analyzer.market_state['trend_strength']
                },
                'circuit_breaker': {
                    'active': self.market_analyzer.circuit_breaker['active'],
                    'triggered_time': self.market_analyzer.circuit_breaker['triggered_time'].isoformat() 
                        if self.market_analyzer.circuit_breaker['triggered_time'] else None,
                    'triggered_symbols': list(self.market_analyzer.circuit_breaker['triggered_symbols'])
                },
                # FIX: Remove direct access to trades, performance tracker already handles them
                # The performance_metrics already include trades, no need to save separately
            }
            
            with open(ENV_CONFIG['state_file'], 'w') as f:
                json.dump(state, f, default=str, indent=2)
            
            logger.info("Strategy state saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving strategy state: {e}")
            
    def load_strategy_state(self):
        """Load strategy state from file"""
        try:
            if os.path.exists(ENV_CONFIG['state_file']):
                with open(ENV_CONFIG['state_file'], 'r') as f:
                    state = json.load(f)
                
                # Load strategy parameters
                STRATEGY_PARAMS.update(state.get('strategy_params', {}))
                RISK_PARAMS.update(state.get('risk_params', {}))
                
                # Load positions - use paper positions if available, otherwise regular positions
                paper_positions = state.get('paper_positions', {})
                if paper_positions:
                    for trade_id, position in paper_positions.items():
                        self.position_manager.add_position(position, paper_trade=True)
                else:
                    # Backward compatibility with older state files
                    positions = state.get('positions', {})
                    for trade_id, position in positions.items():
                        self.position_manager.add_position(position, paper_trade=True)
                
                # Load equity values
                self.total_equity = state.get('total_equity', self.total_equity)
                self.initial_equity = state.get('initial_equity', self.initial_equity)
                
                # Load performance tracker
                self.performance_tracker.daily_pnl = state.get('daily_pnl', 0)
                
                # Load performance metrics (includes trades)
                performance_metrics = state.get('performance_metrics', {})
                if performance_metrics:
                    # Update performance tracker metrics
                    self.performance_tracker.performance_metrics.update(performance_metrics)
                
                # Load exit analytics
                exit_analytics_data = state.get('exit_analytics', {})
                if exit_analytics_data:
                    self.exit_analytics.load_from_dict(exit_analytics_data)
                    
                # Load market state
                market_state = state.get('market_state', {})
                self.market_analyzer.market_state.update(market_state)
                
                # Load circuit breaker
                circuit_breaker = state.get('circuit_breaker', {})
                if circuit_breaker:
                    self.market_analyzer.circuit_breaker['active'] = circuit_breaker.get('active', False)
                    if circuit_breaker.get('triggered_time'):
                        self.market_analyzer.circuit_breaker['triggered_time'] = datetime.fromisoformat(circuit_breaker['triggered_time'])
                    self.market_analyzer.circuit_breaker['triggered_symbols'] = set(circuit_breaker.get('triggered_symbols', []))
                
                logger.info("Strategy state loaded successfully")
                return True
            else:
                logger.info("No strategy state file found")
                return False
                
        except Exception as e:
            logger.error(f"Error loading strategy state: {e}")
            return False
            
    def train_ml_model(self):
        """Train the ML model with historical data"""
        return self.ml_model.train_ml_model(
            self.data_fetcher,
            self.indicator_calculator,
            symbols=['BTCUSDT', 'ETHUSDT'],
            interval='1h'
        )