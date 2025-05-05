"""
Performance metrics and tracking module.
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from config import PERFORMANCE_METRICS_TEMPLATE, ENV_CONFIG

logger = logging.getLogger("BinanceTrading.PerformanceTracker")


class PerformanceTracker:
    """Track and analyze trading performance"""
    
    def __init__(self, initial_equity):
        self.performance_metrics = PERFORMANCE_METRICS_TEMPLATE.copy()
        self.initial_equity = initial_equity
        self.daily_pnl = 0
        
    def update_performance_metrics(self, trades, risk_params):
        """Update performance metrics for strategy optimization"""
        try:
            if not trades:
                return
            
            # Calculate win rate
            winning_trades = sum(1 for trade in trades if trade.get('net_pnl', 0) > 0)
            self.performance_metrics['win_rate'] = winning_trades / len(trades)
            
            # Calculate profit factor
            gross_profit = sum(trade.get('net_pnl', 0) for trade in trades if trade.get('net_pnl', 0) > 0)
            gross_loss = abs(sum(trade.get('net_pnl', 0) for trade in trades if trade.get('net_pnl', 0) < 0))
            self.performance_metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Calculate daily P&L
            daily_pnl = {}
            for trade in trades:
                date = trade.get('exit_time', datetime.now()).date()
                if date in daily_pnl:
                    daily_pnl[date] += trade.get('net_pnl', 0)
                else:
                    daily_pnl[date] = trade.get('net_pnl', 0)
            
            self.performance_metrics['daily_pnl'] = daily_pnl
            
            # Calculate drawdown
            equity_series = []
            equity = self.initial_equity
            dates = sorted(daily_pnl.keys())
            
            for date in dates:
                equity += daily_pnl[date]
                equity_series.append(equity)
            
            if equity_series:
                max_equity = equity_series[0]
                max_drawdown = 0
                
                for eq in equity_series:
                    max_equity = max(max_equity, eq)
                    drawdown = (max_equity - eq) / max_equity
                    max_drawdown = max(max_drawdown, drawdown)
                
                self.performance_metrics['max_drawdown'] = max_drawdown
            
            # Calculate Sharpe ratio
            if len(daily_pnl) > 1:
                returns = [pnl / self.initial_equity for pnl in daily_pnl.values()]
                avg_return = sum(returns) / len(returns)
                std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
                self.performance_metrics['sharpe_ratio'] = avg_return / std_return * (252 ** 0.5) if std_return > 0 else 0
            
            # Update risk parameters based on performance
            if len(trades) > 10:
                # Adjust win rate for Kelly calculation
                risk_params['win_rate'] = self.performance_metrics['win_rate']
                
                # Calculate average win/loss ratio
                avg_win = sum(trade.get('net_pnl', 0) for trade in trades if trade.get('net_pnl', 0) > 0) / winning_trades if winning_trades > 0 else 0
                avg_loss = abs(sum(trade.get('net_pnl', 0) for trade in trades if trade.get('net_pnl', 0) < 0)) / (len(trades) - winning_trades) if len(trades) > winning_trades else 1
                if avg_loss > 0:
                    risk_params['win_loss_ratio'] = avg_win / avg_loss
            
            logger.info(f"Performance metrics updated: Win rate={self.performance_metrics['win_rate']:.2f}, "
                       f"Profit factor={self.performance_metrics['profit_factor']:.2f}")
            
            return self.performance_metrics
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
            return None
            
    def add_trade(self, trade_info):
        """Add a completed trade to performance metrics"""
        try:
            self.performance_metrics['trades'].append(trade_info)
            self.daily_pnl += trade_info.get('net_pnl', 0)
            
            # Update performance if we have enough trades
            if len(self.performance_metrics['trades']) > 10:
                self.update_performance_metrics(self.performance_metrics['trades'], {})
                
            return True
        except Exception as e:
            logger.error(f"Error adding trade: {e}")
            return False
            
    def reset_daily_metrics(self):
        """Reset daily metrics (call at the start of each trading day)"""
        self.daily_pnl = 0
        logger.info("Daily metrics reset")
        
    def get_performance_summary(self):
        """Get summary of performance metrics"""
        return {
            'total_trades': len(self.performance_metrics['trades']),
            'win_rate': self.performance_metrics['win_rate'],
            'profit_factor': self.performance_metrics['profit_factor'],
            'max_drawdown': self.performance_metrics['max_drawdown'],
            'sharpe_ratio': self.performance_metrics['sharpe_ratio'],
            'daily_pnl': self.daily_pnl,
            'total_pnl': sum(trade.get('net_pnl', 0) for trade in self.performance_metrics['trades'])
        }
        
    def save_performance_state(self):
        """Save performance state to file"""
        try:
            state = {
                'performance_metrics': self.performance_metrics,
                'initial_equity': self.initial_equity,
                'daily_pnl': self.daily_pnl,
                'timestamp': datetime.now().isoformat()
            }
            
            with open('performance_state.json', 'w') as f:
                json.dump(state, f, default=str, indent=2)
            
            logger.info("Performance state saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving performance state: {e}")
            return False
            
    def load_performance_state(self):
        """Load performance state from file"""
        try:
            with open('performance_state.json', 'r') as f:
                state = json.load(f)
            
            self.performance_metrics = state['performance_metrics']
            self.initial_equity = state['initial_equity']
            self.daily_pnl = state['daily_pnl']
            
            logger.info("Performance state loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading performance state: {e}")
            return False