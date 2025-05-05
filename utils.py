"""
Utility functions and helpers.
"""

import logging
import os
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import json

logger = logging.getLogger("BinanceTrading.Utils")


def setup_logging(log_file='trading.log', log_level=logging.INFO):
    """Configure logging for the trading system"""
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    

def format_datetime(dt):
    """Format datetime object for logging and display"""
    if dt is None:
        return "N/A"
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt)
        except:
            return dt
    return dt.strftime('%Y-%m-%d %H:%M:%S')
    
def convert_interval_to_timedelta(interval):
    """Convert Binance interval string to timedelta"""
    interval_map = {
        '1m': timedelta(minutes=1),
        '3m': timedelta(minutes=3),
        '5m': timedelta(minutes=5),
        '15m': timedelta(minutes=15),
        '30m': timedelta(minutes=30),
        '1h': timedelta(hours=1),
        '2h': timedelta(hours=2),
        '4h': timedelta(hours=4),
        '6h': timedelta(hours=6),
        '8h': timedelta(hours=8),
        '12h': timedelta(hours=12),
        '1d': timedelta(days=1),
        '3d': timedelta(days=3),
        '1w': timedelta(weeks=1),
        '1M': timedelta(days=30)  # Approximate
    }
    return interval_map.get(interval, timedelta(hours=1))
    
def format_number(number, decimals=2):
    """Format number for display"""
    if number is None:
        return "N/A"
    try:
        return f"{float(number):.{decimals}f}"
    except:
        return str(number)
        
def safe_divide(a, b, default=0):
    """Safely divide two numbers"""
    try:
        if b == 0:
            return default
        return a / b
    except:
        return default
        
class StrategyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for strategy objects"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        return super().default(obj)
        
def calculate_compound_return(trades, initial_equity=10000):
    """Calculate compound return from trade history"""
    try:
        if not trades:
            return 0.0
            
        current_equity = initial_equity
        for trade in trades:
            current_equity += trade.get('net_pnl', 0)
            
        compound_return = (current_equity - initial_equity) / initial_equity
        return compound_return
        
    except Exception as e:
        logger.error(f"Error calculating compound return: {e}")
        return 0.0
        
def convert_symbol_to_base_quote(symbol):
    """Convert symbol string to base and quote assets"""
    try:
        # Most common quote assets
        quote_assets = ['USDT', 'BTC', 'ETH', 'BNB', 'BUSD', 'USD']
        
        for quote in quote_assets:
            if symbol.endswith(quote):
                base = symbol[:-len(quote)]
                return base, quote
                
        # If no match found, assume last 3 characters are quote
        return symbol[:-3], symbol[-3:]
        
    except Exception as e:
        logger.error(f"Error converting symbol: {e}")
        return symbol, 'USDT'
        
def calculate_win_statistics(trades):
    """Calculate win/loss statistics from trade history"""
    try:
        if not trades:
            return {
                'win_rate': 0,
                'profit_factor': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'max_win': 0,
                'max_loss': 0,
                'total_trades': 0
            }
            
        winning_trades = [t for t in trades if t.get('net_pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('net_pnl', 0) < 0]
        
        win_rate = len(winning_trades) / len(trades)
        total_profit = sum(t.get('net_pnl', 0) for t in winning_trades)
        total_loss = abs(sum(t.get('net_pnl', 0) for t in losing_trades))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        avg_win = total_profit / len(winning_trades) if winning_trades else 0
        avg_loss = total_loss / len(losing_trades) if losing_trades else 0
        max_win = max([t.get('net_pnl', 0) for t in winning_trades], default=0)
        max_loss = min([t.get('net_pnl', 0) for t in losing_trades], default=0)
        
        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_win': max_win,
            'max_loss': max_loss,
            'total_trades': len(trades)
        }
        
    except Exception as e:
        logger.error(f"Error calculating win statistics: {e}")
        return {}
        
def plot_equity_curve(trades, initial_equity=10000, save_path='equity_curve.png'):
    """Plot equity curve from trade history"""
    try:
        equity = [initial_equity]
        dates = [trades[0]['entry_time'] if trades else datetime.now()]
        
        for trade in trades:
            equity.append(equity[-1] + trade.get('net_pnl', 0))
            dates.append(trade.get('exit_time', dates[-1]))
            
        plt.figure(figsize=(12, 6))
        plt.plot(dates, equity, label='Equity')
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity (USDT)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Equity curve saved to {save_path}")
        else:
            plt.show()
            
        plt.close()
        
    except Exception as e:
        logger.error(f"Error plotting equity curve: {e}")
        
def validate_order_params(symbol, side, quantity, price=None):
    """Validate order parameters"""
    try:
        # Check required parameters
        if not symbol or not side or not quantity:
            return False, "Missing required parameters"
            
        # Check side
        if side not in ['BUY', 'SELL']:
            return False, "Invalid side parameter"
            
        # Check quantity
        if quantity <= 0:
            return False, "Quantity must be positive"
            
        # Check price if provided
        if price is not None and price <= 0:
            return False, "Price must be positive"
            
        return True, "Parameters valid"
        
    except Exception as e:
        logger.error(f"Error validating order parameters: {e}")
        return False, str(e)
        
def calculate_expectancy(trades):
    """Calculate expectancy from trade results"""
    try:
        if not trades:
            return 0.0
            
        winning_trades = [t for t in trades if t.get('net_pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('net_pnl', 0) < 0]
        
        win_rate = len(winning_trades) / len(trades)
        avg_win = sum(t.get('net_pnl', 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = abs(sum(t.get('net_pnl', 0) for t in losing_trades) / len(losing_trades)) if losing_trades else 0
        
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        return expectancy
    except Exception as e:
        logger.error(f"Error calculating expectancy: {e}")
        return 0.0