"""
Position tracking and management module.
"""

import logging
from datetime import datetime
from config import RISK_PARAMS
from datetime import datetime, timedelta

logger = logging.getLogger("BinanceTrading.PositionManager")


class PositionManager:
    """Track and manage open positions"""
    
    def __init__(self):
        self.positions = {}
        self.closed_positions = []
        # Paper trading storage
        self.paper_positions = {}
        self.paper_closed_positions = []
        self.paper_orders = []
        
    def add_position(self, trade_info, paper_trade=False):
        """Add a new position"""
        try:
            trade_id = trade_info.get('trade_id', f"{trade_info['symbol']}_{trade_info['side']}_{int(datetime.now().timestamp())}")
            
            if paper_trade:
                self.paper_positions[trade_id] = trade_info
                logger.info(f"Paper trading: Position added: {trade_id}")
            else:
                self.positions[trade_id] = trade_info
                logger.info(f"Position added: {trade_id}")
                
            return trade_id
        except Exception as e:
            logger.error(f"Error adding position: {e}")
            return None
            
    def remove_position(self, trade_id, paper_trade=False):
        """Remove a position and store in closed positions"""
        try:
            if paper_trade:
                if trade_id in self.paper_positions:
                    position = self.paper_positions.pop(trade_id)
                    self.paper_closed_positions.append(position)
                    logger.info(f"Paper trading: Position removed: {trade_id}")
                    return position
                else:
                    logger.warning(f"Paper trading: Position not found: {trade_id}")
                    return None
            else:
                if trade_id in self.positions:
                    position = self.positions.pop(trade_id)
                    self.closed_positions.append(position)
                    logger.info(f"Position removed: {trade_id}")
                    return position
                else:
                    logger.warning(f"Position not found: {trade_id}")
                    return None
        except Exception as e:
            logger.error(f"Error removing position: {e}")
            return None
            
    def update_positions(self, client, risk_params=None, paper_trade=False, trade_executor=None):
        """Update open positions and check for completed trades"""
        try:
            risk_params = risk_params or RISK_PARAMS
            
            if paper_trade:
                # Handle paper trading position updates
                # Simplified version without actual API calls
                return self.update_paper_positions(risk_params,trade_executor)
            
            # Get open orders
            try:
                open_orders = client.get_open_orders()
                open_order_ids = {order['orderId'] for order in open_orders}
            except Exception as e:
                logger.warning(f"Failed to get open orders: {e}")
                return 0, []
            
            # Check each position
            closed_trade_ids = []
            updated_positions = []
            
            for trade_id, position in self.positions.items():
                symbol = position['symbol']
                
                # Check if stop loss or take profit orders have been filled
                sl_filled = False
                tp_filled = False
                
                if 'stop_loss' in position['orders']:
                    sl_order_id = position['orders']['stop_loss']['orderId']
                    if sl_order_id not in open_order_ids:
                        # Order not open, check if it was filled
                        try:
                            order = client.get_order(
                                symbol=symbol,
                                orderId=sl_order_id
                            )
                            if order['status'] == 'FILLED':
                                sl_filled = True
                                logger.info(f"Stop loss triggered for {trade_id}")
                        except:
                            # Order might have been canceled
                            pass
                
                if 'take_profit' in position['orders']:
                    tp_order_id = position['orders']['take_profit']['orderId']
                    if tp_order_id not in open_order_ids:
                        try:
                            order = client.get_order(
                                symbol=symbol,
                                orderId=tp_order_id
                            )
                            if order['status'] == 'FILLED':
                                tp_filled = True
                                logger.info(f"Take profit triggered for {trade_id}")
                        except:
                            pass
                
                # If either stop loss or take profit was filled
                if sl_filled or tp_filled:
                    # Calculate profit/loss
                    exit_price = position['stop_loss'] if sl_filled else position['take_profit']
                    entry_price = position['entry_price']
                    quantity = position['quantity']
                    
                    # Calculate fees
                    entry_fee = entry_price * quantity * risk_params['taker_fee']
                    exit_fee = exit_price * quantity * risk_params['taker_fee']
                    total_fees = entry_fee + exit_fee
                    
                    if position['side'] == 'BUY':
                        gross_pnl = (exit_price - entry_price) * quantity
                    else:  # SELL
                        gross_pnl = (entry_price - exit_price) * quantity
                    
                    # Calculate net P&L after fees
                    net_pnl = gross_pnl - total_fees
                    
                    # Log trade result
                    logger.info(f"Position closed: {trade_id}")
                    logger.info(f"Gross P&L: {gross_pnl:.2f} USDT")
                    logger.info(f"Fees: {total_fees:.2f} USDT")
                    logger.info(f"Net P&L: {net_pnl:.2f} USDT")
                    
                    # Update position record
                    position.update({
                        'exit_time': datetime.now(),
                        'exit_price': exit_price,
                        'gross_pnl': gross_pnl,
                        'fees': total_fees,
                        'net_pnl': net_pnl,
                        'exit_reason': 'stop_loss' if sl_filled else 'take_profit'
                    })
                    
                    updated_positions.append(position)
                    
                    # Cancel the other order
                    try:
                        if sl_filled and 'take_profit' in position['orders']:
                            client.cancel_order(
                                symbol=symbol,
                                orderId=position['orders']['take_profit']['orderId']
                            )
                        elif tp_filled and 'stop_loss' in position['orders']:
                            client.cancel_order(
                                symbol=symbol,
                                orderId=position['orders']['stop_loss']['orderId']
                            )
                    except:
                        pass
                    
                    # Mark position for removal
                    closed_trade_ids.append(trade_id)
            
            # Remove closed positions
            for trade_id in closed_trade_ids:
                self.remove_position(trade_id)
            
            return len(closed_trade_ids), updated_positions
            
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
            return 0, []
    
    def update_paper_positions(self, risk_params=None, trade_executor=None):
        """Update paper trading positions and check for exit conditions"""
        try:
            risk_params = risk_params or RISK_PARAMS
            
            # We need the trade_executor to get current prices
            if trade_executor is None:
                logger.warning("Trade executor not provided, cannot check exit conditions")
                return 0, []
            
            # Track closed positions
            closed_trade_ids = []
            updated_positions = []
            
            # Iterate through all paper positions
            for trade_id, position in list(self.paper_positions.items()):
                symbol = position['symbol']
                side = position['side']
                entry_price = position['entry_price']
                quantity = position['quantity']
                
                # Get current price
                current_price = trade_executor.get_current_price(symbol, paper_trade=True)
                if current_price is None:
                    logger.warning(f"Could not get current price for {symbol}, skipping exit check")
                    continue
                
                # Debug stop-loss and take-profit values
                logger.info(f"Checking exit for {symbol} {side} position: Entry price: {entry_price}, Current price: {current_price}")
                logger.info(f"Stop-loss: {position.get('stop_loss', 'Not set')}, Take-profit: {position.get('take_profit', 'Not set')}")
                    
                # Check for stop loss or take profit hit
                stop_loss_hit = False
                take_profit_hit = False
                
                if side == 'BUY':  # Long position
                    # For buy positions: stop loss is below entry, take profit is above entry
                    if current_price <= position.get('stop_loss', 0):
                        stop_loss_hit = True
                        logger.info(f"Paper trading: Stop loss hit for {symbol} BUY position at {current_price}")
                    elif current_price >= position.get('take_profit', float('inf')):
                        take_profit_hit = True
                        logger.info(f"Paper trading: Take profit hit for {symbol} BUY position at {current_price}")
                else:  # SELL/Short position
                    # For sell positions: stop loss is above entry, take profit is below entry
                    if current_price >= position.get('stop_loss', float('inf')):
                        stop_loss_hit = True
                        logger.info(f"Paper trading: Stop loss hit for {symbol} SELL position at {current_price}")
                    elif current_price <= position.get('take_profit', 0):
                        take_profit_hit = True
                        logger.info(f"Paper trading: Take profit hit for {symbol} SELL position at {current_price}")
                
                # Handle position exit if conditions met
                if stop_loss_hit or take_profit_hit:
                    exit_reason = 'stop_loss' if stop_loss_hit else 'take_profit'
                    
                    # Calculate holding time
                    position_entry_time = position.get('entry_time')
                    current_time = datetime.now()
                    holding_days = 0  # Default value
                    
                    # Handle different formats of entry_time
                    if position_entry_time is None:
                        # No entry time available
                        holding_days = 0
                    elif isinstance(position_entry_time, str):
                        try:
                            position_entry_time = datetime.fromisoformat(position_entry_time)
                            current_time = datetime.now()
                            holding_time = current_time - position_entry_time if position_entry_time else timedelta(seconds=0)
                            holding_days = holding_time.total_seconds() / (24 * 60 * 60)
                        except ValueError:
                           # Invalid date format
                           logger.warning(f"Invalid entry_time format for position {trade_id}")
                    else:
                        # Assume entry_time is already a datetime object
                        holding_time = current_time - position_entry_time
                        holding_days = holding_time.total_seconds() / (24 * 60 * 60)
                        
                    # Log comprehensive exit details
                    logger.info(f"=== POSITION EXIT DETAILS ===")
                    logger.info(f"Symbol: {symbol} | Side: {side} | Exit Reason: {exit_reason}")
                    logger.info(f"Entry: {entry_price:.8f} | Exit: {current_price:.8f} | Holding time: {holding_days:.2f} days")
                    logger.info(f"Price change: {((current_price - entry_price) / entry_price) * 100:.2f}% | Direction matched expectation: {(side == 'BUY' and current_price > entry_price) or (side == 'SELL' and current_price < entry_price)}")
                    
                    # Log market conditions at exit
                    try:
                        btc_data = trade_executor.data_fetcher.get_historical_data('BTCUSDT', '1h', '4 hours') 
                        if btc_data is not None and len(btc_data) > 0:
                            latest = btc_data.iloc[-1]
                            logger.info(f"Market conditions: BTC price: ${latest['close']:.2f} | BTC 24h change: {((latest['close'] - btc_data.iloc[-24]['close']) / btc_data.iloc[-24]['close']) * 100:.2f}%")
                            
                    except Exception as e:  
                        logger.debug(f"Could not log market conditions: {e}") 
                        
                    # Record full position lifecycle
                    position['exit_metadata'] = {
                        'entry_time': position_entry_time.isoformat() if isinstance(position_entry_time, datetime) else str(position_entry_time),
                        'exit_time': datetime.now().isoformat(),
                        'holding_days': holding_days,
                        'exit_reason': exit_reason,
                        'price_change_pct': ((current_price - entry_price) / entry_price) * 100,
                        'direction_matched': (side == 'BUY' and current_price > entry_price) or (side == 'SELL' and current_price < entry_price)
                    }        
                             
                    # Calculate P&L
                    initial_value = entry_price * quantity
                    exit_value = current_price * quantity
                    
                    if side == 'BUY':
                        gross_pnl = exit_value - initial_value
                    else:  # SELL
                        gross_pnl = initial_value - exit_value
                    
                    # Add fees (simulate trading fees)
                    entry_fee = initial_value * risk_params.get('taker_fee', 0.001)
                    exit_fee = exit_value * risk_params.get('taker_fee', 0.001)
                    total_fees = entry_fee + exit_fee
                    
                    # Calculate net P&L
                    net_pnl = gross_pnl - total_fees
                    
                    # Update position with exit details
                    position.update({
                        'exit_price': current_price,
                        'exit_time': datetime.now(),
                        'exit_reason': exit_reason,
                        'gross_pnl': gross_pnl,
                        'fees': total_fees,
                        'net_pnl': net_pnl
                    })
                    
                    # Add to updated positions
                    updated_positions.append(position)
                    
                    # Log trade result
                    logger.info(f"Paper trading: Position closed: {trade_id}")
                    logger.info(f"Paper trading: Gross P&L: {gross_pnl:.2f} USDT")
                    logger.info(f"Paper trading: Fees: {total_fees:.2f} USDT")
                    logger.info(f"Paper trading: Net P&L: {net_pnl:.2f} USDT")
                    
                    # Mark position for removal
                    closed_trade_ids.append(trade_id)
            
            # Remove closed positions
            for trade_id in closed_trade_ids:
                self.remove_position(trade_id, paper_trade=True)
            
            return len(closed_trade_ids), updated_positions
            
        except Exception as e:
            logger.error(f"Error updating paper positions: {e}")
            return 0, []
    
    def get_position(self, trade_id, paper_trade=False):
        """Get a specific position"""
        if paper_trade:
            return self.paper_positions.get(trade_id)
        return self.positions.get(trade_id)
        
    def get_all_positions(self, paper_trade=False):
        """Get all open positions"""
        if paper_trade:
            return self.paper_positions
        return self.positions
        
    def get_positions_by_symbol(self, symbol, paper_trade=False):
        """Get all positions for a specific symbol"""
        if paper_trade:
            return {tid: pos for tid, pos in self.paper_positions.items() if pos['symbol'] == symbol}
        return {tid: pos for tid, pos in self.positions.items() if pos['symbol'] == symbol}
        
    def get_total_position_value(self, paper_trade=False):
        """Get total value of all open positions"""
        if paper_trade:
            return sum(pos.get('value', 0) for pos in self.paper_positions.values())
        return sum(pos.get('value', 0) for pos in self.positions.values())
        
    def get_position_summary(self, paper_trade=False):
        """Get summary of positions"""
        positions_to_use = self.paper_positions if paper_trade else self.positions
        
        return {
            'open_positions': len(positions_to_use),
            'total_value': sum(pos.get('value', 0) for pos in positions_to_use.values()),
            'position_details': [
                {
                    'symbol': pos['symbol'],
                    'side': pos['side'],
                    'entry_price': pos['entry_price'],
                    'quantity': pos['quantity'],
                    'value': pos.get('value', 0)
                }
                for pos in positions_to_use.values()
            ]
        }
    
    def add_paper_order(self, order_info):
        """Add a paper trading order"""
        try:
            order_id = order_info.get('orderId', f"paper_order_{int(datetime.now().timestamp())}")
            order_info['orderId'] = order_id
            self.paper_orders.append(order_info)
            logger.info(f"Paper trading: Order added: {order_id}")
            return order_id
        except Exception as e:
            logger.error(f"Error adding paper order: {e}")
            return None
    
    def get_open_orders(self, paper_trade=False):
        """Get open orders with paper trading support"""
        if paper_trade:
            logger.info("Paper trading mode: Using simulated open orders")
            return self.paper_orders
        
        # In a real implementation, this would call the exchange API
        # For now, return an empty list to indicate no live orders
        return []
    
    def add_paper_position(self, symbol, quantity, entry_price, side, position_size=None):
        """Add a new paper trading position"""
        try:
            trade_id = f"{symbol}_{side}_{int(datetime.now().timestamp())}"
            
            # Calculate position value
            value = quantity * entry_price
            
            position = {
                'trade_id': trade_id,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'entry_price': entry_price,
                'value': value,
                'position_size': position_size or value,
                'entry_time': datetime.now(),
                'orders': {}  # Would normally contain stop loss and take profit orders
            }
            
            self.paper_positions[trade_id] = position
            logger.info(f"Paper trading: Added {side} position for {symbol} - {quantity} at {entry_price}")
            return trade_id
        except Exception as e:
            logger.error(f"Error adding paper position: {e}")
            return None
    
    def close_paper_position(self, trade_id, exit_price=None, exit_reason='manual'):
        """Close a paper trading position"""
        try:
            if trade_id in self.paper_positions:
                position = self.paper_positions[trade_id]
                symbol = position['symbol']
                
                # If exit price not provided, use entry price (no profit/loss)
                if exit_price is None:
                    # This would normally get the current market price
                    exit_price = position['entry_price']
                
                entry_price = position['entry_price']
                quantity = position['quantity']
                
                # Calculate P&L
                if position['side'] == 'BUY':
                    gross_pnl = (exit_price - entry_price) * quantity
                else:  # SELL
                    gross_pnl = (entry_price - exit_price) * quantity
                
                # Simulate fees
                fees = (entry_price + exit_price) * quantity * 0.001  # Assume 0.1% fee
                net_pnl = gross_pnl - fees
                
                # Update position with exit details
                position.update({
                    'exit_time': datetime.now(),
                    'exit_price': exit_price,
                    'gross_pnl': gross_pnl,
                    'fees': fees,
                    'net_pnl': net_pnl,
                    'exit_reason': exit_reason
                })
                
                # Move to closed positions
                self.paper_positions.pop(trade_id)
                self.paper_closed_positions.append(position)
                
                logger.info(f"Paper trading: Closed {position['side']} position for {symbol} - PnL: {net_pnl:.2f} USDT")
                return net_pnl
            else:
                logger.warning(f"Paper trading: Position not found: {trade_id}")
                return 0
        except Exception as e:
            logger.error(f"Error closing paper position: {e}")
            return 0