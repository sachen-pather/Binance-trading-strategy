"""
Trade execution logic module.
"""

import logging
import time
from datetime import datetime
from binance.client import Client
from binance.exceptions import BinanceAPIException

logger = logging.getLogger("BinanceTrading.TradeExecutor")


class TradeExecutor:
    """Handle trade execution with Binance API"""
    
    def __init__(self, client, data_fetcher):
        self.client = client
        self.data_fetcher = data_fetcher
        self.paper_trades = []  # Store history of paper trades
        self.paper_orders = {}  # Store paper orders by symbol
        
    def get_minimum_notional(self, symbol):
        """Get minimum notional value for a symbol"""
        try:
            # Try to get from exchange info
            if hasattr(self, '_symbol_info_cache'):
                info = self._symbol_info_cache.get(symbol)
            else:
                self._symbol_info_cache = {}
                info = None
            
            if not info:
                try:
                    exchange_info = self.client.get_exchange_info()
                    for s in exchange_info['symbols']:
                        if s['symbol'] == symbol:
                            info = s
                            self._symbol_info_cache[symbol] = info
                            break
                except:
                    info = None
            
            if info:
                # Find MIN_NOTIONAL filter
                for filter_item in info.get('filters', []):
                    if filter_item['filterType'] == 'MIN_NOTIONAL':
                        return float(filter_item['minNotional'])
                    elif filter_item['filterType'] == 'NOTIONAL':
                        return float(filter_item['minNotional'])
            
            # Fallback minimum values by quote asset
            if 'USDT' in symbol:
                return 10.0  # 10 USDT minimum
            elif 'BTC' in symbol:
                return 0.0001  # 0.0001 BTC minimum
            else:
                return 10.0  # Default 10 USDT
                
        except Exception as e:
            logger.error(f"Error getting minimum notional for {symbol}: {e}")
            return 10.0  # Safe fallback

    def adjust_quantity_for_minimum(self, symbol, quantity, price):
        """Adjust quantity to meet minimum notional requirements"""
        try:
            min_notional = self.get_minimum_notional(symbol)
            current_value = quantity * price
            
            logger.info(f"Order check for {symbol}: Value={current_value:.2f}, Min required={min_notional:.2f}")
            
            if current_value < min_notional:
                # Calculate new quantity to meet minimum
                required_quantity = min_notional / price
                
                # Add 5% buffer to ensure we're above minimum
                adjusted_quantity = required_quantity * 1.05
                
                # Format to correct precision
                adjusted_quantity = self.format_quantity(symbol, adjusted_quantity)
                
                logger.info(f"Adjusted quantity for {symbol}: {quantity:.6f} -> {adjusted_quantity:.6f} (Value: {adjusted_quantity * price:.2f} USDT)")
                
                return adjusted_quantity
            
            return quantity
            
        except Exception as e:
            logger.error(f"Error adjusting quantity for {symbol}: {e}")
            return quantity
        
    def format_quantity(self, symbol, quantity):
        """Format quantity to correct precision for the symbol"""
        try:
            # Get exchange info for precise formatting
            if hasattr(self, '_symbol_info_cache'):
                info = self._symbol_info_cache.get(symbol)
            else:
                self._symbol_info_cache = {}
                info = None
            
            if not info:
                try:
                    exchange_info = self.client.get_exchange_info()
                    for s in exchange_info['symbols']:
                        if s['symbol'] == symbol:
                            info = s
                            self._symbol_info_cache[symbol] = info
                            break
                except:
                    info = None
            
            if info:
                # Find the LOT_SIZE filter for precise quantity formatting
                for filter_item in info.get('filters', []):
                    if filter_item['filterType'] == 'LOT_SIZE':
                        step_size = float(filter_item['stepSize'])
                        
                        # Calculate precision based on step size
                        if step_size >= 1:
                            precision = 0
                        else:
                            precision = len(str(step_size).split('.')[-1].rstrip('0'))
                        
                        # Round to the correct precision
                        formatted_quantity = round(quantity, precision)
                        logger.info(f"Formatted quantity for {symbol}: {quantity} -> {formatted_quantity} (precision: {precision})")
                        return formatted_quantity
            
            # Fallback precision rules if exchange info not available
            if 'BTC' in symbol or 'ETH' in symbol:
                return round(quantity, 6)  # High precision for major coins
            elif 'BNB' in symbol or 'ADA' in symbol or 'DOT' in symbol:
                return round(quantity, 2)  # Medium precision
            else:
                return round(quantity, 3)  # Default precision
                
        except Exception as e:
            logger.error(f"Error formatting quantity for {symbol}: {e}")
            # Safe fallback
            return round(quantity, 3)
    
    def format_price(self, symbol, price):
        """Format price to correct precision for the symbol"""
        try:
            # Get exchange info for precise formatting
            if hasattr(self, '_symbol_info_cache'):
                info = self._symbol_info_cache.get(symbol)
            else:
                self._symbol_info_cache = {}
                info = None
            
            if not info:
                try:
                    exchange_info = self.client.get_exchange_info()
                    for s in exchange_info['symbols']:
                        if s['symbol'] == symbol:
                            info = s
                            self._symbol_info_cache[symbol] = info
                            break
                except:
                    info = None
            
            if info:
                # Find the PRICE_FILTER for precise price formatting
                for filter_item in info.get('filters', []):
                    if filter_item['filterType'] == 'PRICE_FILTER':
                        tick_size = float(filter_item['tickSize'])
                        
                        # Calculate precision based on tick size
                        if tick_size >= 1:
                            precision = 0
                        else:
                            precision = len(str(tick_size).split('.')[-1].rstrip('0'))
                        
                        # Round to the correct precision
                        return round(price, precision)
            
            # Fallback precision rules
            if price > 1000:
                return round(price, 2)  # High value coins
            elif price > 1:
                return round(price, 4)  # Medium value coins
            else:
                return round(price, 8)  # Low value coins
                
        except Exception as e:
            logger.error(f"Error formatting price for {symbol}: {e}")
            return round(price, 4)  # Safe fallback
        
    def execute_trade(self, symbol, side, quantity, risk_manager, data_fetcher, 
                     indicator_calculator, market_state, order_type='MARKET', 
                     paper_trade=True):
        """Execute trade with comprehensive risk management and paper trading option"""
        try:
            # Get current price
            current_price = self.get_current_price(symbol, paper_trade)
            if current_price is None:
                logger.error(f"Failed to get current price for {symbol}")
                return None
            
            # Format quantity to correct precision FIRST
            quantity = self.format_quantity(symbol, quantity)
            
            # THEN adjust for minimum notional requirements
            if not paper_trade:
                quantity = self.adjust_quantity_for_minimum(symbol, quantity, current_price)
                
                # Final check - if still too small, skip the trade
                min_notional = self.get_minimum_notional(symbol)
                final_value = quantity * current_price
                
                if final_value < min_notional:
                    logger.warning(f"Cannot place order for {symbol}: Final value {final_value:.2f} still below minimum {min_notional:.2f}")
                    return None
            
            # Calculate stop loss and take profit levels
            df = data_fetcher.get_historical_data(symbol, interval='1h', lookback='3 days')
            df = indicator_calculator.calculate_indicators(df)
            
            stop_loss, take_profit = risk_manager.calculate_stop_loss_take_profit(
                symbol, side, current_price, market_state, df
            )
            
            # Format prices to correct precision
            stop_loss = self.format_price(symbol, stop_loss)
            take_profit = self.format_price(symbol, take_profit)
            
            # Prepare trade details
            trade_id = f"{symbol}_{side}_{int(time.time())}"
            trade_info = {
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'entry_price': current_price,
                'entry_time': datetime.now(),
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'value': quantity * current_price,
                'orders': {},
                'paper_trade': paper_trade
            }
            
            # Execute main order
            if paper_trade:
                # PAPER TRADE: Simulate order completely (no API calls)
                simulated_order = self.simulate_market_order(symbol, side, quantity, current_price)
                trade_info['orders']['main'] = simulated_order
                
                # Add to paper trades history
                self.paper_trades.append({
                    'trade_id': trade_id,
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'price': current_price,
                    'time': datetime.now(),
                    'value': quantity * current_price,
                    'type': 'MARKET'
                })
                
                logger.info(f"[PAPER TRADE] Market order simulated: {side} {quantity} {symbol} at {current_price}")
                
                # Simulate stop loss order
                sl_side = 'SELL' if side == 'BUY' else 'BUY'
                sl_order = self.simulate_limit_order(symbol, sl_side, quantity, stop_loss, order_type='STOP_LOSS')
                trade_info['orders']['stop_loss'] = sl_order
                logger.info(f"[PAPER TRADE] Stop loss order simulated at {stop_loss}")
                
                # Simulate take profit order
                tp_side = 'SELL' if side == 'BUY' else 'BUY'
                tp_order = self.simulate_limit_order(symbol, tp_side, quantity, take_profit, order_type='LIMIT')
                trade_info['orders']['take_profit'] = tp_order
                logger.info(f"[PAPER TRADE] Take profit order simulated at {take_profit}")
                
                # Store the paper orders for this symbol
                if symbol not in self.paper_orders:
                    self.paper_orders[symbol] = []
                self.paper_orders[symbol].append({
                    'trade_id': trade_id,
                    'main_order': simulated_order,
                    'stop_loss': sl_order,
                    'take_profit': tp_order
                })
                
            else:
                # REAL TRADE: Use actual order endpoint
                try:
                    logger.info(f"Placing LIVE order: {side} {quantity} {symbol} at market price (Value: {quantity * current_price:.2f} USDT)")
                    
                    order = self.client.create_order(
                        symbol=symbol,
                        side=side,
                        type=Client.ORDER_TYPE_MARKET,
                        quantity=quantity
                    )
                    logger.info(f"✅ LIVE Market order placed: {side} {quantity} {symbol}")
                    trade_info['orders']['main'] = order
                    
                    # Place stop loss order
                    try:
                        sl_side = 'SELL' if side == 'BUY' else 'BUY'
                        sl_order = self.client.create_order(
                            symbol=symbol,
                            side=sl_side,
                            type=Client.ORDER_TYPE_STOP_LOSS_LIMIT,
                            timeInForce=Client.TIME_IN_FORCE_GTC,
                            quantity=quantity,
                            stopPrice=stop_loss,
                            price=stop_loss
                        )
                        logger.info(f"✅ LIVE Stop loss order placed at {stop_loss}")
                        trade_info['orders']['stop_loss'] = sl_order
                    except Exception as e:
                        logger.error(f"Error placing stop loss order: {e}")
                    
                    # Place take profit order
                    try:
                        tp_side = 'SELL' if side == 'BUY' else 'BUY'
                        tp_order = self.client.create_order(
                            symbol=symbol,
                            side=tp_side,
                            type=Client.ORDER_TYPE_LIMIT,
                            timeInForce=Client.TIME_IN_FORCE_GTC,
                            quantity=quantity,
                            price=take_profit
                        )
                        logger.info(f"✅ LIVE Take profit order placed at {take_profit}")
                        trade_info['orders']['take_profit'] = tp_order
                    except Exception as e:
                        logger.error(f"Error placing take profit order: {e}")
                
                except BinanceAPIException as e:
                    logger.error(f"❌ API error executing trade: {e}")
                    return None
            
            return trade_info
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return None
    
    def get_current_price(self, symbol, paper_trade=True):
        """Get current price with paper trading support"""
        try:
            # First try to get recent price from historical data
            # This should work without API credentials for public data
            recent_data = self.data_fetcher.get_historical_data(symbol, '1m', '5 minutes')
            if recent_data is not None and len(recent_data) > 0:
                return float(recent_data.iloc[-1]['close'])
            
            # If that fails and we're not in paper trading mode, try direct API call
            if not paper_trade:
                ticker = self.client.get_symbol_ticker(symbol=symbol)
                return float(ticker['price'])
            
            # If all else fails in paper trading mode, return a sensible default
            # This would be improved by having a fallback price mechanism
            logger.warning(f"Could not get price for {symbol} in paper trading mode, using default")
            return 0.0  # Return 0 to make it obvious something is wrong
            
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    def simulate_market_order(self, symbol, side, quantity, price):
        """Simulate a market order for paper trading"""
        order_id = f"paper_{symbol}_{side}_{int(time.time() * 1000)}"
        
        return {
            'orderId': order_id,
            'symbol': symbol,
            'side': side,
            'type': 'MARKET',
            'status': 'FILLED',
            'price': price,
            'origQty': quantity,
            'executedQty': quantity,
            'time': int(time.time() * 1000),
            'paper_trade': True
        }
    
    def simulate_limit_order(self, symbol, side, quantity, price, order_type='LIMIT'):
        """Simulate a limit order for paper trading"""
        order_id = f"paper_{symbol}_{side}_{order_type}_{int(time.time() * 1000)}"
        
        return {
            'orderId': order_id,
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'status': 'NEW',
            'price': price,
            'origQty': quantity,
            'executedQty': '0',
            'time': int(time.time() * 1000),
            'paper_trade': True
        }
            
    def cancel_order(self, symbol, order_id, paper_trade=True):
        """Cancel an order with paper trading support"""
        if paper_trade:
            # Check if this is a paper order
            if symbol in self.paper_orders:
                for trade in self.paper_orders[symbol]:
                    # Check all order types
                    for order_type in ['main_order', 'stop_loss', 'take_profit']:
                        if order_type in trade and trade[order_type].get('orderId') == order_id:
                            trade[order_type]['status'] = 'CANCELED'
                            logger.info(f"[PAPER TRADE] Order cancelled: {symbol} {order_id}")
                            return {'orderId': order_id, 'status': 'CANCELED'}
            
            logger.warning(f"[PAPER TRADE] Order not found for cancellation: {symbol} {order_id}")
            return None
        
        # Real trading mode
        try:
            result = self.client.cancel_order(symbol=symbol, orderId=order_id)
            logger.info(f"Order cancelled: {symbol} {order_id}")
            return result
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return None
            
    def check_order_status(self, symbol, order_id, paper_trade=True):
        """Check the status of an order with paper trading support"""
        if paper_trade:
            # Check if this is a paper order
            if symbol in self.paper_orders:
                for trade in self.paper_orders[symbol]:
                    # Check all order types
                    for order_type in ['main_order', 'stop_loss', 'take_profit']:
                        if order_type in trade and trade[order_type].get('orderId') == order_id:
                            return trade[order_type]
            
            logger.warning(f"[PAPER TRADE] Order not found: {symbol} {order_id}")
            return None
        
        # Real trading mode
        try:
            order = self.client.get_order(symbol=symbol, orderId=order_id)
            return order
        except Exception as e:
            logger.error(f"Error checking order status: {e}")
            return None
    
    def get_paper_trade_history(self):
        """Get all paper trade history"""
        return self.paper_trades
    
    def get_paper_open_orders(self, symbol=None):
        """Get all open paper orders, optionally filtered by symbol"""
        if symbol:
            return self.paper_orders.get(symbol, [])
        
        all_orders = []
        for symbol_orders in self.paper_orders.values():
            all_orders.extend(symbol_orders)
        return all_orders