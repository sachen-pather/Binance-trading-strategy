"""
Main execution script for the trading strategy.
"""

import time
import logging
import json
import asyncio
import websockets
import threading
import numpy as np
from datetime import datetime
from trading_strategy import TradingStrategy
from utils import setup_logging
from config import ENV_CONFIG, RISK_PARAMS

logger = logging.getLogger("BinanceTrading")

# Global variables
websocket_event_loop = None
websocket_server = None
connected_clients = set()

# Create a custom JSON encoder that handles NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return json.JSONEncoder.default(self, obj)

# Helper function to recursively clean data for JSON serialization
def clean_data_for_json(obj):
    """Clean data structure to ensure it's JSON serializable"""
    if isinstance(obj, dict):
        return {key: clean_data_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [clean_data_for_json(item) for item in obj]
    elif isinstance(obj, tuple):
        return [clean_data_for_json(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    # Add more type conversions as needed
    return obj

async def websocket_handler(websocket):
    """Handle WebSocket connections"""
    logger.info("New WebSocket client connected")
    connected_clients.add(websocket)
    try:
        # Send initial data that includes empty structures for all components
        initial_data = {
            "connected": True,
            "message": "Connection established",
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "paper_trade": True,
            "equity": 0,
            "market_regime": "",
            "opportunities": [],
            "executed_trades": [],
            "open_positions": [],
            "closed_trades": [],
            "market_data": {
                "BTC": {},
                "ETH": {}
            },
            "risk_params": {},
            "performance": {
                "daily_pnl": 0,
                "total_pnl": 0,
                "win_rate": 0
            }
        }
        await websocket.send(json.dumps(initial_data))
        
        # Keep connection alive by handling messages
        async for message in websocket:
            logger.debug(f"Received message from client: {message}")
            
            # You could add command handling here if needed
            try:
                msg_data = json.loads(message)
                if msg_data.get('type') == 'ping':
                    await websocket.send(json.dumps({"type": "pong", "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}))
            except json.JSONDecodeError:
                logger.warning(f"Received non-JSON message: {message}")
            
    except websockets.exceptions.ConnectionClosed as e:
        logger.info(f"WebSocket client disconnected: {str(e)}")
    except Exception as e:
        logger.error(f"Error in websocket handler: {str(e)}", exc_info=True)
    finally:
        if websocket in connected_clients:
            connected_clients.remove(websocket)
            logger.info(f"Client removed from connected_clients. Total clients: {len(connected_clients)}")

async def broadcast_data(data):
    """Broadcast data to all connected WebSocket clients"""
    if not connected_clients:
        logger.debug("No connected clients to broadcast to")
        return
    
    try:
        # Use the custom encoder to handle NumPy types
        message = json.dumps(data, cls=NumpyEncoder)
        logger.debug(f"Broadcasting to {len(connected_clients)} clients")
    except Exception as e:
        logger.error(f"JSON serialization error: {str(e)}")
        # Try to clean the data before giving up
        try:
            clean_data = clean_data_for_json(data)
            message = json.dumps(clean_data)
            logger.debug("Using cleaned data for broadcast")
        except Exception as e2:
            logger.error(f"Failed to clean data: {str(e2)}")
            return
    
    # Create a list to track clients with errors
    clients_to_remove = []
    
    for client in connected_clients:
        try:
            await client.send(message)
        except websockets.exceptions.ConnectionClosed:
            logger.debug(f"Client disconnected during broadcast")
            clients_to_remove.append(client)
        except Exception as e:
            logger.error(f"Error sending to client: {str(e)}")
            clients_to_remove.append(client)
            
    # Remove problematic clients
    for client in clients_to_remove:
        if client in connected_clients:  # Check in case it was already removed
            connected_clients.remove(client)
            logger.debug(f"Removed disconnected client. Remaining clients: {len(connected_clients)}")

def run_websocket_server():
    """Run the WebSocket server in its own thread with its own event loop"""
    global websocket_event_loop
    global websocket_server
    
    # Create a new event loop for this thread
    loop = asyncio.new_event_loop()
    websocket_event_loop = loop
    asyncio.set_event_loop(loop)
    
    # Use a threading Event to signal when the server is ready
    ready_event = threading.Event()
    
    async def start_server():
        global websocket_server
        try:
            # Create the server
            websocket_server = await websockets.serve(
                websocket_handler, 
                "0.0.0.0", 
                8765,
                ping_interval=30,  # Keep connections alive
                ping_timeout=10,
                origins=["http://localhost:5173"]  # Match your frontend development server
            )
            
            logger.info("WebSocket server started successfully on ws://0.0.0.0:8765")
            ready_event.set()  # Signal that we're ready
            
            # Keep the server running indefinitely
            await asyncio.Future()  # This will never complete
            
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {str(e)}", exc_info=True)
            ready_event.set()  # Signal even on failure
            raise
    
    # Start the server in the event loop
    try:
        # Schedule the coroutine to run
        loop.create_task(start_server())
        
        # Run the event loop indefinitely
        loop.run_forever()
    except Exception as e:
        logger.error(f"WebSocket server loop failed: {str(e)}", exc_info=True)
    finally:
        # Clean up on exit
        if websocket_server:
            websocket_server.close()
        if not loop.is_closed():
            loop.close()
        logger.info("WebSocket server shut down")

def start_websocket_server():
    """Start the WebSocket server in a separate thread and return the thread"""
    # Create and start the thread for the WebSocket server
    ws_thread = threading.Thread(target=run_websocket_server, daemon=True)
    ws_thread.start()
    logger.info("WebSocket server thread started")
    
    # Give the server a moment to start
    time.sleep(2)
    
    return ws_thread

# Function to safely send data from the main thread
def send_data_to_clients(data):
    """Safely send data to WebSocket clients from the main thread"""
    global websocket_event_loop
    
    if not websocket_event_loop or websocket_event_loop.is_closed():
        logger.warning("WebSocket event loop is not available")
        return False
    
    if not connected_clients:
        logger.debug("No connected clients")
        return False
    
    try:
        # Use the custom encoder to handle NumPy types
        clean_data = clean_data_for_json(data)
        
        # Schedule the broadcast coroutine to run in the WebSocket thread
        future = asyncio.run_coroutine_threadsafe(
            broadcast_data(clean_data), 
            websocket_event_loop
        )
        
        # Wait for the result with a timeout to avoid blocking indefinitely
        future.result(timeout=1.0)
        return True
    except Exception as e:
        logger.error(f"Error broadcasting data: {str(e)}", exc_info=True)
        return False

def display_exit_analytics(strategy):
    """Display detailed exit analytics"""
    exit_stats = strategy.exit_analytics.get_exit_statistics()
    
    if not exit_stats:
        print("No exit data available yet.")
        return
        
    print("\n=== EXIT ANALYTICS ===")
    
    print("\n--- EXIT DISTRIBUTION ---")
    for reason, data in exit_stats['exit_distribution'].items():
        print(f"{reason}: {data['count']} exits ({data['percentage']:.1f}%)")
    
    print("\n--- WIN RATE BY EXIT REASON ---")
    for reason, win_rate in exit_stats['win_rate_by_reason'].items():
        if exit_stats['exit_distribution'][reason]['count'] > 0:
            print(f"{reason}: {win_rate:.1f}% win rate")
    
    print("\n--- AVERAGE P&L BY EXIT REASON ---")
    for reason, avg_pnl in exit_stats['avg_pnl_by_reason'].items():
        if exit_stats['exit_distribution'][reason]['count'] > 0:
            print(f"{reason}: {avg_pnl:.2f} USDT average P&L")

def display_balance_report(strategy, paper_trade=True):
    """Display comprehensive balance and position report with detailed price information"""
    print("\n=== ACCOUNT BALANCE REPORT ===")
    
    # Get current equity
    equity = strategy.total_equity
    
    # Get open positions
    positions = strategy.position_manager.get_all_positions(paper_trade=paper_trade)
    
    # Calculate current market values and unrealized P&L
    total_position_value = 0
    total_unrealized_pnl = 0
    
    print(f"Starting Equity: {strategy.initial_equity:.2f} USDT")
    print(f"Current Equity: {equity:.2f} USDT")
    
    if positions:
        print("\n--- OPEN POSITIONS ---")
        print(f"{'SYMBOL':<10} {'SIDE':<6} {'ENTRY PRICE':<12} {'CURRENT PRICE':<14} {'QUANTITY':<14} {'INITIAL VALUE':<14} {'CURRENT VALUE':<14} {'P&L':<10} {'P&L %':<8}")
        print("-" * 110)
        
        for trade_id, position in positions.items():
            symbol = position['symbol']
            side = position['side']
            entry_price = position['entry_price']
            quantity = position['quantity']
            initial_value = quantity * entry_price
            
            # Get current price - this uses the existing method that works with paper trading
            current_price = strategy.trade_executor.get_current_price(symbol, paper_trade=paper_trade)
            if not current_price:
                # If we can't get the current price, use entry price as a fallback
                current_price = entry_price
            
            # Calculate current value and unrealized P&L
            current_value = quantity * current_price
            
            if side == 'BUY':
                unrealized_pnl = current_value - initial_value
            else:  # SELL
                unrealized_pnl = initial_value - current_value
                
            # Calculate P&L percentage
            pnl_percentage = (unrealized_pnl / initial_value) * 100 if initial_value else 0
                
            # Add to totals
            total_position_value += current_value
            total_unrealized_pnl += unrealized_pnl
            
            # Display position with formatted values
            print(f"{symbol:<10} {side:<6} {entry_price:<12.6f} {current_price:<14.6f} {quantity:<14.6f} {initial_value:<14.2f} {current_value:<14.2f} {unrealized_pnl:<10.2f} {pnl_percentage:<8.2f}%")
    
    else:
        print("\n--- NO OPEN POSITIONS ---")
    
    # Calculate total account value
    total_account_value = equity + total_unrealized_pnl
    
    print("\n--- SUMMARY ---")
    print(f"Total Position Value: {total_position_value:.2f} USDT")
    print(f"Total Unrealized P&L: {total_unrealized_pnl:.2f} USDT ({(total_unrealized_pnl/equity)*100 if equity else 0:.2f}%)")
    print(f"Total Account Value: {total_account_value:.2f} USDT")
    
    # Show recent closed trades
    closed_positions = strategy.position_manager.paper_closed_positions if paper_trade else strategy.position_manager.closed_positions
    if closed_positions:
        recent_trades = sorted(closed_positions, key=lambda x: x.get('exit_time', datetime.now()), reverse=True)[:5]
        
        if recent_trades:
            print("\n--- RECENT CLOSED TRADES ---")
            print(f"{'SYMBOL':<10} {'SIDE':<6} {'ENTRY':<10} {'EXIT':<10} {'P&L':<10} {'P&L %':<8} {'DATE':<20}")
            print("-" * 80)
            
            for trade in recent_trades:
                symbol = trade['symbol']
                side = trade['side']
                entry_price = trade['entry_price']
                exit_price = trade.get('exit_price', 0)
                net_pnl = trade.get('net_pnl', 0)
                
                # Calculate P&L percentage for closed trades
                initial_value = trade.get('value', entry_price * trade.get('quantity', 0))
                pnl_percentage = (net_pnl / initial_value) * 100 if initial_value else 0
                
                exit_time = trade.get('exit_time', 'Open')
                
                if isinstance(exit_time, datetime):
                    exit_time = exit_time.strftime('%Y-%m-%d %H:%M')
                
                print(f"{symbol:<10} {side:<6} {entry_price:<10.6f} {exit_price:<10.6f} {net_pnl:<10.2f} {pnl_percentage:<8.2f}% {exit_time:<20}")
    
    print("\n--- PERFORMANCE ---")
    print(f"Daily P&L: {strategy.performance_tracker.daily_pnl:.2f} USDT")
    print(f"Total Realized P&L: {strategy.performance_tracker.performance_metrics.get('total_pnl', 0):.2f} USDT")
    print(f"Win Rate: {strategy.performance_tracker.performance_metrics.get('win_rate', 0):.2f}%")
    print("=" * 50)

def main():
    """Main execution function"""
    # Setup logging
    setup_logging(ENV_CONFIG['log_file'])
    
    # Start WebSocket server in a separate thread
    ws_thread = start_websocket_server()
    
    # Wait briefly to ensure the server has time to start
    time.sleep(2)
    
    # Create strategy instance with live data but paper trades
    strategy = TradingStrategy(test_mode=False)  # Live data
    
    # Paper trading flag
    paper_trade = True  # Set to False when ready for live trading
    
    # Fix: Disable circuit breaker for testing
    strategy.market_analyzer.circuit_breaker['active'] = False
    strategy.market_analyzer.circuit_breaker['manually_disabled'] = True  # Prevent re-enabling
    
    # Safer settings for testing
    RISK_PARAMS['base_position_size'] = 0.005  # 0.5% position size
    RISK_PARAMS['max_position_size'] = 0.02   # 2% max position
    RISK_PARAMS['max_daily_loss'] = 0.01      # 1% max daily loss
    RISK_PARAMS['extreme_volatility_threshold'] = 0.20  # 20% threshold (increased for testing)
    
    # Display mode clearly
    print("====================================")
    if paper_trade:
        print("PAPER TRADING MODE - No real orders will be placed")
    else:
        print("WARNING: LIVE TRADING MODE ENABLED!")
    # Get account equity with paper_trade flag to avoid API errors
    equity = strategy.data_fetcher.get_account_equity(paper_trade=paper_trade, default_equity=10000)
    print(f"Current equity: {equity} USDT")
    print("====================================")
    
    # Train model manually
    if paper_trade:
        logger.info("Paper trading mode active - no real trades")
    else:
        logger.warning("Live trading mode - real trades will be executed")
        
    print("Training ML model...")
    try:
        success, accuracy = strategy.train_ml_model()
        if success:
            print(f"Model trained successfully. Accuracy: {accuracy:.2f}")
        else:
            print("Model training failed, using default signals")
    except Exception as e:
        print(f"Error training model: {e}")
    
    # Initial portfolio analysis - get with paper_trade flag to avoid API errors
    initial_equity = strategy.data_fetcher.get_account_equity(paper_trade=paper_trade, default_equity=10000)
    logger.info(f"Initial equity: {initial_equity} USDT")
    logger.info(f"Market regime: {strategy.market_analyzer.market_state['regime']}")
    
    # Main strategy loop
    try:
        while True:
            # Check if it's a new day to reset daily metrics
            current_time = datetime.now()
            if current_time.hour == 0 and current_time.minute < 5:
                strategy.performance_tracker.reset_daily_metrics()
                
            # Disable circuit breaker for testing
            strategy.market_analyzer.circuit_breaker['active'] = False
            
            # Run strategy - always pass paper_trade flag
            logger.info("\nRunning strategy check...")
            print("\n=== STRATEGY CHECK ===")
            result = strategy.run_strategy(automatic=True, paper_trade=paper_trade)
            
            # Prepare data to send to frontend
            frontend_data = {
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "paper_trade": paper_trade,
                "equity": strategy.total_equity,
                "market_regime": strategy.market_analyzer.market_state['regime'],
                "opportunities": [],
                "executed_trades": [],
                "market_data": {
                    "BTC": {},
                    "ETH": {}
                },
                "risk_params": {
                    "base_position_size": strategy.risk_manager.risk_params['base_position_size'] * 100,
                    "max_position_size": strategy.risk_manager.risk_params['max_position_size'] * 100,
                    "max_open_positions": strategy.risk_manager.risk_params['max_open_positions'],
                    "max_daily_loss": strategy.risk_manager.risk_params['max_daily_loss'] * 100,
                    "daily_pnl": strategy.performance_tracker.daily_pnl
                },
                # Add Open Positions data
                "open_positions": [],
                # Add Closed Trades data
                "closed_trades": []
            }
            
            if result:
                logger.info(f"Current equity: {result['equity']} USDT")
                logger.info(f"Open positions: {result['open_positions']}")
                logger.info(f"Market regime: {result['market_regime']}")
                
                # Add opportunities to frontend data
                if result['opportunities']:
                    frontend_data["opportunities"] = [
                        {
                            "symbol": opp['symbol'],
                            "signal": opp['signal'],
                            "score": opp['score'],
                            "price": opp['price']
                        } for opp in result['opportunities']
                    ]
                    print(f"\nFound {len(result['opportunities'])} opportunities:")
                    for opp in result['opportunities']:
                        print(f"  {opp['symbol']} {opp['signal']} (Score: {opp['score']:.2f}) - Price: {opp['price']:.2f}")
                
                # Add executed trades to frontend data
                if result['executed_trades']:
                    frontend_data["executed_trades"] = [
                        {
                            "symbol": trade['symbol'],
                            "side": trade['side'],
                            "entry_price": trade['entry_price']
                        } for trade in result['executed_trades']
                    ]
                    print(f"\n[{('PAPER' if paper_trade else 'LIVE')}] Executed {len(result['executed_trades'])} trades:")
                    for trade in result['executed_trades']:
                        print(f"  - {trade['symbol']} {trade['side']} at {trade['entry_price']:.2f}")
                else:
                    print("\nNo trades executed this cycle")
            
            # Always display live market data after strategy check
            print("\n=== LIVE MARKET DATA & METRICS ===")
            print("Fetching live data...")
            try:
                # Fetch BTC data - public API calls should work without authentication
                btc_data = strategy.data_fetcher.get_historical_data('BTCUSDT', '1m', '4 hours')
                if btc_data is not None and len(btc_data) > 0:
                    latest_btc = btc_data.iloc[-1]
                    print(f"BTC Price: ${latest_btc['close']:.2f}")
                    print(f"BTC High (1min): ${latest_btc['high']:.2f}")
                    print(f"BTC Low (1min): ${latest_btc['low']:.2f}")
                    print(f"BTC Volume (1min): {latest_btc['volume']:.2f}")
                    
                    # Add BTC data to frontend data
                    frontend_data["market_data"]["BTC"] = {
                        "price": float(latest_btc['close']),
                        "high": float(latest_btc['high']),
                        "low": float(latest_btc['low']),
                        "volume": float(latest_btc['volume'])
                    }
                    
                    # Get indicators for BTC
                    btc_indicators = strategy.indicator_calculator.calculate_indicators(btc_data)
                    if btc_indicators is not None and len(btc_indicators) > 0:
                        latest_indicators = btc_indicators.iloc[-1]
                        
                        # Add indicators to frontend data - convert NumPy types to Python types
                        frontend_data["market_data"]["BTC"]["indicators"] = {
                            "RSI": float(latest_indicators['RSI']),
                            "MACD": float(latest_indicators['MACD']),
                            "MACD_signal": float(latest_indicators['MACD_signal']),
                            "ADX": float(latest_indicators['ADX']),
                            "ATR": float(latest_indicators['ATR']),
                            "BB_position": float(latest_indicators['BB_position']),
                            "volume_ratio": float(latest_indicators['volume_ratio'])
                        }
                        
                        # Add signal conditions to frontend data - convert NumPy bools to Python bools
                        frontend_data["market_data"]["BTC"]["signal_conditions"] = {
                            "rsi_oversold": bool(latest_indicators['RSI'] < strategy.indicator_calculator.strategy_params['rsi_oversold']),
                            "rsi_overbought": bool(latest_indicators['RSI'] > strategy.indicator_calculator.strategy_params['rsi_overbought']),
                            "bb_lower_touch": bool(latest_btc['close'] < latest_indicators['BB_lower']),
                            "bb_upper_touch": bool(latest_btc['close'] > latest_indicators['BB_upper']),
                            "trend_strength": bool(latest_indicators['ADX'] > strategy.indicator_calculator.strategy_params['adx_threshold']),
                            "volume_spike": bool(latest_indicators['volume_ratio'] > strategy.indicator_calculator.strategy_params['volume_threshold'])
                        }
                        
                        # Add indicator parameters to frontend data
                        frontend_data["market_data"]["BTC"]["indicator_params"] = {
                            "rsi_oversold": float(strategy.indicator_calculator.strategy_params['rsi_oversold']),
                            "rsi_overbought": float(strategy.indicator_calculator.strategy_params['rsi_overbought']),
                            "adx_threshold": float(strategy.indicator_calculator.strategy_params['adx_threshold']),
                            "volume_threshold": float(strategy.indicator_calculator.strategy_params['volume_threshold'])
                        }
                        
                        print("\n--- BTC TECHNICAL INDICATORS ---")
                        print(f"RSI: {latest_indicators['RSI']:.1f} (Buy < {strategy.indicator_calculator.strategy_params['rsi_oversold']}, Sell > {strategy.indicator_calculator.strategy_params['rsi_overbought']})")
                        print(f"MACD: {latest_indicators['MACD']:.4f}, Signal: {latest_indicators['MACD_signal']:.4f}")
                        print(f"ADX: {latest_indicators['ADX']:.1f} (Trend threshold: {strategy.indicator_calculator.strategy_params['adx_threshold']})")
                        print(f"ATR: {latest_indicators['ATR']:.2f}")
                        print(f"BB Position: {latest_indicators['BB_position']:.2f} (0-1 range)")
                        print(f"Volume Ratio: {latest_indicators['volume_ratio']:.2f} (Threshold: {strategy.indicator_calculator.strategy_params['volume_threshold']})")
                        
                        # Check if conditions would trigger trades
                        rsi_oversold = latest_indicators['RSI'] < strategy.indicator_calculator.strategy_params['rsi_oversold']
                        rsi_overbought = latest_indicators['RSI'] > strategy.indicator_calculator.strategy_params['rsi_overbought']
                        bb_lower_touch = latest_btc['close'] < latest_indicators['BB_lower']
                        bb_upper_touch = latest_btc['close'] > latest_indicators['BB_upper']
                        
                        print("\n--- SIGNAL CONDITIONS CHECK ---")
                        print(f"RSI Oversold (Buy): {rsi_oversold}")
                        print(f"RSI Overbought (Sell): {rsi_overbought}")
                        print(f"BB Lower Touch (Buy): {bb_lower_touch}")
                        print(f"BB Upper Touch (Sell): {bb_upper_touch}")
                        print(f"Trend Strength (ADX > threshold): {latest_indicators['ADX'] > strategy.indicator_calculator.strategy_params['adx_threshold']}")
                        print(f"Volume Spike: {latest_indicators['volume_ratio'] > strategy.indicator_calculator.strategy_params['volume_threshold']}")
                else:
                    print("Failed to fetch BTC data")
                    logger.error("BTC data fetch failed: No data returned")
                    
                # Get ETH data - public API call should work without authentication
                eth_data = strategy.data_fetcher.get_historical_data('ETHUSDT', '1m', '4 hours')
                if eth_data is not None and len(eth_data) > 0:
                    latest_eth = eth_data.iloc[-1]
                    print(f"\nETH Price: ${latest_eth['close']:.2f}")
                    
                    frontend_data["market_data"]["ETH"] = {
                        "price": float(latest_eth['close'])
                    }
                else:
                    print("Failed to fetch ETH data")
                    logger.error("ETH data fetch failed: No data returned")
                    
                # Show risk parameters
                print("\n--- RISK PARAMETERS ---")
                print(f"Base Position Size: {strategy.risk_manager.risk_params['base_position_size']*100:.1f}%")
                print(f"Max Position Size: {strategy.risk_manager.risk_params['max_position_size']*100:.1f}%")
                print(f"Max Open Positions: {strategy.risk_manager.risk_params['max_open_positions']}")
                print(f"Daily Loss Limit: {strategy.risk_manager.risk_params['max_daily_loss']*100:.1f}%")
                print(f"Current Daily P&L: {strategy.performance_tracker.daily_pnl:.2f} USDT")
                print(f"Signal Threshold: 0.3 (must exceed for trades)")
                
                # Display comprehensive balance report
                display_balance_report(strategy, paper_trade=paper_trade)
                display_exit_analytics(strategy)
                
                # Add open positions data to frontend_data
                positions = strategy.position_manager.get_all_positions(paper_trade=paper_trade)
                if positions:
                    for trade_id, position in positions.items():
                        symbol = position['symbol']
                        side = position['side']
                        entry_price = position['entry_price']
                        quantity = position['quantity']
                        initial_value = quantity * entry_price
                        
                        # Get current price
                        current_price = strategy.trade_executor.get_current_price(symbol, paper_trade=paper_trade)
                        if not current_price:
                            current_price = entry_price
                        
                        # Calculate current value and unrealized P&L
                        current_value = quantity * current_price
                        
                        if side == 'BUY':
                            unrealized_pnl = current_value - initial_value
                        else:  # SELL
                            unrealized_pnl = initial_value - current_value
                            
                        # Calculate P&L percentage
                        pnl_percentage = (unrealized_pnl / initial_value) * 100 if initial_value else 0
                        
                        # Add to frontend data
                        frontend_data["open_positions"].append({
                            "symbol": symbol,
                            "side": side,
                            "entry_price": float(entry_price),
                            "current_price": float(current_price),
                            "quantity": float(quantity),
                            "initial_value": float(initial_value),
                            "current_value": float(current_value),
                            "pnl": float(unrealized_pnl),
                            "pnl_percentage": float(pnl_percentage)
                        })

                # Add closed trades data to frontend_data
                closed_positions = strategy.position_manager.paper_closed_positions if paper_trade else strategy.position_manager.closed_positions
                if closed_positions:
                    recent_trades = sorted(closed_positions, key=lambda x: x.get('exit_time', datetime.now()), reverse=True)[:5]
                    
                    for trade in recent_trades:
                        symbol = trade['symbol']
                        side = trade['side']
                        entry_price = trade['entry_price']
                        exit_price = trade.get('exit_price', 0)
                        net_pnl = trade.get('net_pnl', 0)
                        
                        # Calculate P&L percentage for closed trades
                        initial_value = trade.get('value', entry_price * trade.get('quantity', 0))
                        pnl_percentage = (net_pnl / initial_value) * 100 if initial_value else 0
                        
                        exit_time = trade.get('exit_time', 'Open')
                        if isinstance(exit_time, datetime):
                            exit_time = exit_time.strftime('%Y-%m-%d %H:%M')
                        
                        # Add to frontend data
                        frontend_data["closed_trades"].append({
                            "symbol": symbol,
                            "side": side,
                            "entry_price": float(entry_price),
                            "exit_price": float(exit_price),
                            "net_pnl": float(net_pnl),
                            "pnl_percentage": float(pnl_percentage),
                            "exit_time": exit_time
                        })

                # Add performance data
                frontend_data["performance"] = {
                    "daily_pnl": float(strategy.performance_tracker.daily_pnl),
                    "total_pnl": float(strategy.performance_tracker.performance_metrics.get('total_pnl', 0)),
                    "win_rate": float(strategy.performance_tracker.performance_metrics.get('win_rate', 0))
                }
                
            except Exception as e:
                print(f"Error fetching live data: {str(e)}")
                logger.error(f"Error fetching live data: {str(e)}", exc_info=True)
            
            # Before broadcasting, ensure the data has the right structure
            if "market_data" not in frontend_data or not frontend_data["market_data"]:
                frontend_data["market_data"] = {"BTC": {}, "ETH": {}}

            # Or initialize with empty objects if they're missing
            if "BTC" not in frontend_data["market_data"]:
                frontend_data["market_data"]["BTC"] = {}
            if "ETH" not in frontend_data["market_data"]:
                 frontend_data["market_data"]["ETH"] = {}
            
            # IMPROVED BROADCASTING LOGIC with NumPy handling
            try:
                success = send_data_to_clients(frontend_data)
                if success:
                    logger.info(f"Data broadcast successful at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                else:
                    logger.warning("Data broadcast failed or no clients connected")
            except Exception as e:
                logger.error(f"Unexpected error during broadcast: {str(e)}", exc_info=True)
                
            print("========================")
            print(f"Check completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("========================")
                
            # Save state after each run
            strategy.save_strategy_state()
            
            # Check if retraining is needed
            current_time = datetime.now()
            last_retrain_time = strategy.ml_model.strategy_params.get('last_retrain_time')
            
            if last_retrain_time:
                if isinstance(last_retrain_time, str):
                    try:
                        last_retrain_time = datetime.fromisoformat(last_retrain_time)
                    except:
                        last_retrain_time = None
                        
                if last_retrain_time:
                    hours_since_retrain = (current_time - last_retrain_time).total_seconds() / 3600
                    if hours_since_retrain > strategy.ml_model.strategy_params.get('ml_retrain_hours', 6):
                        logger.info("Retraining ML model...")
                        strategy.train_ml_model()
            else:
                logger.info("No previous training time found. Retraining ML model...")
                strategy.train_ml_model()
            
            # Wait before next check
            print(f"\nWaiting 1 minute 30 seconds before next check...")
            time.sleep(90)  # 1.5 minutes
            
    except KeyboardInterrupt:
        logger.info("Strategy stopped by user")
    except Exception as e:
        logger.error(f"Strategy error: {e}", exc_info=True)
    finally:
        # Save final state
        strategy.save_strategy_state()
        logger.info("Strategy state saved")

if __name__ == "__main__":
    main()