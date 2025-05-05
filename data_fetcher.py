"""
Data fetching module for historical data and API interactions.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from binance.client import Client
from binance.exceptions import BinanceAPIException
from config import ENV_CONFIG

logger = logging.getLogger("BinanceTrading.DataFetcher")


class DataFetcher:
    """Handle data fetching operations with Binance API"""
    
    def __init__(self, client):
        self.client = client
        
    def get_historical_data(self, symbol='BTCUSDT', interval='1h', lookback='14 days'):
        """Get historical klines/candlestick data with error handling and retries"""
        max_retries = 5
        for attempt in range(max_retries):
            try:
                # Calculate start time
                lookback_time = datetime.now() - pd.Timedelta(lookback)
                
                klines = self.client.get_historical_klines(
                    symbol,
                    interval,
                    str(int(lookback_time.timestamp() * 1000))
                )
                
                # Convert to DataFrame
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 
                    'volume', 'close_time', 'quote_volume', 'trades',
                    'taker_buy_base', 'taker_buy_quote', 'ignore'
                ])
                
                # Convert types
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)
                
                df.set_index('timestamp', inplace=True)
                
                if len(df) == 0:
                    logger.warning(f"No data returned for {symbol} {interval}")
                    return None
                    
                return df
                
            except BinanceAPIException as e:
                logger.error(f"API error getting historical data (attempt {attempt+1}): {e}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(2 ** attempt)  # Exponential backoff
                
            except Exception as e:
                logger.error(f"Error getting historical data (attempt {attempt+1}): {e}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(2 ** attempt)
    
    def get_account_equity(self, paper_trade=False, default_equity=10000):
        """Calculate total account equity in USDT with paper trading support"""
        # If in paper trading mode, return the simulated equity
        if paper_trade:
            logger.info("Paper trading mode: Using simulated equity")
            return default_equity
            
        # Original API-based code for live trading
        max_retries = 5
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                account = self.client.get_account()
                # Get all balances
                balances = {
                    asset['asset']: float(asset['free']) + float(asset['locked'])
                    for asset in account['balances']
                    if float(asset['free']) > 0 or float(asset['locked']) > 0
                }
                
                # Convert all to USDT value
                usdt_values = {}
                for asset, amount in balances.items():
                    if asset == 'USDT':
                        usdt_values[asset] = amount
                    else:
                        try:
                            ticker = self.client.get_symbol_ticker(symbol=f"{asset}USDT")
                            price = float(ticker['price'])
                            usdt_values[asset] = amount * price
                        except:
                            try:
                                # Try BTC intermediate
                                btc_ticker = self.client.get_symbol_ticker(symbol=f"{asset}BTC")
                                btc_price = float(btc_ticker['price'])
                                btc_usdt = float(self.client.get_symbol_ticker(symbol="BTCUSDT")['price'])
                                usdt_values[asset] = amount * btc_price * btc_usdt
                            except:
                                # Try BUSD intermediate
                                try:
                                    busd_ticker = self.client.get_symbol_ticker(symbol=f"{asset}BUSD")
                                    busd_price = float(busd_ticker['price'])
                                    busd_usdt = float(self.client.get_symbol_ticker(symbol="BUSDUSD T")['price'])
                                    usdt_values[asset] = amount * busd_price * busd_usdt
                                except:
                                    # If still cannot convert, skip this asset
                                    logger.debug(f"Skipping {asset}: no conversion path found")
                                    usdt_values[asset] = 0
                
                return sum(usdt_values.values())
            except Exception as e:
                logger.warning(f"Error getting account equity (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"Failed to get account equity after {max_retries} attempts")
                    return 10000  # Default value if cannot fetch
                    
    def get_tradable_symbols(self, volume_threshold=1000000):
        """Get list of tradable symbols with sufficient volume"""
        try:
            exchange_info = self.client.get_exchange_info()
            tickers = self.client.get_ticker()
            
            # Filter symbols
            symbols = []
            for symbol_info in exchange_info['symbols']:
                # Only USDT pairs for simplicity
                if symbol_info['quoteAsset'] == 'USDT' and symbol_info['status'] == 'TRADING':
                    symbol = symbol_info['symbol']
                    
                    # Get 24h volume
                    ticker = next((t for t in tickers if t['symbol'] == symbol), None)
                    if ticker and float(ticker['volume']) * float(ticker['lastPrice']) > volume_threshold:
                        symbols.append({
                            'symbol': symbol,
                            'baseAsset': symbol_info['baseAsset'],
                            'quoteAsset': symbol_info['quoteAsset'],
                            'volume': float(ticker['volume']) * float(ticker['lastPrice'])
                        })
            
            # Sort by volume
            symbols.sort(key=lambda x: x['volume'], reverse=True)
            logger.info(f"Found {len(symbols)} tradable symbols")
            return symbols[:20]  # Top 20 by volume
        except Exception as e:
            logger.error(f"Error getting tradable symbols: {e}")
            return [{'symbol': 'BTCUSDT'}, {'symbol': 'ETHUSDT'}]  # Default