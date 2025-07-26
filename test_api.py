"""
Simple test script to verify Binance API permissions
"""

from binance.client import Client
from config import BINANCE_CONFIG

def test_binance_api():
    """Test Binance API connection and permissions"""
    try:
        # Create client
        client = Client(BINANCE_CONFIG['api_key'], BINANCE_CONFIG['api_secret'])
        
        print("=== BINANCE API PERMISSION TEST ===")
        
        # Test 1: Basic connection
        print("1. Testing basic connection...")
        server_time = client.get_server_time()
        print(f"   ✅ Server connection successful")
        print(f"   Server time: {server_time['serverTime']}")
        
        # Test 2: Account access (requires API key)
        print("\n2. Testing account access...")
        account_info = client.get_account()
        print(f"   ✅ Account access successful")
        print(f"   Account type: {account_info['accountType']}")
        print(f"   Can trade: {account_info['canTrade']}")
        print(f"   Can withdraw: {account_info['canWithdraw']}")
        print(f"   Can deposit: {account_info['canDeposit']}")
        
        # Test 3: Get balances
        print("\n3. Testing balance access...")
        balances = account_info['balances']
        
        # Show non-zero balances
        non_zero_balances = [b for b in balances if float(b['free']) > 0 or float(b['locked']) > 0]
        if non_zero_balances:
            print(f"   ✅ Found {len(non_zero_balances)} assets with balance:")
            for balance in non_zero_balances[:5]:  # Show first 5
                total = float(balance['free']) + float(balance['locked'])
                if total > 0:
                    print(f"   - {balance['asset']}: {total:.8f} (Free: {balance['free']}, Locked: {balance['locked']})")
        else:
            print("   ⚠️ No assets with balance found")
        
        # Test 4: Test order placement capability (without actually placing)
        print("\n4. Testing trading permissions...")
        try:
            # Try to get exchange info (public)
            exchange_info = client.get_exchange_info()
            print(f"   ✅ Exchange info access successful")
            
            # Try to get open orders (requires trading permission)
            open_orders = client.get_open_orders()
            print(f"   ✅ Open orders access successful")
            print(f"   Current open orders: {len(open_orders)}")
            
        except Exception as e:
            print(f"   ❌ Trading permission test failed: {e}")
            return False
        
        print("\n=== RESULT ===")
        if account_info['canTrade']:
            print("✅ ALL TESTS PASSED - API is ready for live trading!")
            return True
        else:
            print("❌ TRADING DISABLED - Check API permissions on Binance")
            return False
            
    except Exception as e:
        print(f"❌ API Test failed: {e}")
        if "Invalid API-key" in str(e):
            print("\n💡 Possible solutions:")
            print("1. Check API key and secret are correct")
            print("2. Enable 'Spot & Margin Trading' permission")
            print("3. Add IP whitelist or re-enable trading permission")
            print("4. Verify API key hasn't expired")
        return False

if __name__ == "__main__":
    test_binance_api()