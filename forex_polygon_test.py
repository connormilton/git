# forex_polygon_test.py - Test script specifically for Polygon.io forex data

import os
import sys
from dotenv import load_dotenv
import time
from datetime import datetime, timedelta

print("=== Polygon.io Forex Data Test ===")
print(f"Current time (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
print("Testing if forex data is available with your API key")

# Load environment variables
print("\nLoading API key...")
load_dotenv()
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')

if not POLYGON_API_KEY:
    print("❌ No POLYGON_API_KEY found in .env file")
    sys.exit(1)

print(f"API Key found: {POLYGON_API_KEY[:4]}...{POLYGON_API_KEY[-4:]}")

# Import the polygon library
try:
    import requests
    import polygon
    from polygon import RESTClient
    print("✅ Successfully imported polygon library")
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("Try installing required packages: pip install polygon-api-client requests")
    sys.exit(1)

# Test various forex endpoints to diagnose the issue
def test_forex_endpoints():
    client = RESTClient(POLYGON_API_KEY)
    
    # Test currency pairs
    forex_pairs = ["EUR_USD", "GBP_USD", "USD_JPY"]
    
    # 1. Test direct REST API call with requests (bypassing client)
    print("\n1. Testing direct REST API call...")
    try:
        today = datetime.today().strftime('%Y-%m-%d')
        yesterday = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
        url = f"https://api.polygon.io/v2/aggs/ticker/C:EURUSD/range/1/hour/{yesterday}/{today}?apiKey={POLYGON_API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Direct REST call successful. Status: {response.status_code}")
            print(f"  Results count: {len(data.get('results', []))}")
            print(f"  First result: {data.get('results', [])[0] if data.get('results') else 'No results'}")
        else:
            print(f"❌ Direct REST call failed. Status: {response.status_code}")
            print(f"  Error message: {response.text}")
    except Exception as e:
        print(f"❌ Exception during direct REST call: {e}")
    
    # 2. Test aggregates with polygon client
    print("\n2. Testing polygon client aggregates call...")
    for pair in forex_pairs:
        try:
            print(f"\nChecking pair: {pair}")
            # Format ticker for polygon
            ticker = f"C:{pair.replace('_', '')}"
            
            # Get today and 5 days ago
            end_date = datetime.today().strftime('%Y-%m-%d')
            start_date = (datetime.today() - timedelta(days=5)).strftime('%Y-%m-%d')
            
            print(f"  Requesting data from {start_date} to {end_date} for {ticker}")
            
            # Request data with explicit timeouts
            aggs = client.get_aggs(
                ticker=ticker,
                multiplier=1,
                timespan="hour",
                from_=start_date,
                to=end_date,
                limit=10
            )
            
            if aggs:
                print(f"  ✅ Successfully retrieved {len(aggs)} aggregates for {ticker}")
                if len(aggs) > 0:
                    print(f"  First aggregate: Open={aggs[0].open}, Close={aggs[0].close}, Timestamp={aggs[0].timestamp}")
            else:
                print(f"  ⚠️ No aggregates returned for {ticker}, but API call succeeded")
                
        except Exception as e:
            print(f"  ❌ Error retrieving data for {pair}: {e}")
            
        # Wait a bit between requests to avoid rate limiting
        time.sleep(1)
    
    # 3. Test forex specific endpoints
    print("\n3. Testing forex specific endpoints...")
    try:
        print("  Trying to get available currency pairs...")
        forex_tickers = client.get_grouped_daily("forex", "2023-01-05", "2023-01-05")
        print(f"  ✅ Successfully retrieved {len(forex_tickers) if forex_tickers else 0} forex tickers")
    except Exception as e:
        print(f"  ❌ Error retrieving forex tickers: {e}")
        
    # 4. Test real-time quotes if available
    print("\n4. Testing real-time quotes (if available in your subscription)...")
    try:
        for pair in forex_pairs[:1]:  # Just test the first pair
            ticker = f"C:{pair.replace('_', '')}"
            print(f"  Requesting last quote for {ticker}...")
            last_quote = client.get_last_quote(ticker)
            if last_quote:
                print(f"  ✅ Last quote for {ticker}: Ask={last_quote.ask_price}, Bid={last_quote.bid_price}")
            else:
                print(f"  ⚠️ No last quote available for {ticker}")
    except Exception as e:
        print(f"  ❌ Error retrieving last quote: {e}")

# Run the tests
try:
    test_forex_endpoints()
    print("\n✅ Test script completed. Check the results above to diagnose issues.")
except Exception as e:
    print(f"\n❌ Fatal error running test script: {e}")