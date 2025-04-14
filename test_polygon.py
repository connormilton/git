# test_polygon.py - Simple test script for Polygon.io API

import sys
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

# Load .env file
print("Loading environment variables...")
load_dotenv()
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
print(f"Polygon API Key found: {'Yes' if POLYGON_API_KEY else 'No'}")
if POLYGON_API_KEY:
    # Show first and last 4 characters for validation without showing full key
    print(f"API Key: {POLYGON_API_KEY[:4]}...{POLYGON_API_KEY[-4:]}")

# Try to import polygon
print("\nTesting polygon-api-client import...")
try:
    import polygon
    # Fix the version display
    try:
        pkg_version = polygon.__version__
    except AttributeError:
        try:
            import pkg_resources
            pkg_version = pkg_resources.get_distribution("polygon-api-client").version
        except:
            pkg_version = "unknown"
    
    print(f"Successfully imported polygon base package (version: {pkg_version})")
    print(f"Polygon package located at: {polygon.__file__}")
    
    try:
        from polygon import RESTClient
        print("Successfully imported polygon.RESTClient")
    except ImportError as e:
        print(f"Failed to import polygon.RESTClient: {e}")
        
    # Try a simple API call
    print("\nTesting API connection...")
    try:
        client = RESTClient(POLYGON_API_KEY)
        print("Successfully created RESTClient")
        
        # Test with forex ticker - EURUSD
        ticker = "C:EURUSD"
        print(f"Testing with ticker: {ticker}")
        
        # Get ticker details
        ticker_details = client.get_ticker_details(ticker=ticker)
        print(f"Successfully retrieved ticker details for {ticker}")
        print(f"  Name: {ticker_details.name}")
        print(f"  Type: {ticker_details.type}")
        print(f"  Market: {ticker_details.market}")
        
        # Get recent aggs - Use dates from the past (2023) to ensure data availability
        # The script was using future dates (2025) which explains the 0 results
        end_date = "2023-12-31"
        start_date = "2023-12-25"
        
        print(f"Getting aggs from {start_date} to {end_date}")
        aggs = client.get_aggs(
            ticker=ticker,
            multiplier=1,
            timespan="day",
            from_=start_date,
            to=end_date,
            limit=10
        )
        
        print(f"Successfully retrieved {len(aggs)} aggs")
        if aggs:
            print("  First agg:")
            print(f"    Open: {aggs[0].open}")
            print(f"    High: {aggs[0].high}")
            print(f"    Low: {aggs[0].low}")
            print(f"    Close: {aggs[0].close}")
            print(f"    Volume: {aggs[0].volume}")
            print(f"    Timestamp: {aggs[0].timestamp}")
            
        print("\n✅ Polygon.io API connection test successful!")
        
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        
except ImportError as e:
    print(f"Failed to import polygon base package: {e}")
    print("Make sure you've installed the package with: pip install polygon-api-client")