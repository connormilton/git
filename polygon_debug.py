# polygon_debug.py
# Simple script to test Polygon.io API connectivity

import os
import sys
from dotenv import load_dotenv
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PolygonDebug")

# Load environment variables
load_dotenv()
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')

logger.info(f"Using Polygon API Key: {POLYGON_API_KEY[:4]}...{POLYGON_API_KEY[-4:] if POLYGON_API_KEY else 'None'}")

# Check if polygon library is installed
try:
    from polygon import RESTClient
    from polygon.exceptions import NoResultsError, BadResponse
    logger.info("✅ Polygon library is installed")
except ImportError:
    logger.error("❌ Polygon library is not installed. Please install with: pip install polygon-api-client")
    sys.exit(1)

# Test API connection
def test_polygon_connection():
    if not POLYGON_API_KEY:
        logger.error("❌ No Polygon API key found in .env file")
        return False
    
    try:
        client = RESTClient(POLYGON_API_KEY, timeout=15)
        logger.info("✅ Successfully created Polygon REST client")
        
        # Test a simple API call
        ticker = "C:EURUSD"
        logger.info(f"Testing API call for ticker: {ticker}")
        
        # Get ticker details
        try:
            ticker_details = client.get_ticker_details(ticker=ticker)
            logger.info(f"✅ Successfully got ticker details for {ticker}")
            logger.info(f"Ticker name: {ticker_details.name}")
            logger.info(f"Ticker type: {ticker_details.type}")
            logger.info(f"Base currency: {ticker_details.base_currency_symbol}")
            logger.info(f"Currency symbol: {ticker_details.currency_symbol}")
        except Exception as e:
            logger.error(f"❌ Failed to get ticker details: {e}")
        
        # Test getting historical data
        try:
            from datetime import datetime, timedelta
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
            
            logger.info(f"Getting aggregates for {ticker} from {start_date} to {end_date}")
            aggs = client.get_aggs(
                ticker=ticker,
                multiplier=1,
                timespan="hour",
                from_=start_date,
                to=end_date,
                limit=100
            )
            
            if aggs:
                logger.info(f"✅ Successfully got {len(aggs)} aggregates")
                logger.info(f"First data point: {aggs[0]}")
                return True
            else:
                logger.warning("⚠️ No aggregates returned, but API call succeeded")
                return True
        except Exception as e:
            logger.error(f"❌ Failed to get aggregates: {e}")
            return False
    
    except Exception as e:
        logger.error(f"❌ Failed to create Polygon client: {e}")
        return False

if __name__ == "__main__":
    success = test_polygon_connection()
    if success:
        logger.info("🎉 Polygon.io API connection test successful!")
    else:
        logger.error("❌ Polygon.io API connection test failed")