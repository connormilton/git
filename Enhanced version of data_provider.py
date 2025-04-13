# Enhanced version of data_provider.py with better debugging

import logging
import decimal
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import concurrent.futures
import time
import os

# Check if polygon library is available
try:
    from polygon import RESTClient
    from polygon.exceptions import NoResultsError, BadResponse
    POLYGON_CLIENT_AVAILABLE = True
    logging.getLogger("DataProvider").info("Polygon library is available")
except ImportError:
    RESTClient = None
    NoResultsError = Exception
    BadResponse = Exception
    POLYGON_CLIENT_AVAILABLE = False
    logging.getLogger("DataProvider").warning("Polygon library is not available, install with: pip install polygon-api-client")

logger = logging.getLogger("TradingBot")

class DataProvider:
    """Handles fetching and processing market data from Polygon.io."""
    def __init__(self, config):
        self.config = config
        self.polygon_api_key = config.get('POLYGON_API_KEY')
        self.polygon_client = None
        
        # Check if polygon library is available
        if not POLYGON_CLIENT_AVAILABLE:
            logger.error("Polygon client library not available. Install with: pip install polygon-api-client")
        elif not self.polygon_api_key:
            logger.error("Polygon API Key missing in configuration")
        else:
            try:
                logger.info(f"Initializing Polygon client with key starting with: {self.polygon_api_key[:4]}...")
                self.polygon_client = RESTClient(self.polygon_api_key, timeout=15)
                logger.info("Polygon.io client initialized successfully")
                
                # Test the connection with a simple API call
                self._test_polygon_connection()
            except Exception as e:
                logger.error(f"Failed to initialize Polygon.io client: {e}")
        
        # Cache for technical indicators
        self.indicator_cache = {}
        self.last_cache_reset = datetime.now(timezone.utc)
        self.cache_ttl = timedelta(minutes=15)  # Cache technical indicators for 15 minutes

    def _test_polygon_connection(self):
        """Test Polygon.io API connection with a simple call"""
        try:
            # Try getting ticker details for a common forex pair
            ticker = "C:EURUSD"
            ticker_details = self.polygon_client.get_ticker_details(ticker=ticker)
            logger.info(f"Polygon connection test successful. Got details for {ticker}: {ticker_details.name}")
            return True
        except Exception as e:
            logger.error(f"Polygon connection test failed: {e}")
            return False

    def reset_cache_if_needed(self):
        """Reset cache if it's older than the TTL"""
        now = datetime.now(timezone.utc)
        if now - self.last_cache_reset > self.cache_ttl:
            self.indicator_cache = {}
            self.last_cache_reset = now
            logger.info("Technical indicator cache reset.")

    def get_polygon_ticker(self, epic):
        """Convert IG epic to Polygon ticker symbol"""
        mapping = self.config.get("EPIC_TO_POLYGON_MAP", {})
        ticker = mapping.get(epic)
        
        if not ticker:
            # Try to infer the ticker from the epic format
            if epic.startswith("CS.D.") and "MINI.IP" in epic:
                # Extract the currency pair from the epic
                parts = epic.split(".")
                if len(parts) > 2:
                    pair = parts[2]
                    if len(pair) == 6:  # EURUSD, GBPUSD, etc.
                        ticker = f"C:{pair}"
                        logger.info(f"Inferred Polygon ticker {ticker} from {epic}")
        
        if not ticker:
            logger.warning(f"No Polygon ticker mapping for IG Epic: {epic}")
            
        return ticker

    def get_historical_data_polygon(self, epic, end_dt=None, days_history=None, timeframe="hour"):
        """Fetch historical price data from Polygon.io for a specific instrument."""
        if not self.polygon_client:
            logger.error("Polygon client not available.")
            return pd.DataFrame()

        polygon_ticker = self.get_polygon_ticker(epic)
        if not polygon_ticker:
            logger.error(f"Cannot fetch history for {epic}: No mapping.")
            return pd.DataFrame()
            
        # Allow overriding the default timeframe
        if not timeframe:
            timeframe = self.config.get("HISTORICAL_DATA_TIMEFRAME", "hour")
            
        if days_history is None:
            days_history = self.config.get("HISTORICAL_DATA_PERIOD_DAYS", 30)
        if end_dt is None:
            end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=days_history)
        end_date_str = end_dt.strftime('%Y-%m-%d')
        start_date_str = start_dt.strftime('%Y-%m-%d')

        logger.info(f"Fetching Polygon history: {polygon_ticker} ({epic}) "
                    f"[{start_date_str} to {end_date_str}], TF: {timeframe}")
        try:
            multiplier = 1
            timespan = timeframe.lower()
            if timespan not in ['minute', 'hour', 'day', 'week', 'month']:
                timespan = 'hour'

            aggs = self.polygon_client.get_aggs(
                ticker=polygon_ticker,
                multiplier=multiplier,
                timespan=timespan,
                from_=start_date_str,
                to=end_date_str,
                adjusted=True,
                limit=50000
            )
            if not aggs:
                logger.warning(f"No results from Polygon for {polygon_ticker}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(aggs)
            df = df.rename(columns={
                'o': 'Open',
                'h': 'High',
                'l': 'Low',
                'c': 'Close',
                'v': 'Volume',
                't': 'Timestamp'
            })
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms', utc=True)
            df = df.set_index('Timestamp')

            for col in ['Open', 'High', 'Low', 'Close']:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: decimal.Decimal(str(x)) if pd.notna(x) else pd.NA)

            if 'Volume' in df.columns:
                df['Volume'] = df['Volume'].apply(lambda x: decimal.Decimal(str(x)) if pd.notna(x) else decimal.Decimal('0'))

            start_dt_aware = pd.Timestamp(start_dt.replace(tzinfo=timezone.utc))
            end_dt_aware = pd.Timestamp(end_dt.replace(tzinfo=timezone.utc))
            df = df[(df.index >= start_dt_aware) & (df.index <= end_dt_aware)]

            logger.info(f"Fetched {len(df)} bars for {polygon_ticker} ({epic}) from Polygon.")
            
            # Log first/last data points for debugging
            if not df.empty:
                logger.debug(f"First data point: {df.iloc[0].to_dict()}")
                logger.debug(f"Last data point: {df.iloc[-1].to_dict()}")
                
            return df
        except NoResultsError:
            logger.warning(f"No results on Polygon for {polygon_ticker}.")
            return pd.DataFrame()
        except BadResponse as br:
            logger.error(f"Polygon API Error for {polygon_ticker}: Status {br.status}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Polygon fetch error for {polygon_ticker}: {e}", exc_info=True)
            return pd.DataFrame()

    # Rest of the class implementation remains unchanged...
    # The apply_technical_indicators method and other methods would follow here