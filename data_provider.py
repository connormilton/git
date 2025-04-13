# updated_polygon_provider.py - Enhanced implementation focusing on real-time forex data

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
    POLYGON_CLIENT_AVAILABLE = True
    logging.getLogger("DataProvider").info("Polygon library is available")
except ImportError:
    RESTClient = None
    POLYGON_CLIENT_AVAILABLE = False
    logging.getLogger("DataProvider").warning("Polygon library is not available, install with: pip install polygon-api-client")

logger = logging.getLogger("TradingBot")

class DataProvider:
    """Handles fetching and processing market data from Polygon.io with focus on real-time data."""
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
                self.polygon_client = RESTClient(self.polygon_api_key)  # Remove the timeout parameter
                logger.info("Polygon.io client initialized successfully")
                
                # Test the connection with a simple API call
                self._test_polygon_connection()
            except Exception as e:
                logger.error(f"Failed to initialize Polygon.io client: {e}")
        
        # Cache for technical indicators
        self.indicator_cache = {}
        self.last_cache_reset = datetime.now(timezone.utc)
        self.cache_ttl = timedelta(minutes=15)  # Cache technical indicators for 15 minutes
        
        # Cache for real-time quotes
        self.quote_cache = {}
        self.last_quote_reset = datetime.now(timezone.utc)
        self.quote_ttl = timedelta(seconds=30)  # Cache quotes for 30 seconds

    def _test_polygon_connection(self):
        """Test Polygon.io API connection with a simple call"""
        try:
            # Try getting a quote for a common forex pair
            ticker = "C:EURUSD"
            last_quote = self.polygon_client.get_last_quote(ticker)
            if last_quote:
                logger.info(f"Polygon connection test successful. Got quote for {ticker}: Ask={last_quote.ask_price}, Bid={last_quote.bid_price}")
                return True
            else:
                logger.warning(f"Polygon connection test: No quote data for {ticker}")
                return False
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
            
        if now - self.last_quote_reset > self.quote_ttl:
            self.quote_cache = {}
            self.last_quote_reset = now
            logger.debug("Quote cache reset.")

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

    def get_latest_quote(self, epic):
        """Get the latest quote for an instrument from Polygon."""
        if not self.polygon_client:
            logger.error("Polygon client not available.")
            return None
            
        polygon_ticker = self.get_polygon_ticker(epic)
        if not polygon_ticker:
            logger.error(f"Cannot fetch quote for {epic}: No mapping.")
            return None
            
        # Check cache first
        self.reset_cache_if_needed()
        if polygon_ticker in self.quote_cache:
            logger.debug(f"Using cached quote for {polygon_ticker}")
            return self.quote_cache[polygon_ticker]
            
        try:
            logger.debug(f"Fetching latest quote for {polygon_ticker}")
            last_quote = self.polygon_client.get_last_quote(polygon_ticker)
            
            if not last_quote:
                logger.warning(f"No quote data available for {polygon_ticker}")
                return None
                
            # Format the quote data
            quote_data = {
                'ask': decimal.Decimal(str(last_quote.ask_price)),
                'bid': decimal.Decimal(str(last_quote.bid_price)),
                'timestamp': last_quote.timestamp / 1000  # Convert milliseconds to seconds
            }
            
            # Cache the quote
            self.quote_cache[polygon_ticker] = quote_data
            
            return quote_data
        except Exception as e:
            logger.error(f"Error fetching quote for {polygon_ticker}: {e}")
            return None

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
            
        # Make sure we're using actual historical dates, not future dates
        end_dt = min(end_dt, datetime.now(timezone.utc))
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
                
                # If no results, use real-time quotes to generate at least a single data point
                quote = self.get_latest_quote(epic)
                if quote:
                    # Create a small dataframe with just the current quote
                    now = datetime.now(timezone.utc)
                    df = pd.DataFrame([{
                        'Timestamp': now,
                        'Open': quote['bid'],
                        'High': quote['ask'], 
                        'Low': quote['bid'],
                        'Close': (quote['ask'] + quote['bid']) / 2,
                        'Volume': decimal.Decimal('100')  # Placeholder
                    }])
                    df = df.set_index('Timestamp')
                    logger.info(f"Created minimal dataframe from current quote for {polygon_ticker}")
                    return df
                else:
                    return pd.DataFrame()

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
            return df
        except Exception as e:
            logger.error(f"Polygon fetch error for {polygon_ticker}: {e}", exc_info=True)
            
            # Try to create a minimal dataframe from real-time quote as fallback
            quote = self.get_latest_quote(epic)
            if quote:
                now = datetime.now(timezone.utc)
                df = pd.DataFrame([{
                    'Timestamp': now,
                    'Open': quote['bid'],
                    'High': quote['ask'], 
                    'Low': quote['bid'],
                    'Close': (quote['ask'] + quote['bid']) / 2,
                    'Volume': decimal.Decimal('100')  # Placeholder
                }])
                df = df.set_index('Timestamp')
                logger.info(f"Created fallback dataframe from current quote for {polygon_ticker}")
                return df
            
            return pd.DataFrame()

    def apply_technical_indicators(self, df):
        """Calculate and add technical indicators to the dataframe."""
        if df.empty:
            return df

        # Convert Decimal columns to float for indicator calculations
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = df[col].astype(float)

        # Simple Moving Averages
        if len(df) >= 20:
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
        else:
            df['SMA_20'] = df['Close']
            
        if len(df) >= 50:
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
        else:
            df['SMA_50'] = df['Close']
            
        if len(df) >= 200:
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
        else:
            df['SMA_200'] = df['Close']

        # Exponential Moving Averages
        if len(df) >= 12:
            df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        else:
            df['EMA_12'] = df['Close']
            
        if len(df) >= 26:
            df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        else:
            df['EMA_26'] = df['Close']

        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        if len(df) >= 9:
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        else:
            df['MACD_Signal'] = df['MACD']
            
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        # Relative Strength Index (RSI)
        if len(df) >= 14:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
        else:
            df['RSI'] = 50  # Neutral value for short data sets

        # Bollinger Bands
        if len(df) >= 20:
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            df['BB_StdDev'] = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + (df['BB_StdDev'] * 2)
            df['BB_Lower'] = df['BB_Middle'] - (df['BB_StdDev'] * 2)
            
            # Bollinger Band Width
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        else:
            df['BB_Middle'] = df['Close']
            df['BB_StdDev'] = 0
            df['BB_Upper'] = df['Close'] * 1.02  # Approximate 2% band
            df['BB_Lower'] = df['Close'] * 0.98  # Approximate 2% band
            df['BB_Width'] = 0.04  # Placeholder for short data sets

        # Average True Range (ATR)
        if len(df) >= 2:  # Need at least 2 rows for ATR
            tr1 = abs(df['High'] - df['Low'])
            tr2 = abs(df['High'] - df['Close'].shift())
            tr3 = abs(df['Low'] - df['Close'].shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            if len(df) >= 14:
                df['ATR_14'] = tr.rolling(window=14).mean()
            else:
                df['ATR_14'] = tr.mean()
        else:
            # For single row, use a placeholder based on High-Low range
            df['ATR_14'] = (df['High'] - df['Low']) * 0.5

        # Stochastic Oscillator
        if len(df) >= 14:
            df['Stoch_K'] = 100 * ((df['Close'] - df['Low'].rolling(window=14).min()) / 
                                    (df['High'].rolling(window=14).max() - df['Low'].rolling(window=14).min()))
            
            if len(df) >= 3:
                df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
            else:
                df['Stoch_D'] = df['Stoch_K']
        else:
            # Placeholder values for short data sets
            df['Stoch_K'] = 50
            df['Stoch_D'] = 50

        # Volatility indicators
        df['Daily_Range'] = df['High'] - df['Low']
        df['Range_Percent'] = (df['Daily_Range'] / df['Close']) * 100
        
        if len(df) >= 20:
            df['Volatility_20'] = df['Range_Percent'].rolling(window=20).std()
        else:
            df['Volatility_20'] = df['Range_Percent'].std() if len(df) > 1 else 1.0

        # Simple trend identification
        df['Trend'] = 0  # 0 = no trend, 1 = uptrend, -1 = downtrend
        
        # Set trend based on available MAs
        if len(df) >= 50:
            df.loc[(df['SMA_20'] > df['SMA_50']), 'Trend'] = 1
            df.loc[(df['SMA_20'] < df['SMA_50']), 'Trend'] = -1
        elif len(df) >= 20:
            # For shorter datasets, compare current price to SMA
            df.loc[(df['Close'] > df['SMA_20']), 'Trend'] = 1
            df.loc[(df['Close'] < df['SMA_20']), 'Trend'] = -1
        else:
            # For very short datasets, compare to previous value if available
            if len(df) > 1:
                df['Trend'] = np.sign(df['Close'].diff())
            else:
                df['Trend'] = 0  # No trend for single point

        # ADX (Simplified calculation for short datasets)
        if len(df) >= 14:
            # Simplified ADX for short datasets
            plus_dm = df['High'].diff()
            minus_dm = df['Low'].diff()
            plus_dm = plus_dm.where(plus_dm > 0, 0)
            minus_dm = minus_dm.where(minus_dm < 0, 0).abs()
            
            atr = df['ATR_14']
            
            # Calculate DI+ and DI-
            plus_di = 100 * plus_dm.rolling(window=14).mean() / atr
            minus_di = 100 * minus_dm.rolling(window=14).mean() / atr
            
            # Calculate DX and ADX
            dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan).fillna(0)
            df['ADX'] = dx.ewm(span=14, adjust=False).mean()
            
            df['Plus_DI'] = plus_di
            df['Minus_DI'] = minus_di
        else:
            # Placeholder values for short datasets
            df['ADX'] = 15  # Low ADX value indicating weak trend
            df['Plus_DI'] = 25
            df['Minus_DI'] = 25

        # Convert indicators back to Decimal for consistency
        for col in df.columns:
            if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Timestamp']:
                df[col] = df[col].apply(lambda x: decimal.Decimal(str(x)) if pd.notna(x) else pd.NA)

        return df

    def get_technical_data(self, epic, days=30):
        """Get OHLCV data with technical indicators for an instrument."""
        # Check cache first
        self.reset_cache_if_needed()
        cache_key = f"{epic}_technical_{days}"
        if cache_key in self.indicator_cache:
            logger.debug(f"Using cached technical data for {epic}")
            return self.indicator_cache[cache_key]
        
        # Fetch historical data
        ohlc_df = self.get_historical_data_polygon(epic, days_history=days)
        if ohlc_df.empty:
            logger.warning(f"No historical data for {epic}, can't calculate technicals.")
            return pd.DataFrame()
            
        # Apply technical indicators
        tech_df = self.apply_technical_indicators(ohlc_df)
        
        # Store in cache
        self.indicator_cache[cache_key] = tech_df.copy()
        
        return tech_df

    def get_latest_technicals(self, epic):
        """Get the latest technical indicators for an instrument."""
        tech_df = self.get_technical_data(epic)
        if tech_df.empty:
            return {}

        # Get the most recent row of data (latest technical values)
        latest = tech_df.iloc[-1].to_dict()
        technicals = {}
        
        # Format decimal values and filter relevant indicators
        for key, value in latest.items():
            # Skip OHLCV data and include only calculated indicators
            if key not in ['Open', 'High', 'Low', 'Close', 'Volume'] and pd.notna(value):
                if isinstance(value, decimal.Decimal):
                    technicals[key] = float(value)
                else:
                    technicals[key] = value
        
        return technicals

    def get_market_regime(self, epic):
        """Determine the current market regime (trending, ranging, volatile)."""
        tech_df = self.get_technical_data(epic)
        if tech_df.empty:
            return "unknown"
            
        # Get the latest values
        latest = tech_df.iloc[-1]
        
        # Check for high volatility
        volatility = latest.get('Volatility_20', 0)
        if isinstance(volatility, decimal.Decimal):
            volatility = float(volatility)
            
        if volatility > 1.5:  # 1.5% average daily range is considered volatile
            regime = "volatile"
        # Check for trending market
        elif float(latest.get('ADX', 0)) > 25:  # ADX > 25 indicates a trend
            if float(latest.get('Trend', 0)) > 0:
                regime = "uptrend"
            else:
                regime = "downtrend"
        # Otherwise ranging/sideways
        else:
            regime = "ranging"
            
        return regime

    def get_multi_timeframe_analysis(self, epic):
        """Analyze multiple timeframes for consistent signals."""
        timeframes = [
            {"name": "daily", "days": 90, "timeframe": "day"},
            {"name": "hourly", "days": 14, "timeframe": "hour"},
            {"name": "minute", "days": 2, "timeframe": "minute"}
        ]
        
        results = {}
        for tf in timeframes:
            df = self.get_historical_data_polygon(
                epic, 
                days_history=tf["days"],
                timeframe=tf["timeframe"]
            )
            if not df.empty:
                tech_df = self.apply_technical_indicators(df)
                if not tech_df.empty:
                    latest = tech_df.iloc[-1]
                    
                    # Determine trend
                    trend = "neutral"
                    trend_val = latest.get('Trend', 0)
                    if isinstance(trend_val, decimal.Decimal):
                        trend_val = float(trend_val)
                        
                    if trend_val > 0:
                        trend = "bullish"
                    elif trend_val < 0:
                        trend = "bearish"
                        
                    # Determine momentum
                    momentum = "neutral"
                    rsi = latest.get('RSI', 50)
                    if isinstance(rsi, decimal.Decimal):
                        rsi = float(rsi)
                        
                    if rsi > 60:
                        momentum = "bullish"
                    elif rsi < 40:
                        momentum = "bearish"
                    
                    # Determine volatility
                    volatility = "normal"
                    vol_val = latest.get('Volatility_20', 1)
                    if isinstance(vol_val, decimal.Decimal):
                        vol_val = float(vol_val)
                        
                    if vol_val > 1.5:
                        volatility = "high"
                    elif vol_val < 0.5:
                        volatility = "low"
                    
                    # Get BB width for analysis
                    bb_width = 0
                    if 'BB_Width' in tech_df.columns:
                        bb_width_val = latest.get('BB_Width', 0)
                        if isinstance(bb_width_val, decimal.Decimal):
                            bb_width = float(bb_width_val) * 100
                    else:
                        # Calculate if not present
                        bb_upper = latest.get('BB_Upper', 0)
                        bb_lower = latest.get('BB_Lower', 0)
                        bb_middle = latest.get('BB_Middle', 1)
                        
                        if all(isinstance(x, decimal.Decimal) for x in [bb_upper, bb_lower, bb_middle]) and bb_middle > 0:
                            bb_width = float((bb_upper - bb_lower) / bb_middle) * 100
                    
                    # Get ADX for trend strength
                    adx = latest.get('ADX', 0)
                    if isinstance(adx, decimal.Decimal):
                        adx = float(adx)
                    
                    results[tf["name"]] = {
                        "trend": trend,
                        "momentum": momentum,
                        "volatility": volatility,
                        "rsi": float(rsi),
                        "adx": adx,
                        "bb_width": bb_width
                    }
        
        return results

    def get_correlation_matrix(self, epics):
        """Calculate correlation matrix between multiple instruments."""
        if not epics or len(epics) < 2:
            return {}
            
        # Fetch close prices for all epics
        price_data = {}
        for epic in epics:
            df = self.get_historical_data_polygon(epic, days_history=30)
            if not df.empty:
                price_data[epic] = df['Close'].astype(float)
        
        # If we have at least 2 instruments with data
        if len(price_data) >= 2:
            # Create a DataFrame with all close prices
            price_df = pd.DataFrame(price_data)
            # Calculate correlation matrix
            corr_matrix = price_df.corr().round(2).to_dict()
            return corr_matrix
        else:
            return {}

    def get_snapshot_with_technicals(self, epics):
        """Get market snapshots with technical indicators for multiple instruments."""
        if not isinstance(epics, list):
            epics = [epics]
            
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Create futures for each epic
            future_to_epic = {
                executor.submit(self.get_technical_data, epic): epic
                for epic in epics
            }
            
            for future in concurrent.futures.as_completed(future_to_epic):
                epic = future_to_epic[future]
                try:
                    tech_df = future.result()
                    
                    # If we don't have technical data, try to get at least current quotes
                    if tech_df.empty:
                        quote = self.get_latest_quote(epic)
                        if quote:
                            mid_price = (quote['ask'] + quote['bid']) / 2
                            results[epic] = {
                                'epic': epic,
                                'price': {
                                    'bid': float(quote['bid']),
                                    'ask': float(quote['ask']),
                                    'mid': float(mid_price)
                                },
                                'indicators': {
                                    'trend': 0,  # Neutral
                                    'rsi': 50,   # Neutral
                                    'volatility': 1.0  # Average
                                },
                                'regime': 'unknown',
                                'timestamp': datetime.now(timezone.utc).isoformat()
                            }
                        continue
                        
                    if not tech_df.empty:
                        # Get the most recent data point
                        latest = tech_df.iloc[-1].to_dict()
                        
                        # Get latest quote if available for more accurate current price
                        current_price = {}
                        quote = self.get_latest_quote(epic)
                        if quote:
                            mid_price = (quote['ask'] + quote['bid']) / 2
                            current_price = {
                                'bid': float(quote['bid']),
                                'ask': float(quote['ask']),
                                'mid': float(mid_price),
                                'open': float(latest.get('Open', mid_price)),
                                'high': float(latest.get('High', quote['ask'])),
                                'low': float(latest.get('Low', quote['bid'])),
                                'close': float(latest.get('Close', mid_price)),
                                'volume': float(latest.get('Volume', 0))
                            }
                        else:
                            # Fallback to historical data
                            current_price = {
                                'open': float(latest.get('Open', 0)),
                                'high': float(latest.get('High', 0)),
                                'low': float(latest.get('Low', 0)),
                                'close': float(latest.get('Close', 0)),
                                'volume': float(latest.get('Volume', 0))
                            }
                        
                        # Create a market snapshot with technicals
                        snapshot = {
                            'epic': epic,
                            'price': current_price,
                            'indicators': {
                                'trend': int(latest.get('Trend', 0)),
                                'rsi': float(latest.get('RSI', 50)),
                                'macd': float(latest.get('MACD', 0)),
                                'macd_signal': float(latest.get('MACD_Signal', 0)),
                                'adx': float(latest.get('ADX', 0)),
                                'bb_upper': float(latest.get('BB_Upper', 0)),
                                'bb_middle': float(latest.get('BB_Middle', 0)),
                                'bb_lower': float(latest.get('BB_Lower', 0)),
                                'atr': float(latest.get('ATR_14', 0)),
                                'stoch_k': float(latest.get('Stoch_K', 0)),
                                'stoch_d': float(latest.get('Stoch_D', 0)),
                                'volatility': float(latest.get('Volatility_20', 0))
                            },
                            'regime': self.get_market_regime(epic),
                            'timestamp': datetime.now(timezone.utc).isoformat()
                        }
                        results[epic] = snapshot
                except Exception as e:
                    logger.error(f"Error getting technicals for {epic}: {e}")
        
        return results

    def get_news_sentiment(self, epic):
        """Get news and basic sentiment for an instrument."""
        # Since Polygon doesn't have news API in most plans, 
        # we return minimal placeholder data
        return {
            "headlines": [],
            "sentiment_score": 0.0,
            "article_count": 0
        }