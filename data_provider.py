# data_provider.py

import logging
import decimal
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import concurrent.futures
import time

try:
    from polygon import RESTClient
    from polygon.exceptions import NoResultsError, BadResponse
    POLYGON_CLIENT_AVAILABLE = True
except ImportError:
    RESTClient = None
    NoResultsError = Exception
    BadResponse = Exception
    POLYGON_CLIENT_AVAILABLE = False

logger = logging.getLogger("TradingBot")

class DataProvider:
    """Handles fetching and processing market data from Polygon.io."""
    def __init__(self, config):
        self.config = config
        self.polygon_api_key = config.get('POLYGON_API_KEY')
        self.polygon_client = None
        if POLYGON_CLIENT_AVAILABLE and self.polygon_api_key:
            try:
                self.polygon_client = RESTClient(self.polygon_api_key, timeout=15)
                logger.info("Polygon.io client initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize Polygon.io client: {e}")
        elif not self.polygon_api_key:
            logger.error("Polygon API Key missing.")
        
        # Cache for technical indicators
        self.indicator_cache = {}
        self.last_cache_reset = datetime.now(timezone.utc)
        self.cache_ttl = timedelta(minutes=15)  # Cache technical indicators for 15 minutes

    def reset_cache_if_needed(self):
        """Reset cache if it's older than the TTL"""
        now = datetime.now(timezone.utc)
        if now - self.last_cache_reset > self.cache_ttl:
            self.indicator_cache = {}
            self.last_cache_reset = now
            logger.info("Technical indicator cache reset.")

    def get_polygon_ticker(self, epic):
        mapping = self.config.get("EPIC_TO_POLYGON_MAP", {})
        ticker = mapping.get(epic)
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
        except NoResultsError:
            logger.warning(f"No results on Polygon for {polygon_ticker}.")
            return pd.DataFrame()
        except BadResponse as br:
            logger.error(f"Polygon API Error for {polygon_ticker}: Status {br.status}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Polygon fetch error for {polygon_ticker}: {e}", exc_info=True)
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
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()

        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        # Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_StdDev'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (df['BB_StdDev'] * 2)
        df['BB_Lower'] = df['BB_Middle'] - (df['BB_StdDev'] * 2)

        # Average True Range (ATR)
        tr1 = abs(df['High'] - df['Low'])
        tr2 = abs(df['High'] - df['Close'].shift())
        tr3 = abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR_14'] = tr.rolling(window=14).mean()

        # Stochastic Oscillator
        df['Stoch_K'] = 100 * ((df['Close'] - df['Low'].rolling(window=14).min()) / 
                                (df['High'].rolling(window=14).max() - df['Low'].rolling(window=14).min()))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()

        # Volatility indicators
        df['Daily_Range'] = df['High'] - df['Low']
        df['Range_Percent'] = (df['Daily_Range'] / df['Close']) * 100
        df['Volatility_20'] = df['Range_Percent'].rolling(window=20).std()

        # Directional movement indicators
        plus_dm = df['High'].diff()
        minus_dm = df['Low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = abs(minus_dm)
        
        # Filter conditions where high diff is greater than low diff and vice versa
        plus_dm_mask = (df['High'].diff() > df['Low'].diff())
        minus_dm_mask = (df['Low'].diff() < df['High'].diff())
        
        plus_dm[~plus_dm_mask] = 0
        minus_dm[~minus_dm_mask] = 0
        
        # Calculate TR
        tr = pd.DataFrame({
            'tr1': df['High'] - df['Low'],
            'tr2': abs(df['High'] - df['Close'].shift(1)),
            'tr3': abs(df['Low'] - df['Close'].shift(1))
        }).max(axis=1)
        
        # Calculate smoothed averages
        smoothing = 14
        # ATR
        atr = tr.ewm(alpha=1/smoothing, adjust=False).mean()
        # +DM & -DM
        plus_di = 100 * (plus_dm.ewm(alpha=1/smoothing, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/smoothing, adjust=False).mean() / atr)
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df['ADX'] = dx.ewm(alpha=1/smoothing, adjust=False).mean()
        df['Plus_DI'] = plus_di
        df['Minus_DI'] = minus_di

        # Simple trend identification
        df['Trend'] = 0  # 0 = no trend, 1 = uptrend, -1 = downtrend
        df.loc[(df['SMA_20'] > df['SMA_50']) & (df['SMA_50'] > df['SMA_200']), 'Trend'] = 1
        df.loc[(df['SMA_20'] < df['SMA_50']) & (df['SMA_50'] < df['SMA_200']), 'Trend'] = -1

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
        if float(latest.get('Volatility_20', 0)) > 1.5:  # 1.5% average daily range is considered volatile
            regime = "volatile"
        # Check for trending market
        elif abs(float(latest.get('ADX', 0))) > 25:  # ADX > 25 indicates a trend
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
                    if float(latest.get('Trend', 0)) > 0:
                        trend = "bullish"
                    elif float(latest.get('Trend', 0)) < 0:
                        trend = "bearish"
                        
                    # Determine momentum
                    momentum = "neutral"
                    rsi = float(latest.get('RSI', 50))
                    if rsi > 60:
                        momentum = "bullish"
                    elif rsi < 40:
                        momentum = "bearish"
                    
                    # Determine volatility
                    volatility = "normal"
                    vol_val = float(latest.get('Volatility_20', 1))
                    if vol_val > 1.5:
                        volatility = "high"
                    elif vol_val < 0.5:
                        volatility = "low"
                    
                    results[tf["name"]] = {
                        "trend": trend,
                        "momentum": momentum,
                        "volatility": volatility,
                        "rsi": rsi,
                        "adx": float(latest.get('ADX', 0)),
                        "bb_width": float((latest.get('BB_Upper', 0) - latest.get('BB_Lower', 0)) / latest.get('BB_Middle', 1)) * 100
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

    def fetch_market_news(self, ticker_symbols, limit=10):
        """Fetch market news for specific tickers from Polygon."""
        if not self.polygon_client:
            return []
        
        if isinstance(ticker_symbols, str):
            ticker_symbols = [ticker_symbols]
        
        news_items = []
        try:
            # Get news from Polygon
            for ticker in ticker_symbols:
                polygon_ticker = self.get_polygon_ticker(ticker)
                if not polygon_ticker:
                    continue
                    
                results = self.polygon_client.get_ticker_news(
                    ticker=polygon_ticker, 
                    limit=limit
                )
                
                for article in results:
                    news_items.append({
                        'title': article.title,
                        'publisher': article.publisher.name if article.publisher else 'Unknown',
                        'published_utc': article.published_utc,
                        'ticker': ticker,
                        'url': article.article_url,
                        'keywords': article.keywords if hasattr(article, 'keywords') else []
                    })
            
            return news_items
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return []

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
                    if not tech_df.empty:
                        # Get the most recent data point
                        latest = tech_df.iloc[-1].to_dict()
                        
                        # Create a market snapshot with technicals
                        snapshot = {
                            'epic': epic,
                            'price': {
                                'open': float(latest.get('Open', 0)),
                                'high': float(latest.get('High', 0)),
                                'low': float(latest.get('Low', 0)),
                                'close': float(latest.get('Close', 0)),
                                'volume': float(latest.get('Volume', 0))
                            },
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
        try:
            polygon_ticker = self.get_polygon_ticker(epic)
            if not polygon_ticker:
                return {"headlines": [], "sentiment_score": 0.0}
                
            news = self.fetch_market_news(epic, limit=5)
            headlines = [item["title"] for item in news]
            
            # Implement a basic sentiment calculation
            # In a real implementation, you might use a more sophisticated model or external API
            positive_keywords = ['surge', 'jump', 'gain', 'rise', 'up', 'high', 'growth', 'profit', 'bullish', 'positive']
            negative_keywords = ['drop', 'fall', 'loss', 'decline', 'down', 'low', 'recession', 'bearish', 'negative']
            
            sentiment_score = 0.0
            if headlines:
                title_text = ' '.join(headlines).lower()
                positive_count = sum(1 for word in positive_keywords if word in title_text)
                negative_count = sum(1 for word in negative_keywords if word in title_text)
                
                if positive_count > 0 or negative_count > 0:
                    sentiment_score = (positive_count - negative_count) / (positive_count + negative_count)
            
            return {
                "headlines": headlines,
                "sentiment_score": round(sentiment_score, 2),
                "article_count": len(headlines)
            }
        except Exception as e:
            logger.error(f"Error getting news sentiment for {epic}: {e}")
            return {"headlines": [], "sentiment_score": 0.0}