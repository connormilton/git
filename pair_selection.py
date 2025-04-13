# pair_selection.py

import logging
import numpy as np
from datetime import datetime, timezone
import pandas as pd

logger = logging.getLogger("TradingBot")

class PairSelector:
    """Advanced pair selection to identify the best forex pairs to trade."""
    
    def __init__(self, config, data_provider, trade_memory):
        self.config = config
        self.data_provider = data_provider
        self.trade_memory = trade_memory
        
        # Get configuration for pairs
        self.available_pairs = config.get('EXPANDED_FOREX_PAIRS', {})
        self.pair_characteristics = config.get('PAIR_CHARACTERISTICS', {})
        self.currency_groups = config.get('CURRENCY_GROUPS', {})
        self.trading_sessions = config.get('TRADING_SESSIONS', {})
        
        # Default selection parameters
        self.max_pairs_to_analyze = config.get('MAX_PAIRS_TO_ANALYZE', 10)
        self.volatility_preference = config.get('VOLATILITY_PREFERENCE', 'balanced')
        
    def get_current_trading_session(self):
        """Determine the current trading session based on UTC time."""
        now = datetime.now(timezone.utc)
        current_hour = now.hour
        
        active_sessions = []
        for session_name, session_data in self.trading_sessions.items():
            start_hour = session_data.get('start_hour', 0)
            end_hour = session_data.get('end_hour', 0)
            
            # Handle sessions that span across midnight
            if start_hour <= end_hour:
                if start_hour <= current_hour < end_hour:
                    active_sessions.append(session_name)
            else:
                if current_hour >= start_hour or current_hour < end_hour:
                    active_sessions.append(session_name)
                    
        return active_sessions
        
    def get_session_recommended_pairs(self):
        """Get pairs recommended for the current trading session."""
        active_sessions = self.get_current_trading_session()
        
        recommended_pairs = set()
        for session in active_sessions:
            session_data = self.trading_sessions.get(session, {})
            session_pairs = session_data.get('active_pairs', [])
            recommended_pairs.update(session_pairs)
            
        return list(recommended_pairs)
        
    def calculate_volatility_score(self, pairs=None):
        """Calculate volatility scores for pairs based on ATR."""
        if pairs is None:
            pairs = list(self.available_pairs.keys())
            
        volatility_scores = {}
        for epic in pairs:
            try:
                # Get technical data that includes ATR
                tech_data = self.data_provider.get_latest_technicals(epic)
                
                if tech_data and 'ATR_14' in tech_data:
                    atr = tech_data['ATR_14']
                    
                    # Get current price for percentage calculation
                    current_price = tech_data.get('Close', 1.0)
                    
                    # Express ATR as percentage of price
                    atr_percent = (atr / current_price) * 100
                    
                    volatility_scores[epic] = atr_percent
                else:
                    # Fallback to hardcoded characteristics if available
                    if epic in self.pair_characteristics:
                        vol_rating = self.pair_characteristics[epic].get('volatility', 'medium')
                        
                        # Convert rating to numeric score
                        vol_score_map = {
                            'very low': 0.2,
                            'low': 0.5, 
                            'medium': 1.0, 
                            'high': 1.5, 
                            'very high': 2.0,
                            'extreme': 3.0
                        }
                        
                        volatility_scores[epic] = vol_score_map.get(vol_rating, 1.0)
            except Exception as e:
                logger.error(f"Error calculating volatility for {epic}: {e}")
                
        return volatility_scores
                
    def calculate_performance_score(self, pairs=None, days=30):
        """Calculate performance scores based on historical trading."""
        if pairs is None:
            pairs = list(self.available_pairs.keys())
            
        performance_scores = {}
        
        # Try to get performance data from trade memory
        try:
            dashboard = self.trade_memory.get_performance_dashboard(timeframe='monthly')
            instruments_data = dashboard.get('instruments', {})
            
            for epic in pairs:
                if epic in instruments_data:
                    data = instruments_data[epic]
                    
                    # Extract metrics
                    win_rate_str = data.get('win_rate', '0%').strip('%')
                    win_rate = float(win_rate_str) / 100 if win_rate_str else 0
                    
                    profit_factor = float(data.get('profit_factor', 0))
                    expectancy = float(data.get('expectancy', 0))
                    
                    # Basic scoring formula
                    score = (win_rate * 0.4) + (min(profit_factor, 3) * 0.2) + (min(expectancy, 2) * 0.4)
                    
                    # Adjust for sample size
                    trade_count = int(data.get('trade_count', 0))
                    confidence_factor = min(trade_count / 10, 1.0)  # Scale up to 10 trades
                    
                    adjusted_score = score * confidence_factor
                    performance_scores[epic] = adjusted_score
                else:
                    # No trading history, assign neutral score
                    performance_scores[epic] = 0.5
        except Exception as e:
            logger.error(f"Error calculating performance scores: {e}")
            # Assign neutral scores if we can't get actual data
            for epic in pairs:
                performance_scores[epic] = 0.5
                
        return performance_scores
        
    def calculate_trend_strength_score(self, pairs=None):
        """Calculate trend strength scores based on ADX values."""
        if pairs is None:
            pairs = list(self.available_pairs.keys())
            
        trend_scores = {}
        for epic in pairs:
            try:
                # Get technical data
                tech_data = self.data_provider.get_latest_technicals(epic)
                
                if tech_data and 'ADX' in tech_data:
                    adx = tech_data['ADX']
                    
                    # ADX interpretation: <20 weak trend, >25 strong trend, >40 very strong
                    if adx < 20:
                        trend_scores[epic] = 0.2  # Weak trend
                    elif adx < 25:
                        trend_scores[epic] = 0.5  # Moderate trend
                    elif adx < 40:
                        trend_scores[epic] = 0.8  # Strong trend
                    else:
                        trend_scores[epic] = 1.0  # Very strong trend
                else:
                    # No ADX data
                    trend_scores[epic] = 0.5  # Neutral
            except Exception as e:
                logger.error(f"Error calculating trend strength for {epic}: {e}")
                trend_scores[epic] = 0.5  # Neutral
                
        return trend_scores
        
    def calculate_setup_quality_score(self, pairs=None):
        """Calculate the quality of current setups based on technical indicators."""
        if pairs is None:
            pairs = list(self.available_pairs.keys())
            
        setup_scores = {}
        for epic in pairs:
            try:
                # Get technical data
                tech_data = self.data_provider.get_latest_technicals(epic)
                if not tech_data:
                    setup_scores[epic] = 0.0
                    continue
                    
                # Calculate composite score based on multiple indicators
                score = 0.0
                indicators_count = 0
                
                # 1. RSI - check for overbought/oversold
                if 'RSI' in tech_data:
                    rsi = tech_data['RSI']
                    indicators_count += 1
                    
                    if rsi < 30:  # Oversold - potential buy
                        score += 0.8
                    elif rsi > 70:  # Overbought - potential sell
                        score += 0.8
                    elif 40 <= rsi <= 60:  # Neutral
                        score += 0.3
                    else:  # Approaching oversold/overbought
                        score += 0.5
                        
                # 2. MACD - check for crossovers
                if 'MACD' in tech_data and 'MACD_Signal' in tech_data:
                    macd = tech_data['MACD']
                    signal = tech_data['MACD_Signal']
                    indicators_count += 1
                    
                    # Check for recent crossover (would need historical data for precision)
                    if abs(macd - signal) < 0.0005:  # Very close - potential crossover
                        score += 0.9
                    elif (macd > signal and macd > 0) or (macd < signal and macd < 0):  # Aligned direction
                        score += 0.7
                    else:  # Not aligned
                        score += 0.3
                        
                # 3. Bollinger Bands - check for price near bands
                if 'BB_Upper' in tech_data and 'BB_Lower' in tech_data and 'Close' in tech_data:
                    upper = tech_data['BB_Upper']
                    lower = tech_data['BB_Lower']
                    close = tech_data['Close']
                    indicators_count += 1
                    
                    # Calculate distance from bands as percentage
                    band_width = upper - lower
                    if band_width > 0:
                        upper_distance = (upper - close) / band_width
                        lower_distance = (close - lower) / band_width
                        
                        # Price near a band
                        if upper_distance < 0.1 or lower_distance < 0.1:
                            score += 0.8
                        # Price in middle - less clear signal
                        elif 0.4 < upper_distance < 0.6 and 0.4 < lower_distance < 0.6:
                            score += 0.3
                        else:
                            score += 0.5
                            
                # 4. Trend Direction - check if price is above/below key MAs
                trend_aligned = False
                if 'SMA_50' in tech_data and 'SMA_200' in tech_data and 'Close' in tech_data:
                    sma50 = tech_data['SMA_50']
                    sma200 = tech_data['SMA_200']
                    close = tech_data['Close']
                    indicators_count += 1
                    
                    # Check for trend alignment
                    if (close > sma50 > sma200) or (close < sma50 < sma200):
                        score += 0.8
                        trend_aligned = True
                    elif (sma50 > sma200 and close < sma50) or (sma50 < sma200 and close > sma50):
                        # Price pulled back to MA in trending market
                        score += 0.7
                    else:
                        score += 0.4
                        
                # 5. Stochastic - check for overbought/oversold with trend
                if 'Stoch_K' in tech_data and 'Stoch_D' in tech_data:
                    k = tech_data['Stoch_K']
                    d = tech_data['Stoch_D']
                    indicators_count += 1
                    
                    # Stochastic signal
                    if k < 20 and d < 20:  # Oversold
                        if trend_aligned and close > sma50:  # Pullback in uptrend
                            score += 0.9
                        else:
                            score += 0.7
                    elif k > 80 and d > 80:  # Overbought
                        if trend_aligned and close < sma50:  # Pullback in downtrend
                            score += 0.9
                        else:
                            score += 0.7
                    elif abs(k - d) < 3:  # K and D very close - potential crossover
                        score += 0.6
                    else:
                        score += 0.4
                        
                # Calculate average score if we have indicators
                if indicators_count > 0:
                    setup_scores[epic] = score / indicators_count
                else:
                    setup_scores[epic] = 0.0
                    
            except Exception as e:
                logger.error(f"Error calculating setup quality for {epic}: {e}")
                setup_scores[epic] = 0.0
                
        return setup_scores
        
    def calculate_correlation_diversity(self, selected_pairs):
        """Calculate how diversified the selected pairs are."""
        if len(selected_pairs) <= 1:
            return 1.0  # No correlation with just one pair
            
        try:
            # Get correlation matrix for these pairs
            corr_matrix = self.data_provider.get_correlation_matrix(selected_pairs)
            
            if not corr_matrix:
                return 0.5  # Neutral score if we can't get data
                
            # Calculate average absolute correlation
            abs_correlations = []
            for epic1 in selected_pairs:
                for epic2 in selected_pairs:
                    if epic1 != epic2 and epic1 in corr_matrix and epic2 in corr_matrix[epic1]:
                        abs_correlations.append(abs(corr_matrix[epic1][epic2]))
            
            if abs_correlations:
                avg_correlation = sum(abs_correlations) / len(abs_correlations)
                # Convert to diversity score (1 - correlation)
                diversity_score = 1.0 - avg_correlation
                return diversity_score
            else:
                return 0.5
        except Exception as e:
            logger.error(f"Error calculating correlation diversity: {e}")
            return 0.5
            
    def select_best_pairs(self, max_pairs=None):
        """Select the best pairs to trade based on multiple factors."""
        if max_pairs is None:
            max_pairs = self.max_pairs_to_analyze
            
        try:
            # Get all available pairs
            all_pairs = list(self.available_pairs.keys())
            
            # 1. First filter: Focus on pairs active in current session
            session_pairs = self.get_session_recommended_pairs()
            
            # If no session-specific pairs, use all pairs
            active_pairs = session_pairs if session_pairs else all_pairs
            
            # 2. Calculate individual scores
            volatility_scores = self.calculate_volatility_score(active_pairs)
            performance_scores = self.calculate_performance_score(active_pairs)
            trend_scores = self.calculate_trend_strength_score(active_pairs)
            setup_scores = self.calculate_setup_quality_score(active_pairs)
            
            # 3. Combine scores based on volatility preference
            combined_scores = {}
            for epic in active_pairs:
                vol_score = volatility_scores.get(epic, 0.5)
                perf_score = performance_scores.get(epic, 0.5)
                trend_score = trend_scores.get(epic, 0.5)
                setup_score = setup_scores.get(epic, 0.0)
                
                # Adjust weights based on volatility preference
                if self.volatility_preference == 'high':
                    # Favor volatile pairs
                    vol_weight = 0.3
                elif self.volatility_preference == 'low':
                    # Favor less volatile pairs
                    vol_weight = -0.1
                else:  # balanced
                    # Middle ground
                    vol_weight = 0.1
                    
                # Calculate combined score
                # Setup quality is most important, followed by trend strength and past performance
                combined_score = (
                    (setup_score * 0.5) +
                    (trend_score * 0.2) +
                    (perf_score * 0.2) +
                    (vol_score * vol_weight)
                )
                
                combined_scores[epic] = combined_score
                
            # 4. Sort pairs by combined score
            ranked_pairs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            
            # 5. Select top pairs while considering diversity
            selected_pairs = []
            for epic, score in ranked_pairs:
                selected_pairs.append(epic)
                
                # Check if we have enough pairs
                if len(selected_pairs) >= max_pairs:
                    break
                    
                # Every 3 pairs, check correlation to ensure diversity
                if len(selected_pairs) % 3 == 0:
                    diversity = self.calculate_correlation_diversity(selected_pairs)
                    
                    # If pairs are too correlated, we might need to adjust our selection
                    if diversity < 0.3:  # High correlation
                        logger.info(f"Selected pairs have high correlation (diversity={diversity:.2f}). Adjusting selection.")
                        
                        # Remove the last added pair and try a different one
                        selected_pairs.pop()
                        
                        # Find the next best pair that improves diversity
                        for alt_epic, alt_score in ranked_pairs:
                            if alt_epic not in selected_pairs:
                                test_pairs = selected_pairs + [alt_epic]
                                test_diversity = self.calculate_correlation_diversity(test_pairs)
                                
                                if test_diversity > diversity:
                                    selected_pairs.append(alt_epic)
                                    break
            
            # Log the selected pairs and their scores
            logger.info(f"Selected {len(selected_pairs)} pairs for analysis:")
            for epic in selected_pairs:
                logger.info(f"  {epic}: Score={combined_scores[epic]:.2f}, "
                           f"Setup={setup_scores.get(epic, 0):.2f}, "
                           f"Trend={trend_scores.get(epic, 0):.2f}, "
                           f"Perf={performance_scores.get(epic, 0):.2f}, "
                           f"Vol={volatility_scores.get(epic, 0):.2f}")
                
            return selected_pairs
            
        except Exception as e:
            logger.error(f"Error in pair selection: {e}")
            # Fallback to default pairs
            default_pairs = list(self.available_pairs.keys())[:max_pairs]
            return default_pairs