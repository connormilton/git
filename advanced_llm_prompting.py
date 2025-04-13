# advanced_llm_prompting.py

import os
import json
import logging
import decimal
import re
import numpy as np
from datetime import datetime, timezone

logger = logging.getLogger("TradingBot")

class AdvancedLLMPrompting:
    """Handles enhanced LLM prompting with context-aware, data-rich inputs for better trading decisions."""
    
    def __init__(self, config, trade_memory):
        self.config = config
        self.trade_memory = trade_memory
        self.prompt_dir = config.get("LLM_PROMPT_DIR")
        
        # Define different prompt templates for different market regimes
        self.prompt_templates = {
            'default': 'advanced_trade_analysis.txt',
            'uptrend': 'uptrend_trade_analysis.txt',
            'downtrend': 'downtrend_trade_analysis.txt',
            'ranging': 'ranging_trade_analysis.txt',
            'volatile': 'volatile_trade_analysis.txt',
            'news_focus': 'news_trade_analysis.txt'
        }
        
        # Default to combined analysis if specific templates don't exist
        for key, template in self.prompt_templates.items():
            if key != 'default' and not os.path.exists(os.path.join(self.prompt_dir, template)):
                self.prompt_templates[key] = self.prompt_templates['default']
                
        # Cache for technical indicator descriptions
        self.indicator_descriptions = {
            'RSI': "Relative Strength Index - measures momentum by comparing recent gains to losses. Values above 70 suggest overbought conditions, while values below 30 suggest oversold conditions.",
            'MACD': "Moving Average Convergence Divergence - trend-following momentum indicator showing the relationship between two moving averages. The MACD line crossing above the signal line is bullish, while crossing below is bearish.",
            'ADX': "Average Directional Index - measures trend strength. Values above 25 indicate a strong trend, while values below 20 indicate a weak trend or ranging market.",
            'BB_Width': "Bollinger Band Width - measures volatility. Wider bands indicate higher volatility, while narrower bands suggest lower volatility and potential breakout setups.",
            'ATR': "Average True Range - measures volatility by calculating the average range between high and low prices. Higher ATR indicates higher volatility.",
            'Stochastic': "Stochastic Oscillator - momentum indicator comparing closing price to price range over a period. Values above 80 suggest overbought conditions, while values below 20 suggest oversold conditions.",
            'SMA': "Simple Moving Average - calculates the average price over a specified period. Used to identify trend direction and potential support/resistance levels.",
            'EMA': "Exponential Moving Average - similar to SMA but gives more weight to recent prices. Often more responsive to price changes than SMA."
        }
        
    def _load_prompt_template(self, template_filename):
        """Load a prompt template from file."""
        filepath = os.path.join(self.prompt_dir, template_filename)
        try:
            if not os.path.exists(filepath):
                logger.error(f"Prompt template '{filepath}' not found.")
                # Fall back to default template
                default_path = os.path.join(self.prompt_dir, self.prompt_templates['default'])
                if os.path.exists(default_path):
                    with open(default_path, 'r', encoding='utf-8') as f:
                        return f.read()
                else:
                    return self._generate_generic_template()
                    
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Could not load prompt template {filepath}: {e}")
            return self._generate_generic_template()
            
    def _generate_generic_template(self):
        """Generate a basic template as a fallback."""
        return """
        You are an expert AI trading analyst for Forex Majors, trading on a {ACCOUNT_CURRENCY} account via IG.
        
        **Current Account & Risk State:**
        *   Account Balance: {ACCOUNT_CURRENCY} {ACCOUNT_BALANCE}
        *   Available Margin: {ACCOUNT_CURRENCY} {AVAILABLE_MARGIN}
        *   Current Total Risk: {CURRENT_RISK_PERCENT}% of balance
        *   Configured Risk Per Trade: {RISK_PER_TRADE_PERCENT}% of balance (before adjustments)
        *   Max Total Portfolio Risk: {MAX_TOTAL_RISK_PERCENT}% of balance
        
        **Current Open Positions:**
        ```json
        {OPEN_POSITIONS_JSON}
        ```
        
        **Current Market Snapshot (Relevant Assets):**
        ```json
        {MARKET_SNAPSHOT_JSON}
        ```
        
        **Technical Indicators:**
        ```json
        {TECHNICAL_INDICATORS_JSON}
        ```
        
        **Recent Performance:**
        ```json
        {PERFORMANCE_METRICS_JSON}
        ```
        
        **Recent Trade History:**
        ```json
        {TRADE_HISTORY_JSON}
        ```
        
        **Market News & Events:**
        ```
        {MARKET_NEWS_TEXT}
        ```
        
        Analyze the provided data and return a JSON response with trading recommendations.
        Response must be in the following format:
        
        ```json
        {
          "tradeActions": [
            { "epic": "EPIC_CODE", "action": "BUY/SELL", "stop_distance": NUM, "limit_distance": NUM, "confidence": "low/medium/high" }
          ],
          "tradeAmendments": [
            { "epic": "EPIC_CODE", "action": "CLOSE/AMEND/BREAKEVEN", "new_stop_distance": NUM, "new_limit_distance": NUM }
          ],
          "reasoning": {
            "epic1": "explanation string",
            "epic2": "explanation string",
            "global": "overall market assessment"
          }
        }
        ```
        
        Ensure that all distances are positive numbers representing points/pips from the current price.
        Base all decisions on the provided market data, technical indicators, and historical performance.
        Only recommend trades for instruments in the provided market snapshot.
        """
            
    def _format_technical_indicators(self, technicals_dict):
        """Format technical indicators with descriptions for better LLM understanding."""
        if not technicals_dict:
            return {}
            
        formatted = {}
        for epic, indicators in technicals_dict.items():
            formatted[epic] = {
                'values': indicators,
                'interpretations': {}
            }
            
            # Add interpretations for key indicators
            if 'RSI' in indicators:
                rsi = indicators['RSI']
                if rsi > 70:
                    formatted[epic]['interpretations']['RSI'] = f"Overbought at {rsi:.1f}"
                elif rsi < 30:
                    formatted[epic]['interpretations']['RSI'] = f"Oversold at {rsi:.1f}"
                else:
                    formatted[epic]['interpretations']['RSI'] = f"Neutral at {rsi:.1f}"
                    
            if 'MACD' in indicators and 'MACD_Signal' in indicators:
                macd = indicators['MACD']
                signal = indicators['MACD_Signal']
                if macd > signal:
                    formatted[epic]['interpretations']['MACD'] = f"Bullish (MACD: {macd:.4f} > Signal: {signal:.4f})"
                else:
                    formatted[epic]['interpretations']['MACD'] = f"Bearish (MACD: {macd:.4f} < Signal: {signal:.4f})"
                    
            if 'ADX' in indicators:
                adx = indicators['ADX']
                if adx > 25:
                    formatted[epic]['interpretations']['ADX'] = f"Strong trend at {adx:.1f}"
                else:
                    formatted[epic]['interpretations']['ADX'] = f"Weak/no trend at {adx:.1f}"
                    
            if 'ATR_14' in indicators:
                formatted[epic]['interpretations']['ATR'] = f"Volatility measure: {indicators['ATR_14']:.5f}"
                
            if 'Trend' in indicators:
                trend = indicators['Trend']
                if trend > 0:
                    formatted[epic]['interpretations']['Trend'] = "Uptrend"
                elif trend < 0:
                    formatted[epic]['interpretations']['Trend'] = "Downtrend"
                else:
                    formatted[epic]['interpretations']['Trend'] = "No clear trend"
                    
            # Add regime information if available
            if 'regime' in indicators:
                formatted[epic]['market_regime'] = indicators['regime']
                
        return formatted
        
    def _format_performance_metrics(self, performance_data):
        """Format performance metrics for the prompt."""
        if not performance_data:
            return {}
            
        # Extract key metrics
        overall = performance_data.get('overall', {})
        instruments = performance_data.get('instruments', {})
        market_regimes = performance_data.get('market_regimes', [])
        
        # Format for better LLM understanding
        formatted = {
            'overall_metrics': {
                'win_rate': f"{overall.get('win_rate', 0) * 100:.1f}%",
                'profit_factor': f"{overall.get('profit_factor', 0):.2f}",
                'expectancy': f"{overall.get('expectancy', 0):.2f}",
                'risk_reward_ratio': f"{overall.get('risk_reward_ratio', 0):.2f}"
            },
            'instruments': {},
            'market_regimes': {}
        }
        
        # Format instrument performance
        for epic, data in instruments.items():
            formatted['instruments'][epic] = {
                'win_rate': f"{data.get('win_rate', 0) * 100:.1f}%",
                'profit_factor': f"{data.get('profit_factor', 0):.2f}",
                'expectancy': f"{data.get('expectancy', 0):.2f}",
                'trade_count': data.get('trade_count', 0),
                'net_pnl': data.get('net_pnl', 0)
            }
            
        # Format market regime performance
        for regime in market_regimes:
            regime_name = regime.get('market_regime', 'unknown')
            if regime_name:
                formatted['market_regimes'][regime_name] = {
                    'win_rate': f"{regime.get('win_rate', 0) * 100:.1f}%",
                    'trade_count': regime.get('total_trades', 0),
                    'net_pnl': regime.get('net_pnl', 0)
                }
                
        return formatted
        
    def _format_market_news(self, news_items):
        """Format market news into a readable text."""
        if not news_items:
            return "No significant market news available."
            
        formatted_text = "Recent Market News:\n\n"
        
        for item in news_items[:5]:  # Limit to 5 news items
            title = item.get('title', 'No title')
            publisher = item.get('publisher', 'Unknown')
            date = item.get('published_utc', '')
            if date:
                try:
                    date_obj = datetime.fromisoformat(date.replace('Z', '+00:00'))
                    date_str = date_obj.strftime('%Y-%m-%d %H:%M UTC')
                except:
                    date_str = date
            else:
                date_str = 'Unknown date'
                
            ticker = item.get('ticker', '')
            
            formatted_text += f"- {title}\n"
            formatted_text += f"  Source: {publisher}, Date: {date_str}, Ticker: {ticker}\n\n"
            
        return formatted_text
        
    def _format_risk_exposure(self, open_position_risk, total_risk, balance):
        """Format current risk exposure."""
        if not open_position_risk:
            return {
                "total_risk_amount": 0,
                "total_risk_percent": "0.00%",
                "positions": {}
            }
            
        # Calculate total risk percentage
        total_risk_percent = (total_risk / balance * 100) if balance > 0 else 0
        
        # Format each position's risk
        positions_risk = {}
        for epic, data in open_position_risk.items():
            positions_risk[epic] = {
                "direction": data.get('direction', 'unknown'),
                "size": float(data.get('size', 0)),
                "risk_amount": float(data.get('risk_amount', 0)),
                "risk_percent": f"{float(data.get('risk_percent', 0)):.2f}%"
            }
            
        return {
            "total_risk_amount": float(total_risk),
            "total_risk_percent": f"{float(total_risk_percent):.2f}%",
            "positions": positions_risk
        }
        
    def _format_multi_timeframe_analysis(self, mtf_data):
        """Format multi-timeframe analysis data."""
        if not mtf_data:
            return {}
            
        formatted = {}
        for timeframe, data in mtf_data.items():
            formatted[timeframe] = {
                "trend": data.get('trend', 'neutral'),
                "momentum": data.get('momentum', 'neutral'),
                "volatility": data.get('volatility', 'normal'),
                "key_indicators": {
                    "RSI": data.get('rsi', 50),
                    "ADX": data.get('adx', 0),
                    "BB_Width": data.get('bb_width', 0)
                }
            }
            
        return formatted

    # Custom JSON encoder to handle Decimal objects
    class DecimalEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, decimal.Decimal):
                return float(o)
            return super().default(o)
        
    def generate_advanced_prompt(self, portfolio, market_data, data_provider, risk_manager):
        """
        Generate an advanced, context-aware prompt for the LLM.
        This includes technical indicators, risk analysis, and market regime detection.
        """
        # Get basic portfolio info
        balance = portfolio.get_balance()
        available = portfolio.get_available_funds()
        positions = portfolio.get_open_positions_dict()
        
        # Get risk exposure
        total_risk = risk_manager.total_open_risk
        position_risk = risk_manager.open_position_risk
        risk_exposure = self._format_risk_exposure(position_risk, total_risk, balance)
        
        # Determine overall market regime
        market_regimes = {}
        technical_data = {}
        for epic, data in market_data.items():
            # Get technical indicators from data provider
            tech_indicators = data_provider.get_latest_technicals(epic)
            technical_data[epic] = tech_indicators
            
            # Get market regime
            regime = data_provider.get_market_regime(epic)
            market_regimes[epic] = regime
            
        # Format technical data for prompt
        formatted_technicals = self._format_technical_indicators(technical_data)
        
        # Get multi-timeframe analysis for active instruments
        mtf_analysis = {}
        for epic in market_data.keys():
            mtf_data = data_provider.get_multi_timeframe_analysis(epic)
            if mtf_data:
                mtf_analysis[epic] = mtf_data
                
        formatted_mtf = self._format_multi_timeframe_analysis(mtf_analysis)
        
        # Get news and sentiment
        news_data = []
        for epic in market_data.keys():
            news = data_provider.get_news_sentiment(epic)
            if news and news.get('headlines'):
                for headline in news.get('headlines', []):
                    news_data.append({
                        'title': headline,
                        'ticker': epic,
                        'published_utc': datetime.now(timezone.utc).isoformat(),
                        'publisher': 'Market News'
                    })
                    
        news_text = self._format_market_news(news_data)
        
        # Get performance metrics from trade memory
        performance_data = self.trade_memory.get_performance_dashboard(timeframe='monthly')
        formatted_performance = self._format_performance_metrics(performance_data)
        
        # Get trade recommendations from memory
        trade_recommendations = self.trade_memory.get_trading_recommendations()
        
        # Recent trade history
        trade_history = portfolio.get_recent_trade_summary()
        
        # Pick appropriate prompt template based on dominant market regime
        regimes_count = {r: 0 for r in set(market_regimes.values())}
        for regime in market_regimes.values():
            regimes_count[regime] = regimes_count.get(regime, 0) + 1
            
        # Determine dominant regime (most common)
        dominant_regime = max(regimes_count.items(), key=lambda x: x[1])[0] if regimes_count else 'default'
        
        # Handle news-driven markets specially
        if news_data and len(news_data) > 3:
            # If we have significant news, potentially override with news-focused template
            dominant_regime = 'news_focus'
            
        template_name = self.prompt_templates.get(dominant_regime, self.prompt_templates['default'])
        template = self._load_prompt_template(template_name)
        
        # Prepare context data for template
        current_risk_percent = (total_risk / balance * 100) if balance > 0 else decimal.Decimal('0.0')
        
        # Create context data with proper JSON serialization of Decimal values
        context_data = {
            "ACCOUNT_BALANCE": f"{balance:.2f}",
            "AVAILABLE_MARGIN": f"{available:.2f}",
            "ACCOUNT_CURRENCY": self.config['ACCOUNT_CURRENCY'],
            "CURRENT_RISK_AMOUNT": f"{total_risk:.2f}",
            "CURRENT_RISK_PERCENT": f"{current_risk_percent:.2f}",
            "RISK_PER_TRADE_PERCENT": f"{self.config['RISK_PER_TRADE_PERCENT']:.2f}",
            "MAX_TOTAL_RISK_PERCENT": f"{self.config['MAX_TOTAL_RISK_PERCENT']:.2f}",
            "PER_CURRENCY_RISK_CAP": f"{self.config['PER_CURRENCY_RISK_CAP']:.2f}",
            "OPEN_POSITIONS_JSON": json.dumps(positions, indent=2, cls=self.DecimalEncoder),
            "MARKET_SNAPSHOT_JSON": json.dumps(market_data, indent=2, cls=self.DecimalEncoder),
            "TECHNICAL_INDICATORS_JSON": json.dumps(formatted_technicals, indent=2, cls=self.DecimalEncoder),
            "RISK_EXPOSURE_JSON": json.dumps(risk_exposure, indent=2, cls=self.DecimalEncoder),
            "MULTI_TIMEFRAME_JSON": json.dumps(formatted_mtf, indent=2, cls=self.DecimalEncoder),
            "PERFORMANCE_METRICS_JSON": json.dumps(formatted_performance, indent=2, cls=self.DecimalEncoder),
            "TRADE_RECOMMENDATIONS_JSON": json.dumps(trade_recommendations, indent=2, cls=self.DecimalEncoder),
            "TRADE_HISTORY_JSON": json.dumps(trade_history, indent=2, cls=self.DecimalEncoder),
            "MARKET_NEWS_TEXT": news_text,
            "CURRENT_MARKET_REGIME": dominant_regime.upper(),
            "DOMINANT_REGIME_DESC": self._get_regime_description(dominant_regime),
            "N_RECENT_TRADES": self.config['N_RECENT_TRADES_FEEDBACK'],
            "MARKET_REGIMES_JSON": json.dumps(market_regimes, indent=2, cls=self.DecimalEncoder)
        }
        
        # Try formatting the template
        try:
            prompt = template.format(**context_data)
        except KeyError as e:
            missing_key = str(e).strip("'")
            logger.error(f"Missing key in prompt template: {missing_key}")
            # Use a simpler template as fallback
            prompt = self._generate_generic_template().format(**context_data)
        except ValueError as e:
            logger.error(f"Error formatting prompt template: {e}")
            # Fix common format string issues with curly braces
            for key, value in context_data.items():
                placeholder = '{' + key + '}'
                if placeholder in template:
                    template = template.replace(placeholder, str(value))
            prompt = template
            
        return prompt, dominant_regime
    
    def _get_regime_description(self, regime):
        """Get a description of a market regime for the LLM."""
        descriptions = {
            'uptrend': "Markets are in an uptrend with bullish momentum. Look for continuation patterns and pullback buying opportunities.",
            'downtrend': "Markets are in a downtrend with bearish momentum. Look for continuation patterns and rally selling opportunities.",
            'ranging': "Markets are in a sideways range without clear directional bias. Look for range-bound trading opportunities and potential breakouts.",
            'volatile': "Markets are showing increased volatility with larger price swings. Use wider stops, reduce position sizes, and focus on short-term opportunities.",
            'news_focus': "Markets are being driven by significant news events. Pay attention to sentiment shifts and potential rapid directional moves.",
            'default': "Analyze the market based on technical and fundamental factors to identify trading opportunities."
        }
        return descriptions.get(regime, descriptions['default'])
        
    def generate_system_prompt(self, market_regime):
        """Generate a tailored system prompt based on market regime."""
        base_prompt = """You are an expert AI trading analyst for Forex Majors, trading on a spread betting account via IG.
Provide specific, actionable trade recommendations in strict JSON format. 
Ensure all stop_distance and limit_distance values are positive numbers representing points from current price.
Focus on identifying high-probability setups based on technical analysis and recent performance data.
"""
        
        # Add regime-specific guidance
        regime_guidance = {
            'uptrend': "The current market is in an UPTREND. Focus on BUY setups with clear support levels for stop placement. Look for pullbacks to moving averages as potential entry points.",
            'downtrend': "The current market is in a DOWNTREND. Focus on SELL setups with clear resistance levels for stop placement. Look for rallies to moving averages as potential entry points.",
            'ranging': "The current market is RANGE-BOUND. Focus on mean-reversion setups, buying near support and selling near resistance. Keep stop distances reasonable as false breakouts are common.",
            'volatile': "The current market is VOLATILE. Use wider stops to accommodate larger price swings, reduce position sizes, and look for clear high-probability setups. Avoid low-confidence trades.",
            'news_focus': "The market is currently NEWS-DRIVEN. Pay close attention to sentiment shifts, use wider stops to accommodate news volatility, and be cautious with new positions."
        }
        
        guidance = regime_guidance.get(market_regime, "")
        if guidance:
            base_prompt += f"\n\n{guidance}"
            
        return base_prompt