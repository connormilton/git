# config_updates.py
# Add these settings to your config_loader.py DEFAULT_CONFIG dictionary or config.yaml file

ENHANCED_CONFIG = {
    # Display settings
    "TERMINAL_COLOR_ENABLED": True,  # Set to False if terminal doesn't support ANSI colors
    
    # Decision logging
    "DECISIONS_LOG_DIR": "data/decisions",
    "AUTO_REVIEW_ENABLED": True,
    
    # Performance analysis
    "PERFORMANCE_ANALYSIS_FREQUENCY_HOURS": 24,  # How often to run performance analysis
    
    # Trading session focus
    "TRADING_SESSIONS": {
        "ASIAN": {
            "start_hour": 0,  # UTC
            "end_hour": 8,    # UTC
            "active_pairs": ["CS.D.USDJPY.MINI.IP", "CS.D.AUDJPY.MINI.IP", "CS.D.EURJPY.MINI.IP"]
        },
        "EUROPEAN": {
            "start_hour": 7,  # UTC
            "end_hour": 16,   # UTC
            "active_pairs": ["CS.D.EURUSD.MINI.IP", "CS.D.GBPUSD.MINI.IP", "CS.D.EURGBP.MINI.IP"]
        },
        "AMERICAN": {
            "start_hour": 12, # UTC
            "end_hour": 21,   # UTC
            "active_pairs": ["CS.D.EURUSD.MINI.IP", "CS.D.GBPUSD.MINI.IP", "CS.D.USDCAD.MINI.IP"]
        },
        "OVERLAP": {
            "start_hour": 12, # UTC
            "end_hour": 16,   # UTC
            "active_pairs": ["CS.D.EURUSD.MINI.IP", "CS.D.GBPUSD.MINI.IP", "CS.D.USDJPY.MINI.IP"]
        }
    },
    
    # Pair selection parameters
    "MAX_PAIRS_TO_ANALYZE": 10,
    "VOLATILITY_PREFERENCE": "balanced",  # Options: "low", "balanced", "high"
    
    # Trading strategy parameters
    "TRADING_STRATEGY": {
        # Risk parameters
        "BASE_RISK_PERCENT": 1.5,
        "MAX_TOTAL_RISK_PERCENT": 15.0,
        "MAX_CURRENCY_EXPOSURE_PERCENT": 6.0,
        "PARTIAL_PROFIT_R_MULTIPLE": 1.5,
        "PARTIAL_PROFIT_PERCENTAGE": 33,
        
        # Entry filters
        "REQUIRED_TIMEFRAME_AGREEMENT": 2,  # Minimum timeframes in agreement
        "REQUIRED_TECHNICAL_CONFIRMATIONS": 3,
        "ECONOMIC_EVENT_BUFFER_MINUTES": 15,
        
        # Position sizing modifiers
        "HISTORICAL_PERFORMANCE_MULTIPLIER": 1.2,  # For pairs that perform well
        "CORRELATION_RISK_REDUCTION": 0.7,  # For correlated pairs
        "SCALE_IN_MAX_COUNT": 3,
        "ABSOLUTE_MAX_RISK_PERCENT": 2.0,
        
        # Market regime adjustments
        "RANGING_STOP_MULTIPLIER": 0.7,  # Tighter stops in ranges
        "RANGING_SIZE_MULTIPLIER": 0.8,  # Smaller positions in ranges
        "VOLATILE_STOP_MULTIPLIER": 1.5,  # Wider stops in volatility
        "VOLATILE_SIZE_MULTIPLIER": 0.6,  # Smaller positions in volatility
        "VOLATILE_PROFIT_R_MULTIPLE": 1.2,  # Take profit sooner in volatility
        
        # Time-based adjustments
        "LOW_LIQUIDITY_SIZE_MULTIPLIER": 0.6,  # Reduced size in low liquidity
        "FRIDAY_AFTERNOON_SIZE_MULTIPLIER": 0.7,  # Reduced size Friday afternoon
        
        # Technical indicator settings
        "ADX_TREND_THRESHOLD": 25,
        "RSI_PERIOD": 14,
        "MACD_SETTINGS": {
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9
        },
        "SUPERTREND_SETTINGS": {
            "atr_period": 10,
            "multiplier": 3.0
        },
        "BOLLINGER_SETTINGS": {
            "period": 20,
            "std_dev": 2.0
        },
        "CHANDELIER_EXIT_SETTINGS": {
            "period": 22,
            "multiplier": 3.0
        }
    }
}

# How to add these settings to your config_loader.py:
"""
def load_and_configure():
    global CONFIG
    
    # Load your existing default config
    DEFAULT_CONFIG = {
        # ... your existing config
    }
    
    # Add the enhanced config settings
    DEFAULT_CONFIG.update(ENHANCED_CONFIG)
    
    # Continue with the rest of your configuration loading
    CONFIG.update(DEFAULT_CONFIG)
    # ...
"""