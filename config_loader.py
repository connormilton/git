# config_loader.py

import os
import sys
import logging
import decimal
import yaml
from dotenv import load_dotenv

# Global CONFIG dict
CONFIG = {}

logger = logging.getLogger("TradingBotConfig")

def load_and_configure():
    """Loads config from defaults, .env, and optional yaml; updates global CONFIG."""
    global CONFIG

    # -------------------------------------------------------------------------
    # 1) Load .env
    # -------------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dotenv_path = os.path.join(script_dir, '.env')
    parent_dotenv_path = os.path.join(os.path.dirname(script_dir), '.env')

    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path, override=True)
        logger.info(f"Loaded .env from: {dotenv_path}")
    elif os.path.exists(parent_dotenv_path):
        load_dotenv(dotenv_path=parent_dotenv_path, override=True)
        logger.info(f"Loaded .env from: {parent_dotenv_path}")
    else:
        logger.warning("'.env' file not found. API keys must be set as environment variables.")

    # -------------------------------------------------------------------------
    # 2) Default Config
    # -------------------------------------------------------------------------
    DEFAULT_CONFIG = {
        # Risk
        "RISK_PER_TRADE_PERCENT": decimal.Decimal("5"),
        "MAX_TOTAL_RISK_PERCENT": decimal.Decimal("30.0"),
        "PER_CURRENCY_RISK_CAP": decimal.Decimal("10.0"),
        "MARGIN_BUFFER_FACTOR": decimal.Decimal("0.70"),
        "ACCOUNT_CURRENCY": "GBP",
        "CONFIDENCE_RISK_MULTIPLIERS": {
            "low": decimal.Decimal("0.5"),
            "medium": decimal.Decimal("1.0"),
            "high": decimal.Decimal("1.2")
        },
        # Strategy/Market
        "N_RECENT_TRADES_FEEDBACK": 5,
        "INITIAL_ASSET_FOCUS_EPICS": [
            "CS.D.EURUSD.MINI.IP",
            "CS.D.USDJPY.MINI.IP",
            "CS.D.GBPUSD.MINI.IP",
            "CS.D.AUDUSD.MINI.IP",
        ],
        "EPIC_TO_POLYGON_MAP": {
            "CS.D.EURUSD.MINI.IP": "C:EURUSD",
            "CS.D.USDJPY.MINI.IP": "C:USDJPY",
            "CS.D.GBPUSD.MINI.IP": "C:GBPUSD",
            "CS.D.AUDUSD.MINI.IP": "C:AUDUSD",
        },
        "HISTORICAL_DATA_TIMEFRAME": "hour",
        "HISTORICAL_DATA_PERIOD_DAYS": 30,
        "TOP_N_ASSETS_TO_TRADE": 2,
        # APIs
        "IG_USERNAME": None,
        "IG_PASSWORD": None,
        "IG_API_KEY": None,
        "IG_ACC_TYPE": "LIVE",
        "IG_ACCOUNT_ID": None,
        "POLYGON_API_KEY": None,
        # LLMs
        "OPENAI_API_KEY": None,
        "CLAUDE_API_KEY": None,
        "LLM_PROVIDER": "OpenAI",
        "LLM_MODEL_ANALYSIS": "gpt-4-turbo",
        "LLM_PROMPT_DIR": "llm_prompts",
        # Optional
        "GOOGLE_API_KEY": None,
        "SLACK_WEBHOOK_URL": None,
        "SLACK_BOT_TOKEN": None,
        "SLACK_SIGNING_SECRET": None,
        "DATABASE_PATH": "data/trading_memory.db",
        # Logging
        "LOG_LEVEL": "INFO",
        "LOG_FILE": "data/trading_bot.log",
        "TRADE_HISTORY_FILE": "data/trade_history.csv",
        # Workflow
        "TRADING_CYCLE_SECONDS": 180,
        
        # Added forex pairs configuration
        "EXPANDED_FOREX_PAIRS": {
            "CS.D.EURUSD.MINI.IP": {
                "description": "Euro/US Dollar",
                "category": "forex",
                "group": "majors"
            },
            "CS.D.USDJPY.MINI.IP": {
                "description": "US Dollar/Japanese Yen",
                "category": "forex",
                "group": "majors"
            },
            "CS.D.GBPUSD.MINI.IP": {
                "description": "British Pound/US Dollar",
                "category": "forex",
                "group": "majors"
            },
            "CS.D.AUDUSD.MINI.IP": {
                "description": "Australian Dollar/US Dollar",
                "category": "forex",
                "group": "majors"
            },
            "CS.D.USDCAD.MINI.IP": {
                "description": "US Dollar/Canadian Dollar",
                "category": "forex",
                "group": "majors"
            },
            "CS.D.EURGBP.MINI.IP": {
                "description": "Euro/British Pound",
                "category": "forex",
                "group": "crosses"
            },
            "CS.D.EURJPY.MINI.IP": {
                "description": "Euro/Japanese Yen",
                "category": "forex",
                "group": "crosses"
            },
            "CS.D.GBPJPY.MINI.IP": {
                "description": "British Pound/Japanese Yen",
                "category": "forex",
                "group": "crosses"
            },
            "CS.D.AUDJPY.MINI.IP": {
                "description": "Australian Dollar/Japanese Yen",
                "category": "forex",
                "group": "crosses"
            },
            "IX.D.FTSE.DAILY.IP": {
                "description": "FTSE 100",
                "category": "index",
                "group": "uk"
            },
            "CS.D.USCGC.TODAY.IP": {
                "description": "Gold",
                "category": "commodity",
                "group": "metals"
            }
        },

        # Trading sessions config
        "TRADING_SESSIONS": {
            "ASIAN": {
                "start_hour": 0,  # UTC
                "end_hour": 8,    # UTC
                "active_pairs": ["CS.D.USDJPY.MINI.IP", "CS.D.AUDJPY.MINI.IP", "CS.D.EURJPY.MINI.IP"]
            },
            "EUROPEAN": {
                "start_hour": 7,  # UTC
                "end_hour": 16,   # UTC
                "active_pairs": ["CS.D.EURUSD.MINI.IP", "CS.D.GBPUSD.MINI.IP", "CS.D.EURGBP.MINI.IP", "IX.D.FTSE.DAILY.IP"]
            },
            "AMERICAN": {
                "start_hour": 12, # UTC
                "end_hour": 21,   # UTC
                "active_pairs": ["CS.D.EURUSD.MINI.IP", "CS.D.GBPUSD.MINI.IP", "CS.D.USDCAD.MINI.IP"]
            }
        },

        # Pair characteristics
        "PAIR_CHARACTERISTICS": {
            "CS.D.EURUSD.MINI.IP": {"volatility": "medium", "liquidity": "high", "pip_value": 10},
            "CS.D.USDJPY.MINI.IP": {"volatility": "medium", "liquidity": "high", "pip_value": 9.35},
            "CS.D.GBPUSD.MINI.IP": {"volatility": "high", "liquidity": "high", "pip_value": 10},
            "CS.D.AUDUSD.MINI.IP": {"volatility": "high", "liquidity": "medium", "pip_value": 10},
            "CS.D.USDCAD.MINI.IP": {"volatility": "medium", "liquidity": "medium", "pip_value": 7.7},
            "CS.D.EURGBP.MINI.IP": {"volatility": "low", "liquidity": "medium", "pip_value": 12.6},
            "CS.D.EURJPY.MINI.IP": {"volatility": "medium", "liquidity": "medium", "pip_value": 9.35},
            "IX.D.FTSE.DAILY.IP": {"volatility": "high", "liquidity": "high", "pip_value": 10},
            "CS.D.USCGC.TODAY.IP": {"volatility": "high", "liquidity": "high", "pip_value": 10}
        },
    }
    CONFIG.update(DEFAULT_CONFIG)

    # -------------------------------------------------------------------------
    # 3) Load from config.yaml (optional override)
    # -------------------------------------------------------------------------
    config_file = os.path.join(script_dir, 'config.yaml')
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
            if isinstance(yaml_config, dict):
                CONFIG.update(yaml_config)
                logger.info(f"Loaded overrides from config.yaml: {config_file}")
        except Exception as e:
            logger.warning(f"Failed to load config.yaml: {e}")

    # -------------------------------------------------------------------------
    # 4) Environment Variable Overrides (highest priority)
    # -------------------------------------------------------------------------
    for key in CONFIG.keys():
        env_val = os.environ.get(key)
        if env_val is not None:
            CONFIG[key] = env_val

    # Convert numeric env values to Decimal if needed
    numeric_keys = [
        "RISK_PER_TRADE_PERCENT",
        "MAX_TOTAL_RISK_PERCENT",
        "PER_CURRENCY_RISK_CAP",
        "MARGIN_BUFFER_FACTOR"
    ]
    for nk in numeric_keys:
        try:
            CONFIG[nk] = decimal.Decimal(str(CONFIG[nk]))
        except:
            pass  # if it fails, keep as is

    # -------------------------------------------------------------------------
    # 5) Validate essential keys
    # -------------------------------------------------------------------------
    essential_keys = ["IG_USERNAME", "IG_PASSWORD", "IG_API_KEY", "IG_ACCOUNT_ID", "POLYGON_API_KEY"]
    llm_provider = CONFIG.get("LLM_PROVIDER", "OpenAI").lower()
    llm_api_key_name = f"{llm_provider.upper()}_API_KEY"
    if llm_api_key_name not in essential_keys:
        essential_keys.append(llm_api_key_name)

    missing_keys = [k for k in essential_keys if not CONFIG.get(k)]
    if missing_keys:
        message = f"CRITICAL CONFIG ERROR: Missing essential keys: {missing_keys}. Check .env or config!"
        logger.critical(message)
        raise ValueError(message)

    # -------------------------------------------------------------------------
    # 6) Final checks / library checks
    # -------------------------------------------------------------------------
    try:
        if llm_provider == 'openai':
            import openai
        elif llm_provider == 'claude':
            import anthropic
    except ImportError as e:
        raise ImportError(f"{llm_provider} provider selected but library not installed: {str(e)}")

    # Additional checks for trading_ig, polygon, etc. could be done here.

    logger.info("Configuration loaded and validated.")

def get_config() -> dict:
    """Returns the global CONFIG dictionary."""
    return CONFIG