#!/usr/bin/env python3
# auto_implement_margin_system.py
# Script to automatically implement margin validation system

import os
import sys
import subprocess
import shutil
import re
from datetime import datetime

# Constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKUP_DIR = os.path.join(SCRIPT_DIR, "backup_before_margin_impl_" + datetime.now().strftime("%Y%m%d_%H%M%S"))

# New files to create
NEW_FILES = {}

# File contents for margin_validator.py
NEW_FILES["margin_validator.py"] = """
# margin_validator.py

import logging
import decimal
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger("TradingBot")

class MarginValidator:
    \"\"\"
    Pre-validates trading opportunities based on margin requirements before analysis.
    This ensures we don't waste resources analyzing trades that can't be executed.
    \"\"\"
    
    def __init__(self, config, broker, portfolio):
        \"\"\"
        Initialize the margin validator.
        
        Args:
            config: Configuration dictionary
            broker: Broker interface for getting instrument details
            portfolio: Portfolio object for account information
        \"\"\"
        self.config = config
        self.broker = broker
        self.portfolio = portfolio
        
        # Minimum deal sizes by broker (hardcoded defaults)
        self.default_min_deal_size = decimal.Decimal('0.125')  # Most common for IG mini contracts
        
        # Instrument detail cache to reduce API calls
        self.instrument_cache = {}
        
        # Hardcoded VPP values for common pairs (as fallback)
        self.APPROX_VPP_GBP = {
            "CS.D.EURUSD.MINI.IP": decimal.Decimal("0.81"),
            "CS.D.USDJPY.MINI.IP": decimal.Decimal("0.74"),
            "CS.D.GBPUSD.MINI.IP": decimal.Decimal("0.81"),
            "CS.D.AUDUSD.MINI.IP": decimal.Decimal("0.81"),
            "CS.D.USDCAD.MINI.IP": decimal.Decimal("0.77"),
            "CS.D.EURGBP.MINI.IP": decimal.Decimal("1.26"),
            "CS.D.EURJPY.MINI.IP": decimal.Decimal("0.94"),
            "CS.D.GBPJPY.MINI.IP": decimal.Decimal("0.94"),
        }
        
    def _get_value_per_point(self, epic: str) -> decimal.Decimal:
        \"\"\"Get the value per point for an instrument.\"\"\"
        # Check if we have it in our hardcoded values
        if epic in self.APPROX_VPP_GBP:
            vpp = self.APPROX_VPP_GBP[epic]
            logger.debug(f"Using hardcoded VPP for {epic}: {vpp}")
            return vpp
            
        # Try to get from broker
        if epic in self.instrument_cache:
            instrument = self.instrument_cache[epic]
        else:
            instrument = self.broker.get_instrument_details(epic)
            if instrument:
                self.instrument_cache[epic] = instrument
                
        if instrument and 'valuePerPoint' in instrument and instrument['valuePerPoint']:
            return decimal.Decimal(str(instrument['valuePerPoint']))
            
        # Fallback
        logger.warning(f"Could not determine VPP for {epic}, using default of 1.0")
        return decimal.Decimal('1.0')
        
    def _get_min_deal_size(self, epic: str) -> decimal.Decimal:
        \"\"\"Get the minimum deal size for an instrument.\"\"\"
        # Try to get from broker
        if epic in self.instrument_cache:
            instrument = self.instrument_cache[epic]
        else:
            instrument = self.broker.get_instrument_details(epic)
            if instrument:
                self.instrument_cache[epic] = instrument
                
        if instrument and 'minDealSize' in instrument and instrument['minDealSize']:
            return decimal.Decimal(str(instrument['minDealSize']))
            
        # Fallback
        return self.default_min_deal_size
        
    def _get_margin_factor(self, epic: str) -> decimal.Decimal:
        \"\"\"Get the margin factor for an instrument.\"\"\"
        # Try to get from broker
        if epic in self.instrument_cache:
            instrument = self.instrument_cache[epic]
        else:
            instrument = self.broker.get_instrument_details(epic)
            if instrument:
                self.instrument_cache[epic] = instrument
                
        if instrument and 'marginFactor' in instrument and instrument['marginFactor']:
            return decimal.Decimal(str(instrument['marginFactor']))
            
        # Fallback - typical margin requirements are around 3-5%
        return decimal.Decimal('0.05')  # 5%
        
    def check_margin_viability(self, epic: str, stop_distance: decimal.Decimal) -> Tuple[bool, str, Dict[str, Any]]:
        \"\"\"
        Check if a trade would be viable based on margin requirements and minimum deal size.
        
        Args:
            epic: The instrument to check
            stop_distance: Planned stop distance in points
            
        Returns:
            Tuple of (is_viable, reason, details)
            - is_viable: Boolean indicating if trade meets margin requirements
            - reason: Description of why trade is/isn't viable
            - details: Dictionary with calculated values like min_size, required_margin, etc.
        \"\"\"
        # Get required values
        available_margin = self.portfolio.get_available_funds()
        balance = self.portfolio.get_balance()
        
        # Get instrument-specific values
        min_deal_size = self._get_min_deal_size(epic)
        vpp = self._get_value_per_point(epic)
        margin_factor = self._get_margin_factor(epic)
        
        # Get current price (bid/ask) from broker
        snapshot = self.broker.fetch_market_snapshot(epic)
        if not snapshot or 'bid' not in snapshot or snapshot['bid'] is None:
            return False, f"No price data available for {epic}", {
                "min_deal_size": min_deal_size,
                "available_margin": available_margin
            }
            
        # Use mid price for simplicity in calculations
        current_price = (snapshot['bid'] + snapshot['offer']) / 2
        
        # Calculate margin required for minimum position size
        min_margin_required = current_price * min_deal_size * margin_factor
        
        # Calculate maximum affordable size given available margin (with buffer)
        margin_buffer = decimal.Decimal('0.9')  # Use 90% of available margin
        max_affordable_size = (available_margin * margin_buffer) / (current_price * margin_factor)
        
        # Calculate risk parameters
        risk_per_trade_pct = decimal.Decimal(str(self.config.get('RISK_PER_TRADE_PERCENT', 2.0)))
        max_risk_amount = balance * (risk_per_trade_pct / 100)
        
        # Calculate size based on risk and stop distance
        risk_based_size = max_risk_amount / (stop_distance * vpp)
        
        # Determine if viable based on various constraints
        if min_deal_size > max_affordable_size:
            return False, f"Cannot meet margin requirements with minimum size {min_deal_size}", {
                "min_deal_size": min_deal_size,
                "min_margin_required": min_margin_required,
                "available_margin": available_margin,
                "max_affordable_size": max_affordable_size,
                "risk_based_size": risk_based_size
            }
            
        if risk_based_size < min_deal_size:
            # Can afford minimum size, but risk parameters would require smaller size
            # We'll return True but note that risk will be higher than preferred
            return True, f"Warning: Minimum size {min_deal_size} exceeds risk-based size {risk_based_size:.2f}. Risk will be higher than configured {risk_per_trade_pct}%.", {
                "min_deal_size": min_deal_size,
                "min_margin_required": min_margin_required,
                "available_margin": available_margin,
                "max_affordable_size": max_affordable_size,
                "risk_based_size": risk_based_size,
                "actual_size": min_deal_size,
                "actual_risk_amount": min_deal_size * stop_distance * vpp,
                "actual_risk_pct": ((min_deal_size * stop_distance * vpp) / balance) * 100
            }
            
        # Trade is viable with preferred risk parameters
        actual_size = max(min_deal_size, risk_based_size)
        if max_affordable_size < actual_size:
            actual_size = max_affordable_size
            
        return True, "Trade meets margin and risk requirements", {
            "min_deal_size": min_deal_size,
            "min_margin_required": min_margin_required,
            "available_margin": available_margin,
            "max_affordable_size": max_affordable_size,
            "risk_based_size": risk_based_size,
            "actual_size": actual_size,
            "actual_risk_amount": actual_size * stop_distance * vpp,
            "actual_risk_pct": ((actual_size * stop_distance * vpp) / balance) * 100
        }
        
    def get_viable_instruments(self, candidate_pairs: List[str], min_stop_distance: decimal.Decimal = decimal.Decimal('5')) -> List[Dict[str, Any]]:
        \"\"\"
        Filter a list of instruments to only those that meet margin requirements.
        
        Args:
            candidate_pairs: List of instrument epics to check
            min_stop_distance: Minimum stop distance to use in viability check
            
        Returns:
            List of dictionaries with viable instruments and their details
        \"\"\"
        viable_instruments = []
        
        for epic in candidate_pairs:
            is_viable, reason, details = self.check_margin_viability(epic, min_stop_distance)
            
            instrument_details = {
                "epic": epic,
                "viable": is_viable,
                "reason": reason,
                **details
            }
            
            if is_viable:
                viable_instruments.append(instrument_details)
            else:
                logger.warning(f"Skipping {epic}: {reason}")
                
        return viable_instruments
        
    def get_adaptive_risk_parameters(self, viable_instruments: List[Dict[str, Any]]) -> Dict[str, Any]:
        \"\"\"
        Calculate adaptive risk parameters based on viable instruments.
        
        Args:
            viable_instruments: List of viable instrument details
            
        Returns:
            Dictionary with adaptive risk parameters
        \"\"\"
        # If we have no viable instruments, return default parameters
        if not viable_instruments:
            return {
                "adjusted_risk_percent": self.config.get('RISK_PER_TRADE_PERCENT', 2.0),
                "allow_higher_risk": False,
                "focus_on_larger_stops": False,
                "reason": "No viable instruments"
            }
            
        # Get balance for calculations
        balance = self.portfolio.get_balance()
        
        # Check if any instruments require higher risk than configured
        higher_risk_required = any(
            inst.get("min_deal_size", 0) > inst.get("risk_based_size", 0)
            for inst in viable_instruments
        )
        
        if higher_risk_required:
            # Calculate the minimum risk percentage needed for the most constrained instrument
            max_required_risk_pct = max(
                ((inst['min_deal_size'] * decimal.Decimal('10') * inst.get('vpp', decimal.Decimal('1.0'))) / balance) * 100
                for inst in viable_instruments
                if inst.get("min_deal_size", 0) > inst.get("risk_based_size", 0)
            )
            
            return {
                "adjusted_risk_percent": float(max_required_risk_pct),
                "allow_higher_risk": True,
                "focus_on_larger_stops": False,
                "reason": "Minimum position sizes require higher risk"
            }
            
        # Check if we have limited margin
        available_margin = self.portfolio.get_available_funds()
        margin_utilization = 1 - (available_margin / balance)
        
        if margin_utilization > decimal.Decimal('0.7'):  # 70% margin used
            # When margin is limited, we focus on trades with larger stops
            # This allows more efficient use of capital
            return {
                "adjusted_risk_percent": float(self.config.get('RISK_PER_TRADE_PERCENT', 2.0)),
                "allow_higher_risk": False,
                "focus_on_larger_stops": True,
                "reason": "Limited margin available"
            }
            
        # Default case - no adjustments needed
        return {
            "adjusted_risk_percent": float(self.config.get('RISK_PER_TRADE_PERCENT', 2.0)),
            "allow_higher_risk": False,
            "focus_on_larger_stops": False,
            "reason": "Standard risk parameters adequate"
        }
"""

# File contents for margin_analyzer.py
NEW_FILES["margin_analyzer.py"] = """
# margin_analyzer.py

import logging
import decimal
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from datetime import datetime

logger = logging.getLogger("TradingBot")

class MarginRequirementAnalyzer:
    \"\"\"
    Advanced analysis of margin requirements for trading instruments.
    Provides solutions for margin-related issues and alternative strategies.
    \"\"\"
    
    def __init__(self, config, broker, portfolio):
        self.config = config
        self.broker = broker
        self.portfolio = portfolio
        
        # Default settings
        self.default_min_size = decimal.Decimal('0.125')  # Common minimum for IG mini contracts
        self.margin_buffer = decimal.Decimal('0.9')  # 90% of available margin as buffer
        self.default_margin_factor = decimal.Decimal('0.05')  # 5% margin requirement
        
        # Default VPP values (Value Per Point) for common pairs
        self.VPP_DEFAULTS = {
            "CS.D.EURUSD.MINI.IP": decimal.Decimal("0.81"),
            "CS.D.USDJPY.MINI.IP": decimal.Decimal("0.74"),
            "CS.D.GBPUSD.MINI.IP": decimal.Decimal("0.81"),
            "CS.D.AUDUSD.MINI.IP": decimal.Decimal("0.81"),
            "CS.D.USDCAD.MINI.IP": decimal.Decimal("0.77"),
            "CS.D.EURGBP.MINI.IP": decimal.Decimal("1.26"),
            "CS.D.EURJPY.MINI.IP": decimal.Decimal("0.94"),
            "CS.D.GBPJPY.MINI.IP": decimal.Decimal("0.94"),
        }
        
        # Cache instrument details
        self.instrument_cache = {}
        
    def analyze_all_instruments(self) -> pd.DataFrame:
        \"\"\"
        Analyze all instruments in the config for margin requirements.
        
        Returns:
            DataFrame with analysis of all instruments
        \"\"\"
        all_pairs = list(self.config.get('EXPANDED_FOREX_PAIRS', {}).keys())
        
        if not all_pairs:
            logger.warning("No pairs found in EXPANDED_FOREX_PAIRS config")
            all_pairs = list(self.VPP_DEFAULTS.keys())
            
        results = []
        
        for epic in all_pairs:
            # Get instrument details
            instrument = self.broker.get_instrument_details(epic)
            if instrument:
                self.instrument_cache[epic] = instrument
            
            # Get VPP
            vpp = self._get_vpp(epic)
            
            # Get min deal size
            min_size = self._get_min_size(epic)
            
            # Get margin factor
            margin_factor = self._get_margin_factor(epic)
            
            # Get current price
            snapshot = self.broker.fetch_market_snapshot(epic)
            if not snapshot or 'bid' not in snapshot or snapshot['bid'] is None:
                current_price = decimal.Decimal('1.0')  # Fallback
                has_price = False
            else:
                current_price = (snapshot['bid'] + snapshot['offer']) / 2
                has_price = True
                
            # Calculate minimum margin
            min_margin = current_price * min_size * margin_factor
            
            # Calculate margin for typical stop distances
            risk_pct = decimal.Decimal(str(self.config.get('RISK_PER_TRADE_PERCENT', 2.0)))
            balance = self.portfolio.get_balance()
            available = self.portfolio.get_available_funds()
            max_risk = balance * (risk_pct / 100)
            
            # Check typical stop distances and find viable ones
            stop_distances = [decimal.Decimal(str(x)) for x in [5, 10, 15, 20, 25, 30, 40, 50]]
            viable_stops = []
            min_viable_stop = None
            
            for stop in stop_distances:
                size_by_risk = max_risk / (stop * vpp)
                if size_by_risk >= min_size:
                    viable_stops.append(int(stop))
                    if min_viable_stop is None:
                        min_viable_stop = int(stop)
            
            # Calculate maximum affordable size
            max_size = (available * self.margin_buffer) / (current_price * margin_factor)
            
            # Add to results
            results.append({
                "epic": epic,
                "description": self.config.get('EXPANDED_FOREX_PAIRS', {}).get(epic, {}).get('description', epic),
                "value_per_point": float(vpp),
                "min_deal_size": float(min_size),
                "margin_factor": float(margin_factor),
                "min_margin_required": float(min_margin),
                "current_price": float(current_price) if has_price else None,
                "max_affordable_size": float(max_size),
                "min_viable_stop": min_viable_stop,
                "viable_stop_distances": viable_stops,
                "has_price_data": has_price,
                "is_viable": min_viable_stop is not None and max_size >= min_size
            })
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(results)
        
        # Add additional metrics
        if not df.empty and 'min_margin_required' in df.columns and 'is_viable' in df.columns:
            available_margin = float(self.portfolio.get_available_funds())
            df['margin_usage_percent'] = (df['min_margin_required'] / available_margin) * 100
            df['viable_count'] = len(df[df['is_viable'] == True])
            
        return df
    
    def get_required_balance(self, epic: str, stop_distance: decimal.Decimal) -> decimal.Decimal:
        \"\"\"
        Calculate the minimum account balance required to trade this instrument.
        
        Args:
            epic: Instrument epic
            stop_distance: Desired stop distance in points
            
        Returns:
            Minimum required balance
        \"\"\"
        # Get parameters
        vpp = self._get_vpp(epic)
        min_size = self._get_min_size(epic)
        
        # Get configured risk percentage
        risk_pct = decimal.Decimal(str(self.config.get('RISK_PER_TRADE_PERCENT', 2.0)))
        
        # Calculate minimum risk amount
        min_risk = min_size * stop_distance * vpp
        
        # Calculate required balance
        required_balance = (min_risk / risk_pct) * 100
        
        return required_balance
        
    def suggest_alternative_strategies(self, current_balance: decimal.Decimal) -> Dict[str, Any]:
        \"\"\"
        Suggest alternative strategies based on current balance.
        
        Args:
            current_balance: Current account balance
            
        Returns:
            Dictionary with suggested strategies
        \"\"\"
        # Analyze margin requirements for all instruments
        analysis = self.analyze_all_instruments()
        
        # Get viable instruments
        viable_instruments = analysis[analysis['is_viable'] == True] if not analysis.empty else pd.DataFrame()
        
        # Minimum balance required for trading with current risk settings
        min_balance_required = decimal.Decimal('0')
        
        if not viable_instruments.empty:
            # We have some viable instruments, use the minimum balance of those
            min_balance_required = current_balance
        else:
            # Calculate minimum balance needed for each instrument with a standard stop
            standard_stop = decimal.Decimal('20')
            all_pairs = list(self.config.get('EXPANDED_FOREX_PAIRS', {}).keys())
            
            if not all_pairs:
                all_pairs = list(self.VPP_DEFAULTS.keys())
                
            required_balances = []
            for epic in all_pairs:
                required_balances.append(self.get_required_balance(epic, standard_stop))
                
            if required_balances:
                min_balance_required = min(required_balances)
                
        # Calculate strategies
        strategies = {
            "current_balance": float(current_balance),
            "min_balance_required": float(min_balance_required),
            "additional_funding_needed": float(max(decimal.Decimal('0'), min_balance_required - current_balance)),
            "viable_instruments_count": len(viable_instruments) if not analysis.empty else 0,
            "alternatives": []
        }
        
        # Strategy 1: Increase risk per trade
        current_risk = decimal.Decimal(str(self.config.get('RISK_PER_TRADE_PERCENT', 2.0)))
        higher_risk_options = [current_risk * decimal.Decimal('1.5'), current_risk * decimal.Decimal('2.0'), decimal.Decimal('5.0')]
        
        for risk in higher_risk_options:
            viable_count = 0
            
            # Check how many instruments would be viable with this risk
            for _, row in analysis.iterrows():
                epic = row['epic']
                vpp = decimal.Decimal(str(row['value_per_point']))
                min_size = decimal.Decimal(str(row['min_deal_size']))
                
                # Find minimum viable stop with this risk
                for stop in [5, 10, 15, 20, 25, 30, 40, 50]:
                    stop_dec = decimal.Decimal(str(stop))
                    max_risk = current_balance * (risk / 100)
                    size_by_risk = max_risk / (stop_dec * vpp)
                    
                    if size_by_risk >= min_size:
                        viable_count += 1
                        break
            
            if viable_count > 0:
                strategies["alternatives"].append({
                    "type": "increase_risk",
                    "risk_percent": float(risk),
                    "viable_count": viable_count,
                    "description": f"Increase risk per trade to {float(risk):.1f}%. {viable_count} instruments would be viable."
                })
        
        # Strategy 2: Use wider stops
        if 'min_viable_stop' in analysis.columns:
            wider_stop_instruments = analysis[analysis['min_viable_stop'].notnull()]
            if not wider_stop_instruments.empty:
                median_stop = wider_stop_instruments['min_viable_stop'].median()
                strategies["alternatives"].append({
                    "type": "wider_stops",
                    "median_stop": float(median_stop),
                    "viable_count": len(wider_stop_instruments),
                    "description": f"Use wider stops (median {median_stop:.0f} points). {len(wider_stop_instruments)} instruments would be viable."
                })
        
        # Strategy 3: Focus on specific instruments
        if not analysis.empty:
            # Find instruments with lowest margin requirements
            analysis_sorted = analysis.sort_values('min_margin_required')
            top_5 = analysis_sorted.head(5)
            
            if not top_5.empty:
                epic_list = top_5['epic'].tolist()
                strategies["alternatives"].append({
                    "type": "focus_instruments",
                    "instruments": epic_list,
                    "description": f"Focus on these lower-margin instruments: {', '.join(epic_list)}"
                })
        
        # Strategy 4: Required top-up amount
        additional_needed = strategies["additional_funding_needed"]
        if additional_needed > 0:
            strategies["alternatives"].append({
                "type": "add_funds",
                "amount": additional_needed,
                "description": f"Add {additional_needed:.2f} {self.config.get('ACCOUNT_CURRENCY', 'GBP')} to your account."
            })
            
        return strategies
    
    def _get_vpp(self, epic: str) -> decimal.Decimal:
        \"\"\"Get Value Per Point for an instrument.\"\"\"
        # Try from instrument cache first
        if epic in self.instrument_cache:
            instrument = self.instrument_cache[epic]
            if 'valuePerPoint' in instrument and instrument['valuePerPoint']:
                return decimal.Decimal(str(instrument['valuePerPoint']))
        
        # Try from hardcoded values
        if epic in self.VPP_DEFAULTS:
            return self.VPP_DEFAULTS[epic]
            
        # Try to get from broker
        instrument = self.broker.get_instrument_details(epic)
        if instrument:
            self.instrument_cache[epic] = instrument
            if 'valuePerPoint' in instrument and instrument['valuePerPoint']:
                return decimal.Decimal(str(instrument['valuePerPoint']))
        
        # Default fallback
        return decimal.Decimal('1.0')
    
    def _get_min_size(self, epic: str) -> decimal.Decimal:
        \"\"\"Get minimum deal size for an instrument.\"\"\"
        # Try from instrument cache first
        if epic in self.instrument_cache:
            instrument = self.instrument_cache[epic]
            if 'minDealSize' in instrument and instrument['minDealSize']:
                return decimal.Decimal(str(instrument['minDealSize']))
        
        # Try to get from broker
        instrument = self.broker.get_instrument_details(epic)
        if instrument:
            self.instrument_cache[epic] = instrument
            if 'minDealSize' in instrument and instrument['minDealSize']:
                return decimal.Decimal(str(instrument['minDealSize']))
        
        # Default fallback
        return self.default_min_size
    
    def _get_margin_factor(self, epic: str) -> decimal.Decimal:
        \"\"\"Get margin factor for an instrument.\"\"\"
        # Try from instrument cache first
        if epic in self.instrument_cache:
            instrument = self.instrument_cache[epic]
            if 'marginFactor' in instrument and instrument['marginFactor']:
                return decimal.Decimal(str(instrument['marginFactor']))
        
        # Try to get from broker
        instrument = self.broker.get_instrument_details(epic)
        if instrument:
            self.instrument_cache[epic] = instrument
            if 'marginFactor' in instrument and instrument['marginFactor']:
                return decimal.Decimal(str(instrument['marginFactor']))
        
        # Default fallback
        return self.default_margin_factor
"""

# File contents for margin_analysis_demo.py
NEW_FILES["margin_analysis_demo.py"] = """
# margin_analysis_demo.py
# Standalone demo script to analyze margin requirements on your account

import os
import sys
import decimal
import json
import pandas as pd
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("MarginAnalysis")

# Import required modules - adjust paths if needed
try:
    from margin_analyzer import MarginRequirementAnalyzer
    from ig_interface import IGInterface
    from portfolio import Portfolio
except ImportError:
    logger.error("Unable to import required modules. Please run this script from your project directory.")
    sys.exit(1)

def load_config():
    \"\"\"Load configuration from config_loader or environment variables.\"\"\"
    try:
        # Try to import from project's config_loader
        from config_loader import get_config
        return get_config()
    except ImportError:
        # Fallback to minimal config from environment variables
        logger.warning("Could not import config_loader. Using environment variables.")
        config = {
            'IG_USERNAME': os.environ.get('IG_USERNAME'),
            'IG_PASSWORD': os.environ.get('IG_PASSWORD'),
            'IG_API_KEY': os.environ.get('IG_API_KEY'),
            'IG_ACC_TYPE': os.environ.get('IG_ACC_TYPE', 'DEMO'),  # Default to DEMO
            'IG_ACCOUNT_ID': os.environ.get('IG_ACCOUNT_ID'),
            'ACCOUNT_CURRENCY': os.environ.get('ACCOUNT_CURRENCY', 'GBP'),
            'RISK_PER_TRADE_PERCENT': decimal.Decimal(os.environ.get('RISK_PER_TRADE_PERCENT', '2.0')),
            'EXPANDED_FOREX_PAIRS': {
                "CS.D.EURUSD.MINI.IP": {"description": "Euro/US Dollar"},
                "CS.D.USDJPY.MINI.IP": {"description": "US Dollar/Japanese Yen"},
                "CS.D.GBPUSD.MINI.IP": {"description": "British Pound/US Dollar"},
                "CS.D.AUDUSD.MINI.IP": {"description": "Australian Dollar/US Dollar"},
                "CS.D.USDCAD.MINI.IP": {"description": "US Dollar/Canadian Dollar"},
                "CS.D.EURGBP.MINI.IP": {"description": "Euro/British Pound"},
                "CS.D.EURJPY.MINI.IP": {"description": "Euro/Japanese Yen"}
            }
        }
        
        # Check if required config is present
        required_keys = ['IG_USERNAME', 'IG_PASSWORD', 'IG_API_KEY', 'IG_ACCOUNT_ID']
        missing_keys = [key for key in required_keys if not config.get(key)]
        
        if missing_keys:
            logger.error(f"Missing required configuration: {missing_keys}")
            print("\\nPlease set the following environment variables:")
            for key in missing_keys:
                print(f"  export {key}='your-value'")
            sys.exit(1)
            
        return config

def save_analysis_to_file(analysis, filename):
    \"\"\"Save the analysis dataframe to a file.\"\"\"
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save as CSV
        analysis.to_csv(filename)
        logger.info(f"Analysis saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving analysis to {filename}: {e}")

def save_strategies_to_file(strategies, filename):
    \"\"\"Save the strategies to a JSON file.\"\"\"
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save as JSON
        with open(filename, 'w') as f:
            json.dump(strategies, f, indent=2)
        logger.info(f"Strategies saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving strategies to {filename}: {e}")

def format_currency(value, currency):
    \"\"\"Format a value as currency.\"\"\"
    return f"{currency} {value:.2f}"

def print_analysis_summary(analysis, currency):
    \"\"\"Print a summary of the analysis.\"\"\"
    # Count viable instruments
    viable_count = len(analysis[analysis['is_viable'] == True])
    total_count = len(analysis)
    
    print("\\n" + "="*80)
    print(f"MARGIN ANALYSIS SUMMARY ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    print("="*80)
    
    print(f"\\nViable Instruments: {viable_count}/{total_count} ({viable_count/total_count*100:.1f}%)")
    
    # Print account status if we have the data
    if 'account_balance' in analysis.columns and 'available_margin' in analysis.columns:
        balance = analysis['account_balance'].iloc[0]
        available = analysis['available_margin'].iloc[0]
        print(f"Account Balance: {format_currency(balance, currency)}")
        print(f"Available Margin: {format_currency(available, currency)}")
        
    # Print margin requirements
    if 'min_margin_required' in analysis.columns:
        avg_margin = analysis['min_margin_required'].mean()
        min_margin = analysis['min_margin_required'].min()
        max_margin = analysis['min_margin_required'].max()
        
        print(f"\\nMargin Requirements:")
        print(f"  Average: {format_currency(avg_margin, currency)}")
        print(f"  Minimum: {format_currency(min_margin, currency)}")
        print(f"  Maximum: {format_currency(max_margin, currency)}")
    
    # Print viable instruments
    if viable_count > 0:
        viable = analysis[analysis['is_viable'] == True]
        print("\\nViable Instruments:")
        for _, row in viable.iterrows():
            print(f"  {row['epic']} - {row['description']}")
            print(f"    Min Size: {row['min_deal_size']:.3f} | VPP: {row['value_per_point']:.4f} | Margin: {format_currency(row['min_margin_required'], currency)}")
            if 'min_viable_stop' in row and row['min_viable_stop'] is not None:
                print(f"    Min Viable Stop: {row['min_viable_stop']} points")
            if 'viable_stop_distances' in row and row['viable_stop_distances'] is not None:
                print(f"    Viable Stops: {row['viable_stop_distances']}")
                
    # Print non-viable instruments
    if viable_count < total_count:
        non_viable = analysis[analysis['is_viable'] == False]
        print("\\nNon-Viable Instruments:")
        for _, row in non_viable.iterrows():
            print(f"  {row['epic']} - {row['description']}")
    
    print("\\n" + "="*80)

def print_strategies(strategies, currency):
    \"\"\"Print alternative strategies.\"\"\"
    print("\\n" + "="*80)
    print("ALTERNATIVE STRATEGIES")
    print("="*80)
    
    print(f"\\nCurrent Balance: {format_currency(strategies['current_balance'], currency)}")
    print(f"Minimum Balance Required: {format_currency(strategies['min_balance_required'], currency)}")
    
    if strategies['additional_funding_needed'] > 0:
        print(f"Additional Funding Needed: {format_currency(strategies['additional_funding_needed'], currency)}")
    
    print(f"\\nViable Instruments Count: {strategies['viable_instruments_count']}")
    
    print("\\nSuggested Alternatives:")
    for i, alt in enumerate(strategies['alternatives'], 1):
        print(f"{i}. {alt['description']}")
    
    print("\\n" + "="*80)

def main():
    logger.info("Starting Margin Analysis Demo")
    
    # Load configuration
    config = load_config()
    
    print("\\n" + "="*80)
    print("MARGIN ANALYSIS DEMO")
    print("="*80)
    print(f"\\nConnecting to IG as {config['IG_USERNAME']} ({config['IG_ACC_TYPE']})")
    
    try:
        # Initialize components
        broker = IGInterface(config)
        portfolio = Portfolio(broker, config)
        
        # Update portfolio state
        portfolio.update_state()
        
        print(f"Connected successfully!")
        print(f"Account Balance: {portfolio.get_balance():.2f} {config['ACCOUNT_CURRENCY']}")
        print(f"Available Funds: {portfolio.get_available_funds():.2f} {config['ACCOUNT_CURRENCY']}")
        
        # Initialize margin analyzer
        margin_analyzer = MarginRequirementAnalyzer(config, broker, portfolio)
        
        print("\\nAnalyzing margin requirements for all instruments...")
        # Perform analysis
        analysis = margin_analyzer.analyze_all_instruments()
        
        # Add account info to analysis
        analysis['account_balance'] = portfolio.get_balance()
        analysis['available_margin'] = portfolio.get_available_funds()
        
        # Save analysis to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_analysis_to_file(analysis, f"data/margin_analysis_{timestamp}.csv")
        
        # Print summary
        print_analysis_summary(analysis, config['ACCOUNT_CURRENCY'])
        
        # Get strategies
        print("\\nGenerating alternative strategies...")
        strategies = margin_analyzer.suggest_alternative_strategies(portfolio.get_balance())
        
        # Save strategies to file
        save_strategies_to_file(strategies, f"data/margin_strategies_{timestamp}.json")
        
        # Print strategies
        print_strategies(strategies, config['ACCOUNT_CURRENCY'])
        
    except Exception as e:
        logger.error(f"Error during margin analysis: {e}", exc_info=True)
        print(f"\\nError during analysis: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
"""

# Files to modify
MODIFIED_FILES = {
    "main.py": {
        "imports_add": """
from margin_validator import MarginValidator
from margin_analyzer import MarginRequirementAnalyzer
""",
        "initialization_add": """        # Initialize our new margin components
        margin_validator = MarginValidator(config, broker, portfolio)
        margin_analyzer = MarginRequirementAnalyzer(config, broker, portfolio)

        # Perform initial margin analysis
        print("\\n🔍 Performing initial margin analysis...")
        initial_analysis = margin_analyzer.analyze_all_instruments()
        viable_count = len(initial_analysis[initial_analysis['is_viable'] == True]) if not initial_analysis.empty else 0
        
        if viable_count == 0:
            print("⚠️ WARNING: No instruments are currently viable with your account balance and risk settings!")
            print("   Analyzing alternatives...")
            
            strategies = margin_analyzer.suggest_alternative_strategies(portfolio.get_balance())
            
            print("\\n🛠️ MARGIN REQUIREMENT ANALYSIS")
            print(f"   Current Balance: {strategies['current_balance']:.2f} {config['ACCOUNT_CURRENCY']}")
            print(f"   Minimum Balance Needed: {strategies['min_balance_required']:.2f} {config['ACCOUNT_CURRENCY']}")
            print(f"   Additional Funding Needed: {strategies['additional_funding_needed']:.2f} {config['ACCOUNT_CURRENCY']}")
            
            print("\\n🔄 SUGGESTED ALTERNATIVES:")
            for i, strategy in enumerate(strategies['alternatives'], 1):
                print(f"   {i}. {strategy['description']}")
                
            # Auto-adopt the first viable strategy involving risk adjustment
            for strategy in strategies['alternatives']:
                if strategy['type'] == 'increase_risk' and strategy['viable_count'] > 0:
                    new_risk = decimal.Decimal(str(strategy['risk_percent']))
                    old_risk = risk_manager.base_risk_per_trade
                    
                    print(f"\\n✅ Auto-adjusting risk parameters: {old_risk}% → {new_risk}%")
                    risk_manager.base_risk_per_trade = new_risk
                    # Update config to reflect change
                    config['RISK_PER_TRADE_PERCENT'] = new_risk
                    break
        else:
            print(f"✅ {viable_count} instruments are viable with current settings")""",
        "pair_selection_replace": {
            "old": """            # 3. Select the best pairs to analyze based on current market conditions
            print("🔍 Selecting optimal trading pairs...")
            best_pairs = pair_selector.select_best_pairs(max_pairs=config.get('MAX_PAIRS_TO_ANALYZE', 10))
            if not best_pairs:
                logger.warning("No valid pairs selected. Will skip this cycle and try again later.")
                print("⚠️ WARNING: No valid pairs selected for this cycle. Will try again next cycle.")
                # Instead of breaking/stopping, we'll continue to the next cycle
                cycle_duration = time.time() - cycle_start_time
                sleep_time = max(10, config.get('TRADING_CYCLE_SECONDS', 180) - cycle_duration)
                logger.info(f"Cycle ended (No pairs selected). Sleeping {sleep_time:.2f}s...")
                print(f"\\n⏱️ Cycle completed in {cycle_duration:.2f}s. Sleeping for {sleep_time:.2f}s until next cycle...")
                print("="*100)
                time.sleep(sleep_time)
                continue  # Skip to next iteration instead of stopping
                
            print(f"✅ Selected {len(best_pairs)} pairs for analysis: {', '.join(best_pairs)}\\n")""",
            "new": """            # 3. Select candidate pairs for analysis
            print("🔍 Selecting optimal trading pairs...")
            candidate_pairs = pair_selector.select_best_pairs(max_pairs=config.get('MAX_PAIRS_TO_ANALYZE', 10))
            if not candidate_pairs:
                logger.warning("No valid pairs selected. Will skip this cycle and try again later.")
                print("⚠️ WARNING: No valid pairs selected for this cycle. Will try again next cycle.")
                # Instead of breaking/stopping, we'll continue to the next cycle
                cycle_duration = time.time() - cycle_start_time
                sleep_time = max(10, config.get('TRADING_CYCLE_SECONDS', 180) - cycle_duration)
                logger.info(f"Cycle ended (No pairs selected). Sleeping {sleep_time:.2f}s...")
                print(f"\\n⏱️ Cycle completed in {cycle_duration:.2f}s. Sleeping for {sleep_time:.2f}s until next cycle...")
                print("="*100)
                time.sleep(sleep_time)
                continue  # Skip to next iteration instead of stopping
            
            # 4. NEW STEP: Pre-filter candidates for margin viability
            viable_instruments = margin_validator.get_viable_instruments(
                candidate_pairs, 
                min_stop_distance=decimal.Decimal('10')  # Conservative initial estimate
            )
            
            viable_epics = [inst['epic'] for inst in viable_instruments]
            
            if not viable_epics:
                logger.warning("No viable instruments after margin check. Will skip this cycle.")
                print("⚠️ WARNING: No instruments meet margin requirements. Cycle skipped.")
                
                # Get suggestions for future cycles
                strategies = margin_analyzer.suggest_alternative_strategies(portfolio.get_balance())
                print("\\n💡 SUGGESTIONS FOR FUTURE CYCLES:")
                for i, strategy in enumerate(strategies['alternatives'], 1):
                    print(f"   {i}. {strategy['description']}")
                
                cycle_duration = time.time() - cycle_start_time
                sleep_time = max(10, config.get('TRADING_CYCLE_SECONDS', 180) - cycle_duration)
                logger.info(f"Cycle ended (No viable instruments). Sleeping {sleep_time:.2f}s...")
                print(f"\\n⏱️ Cycle completed in {cycle_duration:.2f}s. Sleeping for {sleep_time:.2f}s until next cycle...")
                print("="*100)
                time.sleep(sleep_time)
                continue  # Skip to next iteration
                
            print(f"✅ Selected {len(viable_epics)} pairs that meet margin requirements: {', '.join(viable_epics)}\\n")
            
            # 5. Get adaptive risk parameters
            adaptive_params = margin_validator.get_adaptive_risk_parameters(viable_instruments)
            if adaptive_params.get('allow_higher_risk', False):
                print(f"ℹ️ Using adjusted risk: {adaptive_params['adjusted_risk_percent']:.2f}% due to {adaptive_params['reason']}")
                
                # Temporarily adjust risk parameters
                original_risk = risk_manager.base_risk_per_trade
                risk_manager.base_risk_per_trade = decimal.Decimal(str(adaptive_params['adjusted_risk_percent']))"""
        },
        "market_data_replace": {
            "old": """            # 4. Fetch market data and technical indicators for selected pairs
            print("📊 Fetching market data and technical indicators...")
            market_snapshots = {}
            technical_data = {}
            for epic in best_pairs:
                # Get price snapshot
                snapshot = broker.fetch_market_snapshot(epic)
                if snapshot:
                    market_snapshots[epic] = snapshot
                
                # Get technical indicators and market regime
                try:
                    technicals = data_provider.get_latest_technicals(epic)
                    if technicals:
                        technical_data[epic] = technicals
                except Exception as tech_err:
                    logger.warning(f"Error getting technicals for {epic}: {tech_err}")
                
                time.sleep(0.1)  # Avoid API rate limits""",
            "new": """            # 6. Fetch market data for viable pairs only
            print("📊 Fetching market data and technical indicators...")
            market_snapshots = {}
            technical_data = {}
            for epic in viable_epics:
                # Get price snapshot
                snapshot = broker.fetch_market_snapshot(epic)
                if snapshot:
                    market_snapshots[epic] = snapshot
                
                # Get technical indicators and market regime
                try:
                    technicals = data_provider.get_latest_technicals(epic)
                    if technicals:
                        technical_data[epic] = technicals
                except Exception as tech_err:
                    logger.warning(f"Error getting technicals for {epic}: {tech_err}")
                
                time.sleep(0.1)  # Avoid API rate limits"""
        },
        "llm_prompt_replace": {
            "old": """                # 6. Generate advanced LLM prompt based on current market conditions
                print("🧠 Generating advanced LLM analysis prompt...")
                advanced_prompt, market_regime = llm_prompting.generate_advanced_prompt(
                    portfolio, valid_snapshots, data_provider, risk_manager
                )
                
                # 7. Generate system prompt based on market regime
                system_prompt = llm_prompting.generate_system_prompt(market_regime)""",
            "new": """                # 8. Generate advanced LLM prompt with margin context
                print("🧠 Generating advanced LLM analysis prompt...")
                # Add margin information to context
                advanced_prompt, market_regime = llm_prompting.generate_advanced_prompt(
                    portfolio, valid_snapshots, data_provider, risk_manager,
                    additional_context={
                        'MARGIN_CONSTRAINED': True,
                        'VIABLE_INSTRUMENTS': viable_epics,
                        'MIN_POSITION_SIZES': {inst['epic']: float(inst['min_deal_size']) for inst in viable_instruments},
                        'ADAPTIVE_RISK': adaptive_params
                    }
                )
                
                # 9. Generate system prompt with margin awareness
                system_prompt = llm_prompting.generate_system_prompt(market_regime)
                # Add margin constraints to system prompt
                margin_note = f"\\nDue to margin constraints, focus ONLY on these viable instruments: {', '.join(viable_epics)}.\\n"
                if adaptive_params.get('allow_higher_risk', False):
                    margin_note += f"Higher risk per trade ({adaptive_params['adjusted_risk_percent']:.2f}%) is being used due to minimum position size requirements.\\n"
                system_prompt += margin_note"""
        },
        "recommendations_filter_add": """                # 11. Filter recommendations to ensure only viable instruments are included
                if 'tradeActions' in recommendations:
                    original_actions = recommendations['tradeActions']
                    filtered_actions = [
                        action for action in original_actions
                        if action.get('epic') in viable_epics
                    ]
                    
                    if len(filtered_actions) < len(original_actions):
                        logger.info(f"Filtered {len(original_actions) - len(filtered_actions)} trade actions that didn't meet margin requirements")
                        recommendations['tradeActions'] = filtered_actions""",
        "reset_risk_params_add": """            # 13. Reset any temporary risk parameter changes
            if adaptive_params and adaptive_params.get('allow_higher_risk', False):
                risk_manager.base_risk_per_trade = original_risk
                logger.info(f"Reset risk parameters to original value: {original_risk}%")"""
    },
    "advanced_llm_prompting.py": {
        "add_additional_context": {
            "method": "generate_advanced_prompt",
            "old_signature": "def generate_advanced_prompt(self, portfolio, market_data, data_provider, risk_manager):",
            "new_signature": "def generate_advanced_prompt(self, portfolio, market_data, data_provider, risk_manager, additional_context=None):",
            "add_context_code": """    # Add the new margin context if provided
    if additional_context:
        # Format additional context with proper serialization
        for key, value in additional_context.items():
            # Special handling for nested dictionaries or complex objects
            if isinstance(value, dict) or isinstance(value, list):
                context_data[key] = json.dumps(value, indent=2, cls=self.DecimalEncoder)
            else:
                context_data[key] = value
        
        # Add special section for margin constraints if applicable
        if additional_context.get('MARGIN_CONSTRAINED', False):
            # Add information about minimum position sizes
            min_sizes = additional_context.get('MIN_POSITION_SIZES', {})
            viable_instruments = additional_context.get('VIABLE_INSTRUMENTS', [])
            
            # Create a margin constraints section to add to the prompt
            margin_section = "\\n## Margin Constraints\\n"
            margin_section += "Due to account margin requirements, only the following instruments are viable for trading in this cycle:\\n"
            margin_section += ", ".join(viable_instruments) + "\\n\\n"
            
            if min_sizes:
                margin_section += "Minimum position sizes:\\n"
                for epic, size in min_sizes.items():
                    margin_section += f"* {epic}: {size}\\n"
                    
            # Add information about adaptive risk if applicable
            adaptive_risk = additional_context.get('ADAPTIVE_RISK', {})
            if adaptive_risk:
                margin_section += f"\\nRisk per trade has been adjusted to {adaptive_risk.get('adjusted_risk_percent', 0):.2f}% "
                margin_section += f"due to: {adaptive_risk.get('reason', 'margin constraints')}\\n"
                
            # Add this section to the context
            context_data["MARGIN_CONSTRAINTS_TEXT"] = margin_section
            
            # Update template to include margin section if not already present
            if "MARGIN_CONSTRAINTS_TEXT" not in template:
                # Find a good location to insert - typically after account info section
                insertion_point = "## Current Market Data"
                if insertion_point in template:
                    template = template.replace(insertion_point, "{MARGIN_CONSTRAINTS_TEXT}\\n" + insertion_point)"""
        }
    }
}

def backup_file(file_path):
    """Create a backup of a file in the backup directory."""
    if not os.path.exists(file_path):
        print(f"[!] Warning: File {file_path} not found, skipping backup.")
        return False
        
    # Create backup directory if it doesn't exist
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
        
    backup_path = os.path.join(BACKUP_DIR, os.path.basename(file_path))
    try:
        shutil.copy2(file_path, backup_path)
        print(f"[✓] Backed up {file_path} to {backup_path}")
        return True
    except Exception as e:
        print(f"[!] Error backing up {file_path}: {e}")
        return False

def create_new_file(file_name, content):
    """Create a new file with the given content."""
    try:
        file_path = os.path.join(SCRIPT_DIR, file_name)
        
        # Check if file already exists
        if os.path.exists(file_path):
            print(f"[!] File {file_name} already exists.")
            backup_file(file_path)
            
        # Create the file
        with open(file_path, 'w') as f:
            f.write(content.strip())
            
        print(f"[✓] Created {file_name}")
        return True
    except Exception as e:
        print(f"[!] Error creating {file_name}: {e}")
        return False

def modify_file(file_name, modifications):
    """Modify an existing file according to the modification instructions."""
    file_path = os.path.join(SCRIPT_DIR, file_name)
    
    if not os.path.exists(file_path):
        print(f"[!] File {file_name} not found, cannot modify.")
        return False
        
    # Backup the file
    if not backup_file(file_path):
        print(f"[!] Could not backup {file_name}, aborting modification.")
        return False
        
    try:
        # Read the file content
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Apply modifications
        modified = False
        
        # 1. Add imports
        if 'imports_add' in modifications:
            import_code = modifications['imports_add']
            # Find a good spot to add imports - after the last import statement
            import_lines = [i for i in content.split('\n') if i.startswith('import ') or i.startswith('from ')]
            if import_lines:
                last_import = import_lines[-1]
                content = content.replace(last_import, last_import + '\n' + import_code)
                modified = True
                print(f"[✓] Added imports to {file_name}")
                
        # 2. Add initialization code
        if 'initialization_add' in modifications:
            init_code = modifications['initialization_add']
            # Find a good spot to add initialization - typically after "# Initial updates"
            init_marker = "# Initial updates"
            update_complete_marker = "print(f\"✅ All components initialized successfully\")"
            
            if init_marker in content and update_complete_marker in content:
                split_point = content.find(update_complete_marker)
                if split_point > 0:
                    content = content[:split_point] + init_code + '\n        ' + content[split_point:]
                    modified = True
                    print(f"[✓] Added initialization code to {file_name}")
                    
        # 3. Replace specific code blocks
        for key, replacement in modifications.items():
            if key.endswith('_replace'):
                if isinstance(replacement, dict) and 'old' in replacement and 'new' in replacement:
                    old_code = replacement['old']
                    new_code = replacement['new']
                    
                    if old_code in content:
                        content = content.replace(old_code, new_code)
                        modified = True
                        print(f"[✓] Replaced {key} in {file_name}")
                    else:
                        print(f"[!] Warning: Could not find code to replace for {key} in {file_name}")
                        
        # 4. Add additional code at specific points
        for key, add_code in modifications.items():
            if key.endswith('_add') and key not in ['imports_add', 'initialization_add']:
                # Try to find a good insertion point
                if key == 'recommendations_filter_add':
                    marker = "# Log the full decision context"
                    if marker in content:
                        insert_pos = content.find(marker)
                        if insert_pos > 0:
                            content = content[:insert_pos] + add_code + '\n                ' + content[insert_pos:]
                            modified = True
                            print(f"[✓] Added {key} in {file_name}")
                    else:
                        print(f"[!] Warning: Could not find insertion point for {key} in {file_name}")
                        
                elif key == 'reset_risk_params_add':
                    marker = "# 14. Store market data for future analysis"
                    if marker in content:
                        insert_pos = content.find(marker)
                        if insert_pos > 0:
                            content = content[:insert_pos] + add_code + '\n            \n            ' + content[insert_pos:]
                            modified = True
                            print(f"[✓] Added {key} in {file_name}")
                    else:
                        print(f"[!] Warning: Could not find insertion point for {key} in {file_name}")
                        
        # 5. Modify method signatures and add context code
        if file_name == 'advanced_llm_prompting.py' and 'add_additional_context' in modifications:
            ctx_mod = modifications['add_additional_context']
            old_sig = ctx_mod['old_signature']
            new_sig = ctx_mod['new_signature']
            add_code = ctx_mod['add_context_code']
            
            # Replace method signature
            if old_sig in content:
                content = content.replace(old_sig, new_sig)
                modified = True
                print(f"[✓] Updated method signature in {file_name}")
                
                # Find a place to add the context code, before the return statement
                return_statement = "    return prompt, dominant_regime"
                if return_statement in content:
                    insert_pos = content.find(return_statement)
                    if insert_pos > 0:
                        # Find the try/except block for formatting the template
                        try_stmt = "    try:"
                        if try_stmt in content[:insert_pos]:
                            # Insert before the try block
                            try_pos = content[:insert_pos].rfind(try_stmt)
                            if try_pos > 0:
                                content = content[:try_pos] + add_code + '\n\n' + content[try_pos:]
                                modified = True
                                print(f"[✓] Added context handling code in {file_name}")
                            else:
                                print(f"[!] Warning: Could not find try block to add context code in {file_name}")
                        else:
                            print(f"[!] Warning: Could not find try block to add context code in {file_name}")
                else:
                    print(f"[!] Warning: Could not find return statement in {file_name}")
            else:
                print(f"[!] Warning: Could not find method signature in {file_name}")
                
        # Write the modified content back to the file
        if modified:
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"[✓] Successfully modified {file_name}")
            return True
        else:
            print(f"[!] No modifications were made to {file_name}")
            return False
            
    except Exception as e:
        print(f"[!] Error modifying {file_name}: {e}")
        return False

def ensure_llm_prompt_dir():
    """Ensure llm_prompts directory exists."""
    prompt_dir = os.path.join(SCRIPT_DIR, 'llm_prompts')
    if not os.path.exists(prompt_dir):
        try:
            os.makedirs(prompt_dir)
            print(f"[✓] Created llm_prompts directory")
            return True
        except Exception as e:
            print(f"[!] Error creating llm_prompts directory: {e}")
            return False
    return True

def ensure_data_dir():
    """Ensure data directory exists."""
    data_dir = os.path.join(SCRIPT_DIR, 'data')
    if not os.path.exists(data_dir):
        try:
            os.makedirs(data_dir)
            print(f"[✓] Created data directory")
            return True
        except Exception as e:
            print(f"[!] Error creating data directory: {e}")
            return False
    return True

def main():
    print("\n" + "="*80)
    print("🔄 MARGIN VALIDATION SYSTEM AUTO-IMPLEMENTATION")
    print("="*80)
    
    print("\nThis script will create and modify files to implement the margin validation system.")
    print(f"Backup directory: {BACKUP_DIR}")
    
    # Ask for confirmation
    confirm = input("\nDo you want to proceed with installation? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Installation cancelled.")
        return
        
    print("\nStarting implementation...\n")
    
    # Create backup directory
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
        print(f"[✓] Created backup directory: {BACKUP_DIR}")
        
    # Ensure required directories exist
    ensure_llm_prompt_dir()
    ensure_data_dir()
    
    # Create new files
    for file_name, content in NEW_FILES.items():
        create_new_file(file_name, content)
        
    # Modify existing files
    for file_name, modifications in MODIFIED_FILES.items():
        modify_file(file_name, modifications)
        
    print("\n" + "="*80)
    print("✅ IMPLEMENTATION COMPLETED")
    print("="*80)
    
    print("\nTo test the implementation:")
    print("1. Run the margin analysis demo: python margin_analysis_demo.py")
    print("2. Start your trading bot as usual: python main.py")
    print("\nThe system will now filter out instruments that don't meet margin requirements")
    print("and dynamically adjust risk parameters when needed.\n")

if __name__ == '__main__':
    main()