# margin_validator.py

import logging
import decimal
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger("TradingBot")

class MarginValidator:
    """
    Pre-validates trading opportunities based on margin requirements before analysis.
    This ensures we don't waste resources analyzing trades that can't be executed.
    """
    
    def __init__(self, config, broker, portfolio):
        """
        Initialize the margin validator.
        
        Args:
            config: Configuration dictionary
            broker: Broker interface for getting instrument details
            portfolio: Portfolio object for account information
        """
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
        """Get the value per point for an instrument."""
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
        """Get the minimum deal size for an instrument."""
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
        """Get the margin factor for an instrument."""
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
        """
        Check if a trade would be viable based on margin requirements and minimum deal size.
        
        Args:
            epic: The instrument to check
            stop_distance: Planned stop distance in points
            
        Returns:
            Tuple of (is_viable, reason, details)
            - is_viable: Boolean indicating if trade meets margin requirements
            - reason: Description of why trade is/isn't viable
            - details: Dictionary with calculated values like min_size, required_margin, etc.
        """
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
        """
        Filter a list of instruments to only those that meet margin requirements.
        
        Args:
            candidate_pairs: List of instrument epics to check
            min_stop_distance: Minimum stop distance to use in viability check
            
        Returns:
            List of dictionaries with viable instruments and their details
        """
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
        """
        Calculate adaptive risk parameters based on viable instruments.
        
        Args:
            viable_instruments: List of viable instrument details
            
        Returns:
            Dictionary with adaptive risk parameters
        """
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