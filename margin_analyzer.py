# margin_analyzer.py

import logging
import decimal
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from datetime import datetime

logger = logging.getLogger("TradingBot")

class MarginRequirementAnalyzer:
    """
    Advanced analysis of margin requirements for trading instruments.
    Provides solutions for margin-related issues and alternative strategies.
    """
    
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
        """
        Analyze all instruments in the config for margin requirements.
        
        Returns:
            DataFrame with analysis of all instruments
        """
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
        """
        Calculate the minimum account balance required to trade this instrument.
        
        Args:
            epic: Instrument epic
            stop_distance: Desired stop distance in points
            
        Returns:
            Minimum required balance
        """
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
        """
        Suggest alternative strategies based on current balance.
        
        Args:
            current_balance: Current account balance
            
        Returns:
            Dictionary with suggested strategies
        """
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
        """Get Value Per Point for an instrument."""
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
        """Get minimum deal size for an instrument."""
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
        """Get margin factor for an instrument."""
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