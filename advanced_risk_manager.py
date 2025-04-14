# advanced_risk_manager.py

import logging
import decimal
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

logger = logging.getLogger("TradingBot")

class AdvancedRiskManager:
    """
    Advanced risk manager with dynamic position sizing, correlation analysis,
    and adaptive risk based on volatility and performance.
    """
    def __init__(self, config, trade_memory):
        self.config = config
        self.trade_memory = trade_memory
        self.balance = decimal.Decimal('0.0')
        self.available_funds = decimal.Decimal('0.0')
        self.account_currency = config['ACCOUNT_CURRENCY']
        
        # Default Value Per Point mappings (used as fallback)
        self.APPROX_VPP_GBP = {
            "CS.D.EURUSD.MINI.IP": decimal.Decimal("0.81"),
            "CS.D.USDJPY.MINI.IP": decimal.Decimal("0.74"),
            "CS.D.GBPUSD.MINI.IP": decimal.Decimal("0.81"),
            "CS.D.AUDUSD.MINI.IP": decimal.Decimal("0.81"),
            "CS.D.USDCAD.MINI.IP": decimal.Decimal("0.77"),
            "CS.D.EURGBP.MINI.IP": decimal.Decimal("1.26"),
            "CS.D.EURJPY.MINI.IP": decimal.Decimal("0.94"),
            "CS.D.GBPJPY.MINI.IP": decimal.Decimal("0.94"),
            "CS.D.AUDJPY.MINI.IP": decimal.Decimal("0.94"),
        }
        
        # Risk parameters with defaults
        self.base_risk_per_trade = config.get('RISK_PER_TRADE_PERCENT', decimal.Decimal('2.5'))
        self.max_total_risk = config.get('MAX_TOTAL_RISK_PERCENT', decimal.Decimal('30.0'))
        self.per_currency_risk_cap = config.get('PER_CURRENCY_RISK_CAP', decimal.Decimal('10.0'))
        self.margin_buffer = config.get('MARGIN_BUFFER_FACTOR', decimal.Decimal('0.80'))
        
        # Default kelly fraction to apply (0.5 = half-kelly)
        self.kelly_fraction = decimal.Decimal('0.5')
        
        # Track open positions and risk
        self.open_position_risk = {}
        self.total_open_risk = decimal.Decimal('0.0')
        self.total_open_risk_percent = decimal.Decimal('0.0')  # Add this field
        self.currency_exposure = {}
        
        # Correlation matrix cache
        self.correlation_matrix = {}
        self.last_correlation_update = datetime.now(timezone.utc) - timedelta(hours=1)
        
        # Performance adjustment factors
        self.performance_multiplier = decimal.Decimal('1.0')
        self.volatility_adjustment = {}
        
        # Expected trade outcomes from history
        self.expected_outcomes = {}

    def update_account(self, portfolio):
        """Update account balance and available funds from portfolio."""
        self.balance = portfolio.get_balance()
        self.available_funds = portfolio.get_available_funds()
        
        # Update open position risk
        self.update_open_position_risk(portfolio)
        
        # Calculate total risk percent
        if self.balance > 0:
            self.total_open_risk_percent = (self.total_open_risk / self.balance) * 100
        else:
            self.total_open_risk_percent = decimal.Decimal('0.0')
            
        logger.info(f"Risk manager updated: Balance={self.balance:.2f}, Risk Exposure={self.total_open_risk_percent:.2f}%")
        
        # Update performance multiplier based on recent results
        self._update_performance_multiplier()

    def update_open_position_risk(self, portfolio_state):
        """Calculate and update current risk from open positions."""
        open_positions = portfolio_state.get_open_positions_df()
        self.open_position_risk = {}
        self.total_open_risk = decimal.Decimal('0.0')
        self.currency_exposure = {}
        
        if open_positions.empty:
            return
            
        for _, pos in open_positions.iterrows():
            epic = pos.get('epic')
            direction = pos.get('direction')
            size = pos.get('size')
            level = pos.get('level')
            stop_level = pos.get('stopLevel')
            
            if not all([epic, direction, size is not None, level is not None, stop_level is not None]):
                logger.warning(f"Skipping risk calculation for position with missing data: {pos}")
                continue
                
            # Get value per point for risk calculation
            vpp = self._get_value_per_point_for_position(pos)
            
            # Calculate risk in account currency
            risk_per_point = size * vpp
            if direction == 'BUY':
                points_at_risk = level - stop_level
            else:  # SELL
                points_at_risk = stop_level - level
                
            if points_at_risk <= 0:
                logger.warning(f"Position {epic} has negative or zero points at risk. Risk calculation skipped.")
                continue
                
            position_risk = risk_per_point * points_at_risk
            
            # Store calculated risk
            self.open_position_risk[epic] = {
                'size': size,
                'direction': direction,
                'entry_level': level,
                'stop_level': stop_level,
                'vpp': vpp,
                'risk_amount': position_risk,
                'risk_percent': (position_risk / self.balance * 100) if self.balance > 0 else decimal.Decimal('0.0')
            }
            
            # Update total risk
            self.total_open_risk += position_risk
            
            # Update currency exposure
            currency_pair = self._extract_currency_pair(epic)
            if currency_pair:
                base, quote = currency_pair
                
                # Track exposure for both currencies in the pair
                for currency in [base, quote]:
                    if currency not in self.currency_exposure:
                        self.currency_exposure[currency] = decimal.Decimal('0.0')
                    
                    # For simplicity, we're treating the risk as the exposure amount
                    # This is actually a simplification - in reality you'd calculate actual exposure
                    self.currency_exposure[currency] += position_risk / 2  # Split between both currencies
        
        logger.info(f"Current open risk: {self.total_open_risk:.2f} ({(self.total_open_risk / self.balance * 100):.2f}% of account)")

    def _get_value_per_point_for_position(self, position):
        """Get the value per point for a position."""
        epic = position.get('epic')
        
        # First, check if the position already has VPP info
        if hasattr(position, 'valuePerPoint') and position.valuePerPoint:
            return decimal.Decimal(str(position.valuePerPoint))
            
        # Use our approximation mapping
        if epic in self.APPROX_VPP_GBP:
            return self.APPROX_VPP_GBP[epic]
            
        # Default fallback
        logger.warning(f"Could not determine VPP for {epic}, using default of 1.0")
        return decimal.Decimal('1.0')

    def _extract_currency_pair(self, epic):
        """Extract base and quote currencies from epic."""
        if not isinstance(epic, str):
            return None, None
            
        # Handle known non-currency pairs
        if epic in ['CS.D.USCGC.TODAY.IP']:  # Gold
            return ('XAU', 'USD')
        if epic in ['IX.D.FTSE.DAILY.IP']:  # FTSE
            return (None, 'GBP')
            
        # Try to extract from regular pattern
        parts = epic.split('.')
        if len(parts) > 2 and parts[0] == 'CS' and parts[1] == 'D':
            pair = parts[2]
            if len(pair) == 6 and pair.isupper():
                return (pair[:3], pair[3:])
                
        return None, None

    def _update_performance_multiplier(self):
        """Update performance-based risk multiplier."""
        try:
            # Get win rate and profit factor from trade memory
            win_rate = self.trade_memory.calculate_win_rate(days=30)
            profit_factor = self.trade_memory.calculate_profit_factor(days=30)
            
            # Default is 1.0 (no adjustment)
            multiplier = decimal.Decimal('1.0')
            
            # Adjust based on win rate and profit factor
            if win_rate > 0.6 and profit_factor > 1.5:
                # Good performance - can increase risk slightly
                multiplier = decimal.Decimal('1.2')
            elif win_rate < 0.4 or profit_factor < 0.8:
                # Poor performance - reduce risk
                multiplier = decimal.Decimal('0.8')
                
            self.performance_multiplier = multiplier
            logger.info(f"Performance multiplier updated: {multiplier} (Win Rate: {win_rate:.2f}, Profit Factor: {profit_factor:.2f})")
            
        except Exception as e:
            logger.error(f"Error updating performance multiplier: {e}")
            self.performance_multiplier = decimal.Decimal('1.0')

    def update_volatility_adjustments(self, data_provider):
        """Update volatility-based risk adjustments."""
        try:
            # Fetch volatility for active instruments
            for epic in list(self.open_position_risk.keys()):
                technicals = data_provider.get_latest_technicals(epic)
                
                if technicals and 'Volatility_20' in technicals:
                    volatility = decimal.Decimal(str(technicals['Volatility_20']))
                    
                    # Base adjustment on relative volatility (1.0 is normal)
                    normal_volatility = decimal.Decimal('1.0')
                    ratio = volatility / normal_volatility
                    
                    # Reduce position size for higher volatility
                    adjustment = decimal.Decimal('1.0')
                    if ratio > decimal.Decimal('1.5'):
                        adjustment = decimal.Decimal('0.7')  # High volatility - reduce risk
                    elif ratio < decimal.Decimal('0.5'):
                        adjustment = decimal.Decimal('1.2')  # Low volatility - can increase risk slightly
                        
                    self.volatility_adjustment[epic] = adjustment
                    logger.debug(f"Volatility adjustment for {epic}: {adjustment} (Volatility: {volatility})")
            
        except Exception as e:
            logger.error(f"Error updating volatility adjustments: {e}")

    def calculate_optimal_size_kelly(self, win_rate, win_loss_ratio, risk_percentage):
        """
        Calculate optimal position size using the Kelly Criterion.
        
        Parameters:
        - win_rate: Probability of winning (0-1)
        - win_loss_ratio: Average win / average loss ratio
        - risk_percentage: Base risk percentage to adjust
        
        Returns adjusted size as a percentage of account balance.
        """
        try:
            # Convert to Decimal for precise calculation
            win_rate_dec = decimal.Decimal(str(win_rate))
            win_loss_ratio_dec = decimal.Decimal(str(win_loss_ratio))
            
            # Kelly formula: f* = p - (1-p)/R where p=win probability, R=win/loss ratio
            loss_rate = decimal.Decimal('1.0') - win_rate_dec
            
            # Avoid division by zero
            if win_loss_ratio_dec <= 0:
                return decimal.Decimal('0.0')
                
            kelly_percentage = win_rate_dec - (loss_rate / win_loss_ratio_dec)
            
            # Apply Kelly fraction to avoid over-betting
            kelly_percentage = kelly_percentage * self.kelly_fraction
            
            # Constrain to reasonable range
            kelly_percentage = max(decimal.Decimal('0.0'), min(kelly_percentage, decimal.Decimal('0.05')))
            
            # Base risk can't be zero
            if risk_percentage <= 0:
                return decimal.Decimal('0.0')
                
            # Convert risk percentage to actual risk
            base_risk_decimal = risk_percentage / decimal.Decimal('100.0')
            
            # Adjust base risk using kelly
            adjusted_risk = base_risk_decimal * kelly_percentage * decimal.Decimal('2.0')
            
            # Cap at the original risk level
            return min(adjusted_risk, base_risk_decimal)
            
        except Exception as e:
            logger.error(f"Error calculating Kelly criterion: {e}")
            # Return default risk as fallback
            return risk_percentage / decimal.Decimal('100.0')

    def _get_expected_outcome(self, epic, trade_direction):
        """Get expected win rate and win/loss ratio for a specific trade setup."""
        try:
            key = f"{epic}_{trade_direction}"
            
            # Check if we have cached data
            if key in self.expected_outcomes:
                return self.expected_outcomes[key]
                
            # Get historical trades for this asset and direction
            history = self.trade_memory.get_trade_history_summary(days=90)
            matched_history = [h for h in history if h['epic'] == epic and h['direction'] == trade_direction]
            
            if matched_history:
                history_item = matched_history[0]
                total_trades = history_item.get('total_trades', 0)
                
                if total_trades >= 5:  # Need minimum sample size
                    win_rate = history_item.get('winning_trades', 0) / total_trades
                    avg_win = abs(history_item.get('avg_win', 0)) or 1.0
                    avg_loss = abs(history_item.get('avg_loss', 0)) or 1.0
                    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0
                    
                    self.expected_outcomes[key] = {
                        'win_rate': win_rate,
                        'win_loss_ratio': win_loss_ratio,
                        'sample_size': total_trades
                    }
                    
                    return self.expected_outcomes[key]
            
            # Get overall market win rate as fallback
            overall_win_rate = self.trade_memory.calculate_win_rate(days=90)
            
            # Default expectations (conservative)
            default_values = {
                'win_rate': overall_win_rate or 0.5,
                'win_loss_ratio': 1.0,
                'sample_size': 0
            }
            
            self.expected_outcomes[key] = default_values
            return default_values
            
        except Exception as e:
            logger.error(f"Error getting expected outcomes: {e}")
            # Return conservative defaults
            return {'win_rate': 0.5, 'win_loss_ratio': 1.0, 'sample_size': 0}

    def calculate_correlation_adjustment(self, epic, direction, data_provider):
        """Calculate position size adjustment based on portfolio correlation."""
        try:
            # Skip if no open positions
            if not self.open_position_risk:
                return decimal.Decimal('1.0')
                
            # Check if we need to update correlation matrix
            now = datetime.now(timezone.utc)
            if (now - self.last_correlation_update) > timedelta(hours=6):
                # Get active epics for correlation calculation
                active_epics = list(self.open_position_risk.keys()) + [epic]
                if len(active_epics) >= 2:  # Need at least 2 for correlation
                    self.correlation_matrix = data_provider.get_correlation_matrix(active_epics)
                    self.last_correlation_update = now
            
            # Skip if no correlation data
            if not self.correlation_matrix or epic not in self.correlation_matrix:
                return decimal.Decimal('1.0')
                
            # Calculate average correlation with existing positions
            correlations = []
            for open_epic, position in self.open_position_risk.items():
                if open_epic != epic and open_epic in self.correlation_matrix.get(epic, {}):
                    correlation = decimal.Decimal(str(self.correlation_matrix[epic][open_epic]))
                    
                    # Convert correlation based on position directions
                    # If directions are opposite, we invert the correlation
                    if position['direction'] != direction:
                        correlation = -correlation
                        
                    correlations.append(correlation)
            
            if not correlations:
                return decimal.Decimal('1.0')
                
            # Average correlation with open positions
            avg_correlation = sum(correlations) / len(correlations)
            
            # Adjust size based on correlation: higher correlation = lower size
            # Highly correlated (>0.7): reduce to 70% size
            # Negatively correlated (<-0.3): increase to 120% size
            if avg_correlation > decimal.Decimal('0.7'):
                return decimal.Decimal('0.7')
            elif avg_correlation < decimal.Decimal('-0.3'):
                return decimal.Decimal('1.2')
            else:
                # Linear interpolation between points
                slope = decimal.Decimal('0.5') / decimal.Decimal('1.0')  # Change of 0.5 over a range of 1.0
                adjustment = decimal.Decimal('1.0') - (avg_correlation * slope)
                return adjustment
                
        except Exception as e:
            logger.error(f"Error calculating correlation adjustment: {e}")
            return decimal.Decimal('1.0')

    def check_drawdown_limit(self):
        """Check if we've hit maximum drawdown limit."""
        try:
            # Get recent performance metrics
            dashboard = self.trade_memory.get_performance_dashboard(timeframe='monthly')
            
            # Stop trading if experiencing severe drawdown (can define severe as needed)
            # For example, stop if lost more than 10% in a month
            current_balance = self.balance
            
            # Get starting balance from a month ago (if available)
            # This could come from trade history or be calculated from current + net P&L
            monthly_pnl = 0
            for instrument_data in dashboard.get('instruments', {}).values():
                monthly_pnl += instrument_data.get('net_pnl', 0)
                
            if monthly_pnl < 0:
                drawdown_percent = abs(monthly_pnl) / (current_balance - monthly_pnl) * 100
                
                # If severe drawdown, signal to reduce risk
                if drawdown_percent > 10:  # 10% drawdown
                    logger.warning(f"Severe drawdown detected: {drawdown_percent:.2f}%. Reducing risk.")
                    return True, drawdown_percent
            
            return False, 0
            
        except Exception as e:
            logger.error(f"Error checking drawdown limit: {e}")
            return False, 0

    def calculate_trade_details(self, proposed_trade, instrument_details, broker, data_provider=None):
        """
        Calculate optimal trade size and risk based on multiple factors.
        
        This is the main method that implements the advanced position sizing.
        """
        epic = proposed_trade['symbol']
        logger.debug(f"Calculating advanced trade details for: {epic}, Proposal: {proposed_trade}")

        stop_distance_pips = proposed_trade.get('stop_loss_pips')
        limit_pips = proposed_trade.get('limit_pips')
        signal_price = proposed_trade.get('signal_price')
        direction = proposed_trade.get('direction')

        if not isinstance(stop_distance_pips, decimal.Decimal) or stop_distance_pips <= 0:
            return None, "Invalid stop loss distance"
        if not isinstance(signal_price, decimal.Decimal):
            return None, "Invalid signal price"

        if limit_pips is not None and (not isinstance(limit_pips, decimal.Decimal) or limit_pips <= 0):
            limit_pips = None

        vpp = self._get_value_per_point(instrument_details)
        if vpp is None or vpp <= 0:
            return None, f"Invalid Value Per Point for {epic}"

        # Get expected outcomes for this asset/direction
        expected_outcome = self._get_expected_outcome(epic, direction)
        win_rate = expected_outcome['win_rate']
        win_loss_ratio = expected_outcome['win_loss_ratio']
        
        # 1. Base risk as percentage of account
        base_risk_pct = self.base_risk_per_trade
        
        # 2. Adjust based on trade confidence
        confidence = proposed_trade.get('confidence', 'medium').lower()
        conf_mults = self.config.get('CONFIDENCE_RISK_MULTIPLIERS', {})
        confidence_multiplier = conf_mults.get(confidence, decimal.Decimal('1.0'))
        
        # 3. Adjust based on system performance
        risk_pct = base_risk_pct * confidence_multiplier * self.performance_multiplier
        
        # 4. Apply Kelly criterion if we have sufficient historical data
        if expected_outcome['sample_size'] >= 10:
            kelly_risk = self.calculate_optimal_size_kelly(win_rate, win_loss_ratio, risk_pct)
            # Only use Kelly if it doesn't reduce risk too drastically
            risk_pct = max(kelly_risk, risk_pct * decimal.Decimal('0.5'))
        
        # 5. Check for severe drawdown and reduce risk if necessary
        drawdown_detected, _ = self.check_drawdown_limit()
        if drawdown_detected:
            risk_pct = risk_pct * decimal.Decimal('0.5')  # Reduce risk by half during drawdowns
            
        # 6. Apply correlation adjustment if we have data provider
        correlation_adjustment = decimal.Decimal('1.0')
        if data_provider:
            correlation_adjustment = self.calculate_correlation_adjustment(epic, direction, data_provider)
            risk_pct = risk_pct * correlation_adjustment
            
        # 7. Apply volatility adjustment if available
        volatility_adjustment = self.volatility_adjustment.get(epic, decimal.Decimal('1.0'))
        risk_pct = risk_pct * volatility_adjustment
        
        # 8. Convert risk percentage to monetary amount
        if self.balance <= 0:
            return None, "Zero or negative account balance"
            
        risk_pct_decimal = risk_pct / decimal.Decimal('100.0')
        max_risk_target_acc_ccy = self.balance * risk_pct_decimal
        
        # 9. Calculate position size based on risk and stop distance
        try:
            calculated_size = max_risk_target_acc_ccy / (stop_distance_pips * vpp)
            calculated_size = calculated_size.quantize(decimal.Decimal("0.01"), rounding=decimal.ROUND_DOWN)
            if calculated_size <= 0:
                logger.warning(f"Calculated size for {epic} is zero after rounding.")
                calculated_size = decimal.Decimal('0')
        except (decimal.InvalidOperation, ZeroDivisionError) as e:
            logger.error(f"Size calc error {epic}: {e}")
            return None, "Size calculation error"

        # 10. Ensure minimum deal size
        min_deal_size = instrument_details.get('minDealSize', decimal.Decimal("0.01"))
        final_size = max(min_deal_size, calculated_size)
        if calculated_size < min_deal_size:
            logger.warning(f"Calculated size {calculated_size:.2f} < Min {min_deal_size} for {epic}. Using Min.")

        # 11. Check margin requirements and potentially adjust size
        margin_factor = instrument_details.get('marginFactor')
        if margin_factor is not None and margin_factor > 0 and signal_price > 0:
            # Bypass margin check
            logger.info(f"Bypassing margin check for {epic}")

        # 12. Calculate final risk with adjusted size
        estimated_risk_acc_ccy = final_size * stop_distance_pips * vpp

        # 13. Calculate stop and limit levels
        stop_level_abs = None
        limit_level_abs = None

        if direction == 'BUY':
            stop_level_abs = float(signal_price - stop_distance_pips)
            if limit_pips:
                limit_level_abs = float(signal_price + limit_pips)
        else:  # SELL
            stop_level_abs = float(signal_price + stop_distance_pips)
            if limit_pips:
                limit_level_abs = float(signal_price - limit_pips)

        # 14. Create final trade details
        final_details = {
            'epic': epic,
            'direction': direction,
            'size': final_size,
            'stop_level': stop_level_abs,
            'limit_level': limit_level_abs,
            'order_type': 'MARKET',
            'estimated_risk_gbp': float(estimated_risk_acc_ccy.quantize(decimal.Decimal("0.01"))),
            'confidence': confidence,
            'signal_price': signal_price,
            # Add advanced sizing factors for analysis
            'risk_factors': {
                'base_risk_pct': float(base_risk_pct),
                'confidence_mult': float(confidence_multiplier),
                'performance_mult': float(self.performance_multiplier),
                'correlation_adj': float(correlation_adjustment),
                'volatility_adj': float(volatility_adjustment),
                'final_risk_pct': float(risk_pct),
                'expected_win_rate': float(win_rate),
                'win_loss_ratio': float(win_loss_ratio)
            }
        }
        
        # Log detailed trade sizing info
        logger.info(
            f"Advanced sizing for {epic} {direction}: Size={final_size} "
            f"(Base risk: {base_risk_pct}%, Final risk: {risk_pct}%, "
            f"Est. risk amount: {estimated_risk_acc_ccy:.2f})"
        )
        
        return final_details, None

    def _get_value_per_point(self, instrument_details):
        """Get Value Per Point for an instrument."""
        epic = instrument_details.get('epic')
        target_currency = self.account_currency
        vpp_direct = instrument_details.get('valuePerPoint')
        vpp_currency_hint = instrument_details.get('valuePerPointCurrency')

        logger.debug(f"Getting VPP for {epic}. Direct: {vpp_direct}, Currency: {vpp_currency_hint}, Target: {target_currency}")
        if vpp_direct and vpp_direct > 0:
            if vpp_currency_hint == target_currency:
                logger.info(f"Using direct VPP from broker for {epic}: {vpp_direct:.4f} {target_currency}")
                return vpp_direct
            else:
                logger.warning(f"Broker VPP found for {epic} ({vpp_direct:.4f} {vpp_currency_hint}) but needs conversion. NOT IMPLEMENTED. Falling back.")
        if epic in self.APPROX_VPP_GBP:
            approx_vpp = self.APPROX_VPP_GBP[epic]
            logger.warning(f"Using hardcoded approx VPP for {epic}: {approx_vpp:.4f} {target_currency}")
            return approx_vpp

        logger.error(f"Cannot determine VPP for {epic}. Using placeholder VPP=1.0. Sizing may be incorrect.")
        return decimal.Decimal("1.0")

    def check_portfolio_constraints(self, calculated_trade, instrument_details, portfolio_state, data_provider=None):
        """Always approve trades by bypassing all portfolio constraints."""
        epic = calculated_trade['epic']
        direction = calculated_trade['direction']
        
        logger.info(f"Bypassing ALL portfolio constraints for {epic} {direction}")
        
        # Always return True to approve the trade
        return True, "All portfolio constraints bypassed"