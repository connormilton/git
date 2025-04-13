# trade_executor.py

import os
import json
import logging
import decimal
from datetime import datetime, timezone
from csv import DictWriter
import time

logger = logging.getLogger("TradingBot")

class TradeLogger:
    """Logs trade details to CSV."""
    def __init__(self, filename):
        self.filename = filename
        self.fieldnames = [
            "timestamp", "epic", "direction", "size", "signal_price", "entry_price",
            "stop_level", "limit_level", "stop_distance", "limit_distance",
            "confidence", "estimated_risk_gbp", "status", "response_status",
            "reason", "response_reason", "deal_id", "pnl", "outcome", "raw_response"
        ]
        self._ensure_header()

    def _ensure_header(self):
        file_exists = os.path.isfile(self.filename)
        if not file_exists:
            try:
                os.makedirs(os.path.dirname(self.filename), exist_ok=True)
                with open(self.filename, mode="w", newline="", encoding="utf-8") as f:
                    writer = DictWriter(f, fieldnames=self.fieldnames)
                    writer.writeheader()
                    logger.info(f"Created trade log file: {self.filename}")
            except Exception as e:
                logger.error(f"Error ensuring trade log header {self.filename}: {e}")

    def log_trade(self, trade_data):
        log_entry = {field: trade_data.get(field, "") for field in self.fieldnames}
        log_entry["timestamp"] = datetime.now(timezone.utc).isoformat(timespec='seconds')
        try:
            with open(self.filename, mode="a", newline="", encoding="utf-8") as f:
                writer = DictWriter(f, fieldnames=self.fieldnames)
                for key, value in log_entry.items():
                    if isinstance(value, decimal.Decimal):
                        log_entry[key] = f"{value:.6f}"
                    elif isinstance(value, float):
                        log_entry[key] = f"{value:.6f}"
                writer.writerow(log_entry)
        except Exception as e:
            logger.error(f"Error logging trade: {e}", exc_info=True)

class ExecutionHandler:
    """Handles placing trades and logging outcomes."""
    def __init__(self, broker_interface, portfolio, config, trade_memory=None):
        self.broker = broker_interface
        self.portfolio = portfolio
        self.config = config
        self.trade_memory = trade_memory
        self.trade_logger = TradeLogger(config['TRADE_HISTORY_FILE'])

    def execute_new_trade(self, trade_details):
        epic = trade_details['epic']
        
        # Capture technical data if available
        technical_data = None
        market_regime = None
        if self.trade_memory and hasattr(self.trade_memory, 'data_provider'):
            try:
                technical_data = self.trade_memory.data_provider.get_latest_technicals(epic)
                market_regime = self.trade_memory.data_provider.get_market_regime(epic)
            except:
                pass
        
        # Log trade attempt
        log_attempt = trade_details.copy()
        log_attempt["status"] = "ATTEMPT_OPEN"
        log_attempt["outcome"] = "PENDING"
        self.trade_logger.log_trade(log_attempt)

        # Execute trade
        result = self.broker.execute_trade(trade_details)
        status = result.get('status', 'ERROR')
        reason = result.get('reason', 'Unknown')
        deal_id = result.get('dealId')
        success = (status == 'SUCCESS' and deal_id is not None)

        # Prepare trade outcome for logging
        log_outcome = trade_details.copy()
        log_outcome.update({
            "status": "EXECUTED" if success else "FAILED",
            "response_status": status,
            "response_reason": reason,
            "deal_id": deal_id or "N/A",
            "outcome": "OPENED" if success else "FAILED_OPEN",
            "raw_response": json.dumps(result)
        })
        
        # Log to CSV
        self.trade_logger.log_trade(log_outcome)
        
        # Add to portfolio's memory
        self.portfolio.add_trade_to_history(log_outcome)
        
        # Store in trade memory system if available
        if self.trade_memory and success:
            try:
                # Create trade record
                trade_record = {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'epic': epic,
                    'direction': trade_details['direction'],
                    'size': float(trade_details['size']),
                    'entry_price': float(trade_details.get('signal_price', 0)),
                    'stop_level': float(trade_details.get('stop_level', 0)) if trade_details.get('stop_level') else None,
                    'limit_level': float(trade_details.get('limit_level', 0)) if trade_details.get('limit_level') else None,
                    'confidence': trade_details.get('confidence', 'medium'),
                    'deal_id': deal_id,
                    'context': {
                        'reasoning': trade_details.get('reasoning', {}),
                        'risk_factors': trade_details.get('risk_factors', {})
                    }
                }
                # Store in database
                self.trade_memory.store_trade(trade_record, technical_data, market_regime)
            except Exception as e:
                logger.error(f"Error storing trade in memory: {e}")
        
        return success, deal_id if success else reason

    def close_trade(self, deal_id, size, direction, epic="Unknown"):
        # Log attempt
        log_attempt = {
            "deal_id": deal_id,
            "epic": epic,
            "direction": "CLOSE",
            "size": size,
            "status": "ATTEMPT_CLOSE",
            "outcome": "PENDING"
        }
        self.trade_logger.log_trade(log_attempt)

        # Execute close
        result = self.broker.close_trade(deal_id, size, direction)
        status = result.get('status', 'ERROR')
        reason = result.get('reason', 'Unknown')
        success = (status == 'SUCCESS')

        # Prepare outcome for logging
        log_outcome = {
            "deal_id": deal_id,
            "epic": epic,
            "direction": "CLOSE",
            "size": size,
            "status": "EXECUTED" if success else "FAILED",
            "response_status": status,
            "response_reason": reason,
            "outcome": "CLOSED" if success else "FAILED_CLOSE",
            "raw_response": json.dumps(result)
        }
        
        # Try to get PnL 
        pnl = None
        if success:
            try:
                # In a real implementation, you would get the actual PnL from the broker's response
                # For now we'll use a placeholder
                pnl = result.get('pnl', 0)
                log_outcome['pnl'] = pnl
            except:
                pass
        
        # Log to CSV
        self.trade_logger.log_trade(log_outcome)
        
        # Add to portfolio's memory
        self.portfolio.add_trade_to_history(log_outcome)
        
        # Update trade in memory system if available
        if self.trade_memory and success:
            try:
                # Get technical data if available
                technical_data = None
                market_regime = None
                if hasattr(self.trade_memory, 'data_provider'):
                    try:
                        technical_data = self.trade_memory.data_provider.get_latest_technicals(epic)
                        market_regime = self.trade_memory.data_provider.get_market_regime(epic)
                    except:
                        pass
                
                # Create trade record
                trade_record = {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'epic': epic,
                    'direction': 'CLOSE',
                    'size': float(size),
                    'exit_price': 0,  # Would be filled from broker response in real implementation
                    'pnl': pnl,
                    'outcome': 'PROFIT' if pnl and pnl > 0 else 'LOSS',
                    'deal_id': deal_id,
                    'reason': 'LLM_DECISION'
                }
                # Store in database
                self.trade_memory.store_trade(trade_record, technical_data, market_regime)
            except Exception as e:
                logger.error(f"Error storing trade close in memory: {e}")
        
        return success

    def amend_trade(self, deal_id, stop_level=None, limit_level=None, epic="Unknown"):
        # Log attempt
        log_attempt = {
            "deal_id": deal_id,
            "epic": epic,
            "direction": "AMEND",
            "status": "ATTEMPT_AMEND",
            "outcome": "PENDING",
            "stop_level": stop_level,
            "limit_level": limit_level
        }
        self.trade_logger.log_trade(log_attempt)

        # Validate inputs
        if stop_level is None and limit_level is None:
            logger.warning("Amend trade called with no stop or limit level.")
            return False
            
        # Execute amendment
        result = self.broker.amend_trade(deal_id, stop_level, limit_level)
        status = result.get('status', 'ERROR')
        reason = result.get('reason', 'Unknown')
        success = (status == 'SUCCESS')

        # Prepare outcome for logging
        log_outcome = {
            "deal_id": deal_id,
            "epic": epic,
            "direction": "AMEND",
            "status": "EXECUTED" if success else "FAILED",
            "response_status": status,
            "response_reason": reason,
            "outcome": "AMENDED" if success else "FAILED_AMEND",
            "stop_level": stop_level,
            "limit_level": limit_level,
            "raw_response": json.dumps(result)
        }
        
        # Log to CSV
        self.trade_logger.log_trade(log_outcome)
        
        # Add to portfolio's memory
        self.portfolio.add_trade_to_history(log_outcome)
        
        # Update trade record is an option, but not critical for amendments
        # since we'll capture it on close
        
        return success