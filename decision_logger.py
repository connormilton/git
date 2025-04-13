# decision_logger.py

import os
import json
import logging
from datetime import datetime, timezone
import pandas as pd

logger = logging.getLogger("TradingBot")

class DecisionLogger:
    """Logs detailed LLM decisions and outcomes for review and self-learning."""
    
    def __init__(self, config):
        self.config = config
        self.decisions_dir = config.get('DECISIONS_LOG_DIR', 'data/decisions')
        self.review_file = os.path.join(self.decisions_dir, 'decision_reviews.csv')
        
        # Ensure directory exists
        os.makedirs(self.decisions_dir, exist_ok=True)
        
        # Initialize review file if it doesn't exist
        if not os.path.exists(self.review_file):
            self._init_review_file()
    
    def _init_review_file(self):
        """Initialize the decision review file with headers."""
        headers = [
            "decision_id", "timestamp", "market_regime", "instrument",
            "action_type", "action_details", "reasoning", "outcome",
            "pnl", "outcome_notes", "quality_score", "reviewed"
        ]
        
        df = pd.DataFrame(columns=headers)
        df.to_csv(self.review_file, index=False)
        logger.info(f"Initialized decision review file: {self.review_file}")
    
    def log_decisions(self, llm_response, raw_response, prompt, market_regime):
        """Log the full LLM decision context for later review."""
        # Create a unique ID for this decision set
        timestamp = datetime.now(timezone.utc).isoformat()
        decision_id = f"decision_{timestamp.replace(':', '-').replace('.', '-')}"
        
        # Prepare the full decision context
        decision_data = {
            "decision_id": decision_id,
            "timestamp": timestamp,
            "market_regime": market_regime,
            "llm_decisions": llm_response,
            "raw_llm_response": raw_response,
            "prompt_used": prompt,
            "trade_executions": [],  # Will be populated later
            "amendments_executed": [],  # Will be populated later
            "review_notes": "",
            "outcome_summary": {}
        }
        
        # Save as JSON file
        decision_file = os.path.join(self.decisions_dir, f"{decision_id}.json")
        with open(decision_file, 'w', encoding='utf-8') as f:
            json.dump(decision_data, indent=2, default=str, fp=f)
        
        # Log in review file for each action/amendment
        self._log_to_review_file(decision_id, timestamp, market_regime, llm_response)
        
        logger.info(f"Logged decision set {decision_id} for later review")
        return decision_id
    
    def _log_to_review_file(self, decision_id, timestamp, market_regime, llm_response):
        """Add entries to the review CSV file for each decision."""
        review_entries = []
        
        # Process trade actions
        for action in llm_response.get("tradeActions", []):
            epic = action.get("epic", "unknown")
            action_type = "NEW_TRADE"
            action_details = json.dumps({
                "direction": action.get("action"),
                "stop_distance": action.get("stop_distance"),
                "limit_distance": action.get("limit_distance"),
                "confidence": action.get("confidence")
            })
            reasoning = llm_response.get("reasoning", {}).get(epic, "")
            
            review_entries.append({
                "decision_id": decision_id,
                "timestamp": timestamp,
                "market_regime": market_regime,
                "instrument": epic,
                "action_type": action_type,
                "action_details": action_details,
                "reasoning": reasoning,
                "outcome": "PENDING",
                "pnl": 0,
                "outcome_notes": "",
                "quality_score": 0,
                "reviewed": False
            })
        
        # Process trade amendments
        for amend in llm_response.get("tradeAmendments", []):
            epic = amend.get("epic", "unknown")
            action_type = f"AMENDMENT_{amend.get('action', 'UNKNOWN')}"
            
            details = {}
            if amend.get("action") == "AMEND":
                details = {
                    "new_stop_distance": amend.get("new_stop_distance"),
                    "new_limit_distance": amend.get("new_limit_distance")
                }
            
            action_details = json.dumps(details)
            reasoning = llm_response.get("reasoning", {}).get(epic, "")
            
            review_entries.append({
                "decision_id": decision_id,
                "timestamp": timestamp,
                "market_regime": market_regime,
                "instrument": epic,
                "action_type": action_type,
                "action_details": action_details,
                "reasoning": reasoning,
                "outcome": "PENDING",
                "pnl": 0,
                "outcome_notes": "",
                "quality_score": 0,
                "reviewed": False
            })
        
        # Add global market assessment as a separate entry
        global_reasoning = llm_response.get("reasoning", {}).get("global")
        if global_reasoning:
            review_entries.append({
                "decision_id": decision_id,
                "timestamp": timestamp,
                "market_regime": market_regime,
                "instrument": "GLOBAL",
                "action_type": "MARKET_ASSESSMENT",
                "action_details": "",
                "reasoning": global_reasoning,
                "outcome": "N/A",
                "pnl": 0,
                "outcome_notes": "",
                "quality_score": 0,
                "reviewed": False
            })
        
        # Append to review file
        if review_entries:
            df = pd.DataFrame(review_entries)
            df.to_csv(self.review_file, mode='a', header=False, index=False)
    
    def update_execution_result(self, decision_id, instrument, action_type, 
                                execution_result, trade_details=None):
        """Update a decision with execution results."""
        # Find the decision file
        decision_file = os.path.join(self.decisions_dir, f"{decision_id}.json")
        if not os.path.exists(decision_file):
            logger.warning(f"Decision file not found for ID {decision_id}")
            return False
        
        # Load the decision data
        with open(decision_file, 'r', encoding='utf-8') as f:
            decision_data = json.load(f)
        
        # Add execution result
        execution_info = {
            "instrument": instrument,
            "action_type": action_type,
            "execution_result": execution_result,
            "execution_timestamp": datetime.now(timezone.utc).isoformat(),
            "trade_details": trade_details
        }
        
        if action_type.startswith("NEW_TRADE"):
            decision_data["trade_executions"].append(execution_info)
        else:
            decision_data["amendments_executed"].append(execution_info)
        
        # Save updated data
        with open(decision_file, 'w', encoding='utf-8') as f:
            json.dump(decision_data, indent=2, default=str, fp=f)
        
        # Update review file entry
        self._update_review_entry(decision_id, instrument, action_type, 
                                 "EXECUTED" if execution_result.get("status") == "SUCCESS" else "FAILED",
                                 execution_result)
        
        logger.info(f"Updated decision {decision_id} with execution result for {instrument}")
        return True
    
    def update_trade_outcome(self, decision_id, instrument, pnl, outcome_status, notes=""):
        """Update a decision with final trade outcome."""
        # Find the decision file
        decision_file = os.path.join(self.decisions_dir, f"{decision_id}.json")
        if not os.path.exists(decision_file):
            logger.warning(f"Decision file not found for ID {decision_id}")
            return False
        
        # Load the decision data
        with open(decision_file, 'r', encoding='utf-8') as f:
            decision_data = json.load(f)
        
        # Add outcome to the right trade execution
        for trade in decision_data["trade_executions"]:
            if trade["instrument"] == instrument:
                trade["final_outcome"] = {
                    "pnl": pnl,
                    "status": outcome_status,
                    "notes": notes,
                    "closed_timestamp": datetime.now(timezone.utc).isoformat()
                }
        
        # Update outcome summary
        if "outcome_summary" not in decision_data:
            decision_data["outcome_summary"] = {}
            
        if instrument not in decision_data["outcome_summary"]:
            decision_data["outcome_summary"][instrument] = []
            
        decision_data["outcome_summary"][instrument].append({
            "pnl": pnl,
            "status": outcome_status,
            "notes": notes,
            "closed_timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Save updated data
        with open(decision_file, 'w', encoding='utf-8') as f:
            json.dump(decision_data, indent=2, default=str, fp=f)
        
        # Update review file entry
        self._update_review_entry(decision_id, instrument, "NEW_TRADE", 
                                 outcome_status, None, pnl, notes)
        
        logger.info(f"Updated decision {decision_id} with final outcome for {instrument}: {outcome_status}, PnL: {pnl}")
        return True
    
    def _update_review_entry(self, decision_id, instrument, action_type, 
                            outcome, execution_result=None, pnl=None, notes=None):
        """Update the corresponding entry in the review CSV file."""
        try:
            # Read the current review file
            df = pd.read_csv(self.review_file)
            
            # Find the matching row
            mask = (df['decision_id'] == decision_id) & \
                   (df['instrument'] == instrument) & \
                   (df['action_type'] == action_type)
            
            if not any(mask):
                logger.warning(f"No matching review entry found for {decision_id}, {instrument}, {action_type}")
                return
            
            # Update the row
            df.loc[mask, 'outcome'] = outcome
            
            if pnl is not None:
                df.loc[mask, 'pnl'] = pnl
                
            if notes is not None:
                df.loc[mask, 'outcome_notes'] = notes
                
            if execution_result is not None:
                # Add execution details to notes
                current_notes = df.loc[mask, 'outcome_notes'].values[0]
                execution_summary = f"Execution: {execution_result.get('status')} - {execution_result.get('reason', 'N/A')}"
                
                if current_notes:
                    updated_notes = f"{current_notes}; {execution_summary}"
                else:
                    updated_notes = execution_summary
                    
                df.loc[mask, 'outcome_notes'] = updated_notes
            
            # Save updated file
            df.to_csv(self.review_file, index=False)
            
        except Exception as e:
            logger.error(f"Error updating review entry: {e}")
    
    def get_pending_reviews(self):
        """Get decisions that need review."""
        try:
            df = pd.read_csv(self.review_file)
            pending = df[(df['outcome'] != 'PENDING') & (~df['reviewed'])]
            return pending.to_dict('records')
        except Exception as e:
            logger.error(f"Error getting pending reviews: {e}")
            return []
    
    def add_review(self, decision_id, instrument, action_type, quality_score, notes):
        """Add a review for a decision."""
        try:
            # Read the current review file
            df = pd.read_csv(self.review_file)
            
            # Find the matching row
            mask = (df['decision_id'] == decision_id) & \
                   (df['instrument'] == instrument) & \
                   (df['action_type'] == action_type)
            
            if not any(mask):
                logger.warning(f"No matching review entry found for {decision_id}, {instrument}, {action_type}")
                return False
            
            # Update the row
            df.loc[mask, 'quality_score'] = quality_score
            df.loc[mask, 'reviewed'] = True
            
            # Append to existing notes
            current_notes = df.loc[mask, 'outcome_notes'].values[0]
            if current_notes:
                updated_notes = f"{current_notes}; Review: {notes}"
            else:
                updated_notes = f"Review: {notes}"
                
            df.loc[mask, 'outcome_notes'] = updated_notes
            
            # Save updated file
            df.to_csv(self.review_file, index=False)
            
            # Also update the JSON file
            self._add_review_to_json(decision_id, instrument, action_type, quality_score, notes)
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding review: {e}")
            return False
    
    def _add_review_to_json(self, decision_id, instrument, action_type, quality_score, notes):
        """Add review information to the JSON decision file."""
        # Find the decision file
        decision_file = os.path.join(self.decisions_dir, f"{decision_id}.json")
        if not os.path.exists(decision_file):
            logger.warning(f"Decision file not found for ID {decision_id}")
            return
        
        # Load the decision data
        with open(decision_file, 'r', encoding='utf-8') as f:
            decision_data = json.load(f)
        
        # Add review info
        if "reviews" not in decision_data:
            decision_data["reviews"] = []
            
        decision_data["reviews"].append({
            "instrument": instrument,
            "action_type": action_type,
            "quality_score": quality_score,
            "notes": notes,
            "review_timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Save updated data
        with open(decision_file, 'w', encoding='utf-8') as f:
            json.dump(decision_data, indent=2, default=str, fp=f)