import logging
import time
import decimal
from datetime import datetime, timezone

logger = logging.getLogger("TradingBot")

def process_llm_recommendations_with_logging(recommendations, portfolio, risk_manager, 
                                            data_provider, broker, executor, 
                                            decision_logger, decision_id):
    """Process LLM recommendations with execution logging."""
    try:
        open_positions_df = portfolio.get_open_positions_df()

        # --- Amendments ---
        trade_amendments = recommendations.get("tradeAmendments", [])
        if trade_amendments:
            logger.info(f"Processing {len(trade_amendments)} LLM amendments...")
            print(f"⚙️ Processing {len(trade_amendments)} position amendments...")
            
            for amend in trade_amendments:
                epic = amend.get("epic")
                action = amend.get("action")
                if not epic or not action:
                    logger.warning("Skipping invalid amendment structure (no epic/action)")
                    continue
                    
                # Check if the epic might actually be a dealId
                position_rows = open_positions_df[open_positions_df['epic'] == epic]
                
                # If no position found by epic, try looking up by dealId instead
                if position_rows.empty and epic.startswith('DI'):
                    position_rows = open_positions_df[open_positions_df['dealId'] == epic]
                    if not position_rows.empty:
                        # If found by dealId, log this for clarity
                        logger.info(f"Position found using dealId instead of epic: {epic}")
                
                if position_rows.empty:
                    logger.warning(f"Cannot {action} {epic}: No open position.")
                    continue

                position = position_rows.iloc[0].to_dict()
                deal_id = position.get('dealId')
                pos_size = position.get('size')
                pos_dir = position.get('direction')
                entry_level = position.get('level')
                
                # Get the actual instrument epic (not the dealId)
                real_epic = position.get('epic')
                if not real_epic:
                    logger.warning(f"Cannot determine real epic for position: {deal_id}")
                    real_epic = epic  # Fallback to original value, though likely incorrect

                if not all([deal_id, pos_size is not None, pos_dir]):
                    logger.warning(f"Missing critical data for open position {epic}")
                    continue

                # Process different amendment types
                execution_result = None
                
                if action == "CLOSE":
                    logger.info(f"LLM Recommends CLOSE for {epic} (DealID: {deal_id})")
                    print(f"🔄 Closing position {epic} (DealID: {deal_id})")
                    success = executor.close_trade(deal_id, pos_size, pos_dir, epic=real_epic)
                    
                    # Create execution result for logging
                    execution_result = {
                        "status": "SUCCESS" if success else "FAILED",
                        "action": "CLOSE",
                        "dealId": deal_id,
                        "reason": "LLM recommendation"
                    }
                    
                    # Update decision log
                    decision_logger.update_execution_result(
                        decision_id, 
                        real_epic, 
                        f"AMENDMENT_{action}",
                        execution_result
                    )
                    
                    if success:
                        print(f"✅ Successfully closed position {epic}")
                    else:
                        print(f"❌ Failed to close position {epic}")

                elif action == "AMEND":
                    new_stop_dist_dec = amend.get("new_stop_distance_dec")
                    new_limit_dist_dec = amend.get("new_limit_distance_dec")
                    if new_stop_dist_dec is None and new_limit_dist_dec is None:
                        logger.warning(f"AMEND action for {epic} has no new distances.")
                        continue

                    # Always use the real instrument epic for market data, not dealId
                    snapshot = broker.fetch_market_snapshot(real_epic)
                        
                    if not snapshot or snapshot.get('bid') is None:
                        logger.warning(f"Cannot calculate AMEND levels for {real_epic}: No snapshot.")
                        continue

                    current_price = snapshot['offer'] if pos_dir == 'BUY' else snapshot['bid']
                    new_stop_level, new_limit_level = None, None
                    reason = "Validation failed"
                    
                    try:
                        if new_stop_dist_dec is not None:
                            new_stop_level_dec = (current_price - new_stop_dist_dec) if pos_dir == 'BUY' else (current_price + new_stop_dist_dec)
                            current_stop_dec = position.get('stopLevel')
                            if current_stop_dec is not None:
                                # For a BUY, moving stop lower (away from current price) increases risk
                                # For a SELL, moving stop higher (away from current price) increases risk
                                is_loss_increasing = (
                                    (pos_dir == 'BUY' and new_stop_level_dec < current_stop_dec) or
                                    (pos_dir == 'SELL' and new_stop_level_dec > current_stop_dec)
                                )
                                
                                if is_loss_increasing and new_stop_level_dec != current_stop_dec:
                                    logger.warning(f"REJECTING AMEND Stop for {epic}: New stop {new_stop_level_dec} increases risk from {current_stop_dec}.")
                                    print(f"❌ Rejecting stop amendment for {epic}: New stop {new_stop_level_dec} increases risk from current {current_stop_dec}")
                                    reason = f"New stop {new_stop_level_dec} increases risk from {current_stop_dec}"
                                else:
                                    new_stop_level = float(new_stop_level_dec)
                            else:
                                new_stop_level = float(new_stop_level_dec)

                        if new_limit_dist_dec is not None:
                            new_limit_level_dec = (current_price + new_limit_dist_dec) if pos_dir == 'BUY' else (current_price - new_limit_dist_dec)
                            new_limit_level = float(new_limit_level_dec)

                        # Check if positions can be amended (gets min distances from broker)
                        can_amend, amend_reason = broker.check_amendment_viability(
                            real_epic, current_price, pos_dir, new_stop_level, new_limit_level
                        )
                        
                        if not can_amend:
                            logger.warning(f"Cannot AMEND {epic}: {amend_reason}")
                            print(f"❌ Cannot amend {epic}: {amend_reason}")
                            reason = amend_reason
                            success = False
                        else:
                            if new_stop_level or new_limit_level:
                                logger.info(f"LLM Recommends AMEND for {epic} (DealID: {deal_id}): Stop={new_stop_level}, Limit={new_limit_level}")
                                print(f"🔄 Amending position {epic} (DealID: {deal_id}): Stop={new_stop_level}, Limit={new_limit_level}")
                                success = executor.amend_trade(deal_id, new_stop_level, new_limit_level, epic=real_epic)
                                
                                if success:
                                    print(f"✅ Successfully amended position {epic}")
                                else:
                                    print(f"❌ Failed to amend position {epic}")
                            else:
                                logger.info(f"No valid levels to AMEND for {epic} after safety check.")
                                print(f"ℹ️ No valid levels to amend for {epic} after safety checks")
                                reason = "No valid levels after safety checks"
                                success = False
                        
                        # Create execution result for logging
                        execution_result = {
                            "status": "SUCCESS" if success else "FAILED",
                            "action": "AMEND",
                            "dealId": deal_id,
                            "new_stop": new_stop_level,
                            "new_limit": new_limit_level,
                            "reason": reason if not success else "LLM recommendation"
                        }
                        
                        # Update decision log
                        decision_logger.update_execution_result(
                            decision_id, 
                            real_epic, 
                            f"AMENDMENT_{action}",
                            execution_result
                        )
                        
                    except Exception as calc_err:
                        logger.error(f"Error calculating AMEND levels for {epic}: {calc_err}", exc_info=True)
                        print(f"❌ Error calculating amendment levels for {epic}: {calc_err}")
                        
                        # Log the error
                        execution_result = {
                            "status": "FAILED",
                            "action": "AMEND",
                            "dealId": deal_id,
                            "reason": f"Error: {str(calc_err)}"
                        }
                        
                        decision_logger.update_execution_result(
                            decision_id, 
                            real_epic, 
                            f"AMENDMENT_{action}",
                            execution_result
                        )

                elif action == "BREAKEVEN":
                    if entry_level is not None:
                        # Check if breakeven is viable (not too close to market)
                        snapshot = broker.fetch_market_snapshot(real_epic)
                        if not snapshot or snapshot.get('bid') is None:
                            logger.warning(f"Cannot check BREAKEVEN viability for {real_epic}: No snapshot.")
                            print(f"❌ Cannot check breakeven viability for {real_epic}: No market data")
                            continue
                            
                        current_price = snapshot['offer'] if pos_dir == 'BUY' else snapshot['bid']
                        can_amend, reason = broker.check_amendment_viability(
                            real_epic, current_price, pos_dir, float(entry_level), None
                        )
                        
                        if not can_amend:
                            logger.warning(f"Cannot BREAKEVEN {epic}: {reason}")
                            print(f"❌ Cannot set breakeven for {epic}: {reason}")
                            
                            # Log the failure
                            execution_result = {
                                "status": "FAILED",
                                "action": "BREAKEVEN",
                                "dealId": deal_id,
                                "breakeven_level": float(entry_level),
                                "reason": reason
                            }
                        else:
                            logger.info(f"LLM Recommends BREAKEVEN for {epic} (DealID: {deal_id})")
                            print(f"🔄 Setting breakeven stop for {epic} (DealID: {deal_id}) at {entry_level}")
                            success = executor.amend_trade(deal_id, stop_level=float(entry_level), limit_level=None, epic=real_epic)
                            
                            if success:
                                print(f"✅ Successfully set breakeven stop for {epic}")
                            else:
                                print(f"❌ Failed to set breakeven stop for {epic}")
                            
                            # Log the result
                            execution_result = {
                                "status": "SUCCESS" if success else "FAILED",
                                "action": "BREAKEVEN",
                                "dealId": deal_id,
                                "breakeven_level": float(entry_level),
                                "reason": "LLM recommendation" if success else "Execution failed"
                            }
                        
                        # Update decision log
                        decision_logger.update_execution_result(
                            decision_id, 
                            real_epic, 
                            f"AMENDMENT_{action}",
                            execution_result
                        )
                    else:
                        logger.warning(f"Cannot set breakeven for {epic}: Entry level missing.")
                        print(f"❌ Cannot set breakeven for {epic}: Entry level missing")
                else:
                    logger.warning(f"Unsupported amendment action '{action}' for {epic}.")
                    print(f"❓ Unsupported amendment action '{action}' for {epic}")
                
                time.sleep(0.5)
        else:
            logger.info("No LLM amendments to process.")
            print("ℹ️ No position amendments to process")

        # --- New Trades ---
        trade_actions = recommendations.get("tradeActions", [])
        if trade_actions:
            logger.info(f"Processing {len(trade_actions)} LLM new trade actions...")
            print(f"🆕 Processing {len(trade_actions)} new trade opportunities...")
            
            for action in trade_actions:
                epic = action.get("epic")
                direction = action.get("action")
                stop_dist_dec = action.get("stop_loss_pips")
                limit_dist_dec = action.get("limit_pips")
                confidence = action.get("confidence")

                if not epic or not direction or stop_dist_dec is None:
                    logger.warning(f"Skipping invalid action structure: {action}")
                    continue

                if not open_positions_df[open_positions_df['epic'] == epic].empty:
                    logger.info(f"Skipping new {direction} for {epic}: Position exists.")
                    print(f"ℹ️ Skipping new {direction} for {epic}: Position already exists")
                    continue

                instrument_details = broker.get_instrument_details(epic)
                snapshot = broker.fetch_market_snapshot(epic)
                if not instrument_details or not snapshot or snapshot.get('bid') is None:
                    logger.warning(f"Skipping {epic}: Missing details or snapshot.")
                    print(f"❌ Skipping {epic}: Missing instrument details or market data")
                    continue

                signal_price = snapshot['offer'] if direction == 'BUY' else snapshot['bid']
                proposed_trade = {
                    'symbol': epic,
                    'direction': direction,
                    'signal_price': signal_price,
                    'stop_loss_pips': stop_dist_dec,
                    'limit_pips': limit_dist_dec,
                    'confidence': confidence,
                }

                # Verify stop/limit distances meet broker requirements
                can_place, reason = broker.check_order_viability(
                    epic, signal_price, direction, 
                    float(stop_dist_dec), 
                    float(limit_dist_dec) if limit_dist_dec else None
                )
                
                if not can_place:
                    logger.warning(f"Skipping {epic} {direction}: {reason}")
                    print(f"❌ Skipping {epic} {direction}: {reason}")
                    
                    # Log the rejection
                    execution_result = {
                        "status": "REJECTED",
                        "reason": reason,
                        "direction": direction,
                        "stop_distance": float(stop_dist_dec),
                        "limit_distance": float(limit_dist_dec) if limit_dist_dec else None
                    }
                    
                    decision_logger.update_execution_result(
                        decision_id,
                        epic,
                        "NEW_TRADE",
                        execution_result
                    )
                    
                    continue

                final_trade_details, calc_reason = risk_manager.calculate_trade_details(proposed_trade, instrument_details, broker, data_provider)
                if not final_trade_details:
                    logger.warning(f"Trade calc failed for {epic}: {calc_reason}")
                    print(f"❌ Trade calculation failed for {epic}: {calc_reason}")
                    
                    # Log the rejection
                    execution_result = {
                        "status": "REJECTED",
                        "reason": calc_reason,
                        "direction": direction
                    }
                    
                    decision_logger.update_execution_result(
                        decision_id,
                        epic,
                        "NEW_TRADE",
                        execution_result
                    )
                    
                    continue

                # Note: Changed to call check_portfolio_constraints instead of check_portfolio_constraints
                is_viable, constraint_reason = risk_manager.check_portfolio_constraints(final_trade_details, instrument_details, portfolio, data_provider)
                if is_viable:
                    logger.info(f"Executing viable trade for {epic}...")
                    print(f"🔄 Executing {direction} trade for {epic} | Size: {final_trade_details['size']:.2f} | Stop: {final_trade_details['stop_level']}")
                    
                    success, result = executor.execute_new_trade(final_trade_details)
                    if success:
                        logger.info(f"Trade submitted successfully for {epic}. Deal ID: {result}")
                        print(f"✅ Trade submitted successfully for {epic}. Deal ID: {result}")
                        
                        portfolio.update_state()
                        risk_manager.update_account(portfolio)
                        
                        # Log successful execution
                        execution_result = {
                            "status": "SUCCESS",
                            "dealId": result,
                            "direction": direction,
                            "size": float(final_trade_details['size']),
                            "stop_level": float(final_trade_details['stop_level']),
                            "limit_level": float(final_trade_details['limit_level']) if final_trade_details.get('limit_level') else None,
                            "reason": "LLM recommendation"
                        }
                    else:
                        logger.error(f"Execution failed for {epic}: {result}")
                        print(f"❌ Execution failed for {epic}: {result}")
                        
                        # Log failed execution
                        execution_result = {
                            "status": "FAILED",
                            "reason": result,
                            "direction": direction
                        }
                    
                    # Update decision log
                    decision_logger.update_execution_result(
                        decision_id,
                        epic,
                        "NEW_TRADE",
                        execution_result,
                        final_trade_details
                    )
                    
                    time.sleep(1.5)
                else:
                    logger.warning(f"Trade for {epic} rejected by constraints: {constraint_reason}")
                    print(f"❌ Trade for {epic} rejected by risk constraints: {constraint_reason}")
                    
                    # Log the rejection
                    execution_result = {
                        "status": "REJECTED",
                        "reason": constraint_reason,
                        "direction": direction
                    }
                    
                    decision_logger.update_execution_result(
                        decision_id,
                        epic,
                        "NEW_TRADE",
                        execution_result
                    )
        else:
            logger.info("No new LLM trade actions to process.")
            print("ℹ️ No new trade opportunities to process")
                
    except Exception as e:
        logger.error(f"Error processing LLM recommendations: {e}", exc_info=True)
        print(f"❌ Error processing LLM recommendations: {e}")
        
        # Attempt to log the error in the decision log
        try:
            decision_logger.update_execution_result(
                decision_id,
                "global",
                "ERROR",
                {
                    "status": "FAILED",
                    "reason": f"Processing error: {str(e)}"
                }
            )
        except Exception as log_err:
            logger.error(f"Could not log error to decision logger: {log_err}")
            
    return