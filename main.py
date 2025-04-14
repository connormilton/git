# main.py

import os
import time
import logging
import decimal
from datetime import datetime, timezone

# Local imports
from config_loader import load_and_configure, get_config
from logging_setup import setup_logging
from ig_interface import IGInterface
from data_provider import DataProvider
from trade_memory import TradeMemory
from advanced_risk_manager import AdvancedRiskManager
from pair_selector import PairSelector
from llm_interface import LLMInterface
from advanced_llm_prompting import AdvancedLLMPrompting
from portfolio import Portfolio
from trade_executor import ExecutionHandler
from decision_logger import DecisionLogger
from trading_ig.rest import ApiExceededException

# Import error handling for process_llm_recommendations_with_logging
try:
    from process_llm_recommendations_with_logging import process_llm_recommendations_with_logging
except ImportError:
    # Fallback to simplified version if import fails
    from llm_process import process_llm_recommendations_with_logging
    logger = logging.getLogger("TradingBot")
    logger.warning("Using simplified LLM recommendation processing due to import failure")

logger = logging.getLogger("TradingBot")

def process_llm_recommendations(recommendations, portfolio, risk_manager, data_provider, broker, executor):
    open_positions_df = portfolio.get_open_positions_df()

    # --- Amendments ---
    trade_amendments = recommendations.get("tradeAmendments", [])
    if trade_amendments:
        logger.info(f"Processing {len(trade_amendments)} LLM amendments...")
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

            if action == "CLOSE":
                logger.info(f"LLM Recommends CLOSE for {epic} (DealID: {deal_id})")
                executor.close_trade(deal_id, pos_size, pos_dir, epic=real_epic)

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
                try:
                    if new_stop_dist_dec is not None:
                        new_stop_level_dec = (current_price - new_stop_dist_dec) if pos_dir == 'BUY' else (current_price + new_stop_dist_dec)
                        current_stop_dec = position.get('stopLevel')
                        if current_stop_dec is not None:
                            # For a BUY, moving stop lower (towards current price) decreases risk
                            # For a SELL, moving stop higher (towards current price) decreases risk
                            is_loss_increasing = (
                                (pos_dir == 'BUY' and new_stop_level_dec > current_stop_dec) or
                                (pos_dir == 'SELL' and new_stop_level_dec < current_stop_dec)
                            )
                            if is_loss_increasing:
                                logger.warning(f"REJECTING AMEND Stop for {epic}: New stop {new_stop_level_dec} increases risk from {current_stop_dec}.")
                            else:
                                new_stop_level = float(new_stop_level_dec)
                        else:
                            new_stop_level = float(new_stop_level_dec)

                    if new_limit_dist_dec is not None:
                        new_limit_level_dec = (current_price + new_limit_dist_dec) if pos_dir == 'BUY' else (current_price - new_limit_dist_dec)
                        new_limit_level = float(new_limit_level_dec)

                    # Check if positions can be amended (gets min distances from broker)
                    can_amend, reason = broker.check_amendment_viability(
                        real_epic, current_price, pos_dir, new_stop_level, new_limit_level
                    )
                    
                    if not can_amend:
                        logger.warning(f"Cannot AMEND {epic}: {reason}")
                        continue

                    if new_stop_level or new_limit_level:
                        logger.info(f"LLM Recommends AMEND for {epic} (DealID: {deal_id}): Stop={new_stop_level}, Limit={new_limit_level}")
                        executor.amend_trade(deal_id, new_stop_level, new_limit_level, epic=real_epic)
                    else:
                        logger.info(f"No valid levels to AMEND for {epic} after safety check.")
                except Exception as calc_err:
                    logger.error(f"Error calculating AMEND levels for {epic}: {calc_err}", exc_info=True)

            elif action == "BREAKEVEN":
                if entry_level is not None:
                    # Check if breakeven is viable (not too close to market)
                    snapshot = broker.fetch_market_snapshot(real_epic)
                    if not snapshot or snapshot.get('bid') is None:
                        logger.warning(f"Cannot check BREAKEVEN viability for {real_epic}: No snapshot.")
                        continue
                        
                    current_price = snapshot['offer'] if pos_dir == 'BUY' else snapshot['bid']
                    can_amend, reason = broker.check_amendment_viability(
                        real_epic, current_price, pos_dir, float(entry_level), None
                    )
                    
                    if not can_amend:
                        logger.warning(f"Cannot BREAKEVEN {epic}: {reason}")
                        continue
                        
                    logger.info(f"LLM Recommends BREAKEVEN for {epic} (DealID: {deal_id})")
                    executor.amend_trade(deal_id, stop_level=float(entry_level), limit_level=None, epic=real_epic)
                else:
                    logger.warning(f"Cannot set breakeven for {epic}: Entry level missing.")
            else:
                logger.warning(f"Unsupported amendment action '{action}' for {epic}.")
            time.sleep(0.5)
    else:
        logger.info("No LLM amendments to process.")

    # --- New Trades ---
    trade_actions = recommendations.get("tradeActions", [])
    if trade_actions:
        logger.info(f"Processing {len(trade_actions)} LLM new trade actions...")
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
                continue

            instrument_details = broker.get_instrument_details(epic)
            snapshot = broker.fetch_market_snapshot(epic)
            if not instrument_details or not snapshot or snapshot.get('bid') is None:
                logger.warning(f"Skipping {epic}: Missing details or snapshot.")
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
                continue

            final_trade_details, calc_reason = risk_manager.calculate_trade_details(proposed_trade, instrument_details, broker, data_provider)
            if not final_trade_details:
                logger.warning(f"Trade calc failed for {epic}: {calc_reason}")
                continue

            is_viable, constraint_reason = risk_manager.check_portfolio_constraints(final_trade_details, instrument_details, portfolio, data_provider)
            if is_viable:
                logger.info(f"Executing viable trade for {epic}...")
                success, result = executor.execute_new_trade(final_trade_details)
                if success:
                    logger.info(f"Trade submitted successfully for {epic}. Deal ID: {result}")
                    portfolio.update_state()
                    risk_manager.update_account(portfolio)
                    time.sleep(1.5)
                else:
                    logger.error(f"Execution failed for {epic}: {result}")
            else:
                logger.warning(f"Trade for {epic} rejected by constraints: {constraint_reason}")
    else:
        logger.info("No new LLM trade actions to process.")

def run_trading_cycle(config, broker, data_provider, trade_memory, risk_manager, portfolio, pair_selector, llm_prompting, llm_interface, decision_logger):
    """Standard trading cycle without margin validation."""
    now = datetime.now(timezone.utc)
    logger.info(f"--- Trading Cycle {now.isoformat(timespec='seconds')} ---")
    
    # Update portfolio and risk management
    portfolio.update_state()
    risk_manager.update_account(portfolio)
    
    # Print account status to terminal
    print(f"💰 Account Balance: {portfolio.get_balance():.2f} {config['ACCOUNT_CURRENCY']}")
    print(f"💸 Available Margin: {portfolio.get_available_funds():.2f} {config['ACCOUNT_CURRENCY']}")
    print(f"⚠️ Current Risk Exposure: {risk_manager.total_open_risk_percent:.2f}% of balance\n")
    
    # 1. Select potential pairs to analyze
    print("🔍 Selecting optimal trading pairs...")
    candidate_pairs = pair_selector.select_best_pairs(max_pairs=config.get('MAX_PAIRS_TO_ANALYZE', 10))
    if not candidate_pairs:
        logger.warning("No valid pairs selected. Will skip this cycle.")
        print("⚠️ WARNING: No valid pairs selected for this cycle. Will try again next cycle.")
        return False
        
    logger.info(f"Selected {len(candidate_pairs)} candidate pairs")
    print(f"✅ Selected {len(candidate_pairs)} pairs for analysis: {', '.join(candidate_pairs)}\n")
    
    # 2. Fetch market data for candidate pairs
    print("📊 Fetching market data and technical indicators...")
    market_snapshots = {}
    technical_data = {}
    
    for epic in candidate_pairs:
        snapshot = broker.fetch_market_snapshot(epic)
        if snapshot:
            market_snapshots[epic] = snapshot
            
        # Get technical indicators
        try:
            technicals = data_provider.get_latest_technicals(epic)
            if technicals:
                technical_data[epic] = technicals
        except Exception as e:
            logger.warning(f"Error getting technicals for {epic}: {e}")
        
        time.sleep(0.1)  # Avoid API rate limits
    
    valid_snapshots = {
        e: s for e, s in market_snapshots.items() if s and s.get('bid') is not None
    }
    
    if not valid_snapshots:
        logger.warning("No valid snapshots. Skipping LLM call.")
        print("⚠️ WARNING: No valid market data available. Skipping trading decisions.")
        return False
    
    print(f"✅ Fetched market data for {len(valid_snapshots)} instruments\n")
    
    # 3. Update volatility adjustments for risk management
    risk_manager.update_volatility_adjustments(data_provider)
    
    # 4. Generate LLM prompt
    print("🧠 Generating advanced LLM analysis prompt...")
    advanced_prompt, market_regime = llm_prompting.generate_advanced_prompt(
        portfolio, valid_snapshots, data_provider, risk_manager
    )
    
    # 5. Generate system prompt 
    system_prompt = llm_prompting.generate_system_prompt(market_regime)
    
    # 6. Get trade recommendations
    recommendations = llm_interface.get_trade_recommendations_with_prompt(
        system_prompt, advanced_prompt, market_regime
    )
    
    # Store the raw LLM response for logging
    raw_llm_response = recommendations.get("raw_response", "{}")
    
    # Log the full decision context
    current_decision_id = decision_logger.log_decisions(
        recommendations, 
        raw_llm_response, 
        advanced_prompt, 
        market_regime
    )
    
    # Return the recommendations and decision ID
    return recommendations, current_decision_id

def run_trading_bot():
    logger.info(f"🚀 Initializing Enhanced Forex AI Trader PID: {os.getpid()}")
    print(f"\n{'='*80}\n🚀 INITIALIZING ENHANCED FOREX AI TRADER\n{'='*80}")
    config = get_config()

    try:
        # Initialize components
        broker = IGInterface(config)
        data_provider = DataProvider(config)
        trade_memory = TradeMemory(config)
        risk_manager = AdvancedRiskManager(config, trade_memory)
        pair_selector = PairSelector(config, data_provider, trade_memory)
        portfolio = Portfolio(broker, config)
        executor = ExecutionHandler(broker, portfolio, config, trade_memory)
        llm_prompting = AdvancedLLMPrompting(config, trade_memory)
        llm_interface = LLMInterface(config)
        decision_logger = DecisionLogger(config)

        # Initial updates
        portfolio.update_state()
        risk_manager.update_account(portfolio)
        
        print(f"✅ All components initialized successfully")
        print(f"💰 Initial account balance: {portfolio.get_balance():.2f} {config['ACCOUNT_CURRENCY']}")
    except Exception as init_err:
        logger.critical(f"Initialization failed: {init_err}", exc_info=True)
        print(f"❌ ERROR: Initialization failed: {init_err}")
        return

    # Store current decision ID for this cycle
    current_decision_id = None

    while True:
        cycle_start_time = time.time()
        now = datetime.now(timezone.utc)
        
        # Print clear cycle separator to terminal
        print("\n\n" + "="*100)
        print(f"🔄 TRADING CYCLE: {now.isoformat(timespec='seconds')}")
        print("="*100 + "\n")
        
        try:
            # 1. Update portfolio state
            portfolio.update_state()
            if portfolio.get_balance() <= 0:
                logger.error("Balance zero or negative. Stopping.")
                print("❌ ERROR: Account balance zero or negative. Stopping trading bot.")
                break

            # 2. Run the trading cycle without margin validation
            results = run_trading_cycle(
                config, broker, data_provider, trade_memory, risk_manager, 
                portfolio, pair_selector, llm_prompting, llm_interface, decision_logger
            )
            
            if results is False:
                # No pairs were selected
                cycle_duration = time.time() - cycle_start_time
                sleep_time = max(10, config.get('TRADING_CYCLE_SECONDS', 180) - cycle_duration)
                logger.info(f"Cycle ended (No pairs selected). Sleeping {sleep_time:.2f}s...")
                print(f"\n⏱️ Cycle completed in {cycle_duration:.2f}s. Sleeping for {sleep_time:.2f}s until next cycle...")
                print("="*100)
                time.sleep(sleep_time)
                continue
            
            # Unpack results
            recommendations, current_decision_id = results
            
            # 3. Process LLM recommendations with decision tracking
            if current_decision_id:
                process_llm_recommendations_with_logging(
                    recommendations,
                    portfolio,
                    risk_manager,
                    data_provider,
                    broker,
                    executor,
                    decision_logger,
                    current_decision_id
                )
            else:
                process_llm_recommendations(
                    recommendations,
                    portfolio,
                    risk_manager,
                    data_provider,
                    broker,
                    executor
                )
            
            # 4. Store market data for future analysis
            valid_snapshots = {
                e: s for e, s in broker.fetch_market_snapshots(candidate_pairs).items() 
                if s and s.get('bid') is not None
            }
            
            for epic, data in valid_snapshots.items():
                technicals = data_provider.get_latest_technicals(epic)
                regime = data_provider.get_market_regime(epic)
                trade_memory.store_market_data(epic, data, technicals, regime)
            
            # 5. Update performance metrics periodically
            if now.hour % 6 == 0 and now.minute < 5:
                print("📈 Updating performance metrics...")
                # Update metrics every 6 hours
                trade_memory.update_performance_metrics()
                # Analyze trade patterns
                trade_memory.analyze_trade_patterns()
                print("✅ Performance metrics updated")

        except KeyboardInterrupt:
            logger.info("Stop requested by user.")
            print("\n⛔ Stop requested by user. Shutting down trading bot...")
            break
        except ApiExceededException:
            logger.error("IG API Rate Limit hit. Pausing 60s.")
            print("⚠️ IG API Rate Limit hit. Pausing for 60 seconds...")
            time.sleep(60)
        except Exception as loop_err:
            logger.critical("Unhandled exception in main loop!", exc_info=True)
            logger.error(f"Error: {loop_err}")
            logger.error("Pausing 30s before retry.")
            print(f"❌ ERROR: Unhandled exception: {loop_err}")
            print("⏳ Pausing for 30 seconds before retry...")
            time.sleep(30)

        # Calculate cycle duration and sleep time
        cycle_duration = time.time() - cycle_start_time
        sleep_time = max(10, config.get('TRADING_CYCLE_SECONDS', 180) - cycle_duration)
        logger.info(f"Cycle ended (Duration: {cycle_duration:.2f}s). Sleeping {sleep_time:.2f}s...")
        
        print(f"\n⏱️ Cycle completed in {cycle_duration:.2f}s. Sleeping for {sleep_time:.2f}s until next cycle...")
        print("="*100)
        
        time.sleep(sleep_time)
    
    # Clean up
    trade_memory.close()
    logger.info("Trading bot shutdown complete.")
    print("\n👋 Trading bot shutdown complete.")

def main():
    try:
        load_and_configure()  # This populates the global CONFIG in config_loader
        config = get_config()
        logger_ = setup_logging(config)
        run_trading_bot()
    except Exception as e:
        logger.critical(f"Fatal error starting up: {e}", exc_info=True)
        print(f"❌ FATAL ERROR: {e}")

if __name__ == "__main__":
    main()