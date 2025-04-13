# llm_process.py

import logging
logger = logging.getLogger("TradingBot")

def process_llm_recommendations_with_logging(recommendations, portfolio, risk_manager, 
                                           data_provider, broker, executor, 
                                           decision_logger, decision_id):
    """Wrapper function to import the actual implementation."""
    try:
        # Import the actual implementation
        from process_llm_recommendations_with_logging import process_llm_recommendations_with_logging as actual_function
        
        # Call the actual implementation
        return actual_function(
            recommendations, portfolio, risk_manager, 
            data_provider, broker, executor, 
            decision_logger, decision_id
        )
    except ImportError as e:
        logger.error(f"Failed to import process_llm_recommendations_with_logging: {e}")
        print(f"⚠️ ERROR: Failed to import process_llm_recommendations_with_logging: {e}")
        
        # Fallback to the original function
        from main import process_llm_recommendations
        process_llm_recommendations(
            recommendations, portfolio, risk_manager, 
            data_provider, broker, executor
        )
        return True