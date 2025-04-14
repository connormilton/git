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
    """Load configuration from config_loader or environment variables."""
    try:
        # Try to import from project's config_loader
        from config_loader import get_config
        return get_config()
    except ImportError:
        # Fallback to hardcoded config with your credentials
        logger.warning("Could not import config_loader. Using hardcoded configuration.")
        config = {
            'IG_USERNAME': 'connormilton',
            'IG_PASSWORD': 'noisyFalseCar88',
            'IG_API_KEY': '95521b9a41b7fd311aef327e4ecddec775073be5',
            'IG_ACC_TYPE': 'LIVE',
            'IG_ACCOUNT_ID': 'INRKZ',
            'ACCOUNT_CURRENCY': 'GBP',
            'RISK_PER_TRADE_PERCENT': decimal.Decimal('2.0'),
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
            print("\nPlease set the following environment variables:")
            for key in missing_keys:
                print(f"  export {key}='your-value'")
            sys.exit(1)
            
        return config

def save_analysis_to_file(analysis, filename):
    """Save the analysis dataframe to a file."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save as CSV
        analysis.to_csv(filename)
        logger.info(f"Analysis saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving analysis to {filename}: {e}")

def save_strategies_to_file(strategies, filename):
    """Save the strategies to a JSON file."""
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
    """Format a value as currency."""
    return f"{currency} {value:.2f}"

def print_analysis_summary(analysis, currency):
    """Print a summary of the analysis."""
    # Count viable instruments
    viable_count = len(analysis[analysis['is_viable'] == True])
    total_count = len(analysis)
    
    print("\n" + "="*80)
    print(f"MARGIN ANALYSIS SUMMARY ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    print("="*80)
    
    print(f"\nViable Instruments: {viable_count}/{total_count} ({viable_count/total_count*100:.1f}%)")
    
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
        
        print(f"\nMargin Requirements:")
        print(f"  Average: {format_currency(avg_margin, currency)}")
        print(f"  Minimum: {format_currency(min_margin, currency)}")
        print(f"  Maximum: {format_currency(max_margin, currency)}")
    
    # Print viable instruments
    if viable_count > 0:
        viable = analysis[analysis['is_viable'] == True]
        print("\nViable Instruments:")
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
        print("\nNon-Viable Instruments:")
        for _, row in non_viable.iterrows():
            print(f"  {row['epic']} - {row['description']}")
    
    print("\n" + "="*80)

def print_strategies(strategies, currency):
    """Print alternative strategies."""
    print("\n" + "="*80)
    print("ALTERNATIVE STRATEGIES")
    print("="*80)
    
    print(f"\nCurrent Balance: {format_currency(strategies['current_balance'], currency)}")
    print(f"Minimum Balance Required: {format_currency(strategies['min_balance_required'], currency)}")
    
    if strategies['additional_funding_needed'] > 0:
        print(f"Additional Funding Needed: {format_currency(strategies['additional_funding_needed'], currency)}")
    
    print(f"\nViable Instruments Count: {strategies['viable_instruments_count']}")
    
    print("\nSuggested Alternatives:")
    for i, alt in enumerate(strategies['alternatives'], 1):
        print(f"{i}. {alt['description']}")
    
    print("\n" + "="*80)

def main():
    logger.info("Starting Margin Analysis Demo")
    
    # Load configuration
    config = load_config()
    
    # Force credentials regardless of what load_config returned
    config['IG_USERNAME'] = 'connormilton'
    config['IG_PASSWORD'] = 'noisyFalseCar88'
    config['IG_API_KEY'] = '95521b9a41b7fd311aef327e4ecddec775073be5'
    config['IG_ACC_TYPE'] = 'LIVE'
    config['IG_ACCOUNT_ID'] = 'INRKZ'
    config['ACCOUNT_CURRENCY'] = config.get('ACCOUNT_CURRENCY', 'GBP')
    config['RISK_PER_TRADE_PERCENT'] = config.get('RISK_PER_TRADE_PERCENT', decimal.Decimal('2.0'))
    
    print("\n" + "="*80)
    print("MARGIN ANALYSIS DEMO")
    print("="*80)
    print(f"\nConnecting to IG as {config['IG_USERNAME']} ({config['IG_ACC_TYPE']})")
    
    # Rest of your code...
    
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
        
        print("\nAnalyzing margin requirements for all instruments...")
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
        print("\nGenerating alternative strategies...")
        strategies = margin_analyzer.suggest_alternative_strategies(portfolio.get_balance())
        
        # Save strategies to file
        save_strategies_to_file(strategies, f"data/margin_strategies_{timestamp}.json")
        
        # Print strategies
        print_strategies(strategies, config['ACCOUNT_CURRENCY'])
        
    except Exception as e:
        logger.error(f"Error during margin analysis: {e}", exc_info=True)
        print(f"\nError during analysis: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())