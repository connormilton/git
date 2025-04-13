#!/usr/bin/env python
# Script to apply fixes to the Forex AI Trader

import os
import sys
import shutil
from datetime import datetime

def backup_file(filepath):
    """Create a backup of a file before modifying it."""
    if os.path.exists(filepath):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{filepath}.backup_{timestamp}"
        shutil.copy2(filepath, backup_path)
        print(f"Created backup of {filepath} at {backup_path}")
        return True
    return False

def apply_data_provider_fix():
    """Apply the fix to data_provider.py"""
    filepath = "data_provider.py"
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found in current directory!")
        return False
    
    # Create backup
    if backup_file(filepath):
        try:
            # Open the fixed version file
            with open("fixed_data_provider.py", "r") as fixed_file:
                fixed_content = fixed_file.read()
            
            # Write to the original file
            with open(filepath, "w") as original_file:
                original_file.write(fixed_content)
                
            print(f"✅ Successfully applied fix to {filepath}")
            return True
        except Exception as e:
            print(f"Error applying fix to {filepath}: {e}")
            return False
    return False

def apply_risk_manager_fix():
    """Apply fixes to advanced_risk_manager.py"""
    filepath = "advanced_risk_manager.py"
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found in current directory!")
        return False
    
    # Create backup
    if backup_file(filepath):
        try:
            # Read original file
            with open(filepath, "r") as f:
                content = f.read()
            
            # Read replacement function for check_portfolio_constraints
            with open("fixed_check_portfolio_constraints.py", "r") as f:
                constraints_func = f.read()
                
            # Read replacement function for calculate_trade_details
            with open("fixed_calculate_trade_details.py", "r") as f:
                trade_details_func = f.read()
            
            # Replace the functions
            # This is a simplistic approach - in a real scenario, we'd use AST parsing
            # to properly modify the functions, but this should work for our purposes
            
            # Find the check_portfolio_constraints function
            check_constraints_start = content.find("def check_portfolio_constraints")
            if check_constraints_start == -1:
                print("Error: Could not find check_portfolio_constraints function!")
                return False
            
            # Find next function definition after check_portfolio_constraints
            next_func_start = content.find("def ", check_constraints_start + 30)
            if next_func_start == -1:
                next_func_start = len(content)  # If not found, use end of file
            
            # Replace the function
            new_content = content[:check_constraints_start] + constraints_func + content[next_func_start:]
            
            # Find the calculate_trade_details function
            calc_details_start = new_content.find("def calculate_trade_details")
            if calc_details_start == -1:
                print("Error: Could not find calculate_trade_details function!")
                return False
            
            # Find next function definition after calculate_trade_details
            next_func_start = new_content.find("def ", calc_details_start + 30)
            if next_func_start == -1:
                next_func_start = len(new_content)  # If not found, use end of file
            
            # Replace the function
            final_content = new_content[:calc_details_start] + trade_details_func + new_content[next_func_start:]
            
            # Write the modified content back to the file
            with open(filepath, "w") as f:
                f.write(final_content)
                
            print(f"✅ Successfully applied fixes to {filepath}")
            return True
        except Exception as e:
            print(f"Error applying fixes to {filepath}: {e}")
            return False
    return False

def apply_llm_recommendations_fix():
    """Apply fix to process_llm_recommendations_with_logging.py"""
    filepath = "process_llm_recommendations_with_logging.py"
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found in current directory!")
        return False
    
    # Create backup
    if backup_file(filepath):
        try:
            # Open the fixed version file
            with open("fixed_process_llm_recommendations_with_logging.py", "r") as fixed_file:
                fixed_content = fixed_file.read()
            
            # Write to the original file
            with open(filepath, "w") as original_file:
                original_file.write(fixed_content)
                
            print(f"✅ Successfully applied fix to {filepath}")
            return True
        except Exception as e:
            print(f"Error applying fix to {filepath}: {e}")
            return False
    return False

def extract_functions_from_artifacts():
    """Extract functions from the artifacts files we created."""
    # Extract data_provider fix
    with open("fixed-data-provider.txt", "r") as f:
        data_provider_content = f.read()
        with open("fixed_data_provider.py", "w") as out:
            out.write(data_provider_content)
    
    # Extract risk manager functions
    risk_manager_content = ""
    with open("fixed-risk-manager.txt", "r") as f:
        risk_manager_content = f.read()
    
    # Extract each function 
    check_constraints_start = risk_manager_content.find("def check_portfolio_constraints")
    calc_trade_start = risk_manager_content.find("def calculate_trade_details")
    
    if check_constraints_start != -1 and calc_trade_start != -1:
        # Determine which comes first to properly split
        if check_constraints_start < calc_trade_start:
            check_constraints_content = risk_manager_content[check_constraints_start:calc_trade_start]
            calc_trade_content = risk_manager_content[calc_trade_start:]
        else:
            calc_trade_content = risk_manager_content[calc_trade_start:check_constraints_start]
            check_constraints_content = risk_manager_content[check_constraints_start:]
    
        # Write to individual files
        with open("fixed_check_portfolio_constraints.py", "w") as f:
            f.write(check_constraints_content)
        
        with open("fixed_calculate_trade_details.py", "w") as f:
            f.write(calc_trade_content)
    
    # Extract process_llm_recommendations_with_logging fix
    with open("process-llm-recommendations-fix.txt", "r") as f:
        process_llm_content = f.read()
        with open("fixed_process_llm_recommendations_with_logging.py", "w") as out:
            out.write(process_llm_content)

def main():
    print("📊 Forex AI Trader Bug Fix Utility")
    print("----------------------------------")
    
    # Extract functions from artifacts
    print("Extracting fixes from artifacts...")
    extract_functions_from_artifacts()
    
    # Apply fixes
    data_provider_fixed = apply_data_provider_fix()
    risk_manager_fixed = apply_risk_manager_fix()
    llm_recommendations_fixed = apply_llm_recommendations_fix()
    
    # Summary
    print("\n🔍 Fix Application Summary:")
    print(f"  - data_provider.py: {'✅ Fixed' if data_provider_fixed else '❌ Failed'}")
    print(f"  - advanced_risk_manager.py: {'✅ Fixed' if risk_manager_fixed else '❌ Failed'}")
    print(f"  - process_llm_recommendations_with_logging.py: {'✅ Fixed' if llm_recommendations_fixed else '❌ Failed'}")
    
    if data_provider_fixed and risk_manager_fixed and llm_recommendations_fixed:
        print("\n🎉 All fixes successfully applied!")
        print("\nTo test your fixes, run: python main.py")
    else:
        print("\n⚠️ Some fixes could not be applied. Check the log above for details.")
    
    # Cleanup
    print("\nCleaning up temporary files...")
    try:
        cleanup_files = [
            "fixed_data_provider.py", 
            "fixed_check_portfolio_constraints.py", 
            "fixed_calculate_trade_details.py",
            "fixed_process_llm_recommendations_with_logging.py"
        ]
        for file in cleanup_files:
            if os.path.exists(file):
                os.remove(file)
        print("Cleanup complete.")
    except Exception as e:
        print(f"Error during cleanup: {e}")

if __name__ == "__main__":
    main()