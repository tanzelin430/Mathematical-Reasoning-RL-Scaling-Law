#!/usr/bin/env python3
"""
Batch Execution Example for Scaling Law Analysis

This script demonstrates different ways to run the scaling law analysis:
1. Single command execution (equivalent to your CLI command)
2. Batch execution from JSON config
3. Programmatic execution with custom parameters
"""

import subprocess
import sys
from pathlib import Path


def run_single_command():
    """
    Execute the equivalent of your original command:
    python run_plot_multi_fit.py --data-source base --plot-curve N -x C,E --eval holdout_score 
    --metric ErrRate --warmup-clip-frac 0.02 --fit --fit-curve N --fit-x C --fit-metric ErrRate 
    --plot-x-scale log --plot-y-scale log
    """
    print("=" * 60)
    print("Running Single Command (equivalent to your CLI example)")
    print("=" * 60)
    
    cmd = [
        "uv", "run", "python", "run_plot_multi_fit.py",
        "--data-source", "exp2-base",  # Note: using exp2-base instead of 'base'
        "--plot-curve", "N",
        "-x", "C,E",
        "--eval", "holdout_score",
        "--metric", "ErrRate",
        "--warmup-clip-frac", "0.02",
        "--fit",
        "--fit-curve", "N",
        "--fit-x", "C",
        "--fit-metric", "ErrRate",
        "--plot-x-scale", "log",
        "--plot-y-scale", "log",
        "--plot-curve-mask", "5e8,1.5e9,3e9,7e9,14e9",
        "--highlight-curves", "7e9",
        "--output-prefix", "single_cmd_example_"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print("✅ Single command executed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error executing single command: {e}")
        return False


def run_batch_config():
    """
    Execute batch processing from JSON config file
    """
    print("\n" + "=" * 60)
    print("Running Batch Configuration")
    print("=" * 60)
    
    config_file = "example_batch_config.json"
    if not Path(config_file).exists():
        print(f"❌ Config file {config_file} not found!")
        return False
    
    cmd = [
        "uv", "run", "python", "run_plot_multi_fit.py",
        "--config-file", config_file
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print("✅ Batch configuration executed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error executing batch config: {e}")
        return False


def run_programmatic_example():
    """
    Example of how to programmatically create and run configurations
    """
    print("\n" + "=" * 60)
    print("Running Programmatic Example")
    print("=" * 60)
    
    # Example: Create a quick comparison between different data sources
    configs = [
        {
            "name": "Base Model Quick Analysis",
            "args": [
                "--data-source", "exp2-base",
                "--plot-curve", "Tau",
                "-x", "E",
                "--eval", "holdout_score",
                "--metric", "ErrRate",
                "--plot-curve-mask", "1,5,20,100",
                "--output-prefix", "quick_base_"
            ]
        },
        {
            "name": "Instruct Model Quick Analysis",
            "args": [
                "--data-source", "exp2-instruct",
                "--plot-curve", "Tau",
                "-x", "E",
                "--eval", "holdout_score",
                "--metric", "ErrRate",
                "--plot-curve-mask", "1,5,20,100",
                "--output-prefix", "quick_instruct_"
            ]
        }
    ]
    
    for config in configs:
        print(f"\n--- {config['name']} ---")
        cmd = ["uv", "run", "python", "run_plot_multi_fit.py"] + config['args']
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=False, text=True)
            print(f"✅ {config['name']} completed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error in {config['name']}: {e}")
            continue
    
    return True


def main():
    """
    Main execution function - you can comment/uncomment different sections
    """
    print("🚀 Scaling Law Analysis - Batch Execution Examples")
    print("Choose execution mode:")
    print("1. Single command (your original CLI example)")
    print("2. Batch configuration from JSON")
    print("3. Programmatic execution")
    print("4. All of the above")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    success_count = 0
    total_count = 0
    
    if choice in ['1', '4']:
        total_count += 1
        if run_single_command():
            success_count += 1
    
    if choice in ['2', '4']:
        total_count += 1
        if run_batch_config():
            success_count += 1
    
    if choice in ['3', '4']:
        total_count += 1
        if run_programmatic_example():
            success_count += 1
    
    print(f"\n🎯 Summary: {success_count}/{total_count} execution modes completed successfully")
    
    if success_count == total_count:
        print("🎉 All executions completed successfully!")
        print("📁 Check the 'outputs/' directory for generated plots and models")
    else:
        print("⚠️  Some executions failed. Check error messages above.")


if __name__ == "__main__":
    main()
