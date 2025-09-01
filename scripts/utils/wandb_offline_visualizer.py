#!/usr/bin/env python3
"""
WandB Offline Data Visualizer

This script reads WandB offline run data from multiple directories and creates visualization plots.
It's especially useful when WandB is in offline mode and you need to combine data from multiple
resumed runs.

Usage:
    python wandb_offline_visualizer.py --wandb-dirs dir1 dir2 dir3 --output-dir ./plots
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import glob
import re

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def find_wandb_runs(wandb_dir: str) -> List[str]:
    """Find all run directories in a WandB directory."""
    wandb_path = Path(wandb_dir)
    run_dirs = []
    
    # Look for directories with pattern offline-run-* or run-*
    for pattern in ['offline-run-*', 'run-*']:
        run_dirs.extend(glob.glob(str(wandb_path / pattern)))
    
    # Also look for directories with timestamp patterns
    for item in wandb_path.iterdir():
        if item.is_dir() and re.match(r'\d{8}_\d{6}', item.name):
            run_dirs.append(str(item))
    
    return sorted(run_dirs)


def read_wandb_history(run_dir: str) -> pd.DataFrame:
    """Read wandb history from a run directory."""
    history_path = Path(run_dir) / 'files' / 'wandb-history.jsonl'
    
    if not history_path.exists():
        # Try alternative path
        history_path = Path(run_dir) / 'wandb-history.jsonl'
    
    if not history_path.exists():
        print(f"Warning: No history file found in {run_dir}")
        return pd.DataFrame()
    
    # Read jsonl file
    data = []
    with open(history_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    
    if not data:
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    
    # Extract run info
    run_config_path = Path(run_dir) / 'files' / 'config.yaml'
    if not run_config_path.exists():
        run_config_path = Path(run_dir) / 'config.yaml'
    
    if run_config_path.exists():
        # Add run name to dataframe for identification
        df['run_dir'] = run_dir
        df['run_name'] = Path(run_dir).name
    
    return df


def merge_runs_data(run_dirs: List[str]) -> pd.DataFrame:
    """Merge data from multiple runs, handling resumed training."""
    all_data = []
    
    for run_dir in run_dirs:
        df = read_wandb_history(run_dir)
        if not df.empty:
            all_data.append(df)
    
    if not all_data:
        print("No data found in any of the run directories.")
        return pd.DataFrame()
    
    # Concatenate all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by step if available
    if '_step' in combined_df.columns:
        combined_df = combined_df.sort_values('_step')
    elif 'training/global_step' in combined_df.columns:
        combined_df = combined_df.sort_values('training/global_step')
    
    return combined_df


def get_metric_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Categorize metric columns by their prefix."""
    metric_cols = defaultdict(list)
    
    for col in df.columns:
        if col.startswith('_') or col in ['run_dir', 'run_name']:
            continue
        
        # Categorize by prefix
        if '/' in col:
            category = col.split('/')[0]
            metric_cols[category].append(col)
        else:
            metric_cols['other'].append(col)
    
    return dict(metric_cols)


def plot_metrics(df: pd.DataFrame, metric_columns: List[str], title: str, output_dir: str, 
                 x_col: str = None, smooth_factor: float = 0.9):
    """Plot multiple metrics on the same figure."""
    if not metric_columns:
        return
    
    # Determine x-axis column
    if x_col is None:
        if 'training/global_step' in df.columns:
            x_col = 'training/global_step'
        elif '_step' in df.columns:
            x_col = '_step'
        else:
            x_col = df.index.name if df.index.name else 'index'
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each metric
    for metric in metric_columns:
        if metric not in df.columns:
            continue
        
        # Get non-null values
        mask = df[metric].notna()
        x_values = df[mask][x_col] if x_col in df.columns else df[mask].index
        y_values = df[mask][metric]
        
        if len(y_values) == 0:
            continue
        
        # Plot raw data with low alpha
        ax.plot(x_values, y_values, alpha=0.3, label=f'{metric} (raw)')
        
        # Apply exponential moving average for smoothing
        if len(y_values) > 1 and smooth_factor > 0:
            smoothed = pd.Series(y_values).ewm(alpha=1-smooth_factor).mean()
            ax.plot(x_values, smoothed, label=f'{metric} (smoothed)', linewidth=2)
    
    ax.set_xlabel(x_col.replace('_', ' ').title())
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
    output_path = Path(output_dir) / f'{safe_title}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot: {output_path}")


def plot_training_overview(df: pd.DataFrame, output_dir: str):
    """Create an overview plot with key training metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Define key metrics to plot
    key_metrics = [
        ('training/actor_loss', 'Actor Loss'),
        ('training/critic_loss', 'Critic Loss'),
        ('training/reward_mean', 'Mean Reward'),
        ('training/kl_div', 'KL Divergence')
    ]
    
    x_col = 'training/global_step' if 'training/global_step' in df.columns else '_step'
    
    for idx, (metric, title) in enumerate(key_metrics):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        if metric in df.columns:
            mask = df[metric].notna()
            x_values = df[mask][x_col] if x_col in df.columns else df[mask].index
            y_values = df[mask][metric]
            
            if len(y_values) > 0:
                # Plot raw data
                ax.plot(x_values, y_values, alpha=0.3, color='blue')
                
                # Plot smoothed data
                if len(y_values) > 1:
                    smoothed = pd.Series(y_values).ewm(alpha=0.1).mean()
                    ax.plot(x_values, smoothed, color='red', linewidth=2)
                
                ax.set_title(title)
                ax.set_xlabel('Global Step')
                ax.set_ylabel(title)
                ax.grid(True, alpha=0.3)
    
    plt.suptitle('Training Overview', fontsize=16)
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'training_overview.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved overview plot: {output_path}")


def create_summary_table(df: pd.DataFrame, output_dir: str):
    """Create a summary table of final metrics."""
    summary = {}
    
    # Get the last non-null value for each metric
    for col in df.columns:
        if col.startswith('_') or col in ['run_dir', 'run_name']:
            continue
        
        last_value = df[col].dropna().iloc[-1] if not df[col].dropna().empty else None
        if last_value is not None:
            summary[col] = last_value
    
    # Convert to DataFrame for nice formatting
    summary_df = pd.DataFrame([summary]).T
    summary_df.columns = ['Final Value']
    
    # Save as CSV
    output_path = Path(output_dir) / 'final_metrics_summary.csv'
    summary_df.to_csv(output_path)
    
    print(f"\nFinal Metrics Summary:")
    print(summary_df)
    print(f"\nSaved summary to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize WandB offline run data')
    parser.add_argument('--wandb-dirs', nargs='+', required=True,
                        help='WandB directories to process (can specify multiple for resumed runs)')
    parser.add_argument('--output-dir', default='./wandb_plots',
                        help='Directory to save plots')
    parser.add_argument('--smooth-factor', type=float, default=0.9,
                        help='Smoothing factor for exponential moving average (0-1, higher = more smooth)')
    parser.add_argument('--find-runs', action='store_true',
                        help='Automatically find all runs in the specified directories')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all run directories
    all_run_dirs = []
    for wandb_dir in args.wandb_dirs:
        if args.find_runs:
            # Find all runs in this directory
            runs = find_wandb_runs(wandb_dir)
            all_run_dirs.extend(runs)
            print(f"Found {len(runs)} runs in {wandb_dir}")
        else:
            # Use the directory directly
            all_run_dirs.append(wandb_dir)
    
    if not all_run_dirs:
        print("No run directories found.")
        return
    
    print(f"\nProcessing {len(all_run_dirs)} run directories...")
    
    # Merge data from all runs
    df = merge_runs_data(all_run_dirs)
    
    if df.empty:
        print("No data to plot.")
        return
    
    print(f"\nTotal data points: {len(df)}")
    
    # Get metric columns organized by category
    metric_categories = get_metric_columns(df)
    
    # Create training overview plot
    plot_training_overview(df, output_dir)
    
    # Plot metrics by category
    for category, metrics in metric_categories.items():
        if metrics:
            print(f"\nPlotting {category} metrics...")
            plot_metrics(df, metrics, f'{category.title()} Metrics', 
                        output_dir, smooth_factor=args.smooth_factor)
    
    # Create summary table
    create_summary_table(df, output_dir)
    
    print(f"\nAll plots saved to: {output_dir}")


if __name__ == '__main__':
    main()