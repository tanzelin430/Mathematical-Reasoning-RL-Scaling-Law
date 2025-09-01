#!/usr/bin/env python3
"""
VeRL WandB Offline Plotter

A simplified script specifically for plotting VeRL training metrics from WandB offline data.
Handles multiple resumed runs and creates focused plots for RL training analysis.

Usage:
    # Plot a single run
    python plot_verl_wandb_offline.py --wandb-dir ~/Agentic-RL-Scaling-Law/wandb_tanzl/
    
    # Plot multiple resumed runs
    python plot_verl_wandb_offline.py --wandb-dir ~/Agentic-RL-Scaling-Law/wandb_tanzl/ --run-pattern "qwen*"
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import glob

# Set up plotting style
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")


def load_wandb_offline_data(run_path: Path) -> pd.DataFrame:
    """Load data from a single WandB offline run."""
    history_file = run_path / "files" / "wandb-history.jsonl"
    
    if not history_file.exists():
        # Try alternative locations
        history_file = run_path / "wandb-history.jsonl"
        if not history_file.exists():
            print(f"Warning: No history file found in {run_path}")
            return pd.DataFrame()
    
    data = []
    with open(history_file, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    if not data:
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    df['run_name'] = run_path.name
    return df


def merge_resumed_runs(dfs: list) -> pd.DataFrame:
    """Merge multiple dataframes from resumed runs."""
    if not dfs:
        return pd.DataFrame()
    
    # Concatenate all dataframes
    merged = pd.concat(dfs, ignore_index=True)
    
    # Sort by global step if available
    if 'training/global_step' in merged.columns:
        merged = merged.sort_values('training/global_step')
    elif '_step' in merged.columns:
        merged = merged.sort_values('_step')
    
    # Remove duplicates based on step
    if 'training/global_step' in merged.columns:
        merged = merged.drop_duplicates(subset=['training/global_step'], keep='last')
    
    return merged


def plot_reward_metrics(df: pd.DataFrame, output_dir: Path):
    """Plot reward-related metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # Define metrics to plot
    reward_metrics = [
        ('training/reward_mean', 'Mean Reward', axes[0]),
        ('training/reward_std', 'Reward Std', axes[1]),
        ('val/math/pass1', 'Math Pass@1', axes[2]),
        ('val/overall_pass1', 'Overall Pass@1', axes[3])
    ]
    
    x_col = 'training/global_step' if 'training/global_step' in df.columns else df.index
    
    for metric, title, ax in reward_metrics:
        if metric in df.columns:
            y_data = df[metric].dropna()
            if not y_data.empty:
                x_data = df.loc[y_data.index, 'training/global_step'] if 'training/global_step' in df.columns else y_data.index
                
                # Plot raw data
                ax.plot(x_data, y_data, 'o-', alpha=0.6, markersize=3, label='Raw')
                
                # Add smoothed line if enough data
                if len(y_data) > 5:
                    smoothed = y_data.rolling(window=5, min_periods=1).mean()
                    ax.plot(x_data, smoothed, linewidth=2, label='Smoothed')
                
                ax.set_xlabel('Global Step')
                ax.set_ylabel(title)
                ax.set_title(title)
                ax.legend()
                ax.grid(True, alpha=0.3)
    
    plt.suptitle('Reward Metrics', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / 'reward_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_loss_metrics(df: pd.DataFrame, output_dir: Path):
    """Plot loss-related metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # Define metrics to plot
    loss_metrics = [
        ('training/actor_loss', 'Actor Loss', axes[0]),
        ('training/critic_loss', 'Critic Loss', axes[1]),
        ('training/kl_div', 'KL Divergence', axes[2]),
        ('training/entropy', 'Entropy', axes[3])
    ]
    
    x_col = 'training/global_step' if 'training/global_step' in df.columns else df.index
    
    for metric, title, ax in loss_metrics:
        if metric in df.columns:
            y_data = df[metric].dropna()
            if not y_data.empty:
                x_data = df.loc[y_data.index, 'training/global_step'] if 'training/global_step' in df.columns else y_data.index
                
                # Plot
                ax.plot(x_data, y_data, '-', alpha=0.8, linewidth=1.5)
                
                ax.set_xlabel('Global Step')
                ax.set_ylabel(title)
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
                
                # Add trend line for losses
                if 'loss' in metric.lower() and len(y_data) > 10:
                    z = np.polyfit(range(len(y_data)), y_data.values, 1)
                    p = np.poly1d(z)
                    ax.plot(x_data, p(range(len(y_data))), "--", alpha=0.5, label=f'Trend: {z[0]:.2e}')
                    ax.legend()
    
    plt.suptitle('Loss Metrics', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_validation_by_domain(df: pd.DataFrame, output_dir: Path):
    """Plot validation metrics by domain."""
    domains = ['math', 'code', 'logic', 'stem']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for domain in domains:
        metric = f'val/{domain}/pass1'
        if metric in df.columns:
            y_data = df[metric].dropna()
            if not y_data.empty:
                x_data = df.loc[y_data.index, 'training/global_step'] if 'training/global_step' in df.columns else y_data.index
                ax.plot(x_data, y_data, 'o-', label=domain.capitalize(), markersize=6, linewidth=2)
    
    ax.set_xlabel('Global Step')
    ax.set_ylabel('Pass@1')
    ax.set_title('Validation Pass@1 by Domain')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'validation_by_domain.png', dpi=150, bbox_inches='tight')
    plt.close()


def generate_summary_report(df: pd.DataFrame, output_dir: Path):
    """Generate a text summary report."""
    report_lines = []
    report_lines.append("VeRL Training Summary Report")
    report_lines.append("=" * 50)
    report_lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Training progress
    if 'training/global_step' in df.columns:
        total_steps = df['training/global_step'].max()
        report_lines.append(f"Total training steps: {total_steps}")
    
    # Final metrics
    report_lines.append("\nFinal Metrics:")
    report_lines.append("-" * 30)
    
    final_metrics = [
        ('training/reward_mean', 'Final mean reward'),
        ('val/overall_pass1', 'Overall Pass@1'),
        ('val/math/pass1', 'Math Pass@1'),
        ('val/code/pass1', 'Code Pass@1'),
        ('val/logic/pass1', 'Logic Pass@1'),
        ('val/stem/pass1', 'STEM Pass@1'),
        ('training/actor_loss', 'Final actor loss'),
        ('training/critic_loss', 'Final critic loss'),
    ]
    
    for metric, name in final_metrics:
        if metric in df.columns:
            values = df[metric].dropna()
            if not values.empty:
                final_value = values.iloc[-1]
                report_lines.append(f"{name}: {final_value:.4f}")
    
    # Best validation scores
    report_lines.append("\nBest Validation Scores:")
    report_lines.append("-" * 30)
    
    for domain in ['math', 'code', 'logic', 'stem', 'overall']:
        metric = f'val/{domain}/pass1' if domain != 'overall' else 'val/overall_pass1'
        if metric in df.columns:
            values = df[metric].dropna()
            if not values.empty:
                best_value = values.max()
                best_step = df.loc[values.idxmax(), 'training/global_step'] if 'training/global_step' in df.columns else values.idxmax()
                report_lines.append(f"{domain.capitalize()}: {best_value:.4f} (at step {best_step})")
    
    # Save report
    report_path = output_dir / 'training_summary.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print('\n'.join(report_lines))
    print(f"\nReport saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot VeRL training metrics from WandB offline data')
    parser.add_argument('--wandb-dir', type=str, default='~/Agentic-RL-Scaling-Law/wandb_tanzl',
                        help='Base WandB directory')
    parser.add_argument('--run-pattern', type=str, default=None,
                        help='Pattern to match run names (e.g., "qwen*14B*")')
    parser.add_argument('--output-dir', type=str, default='./verl_plots',
                        help='Directory to save plots')
    parser.add_argument('--latest-only', action='store_true',
                        help='Only use the latest run matching the pattern')
    
    args = parser.parse_args()
    
    # Expand paths
    wandb_dir = Path(args.wandb_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find runs
    if args.run_pattern:
        run_dirs = sorted(wandb_dir.glob(f"*{args.run_pattern}*"))
    else:
        # Find all directories that look like wandb runs
        run_dirs = [d for d in wandb_dir.iterdir() if d.is_dir() and 
                    (d.name.startswith('offline-run-') or d.name.startswith('run-'))]
    
    if args.latest_only and run_dirs:
        run_dirs = [max(run_dirs, key=lambda x: x.stat().st_mtime)]
    
    print(f"Found {len(run_dirs)} run(s) to process")
    
    # Load data from all runs
    all_dfs = []
    for run_dir in run_dirs:
        print(f"Loading data from: {run_dir.name}")
        df = load_wandb_offline_data(run_dir)
        if not df.empty:
            all_dfs.append(df)
    
    if not all_dfs:
        print("No data found in any runs!")
        return
    
    # Merge data
    merged_df = merge_resumed_runs(all_dfs)
    print(f"Total data points: {len(merged_df)}")
    
    # Create plots
    print("\nGenerating plots...")
    plot_reward_metrics(merged_df, output_dir)
    print("✓ Reward metrics plot saved")
    
    plot_loss_metrics(merged_df, output_dir)
    print("✓ Loss metrics plot saved")
    
    plot_validation_by_domain(merged_df, output_dir)
    print("✓ Validation by domain plot saved")
    
    # Generate summary report
    generate_summary_report(merged_df, output_dir)
    
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()