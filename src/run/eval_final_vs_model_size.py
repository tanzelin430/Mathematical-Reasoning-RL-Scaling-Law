#!/usr/bin/env python3
"""
Plot best test loss vs model size for multiple evaluations
Shows only the BEST performance (lowest test loss) for each model size
Creates a 2x4 subplot grid with one evaluation per subplot
Generates separate PDFs for base and instruct (16 subplots total)

Usage:
  python eval_final_vs_model_size.py --metric ErrRate --data-source base
  python eval_final_vs_model_size.py --metric ErrRate --data-source instruct

Options:
- metric: Metric to plot (e.g., 'ErrRate', 'R')
- data_source: Data source to use ('base' or 'instruct')
"""

import argparse
from src.common import data_proc
from src.common import config
from src.common import plot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Final Test Loss vs Model Size - Subplot Grid')
    parser.add_argument('--metric', type=str, default='ErrRate',
                        help='Metric to plot (default: ErrRate)')
    parser.add_argument('--data-source', type=str, default='base',
                        choices=['base', 'instruct', 'llama-base', 'llama-instruct'],
                        help='Data source to use (default: base)')
    parser.add_argument('--warmup-clip', type=int, default=0,
                        help='Number of warmup steps to clip (default: 0)')
    args = parser.parse_args()

    metric = args.metric
    data_source = args.data_source
    warmup_clip = args.warmup_clip

    print(f"Configuration:")
    print(f"  Metric: {metric}")
    print(f"  Data source: {data_source}")
    print(f"  Warmup clip: {warmup_clip}")
    print()

    # Load data
    df = data_proc.load_and_preprocess(config.CSV_MAP[data_source])

    # Define all evaluations (8 total for 2x4 grid)
    all_eval_keys = [
        'holdout_score',
        'val/test_score/openai/gsm8k',
        'val/test_score/math__math',
        'val/test_score/aime2024',
        'val/test_score/aimeamc2023',
        'val/test_score/stem__supergpqa',
        'val/test_score/codegen__humaneval',
        'val/test_score/logic__zebra_puzzle_dataset'
    ]

    test_evals = {k: config.TEST_EVALS[k] for k in all_eval_keys if k in config.TEST_EVALS}

    # Define evaluation order for subplots (2 rows x 4 columns)
    eval_order_list = [
        "Holdout Validation",
        "GSM8K",
        "MATH-500",
        "AIME2024",
        "AMC2023",
        "SuperGPQA",
        "CodeGen - HumanEval",
        "Logic - Zebra Puzzle"
    ]

    # Get all model sizes
    model_sizes = sorted(df['N'].unique())
    print(f"Model sizes found: {[plot.human_format_N(n) for n in model_sizes]}")

    # Create 2x4 subplot grid
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()  # Flatten to 1D array for easy iteration

    # Sort evaluations by the defined order
    ordered_evals = []
    for eval_label in eval_order_list:
        for eval_name, eval_config in test_evals.items():
            if eval_config['plot_str'] == eval_label:
                ordered_evals.append((eval_name, eval_config))
                break

    # Add any remaining evals not in the order list
    for eval_name, eval_config in test_evals.items():
        if not any(eval_config['plot_str'] == item[1]['plot_str'] for item in ordered_evals):
            ordered_evals.append((eval_name, eval_config))

    # Plot each evaluation in its own subplot
    for subplot_idx, (eval_name, eval_config) in enumerate(ordered_evals):
        if subplot_idx >= 8:  # Only plot first 8 evals (2x4 grid)
            break

        ax = axes[subplot_idx]
        # Check if evaluation column exists in data
        if eval_name not in df.columns:
            print(f"Warning: {eval_name} not found in data, skipping...")
            ax.set_visible(False)
            continue

        # For each model size, get the BEST performance and step 0 performance
        best_values = []
        step0_values = []
        common_model_sizes = []

        for N in model_sizes:
            df_n = df[df['N'] == N].copy()

            # Get step 0 value (untrained performance)
            df_step0 = df_n[df_n['step'] == 0]
            if len(df_step0) > 0:
                if metric == 'ErrRate':
                    step0_value = 1 - df_step0[eval_name].iloc[0]
                elif metric == 'R':
                    step0_value = df_step0[eval_name].iloc[0]
                else:
                    step0_value = df_step0[eval_name].iloc[0]
            else:
                step0_value = None

            # Apply warmup clip for trained data
            if warmup_clip > 0:
                df_n = df_n[df_n['step'] >= warmup_clip]

            if len(df_n) == 0:
                continue

            # Get the BEST metric value (not the last one)
            if metric == 'ErrRate':
                # ErrRate = 1 - score, lower is better
                df_n['metric_value'] = 1 - df_n[eval_name]
                best_value = df_n['metric_value'].min()  # Best = minimum ErrRate
            elif metric == 'R':
                # R (reward/score), higher is better
                best_value = df_n[eval_name].max()  # Best = maximum R
            else:
                # Default: assume lower is better
                best_value = df_n[eval_name].min()

            # Add data points - include step0 if available
            if step0_value is not None:
                best_values.append(best_value)
                step0_values.append(step0_value)
                common_model_sizes.append(N)
            else:
                # If no step0, just add best value
                best_values.append(best_value)
                common_model_sizes.append(N)

        if len(best_values) == 0:
            print(f"Warning: No data for {eval_name}, skipping...")
            ax.set_visible(False)
            continue

        # Get color for this evaluation
        eval_color = config.get_color_for_curve(eval_name)

        # Plot with step 0 baseline if available
        if len(step0_values) > 0:
            # Plot step 0 baseline (dashed line)
            ax.plot(common_model_sizes, step0_values,
                   color=eval_color, linewidth=1.5, alpha=0.5,
                   linestyle='--', marker='s', markersize=6,
                   markerfacecolor='none', markeredgecolor=eval_color,
                   markeredgewidth=1.5, label='Pre-training')

            # Plot best performance (solid line)
            ax.plot(common_model_sizes, best_values,
                   color=eval_color, linewidth=2.0, alpha=1.0,
                   marker='o', markersize=8, markerfacecolor=eval_color,
                   markeredgecolor='none', label='RL-Post Training')

            # Fill area between step 0 and best performance
            ax.fill_between(common_model_sizes, step0_values, best_values,
                           color=eval_color, alpha=0.2, label='Improvement')
        else:
            # If no step0 data, just plot best performance
            ax.plot(common_model_sizes, best_values,
                   color=eval_color, linewidth=2.0, alpha=1.0,
                   marker='o', markersize=8, markerfacecolor=eval_color,
                   markeredgecolor='none', label='RL-Post Training')

        # Apply settings to this subplot
        metric_label = config.DEFAULT_LABELS.get(metric, metric)
        ax.set_xscale('log')
        ax.set_xlabel('Model Size', fontsize=9)
        ax.set_ylabel(metric_label, fontsize=9)
        ax.set_title(eval_config['plot_str'], fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Format x-axis to show powers of 10
        from matplotlib.ticker import LogLocator, FuncFormatter

        def format_func(value, tick_number):
            if value <= 0:
                return ''
            power = np.log10(value)
            if abs(power - round(power)) < 0.01:
                return f'$10^{{{int(round(power))}}}$'
            return ''

        ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
        ax.xaxis.set_major_formatter(FuncFormatter(format_func))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda x, p: ''))

        # Add legend only to first subplot
        if subplot_idx == 0 and len(step0_values) > 0:
            ax.legend(loc='best', fontsize=8)

    # Add overall title first
    metric_label = config.DEFAULT_LABELS.get(metric, metric)
    fig.suptitle(f'Cross-Domain Evaluation - Post-Training {metric_label} vs Model Size ({data_source.title()})',
                 fontsize=16, fontweight='bold', y=0.99)

    # Adjust layout to prevent overlap, leaving space for suptitle
    plt.tight_layout(rect=[0, 0, 1, 1])

    # Save figure
    filename = config.OUTPUT_BASE_DIR / f"crossdomain_{data_source}_{metric}_vs_N.pdf"
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved figure: {filename}")

    plt.close(fig)
    print(f"\nPlot complete!")

if __name__ == "__main__":
    main()
