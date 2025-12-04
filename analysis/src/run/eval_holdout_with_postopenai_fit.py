#!/usr/bin/env python3
"""
Plot holdout test loss vs model size with PostOpenAI model fitting curve

Shows:
1. Pre-training performance (step 0, dashed line)
2. Best post-training performance (solid line with markers)
3. PostOpenAI model prediction at C→∞ (theoretical limit, dashed line)

PostOpenAI model: L(N, C) = [(N0/N)^β + B0/(C+C0)]^α
As C→∞: L(N, ∞) = (N0/N)^(α×β)

Usage:
  python eval_holdout_with_postopenai_fit.py --data-source base
  python eval_holdout_with_postopenai_fit.py --data-source instruct
"""

import argparse
from src.common import data_proc
from src.common import config
from src.common import plot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from pathlib import Path


def load_postopenai_params(data_source):
    """Load PostOpenAI model parameters from fitted results"""
    fit_file = Path("outputs/fits_postopenai_full.json")

    if not fit_file.exists():
        print(f"Warning: {fit_file} not found, skipping PostOpenAI fit curve")
        return None

    with open(fit_file, 'r') as f:
        fit_data = json.load(f)

    # Find the matching fit for this data source
    for fit in fit_data['fits']:
        if fit['context']['data_source'] == data_source:
            params = fit['params']

            # Build lookup tables for B0 and C0
            B0_lookup = {
                0.5e9: params['B0_0_5'],
                1.5e9: params['B0_1_5'],
                3e9: params['B0_3'],
                7e9: params['B0_7'],
                14e9: params['B0_14'],
                32e9: params['B0_32'],
                72e9: params['B0_72']
            }
            C0_lookup = {
                0.5e9: params['C0_0_5'],
                1.5e9: params['C0_1_5'],
                3e9: params['C0_3'],
                7e9: params['C0_7'],
                14e9: params['C0_14'],
                32e9: params['C0_32'],
                72e9: params['C0_72']
            }

            return {
                'N0': params['N0'],
                'beta': params['beta'],
                'alpha': params['alpha'],
                'r2': fit['info']['r2'],
                'B0_lookup': B0_lookup,
                'C0_lookup': C0_lookup
            }

    return None


def postopenai_limit(N, N0, beta, alpha):
    """
    Calculate PostOpenAI model prediction as C→∞

    L(N, ∞) = (N0/N)^(α×β)
    """
    return np.power(N0 / N, alpha * beta)


def postopenai_at_C0(N_values, N0, beta, alpha, B0_lookup, C0_lookup):
    """
    Calculate PostOpenAI model prediction at C=0

    L(N, 0) = [(N0/N)^β + B0(N)/C0(N)]^α

    Args:
        N_values: array of model sizes
        N0, beta, alpha: global parameters
        B0_lookup, C0_lookup: lookup tables for B0 and C0
    """
    results = []
    for N in N_values:
        # Find closest N in lookup table
        closest_N = min(B0_lookup.keys(), key=lambda n: abs(n - N))
        if abs(closest_N - N) > 1:  # Tolerance
            # If not in lookup, skip this point
            results.append(np.nan)
            continue

        B0 = B0_lookup[closest_N]
        C0 = C0_lookup[closest_N]

        # Calculate: [(N0/N)^β + B0/C0]^α
        term1 = np.power(N0 / N, beta)
        term2 = B0 / C0
        result = np.power(term1 + term2, alpha)
        results.append(result)

    return np.array(results)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Holdout Test Loss vs Model Size with PostOpenAI Fit')
    parser.add_argument('--data-source', type=str, default='base',
                        choices=['base', 'instruct', 'llama-base', 'llama-instruct'],
                        help='Data source to use (default: base)')
    parser.add_argument('--warmup-clip', type=int, default=None,
                        help='Number of warmup steps to clip (default: 0 for base, 4 for instruct)')
    parser.add_argument('--metric', type=str, default='ErrRate',
                        help='Metric to plot (default: ErrRate)')
    args = parser.parse_args()

    data_source = args.data_source
    metric = args.metric

    # Set default warmup_clip based on data source
    if args.warmup_clip is None:
        warmup_clip = 0 if data_source == 'base' else 4
    else:
        warmup_clip = args.warmup_clip

    print(f"Configuration:")
    print(f"  Data source: {data_source}")
    print(f"  Metric: {metric}")
    print(f"  Warmup clip: {warmup_clip}")
    print()

    # Load data
    df = data_proc.load_and_preprocess(config.CSV_MAP[data_source])

    # Use holdout_score evaluation
    eval_name = 'holdout_score'
    eval_config = config.TEST_EVALS[eval_name]

    # Check if evaluation column exists
    if eval_name not in df.columns:
        print(f"Error: {eval_name} not found in data")
        return

    # Get all model sizes
    model_sizes = sorted(df['N'].unique())
    print(f"Model sizes found: {[plot.human_format_N(n) for n in model_sizes]}")

    # Load PostOpenAI parameters
    postopenai_params = load_postopenai_params(data_source)
    if postopenai_params:
        print(f"\nPostOpenAI model parameters loaded:")
        print(f"  N0 = {postopenai_params['N0']:.2e}")
        print(f"  β = {postopenai_params['beta']:.4f}")
        print(f"  α = {postopenai_params['alpha']:.4f}")
        print(f"  R² = {postopenai_params['r2']:.4f}")
        print(f"  α×β = {postopenai_params['alpha'] * postopenai_params['beta']:.4f}")

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

        # Add data points
        if step0_value is not None:
            best_values.append(best_value)
            step0_values.append(step0_value)
            common_model_sizes.append(N)
        else:
            best_values.append(best_value)
            common_model_sizes.append(N)

    if len(best_values) == 0:
        print(f"Error: No data points found")
        return

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Get color for holdout evaluation
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
               markeredgecolor='none', label='RL-Post Training (Best)')

        # Fill area between step 0 and best performance
        ax.fill_between(common_model_sizes, step0_values, best_values,
                       color=eval_color, alpha=0.2, label='Improvement')
    else:
        # If no step0 data, just plot best performance
        ax.plot(common_model_sizes, best_values,
               color=eval_color, linewidth=2.0, alpha=1.0,
               marker='o', markersize=8, markerfacecolor=eval_color,
               markeredgecolor='none', label='RL-Post Training (Best)')

    # Plot PostOpenAI model prediction (C→∞)
    if postopenai_params:
        # Create smooth curve for model prediction
        N_smooth = np.logspace(np.log10(min(common_model_sizes)),
                               np.log10(max(common_model_sizes)), 100)
        L_limit = postopenai_limit(N_smooth,
                                    postopenai_params['N0'],
                                    postopenai_params['beta'],
                                    postopenai_params['alpha'])

        ax.plot(N_smooth, L_limit,
               color='red', linewidth=2.0, alpha=0.8,
               linestyle='-.',
               label=f'PostOpenAI Fit (C→∞, R²={postopenai_params["r2"]:.3f})')

        # Plot PostOpenAI model prediction at C=0
        L_at_C0 = postopenai_at_C0(np.array(common_model_sizes),
                                    postopenai_params['N0'],
                                    postopenai_params['beta'],
                                    postopenai_params['alpha'],
                                    postopenai_params['B0_lookup'],
                                    postopenai_params['C0_lookup'])

        ax.plot(common_model_sizes, L_at_C0,
               color='blue', linewidth=2.0, alpha=0.8,
               linestyle='--',
               label='PostOpenAI Fit (C=0, No Training)')

    # Apply settings
    metric_label = config.DEFAULT_LABELS.get(metric, metric)
    ax.set_xscale('log')
    ax.set_xlabel('Model Size (N)', fontsize=12)
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_title(f'{eval_config["plot_str"]} - {metric_label} vs Model Size ({data_source.title()})',
                 fontsize=14, fontweight='bold')
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

    # Add legend
    ax.legend(loc='best', fontsize=10)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    filename = config.OUTPUT_BASE_DIR / f"holdout_with_postopenai_{data_source}_{metric}_vs_N.pdf"
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved figure: {filename}")

    plt.close(fig)
    print(f"\nPlot complete!")


if __name__ == "__main__":
    main()
