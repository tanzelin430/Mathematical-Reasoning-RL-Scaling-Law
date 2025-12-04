#!/usr/bin/env python3
"""
Analyze and visualize uncertainty metrics for scaling law fits.

This script generates:
1. Bar plots showing mean ± std for each model size
2. LaTeX tables with comprehensive uncertainty metrics
3. Confidence interval estimates (bootstrap if needed)

Usage:
    uv run python scripts/analyze_uncertainty.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.config import CSV_MAP, get_physical_dimensions
from src.common.data_proc import prepare_eval_data, load_and_preprocess, apply_clip


def load_fit_results(fit_path):
    """Load fitting results from JSON."""
    with open(fit_path, 'r') as f:
        return json.load(f)


def load_raw_data_for_source(data_source, eval_name, warmup_clip=0):
    """Load raw data (before averaging runs) for a specific data source.

    Returns DataFrame with runid column preserved.
    """
    # Load raw data
    csv_paths = CSV_MAP[data_source]
    df = load_and_preprocess(csv_paths)

    # Rename eval column to a standard name
    df['ErrRate'] = 1 - df[eval_name]

    # Get physical dimensions
    physical_dimensions = get_physical_dimensions(data_source)
    physical_curve_column = physical_dimensions[0]  # N
    physical_x_column = physical_dimensions[1]      # step

    # Remove step=0
    df = df[df[physical_x_column] > 0].reset_index(drop=True)

    # Apply warmup clipping
    if warmup_clip > 0:
        df = apply_clip(df, curve_column=physical_curve_column, warmup_clip=warmup_clip)

    return df


def compute_uncertainty_metrics(df_raw, model_sizes, metric='ErrRate'):
    """
    Compute uncertainty metrics for each model size.

    Strategy: For each model size, compute std for each step across runs,
    then average all step-level stds. If a step has only 1 run, treat std=0.

    Args:
        df_raw: Raw dataframe with runid column (before averaging)
        model_sizes: List of model sizes to analyze
        metric: Metric to analyze (default: ErrRate)

    Returns:
        DataFrame with columns: N, mean, avg_std, sem, ci_lower, ci_upper, n_steps
    """
    from scipy import stats

    results = []

    for N in model_sizes:
        df_n = df_raw[df_raw['N'] == N]

        if len(df_n) == 0:
            print(f"Warning: No data for N={N/1e9:.1f}B")
            continue

        # Collect std for each step
        all_steps = sorted(df_n['step'].unique())
        step_stds = []
        step_means = []

        for step in all_steps:
            df_step = df_n[df_n['step'] == step]
            err_rates = df_step[metric].values

            step_mean = np.mean(err_rates)
            step_means.append(step_mean)

            if len(err_rates) >= 2:
                # Multiple runs: compute std
                step_std = np.std(err_rates, ddof=1)
            else:
                # Single run: std = 0
                step_std = 0.0

            step_stds.append(step_std)

        # Average std across all steps
        avg_std = np.mean(step_stds)

        # Final performance (mean of last step)
        final_mean = step_means[-1]

        # Compute SEM and CI based on average std
        n_runs = 3  # Assume 3 runs
        sem_val = avg_std / np.sqrt(n_runs)
        ci = stats.t.interval(0.95, n_runs - 1, loc=final_mean, scale=sem_val)

        results.append({
            'N': N,
            'mean': final_mean,
            'avg_std': avg_std,
            'sem': sem_val,
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'n_steps': len(all_steps)
        })

    return pd.DataFrame(results)


def plot_uncertainty_bars(uncertainty_df, data_source, eval_name, output_dir='outputs'):
    """Generate bar plot with error bars."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Format labels properly for 0.5B, 1.5B, etc.
    model_labels = []
    for n in uncertainty_df['N']:
        if n < 1e9:
            model_labels.append(f"{n/1e9:.1f}B")
        else:
            model_labels.append(f"{int(n/1e9)}B")

    x_pos = np.arange(len(model_labels))

    # Bar plot with error bars (avg_std)
    ax.bar(x_pos, uncertainty_df['mean'],
           yerr=uncertainty_df['avg_std'],
           capsize=5, alpha=0.7, color='steelblue',
           label='Mean ± Avg Std')

    # Add confidence intervals as markers
    ax.errorbar(x_pos, uncertainty_df['mean'],
                yerr=[uncertainty_df['mean'] - uncertainty_df['ci_lower'],
                      uncertainty_df['ci_upper'] - uncertainty_df['mean']],
                fmt='none', ecolor='red', capsize=3, linewidth=1.5,
                label='95% CI')

    ax.set_xlabel('Model Size', fontsize=12)
    ax.set_ylabel('Error Rate', fontsize=12)
    ax.set_title(f'Uncertainty Analysis: {data_source.capitalize()} - {eval_name}',
                 fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    output_path = Path(output_dir) / f'uncertainty_{data_source}_{eval_name}.pdf'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved: {output_path}")
    plt.close()


def generate_latex_table(uncertainty_df, data_source, eval_name):
    """Generate LaTeX table with uncertainty metrics.

    Note: Mean is the final step performance (averaged across runs).
    """
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append(f"\\caption{{Uncertainty Analysis: {data_source.capitalize()} - {eval_name}}}")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\hline")
    lines.append("Model & Final ErrRate & Avg Std & SEM & 95\\% CI \\\\")
    lines.append("\\hline")

    for _, row in uncertainty_df.iterrows():
        n = row['N']
        if n < 1e9:
            model_label = f"{n/1e9:.1f}B"
        else:
            model_label = f"{int(n/1e9)}B"

        mean_str = f"{row['mean']:.4f}"
        std_str = f"{row['avg_std']:.4f}"
        sem_str = f"{row['sem']:.4f}"
        ci_str = f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"

        lines.append(f"{model_label} & {mean_str} & {std_str} & {sem_str} & {ci_str} \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def main():
    """Main analysis workflow."""
    # Configuration
    data_sources = ['base', 'instruct']
    eval_name = 'holdout_score'
    model_sizes = [0.5e9, 1.5e9, 3e9, 7e9, 14e9, 32e9, 72e9]
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)

    all_tables = []

    for data_source in data_sources:
        print(f"\n=== Processing {data_source} ===")

        # Load raw data (with runid preserved)
        warmup_clip = 0 if data_source == 'base' else 4
        df_raw = load_raw_data_for_source(data_source, eval_name, warmup_clip)

        # Compute uncertainty metrics
        uncertainty_df = compute_uncertainty_metrics(df_raw, model_sizes, metric='ErrRate')

        # Generate plots
        plot_uncertainty_bars(uncertainty_df, data_source, eval_name, output_dir)

        # Generate LaTeX table
        latex_table = generate_latex_table(uncertainty_df, data_source, eval_name)
        all_tables.append(f"\n% {data_source.upper()} - {eval_name}\n")
        all_tables.append(latex_table)

        # Print summary
        print(f"\nUncertainty Summary for {data_source}:")
        print(uncertainty_df.to_string(index=False))

    # Save all tables to file
    table_path = output_dir / 'uncertainty_tables.tex'
    with open(table_path, 'w') as f:
        f.write('\n\n'.join(all_tables))
    print(f"\n\nSaved LaTeX tables to: {table_path}")

    print("\n=== Analysis Complete ===")


if __name__ == '__main__':
    main()
