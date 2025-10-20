#!/usr/bin/env python3
"""
Scaling Law Pipeline - Multi-Eval Analysis
Processes multiple test evals from Experiment1 data and generates scaling law plots for each eval
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
import itertools
import matplotlib
import argparse

# Enable LaTeX rendering for titles
matplotlib.rcParams['text.usetex'] = False  # Use mathtext instead of LaTeX for better compatibility
matplotlib.rcParams['font.family'] = 'serif'
    

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.common import intrinsic
from src.common import data_proc
from src.common.plot import plot_basic, plot_basic_settings
from src.common import config

# =============================================================================
# CONFIGURATION - Use config constants
# =============================================================================

phi_global = 1.0


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Scaling Law Pipeline - Multi-Eval Analysis')
    
    parser.add_argument('--plot-enable', 
                        nargs='+', 
                        default=["k"], 
                        choices=["k", "E0"],
                        help='Specify which plots to generate (default: ["k"])')
    
    parser.add_argument('--fitting-type', 
                        default="step", 
                        choices=["C", "C_raw", "step", "E", "response_length"],
                        help='Fitting type: "C" for L(N,C), "C_raw" for L(N,C_raw), "step" for L(N,step), "E" for L(N,D) (default: "step")')
    
    parser.add_argument('--data-sources', 
                        nargs='+', 
                        default=["exp2-base", "exp2-instruct"],
                        choices=["base", "instruct", "exp2-base", "exp2-instruct"],
                        help='Model types to compare (default: ["exp2-base", "exp2-instruct"])')
    
    parser.add_argument('--curve-column', 
                        default="Tau", 
                        choices=["N", "Tau"],
                        help='Column to use for curve fitting and plotting (default: "Tau")')
    
    parser.add_argument('--warmup-clip-num', 
                        type=int, 
                        default=5,
                        help='Number of warmup steps to clip (default: 5)')
    
    parser.add_argument('--title', 
                        type=str, 
                        default=None,
                        help='Base title for plots (if not specified, uses default format)')
    
    parser.add_argument('--title-k', 
                        type=str, 
                        default=None,
                        help='Title for k plots (if not specified, uses --title or default format)')
    
    parser.add_argument('--title-E0', 
                        type=str, 
                        default=None,
                        help='Title for E0 plots (if not specified, uses --title or default format)')
    
    return parser.parse_args()

# Parse command line arguments
args = parse_arguments()

# Use holdout score evaluation
eval_name = config.DEFAULT_TEST_EVAL
warmup_clip_num = args.warmup_clip_num

# Plot control: specify which plots to generate
plot_enable = args.plot_enable
fitting_type = args.fitting_type
data_sources = args.data_sources
curve_column = args.curve_column

# Title configuration
base_title = args.title
title_k = args.title_k if args.title_k is not None else base_title
title_E0 = args.title_E0 if args.title_E0 is not None else base_title

curve_mask = [1, 2, 5, 20, 25, 50, 100] if curve_column == "Tau" else None
#[14e9] if curve_column == "N" else None

def perform_fitting(df_mean, fitting_type, data_source, display_names, curve_column):
    """
    Perform fitting based on fitting_type
    """
    from sklearn.linear_model import LinearRegression
    
    display_name = display_names.get(fitting_type, fitting_type)
    curve_label = config.DEFAULT_LABELS.get(curve_column, curve_column)
    print(f"  Performing L({curve_label},{display_name}) fitting for {data_source} model")
    
    # Get unique curve values (e.g., model sizes)
    unique_curve_values = sorted(df_mean[curve_column].unique())
    fit_results = {}
    
    for curve_val in unique_curve_values:
        subdf = df_mean[df_mean[curve_column] == curve_val]
        if len(subdf) < 2:
            continue
            
        # Select X column based on fitting_type
        if fitting_type == "C":
            X_vals = subdf['C'].to_numpy(dtype=float)
            X_name = "C"
        elif fitting_type == "C_raw":
            X_vals = subdf['C_raw'].to_numpy(dtype=float)
            X_name = "C_raw"
        elif fitting_type == "step":
            X_vals = subdf['step'].to_numpy(dtype=float)
            X_name = "step"
        elif fitting_type == "E": 
            X_vals = subdf['E'].to_numpy(dtype=float)  # E represents D (data size)
            X_name = "D"
        elif fitting_type == "response_length":
            X_vals = subdf['response_length'].to_numpy(dtype=float)
            X_name = "response_length"
        else:
            raise ValueError(f"Invalid fitting type: {fitting_type}")
            
        ErrRate_vals = np.clip(subdf['ErrRate'].to_numpy(dtype=float), 1e-12, None)
        
        log_X = np.log10(X_vals)
        log_ErrRate = np.log10(ErrRate_vals)
        
        # Remove invalid values
        valid_mask = np.isfinite(log_X) & np.isfinite(log_ErrRate)
        log_X = log_X[valid_mask]
        log_ErrRate = log_ErrRate[valid_mask]
        
        if len(log_X) < 2:
            continue
        
        # Linear regression: log_ErrRate = -k * log_X + E0 (so k should be positive)
        reg = LinearRegression()
        reg.fit(log_X.reshape(-1, 1), log_ErrRate)
        k = -reg.coef_[0]  # Take negative to make k positive
        E0 = reg.intercept_
        r2 = reg.score(log_X.reshape(-1, 1), log_ErrRate)
        
        # Use display name for output
        display_X_name = display_names.get(X_name, X_name)
        
        fit_results[curve_val] = {
            'k': k, 'E0': E0, 'r2': r2, 'n_points': len(log_X),
            'fitting_type': fitting_type, 'X_name': X_name
        }
        
        # Format the curve value for display (model size in billions)
        curve_display = f"{curve_val/1e9:>4.1f}B"
        print(f"  N={curve_display}: k_{display_X_name}={k:>8.4f}, E0_{display_X_name}={E0:>8.4f}, R²={r2:>6.4f}")
    
    return fit_results

# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================
def main():
    """Main processing function"""
    global phi_global
    
    # Display name mapping
    display_names = {
        "C": "C",
        "C_raw": "C", 
        "step": "steps",
        "E": "D"
    }

    print("=== Multi-Eval Scaling Law Analysis - Base vs Instruct Comparison ===")
    print(f"Fitting type: L(N,{fitting_type}) fitting")
    print(f"Comparing models: {data_sources}")
    
    # Storage for results from both model types
    all_results = {}
    
    # Process each model type
    for data_source in data_sources:
        print(f"\n=== Processing {data_source} model ===")
        
        # Load and preprocess data
        df = data_proc.load_and_preprocess(config.CSV_MAP[data_source])
        print(f"Loaded {data_source} data with {len(df)} rows")
        
        # Get phi_global
        phi_global, _, _ = data_proc.estimate_phi_from_runs(
            df, 
            sample_size_per_step=config.SAMPLE_SIZE_PER_STEP, 
            tail_fraction=0.5
        )
        
        # Process data (same as before)
        df['ErrRate'] = 1 - df[eval_name]
        
        # 计算Improvement Rate：相对于step=0的改进率
        def calc_improvement_rate_for_group(group):
            step_0_rows = group[group['step'] == 0]
            if len(step_0_rows) == 0:
                raise ValueError(f"No step=0 found for {curve_column}={group[curve_column].iloc[0]}")
            baseline_score = step_0_rows[eval_name].iloc[0]
            group = group.copy()
            group['ImprovementRate'] = group[eval_name] / baseline_score
            return group
        
        df = df.groupby(curve_column, group_keys=False).apply(calc_improvement_rate_for_group).reset_index(drop=True)
        
        # 计算完 ImprovementRate 后，丢弃 step=0 的数据（因为 E=0 会导致 log10(E) = -inf）
        df = df[df['step'] > 0].reset_index(drop=True)
        
        # apply curve mask 
        if curve_mask is not None:
            df = df[df[curve_column].isin(curve_mask)]

        # 丢掉每个 (curve_column, runid) 的前 warmup_clip_num 个点
        if warmup_clip_num and warmup_clip_num > 0:
            df = (
                df.groupby([curve_column, 'runid'], as_index=False, group_keys=False)
                  .apply(lambda g: g.iloc[warmup_clip_num:])
                  .reset_index(drop=True)
            )
        
        print('-------', df.columns)
        # 对相同横坐标（同一 curve_column 与 step → 同一 E）聚合：只显示三个纵坐标（不同 run）的平均值
        df_mean = (
            df.groupby([curve_column, 'step'], as_index=False)
              .agg(N=('N', 'last'), Tau=('Tau', 'last'), C=('C', 'last'), C_raw=('C_raw', 'last'), E=('E', 'last'), ErrRate=('ErrRate', 'mean'), ImprovementRate=('ImprovementRate', 'mean'), response_length=('response_length', 'mean'))
        )
        
        # Perform fitting based on fitting_type
        fit_results = perform_fitting(df_mean, fitting_type, data_source, display_names, curve_column)
        all_results[data_source] = fit_results
    
    # Now create comparison plots
    create_comparison_plots(all_results, fitting_type, display_names, plot_enable, curve_column)

def create_comparison_plots(all_results, fitting_type, display_names, plot_enable, curve_column):
    """Create comparison plots for Base vs Instruct models"""
    
    # Extract data for plotting
    plot_data = {}
    
    for data_source, fit_results in all_results.items():
        curve_values = sorted(fit_results.keys())
        k_values = [fit_results[curve_val]['k'] for curve_val in curve_values]
        E0_values = [fit_results[curve_val]['E0'] for curve_val in curve_values]
        
        plot_data[data_source] = {
            'curve_values': curve_values,
            'k_values': k_values,
            'E0_values': E0_values
        }
    
    # Determine which plots to create based on plot_enable
    n_plots = len(plot_enable)
    if n_plots == 0:
        print("Warning: plot_enable is empty, no plots will be generated")
        return
    
    # Create subplots dynamically
    if n_plots == 1:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 4))
        if n_plots == 1:
            axes = [axes]
    
    ax_idx = 0
    
    # Get labels for display
    curve_label = config.DEFAULT_LABELS.get(curve_column, curve_column)
    curve_short_name = config.DEFAULT_SHORT_NAME.get(curve_column, curve_column)
    fitting_short_name = config.DEFAULT_SHORT_NAME.get(fitting_type, fitting_type)
    
    # Plot k(curve_column) comparison if enabled
    if "k" in plot_enable:
        ax = axes[ax_idx]
        
        # Plot data for each model type using plot_basic
        for data_source, data in plot_data.items():
            plot_basic(
                x=np.array(data['curve_values']),
                y=np.array(data['k_values']),
                use_scatter=True,
                scatter_alpha=1.0,
                scatter_s=100,
                scatter_marker=config.DEFAULT_MARKERS[data_source],
                color=config.COLOR_MAPPING[data_source],
                ax=ax
            )
            # Add manual legend entry since plot_basic doesn't handle labels
            ax.scatter([], [], color=config.COLOR_MAPPING[data_source], 
                      marker=config.DEFAULT_MARKERS[data_source], s=100,
                      label=config.DEFAULT_LABELS[data_source])
        
        # Apply settings using plot_basic_settings
        plot_basic_settings(
            ax=ax,
            x_scale='log',
            x_label=f'{curve_label} ({curve_short_name})',
            y_label=f'$k_{{{fitting_short_name}}}({curve_short_name})$',
            # title=title_k if title_k is not None else f'k({curve_short_name})',
            use_legend=True,
            # x_tick_spacing=0.1,
            x_tick_on_data=True
        )
        ax_idx += 1
    
    # Plot E0(curve_column) comparison if enabled
    if "E0" in plot_enable:
        ax = axes[ax_idx]
        
        # Plot data for each model type using plot_basic
        for data_source, data in plot_data.items():
            plot_basic(
                x=np.array(data['curve_values']),
                y=np.array(data['E0_values']),
                use_scatter=True,
                scatter_alpha=1.0,
                scatter_s=100,
                scatter_marker=config.DEFAULT_MARKERS[data_source],
                color=config.COLOR_MAPPING[data_source],
                ax=ax
            )
            # Add manual legend entry since plot_basic doesn't handle labels
            ax.scatter([], [], color=config.COLOR_MAPPING[data_source], 
                      marker=config.DEFAULT_MARKERS[data_source], s=100,
                      label=config.DEFAULT_LABELS[data_source])
        
        # Apply settings using plot_basic_settings
        plot_basic_settings(
            ax=ax,
            x_scale='log',
            x_label=f'{curve_label} ({curve_short_name})',
            y_label=f'$E_{{{fitting_short_name}}}({curve_short_name})$',
            # title=title_E0 if title_E0 is not None else f'E0({curve_label})',
            use_legend=True,
            x_tick_on_data=True,
            # x_tick_spacing=0.2
        )
        ax_idx += 1
    
    plt.tight_layout()
    
    # Save the plot with descriptive filename
    plot_suffix = "_".join(plot_enable)
    comparison_path = config.OUTPUT_BASE_DIR / f"base_vs_instruct_L-{curve_column}-{fitting_type}_fitting_comparison_{plot_suffix}.pdf"
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"\n=== Base vs Instruct Comparison Plot ===")
    print(f"Enabled plots: {plot_enable}")
    print(f"Comparison plot saved to: {comparison_path}")
    
    # Print summary statistics
    print(f"\n=== Summary Statistics ===")
    for data_source, data in plot_data.items():
        k_values = np.array(data['k_values'])
        E0_values = np.array(data['E0_values'])
        
        print(f"\n{config.DEFAULT_LABELS[data_source]}:")
        if "k" in plot_enable:
            print(f"  k({curve_label}) range: {k_values.min():.4f} to {k_values.max():.4f}")
            print(f"  k({curve_label}) variation: {k_values.std():.4f}")
        if "E0" in plot_enable:
            print(f"  E0({curve_label}) range: {E0_values.min():.4f} to {E0_values.max():.4f}")
            print(f"  E0({curve_label}) variation: {E0_values.std():.4f}")

if __name__ == "__main__":
    main()