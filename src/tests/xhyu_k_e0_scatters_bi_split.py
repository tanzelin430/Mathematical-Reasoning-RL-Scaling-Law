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
    

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.common import intrinsic
from src.common import data_proc
from src.common import plot
from src.common import config

# =============================================================================
# CONFIGURATION - Use config constants
# =============================================================================

# Use holdout score evaluation
eval_name = config.DEFAULT_TEST_EVAL
warmup_clip_num = 5  # Local setting for this script
phi_global = 1.0

def perform_fitting(df_mean, fitting_type, model_type, display_names):
    """
    Perform fitting based on fitting_type
    """
    from sklearn.linear_model import LinearRegression
    
    display_name = display_names.get(fitting_type, fitting_type)
    print(f"  Performing L(N,{display_name}) fitting for {model_type} model")
    
    # Get unique model sizes
    unique_model_sizes = sorted(df_mean['N'].unique())
    fit_results = {}
    
    for N_val in unique_model_sizes:
        subdf = df_mean[df_mean['N'] == N_val]
        if len(subdf) < 2:
            continue
            
        # Select X column based on fitting_type
        if fitting_type == "C":
            X_vals = subdf['C'].to_numpy(dtype=float)
            X_name = "C"
        elif fitting_type == "C_raw":
            X_vals = subdf['C_raw'].to_numpy(dtype=float)
            X_name = "C_raw"
        else: 
            X_vals = subdf['E'].to_numpy(dtype=float)  # E represents D (data size)
            X_name = "D"
            
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
        
        fit_results[N_val] = {
            'k': k, 'E0': E0, 'r2': r2, 'n_points': len(log_X),
            'fitting_type': fitting_type, 'X_name': X_name
        }
        
        print(f"  N={N_val/1e9:>4.1f}B: k_{display_X_name}={k:>8.4f}, E0_{display_X_name}={E0:>8.4f}, R²={r2:>6.4f}")
    
    return fit_results

# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================
def main():
    """Main processing function"""
    global phi_global
    
    # Configuration
    fitting_type = "C_raw"  # "C" for L(N,C) fitting, "C_raw" for L(N,C_raw) fitting, "D" for L(N,D) fitting
    model_types = ["base", "instruct"]  # Both model types to compare
    
    # Display name mapping
    display_names = {
        "C": "C",
        "C_raw": "C", 
        "E": "D"
    }

    print("=== Multi-Eval Scaling Law Analysis - Base vs Instruct Comparison ===")
    print(f"Fitting type: L(N,{fitting_type}) fitting")
    print(f"Comparing models: {model_types}")
    
    # Storage for results from both model types
    all_results = {}
    
    # Process each model type
    for model_type in model_types:
        print(f"\n=== Processing {model_type} model ===")
        
        # Load and preprocess data
        df = data_proc.load_and_preprocess(config.CSV_MAP[model_type])
        print(f"Loaded {model_type} data with {len(df)} rows")
        
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
                raise ValueError(f"No step=0 found for model_size={group['model_size'].iloc[0]}")
            baseline_score = step_0_rows[eval_name].iloc[0]
            group = group.copy()
            group['ImprovementRate'] = group[eval_name] / baseline_score
            return group
        
        df = df.groupby('model_size', group_keys=False).apply(calc_improvement_rate_for_group).reset_index(drop=True)
        
        # 计算完 ImprovementRate 后，丢弃 step=0 的数据（因为 E=0 会导致 log10(E) = -inf）
        df = df[df['step'] > 0].reset_index(drop=True)
        
        # 丢掉每个 (model_size, runid) 的前 warmup_clip_num 个点
        if warmup_clip_num and warmup_clip_num > 0:
            df = (
                df.groupby(['model_size', 'runid'], as_index=False, group_keys=False)
                  .apply(lambda g: g.iloc[warmup_clip_num:])
                  .reset_index(drop=True)
            )
        
        # 对相同横坐标（同一 model_size 与 step → 同一 E）聚合：只显示三个纵坐标（不同 run）的平均值
        df_mean = (
            df.groupby(['model_size', 'step'], as_index=False)
              .agg(N=('N', 'first'), C=('C', 'first'), C_raw=('C_raw', 'first'), E=('E', 'first'), ErrRate=('ErrRate', 'mean'), ImprovementRate=('ImprovementRate', 'mean'))
        )
        
        # Perform fitting based on fitting_type
        fit_results = perform_fitting(df_mean, fitting_type, model_type, display_names)
        all_results[model_type] = fit_results
    
    # Now create comparison plots
    create_comparison_plots(all_results, fitting_type, display_names)

def create_comparison_plots(all_results, fitting_type, display_names):
    """Create comparison plots for Base vs Instruct models"""
    
    # Extract data for plotting
    plot_data = {}
    
    for model_type, fit_results in all_results.items():
        N_values = sorted(fit_results.keys())
        k_values = [fit_results[N]['k'] for N in N_values]
        E0_values = [fit_results[N]['E0'] for N in N_values]
        
        plot_data[model_type] = {
            'N_values': N_values,
            'k_values': k_values,
            'E0_values': E0_values
        }
    
    # Define colors and markers for different model types (使用 config 中的统一配色)
    colors = {'base': config.COLOR_MAPPING['base'], 'instruct': config.COLOR_MAPPING['instruct']}
    markers = {'base': 'o', 'instruct': 's'}
    labels = {'base': 'Base Model', 'instruct': 'Instruct Model'}
    
    # =============================================================================
    # Figure 1: k(N) comparison (Base vs Instruct)
    # =============================================================================
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: k(N) Base
    if 'base' in plot_data:
        data = plot_data['base']
        ax1.scatter(data['N_values'], data['k_values'], 
                   color=colors['base'], marker=markers['base'], 
                   s=100, alpha=0.7)
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Model Size N')
    ax1.set_ylabel(f'k(N)')
    ax1.set_title(f'k(N) of L(N,{display_names[fitting_type]}) on Base model')
    
    # Right: k(N) Instruct
    if 'instruct' in plot_data:
        data = plot_data['instruct']
        ax2.scatter(data['N_values'], data['k_values'], 
                   color=colors['instruct'], marker=markers['instruct'], 
                   s=100, alpha=0.7)
    
    ax2.set_xscale('log')
    ax2.set_xlabel('Model Size N')
    ax2.set_ylabel(f'k(N)')
    ax2.set_title(f'k(N) of L(N,{display_names[fitting_type]}) on Instruct model')
    
    plt.tight_layout()
    
    # Save k(N) plot
    k_comparison_path = config.OUTPUT_BASE_DIR / f"comparison_k_N_{fitting_type}.pdf"
    plt.savefig(k_comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # =============================================================================
    # Figure 2: E0(N) comparison (Base vs Instruct)
    # =============================================================================
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: E0(N) Base
    if 'base' in plot_data:
        data = plot_data['base']
        ax3.scatter(data['N_values'], data['E0_values'], 
                   color=colors['base'], marker=markers['base'], 
                   s=100, alpha=0.7)
    
    ax3.set_xscale('log')
    ax3.set_xlabel('Model Size N')
    ax3.set_ylabel(f'E0(N)')
    ax3.set_title(f'E0(N) of L(N,{display_names[fitting_type]}) on Base model')
    
    # Right: E0(N) Instruct
    if 'instruct' in plot_data:
        data = plot_data['instruct']
        ax4.scatter(data['N_values'], data['E0_values'], 
                   color=colors['instruct'], marker=markers['instruct'], 
                   s=100, alpha=0.7)
    
    ax4.set_xscale('log')
    ax4.set_xlabel('Model Size N')
    ax4.set_ylabel(f'E0(N)')
    ax4.set_title(f'E0(N) of L(N,{display_names[fitting_type]}) on Instruct model')
    
    plt.tight_layout()
    
    # Save E0(N) plot
    E0_comparison_path = config.OUTPUT_BASE_DIR / f"comparison_E0_N_{fitting_type}.pdf"
    plt.savefig(E0_comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n=== Base vs Instruct Comparison Plots ===")
    print(f"k(N) comparison plot saved to: {k_comparison_path}")
    print(f"E0(N) comparison plot saved to: {E0_comparison_path}")
    
    # Print summary statistics
    print(f"\n=== Summary Statistics ===")
    for model_type, data in plot_data.items():
        k_values = np.array(data['k_values'])
        E0_values = np.array(data['E0_values'])
        
        print(f"\n{labels[model_type]}:")
        print(f"  k(N) range: {k_values.min():.4f} to {k_values.max():.4f}")
        print(f"  E0(N) range: {E0_values.min():.4f} to {E0_values.max():.4f}")
        print(f"  k(N) variation: {k_values.std():.4f}")
        print(f"  E0(N) variation: {E0_values.std():.4f}")

if __name__ == "__main__":
    main()