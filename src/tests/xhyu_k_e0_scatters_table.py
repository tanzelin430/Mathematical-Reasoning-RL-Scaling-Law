#!/usr/bin/env python3
"""
Scaling Law Pipeline - LaTeX Table Generation
Processes multiple test evals from Experiment1 data and generates LaTeX tables for L(N,D) and L(N,C_raw) fitting results
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import itertools
import warnings

# Suppress pandas deprecation warnings for groupby operations
warnings.filterwarnings("ignore", message=".*DataFrameGroupBy.apply operated on the grouping columns.*")
    

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

# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================
def process_model_type(model_type):
    """Process a single model type and return fitting results"""
    print(f"Loading {model_type} model data...")
    
    # Load and preprocess data using the unified function
    df = data_proc.load_and_preprocess(config.CSV_MAP[model_type])
    print(f"Columns: {list(df.columns)}")
    
    # Get phi_global from the load_and_preprocess function
    phi_global, _, _ = data_proc.estimate_phi_from_runs(
        df, 
        sample_size_per_step=config.SAMPLE_SIZE_PER_STEP, 
        tail_fraction=0.5
    )
    # 计算Error Rate
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
    
    # 按数值 N 排序（确保大模型如 14B 排在最后）
    model_order = (
        df_mean[['model_size', 'N']]
        .drop_duplicates()
        .sort_values('N')
        ['model_size']
        .tolist()
    )
    unique_model_sizes = model_order

    # ===========================
    # 测试C和E拟合的k(N)和E0(N)关系
    # ===========================
    print(f"\n=== L(N,D)和L(N,C)拟合的k(N)和E0(N)关系 ===")
    
    # 对每个N分别拟合 log10(ErrRate) = -k * log10(X) + E0，其中X为D或C
    from sklearn.linear_model import LinearRegression
    
    fit_results_E = {}  # 拟合E的结果
    fit_results_C = {}  # 拟合C的结果
    
    for ms in unique_model_sizes:
        subdf = df_mean[df_mean['model_size'] == ms]
        if len(subdf) < 2:
            continue
            
        N_val = float(subdf['N'].iloc[0])
        E_vals = subdf['E'].to_numpy(dtype=float)
        C_vals = subdf['C_raw'].to_numpy(dtype=float)
        ErrRate_vals = np.clip(subdf['ErrRate'].to_numpy(dtype=float), 1e-12, None)
        
        log_E = np.log10(E_vals)
        log_C = np.log10(C_vals)
        log_ErrRate = np.log10(ErrRate_vals)
        
        # 移除无效值
        valid_mask = np.isfinite(log_E) & np.isfinite(log_C) & np.isfinite(log_ErrRate)
        log_E = log_E[valid_mask]
        log_C = log_C[valid_mask]
        log_ErrRate = log_ErrRate[valid_mask]
        
        if len(log_E) < 2:
            continue
        
        # 拟合D: log_ErrRate = -k_D * log_D + E0_D (所以k_D应该是正数)
        reg_E = LinearRegression()
        reg_E.fit(log_E.reshape(-1, 1), log_ErrRate)
        k_E = -reg_E.coef_[0]  # 取负号，使k为正数
        E0_E = reg_E.intercept_
        r2_E = reg_E.score(log_E.reshape(-1, 1), log_ErrRate)
        
        # 拟合C: log_ErrRate = -k_C * log_C + E0_C (所以k_C应该是正数)
        reg_C = LinearRegression()
        reg_C.fit(log_C.reshape(-1, 1), log_ErrRate)
        k_C = -reg_C.coef_[0]  # 取负号，使k为正数
        E0_C = reg_C.intercept_
        r2_C = reg_C.score(log_C.reshape(-1, 1), log_ErrRate)
        
        fit_results_E[N_val] = {'k': k_E, 'E0': E0_E, 'r2': r2_E, 'n_points': len(log_E), 'model_size': ms}
        fit_results_C[N_val] = {'k': k_C, 'E0': E0_C, 'r2': r2_C, 'n_points': len(log_C), 'model_size': ms}
        
        print(f"N={N_val/1e9:>4.1f}B: k_D={k_E:>8.4f}, E0_D={E0_E:>8.4f}, R²_D={r2_E:>6.4f}")
        print(f"N={N_val/1e9:>4.1f}B: k_C={k_C:>8.4f}, E0_C={E0_C:>8.4f}, R²_C={r2_C:>6.4f}")
        print(f"        k差异={k_C-k_E:>8.6f}, E0差异={E0_C-E0_E:>8.4f}")
        print()
    
    return fit_results_E, fit_results_C

def generate_latex_tables(results_base, results_instruct):
    """Generate LaTeX tables for fitting results using booktabs style (Version A only)"""
    
    def format_number(val, precision=4):
        """Format number for LaTeX table"""
        # Always use fixed-point notation, no scientific notation
        return f"{val:.{precision}f}"
    
    # Collect all model sizes from both base and instruct
    all_sizes = set()
    for res_e, res_c in [(results_base[0], results_base[1]), (results_instruct[0], results_instruct[1])]:
        all_sizes.update(res_e.keys())
        all_sizes.update(res_c.keys())
    all_sizes = sorted(all_sizes)
    
    # =========================
    # Table 1: L(N,D) fitting results
    # =========================
    
    print("\n" + "="*80)
    print("LaTeX Table 1: L(N,D) Fitting Results")
    print("="*80)
    
    print("\\begin{table}[H]")
    print("\\centering")
    print("\\caption{$L(N,D)$ Fitting Results: $\\log\\text{TestLoss} = -k_D \\log D + E_D$}")
    print("\\label{tab:fitting_results_LD}")
    print("\\begin{tabular}{llll}")
    print("\\toprule")
    print("\\textbf{Model} & \\textbf{$k_D$} & \\textbf{$E_D$} & \\textbf{$R_D^2$} \\\\")
    print("\\midrule")
    
    # First, print all Base models
    for N in all_sizes:
        # Format model size without .0 for integers
        N_val = N/1e9
        if N_val == int(N_val):
            N_str = f"{int(N_val)}B"
        else:
            N_str = f"{N_val:.1f}B"
        if N in results_base[0]:
            res = results_base[0][N]
            k_val = format_number(res['k'])
            e0_val = format_number(res['E0'])
            r2_val = format_number(res['r2'], 3)
            print(f"{N_str}-Base & {k_val} & {e0_val} & {r2_val} \\\\")
    
    # Then, print all Instruct models
    for N in all_sizes:
        # Format model size without .0 for integers
        N_val = N/1e9
        if N_val == int(N_val):
            N_str = f"{int(N_val)}B"
        else:
            N_str = f"{N_val:.1f}B"
        if N in results_instruct[0]:
            res = results_instruct[0][N]
            k_val = format_number(res['k'])
            e0_val = format_number(res['E0'])
            r2_val = format_number(res['r2'], 3)
            print(f"{N_str}-Instruct & {k_val} & {e0_val} & {r2_val} \\\\")
            
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    # =========================
    # Table 2: L(N,C_raw) fitting results
    # =========================
    
    print("\n" + "="*80)
    print("LaTeX Table 2: L(N,C_raw) Fitting Results")
    print("="*80)
    
    print("\\begin{table}[H]")
    print("\\centering")
    print("\\caption{$L(N,C)$ Fitting Results: $\\log\\text{TestLoss} = -k_C \\log C + E_C$}")
    print("\\label{tab:fitting_results_LC}")
    print("\\begin{tabular}{llll}")
    print("\\toprule")
    print("\\textbf{Model} & \\textbf{$k_C$} & \\textbf{$E_C$} & \\textbf{$R_C^2$} \\\\")
    print("\\midrule")
    
    # First, print all Base models
    for N in all_sizes:
        # Format model size without .0 for integers
        N_val = N/1e9
        if N_val == int(N_val):
            N_str = f"{int(N_val)}B"
        else:
            N_str = f"{N_val:.1f}B"
        if N in results_base[1]:
            res = results_base[1][N]
            k_val = format_number(res['k'])
            e0_val = format_number(res['E0'])
            r2_val = format_number(res['r2'], 3)
            print(f"{N_str}-Base & {k_val} & {e0_val} & {r2_val} \\\\")
    
    # Then, print all Instruct models
    for N in all_sizes:
        # Format model size without .0 for integers
        N_val = N/1e9
        if N_val == int(N_val):
            N_str = f"{int(N_val)}B"
        else:
            N_str = f"{N_val:.1f}B"
        if N in results_instruct[1]:
            res = results_instruct[1][N]
            k_val = format_number(res['k'])
            e0_val = format_number(res['E0'])
            r2_val = format_number(res['r2'], 3)
            print(f"{N_str}-Instruct & {k_val} & {e0_val} & {r2_val} \\\\")
            
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    # =========================
    # Table 3: L(N,D) fitting results - Compact Version (Base vs Instruct as columns)
    # =========================
    
    print("\n" + "="*80)
    print("LaTeX Table 3: L(N,D) Fitting Results - Compact Version")
    print("="*80)
    
    print("\\begin{table}[H]")
    print("\\centering")
    print("\\caption{$L(N,D)$ Fitting Results: $\\log\\text{TestLoss} = -k_D \\log D + E_D$}")
    print("\\label{tab:fitting_results_LD}")
    print("\\begin{tabular}{lcccccc}")
    print("\\toprule")
    print("\\multirow{2}{*}{\\textbf{Model Size}} & \\multicolumn{3}{c}{\\textbf{Base}} & \\multicolumn{3}{c}{\\textbf{Instruct}} \\\\")
    print("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}")
    print(" & $k_D$ & $E_D$ & $R_D^2$ & $k_D$ & $E_D$ & $R_D^2$ \\\\")
    print("\\midrule")
    
    for N in all_sizes:
        # Format model size without .0 for integers
        N_val = N/1e9
        if N_val == int(N_val):
            N_str = f"{int(N_val)}B"
        else:
            N_str = f"{N_val:.1f}B"
            
        # Get values for both base and instruct
        k_base = format_number(results_base[0][N]['k']) if N in results_base[0] else "---"
        k_instruct = format_number(results_instruct[0][N]['k']) if N in results_instruct[0] else "---"
        e0_base = format_number(results_base[0][N]['E0']) if N in results_base[0] else "---"
        e0_instruct = format_number(results_instruct[0][N]['E0']) if N in results_instruct[0] else "---"
        r2_base = format_number(results_base[0][N]['r2'], 3) if N in results_base[0] else "---"
        r2_instruct = format_number(results_instruct[0][N]['r2'], 3) if N in results_instruct[0] else "---"
        
        print(f"{N_str} & {k_base} & {e0_base} & {r2_base} & {k_instruct} & {e0_instruct} & {r2_instruct} \\\\")
        
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    # =========================
    # Table 4: L(N,C) fitting results - Compact Version (Base vs Instruct as columns)
    # =========================
    
    print("\n" + "="*80)
    print("LaTeX Table 4: L(N,C) Fitting Results - Compact Version")
    print("="*80)
    
    print("\\begin{table}[H]")
    print("\\centering")
    print("\\caption{$L(N,C)$ Fitting Results: $\\log\\text{TestLoss} = -k_C \\log C + E_C$}")
    print("\\label{tab:fitting_results_LC}")
    print("\\begin{tabular}{lcccccc}")
    print("\\toprule")
    print("\\multirow{2}{*}{\\textbf{Model Size}} & \\multicolumn{3}{c}{\\textbf{Base}} & \\multicolumn{3}{c}{\\textbf{Instruct}} \\\\")
    print("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}")
    print(" & $k_C$ & $E_C$ & $R_C^2$ & $k_C$ & $E_C$ & $R_C^2$ \\\\")
    print("\\midrule")
    
    for N in all_sizes:
        # Format model size without .0 for integers
        N_val = N/1e9
        if N_val == int(N_val):
            N_str = f"{int(N_val)}B"
        else:
            N_str = f"{N_val:.1f}B"
            
        # Get values for both base and instruct
        k_base = format_number(results_base[1][N]['k']) if N in results_base[1] else "---"
        k_instruct = format_number(results_instruct[1][N]['k']) if N in results_instruct[1] else "---"
        e0_base = format_number(results_base[1][N]['E0']) if N in results_base[1] else "---"
        e0_instruct = format_number(results_instruct[1][N]['E0']) if N in results_instruct[1] else "---"
        r2_base = format_number(results_base[1][N]['r2'], 3) if N in results_base[1] else "---"
        r2_instruct = format_number(results_instruct[1][N]['r2'], 3) if N in results_instruct[1] else "---"
        
        print(f"{N_str} & {k_base} & {e0_base} & {r2_base} & {k_instruct} & {e0_instruct} & {r2_instruct} \\\\")
        
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    # =========================
    # Additional note for users
    # =========================
    print("\n" + "="*80)
    print("Note: These tables use the booktabs package for professional styling.")
    print("Make sure to include \\usepackage{booktabs} and \\usepackage{multirow} in your LaTeX preamble.")
    print("Also include \\usepackage{float} for the [H] table positioning option.")
    print("="*80)

def main():
    """Main processing function"""
    print("=== Multi-Eval Scaling Law Analysis ===")
    
    # Process both base and instruct models
    print("\n--- Processing Base Models ---")
    results_base = process_model_type("base")
    
    print("\n--- Processing Instruct Models ---")
    results_instruct = process_model_type("instruct")
    
    # Generate LaTeX tables
    generate_latex_tables(results_base, results_instruct)

if __name__ == "__main__":
    main()