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

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import intrinsic
import data_proc
import plot

# =============================================================================
# CONFIGURATION - Edit these variables to customize the run
# =============================================================================

# Data source configuration - use absolute paths based on script location
SCRIPT_DIR = Path(__file__).parent
OUTPUT_BASE_DIR = SCRIPT_DIR / "outputs"  # Base output directory for PNG plots
SAMPLE_SIZE_PER_STEP = 512
BUILD_I_ON_SMOOTHED = True
WARMUP_CLIPPING_NUM = 5 


PLOT_BASIC_CURVES = False # True for Intrinsic Curves

HOLDOUT=True
if HOLDOUT:
    # Test evals to process (from the CSV columns)
    TEST_EVALS = [
        'holdout_score',
    ]
    FIGURE_PREFIX = 'holdout'
    FIGURE_COLUMNS = 1 # note: if total > figure_columns, [row, col] -> [i]
    FIGURE_SIZE=(5, 5)
else:
    # Test evals to process (from the CSV columns)
    TEST_EVALS = [
        'overall_pass1', 
        'val/test_score/openai/gsm8k',
        'val/test_score/codegen__humaneval',
        'val/test_score/stem__supergpqa',
        'val/test_score/math__math',
        'val/test_score/logic__zebra_puzzle_dataset',
        'val/test_score/aimeamc2023',
        'val/test_score/aime2024',
        # 'holdout_score',
        # 'val/test_score/math__deepscaler_preview',
        # 'val/test_score/math__merged_deduped_dapo_or1_dataset',
    ]
    # FIGURE_PREFIX = 'holdout'
    FIGURE_PREFIX = 'all'
    FIGURE_COLUMNS = 2 # note: if total > figure_columns, [row, col] -> [i]
    FIGURE_SIZE=(10, 10)


total_evals = len(TEST_EVALS)
phi_global = 1.0

DEBUG = False  # Set to False to disable data statistics printing

# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================
def main():
    """Main processing function"""
    global phi_global
    

    csv_exp1_run0 = SCRIPT_DIR / "csv" / "scaling_law_data_experiment1_instruct_run0.csv" 
    csv_exp1_run1 = SCRIPT_DIR / "csv" / "scaling_law_data_experiment1_instruct_run1.csv" 
    csv_exp1_run2 = SCRIPT_DIR / "csv" / "scaling_law_data_experiment1_instruct_run2.csv" 

    print("=== Multi-Eval Scaling Law Analysis ===")
    print(f"Processing CSV: {csv_exp1_run0} and {csv_exp1_run1}")
    print(f"Output directory: {OUTPUT_BASE_DIR}")
    print(f"Evals to process: {len(TEST_EVALS)}")
    
    # Load data
    if not csv_exp1_run0.exists():
        print(f"❌ CSV file not found: {csv_exp1_run0}")
        return
    if not csv_exp1_run1.exists():
        print(f"❌ CSV file not found: {csv_exp1_run1}")
        return
    
    df_run0 = pd.read_csv(csv_exp1_run0)
    df_run1 = pd.read_csv(csv_exp1_run1)
    df_run2 = pd.read_csv(csv_exp1_run2)

    df = pd.concat([df_run0, df_run1, df_run2], ignore_index=True)
    # df = df_run2
    print (df.columns)
    
    # Sort, Inspect, Validate and Normalize data
    df = df.sort_values(['model_size','runid','step']).reset_index(drop=True)
    # data_proc.inspect_data(df)
    data_proc.validate_data(df, eval_columns=TEST_EVALS)
    df = data_proc.rename_columns(df)
    # Calculate E = step * sample_size_per_step
    df['E'] = df['step'] * float(SAMPLE_SIZE_PER_STEP)

    # Estimate global efficiency parameter phi
    phi_global, phi_by_N, phi_stats_df = data_proc.estimate_phi_from_runs(
        df, 
        sample_size_per_step=SAMPLE_SIZE_PER_STEP, 
        tail_fraction=0.5
    )
    # print(f"phi (global tail median) = {phi_global}")
    
    # Recalculate C using the estimated phi_global
    df['C'] = df['N'] * df['E'] * phi_global
    
    # # Print data statistics if enabled
    # if DEBUG:
    #     print("\n" + "="*50)
    #     print("Raw data statistics:")
    #     data_proc.print_data_statistics(df_merged)
    
    # ===========================
    # 只画一个散点图：横轴为Compute（C），纵轴为Error Rate，不同模型大小用不同颜色
    # ===========================
    import matplotlib.pyplot as plt

    # 只处理第一个eval
    eval_name = TEST_EVALS[0]
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
    
    # 丢掉每个 (model_size, runid) 的前 WARMUP_CLIPPING_NUM 个点
    if WARMUP_CLIPPING_NUM and WARMUP_CLIPPING_NUM > 0:
        df = (
            df.groupby(['model_size', 'runid'], as_index=False, group_keys=False)
              .apply(lambda g: g.iloc[WARMUP_CLIPPING_NUM:])
              .reset_index(drop=True)
        )
    
    # 对相同横坐标（同一 model_size 与 step → 同一 E）聚合：只显示三个纵坐标（不同 run）的平均值
    df_mean = (
        df.groupby(['model_size', 'step'], as_index=False)
          .agg(N=('N', 'first'), C=('C', 'first'), E=('E', 'first'), ErrRate=('ErrRate', 'mean'), ImprovementRate=('ImprovementRate', 'mean'))
    )
    # 颜色映射
    color_map = {
        0.5e9: '#1f77b4',
        1.5e9: '#ff7f0e',
        3e9: '#d62728',
        7e9: '#2ca02c',
        14e9: '#9467bd',
    }
    # 如果有更多模型大小，按数值 N 排序（确保大模型如 14B 排在最后）
    model_order = (
        df_mean[['model_size', 'N']]
        .drop_duplicates()
        .sort_values('N')
        ['model_size']
        .tolist()
    )
    unique_model_sizes = model_order
    import itertools
    import matplotlib
    color_cycle = itertools.cycle(matplotlib.colormaps['tab10'].colors)
    for ms in unique_model_sizes:
        if ms not in color_map:
            color_map[ms] = next(color_cycle)

    plt.figure(figsize=(7, 5))
    for ms in unique_model_sizes:
        subdf = df_mean[df_mean['model_size'] == ms]
        plt.scatter(
            subdf['E'], 
            subdf['ErrRate'], 
            # 直接用小写b显示（如1.5b），不区分大小
            label=f"{ms}",
            color=color_map[ms], 
            alpha=0.7, 
            s=12
        )
    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel("Training Examples E (log)")
    plt.ylabel("Error Rate")
    plt.title("Error Rate vs Training Examples")
    plt.legend(title="model size", loc="best")
    plt.tight_layout()
    plt.savefig(OUTPUT_BASE_DIR / f"{FIGURE_PREFIX}_scatter_errrate_vs_E.pdf", dpi=300, bbox_inches='tight')
    print(f"散点图已保存到 {OUTPUT_BASE_DIR / f'{FIGURE_PREFIX}_scatter_errrate_vs_E.pdf'}")

    # ===========================
    # 新增：ImprovementRate vs Training Examples 散点图
    # ===========================
    plt.figure(figsize=(7, 5))
    for ms in unique_model_sizes:
        subdf = df_mean[df_mean['model_size'] == ms]
        plt.scatter(
            subdf['E'], 
            -np.log(subdf['ImprovementRate']), 
            label=f"{ms}",
            color=color_map[ms], 
            alpha=0.7, 
            s=12
        )
    plt.xscale('log')
    # 注意：y轴不再是log scale，因为我们已经取了-log

    plt.xlabel("Training Examples E (log)")
    plt.ylabel("-log(Improvement Rate)")
    plt.title("-log(Improvement Rate) vs Training Examples")
    plt.legend(title="model size", loc="best")
    plt.tight_layout()
    plt.savefig(OUTPUT_BASE_DIR / f"{FIGURE_PREFIX}_scatter_improvementrate_vs_E.pdf", dpi=300, bbox_inches='tight')
    print(f"改进率散点图已保存到 {OUTPUT_BASE_DIR / f'{FIGURE_PREFIX}_scatter_improvementrate_vs_E.pdf'}")

    # ===========================
    # 拟合模型：对数线性模型
    # 变量命名：ErrRate=Error Rate, N=model_size, E=Training Examples
    # ===========================
    N_all = df_mean['N'].to_numpy(dtype=float)
    E_all = df_mean['E'].to_numpy(dtype=float)  # Training Examples (x-axis)
    ErrRate_all = df_mean['ErrRate'].to_numpy(dtype=float)  # Error Rate (y-axis)

    # 安全处理：避免 log(0)
    eps = 1e-12

    # ---------------------------
    # 模型 0（最简单）：对每个 N，拟合 log10(ErrRate) = -k * log10(E) + E0
    # 保存每个 N 的 k、E0、R2_log10，并单独绘图（保留原输出）
    # ---------------------------
    print("\n=== 模型0：按 N 的对数线性拟合 log10(ErrRate) = -k * log10(E) + E0 ===")
    model0_stats = []  # 列表元素: dict(N, k, E0, r2_log, m, b)
    plt.figure(figsize=(7, 5))
    # 先画散点（模型0：使用 x=log10(E)，y=log10(ErrRate)-log10(ErrRate0) 使起点为0）
    for ms in unique_model_sizes:
        subdf = df_mean[df_mean['model_size'] == ms].sort_values('E')
        E_vals = subdf['E'].to_numpy(dtype=float)
        ErrRate_vals = np.clip(subdf['ErrRate'].to_numpy(dtype=float), 1e-12, None)
        x = np.log10(E_vals)
        y = np.log10(ErrRate_vals)
        y0 = y[0]
        plt.scatter(
            x, y - y0, label=f"{ms}",
            color=color_map[ms], alpha=0.6, s=12
        )
    # 拟合与画线
    for ms in unique_model_sizes:
        subdf = df_mean[df_mean['model_size'] == ms]
        if len(subdf) < 2:
            continue
        E_vals = subdf['E'].to_numpy(dtype=float)
        ErrRate_vals = np.clip(subdf['ErrRate'].to_numpy(dtype=float), 1e-12, None)
        x = np.log10(E_vals)
        y = np.log10(ErrRate_vals)
        y0 = y[0]
        # 线性回归：y = m x + b  =>  k = -m, E0 = b
        m, b = np.polyfit(x, y, deg=1)
        y_pred = m * x + b
        # R2 in log space
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2_log = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        k = -m
        N_val = float(subdf['N'].iloc[0])
        model0_stats.append({
            'N': N_val, 'k': k, 'E0': b, 'r2_log': r2_log, 'm': m, 'b': b,
            'E_min': E_vals.min(), 'E_max': E_vals.max(), 'y0': y0
        })
        print(f"N={N_val:.3g}: k={k:.4f}, E0={b:.4f}, R2(log10 ErrRate)={r2_log:.4f}")
        # 画拟合线
        E_grid = np.logspace(np.log10(E_vals.min()), np.log10(E_vals.max()), 200)
        x_grid = np.log10(E_grid)
        y_grid = m * x_grid + b
        plt.plot(x_grid, y_grid - y0, color=color_map[ms], linewidth=2)

    # 线性坐标：x=log10(E), y=Δlog10(ErrRate)
    plt.xlabel(r"$\log_{10}E$")
    plt.ylabel(r"$\Delta\log_{10}ErrRate$")
    plt.title(r"$\log_{10}ErrRate = -k(N)\,\log_{10}E + e_0(N)$")
    plt.legend(title="model size", loc="best")
    plt.tight_layout()
    out_path0 = OUTPUT_BASE_DIR / f"{FIGURE_PREFIX}_fit_model0.pdf"
    plt.savefig(out_path0, dpi=300, bbox_inches='tight')
    print(f"拟合曲线图已保存到 {out_path0}")

    # ===========================
    # 新增：model0 的 k 和 E0 参数随 N 变化的散点图
    # ===========================
    if model0_stats:
        # 提取 N, k, E0 数据
        N_vals = [s['N'] for s in model0_stats]
        k_vals = [s['k'] for s in model0_stats]
        E0_vals = [s['E0'] for s in model0_stats]
        
        # 图1：k vs N 
        plt.figure(figsize=(7, 5))
        plt.scatter(N_vals, k_vals, color='red', alpha=0.7, s=50, marker='o')
        # plt.xscale('log')
        plt.xlabel('Model Size N (parameters)')
        plt.ylabel('k (slope parameter)')
        plt.title('Model0 Parameter k vs Model Size N')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path_k = OUTPUT_BASE_DIR / f"{FIGURE_PREFIX}_model0_k_vs_N.pdf"
        plt.savefig(out_path_k, dpi=300, bbox_inches='tight')
        print(f"k参数散点图已保存到 {out_path_k}")
        
        # 图2：E0 vs N
        plt.figure(figsize=(7, 5))
        plt.scatter(N_vals, E0_vals, color='blue', alpha=0.7, s=50, marker='s')
        # plt.xscale('log')
        plt.xlabel('Model Size N (parameters)')
        plt.ylabel('E0 (intercept parameter)')
        plt.title('Model0 Parameter E0 vs Model Size N')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path_e0 = OUTPUT_BASE_DIR / f"{FIGURE_PREFIX}_model0_E0_vs_N.pdf"
        plt.savefig(out_path_e0, dpi=300, bbox_inches='tight')
        print(f"E0参数散点图已保存到 {out_path_e0}")

        # # ===========================
        # # 分析 k = f(N) 的函数形式
        # # ===========================
        # 幂律函数 k = 1.01e-08 * N^0.664, R² = 0.8870
        # 对数函数 k = 1.56e-02 * log(N) + -0.3145, R² = 0.8545
        # 线性函数 k = 3.65e-12 * N + 0.0061, R² = 0.8382
        # 指数函数 k = 1.34e-02 * exp(1.01e-10 * N), R² = 0.7026
        
        # print(f"\n=== 分析 k = f(N) 的函数形式 ===")
        # N_vals_arr = np.array(N_vals)
        # k_vals_arr = np.array(k_vals)
        
        # # 尝试不同的函数形式
        # from scipy.optimize import curve_fit
        
        # # 1. 幂律函数: k = a * N^b
        # def power_law(N, a, b):
        #     return a * np.power(N, b)
        
        # try:
        #     popt_power, _ = curve_fit(power_law, N_vals_arr, k_vals_arr, p0=[1e-10, 0.5])
        #     k_pred_power = power_law(N_vals_arr, *popt_power)
        #     r2_power = 1 - np.sum((k_vals_arr - k_pred_power)**2) / np.sum((k_vals_arr - np.mean(k_vals_arr))**2)
        #     print(f"幂律函数 k = {popt_power[0]:.2e} * N^{popt_power[1]:.3f}, R² = {r2_power:.4f}")
        # except Exception as e:
        #     print(f"幂律函数拟合失败: {e}")
        #     r2_power = -1
        
        # # 2. 对数函数: k = a * log(N) + b
        # def log_func(N, a, b):
        #     return a * np.log(N) + b
        
        # try:
        #     popt_log, _ = curve_fit(log_func, N_vals_arr, k_vals_arr)
        #     k_pred_log = log_func(N_vals_arr, *popt_log)
        #     r2_log = 1 - np.sum((k_vals_arr - k_pred_log)**2) / np.sum((k_vals_arr - np.mean(k_vals_arr))**2)
        #     print(f"对数函数 k = {popt_log[0]:.2e} * log(N) + {popt_log[1]:.4f}, R² = {r2_log:.4f}")
        # except Exception as e:
        #     print(f"对数函数拟合失败: {e}")
        #     r2_log = -1
        
        # # 3. 线性函数: k = a * N + b
        # def linear_func(N, a, b):
        #     return a * N + b
        
        # try:
        #     popt_linear, _ = curve_fit(linear_func, N_vals_arr, k_vals_arr)
        #     k_pred_linear = linear_func(N_vals_arr, *popt_linear)
        #     r2_linear = 1 - np.sum((k_vals_arr - k_pred_linear)**2) / np.sum((k_vals_arr - np.mean(k_vals_arr))**2)
        #     print(f"线性函数 k = {popt_linear[0]:.2e} * N + {popt_linear[1]:.4f}, R² = {r2_linear:.4f}")
        # except Exception as e:
        #     print(f"线性函数拟合失败: {e}")
        #     r2_linear = -1
        
        # # 4. 指数函数: k = a * exp(b * N)
        # def exp_func(N, a, b):
        #     return a * np.exp(b * N)
        
        # try:
        #     popt_exp, _ = curve_fit(exp_func, N_vals_arr, k_vals_arr, p0=[0.001, 1e-10])
        #     k_pred_exp = exp_func(N_vals_arr, *popt_exp)
        #     r2_exp = 1 - np.sum((k_vals_arr - k_pred_exp)**2) / np.sum((k_vals_arr - np.mean(k_vals_arr))**2)
        #     print(f"指数函数 k = {popt_exp[0]:.2e} * exp({popt_exp[1]:.2e} * N), R² = {r2_exp:.4f}")
        # except Exception as e:
        #     print(f"指数函数拟合失败: {e}")
        #     r2_exp = -1
        
        # # 找出最佳拟合
        # r2_values = {'幂律': r2_power, '对数': r2_log, '线性': r2_linear, '指数': r2_exp}
        # valid_r2 = {k: v for k, v in r2_values.items() if v > -1}
        # if valid_r2:
        #     best_func = max(valid_r2, key=valid_r2.get)
        #     print(f"\n最佳拟合函数形式: {best_func} (R² = {valid_r2[best_func]:.4f})")
        
        # # 绘制比较图
        # plt.figure(figsize=(10, 6))
        # plt.scatter(N_vals_arr, k_vals_arr, color='black', s=50, label='data points', zorder=5)
        
        # N_smooth = np.logspace(np.log10(N_vals_arr.min()), np.log10(N_vals_arr.max()), 100)
        
        # if r2_power > 0:
        #     k_smooth_power = power_law(N_smooth, *popt_power)
        #     plt.plot(N_smooth, k_smooth_power, '--', label=f'pow (R²={r2_power:.3f})', linewidth=2)
        
        # if r2_log > 0:
        #     k_smooth_log = log_func(N_smooth, *popt_log)
        #     plt.plot(N_smooth, k_smooth_log, '-', label=f'log (R²={r2_log:.3f})', linewidth=2)
        
        # if r2_linear > 0:
        #     k_smooth_linear = linear_func(N_smooth, *popt_linear)
        #     plt.plot(N_smooth, k_smooth_linear, ':', label=f'linear (R²={r2_linear:.3f})', linewidth=2)
        
        # # plt.xscale('log')
        # plt.xlabel('Model Size N (parameters)')
        # plt.ylabel('k parameter')
        # plt.title('k = f(N) Function Form Analysis')
        # plt.legend()
        # plt.grid(True, alpha=0.3)
        # plt.tight_layout()
        # out_path_analysis = OUTPUT_BASE_DIR / f"{FIGURE_PREFIX}_model0_k_function_analysis.pdf"
        # plt.savefig(out_path_analysis, dpi=300, bbox_inches='tight')
        # print(f"函数形式分析图已保存到 {out_path_analysis}")

    print(f"\n=== ErrRate 模型拟合完成 ===")
    print("模型：对数线性拟合 log10(ErrRate) = -k * log10(E) + E0")
    if model0_stats:
        k_mean = np.mean([s['k'] for s in model0_stats])
        r2_mean = np.mean([s['r2_log'] for s in model0_stats])
        print(f"平均 k ≈ {k_mean:.3f}")
        print(f"平均 R2(log10 ErrRate) ≈ {r2_mean:.3f}")

    return
    # ---------------------------
    # ImprovementRate 模型拟合
    # ---------------------------
    print("\n=== 模型1：新公式拟合 -log(ImprovementRate) = (N/N_c)^α_N * D_c/(D+D_0) ===")
    
    # 定义新的拟合模型
    def new_scaling_model(params, N, D):
        N_c, alpha_N, D_c, D_0 = params
        # 避免除零和负数
        N = np.maximum(N, 1e-10)
        D = np.maximum(D, 1e-10)
        N_c = max(N_c, 1e-10)
        D_c = max(D_c, 1e-10)
        D_0 = max(D_0, 0)  # D_0 可以为0但不能为负
        
        term1 = np.power(N / N_c, alpha_N)  # (N/N_c)^α_N
        term2 = D_c / (D + D_0)  # D_c/(D+D_0)
        return term1 * term2

    # 准备数据
    N_all_improve = df_mean['N'].to_numpy(dtype=float)
    D_all_improve = df_mean['E'].to_numpy(dtype=float)  # D = E (Training Examples)
    y_all_improve = -np.log(np.clip(df_mean['ImprovementRate'].to_numpy(dtype=float), 1e-12, None))

    # 使用scipy进行非线性拟合
    from scipy.optimize import curve_fit
    
    # 检查数据范围并设置合理的初始参数估计
    N_min, N_max = N_all_improve.min(), N_all_improve.max()
    D_min, D_max = D_all_improve.min(), D_all_improve.max()
    y_min, y_max = y_all_improve.min(), y_all_improve.max()
    
    # 基于数据范围和物理意义设置合理的初始参数
    print(f"数据范围: N=[{N_min:.2g}, {N_max:.2g}], D=[{D_min:.2g}, {D_max:.2g}], y=[{y_min:.3f}, {y_max:.3f}]")
    
    # 初始参数估计
    N_c_init = N_max * 5      # N_c 应该大于最大模型大小，作为特征尺度
    alpha_N_init = 0.2        # 模型大小的scaling指数，通常在0.1-0.5范围
    D_c_init = D_max * 2      # D_c 应该在数据量的合理范围内
    D_0_init = D_min * 0.5    # D_0 作为小的基础偏移量
    
    p0 = [N_c_init, alpha_N_init, D_c_init, D_0_init]
    
    # 边界设置 - 扩大范围给优化更多空间
    # N_c: 模型大小特征尺度，大范围探索
    # alpha_N: scaling指数，扩大到更宽的范围
    # D_c: 数据特征尺度，大范围探索
    # D_0: 基础偏移，允许更大的范围
    bounds = (
        [N_min * 0.1, 0.001, D_min * 0.1, 0],           # 下界
        [N_max * 1000, 10.0, D_max * 1000, D_max * 100]  # 上界
    )
    
    print(f"初始参数: N_c={N_c_init:.2g}, α_N={alpha_N_init:.3f}, D_c={D_c_init:.2g}, D_0={D_0_init:.2g}")
    print(f"参数边界: N_c=[{bounds[0][0]:.1g}, {bounds[1][0]:.1g}], α_N=[{bounds[0][1]:.2f}, {bounds[1][1]:.2f}]")
    print(f"         D_c=[{bounds[0][2]:.1g}, {bounds[1][2]:.1g}], D_0=[{bounds[0][3]:.1g}, {bounds[1][3]:.1g}]")
    
    try:
        popt, pcov = curve_fit(
            lambda data, N_c, alpha_N, D_c, D_0: new_scaling_model([N_c, alpha_N, D_c, D_0], data[0], data[1]),
            (N_all_improve, D_all_improve), 
            y_all_improve, 
            p0=p0, 
            bounds=bounds, 
            maxfev=50000,  # 增加最大迭代次数
            method='trf'   # 使用Trust Region Reflective算法，对边界约束更友好
        )
        
        N_c_fit, alpha_N_fit, D_c_fit, D_0_fit = popt
        y_pred_all = new_scaling_model(popt, N_all_improve, D_all_improve)
        
        # 计算R²
        ss_res = np.sum((y_all_improve - y_pred_all) ** 2)
        ss_tot = np.sum((y_all_improve - np.mean(y_all_improve)) ** 2)
        r2_new = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        
        print(f"新模型拟合参数:")
        print(f"N_c = {N_c_fit:.3g}")
        print(f"α_N = {alpha_N_fit:.4f}")
        print(f"D_c = {D_c_fit:.3g}")
        print(f"D_0 = {D_0_fit:.3g}")
        print(f"R² = {r2_new:.4f}")
        
        # 简单绘图
        plt.figure(figsize=(7, 5))
        for ms in unique_model_sizes:
            subdf = df_mean[df_mean['model_size'] == ms]
            plt.scatter(
                subdf['E'], 
                -np.log(subdf['ImprovementRate']), 
                label=f"{ms}",
                color=color_map[ms], 
                alpha=0.7, 
                s=12
            )
        
        # 为每个模型大小画拟合曲线
        for ms in unique_model_sizes:
            subdf = df_mean[df_mean['model_size'] == ms]
            if len(subdf) < 2:
                continue
            N_val = float(subdf['N'].iloc[0])
            D_vals = subdf['E'].to_numpy()
            D_grid = np.logspace(np.log10(D_vals.min()), np.log10(D_vals.max()), 100)
            N_grid = np.full_like(D_grid, N_val)
            y_grid = new_scaling_model(popt, N_grid, D_grid)
            plt.plot(D_grid, y_grid, color=color_map[ms], linewidth=2)
        
        plt.xscale('log')
        plt.xlabel('Training Examples D (log)')
        plt.ylabel('-log(ImprovementRate)')
        plt.title(r'$-\log(ImprovementRate) = \left(\frac{N}{N_c}\right)^{\alpha_N} \cdot \frac{D_c}{D+D_0}$')
        plt.legend(title="model size", loc="best")
        plt.tight_layout()
        out_path1 = OUTPUT_BASE_DIR / f"{FIGURE_PREFIX}_fit_improvementrate_model.pdf"
        plt.savefig(out_path1, dpi=300, bbox_inches='tight')
        print(f"新模型拟合图已保存到 {out_path1}")
        
    except Exception as e:
        print(f"新模型拟合失败: {e}")

    print(f"\n=== ImprovementRate 模型拟合完成 ===")
    print("模型：新公式拟合 -log(ImprovementRate) = (N/N_c)^α_N * D_c/(D+D_0)")

if __name__ == "__main__":
    main()