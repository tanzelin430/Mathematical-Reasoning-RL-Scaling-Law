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
    

    csv_exp1_run0 = SCRIPT_DIR / "csv" / "scaling_law_data_experiment1_base_run0.csv" 
    csv_exp1_run1 = SCRIPT_DIR / "csv" / "scaling_law_data_experiment1_base.csv" 

    # csv_exp1_run0 = SCRIPT_DIR / "csv" / "scaling_law_data_experiment1_instruct_run0.csv" 
    # csv_exp1_run1 = SCRIPT_DIR / "csv" / "scaling_law_data_experiment1_instruct_run1.csv" 
    # csv_exp1_run2 = SCRIPT_DIR / "csv" / "scaling_law_data_experiment1_instruct_run2.csv" 

    print("=== Multi-Eval Scaling Law Analysis ===")
    
    df_run0 = pd.read_csv(csv_exp1_run0)
    df_run1 = pd.read_csv(csv_exp1_run1)

    df = pd.concat([df_run0, df_run1], ignore_index=True)
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
    # 模型 0 (全局拟合版)：log10(ErrRate) = -(a * N + b) * log10(E) + E0
    # 其中 a, b, E0 作为全局参数一起拟合
    # ---------------------------
    print("\n=== 模型0全局拟合版：log10(ErrRate) = -(a * N + b) * log10(E) + E0 ===")
    
    # 准备所有数据
    N_all_data = []
    E_all_data = []
    ErrRate_all_data = []
    
    for ms in unique_model_sizes:
        subdf = df_mean[df_mean['model_size'] == ms]
        N_val = float(subdf['N'].iloc[0])
        E_vals = subdf['E'].to_numpy(dtype=float)
        ErrRate_vals = np.clip(subdf['ErrRate'].to_numpy(dtype=float), 1e-12, None)
        
        N_all_data.extend([N_val] * len(E_vals))
        E_all_data.extend(E_vals)
        ErrRate_all_data.extend(ErrRate_vals)
    
    N_all_data = np.array(N_all_data)
    E_all_data = np.array(E_all_data)
    ErrRate_all_data = np.array(ErrRate_all_data)
    
    # 转换为对数空间
    log10_E_all = np.log10(E_all_data)
    log10_ErrRate_all = np.log10(ErrRate_all_data)
    
    # 定义全局拟合函数：log10(ErrRate) = -(a * N + b) * log10(E) + E0
    def global_model(params, N, log10_E):
        a, b, E0 = params
        k = a * N + b  # k(N) = a * N + b
        return -k * log10_E + E0
    
    # 使用scipy进行非线性拟合
    from scipy.optimize import curve_fit
    
    # 初始参数估计
    a_init = 3.65e-12  # 基于之前的分析
    b_init = 0.0061    # 基于之前的分析
    E0_init = 0.0      # 从0开始
    
    try:
        # 定义拟合函数包装器
        def fit_func(data, a, b, E0):
            N, log10_E = data
            return global_model([a, b, E0], N, log10_E)
        
        popt, pcov = curve_fit(
            fit_func,
            (N_all_data, log10_E_all),
            log10_ErrRate_all,
            p0=[a_init, b_init, E0_init],
            maxfev=10000
        )
        
        a_fit, b_fit, E0_fit = popt
        
        # 计算全局R²
        y_pred_all = global_model(popt, N_all_data, log10_E_all)
        ss_res = np.sum((log10_ErrRate_all - y_pred_all) ** 2)
        ss_tot = np.sum((log10_ErrRate_all - np.mean(log10_ErrRate_all)) ** 2)
        r2_global = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        
        print(f"全局拟合参数 k(N) = a * N + b:")
        print(f"a = {a_fit:.2e}")
        print(f"b = {b_fit:.6f}")
        print(f"E0 = {E0_fit:.6f}")
        print(f"k(N) = {a_fit:.2e} * N + {b_fit:.6f}")
        print(f"log10(ErrRate) = -k(N) * log10(E) + {E0_fit:.6f}")
        print(f"全局 R² = {r2_global:.4f}")
        
        # 绘图
        model0_stats = []
        plt.figure(figsize=(7, 5))
        
        # 先画散点
        for ms in unique_model_sizes:
            subdf = df_mean[df_mean['model_size'] == ms].sort_values('E')
            E_vals = subdf['E'].to_numpy(dtype=float)
            ErrRate_vals = np.clip(subdf['ErrRate'].to_numpy(dtype=float), 1e-12, None)
            x = np.log10(E_vals)
            y = np.log10(ErrRate_vals)
            y0 = y[0]  # 起始点，用于相对显示
            plt.scatter(
                # x, y - y0, label=f"{ms}",
                x, y, label=f"{ms}",
                color=color_map[ms], alpha=0.6, s=12
            )
        
        # 画拟合线
        for ms in unique_model_sizes:
            subdf = df_mean[df_mean['model_size'] == ms]
            if len(subdf) < 2:
                continue
            E_vals = subdf['E'].to_numpy(dtype=float)
            ErrRate_vals = np.clip(subdf['ErrRate'].to_numpy(dtype=float), 1e-12, None)
            x = np.log10(E_vals)
            y = np.log10(ErrRate_vals)
            y0 = y[0]
            
            N_val = float(subdf['N'].iloc[0])
            k_global = a_fit * N_val + b_fit  # 使用全局拟合的k(N)关系
            
            # 计算该模型大小的R²
            y_pred = global_model(popt, np.array([N_val] * len(x)), x)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2_local = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            
            model0_stats.append({
                'N': N_val, 'k': k_global, 'E0': E0_fit, 'r2_log': r2_local,
                'E_min': E_vals.min(), 'E_max': E_vals.max(), 'y0': y0
            })
            print(f"N={N_val:.3g}: k={k_global:.6f} (全局), E0={E0_fit:.6f} (全局), R2(log10 ErrRate)={r2_local:.4f}")
            
            # 画拟合线
            E_grid = np.logspace(np.log10(E_vals.min()), np.log10(E_vals.max()), 200)
            x_grid = np.log10(E_grid)
            y_grid = global_model(popt, np.array([N_val] * len(x_grid)), x_grid)
            # plt.plot(x_grid, y_grid - y0, color=color_map[ms], linewidth=2, linestyle='--')
            plt.plot(x_grid, y_grid, color='black', linewidth=2, linestyle='--')
        
    except Exception as e:
        print(f"全局拟合失败: {e}")
        model0_stats = []

    # 线性坐标：x=log10(E), y=Δlog10(ErrRate)
    plt.xlabel(r"$\log_{10}E$")
    plt.ylabel(r"$\log_{10}ErrRate$")
    plt.title(r"Model0 Fitting: $\log_{10}ErrRate = -(a N + b) \log_{10}E + E_0$")
    plt.legend(title="model size", loc="best")
    plt.tight_layout()
    out_path0 = OUTPUT_BASE_DIR / f"{FIGURE_PREFIX}_fit_model0.pdf"
    plt.savefig(out_path0, dpi=300, bbox_inches='tight')
    print(f"拟合曲线图已保存到 {out_path0}")

    print(f"\n=== ErrRate 模型拟合完成 ===")
    print("模型：对数线性拟合 log10(ErrRate) = -k * log10(E) + E0")
    if model0_stats:
        k_mean = np.mean([s['k'] for s in model0_stats])
        r2_mean = np.mean([s['r2_log'] for s in model0_stats])
        print(f"平均 k ≈ {k_mean:.3f}")
        print(f"平均 R2(log10 ErrRate) ≈ {r2_mean:.3f}")

    # ===========================
    # 测试C和E拟合的k(N)和E0(N)关系
    # ===========================
    print(f"\n=== 测试C和E拟合的k(N)和E0(N)关系 ===")
    
    # 对每个N分别拟合 log10(ErrRate) = k * log10(X) + E0，其中X为E或C
    from sklearn.linear_model import LinearRegression
    
    fit_results_E = {}  # 拟合E的结果
    fit_results_C = {}  # 拟合C的结果
    
    for ms in unique_model_sizes:
        subdf = df_mean[df_mean['model_size'] == ms]
        if len(subdf) < 2:
            continue
            
        N_val = float(subdf['N'].iloc[0])
        E_vals = subdf['E'].to_numpy(dtype=float)
        C_vals = subdf['C'].to_numpy(dtype=float)
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
        
        # 拟合E: log_ErrRate = k_E * log_E + E0_E
        reg_E = LinearRegression()
        reg_E.fit(log_E.reshape(-1, 1), log_ErrRate)
        k_E = -reg_E.coef_[0]  # Take negative to store positive k
        E0_E = reg_E.intercept_
        r2_E = reg_E.score(log_E.reshape(-1, 1), log_ErrRate)
        
        # 拟合C: log_ErrRate = k_C * log_C + E0_C
        reg_C = LinearRegression()
        reg_C.fit(log_C.reshape(-1, 1), log_ErrRate)
        k_C = -reg_C.coef_[0]  # Take negative to store positive k
        E0_C = reg_C.intercept_
        r2_C = reg_C.score(log_C.reshape(-1, 1), log_ErrRate)
        
        fit_results_E[N_val] = {'k': k_E, 'E0': E0_E, 'r2': r2_E, 'n_points': len(log_E)}
        fit_results_C[N_val] = {'k': k_C, 'E0': E0_C, 'r2': r2_C, 'n_points': len(log_C)}
        
        print(f"N={N_val/1e9:>4.1f}B: k_E={k_E:>8.4f}, E0_E={E0_E:>8.4f}, R²_E={r2_E:>6.4f}")
        print(f"N={N_val/1e9:>4.1f}B: k_C={k_C:>8.4f}, E0_C={E0_C:>8.4f}, R²_C={r2_C:>6.4f}")
        print(f"        k差异={k_C-k_E:>8.6f}, E0差异={E0_C-E0_E:>8.4f}")
        print()
    
    # 提取数据用于绘图
    N_values = sorted(fit_results_E.keys())
    k_E_values = [fit_results_E[N]['k'] for N in N_values]
    k_C_values = [fit_results_C[N]['k'] for N in N_values]
    E0_E_values = [fit_results_E[N]['E0'] for N in N_values]
    E0_C_values = [fit_results_C[N]['E0'] for N in N_values]
    
    N_billions = np.array(N_values) / 1e9
    
    # 绘制k(N)对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # k(N)散点图
    ax1.scatter(N_billions, np.abs(k_E_values), color='blue', s=100, alpha=0.7, label='k from E fitting')
    ax1.scatter(N_billions, np.abs(k_C_values), color='red', s=100, alpha=0.7, label='k from C fitting')
    ax1.set_xscale('log')
    ax1.set_xlabel('Model Size N (Billions of Parameters)')
    ax1.set_ylabel('|k(N)| (Data Efficiency Slope)')
    ax1.set_title('k(N) Comparison: E vs C Fitting')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # E0(N)散点图
    ax2.scatter(N_billions, E0_E_values, color='blue', s=100, alpha=0.7, label='E0 from E fitting')
    ax2.scatter(N_billions, E0_C_values, color='red', s=100, alpha=0.7, label='E0 from C fitting')
    ax2.set_xscale('log')
    ax2.set_xlabel('Model Size N (Billions of Parameters)')
    ax2.set_ylabel('E0(N) (Error Offset)')
    ax2.set_title('E0(N) Comparison: E vs C Fitting')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    comparison_path = OUTPUT_BASE_DIR / f"{FIGURE_PREFIX}_k_E0_comparison.pdf"
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"k(N)和E0(N)对比图已保存到 {comparison_path}")
    
    # 分析k值差异
    k_differences = np.array(k_C_values) - np.array(k_E_values)
    max_k_diff = np.max(np.abs(k_differences))
    print(f"\n=== k(N)差异分析 ===")
    print(f"k值最大差异: {max_k_diff:.15f}")
    if max_k_diff < 1e-10:
        print("✅ k值在E和C拟合中完全相同")
    else:
        print("❌ k值存在差异")
    
    # 分析E0变化模式
    E0_E_range = np.max(E0_E_values) - np.min(E0_E_values)
    E0_C_range = np.max(E0_C_values) - np.min(E0_C_values)
    print(f"\n=== E0(N)变化幅度分析 ===")
    print(f"E0_E变化幅度: {E0_E_range:.6f}")
    print(f"E0_C变化幅度: {E0_C_range:.6f}")
    print(f"变化幅度比值: {E0_C_range/E0_E_range:.2f}")
    
    # 计算相关性
    from scipy.stats import pearsonr
    k_E_corr_with_N = pearsonr(np.log10(N_values), k_E_values)[0]
    k_C_corr_with_N = pearsonr(np.log10(N_values), k_C_values)[0]
    E0_E_corr_with_N = pearsonr(np.log10(N_values), E0_E_values)[0]
    E0_C_corr_with_N = pearsonr(np.log10(N_values), E0_C_values)[0]
    
    print(f"\n=== 与模型大小N的相关性分析 ===")
    print(f"k_E与log(N)的相关系数: {k_E_corr_with_N:.4f}")
    print(f"k_C与log(N)的相关系数: {k_C_corr_with_N:.4f}")
    print(f"E0_E与log(N)的相关系数: {E0_E_corr_with_N:.4f}")
    print(f"E0_C与log(N)的相关系数: {E0_C_corr_with_N:.4f}")
    
    # 验证数学关系 E0_C = E0_E + k_E * log10(N * phi_global)
    print(f"\n=== 验证数学关系 ===")
    print(f"phi_global = {phi_global:.2e}")
    print("验证关系: E0_C(N) = E0_E(N) + k(N) * log10(N * phi_global)")
    print("N(B)\t\tE0_C实际\t\tE0_C理论\t\t差异")
    print("-" * 60)
    
    for i, N in enumerate(N_values):
        E0_C_actual = E0_C_values[i]
        E0_E_actual = E0_E_values[i]
        k_actual = k_E_values[i]
        
        log_N_phi = np.log10(N * phi_global)
        E0_C_theoretical = E0_E_actual + k_actual * log_N_phi
        difference = E0_C_actual - E0_C_theoretical
        
        print(f"{N/1e9:.1f}\t\t{E0_C_actual:.6f}\t\t{E0_C_theoretical:.6f}\t\t{difference:.6f}")

if __name__ == "__main__":
    main()