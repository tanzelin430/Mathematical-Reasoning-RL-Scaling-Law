#!/usr/bin/env python3
"""
Scaling Law Pipeline - Multi-Metric Analysis
Processes multiple test metrics from Experiment1 data and generates scaling law plots for each metric
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
WARMUP_CLIPPING_NUM = 20


PLOT_BASIC_CURVES = False # True for Intrinsic Curves

HOLDOUT=True
if HOLDOUT:
    # Test metrics to process (from the CSV columns)
    TEST_METRICS = [
        'holdout_score',
    ]
    FIGURE_PREFIX = 'holdout'
    FIGURE_COLUMNS = 1 # note: if total > figure_columns, [row, col] -> [i]
    FIGURE_SIZE=(5, 5)
else:
    # Test metrics to process (from the CSV columns)
    TEST_METRICS = [
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


total_metrics = len(TEST_METRICS)
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

    print("=== Multi-Metric Scaling Law Analysis ===")
    print(f"Processing CSV: {csv_exp1_run0} and {csv_exp1_run1}")
    print(f"Output directory: {OUTPUT_BASE_DIR}")
    print(f"Metrics to process: {len(TEST_METRICS)}")
    
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
    data_proc.validate_data(df, metric_columns=TEST_METRICS)
    df = data_proc.normalize_data(df)
    # Calculate E = step * sample_size_per_step
    df['E'] = df['step'] * float(SAMPLE_SIZE_PER_STEP)

    # Estimate global efficiency parameter phi
    phi_global, phi_by_N, phi_stats_df = data_proc.estimate_phi_from_runs(
        df, 
        sample_size_per_step=SAMPLE_SIZE_PER_STEP, 
        tail_fraction=0.5
    )
    print(f"phi (global tail median) = {phi_global}")
    
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

    # 只处理第一个metric
    metric_name = TEST_METRICS[0]
    # 计算Error Rate
    df['ErrRate'] = 1 - df[metric_name]
    
    # 计算Improvement Rate：相对于step=0的改进率
    def calc_improvement_rate_for_group(group):
        step_0_rows = group[group['step'] == 0]
        if len(step_0_rows) == 0:
            raise ValueError(f"No step=0 found for model_size={group['model_size'].iloc[0]}")
        baseline_score = step_0_rows[metric_name].iloc[0]
        group = group.copy()
        group['ImprovementRate'] = group[metric_name] / baseline_score
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
    # 拟合模型：对数线性模型
    # 变量命名：ErrRate=Error Rate, N=model_size, E=Training Examples
    # ===========================
    N_all = df_mean['N'].to_numpy(dtype=float)
    E_all = df_mean['E'].to_numpy(dtype=float)  # Training Examples (x-axis)
    ErrRate_all = df_mean['ErrRate'].to_numpy(dtype=float)  # Error Rate (y-axis)

    # 安全处理：避免 log(0)
    eps = 1e-12

    # ---------------------------
    # 模型 0 (双Logistic拟合版)：log10(ErrRate) = -k(N) * log10(E) + E0(N)
    # 其中 k(N) = L / (1 + exp(-r * (N - N0)))，E0(N) = A_e0 / (1 + exp(r_e0 * (N - N0_e0))) + B_e0
    # ---------------------------
    print("\n=== 模型0 双Logistic拟合版：log10(ErrRate) = -k(N) * log10(E) + E0(N) ===")
    print("其中 k(N) = L / (1 + exp(-r * (N - N0)))")
    print("E0(N) = A_e0 / (1 + exp(r_e0 * (N - N0_e0))) + B_e0")
    
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
    

    # 局域函数定义
    def logistic_k(N, L, r, N0_k):
        """Logistic函数：k(N) = L / (1 + exp(-r * (N - N0_k)))"""
        return L / (1 + np.exp(-r * (N - N0_k)))

    def logistic_e0(N, L, r_e0, N0_e0, B_e0):
        """Logistic函数用于E0(N)：E0(N) = L / (1 + exp(r_e0 * (N - N0_e0))) + B_e0"""
        return L / (1 + np.exp(r_e0 * (N - N0_e0))) + B_e0

    def global_model(params, N, log10_E):
        """双Logistic拟合函数：log10(ErrRate) = -k(N) * log10(E) + E0(N)
        其中 k(N) = L / (1 + exp(-r * (N - N0_k)))
        E0(N) = L / (1 + exp(r_e0 * (N - N0_e0))) + B_e0
        """
        # 共享参数：L
        # k(N)参数：r, N0_k
        # E0(N)参数：r_e0, N0_e0, B_e0
        L, r, N0_k, r_e0, N0_e0, B_e0 = params

        k = logistic_k(N, L, r, N0_k)
        E0 = logistic_e0(N, L, r_e0, N0_e0, B_e0)

        return -k * log10_E + E0

    # # 定义全局拟合函数：log10(ErrRate) = -(a * N + b) * log10(E) + E0
    # def global_model(params, N, log10_E):
    #     a, b, E0 = params
    #     k = a * N + b  # k(N) = a * N + b
    #     return -k * log10_E + E0
    
    # 使用scipy进行非线性拟合
    from scipy.optimize import curve_fit
    
    # # 初始参数估计
    # a_init = 3.65e-12  # 基于之前的分析
    # b_init = 0.0061    # 基于之前的分析
    # E0_init = 0.0      # 从0开始

    # 获取所有唯一的N值
    unique_N_list = sorted([float(subdf['N'].iloc[0]) for ms in unique_model_sizes 
                           for subdf in [df_mean[df_mean['model_size'] == ms]] if len(subdf) > 0])
    n_models = len(unique_N_list)
    
    print(f"发现 {n_models} 个不同的模型大小: {[f'{n:.1e}' for n in unique_N_list]}")
    
    # 初始参数估计
    # 共享参数
    L_init = 0.06      # 共享的最大振幅
    
    # k(N) 参数
    r_init = 2e-10     # k的增长率
    N0_k_init = 5e9    # k的拐点位置
    
    # E0(N) 参数
    r_e0_init = 1e-9   # E0的增长率
    N0_e0_init = 3e9   # E0的拐点位置
    B_e0_init = 0  # E0的下渐近线
    
    # try:
    if True:
        # 定义拟合函数包装器
        # def fit_func(data, a, b, E0):
        #     N, log10_E = data
        #     return global_model([a, b, E0], N, log10_E)
        def fit_func(data, L, r, N0_k, r_e0, N0_e0, B_e0):
            N, log10_E = data
            params = [L, r, N0_k, r_e0, N0_e0, B_e0]
            return global_model(params, N, log10_E)
        
        # popt, pcov = curve_fit(
        #     fit_func,
        #     (N_all_data, log10_E_all),
        #     log10_ErrRate_all,
        #     p0=[a_init, b_init, E0_init],
        #     maxfev=10000
        # )

        # 构建初始参数和边界
        p0 = [L_init, r_init, N0_k_init, r_e0_init, N0_e0_init, B_e0_init]
        # 更合理的边界设置
        lower_bounds = [0, 0, 0, 0, 0, -0.5]  # L>0, r>0, N0_k>0, r_e0>0, N0_e0>0, B_e0>-0.5
        upper_bounds = [1, 1e-6, 1e12, 1e-6, 1e12, 0.5]  # B_e0的范围缩小到[-0.5, 0.5]
        
        print(f"参数数量: {len(p0)} (共享参数: L; k(N): r, N0_k; E0(N): r_e0, N0_e0, B_e0)")
        print(f"参数边界: L[0,1], r[0,1e-6], N0_k[0,1e12], r_e0[0,1e-6], N0_e0[0,1e12], B_e0[-0.5,0.5]")
        
        # 使用更严格的收敛条件和多次尝试不同初始值
        best_popt = None
        best_r2 = -np.inf
        best_result = None
        
        # 尝试不同的B_e0初始值以避免局部最优
        B_e0_candidates = [0, -0.1, 0.1, -0.05, 0.05]
        
        for i, B_e0_try in enumerate(B_e0_candidates):
            try:
                p0_try = [L_init, r_init, N0_k_init, r_e0_init, N0_e0_init, B_e0_try]
                print(f"尝试 #{i+1}: B_e0_init = {B_e0_try}")
                
                popt_try, pcov_try = curve_fit(
                    fit_func,
                    (N_all_data, log10_E_all),
                    log10_ErrRate_all,
                    p0=p0_try,
                    bounds=(lower_bounds, upper_bounds),
                    maxfev=100000,  # 增加最大迭代次数
                    ftol=1e-12,     # 设置函数收敛精度
                    xtol=1e-12,     # 设置参数收敛精度
                    gtol=1e-12      # 设置梯度收敛精度
                )
                
                # 计算R²
                y_pred_try = global_model(popt_try, N_all_data, log10_E_all)
                ss_res_try = np.sum((log10_ErrRate_all - y_pred_try) ** 2)
                ss_tot_try = np.sum((log10_ErrRate_all - np.mean(log10_ErrRate_all)) ** 2)
                r2_try = 1.0 - ss_res_try / ss_tot_try if ss_tot_try > 0 else np.nan
                
                print(f"  R² = {r2_try:.6f}")
                
                if r2_try > best_r2:
                    best_r2 = r2_try
                    best_popt = popt_try
                    best_result = (popt_try, pcov_try)
                    
            except Exception as e:
                print(f"  拟合失败: {e}")
                continue
        
        if best_popt is None:
            raise RuntimeError("所有初始值尝试都失败了，请检查数据和参数设置")
            
        popt, pcov = best_result
        print(f"\n✅ 最佳拟合结果: R² = {best_r2:.6f}")
        
        # 提取拟合参数
        L_fit, r_fit, N0_k_fit, r_e0_fit, N0_e0_fit, B_e0_fit = popt
        
        # 计算全局R²
        y_pred_all = global_model(popt, N_all_data, log10_E_all)
        ss_res = np.sum((log10_ErrRate_all - y_pred_all) ** 2)
        ss_tot = np.sum((log10_ErrRate_all - np.mean(log10_ErrRate_all)) ** 2)
        r2_global = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        
        # print(f"全局拟合参数 k(N) = a * N + b:")
        # print(f"a = {a_fit:.2e}")
        # print(f"b = {b_fit:.6f}")
        # print(f"E0 = {E0_fit:.6f}")
        # print(f"k(N) = {a_fit:.2e} * N + {b_fit:.6f}")
        # print(f"log10(ErrRate) = -k(N) * log10(E) + {E0_fit:.6f}")
        # print(f"全局 R² = {r2_global:.4f}")
        
        print(f"\n✅ 双Logistic拟合成功！")
        print(f"\n共享参数:")
        print(f"  L = {L_fit:.6f} (共享最大振幅)")
        
        print(f"\nk(N) 参数:")
        print(f"  r = {r_fit:.2e} (k的增长率)")
        print(f"  N0_k = {N0_k_fit:.2e} (k的拐点位置)")
        print(f"  k(N) = {L_fit:.6f} / (1 + exp(-{r_fit:.2e} * (N - {N0_k_fit:.2e})))")
        
        print(f"\nE0(N) 参数:")
        print(f"  r_e0 = {r_e0_fit:.2e} (E0的增长率)")
        print(f"  N0_e0 = {N0_e0_fit:.2e} (E0的拐点位置)")
        print(f"  B_e0 = {B_e0_fit:.6f} (E0的下渐近线)")
        print(f"  E0(N) = {L_fit:.6f} / (1 + exp({r_e0_fit:.2e} * (N - {N0_e0_fit:.2e}))) + {B_e0_fit:.6f}")
        
        print(f"\n全局 R² = {r2_global:.4f}")
        
        # 计算并显示每个模型大小对应的E0值
        print(f"\n每个模型大小对应的E0(N)预测值:")
        E0_fit_list = [logistic_e0(n_val, L_fit, r_e0_fit, N0_e0_fit, B_e0_fit) for n_val in unique_N_list]
        for i, n_val in enumerate(unique_N_list):
            print(f"  N = {n_val:.2e}: E0(N) = {E0_fit_list[i]:.6f}")
        
        # ===========================
        # 绘制E0 vs N的scatter图
        # ===========================
        plt.figure(figsize=(10, 6))
        
        # 创建子图
        plt.subplot(1, 2, 1)
        # E0 vs N scatter图 - 对数尺度
        plt.scatter(unique_N_list, E0_fit_list, color='red', s=80, alpha=0.8, 
                   marker='o', edgecolor='black', linewidth=1, zorder=5)
        
        # 添加数值标签
        for i, (n_val, e0_val) in enumerate(zip(unique_N_list, E0_fit_list)):
            plt.annotate(f'{e0_val:.3f}', 
                        (n_val, e0_val), 
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center', 
                        fontsize=9)
        
        plt.xscale('log')
        plt.xlabel('Model Size N (parameters)')
        plt.ylabel('E0 parameter')
        plt.title('E0 vs Model Size N (Log Scale)')
        plt.grid(True, alpha=0.3)
        
        # 线性尺度的E0 vs N图
        plt.subplot(1, 2, 2)
        plt.scatter([n/1e9 for n in unique_N_list], E0_fit_list, color='blue', s=80, alpha=0.8,
                   marker='s', edgecolor='black', linewidth=1, zorder=5)
        
        # 添加数值标签
        for i, (n_val, e0_val) in enumerate(zip(unique_N_list, E0_fit_list)):
            plt.annotate(f'{e0_val:.3f}', 
                        (n_val/1e9, e0_val), 
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center', 
                        fontsize=9)
        
        plt.xlabel('Model Size N (Billions of parameters)')
        plt.ylabel('E0 parameter')
        plt.title('E0 vs Model Size N (Linear Scale)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        out_path_e0_scatter = OUTPUT_BASE_DIR / f"{FIGURE_PREFIX}_E0_vs_N_scatter.pdf"
        plt.savefig(out_path_e0_scatter, dpi=300, bbox_inches='tight')
        print(f"\nE0 vs N scatter图已保存到 {out_path_e0_scatter}")
        
        # 分析E0(N)的趋势
        print(f"\n=== E0(N)关系分析 ====")
        
        # 计算E0的统计信息
        E0_min = min(E0_fit_list)
        E0_max = max(E0_fit_list)
        E0_mean = np.mean(E0_fit_list)
        E0_std = np.std(E0_fit_list)
        
        print(f"E0范围: [{E0_min:.6f}, {E0_max:.6f}]")
        print(f"E0均值: {E0_mean:.6f} ± {E0_std:.6f}")
        
        # 简单拟合E0与N的关系
        # 尝试线性关系: E0 = a_e0 * N + b_e0
        try:
            coeffs_linear = np.polyfit(unique_N_list, E0_fit_list, 1)
            a_e0, b_e0 = coeffs_linear
            E0_pred_linear = a_e0 * np.array(unique_N_list) + b_e0
            r2_e0_linear = 1 - np.sum((np.array(E0_fit_list) - E0_pred_linear)**2) / np.sum((np.array(E0_fit_list) - E0_mean)**2)
            print(f"\n线性拟合 E0(N) = a*N + b:")
            print(f"  a = {a_e0:.2e}")
            print(f"  b = {b_e0:.6f}")
            print(f"  R² = {r2_e0_linear:.4f}")
        except:
            print("线性拟合失败")
        
        # 尝试对数关系: E0 = a_e0 * log(N) + b_e0
        try:
            coeffs_log = np.polyfit(np.log(unique_N_list), E0_fit_list, 1)
            a_e0_log, b_e0_log = coeffs_log
            E0_pred_log = a_e0_log * np.log(unique_N_list) + b_e0_log
            r2_e0_log = 1 - np.sum((np.array(E0_fit_list) - E0_pred_log)**2) / np.sum((np.array(E0_fit_list) - E0_mean)**2)
            print(f"\n对数拟合 E0(N) = a*log(N) + b:")
            print(f"  a = {a_e0_log:.6f}")
            print(f"  b = {b_e0_log:.6f}")
            print(f"  R² = {r2_e0_log:.4f}")
        except:
            print("对数拟合失败")
        
        # 尝试Logistic函数拟合 E0(N)
        # 对于递减的趋势，使用反向Logistic: E0(N) = L_e0 - A_e0 / (1 + exp(-r_e0 * (N - N0_e0)))
        # 或者简化为: E0(N) = A_e0 / (1 + exp(r_e0 * (N - N0_e0))) + B_e0
        def logistic_e0_func(N, A_e0, r_e0, N0_e0, B_e0):
            """
            Logistic函数拟合E0(N)
            E0(N) = A_e0 / (1 + exp(r_e0 * (N - N0_e0))) + B_e0
            """
            return A_e0 / (1 + np.exp(r_e0 * (N - N0_e0))) + B_e0
        
        try:
            # 初始参数估计
            A_e0_init = max(E0_fit_list) - min(E0_fit_list)  # 振幅
            r_e0_init = 1e-9  # 增长率（正值表示递减）
            N0_e0_init = np.median(unique_N_list)  # 拐点
            B_e0_init = min(E0_fit_list)  # 下渐近线
            
            print(f"\nLogistic拟合 E0(N) = A/(1 + exp(r*(N-N0))) + B:")
            print(f"初始参数估计: A={A_e0_init:.6f}, r={r_e0_init:.2e}, N0={N0_e0_init:.2e}, B={B_e0_init:.6f}")
            
            popt_e0_logistic, pcov_e0_logistic = curve_fit(
                logistic_e0_func, 
                unique_N_list, 
                E0_fit_list,
                p0=[A_e0_init, r_e0_init, N0_e0_init, B_e0_init],
                bounds=([0, 0, 0, -0.5], [1, 1e-6, 1e12, 0.5]),  # 同样限制B的范围
                maxfev=50000,   # 增加迭代次数
                ftol=1e-12,     # 设置收敛精度
                xtol=1e-12,
                gtol=1e-12
            )
            
            A_e0_fit, r_e0_fit, N0_e0_fit, B_e0_fit = popt_e0_logistic
            E0_pred_logistic = logistic_e0_func(unique_N_list, *popt_e0_logistic)
            r2_e0_logistic = 1 - np.sum((np.array(E0_fit_list) - E0_pred_logistic)**2) / np.sum((np.array(E0_fit_list) - E0_mean)**2)
            
            print(f"  A = {A_e0_fit:.6f} (振幅)")
            print(f"  r = {r_e0_fit:.2e} (增长率)")
            print(f"  N0 = {N0_e0_fit:.2e} (拐点位置)")
            print(f"  B = {B_e0_fit:.6f} (下渐近线)")
            print(f"  R² = {r2_e0_logistic:.4f}")
            
            # 在E0 vs N图中添加Logistic拟合线
            # 重新绘制E0 vs N图，加上Logistic拟合线
            plt.figure(figsize=(12, 8))
            
            # 子图1: 对数尺度 + 所有拟合线
            plt.subplot(2, 2, 1)
            plt.scatter(unique_N_list, E0_fit_list, color='red', s=80, alpha=0.8, 
                       marker='o', edgecolor='black', linewidth=1, zorder=5, label='Actual E0 values')
            
            # 绘制拟合线
            N_smooth = np.logspace(np.log10(min(unique_N_list)), np.log10(max(unique_N_list)), 200)
            
            # Logistic拟合线
            E0_smooth_logistic = logistic_e0_func(N_smooth, *popt_e0_logistic)
            plt.plot(N_smooth, E0_smooth_logistic, 'b-', linewidth=2, label=f'Logistic (R²={r2_e0_logistic:.3f})')
            
            # 线性拟合线
            if 'r2_e0_linear' in locals():
                E0_smooth_linear = a_e0 * N_smooth + b_e0
                plt.plot(N_smooth, E0_smooth_linear, 'g--', linewidth=2, label=f'Linear (R²={r2_e0_linear:.3f})')
            
            # 对数拟合线
            if 'r2_e0_log' in locals():
                E0_smooth_log = a_e0_log * np.log(N_smooth) + b_e0_log
                plt.plot(N_smooth, E0_smooth_log, 'm:', linewidth=2, label=f'Logarithmic (R²={r2_e0_log:.3f})')
            
            plt.xscale('log')
            plt.xlabel('Model Size N (parameters)')
            plt.ylabel('E0 parameter')
            plt.title('E0(N) Fitting Comparison - Log Scale')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 子图2: 线性尺度
            plt.subplot(2, 2, 2)
            plt.scatter([n/1e9 for n in unique_N_list], E0_fit_list, color='blue', s=80, alpha=0.8,
                       marker='s', edgecolor='black', linewidth=1, zorder=5, label='Actual E0 values')
            
            N_smooth_linear = np.linspace(min(unique_N_list), max(unique_N_list), 200)
            
            # Logistic拟合线
            E0_smooth_logistic_linear = logistic_e0_func(N_smooth_linear, *popt_e0_logistic)
            plt.plot([n/1e9 for n in N_smooth_linear], E0_smooth_logistic_linear, 'b-', linewidth=2, label=f'Logistic (R²={r2_e0_logistic:.3f})')
            
            plt.xlabel('Model Size N (Billions of parameters)')
            plt.ylabel('E0 parameter')
            plt.title('E0(N) Fitting Comparison - Linear Scale')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 子图3: 残差分析
            plt.subplot(2, 2, 3)
            residuals_logistic = np.array(E0_fit_list) - E0_pred_logistic
            plt.scatter(unique_N_list, residuals_logistic, color='red', alpha=0.8)
            plt.axhline(y=0, color='black', linestyle='--')
            plt.xscale('log')
            plt.xlabel('Model Size N (parameters)')
            plt.ylabel('Logistic Fit Residuals')
            plt.title('Logistic Fit Residual Analysis')
            plt.grid(True, alpha=0.3)
            
            # 子图4: 拟合质量比较
            plt.subplot(2, 2, 4)
            methods = []
            r2_values = []
            
            if 'r2_e0_linear' in locals():
                methods.append('Linear')
                r2_values.append(r2_e0_linear)
            if 'r2_e0_log' in locals():
                methods.append('Logarithmic')
                r2_values.append(r2_e0_log)
            methods.append('Logistic')
            r2_values.append(r2_e0_logistic)
            
            bars = plt.bar(methods, r2_values, color=['green', 'magenta', 'blue'][:len(methods)], alpha=0.7)
            plt.ylabel('R² Value')
            plt.title('E0(N) Fitting Quality Comparison')
            plt.ylim(0, 1)
            
            # 在柱状图上添加数值标签
            for bar, r2 in zip(bars, r2_values):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{r2:.3f}', ha='center', va='bottom')
            
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            out_path_e0_analysis = OUTPUT_BASE_DIR / f"{FIGURE_PREFIX}_E0_analysis_comprehensive.pdf"
            plt.savefig(out_path_e0_analysis, dpi=300, bbox_inches='tight')
            print(f"\nE0(N)综合分析图已保存到 {out_path_e0_analysis}")
            
            # 比较不同拟合方法的质量
            print(f"\n=== E0(N)拟合质量比较 ====")
            if 'r2_e0_linear' in locals():
                print(f"线性拟合 R²: {r2_e0_linear:.4f}")
            if 'r2_e0_log' in locals():
                print(f"对数拟合 R²: {r2_e0_log:.4f}")
            print(f"Logistic拟合 R²: {r2_e0_logistic:.4f}")
            
            best_r2 = max([r2 for r2 in [r2_e0_linear if 'r2_e0_linear' in locals() else 0, 
                          r2_e0_log if 'r2_e0_log' in locals() else 0, 
                          r2_e0_logistic]])
            if best_r2 == r2_e0_logistic:
                print(f"\n🏆 Logistic函数拟合效果最佳！")
                print(f"E0(N) = {A_e0_fit:.6f} / (1 + exp({r_e0_fit:.2e} * (N - {N0_e0_fit:.2e}))) + {B_e0_fit:.6f}")
            
        except Exception as e:
            print(f"Logistic拟合失败: {e}")
            
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
            k_logistic = logistic_k(N_val, L_fit, r_fit, N0_k_fit)  # 使用Logistic拟合的k(N)关系
            
            # 计算E0值（使用Logistic函数）
            E0_local = logistic_e0(N_val, L_fit, r_e0_fit, N0_e0_fit, B_e0_fit)
            
            # 计算该模型大小的R²
            y_pred = global_model(popt, np.array([N_val] * len(x)), x)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2_local = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            
            model0_stats.append({
                'N': N_val, 'k': k_logistic, 'E0': E0_local, 'r2_log': r2_local,
                'E_min': E_vals.min(), 'E_max': E_vals.max(), 'y0': y0
            })
            print(f"N={N_val:.3g}: k={k_logistic:.6f} (Logistic), E0={E0_local:.6f} (Logistic), R2(log10 ErrRate)={r2_local:.4f}")
            
            # 画拟合线
            E_grid = np.logspace(np.log10(E_vals.min()), np.log10(E_vals.max()), 200)
            x_grid = np.log10(E_grid)
            y_grid = global_model(popt, np.array([N_val] * len(x_grid)), x_grid)
            # plt.plot(x_grid, y_grid - y0, color=color_map[ms], linewidth=2, linestyle='--')
            plt.plot(x_grid, y_grid, color=color_map[ms], linewidth=2, linestyle='--')
        
    # except Exception as e:
    #     print(f"全局拟合失败: {e}")
    #     model0_stats = []

    # 线性坐标：x=log10(E), y=Δlog10(ErrRate)
    plt.xlabel(r"$\log_{10}E$")
    plt.ylabel(r"$\log_{10}ErrRate$")
    
    # 更新标题显示完整的双 Logistic 公式（抽象形式）
    title_text = (r"Dual-Logistic Model: $\log_{10}ErrRate = -k(N) \cdot \log_{10}E + E_0(N)$" + "\n" +
                 r"$k(N) = \frac{L}{1 + \exp(-r \cdot (N - N_{0k}))}$" + "\n" +
                 r"$E_0(N) = \frac{L}{1 + \exp(r_{e0} \cdot (N - N_{0e0}))} + B$")
    plt.title(title_text, fontsize=11, pad=20)
    
    # 在图上添加拟合参数信息
    info_text = (
        f"Fitting Results:\n"
        f"Global R² = {r2_global:.4f}\n"
        f"Shared Parameter:\n"
        f"  L = {L_fit:.6f}\n"
        f"k(N) Parameters:\n"
        f"  r = {r_fit:.2e}\n"
        f"  N₀k = {N0_k_fit:.2e}\n"
        f"E₀(N) Parameters:\n"
        f"  r_e0 = {r_e0_fit:.2e}\n"
        f"  N₀e0 = {N0_e0_fit:.2e}\n"
        f"  B = {B_e0_fit:.6f}"
    )
    
    # 在图的右上角添加信息框
    plt.text(0.98, 0.98, info_text, transform=plt.gca().transAxes, 
             fontsize=8, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.legend(title="Model Size", loc="lower left")
    plt.tight_layout()
    out_path0 = OUTPUT_BASE_DIR / f"{FIGURE_PREFIX}_fit_model0.pdf"
    plt.savefig(out_path0, dpi=300, bbox_inches='tight')
    print(f"拟合曲线图已保存到 {out_path0}")

    print(f"\n=== ErrRate 模型拟合完成 ===")
    print("模型：双Logistic拟合 log10(ErrRate) = -k(N) * log10(E) + E0(N)")
    print(f"其中 k(N) = {L_fit:.6f} / (1 + exp(-{r_fit:.2e} * (N - {N0_k_fit:.2e})))")
    print(f"E0(N) = {L_fit:.6f} / (1 + exp({r_e0_fit:.2e} * (N - {N0_e0_fit:.2e}))) + {B_e0_fit:.6f}")
    if model0_stats:
        k_mean = np.mean([s['k'] for s in model0_stats])
        r2_mean = np.mean([s['r2_log'] for s in model0_stats])
        print(f"平均 k ≈ {k_mean:.3f}")
        print(f"平均 R2(log10 ErrRate) ≈ {r2_mean:.3f}")

if __name__ == "__main__":
    main()