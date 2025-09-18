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


# Y="Reward"
Y="ErrRate"
# Y="log(ErrRate)"

# X="logE"
# X="C"
# X="E"
X="N"
X="T"

X_SCALE = "log"
Y_SCALE = None

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

    def logistic_e0(N, L, r_e0, N0_e0):
        """Logistic函数用于E0(N)：E0(N) = L / (1 + exp(r_e0 * (N - N0_e0)))"""
        return L / (1 + np.exp(r_e0 * (N - N0_e0)))

    def global_model(params, N, log10_E):
        """双Logistic拟合函数：log10(ErrRate) = -k(N) * log10(E) + E0(N)
        其中 k(N) = L / (1 + exp(-r * (N - N0_k)))
        E0(N) = L / (1 + exp(r_e0 * (N - N0_e0)))
        """
        # 共享参数：L
        # k(N)参数：r, N0_k
        # E0(N)参数：r_e0, N0_e0
        L, r, N0_k, r_e0, N0_e0 = params

        k = logistic_k(N, L, r, N0_k)
        E0 = logistic_e0(N, L, r_e0, N0_e0)

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
    
    
    # try:
    if True:
        # 定义拟合函数包装器
        # def fit_func(data, a, b, E0):
        #     N, log10_E = data
        #     return global_model([a, b, E0], N, log10_E)
        def fit_func(data, L, r, N0_k, r_e0, N0_e0):
            N, log10_E = data
            params = [L, r, N0_k, r_e0, N0_e0]
            return global_model(params, N, log10_E)
        
        # popt, pcov = curve_fit(
        #     fit_func,
        #     (N_all_data, log10_E_all),
        #     log10_ErrRate_all,
        #     p0=[a_init, b_init, E0_init],
        #     maxfev=10000
        # )

        # 构建初始参数和边界 - 根据数据特征调整边界
        p0 = [L_init, r_init, N0_k_init, r_e0_init, N0_e0_init]
        
        # 更合理的边界设置：基于初始值的合理范围
        lower_bounds = [0.001, 1e-12, 1e8, 1e-12, 1e8]   # 避免0边界
        upper_bounds = [0.5, 1e-8, 2e10, 1e-8, 2e10]     # 缩小边界范围，围绕初始值
        
        print(f"参数数量: {len(p0)} (共享参数: L; k(N): r, N0_k; E0(N): r_e0, N0_e0)")
        print(f"参数边界: L[0.001,0.5], r[1e-12,1e-8], N0_k[1e8,2e10], r_e0[1e-12,1e-8], N0_e0[1e8,2e10]")
        
        # 由于移除了B参数，模型更稳定，使用单次拟合即可
        try:
            print("开始拟合...")
            
            print(f"初始参数: {p0}")
            print(f"下边界: {lower_bounds}")
            print(f"上边界: {upper_bounds}")
            
            popt, pcov = curve_fit(
                fit_func,
                (N_all_data, log10_E_all),
                log10_ErrRate_all,
                p0=p0,
                bounds=(lower_bounds, upper_bounds),
                method='trf',     # 使用Trust Region算法
                maxfev=50000,     # 适中的迭代次数
                ftol=1e-10,       # 函数收敛精度
                xtol=1e-8,        # 参数收敛精度 (放宽)
                gtol=1e-10        # 梯度收敛精度
            )
            
            print(f"拟合后参数: {popt}")
            print(f"参数变化: {[f'{(new-old)/old*100:.1f}%' if old != 0 else 'N/A' for old, new in zip(p0, popt)]}")
            
            # 计算R²
            y_pred_all = global_model(popt, N_all_data, log10_E_all)
            ss_res = np.sum((log10_ErrRate_all - y_pred_all) ** 2)
            ss_tot = np.sum((log10_ErrRate_all - np.mean(log10_ErrRate_all)) ** 2)
            r2_global = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            
            print(f"✅ 拟合成功: R² = {r2_global:.6f}")
            
        except Exception as e:
            raise RuntimeError(f"拟合失败: {e}")
        
        # 提取拟合参数
        L_fit, r_fit, N0_k_fit, r_e0_fit, N0_e0_fit = popt
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
        print(f"  E0(N) = {L_fit:.6f} / (1 + exp({r_e0_fit:.2e} * (N - {N0_e0_fit:.2e})))")
        
        print(f"\n全局 R² = {r2_global:.4f}")
        
        # 计算并显示每个模型大小对应的E0值
        print(f"\n每个模型大小对应的E0(N)预测值:")
        E0_fit_list = [logistic_e0(n_val, L_fit, r_e0_fit, N0_e0_fit) for n_val in unique_N_list]
        for i, n_val in enumerate(unique_N_list):
            print(f"  N = {n_val:.2e}: E0(N) = {E0_fit_list[i]:.6f}")
        
        # 绘图
        model0_stats = []
        plt.figure(figsize=(7, 5))

        unique_N = sorted(df['N'].unique())
        max_step = min(df_mean[df_mean['N'] == ms]['E'].max() for ms in unique_N)

        x_models = unique_N
        
        y_errrate_step_max = [df_mean[(df_mean['N'] == ms) & (df_mean['E'] == max_step)]['ErrRate'].iloc[0] 
                    if len(df_mean[(df_mean['N'] == ms) & (df_mean['E'] == max_step)]) > 0 
                    else None for ms in unique_N]
        
        if Y == 'ErrRate':
            y = y_errrate_step_max
        if Y == 'log(ErrRate)':
            y = np.log10(y_errrate_step_max)
        if Y == 'Reward':
            y = 1 - y_errrate_step_max
        
        print (f"----------------x_models: {x_models}, y: {y}")
        # # 绘制step=10时各模型的E值
        # for i, (ms, e_val) in enumerate(zip(x_models, y_step10)):
        #     if e_val is not None:
        plt.scatter(
            x_models, y, label=f"step=max",
            color=color_map[ms], alpha=0.6, s=12, marker='o'
        )

        print("--------------max_step", max_step)
        _y_grid = global_model(popt, x_models, np.log10(np.array([max_step]*len(x_models))))
        print ("-------------_y_grid", _y_grid)
        # TODO log(ErrRate)
        if Y == "log(ErrRate)":
            y_grid = _y_grid
        # TODO ErrRate
        if Y == "ErrRate":
            y_grid = 10**_y_grid
        # TODO Reward
        if Y == "Reward":
            y_grid = 1 - 10**_y_grid
        # # TODO DeltaReward
        # y_grid = 1 - 10**_y_grid - (1 - ErrRate_vals[0])
        
        # plt.plot(x_grid, y_grid - y0, color=color_map[ms], linewidth=2, linestyle='--')
        plt.plot(x_models, y_grid, color=color_map[ms], linewidth=2, linestyle='--')
    
        # # 先画散点
        # for ms in unique_model_sizes:
        #     subdf = df_mean[df_mean['model_size'] == ms].sort_values('E')
        #     E_vals = subdf['E'].to_numpy(dtype=float)
        #     C_vals = subdf['C'].to_numpy(dtype=float)
        #     ErrRate_vals = np.clip(subdf['ErrRate'].to_numpy(dtype=float), 1e-12, None)
        #     # x = np.log10(E_vals)
        #     if X == "logE":
        #         x = np.log10(E_vals)
        #     if X == "E":
        #         print("-------------E_vals", E_vals)
        #         x = E_vals
        #     if X == "C":
        #         x = C_vals

        #     # TODO log(ErrRate)
        #     if Y == "log(ErrRate)":
        #         y = np.log10(ErrRate_vals)
        #     # TODO ErrRate
        #     if Y == "ErrRate":
        #         y = ErrRate_vals
        #     # TODO Reward
        #     if Y == "Reward":
        #         y = 1 - ErrRate_vals
        #     # # TODO DeltaReward
        #     # y = 1 - ErrRate_vals - (1 - ErrRate_vals[0])

        #     y0 = y[0]  # 起始点，用于相对显示
        #     plt.scatter(
        #         # x, y - y0, label=f"{ms}",
        #         x, y, label=f"{ms}",
        #         color=color_map[ms], alpha=0.6, s=12
        #     )
        
        # # 画拟合线
        # for ms in unique_model_sizes:
        #     subdf = df_mean[df_mean['model_size'] == ms]
        #     if len(subdf) < 2:
        #         continue
        #     E_vals = subdf['E'].to_numpy(dtype=float)
        #     C_vals = subdf['C'].to_numpy(dtype=float)
        #     ErrRate_vals = np.clip(subdf['ErrRate'].to_numpy(dtype=float), 1e-12, None)
        #     x = np.log10(E_vals)

        #     # log(ErrRate)
        #     # y = np.log10(ErrRate_vals)
        #     # ErrRate
        #     # y = ErrRate_vals

        #     # Reward

        #     # y0 = y[0]
            
        #     N_val = float(subdf['N'].iloc[0])
        #     k_logistic = logistic_k(N_val, L_fit, r_fit, N0_k_fit)  # 使用Logistic拟合的k(N)关系
            
        #     # # 计算E0值（使用Logistic函数）
        #     # E0_local = logistic_e0(N_val, L_fit, r_e0_fit, N0_e0_fit)
            
        #     # # 计算该模型大小的R²
        #     # y_pred = global_model(popt, np.array([N_val] * len(x)), x)
        #     # ss_res = np.sum((y - y_pred) ** 2)
        #     # ss_tot = np.sum((y - np.mean(y)) ** 2)
        #     # r2_local = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            
        #     # model0_stats.append({
        #     #     'N': N_val, 'k': k_logistic, 'E0': E0_local, 'r2_log': r2_local,
        #     #     'E_min': E_vals.min(), 'E_max': E_vals.max(), 'y0': y0
        #     # })
        #     # print(f"N={N_val:.3g}: k={k_logistic:.6f} (Logistic), E0={E0_local:.6f} (Logistic), R2(log10 ErrRate)={r2_local:.4f}")
            
            # # 画拟合线
            # E_grid = np.logspace(np.log10(E_vals.min()), np.log10(E_vals.max()), 200)
            # # x_grid = np.log10(E_grid)


            # if X == "logE":
            #     x_grid = np.log10(E_vals)
            #     log10_E_grid = x_grid  # 已经是log10(E)
            # if X == "E":
            #     print("-------------E_grid", E_grid)
            #     x_grid = E_grid
            #     log10_E_grid = np.log10(E_grid)  # ✅ 修复：转换为log10(E)给模型
            # if X == "C":
            #     x_grid = E_grid * N_val * phi_global
            #     log10_E_grid = np.log10(E_grid)  # ✅ 修复：转换为log10(E)给模型
            
            # _y_grid = global_model(popt, np.array([N_val] * len(x_grid)), log10_E_grid)
            # # TODO log(ErrRate)
            # if Y == "log(ErrRate)":
            #     y_grid = _y_grid
            # # TODO ErrRate
            # if Y == "ErrRate":
            #     y_grid = 10**_y_grid
            # # TODO Reward
            # if Y == "Reward":
            #     y_grid = 1 - 10**_y_grid
            # # # TODO DeltaReward
            # # y_grid = 1 - 10**_y_grid - (1 - ErrRate_vals[0])
            
            # # plt.plot(x_grid, y_grid - y0, color=color_map[ms], linewidth=2, linestyle='--')
            # plt.plot(x_grid, y_grid, color=color_map[ms], linewidth=2, linestyle='--')
        
    # except Exception as e:
    #     print(f"全局拟合失败: {e}")
    #     model0_stats = []

    # 线性坐标：x=log10(E), y=Δlog10(ErrRate)
    if X == "logE":
        plt.xlabel(r"$\log_{10}E$")
    if X == "E":
        plt.xlabel(r"Data Size")
    if X == "C":
        plt.xlabel(r"Compute")
    if X == "N":
        plt.xlabel(r"Model Size")

    if Y == "log(ErrRate)":
        plt.ylabel(r"Err Rate (log)")
    if Y == "ErrRate":
        plt.ylabel(r"Err Rate")
    if Y == "Reward":
        plt.ylabel(r"Reward")
    if Y == "DeltaReward":
        plt.ylabel(r"DeltaReward")
    
    if X_SCALE:
        plt.xscale(X_SCALE)
    if Y_SCALE:
        plt.yscale(Y_SCALE)
    
    # 更新标题显示完整的双 Logistic 公式（抽象形式）
    title_text = (r"Dual-Logistic Model: $\log_{10}ErrRate = -k(N) \cdot \log_{10}E + E_0(N)$" + "\n" +
                 r"$k(N) = \frac{L}{1 + \exp(-r \cdot (N - N_{0k}))}$" + "\n" +
                 r"$E_0(N) = \frac{L}{1 + \exp(r_{e0} \cdot (N - N_{0e0}))}$")
    # plt.title(title_text, fontsize=11, pad=20)
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
        f"  N₀e0 = {N0_e0_fit:.2e}"
    )
    
    # # 在图的右上角添加信息框
    # plt.text(0.98, 0.98, info_text, transform=plt.gca().transAxes, 
    #          fontsize=8, verticalalignment='top', horizontalalignment='right',
    #          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.legend(title="Model Size", loc="lower left")
    plt.tight_layout()
    out_path0 = OUTPUT_BASE_DIR / f"{FIGURE_PREFIX}_fit_model0.pdf"
    plt.savefig(out_path0, dpi=300, bbox_inches='tight')
    print(f"拟合曲线图已保存到 {out_path0}")

    print(f"\n=== ErrRate 模型拟合完成 ===")
    print("模型：双Logistic拟合 log10(ErrRate) = -k(N) * log10(E) + E0(N)")
    print(f"其中 k(N) = {L_fit:.6f} / (1 + exp(-{r_fit:.2e} * (N - {N0_k_fit:.2e})))")
    print(f"E0(N) = {L_fit:.6f} / (1 + exp({r_e0_fit:.2e} * (N - {N0_e0_fit:.2e})))")
    if model0_stats:
        k_mean = np.mean([s['k'] for s in model0_stats])
        r2_mean = np.mean([s['r2_log'] for s in model0_stats])
        print(f"平均 k ≈ {k_mean:.3f}")
        print(f"平均 R2(log10 ErrRate) ≈ {r2_mean:.3f}")

if __name__ == "__main__":
    main()