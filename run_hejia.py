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
WARMUP_CLIPPING_NUM = 5 


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
    # 丢掉每个 (model_size, runid) 的前 WARMUP_CLIPPING_NUM 个点
    if WARMUP_CLIPPING_NUM and WARMUP_CLIPPING_NUM > 0:
        df = (
            df.groupby(['model_size', 'runid'], as_index=False, group_keys=False)
              .apply(lambda g: g.iloc[WARMUP_CLIPPING_NUM:])
              .reset_index(drop=True)
        )
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
    # 对相同横坐标（同一 model_size 与 step → 同一 C）聚合：只显示三个纵坐标（不同 run）的平均值
    df_mean = (
        df.groupby(['model_size', 'step'], as_index=False)
          .agg(N=('N', 'first'), C=('C', 'first'), ErrRate=('ErrRate', 'mean'))
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
            subdf['C'], 
            subdf['ErrRate'], 
            # 直接用小写b显示（如1.5b），不区分大小
            label=f"{ms}",
            color=color_map[ms], 
            alpha=0.7, 
            s=12
        )
    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel("Compute C (FLOPs, log)")
    plt.ylabel("Error Rate")
    plt.title("Error Rate vs Compute")
    plt.legend(title="model size", loc="best")
    plt.tight_layout()
    plt.savefig(OUTPUT_BASE_DIR / f"{FIGURE_PREFIX}_scatter_errrate_vs_C.pdf", dpi=300, bbox_inches='tight')
    print(f"散点图已保存到 {OUTPUT_BASE_DIR / f'{FIGURE_PREFIX}_scatter_errrate_vs_C.pdf'}")

    # ===========================
    # 拟合三种模型并绘图
    # 变量命名：E=ErrRate, N=model_size, C=Compute
    # ===========================
    N_all = df_mean['N'].to_numpy(dtype=float)
    C_all = df_mean['C'].to_numpy(dtype=float)
    E_all = df_mean['ErrRate'].to_numpy(dtype=float)

    # 安全处理：避免 log(0)
    eps = 1e-12
    E_all_clipped = np.clip(E_all, eps, 1.0)

    # 实用函数：计算 R^2（线性空间）与 R^2（log10 空间）
    def compute_r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    def compute_r2_log10(y_true, y_pred):
        y_true_log = np.log10(np.clip(y_true, eps, None))
        y_pred_log = np.log10(np.clip(y_pred, eps, None))
        ss_res = np.sum((y_true_log - y_pred_log) ** 2)
        ss_tot = np.sum((y_true_log - np.mean(y_true_log)) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # 颜色与线型
    linestyles = ['-', '--', '-.']

    # 工具：针对每个 N 画一条拟合曲线（随 C 变化）
    def plot_model_curves(model_func, params, title_suffix, filename_suffix):
        plt.figure(figsize=(7, 5))
        # 散点
        for ms in unique_model_sizes:
            subdf = df_mean[df_mean['model_size'] == ms]
            plt.scatter(
                subdf['C'], subdf['ErrRate'], label=f"{ms}",
                color=color_map[ms], alpha=0.6, s=12
            )
        # 拟合线（每个 N 一条）
        for idx, ms in enumerate(unique_model_sizes):
            subdf = df_mean[df_mean['model_size'] == ms]
            if len(subdf) < 2:
                continue
            C_min = subdf['C'].min()
            C_max = subdf['C'].max()
            C_grid = np.logspace(np.log10(C_min), np.log10(C_max), 200)
            N_val = float(subdf['N'].iloc[0])
            N_grid = np.full_like(C_grid, N_val)
            E_pred = model_func((N_grid, C_grid), *params)
            plt.plot(C_grid, E_pred, color=color_map[ms], linewidth=2)

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Compute C (FLOPs, log)")
        plt.ylabel("Error Rate")
        plt.title(f"Error Rate vs Compute - {title_suffix}")
        plt.legend(title="model size", loc="best")
        plt.tight_layout()
        out_path = OUTPUT_BASE_DIR / f"{FIGURE_PREFIX}_fit_{filename_suffix}.pdf"
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"拟合曲线图已保存到 {out_path}")

    # ---------------------------
    # 模型 1: E = [(N0/N)^(an/ac) + Co/C]^(ac)
    # 参数: N0, an, ac, Co
    # ---------------------------
    def model1(X, N0, a_n, a_c, C0):
        N, C = X
        term = np.power((N0 / N), (a_n / a_c)) + (C0 / C)
        return np.power(np.clip(term, eps, None), a_c)

    # 初值与边界
    N0_init = np.median(N_all)
    C0_init = np.median(C_all)
    p0_1 = [N0_init, 0.5, 0.5, C0_init]
    bounds_1 = ([1e6, 1e-6, 1e-6, 1e12], [1e13, 5.0, 5.0, 1e26])
    try:
        popt1, _ = curve_fit(model1, (N_all, C_all), E_all_clipped, p0=p0_1, bounds=bounds_1, maxfev=200000)
        E_pred1 = model1((N_all, C_all), *popt1)
        r2_1 = compute_r2(E_all, E_pred1)
        r2log_1 = compute_r2_log10(E_all, E_pred1)
        print(f"模型1参数: N0={popt1[0]:.3g}, a_n={popt1[1]:.3g}, a_c={popt1[2]:.3g}, C0={popt1[3]:.3g}")
        print(f"模型1拟合度: R2(E)={r2_1:.4f}, R2(log10 E)={r2log_1:.4f}")
        plot_model_curves(model1, popt1, "Model 1", "model1")
    except Exception as e:
        print(f"模型1拟合失败: {e}")

    # ---------------------------
    # 模型 2: E = A/N^a + B/C^b + const
    # 参数: A, a, B, b, const
    # ---------------------------
    def model2(X, A, a, B, b, const):
        N, C = X
        return A / np.power(N, a) + B / np.power(C, b) + const

    p0_2 = [0.1, 0.5, 0.1, 0.5, 1e-3]
    bounds_2 = ([1e-12, 1e-6, 1e-12, 1e-6, 0.0], [1.0, 5.0, 1.0, 5.0, 0.5])
    try:
        popt2, _ = curve_fit(model2, (N_all, C_all), E_all_clipped, p0=p0_2, bounds=bounds_2, maxfev=200000)
        E_pred2 = model2((N_all, C_all), *popt2)
        r2_2 = compute_r2(E_all, E_pred2)
        r2log_2 = compute_r2_log10(E_all, E_pred2)
        print(f"模型2参数: A={popt2[0]:.3g}, a={popt2[1]:.3g}, B={popt2[2]:.3g}, b={popt2[3]:.3g}, const={popt2[4]:.3g}")
        print(f"模型2拟合度: R2(E)={r2_2:.4f}, R2(log10 E)={r2log_2:.4f}")
        plot_model_curves(model2, popt2, "Model 2", "model2")
    except Exception as e:
        print(f"模型2拟合失败: {e}")

    # ---------------------------
    # 模型 3: E^beta = (N0/N)^(an) + (Co/C)^(ac)
    # 参数: beta, N0, an, Co, ac
    # ---------------------------
    def model3_raw(X, beta, N0, a_n, C0, a_c):
        N, C = X
        return np.power((N0 / N), a_n) + np.power((C0 / C), a_c)

    def model3(X, beta, N0, a_n, C0, a_c):
        return np.power(np.clip(model3_raw(X, beta, N0, a_n, C0, a_c), eps, None), 1.0 / beta)

    p0_3 = [1.0, N0_init, 0.5, C0_init, 0.5]
    bounds_3 = ([0.2, 1e6, 1e-6, 1e12, 1e-6], [5.0, 1e13, 5.0, 1e26, 5.0])
    try:
        popt3, _ = curve_fit(model3, (N_all, C_all), E_all_clipped, p0=p0_3, bounds=bounds_3, maxfev=200000)
        E_pred3 = model3((N_all, C_all), *popt3)
        r2_3 = compute_r2(E_all, E_pred3)
        r2log_3 = compute_r2_log10(E_all, E_pred3)
        print(f"模型3参数: beta={popt3[0]:.3g}, N0={popt3[1]:.3g}, a_n={popt3[2]:.3g}, C0={popt3[3]:.3g}, a_c={popt3[4]:.3g}")
        print(f"模型3拟合度: R2(E)={r2_3:.4f}, R2(log10 E)={r2log_3:.4f}")
        plot_model_curves(model3, popt3, "Model 3", "model3")
    except Exception as e:
        print(f"模型3拟合失败: {e}")

    # ---------------------------
    # 模型 0（最简单）：对每个 N，拟合 log10(E) = -k * log10(C) + C0
    # 保存每个 N 的 k、C0、R2_log10，并单独绘图（保留原输出）
    # ---------------------------
    print("\n=== 模型0：按 N 的对数线性拟合 log10(E) = -k * log10(C) + C0 ===")
    model0_stats = []  # 列表元素: dict(N, k, C0, r2_log, m, b)
    plt.figure(figsize=(7, 5))
    # 先画散点（模型0：使用 x=log10(C)，y=log10(E)-log10(E0) 使起点为0）
    for ms in unique_model_sizes:
        subdf = df_mean[df_mean['model_size'] == ms].sort_values('C')
        C_vals = subdf['C'].to_numpy(dtype=float)
        E_vals = np.clip(subdf['ErrRate'].to_numpy(dtype=float), 1e-12, None)
        x = np.log10(C_vals)
        y = np.log10(E_vals)
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
        C_vals = subdf['C'].to_numpy(dtype=float)
        E_vals = np.clip(subdf['ErrRate'].to_numpy(dtype=float), 1e-12, None)
        x = np.log10(C_vals)
        y = np.log10(E_vals)
        y0 = y[0]
        # 线性回归：y = m x + b  =>  k = -m, C0 = b
        m, b = np.polyfit(x, y, deg=1)
        y_pred = m * x + b
        # R2 in log space
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2_log = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        k = -m
        N_val = float(subdf['N'].iloc[0])
        model0_stats.append({
            'N': N_val, 'k': k, 'C0': b, 'r2_log': r2_log, 'm': m, 'b': b,
            'C_min': C_vals.min(), 'C_max': C_vals.max(), 'y0': y0
        })
        print(f"N={N_val:.3g}: k={k:.4f}, C0={b:.4f}, R2(log10 E)={r2_log:.4f}")
        # 画拟合线
        C_grid = np.logspace(np.log10(C_vals.min()), np.log10(C_vals.max()), 200)
        x_grid = np.log10(C_grid)
        y_grid = m * x_grid + b
        plt.plot(x_grid, y_grid - y0, color=color_map[ms], linewidth=2)

    # 线性坐标：x=log10(C), y=Δlog10(E)
    plt.xlabel(r"$\log_{10}C$")
    plt.ylabel(r"$\Delta\log_{10}E$")
    plt.title(r"$\log_{10}E = -k(N)\,\log_{10}C + c_0(N)$")
    plt.legend(title="model size", loc="best")
    plt.tight_layout()
    out_path0 = OUTPUT_BASE_DIR / f"{FIGURE_PREFIX}_fit_model0.pdf"
    plt.savefig(out_path0, dpi=300, bbox_inches='tight')
    print(f"拟合曲线图已保存到 {out_path0}")

    # ---------------------------
    # 组合图：4个子图（模型0-3）
    # ---------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    def draw_scatter(ax):
        for ms in unique_model_sizes:
            subdf = df_mean[df_mean['model_size'] == ms]
            ax.scatter(
                subdf['C'], subdf['ErrRate'], label=f"{ms}",
                color=color_map[ms], alpha=0.6, s=12
            )
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("Compute C (FLOPs, log)")
        ax.set_ylabel("Error Rate")

    # 子图0：模型0
    ax0 = axes[0]
    # 模型0子图：x=log10(C), y=log10(E)-y0
    handles0 = []
    labels0 = []
    for ms in unique_model_sizes:
        subdf = df_mean[df_mean['model_size'] == ms].sort_values('C')
        if len(subdf) < 2:
            continue
        C_vals = subdf['C'].to_numpy(dtype=float)
        E_vals = np.clip(subdf['ErrRate'].to_numpy(dtype=float), 1e-12, None)
        x = np.log10(C_vals)
        y = np.log10(E_vals)
        y0 = y[0]
        sc = ax0.scatter(x, y - y0, label=f"{ms}", color=color_map[ms], alpha=0.6, s=12)
        handles0.append(sc)
        labels0.append(f"{ms}")
    for st in model0_stats:
        C_grid = np.logspace(np.log10(st['C_min']), np.log10(st['C_max']), 200)
        x_grid = np.log10(C_grid)
        y_grid = (-st['k']) * x_grid + st['C0'] - st['y0']
        # 找颜色：按 N 匹配 model_size
        color = None
        for ms in unique_model_sizes:
            subdf = df_mean[df_mean['model_size'] == ms]
            if len(subdf) and abs(float(subdf['N'].iloc[0]) - st['N']) < 1e-6:
                color = color_map[ms]
                break
        ax0.plot(x_grid, y_grid, color=color if color else 'k', linewidth=2)
    ax0.set_xlabel(r"$\log_{10}C$")
    ax0.set_ylabel(r"$\Delta\log_{10}E$")
    # 文本：用均值展示
    if model0_stats:
        k_mean = np.mean([s['k'] for s in model0_stats])
        r2_mean = np.mean([s['r2_log'] for s in model0_stats])
        ax0.set_title(r"$\log_{10}E = -k(N)\,\log_{10}C + c_0(N)$")
        ax0.text(0.97, 0.97,
                 f"k≈{k_mean:.3f}\nR2(log10 E)≈{r2_mean:.3f}",
                 transform=ax0.transAxes, ha='right', va='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, lw=0))

    # 子图1：模型1
    ax1 = axes[1]
    draw_scatter(ax1)
    try:
        E_pred1  # noqa: check existence
        for ms in unique_model_sizes:
            subdf = df_mean[df_mean['model_size'] == ms]
            if len(subdf) < 2:
                continue
            C_min = subdf['C'].min(); C_max = subdf['C'].max()
            C_grid = np.logspace(np.log10(C_min), np.log10(C_max), 200)
            N_val = float(subdf['N'].iloc[0])
            N_grid = np.full_like(C_grid, N_val)
            E_grid = model1((N_grid, C_grid), *popt1)
            ax1.plot(C_grid, E_grid, color=color_map[ms], linewidth=2)
        ax1.set_title(r"$E=[(N_0/N)^{a_n/a_c} + C_0/C]^{a_c}$")
        ax1.text(0.97, 0.97,
                 f"N0={popt1[0]:.2g}, a_n={popt1[1]:.2g}\na_c={popt1[2]:.2g}, C0={popt1[3]:.2g}\nR2={r2_1:.3f}, R2log={r2log_1:.3f}",
                 transform=ax1.transAxes, ha='right', va='top', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, lw=0))
    except Exception:
        ax1.set_title("Model 1: 拟合失败")

    # 子图2：模型2
    ax2 = axes[2]
    draw_scatter(ax2)
    try:
        E_pred2
        for ms in unique_model_sizes:
            subdf = df_mean[df_mean['model_size'] == ms]
            if len(subdf) < 2:
                continue
            C_min = subdf['C'].min(); C_max = subdf['C'].max()
            C_grid = np.logspace(np.log10(C_min), np.log10(C_max), 200)
            N_val = float(subdf['N'].iloc[0])
            N_grid = np.full_like(C_grid, N_val)
            E_grid = model2((N_grid, C_grid), *popt2)
            ax2.plot(C_grid, E_grid, color=color_map[ms], linewidth=2)
        ax2.set_title(r"$E = A N^{-a} + B C^{-b} + c$")
        ax2.text(0.97, 0.97,
                 f"A={popt2[0]:.2g}, a={popt2[1]:.2g}\nB={popt2[2]:.2g}, b={popt2[3]:.2g}\nconst={popt2[4]:.2g}\nR2={r2_2:.3f}, R2log={r2log_2:.3f}",
                 transform=ax2.transAxes, ha='right', va='top', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, lw=0))
    except Exception:
        ax2.set_title("Model 2: 拟合失败")

    # 子图3：模型3
    ax3 = axes[3]
    draw_scatter(ax3)
    try:
        E_pred3
        for ms in unique_model_sizes:
            subdf = df_mean[df_mean['model_size'] == ms]
            if len(subdf) < 2:
                continue
            C_min = subdf['C'].min(); C_max = subdf['C'].max()
            C_grid = np.logspace(np.log10(C_min), np.log10(C_max), 200)
            N_val = float(subdf['N'].iloc[0])
            N_grid = np.full_like(C_grid, N_val)
            E_grid = model3((N_grid, C_grid), *popt3)
            ax3.plot(C_grid, E_grid, color=color_map[ms], linewidth=2)
        ax3.set_title(r"$E^{\beta} = (N_0/N)^{a_n} + (C_0/C)^{a_c}$")
        ax3.text(0.97, 0.97,
                 f"β={popt3[0]:.2g}, N0={popt3[1]:.2g}\na_n={popt3[2]:.2g}, C0={popt3[3]:.2g}\na_c={popt3[4]:.2g}\nR2={r2_3:.3f}, R2log={r2log_3:.3f}",
                 transform=ax3.transAxes, ha='right', va='top', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, lw=0))
    except Exception:
        ax3.set_title("Model 3: 拟合失败")

    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(handles, labels, title="model size", loc='lower center', ncol=len(labels))
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    out_path_all = OUTPUT_BASE_DIR / f"{FIGURE_PREFIX}_fits_all.pdf"
    fig.savefig(out_path_all, dpi=300, bbox_inches='tight')
    print(f"组合拟合图已保存到 {out_path_all}")

if __name__ == "__main__":
    main()