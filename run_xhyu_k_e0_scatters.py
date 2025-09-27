#!/usr/bin/env python3
"""
Scaling Law Pipeline - Multi-Eval Analysis
Processes multiple test evals from Experiment1 data and generates scaling law plots for each eval

Usage:
  python run_xhyu_k_e0_scatters.py --model-type base --warmup-clip-num 10
  python run_xhyu_k_e0_scatters.py --model-type instruct --warmup-clip-num 5
"""

import argparse
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

import intrinsic
import data_proc
import plot
import config

# =============================================================================
# CONFIGURATION - Use config constants
# =============================================================================

# Use holdout score evaluation
eval_name = config.DEFAULT_TEST_EVAL
phi_global = 1.0

# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================
def main():
    """Main processing function"""
    global phi_global
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Multi-Eval Scaling Law Analysis')
    parser.add_argument('--model-type', type=str, default='base',
                        choices=['base', 'instruct', 'llama-base', 'llama-instruct', 
                                'exp2-base', 'exp2-instruct', 'grpo-base'],
                        help='Model type to analyze (default: base)')
    parser.add_argument('--warmup-clip-num', type=int, default=10,
                        help='Number of warmup steps to clip (default: 10)')
    args = parser.parse_args()
    
    # Use parsed arguments
    model_type = args.model_type
    warmup_clip_num = args.warmup_clip_num
    
    # Fixed figure sizes
    figure_width = 6.0
    figure_height = 4.0

    print("=== Multi-Eval Scaling Law Analysis ===")
    print(f"Configuration:")
    print(f"  Model type: {model_type}")
    print(f"  Warmup clip num: {warmup_clip_num}")
    print()
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
    # ===========================
    # 只画一个散点图：横轴为Compute（C），纵轴为Error Rate，不同模型大小用不同颜色
    # ===========================
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
    # 使用 config 中的统一颜色映射
    color_map = {}
    # 如果有更多模型大小，按数值 N 排序（确保大模型如 14B 排在最后）
    model_order = (
        df_mean[['model_size', 'N']]
        .drop_duplicates()
        .sort_values('N')
        ['model_size']
        .tolist()
    )
    unique_model_sizes = model_order
    
    # 为每个模型大小分配config中的颜色
    for ms in unique_model_sizes:
        subdf = df_mean[df_mean['model_size'] == ms]
        N_val = float(subdf['N'].iloc[0])
        color_map[ms] = config.get_color_for_curve(N_val)

    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    for ms in unique_model_sizes:
        subdf = df_mean[df_mean['model_size'] == ms]
        plot.plot_basic(
            x=subdf['E'].values,
            y=subdf['ErrRate'].values,
            use_scatter=True,
            scatter_alpha=0.7,
            scatter_s=12,
            scatter_marker='o',
            color=color_map[ms],
            ax=ax
        )
        # Add manual legend entry
        ax.scatter([], [], color=color_map[ms], s=12, alpha=0.7, label=f"{ms}")
    
    plot.plot_basic_settings(
        ax=ax,
        x_scale='log',
        y_scale='log',
        x_label="Training Examples E (log)",
        y_label="Error Rate",
        title="Error Rate vs Training Examples",
        use_legend=True,
        legend_loc="best"
    )
    ax.legend(title="model size", loc="best")
    plt.tight_layout()
    plt.savefig(config.OUTPUT_BASE_DIR / f"{config.DEFAULT_FIGURE_PREFIX}_scatter_errrate_vs_E.pdf", dpi=300, bbox_inches='tight')
    print(f"散点图已保存到 {config.OUTPUT_BASE_DIR / f'{config.DEFAULT_FIGURE_PREFIX}_scatter_errrate_vs_E.pdf'}")

    # ===========================
    # 新增：ImprovementRate vs Training Examples 散点图
    # ===========================
    fig2, ax2 = plt.subplots(figsize=(figure_width, figure_height))
    for ms in unique_model_sizes:
        subdf = df_mean[df_mean['model_size'] == ms]
        plot.plot_basic(
            x=subdf['E'].values,
            y=-np.log(subdf['ImprovementRate']).values,
            use_scatter=True,
            scatter_alpha=0.7,
            scatter_s=12,
            scatter_marker='o',
            color=color_map[ms],
            ax=ax2
        )
        # Add manual legend entry
        ax2.scatter([], [], color=color_map[ms], s=12, alpha=0.7, label=f"{ms}")
    
    plot.plot_basic_settings(
        ax=ax2,
        x_scale='log',
        # 注意：y轴不再是log scale，因为我们已经取了-log
        x_label="Training Examples E (log)",
        y_label="-log(Improvement Rate)",
        title="-log(Improvement Rate) vs Training Examples",
        use_legend=True,
        legend_loc="best"
    )
    ax2.legend(title="model size", loc="best")
    plt.tight_layout()
    plt.savefig(config.OUTPUT_BASE_DIR / f"{config.DEFAULT_FIGURE_PREFIX}_scatter_improvementrate_vs_E.pdf", dpi=300, bbox_inches='tight')
    print(f"改进率散点图已保存到 {config.OUTPUT_BASE_DIR / f'{config.DEFAULT_FIGURE_PREFIX}_scatter_improvementrate_vs_E.pdf'}")

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
        plt.figure(figsize=(figure_width, figure_height))
        
        # 先画散点
        for ms in unique_model_sizes:
            subdf = df_mean[df_mean['model_size'] == ms].sort_values('E')
            E_vals = subdf['E'].to_numpy(dtype=float)
            ErrRate_vals = np.clip(subdf['ErrRate'].to_numpy(dtype=float), 1e-12, None)
            x = np.log10(E_vals)
            y = np.log10(ErrRate_vals)
            y0 = y[0]  # 起始点，用于相对显示
            plot.plot_basic(
                x=x,
                y=y,
                use_scatter=True,
                scatter_alpha=0.6,
                scatter_s=12,
                scatter_marker='o',
                color=color_map[ms],
                ax=plt.gca()
            )
            # Add manual legend entry
            plt.scatter([], [], color=color_map[ms], s=12, alpha=0.6, label=f"{ms}")
        
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
            plot.plot_basic(
                x=x_grid,
                y=y_grid,
                use_line=True,
                line_alpha=1.0,
                line_width=2,
                color='black',
                ax=plt.gca()
            )
        
    except Exception as e:
        print(f"全局拟合失败: {e}")
        model0_stats = []

    # 使用plot_basic_settings设置坐标轴
    plot.plot_basic_settings(
        ax=plt.gca(),
        x_label=r"$\log_{10}E$",
        y_label=r"$\log_{10}ErrRate$",
        title=r"Model0 Fitting: $\log_{10}ErrRate = -(a N + b) \log_{10}E + E_0$",
        use_legend=True,
        legend_loc="best"
    )
    plt.gca().legend(title="model size", loc="best")
    plt.tight_layout()
    out_path0 = config.OUTPUT_BASE_DIR / f"{config.DEFAULT_FIGURE_PREFIX}_fit_model0.pdf"
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
    print(f"\n=== 测试L(N,D)和L(N,C)拟合的k(N)和E0(N)关系 ===")
    
    # 对每个N分别拟合 log10(ErrRate) = k * log10(X) + E0，其中X为D或C
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
        
        fit_results_E[N_val] = {'k': k_E, 'E0': E0_E, 'r2': r2_E, 'n_points': len(log_E)}
        fit_results_C[N_val] = {'k': k_C, 'E0': E0_C, 'r2': r2_C, 'n_points': len(log_C)}
        
        print(f"N={N_val/1e9:>4.1f}B: k_D={k_E:>8.4f}, E0_D={E0_E:>8.4f}, R²_D={r2_E:>6.4f}")
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figure_width * 1.67, figure_height))
    
    # k(N)散点图 - 使用config配色和plot_basic
    plot.plot_basic(
        x=np.array(N_values),
        y=np.array(k_E_values),
        use_scatter=True,
        scatter_alpha=0.7,
        scatter_s=100,
        scatter_marker='o',
        color=config.COLOR_MAPPING['base'],
        ax=ax1
    )
    plot.plot_basic(
        x=np.array(N_values),
        y=np.array(k_C_values),
        use_scatter=True,
        scatter_alpha=0.7,
        scatter_s=100,
        scatter_marker='o',
        color=config.COLOR_MAPPING['instruct'],
        ax=ax1
    )
    # Add manual legend entries
    ax1.scatter([], [], color=config.COLOR_MAPPING['base'], s=100, alpha=0.7, label='L(N,D) fitting')
    ax1.scatter([], [], color=config.COLOR_MAPPING['instruct'], s=100, alpha=0.7, label='L(N,C) fitting')
    
    plot.plot_basic_settings(
        ax=ax1,
        x_scale='log',
        x_label='Model Size (N)',
        y_label=' $k(N)$',
        x_tick_on_data=True,
        title=f'$k(N)$ Comparison on {model_type.capitalize()} Models',
        use_legend=True
    )
    ax1.grid(True, alpha=0.3)
    
    # E0(N)散点图 - 使用config配色和plot_basic
    plot.plot_basic(
        x=np.array(N_values),
        y=np.array(E0_E_values),
        use_scatter=True,
        scatter_alpha=0.7,
        scatter_s=100,
        scatter_marker='o',
        color=config.COLOR_MAPPING['base'],
        ax=ax2
    )
    plot.plot_basic(
        x=np.array(N_values),
        y=np.array(E0_C_values),
        use_scatter=True,
        scatter_alpha=0.7,
        scatter_s=100,
        scatter_marker='o',
        color=config.COLOR_MAPPING['instruct'],
        ax=ax2
    )
    # Add manual legend entries
    ax2.scatter([], [], color=config.COLOR_MAPPING['base'], s=100, alpha=0.7, label='L(N,D) fitting')
    ax2.scatter([], [], color=config.COLOR_MAPPING['instruct'], s=100, alpha=0.7, label='L(N,C) fitting')
    
    plot.plot_basic_settings(
        ax=ax2,
        x_scale='log',
        x_label='Model Size (N)',
        y_label=' $E(N)$',
        x_tick_on_data=True,
        title=f'$E(N)$ Comparison on {model_type.capitalize()} Models',
        use_legend=True
    )
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    comparison_path = config.OUTPUT_BASE_DIR / f"fit_{model_type}_{config.DEFAULT_FIGURE_PREFIX}_k_E0_comparison.pdf"
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"k(N)和E0(N)对比图已保存到 {comparison_path}")

if __name__ == "__main__":
    main()