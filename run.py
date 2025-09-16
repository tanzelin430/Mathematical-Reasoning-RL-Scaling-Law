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
WARMUP_CLIPPING_FACTOR = 1/64 # warmup clipping: Important even for LLM RL
# WARMUP_CLIPPING_FACTOR = 1
# WARMUP_CLIPPING_FACTOR = 0

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

# Metric name formatting function
def format_metric_name(metric_name):
    """
    Format metric name for folder and label display:
    - Remove 'val/test_score/' prefix
    - Remove '/unknown' suffix  
    - Replace '/' with '-' in the middle
    """
    # Remove prefixes and suffixes
    name = metric_name.replace('val/test_score/', '').replace('val/', '').replace('/unknown', '')
    name = name.replace('merged_deduped_dapo_or1_dataset', 'merged_deduped_dapo')
    # Replace '/' with '-' for folder names
    folder_name = name
    # folder_name = name.replace('/', '-')
    # Keep original for display labels
    display_name = name
    return folder_name, display_name

# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def process_single_metric(df, plot_x_column: str, metric_name, curve_column: str, output_dir, axes: dict[str, plt.Axes]):
    """Process a single test metric and generate all scaling law plots"""
    
    print(f"\n{'='*60}")
    print(f"Processing metric: {metric_name}")
    print(f"{'='*60}")
    
    # Format metric names
    folder_name, display_name = format_metric_name(metric_name)
    
    # Create metric-specific output directory
    metric_output_dir = Path(output_dir) / folder_name
    metric_output_dir.mkdir(parents=True, exist_ok=True)
    
    # =============================================================================
    # PHASE 1: DATA PREPROCESSING
    # =============================================================================
    
    df = df.rename(columns={metric_name: 'R'})
    # Calculate error rate
    df['ErrRate'] = 1 - df['R']
    df['Improve'] = data_proc.calc_improve(df, 'R', curve_column)
    # Calculate std
    R_std = df.groupby([curve_column, 'step'])['R'].std().to_frame('R_std')
    ErrRate_std = df.groupby([curve_column, 'step'])['ErrRate'].std().to_frame('ErrRate_std')
    Improve_std = df.groupby([curve_column, 'step'])['Improve'].std().to_frame('Improve_std')

    # Merge multi rollout in same step
    df = data_proc.merge_duplicate_steps(df, group_columns=[curve_column, 'step'], mode='mean')

    df = df.merge(R_std, on=[curve_column, 'step'])
    df = df.merge(ErrRate_std, on=[curve_column, 'step'])
    df = df.merge(Improve_std, on=[curve_column, 'step'])

    df.sort_values(plot_x_column, inplace=True)

    # Validate phi parameter estimation
    ax, stats, kappa_hat = plot.plot_phi_over_steps(df, sample_size_per_step=SAMPLE_SIZE_PER_STEP, tail_fraction=0.25)
    plt.tight_layout()
    plt.savefig(metric_output_dir / "phi.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    print("Global kappa (tail median):", kappa_hat)
    print("Expected phi_global:", phi_global if phi_global is not None else "N/A")
    print(stats.head())

    # runs_raw_dfs = data_proc.sort_dfs(runs_raw_dfs)

    # monotonic smoothing
    df_R_smooth = data_proc.smooth_df(
        df, 
        curve_column=curve_column,
        col_x=plot_x_column, 
        col_y='R', 
        col_y_out='R_smooth',
        # monotonic=True,
        monotonic=False,
        increasing=None, # auto determine
        # strict=True,
        strict=False,
        s_factor=1, 
        k_spline=5,
        rolling_window=200, 
        min_se=1e-6, 
        x_inv_weight_power=0.3
    )
    
    df_R_smooth = data_proc.apply_warmup_clipping(df_R_smooth, curve_column=curve_column, warmup_frac=WARMUP_CLIPPING_FACTOR)
    
    # =============================================================================
    # FIGURE 1A: REWARD CURVES (C vs R)
    # =============================================================================
    
    # Generate Figure 1a with both smoothed curves and raw data points
    ax = plot.plot_generic_curve(
        df,
        curve_column=curve_column,
        x_column=plot_x_column,
        y_column="R",
        df_smooth=df_R_smooth,
        y_smooth_column="R_smooth",
        # x_column='R_std',
        y_std_column='R_std',
        # xlabel="Compute C (FLOPs, log)",
        # ylabel=f"Return R ({display_name})",
        title=f"{display_name}",
        ax=axes["score-norm"]
    )
    # plt.tight_layout()
    # plt.savefig(metric_output_dir / "figure_1a.png", dpi=300, bbox_inches='tight')
    # plt.close()
    
    # ============================
    # Error Rate
    # ============================
    # monotonic smoothing
    df_ErrRate_smooth = data_proc.smooth_df(
        df, 
        curve_column=curve_column,
        col_x=plot_x_column, 
        col_y='ErrRate', 
        col_y_out='ErrRate_smooth',
        # monotonic=True,
        monotonic=False,
        increasing=None, # auto determine
        # strict=True,
        s_factor=1, 
        k_spline=5,
        rolling_window=200, 
        min_se=1e-6, 
        x_inv_weight_power=0.3
    )
    df_ErrRate_smooth = data_proc.apply_warmup_clipping(df_ErrRate_smooth, curve_column=curve_column, warmup_frac=WARMUP_CLIPPING_FACTOR)
    
    ax = plot.plot_generic_curve(
        df,
        curve_column=curve_column,
        x_column=plot_x_column,
        y_column="ErrRate",
        df_smooth=df_ErrRate_smooth,
        y_smooth_column="ErrRate_smooth",
        y_std_column='ErrRate_std',
        # xlabel="Compute C (FLOPs, log)",
        # ylabel=f"1-R({display_name})",
        title=f"{display_name}",
        ax=axes["err-rate"]
    )
    # plt.tight_layout()
    # plt.savefig(metric_output_dir / "figure_1c.png", dpi=300, bbox_inches='tight')
    # plt.close()

    # ============================
    # Improve Rate: 相对于step==0时的R的改进率
    # ============================

    df_Improve_smooth = data_proc.smooth_df(
        df, 
        curve_column=curve_column,
        col_x=plot_x_column, 
        col_y='Improve', 
        col_y_out='Improve_smooth',
        # monotonic=True,
        monotonic=False,
        increasing=None, # auto determine
        # strict=True,
        s_factor=1, 
        k_spline=5,
        rolling_window=200, 
        min_se=1e-6, 
        x_inv_weight_power=0.3
    )
    df_Improve_smooth = data_proc.apply_warmup_clipping(df_Improve_smooth, curve_column=curve_column, warmup_frac=WARMUP_CLIPPING_FACTOR)
    # df_Improve_smooth = df_R_smooth
    # df_Improve_smooth['Improve_smooth'] = data_proc.calc_improve(df_Improve_smooth, 'R_smooth')
    
    ax = plot.plot_generic_curve(
        df,
        curve_column=curve_column,
        x_column=plot_x_column,
        y_column="Improve",
        df_smooth=df_Improve_smooth,
        y_smooth_column="Improve_smooth",
        # y_std_column="Improve_std",
        # xlabel="Compute C (FLOPs, log)",
        # ylabel=f"1-R({display_name})",
        title=f"{display_name}",
        ax=axes["improve-rate"]
    )
    
def process_single_metric_intrinsic(df, plot_x_column: str, metric_name, curve_column: str, output_dir, axes: dict[str, plt.Axes]):
    
    assert(plot_x_column == "C") # only support 'C'
    
    # Format metric names
    folder_name, display_name = format_metric_name(metric_name)
    
    # Create metric-specific output directory
    metric_output_dir = Path(output_dir) / folder_name
    metric_output_dir.mkdir(parents=True, exist_ok=True)
    
    # =============================================================================
    # PHASE 1: DATA PREPROCESSING
    # =============================================================================
    
    df = df.rename(columns={metric_name: 'R'})
    # # Calculate error rate
    # df['ErrRate'] = 1 - df['R']
    # df['Improve'] = data_proc.calc_improve(df, 'R', curve_column)
    # # Calculate std
    # R_std = df.groupby([curve_column, 'step'])['R'].std().to_frame('R_std')
    # ErrRate_std = df.groupby([curve_column, 'step'])['ErrRate'].std().to_frame('ErrRate_std')
    # Improve_std = df.groupby([curve_column, 'step'])['Improve'].std().to_frame('Improve_std')

    # Merge multi rollout in same step
    df = data_proc.merge_duplicate_steps(df, group_columns=[curve_column, 'step'], mode='mean')

    # df = df.merge(R_std, on=[curve_column, 'step'])
    # df = df.merge(ErrRate_std, on=[curve_column, 'step'])
    # df = df.merge(Improve_std, on=[curve_column, 'step'])

    df.sort_values(plot_x_column, inplace=True)

    # Validate phi parameter estimation
    ax, stats, kappa_hat = plot.plot_phi_over_steps(df, sample_size_per_step=SAMPLE_SIZE_PER_STEP, tail_fraction=0.25)
    plt.tight_layout()
    plt.savefig(metric_output_dir / "phi.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    print("Global kappa (tail median):", kappa_hat)
    print("Expected phi_global:", phi_global if phi_global is not None else "N/A")
    print(stats.head())

    # runs_raw_dfs = data_proc.sort_dfs(runs_raw_dfs)

    df = data_proc.apply_warmup_clipping(df, curve_column=curve_column, warmup_frac=WARMUP_CLIPPING_FACTOR)
    # monotonic smoothing
    df_R_smooth = data_proc.smooth_df(
        df, 
        curve_column=curve_column,
        col_x=plot_x_column, 
        col_y='R', 
        col_y_out='R_smooth',
        # monotonic=True,
        monotonic=False,
        increasing=None, # auto determine
        # strict=True,
        strict=False,
        s_factor=1, 
        k_spline=5,
        rolling_window=200, 
        min_se=1e-6, 
        x_inv_weight_power=0.3
    )
    
    df_R_smooth = data_proc.apply_warmup_clipping(df_R_smooth, curve_column=curve_column, warmup_frac=WARMUP_CLIPPING_FACTOR)
    
    # =============================================================================
    # PHASE 3: INTRINSIC TABLE
    # =============================================================================
    
    # Build intrinsic capability table I(R)
    I_of_R = intrinsic.build_intrinsic_table(
        df_R_smooth if BUILD_I_ON_SMOOTHED else df, 
        column_R= "R_smooth" if BUILD_I_ON_SMOOTHED else "R"
    )
    I_interp = intrinsic.make_I_interpolator(
        I_of_R, 
        # mode="step"
        mode="loglinear"
    )

    # Visualize the R->I mapping relationship
    plot.vplot_empirical_f_of_R(
        df=df_R_smooth if BUILD_I_ON_SMOOTHED else df,
        I_of_R=I_of_R,
        use_smooth=BUILD_I_ON_SMOOTHED,
        label_by="N",        # or "runid" / None
        legend_max=12,
        alpha_runs=0.1
    )
    plt.tight_layout()
    plt.savefig(metric_output_dir / "f_of_R_empirical.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    # =============================================================================
    # FIGURE 1B: INTRINSIC PERFORMANCE CURVES (C vs I(R))
    # =============================================================================
    
    # Map all data points to intrinsic coordinates
    df_Imap = intrinsic.map_points_to_intrinsic(
        df_R_smooth if BUILD_I_ON_SMOOTHED else df, 
        I_interp, 
        R_column="R_smooth" if BUILD_I_ON_SMOOTHED else "R", 
    )

    # Validate the mapping (sanity check)
    # assert np.all(intrinsic_points["I_map"] <= intrinsic_points["C"] + 1e-9), f"I > C violation: max(I-C) = {(intrinsic_points['I_map'] - intrinsic_points['C']).max():.2e}"

    # # Optional: Smooth the I_map curve
    # df_Imap = data_proc.smooth_df(
    #     df_Imap,
    #     curve_column=curve_column,
    #     col_x=plot_x_column,
    #     col_y="I_map",
    #     col_y_out="I_map_smooth",
    #     monotonic=True,
    #     increasing=None, # auto determine
    #     strict=True,
    #     s_factor=0.3, 
    #     k_spline=3,
    #     rolling_window=50, 
    #     min_se=1e-4, 
    #     x_inv_weight_power=10
    # )

    # Optional: Smooth the I_map curve
    # df_Imap = data_proc.smooth_df(
    #     df_Imap,
    #     curve_column=curve_column,
    #     col_x=plot_x_column,
    #     col_y="I_map",
    #     col_y_out="I_map_smooth",
    #     monotonic=True,
    #     increasing=None, # auto determine
    #     strict=True,
    #     s_factor=0.1, 
    #     k_spline=3,
    #     rolling_window=200, 
    #     min_se=1e-6, 
    #     x_inv_weight_power=0.3
    # )
    # df_Imap = pd.concat(_dfs, ignore_index=True)
    # Generate Figure 1b - intrinsic performance curves
    ax = plot.plot_ip_c_1b(
        df_Imap,
        y_column="I_map",
        # y_smooth_column="I_map",
        # y_smooth_column="I_map_smooth",
        # xlabel="Compute C (FLOPs, log)",
        # ylabel=f"Intrinsic I(R) (log)",
        title=f"{display_name}",
        ax=axes["ip"]
    )
    # plt.tight_layout()
    # plt.savefig(metric_output_dir / "figure_1b.png", dpi=300, bbox_inches='tight')
    # plt.close()
    
    # =============================================================================
    # PHASE 4: JOINT FITTING OF SCALING LAW PARAMETERS
    # =============================================================================
    

    df_NERCI = df_Imap
    
    fit_result = intrinsic.fit_scaling_law_joint_alternating(
        df_NERCI.copy(),
        R_column="R_smooth",
        I_column="I_map",
        fit_fR=True,
        seed=0,
        n_trials=3,             # 3 rounds of iteration
        # max_iters=10,
        max_iters=160,
        popsize=20, 
        sigma0=0.5, 
        early_stop_tol=1e-12,       # Early stopping threshold
        w_eq=1,
        w_env=10,
        # Boundary parameters
        alpha_range=(0.1, 3.0),
        beta_min=0.1,
        soft_log_margin_percentage=10.0
    )
    intrinsic_func, f_R_logI = fit_result['intrinsic'], fit_result['f_R_logI']

    # =============================================================================
    # FIGURE 2A: FITTED SCALING LAW IN RETURN SPACE (C vs R_predicted)
    # =============================================================================
    
    df_I_pred_CNE_newX = intrinsic.predict_intrinsic_curves(df, intrinsic_func, phi=phi_global, warmup_clipping_factor=WARMUP_CLIPPING_FACTOR, sample_size_per_step=SAMPLE_SIZE_PER_STEP)

    tangent_points = intrinsic.calc_tangent_points(df_I_pred_CNE_newX)
    
    df_R_pred_I_pred_CNE_newX = intrinsic.predict_return_curves(df_I_pred_CNE_newX, f_R_logI)

    df_R_pred_I_pred_CNE_newX = data_proc.apply_warmup_clipping(df_R_pred_I_pred_CNE_newX, curve_column=curve_column, warmup_frac=WARMUP_CLIPPING_FACTOR)

    ax = plot.plot_fit_score_c_2a(
        df_R_smooth,
        # curve_column=curve_column,
        df_R_pred_I_pred_CNE_newX,
        # xlabel="Compute C (FLOPs, log)",
        # ylabel=f"Return R ({display_name}, predicted)",
        title=f"{display_name}",
        ax=axes["fit-score-norm"]
    )
    
    plt.tight_layout()
    # plt.savefig(metric_output_dir / "figure_2a.png", dpi=300, bbox_inches='tight')
    # plt.close()

    # =============================================================================
    # FIGURE 2B: FITTED SCALING LAW IN INTRINSIC SPACE (C vs I_predicted)
    # =============================================================================
    
    # Validate predictions: I_pred ≤ C
    # assert np.all(pred_intrinsic_curves["I_pred"] <= pred_intrinsic_curves["C"] + 1e-9), f"I_pred > C violation: max(I_pred-C) = {(pred_intrinsic_curves['I_pred'] - pred_intrinsic_curves['C']).max():.2e}"
    

    ax = plot.plot_fit_ip_2b(
        df_Imap,
        df_I_pred_CNE_newX, 
        tangent_points,
        # xlabel="Compute C (FLOPs, log)",
        # ylabel=f"Intrinsic I (log)",
        title=f"{display_name}",
        ax=axes["fit-ip"]
    )
    plt.tight_layout()
    # plt.savefig(metric_output_dir / "figure_2b.png", dpi=300, bbox_inches='tight')
    # plt.close()

    # Save fit results
    import json
    fit_result_serializable = {}
    for k, v in fit_result.items():
        if isinstance(v, np.ndarray):
            fit_result_serializable[k] = v.tolist()
        elif isinstance(v, (int, float, str, bool)):
            fit_result_serializable[k] = v
        elif hasattr(v, '__call__'):
            # Skip function objects
            continue
        else:
            # Skip other non-serializable objects
            fit_result_serializable[k] = str(type(v))
    with open(metric_output_dir / "fit_result.json", 'w') as f:
        json.dump(fit_result_serializable, f, indent=2)

    print(f"Done. Outputs saved to {metric_output_dir}")
    print("fit_result keys:", list(fit_result.keys()))
    
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
    # Plot Basic Curves
    # ===========================
    if PLOT_BASIC_CURVES:
        # keys = ["score-norm", "ip-c", "err-rate", "fit-score-norm", "fit-ip-c"]
        keys = ["score-norm", "err-rate", "improve-rate"]

        xlabels = {
            "T": "Tokens (log)",
            "C": "Compute C (FLOPs, log)",
            "E": "Data Size (log)"
        }

        for x_column, xlabel in xlabels.items():
            # Create fig and ax_list lists
            fig_axes = {key: plt.subplots(
                (total_metrics+FIGURE_COLUMNS-1) // FIGURE_COLUMNS, FIGURE_COLUMNS, 
                figsize=FIGURE_SIZE,
                constrained_layout=True
            ) for key in keys}

            for i, metric_name in enumerate(TEST_METRICS):
                if len(TEST_METRICS) > FIGURE_COLUMNS:
                    row = i // FIGURE_COLUMNS
                    col = i % FIGURE_COLUMNS
                    axes = {key: fig_axes[key][1][row, col] for key in keys}
                else:
                    axes = {key: fig_axes[key][1] for key in keys}
                process_single_metric(df.copy(), plot_x_column=x_column, metric_name=metric_name, curve_column='N', output_dir=OUTPUT_BASE_DIR, axes=axes)
            
            fig_axes['score-norm'][0].supxlabel(xlabel)
            fig_axes['score-norm'][0].supylabel("Return")

            fig_axes['err-rate'][0].supxlabel(xlabel)
            fig_axes['err-rate'][0].supylabel("Error Rate")

            fig_axes['improve-rate'][0].supxlabel(xlabel)
            fig_axes['improve-rate'][0].supylabel("Improve Rate")

            # fig_axes['fit-ip-c'][0].supxlabel("Compute C (FLOPs, log)")
            # fig_axes['fit-ip-c'][0].supylabel("Fitted Intrinsic Performance (log)")

            # fig_axes['fit-score-norm'][0].supxlabel("Compute C (FLOPs, log)")
            # fig_axes['fit-score-norm'][0].supylabel("Fitted Return")
            # Save files
            # [fig_axes[key][0].layout() for key in keys]
            [fig_axes[key][0].savefig(OUTPUT_BASE_DIR / (FIGURE_PREFIX+"_"+x_column+"_"+key+".pdf"), dpi=300, bbox_inches='tight') for key in keys]
            print(f"\n saved {x_column} figures")

        print(f"\n Basic curves - {FIGURE_PREFIX} complete! Check {OUTPUT_BASE_DIR} for results")
    # ===========================
    # Plot Intrinsic Curves
    # ===========================
    else: 
        keys = ["ip", "fit-score-norm", "fit-ip"]

        xlabels = {
            # "T": "Tokens (log)",
            "C": "Compute C (FLOPs, log)",
            # "E": "Data Size (log)"
        }

        fig_axes = {key: plt.subplots(
            (total_metrics+FIGURE_COLUMNS-1) // FIGURE_COLUMNS, FIGURE_COLUMNS, 
            figsize=FIGURE_SIZE,
            constrained_layout=True
        ) for key in keys}

        for i, metric_name in enumerate(TEST_METRICS):
            if len(TEST_METRICS) > FIGURE_COLUMNS:
                row = i // FIGURE_COLUMNS
                col = i % FIGURE_COLUMNS
                axes = {key: fig_axes[key][1][row, col] for key in keys}
            else:
                axes = {key: fig_axes[key][1] for key in keys}
            process_single_metric_intrinsic(df.copy(), plot_x_column="C", metric_name=metric_name, curve_column='N', output_dir=OUTPUT_BASE_DIR / "intrinsic", axes=axes)

        fig_axes['ip'][0].supxlabel("Compute C (FLOPs, log)")
        fig_axes['ip'][0].supylabel("Intrinsic Performance (log)")

        # fig_axes['fit-score-norm'][0].supxlabel("Compute C (FLOPs, log)")
        # fig_axes['fit-score-norm'][0].supylabel("Fitted Return")

        # fig_axes['fit-ip'][0].supxlabel("Compute C (FLOPs, log)")
        # fig_axes['fit-ip'][0].supylabel("Fitted Intrinsic Performance (log)")

        [fig_axes[key][0].savefig(OUTPUT_BASE_DIR / "intrinsic" / (FIGURE_PREFIX+"_"+key+".pdf"), dpi=300, bbox_inches='tight') for key in keys]
        # print(f"\n saved intrinsic figures")

        print(f"\n Intrinsic curves - {FIGURE_PREFIX} complete! Check {OUTPUT_BASE_DIR / 'intrinsic'} for results")

if __name__ == "__main__":
    main()