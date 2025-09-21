#!/usr/bin/env python3
"""
Scaling Law Pipeline - Multi-Eval Analysis
Processes multiple test evals from Experiment1 data and generates scaling law plots for each eval
"""

import data_proc
import fit_models
import fit_models_simple
import plot_data
import config
import fit
import plot
import matplotlib.pyplot as plt
import numpy as np

def main():

    model_type = "instruct"
    # df = data_proc.load_and_preprocess(config.CSV_INSTRUCT_RUNS if model_type == "instruct" else config.CSV_BASE_RUNS)
    df = data_proc.load_and_preprocess([config.CSV_INSTRUCT_RUNS[0]])
    
    # predicter = fit.FitLogErrRate(L=0.06333, r=1.73e-10, N0_k=4.95e9, r_e0=1e-9, N0_e0=3e9)
    # df["R_pred"] = predicter.predict_reward_df(df, "N", "E")
    # df["ErrRate_pred"] = predicter.predict_errrate_df(df, "N", "E")
    # df["DeltaReward_pred"] = data_proc.calc_delta_y(df, 'R_pred', base_step=1, curve_column="N")
    # df['DeltaErrRate_pred'] = data_proc.calc_delta_y(df, 'ErrRate_pred', base_step=1, curve_column="N")

    # eval_name = "val/test_score/openai/gsm8k" # must be one of config.TEST_EVALS.keys()
    eval_name = "holdout_score"
    x_column = "T" # key must be one of 'T', 'C', 'E'
    pred_metric = 'R' # key must be one of 'R', 'ErrRate', 'DeltaReward', 'DeltaErrRate'
    metric = 'R' # key must be one of 'R', 'ErrRate', 'DeltaReward', 'DeltaErrRate'
    curve_column = 'N' # key must be one of 'N', 'data_fator'
    # is_delta = True

    # predicter = fit.fit_log_errrate(df, eval_name)
    # Use simple lookup table fitting instead of complex logistic fitting
    predicter = fit_models_simple.fit_simple_lookup(df, eval_name)
    
    # Get k(N) and E0(N) arrays for further analysis
    N_values, k_values = predicter.get_k_array()
    N_values_E0, E0_values = predicter.get_E0_array()
    
    print("\n=== k(N) and E0(N) Arrays for Analysis ===")
    print("N_values:", N_values)
    print("k_values:", k_values)
    print("E0_values:", E0_values)
    
    # Plot k(N) and E0(N) scatter plots
    N_billions = N_values / 1e9  # Convert to billions for better readability
    
    # Plot k(N)
    fig1, ax1 = plt.subplots(figsize=(8, 6), dpi=150)
    ax1 = plot.plot_basic(
        x=N_billions,
        y=np.abs(k_values),  # Use absolute value since k is negative
        use_scatter=True,
        scatter_s=100,
        color='blue',
        ax=ax1
    )
    ax1 = plot.plot_basic_settings(
        ax=ax1,
        x_scale="log",
        y_scale="log", 
        x_label="Model Size N (Billions of Parameters)",
        y_label="|k(N)| (Data Efficiency Slope)",
        title="Data Efficiency k(N) vs Model Size",
        use_legend=False
    )
    plt.tight_layout()
    eval_file_str = config.TEST_EVALS[eval_name]['file_str']
    plt.savefig(config.OUTPUT_BASE_DIR / f"fit_{model_type}_{eval_file_str}_N_E_k_scatter.pdf", bbox_inches='tight')
    plt.savefig(config.OUTPUT_BASE_DIR / f"fit_{model_type}_{eval_file_str}_N_E_k_scatter.png", bbox_inches='tight', dpi=150)
    print(f"Saved k(N) plot: {config.OUTPUT_BASE_DIR}/fit_{model_type}_{eval_file_str}_N_E_k_scatter.pdf")
    
    # Plot E0(N)
    fig2, ax2 = plt.subplots(figsize=(8, 6), dpi=150)
    ax2 = plot.plot_basic(
        x=N_billions,
        y=E0_values,
        use_scatter=True,
        scatter_s=100,
        color='red',
        ax=ax2
    )
    ax2 = plot.plot_basic_settings(
        ax=ax2,
        x_scale="log",
        y_scale=None,  # Linear scale for E0 since it can be negative
        x_label="Model Size N (Billions of Parameters)",
        y_label="E0(N) (Error Offset)",
        title="Error Offset E0(N) vs Model Size",
        use_legend=False
    )
    plt.tight_layout()
    plt.savefig(config.OUTPUT_BASE_DIR / f"fit_{model_type}_{eval_file_str}_N_E_E0_scatter.pdf", bbox_inches='tight')
    plt.savefig(config.OUTPUT_BASE_DIR / f"fit_{model_type}_{eval_file_str}_N_E_E0_scatter.png", bbox_inches='tight', dpi=150)
    print(f"Saved E0(N) plot: {config.OUTPUT_BASE_DIR}/fit_{model_type}_{eval_file_str}_N_E_E0_scatter.pdf")
    
    # df_fit_plot = data_proc.apply_warmup_clipping(df, curve_column="N", warmup_frac=config.WARMUP_CLIPPING_FACTOR_FOR_RAW)
    # ax = plot.plot_curves(df_fit_plot, curve_column=curve_column, x_column=x_column, y_column=metric+"_pred", use_line=True, x_scale="log")

    ax = plot_data.predict_and_plot(
        df,
        predicter.predict_errrate_df,
        predict_x_column_list=["N", "E"],
        metric_column=pred_metric,
        plot_curve_column=curve_column,
        plot_x_column=x_column,
        plot_use_line=True,
        plot_y_lambda=lambda y: 1 - y,
        # plot_use_delta=is_delta,
        # plot_delta_base_step=0,
        plot_x_scale="log",
        warmup_frac_raw=config.WARMUP_CLIPPING_FACTOR_FOR_RAW,
        # ax=ax,
    )
    ax = plot_data.process_single_eval(
        df, 
        plot_x_column=x_column, 
        plot_eval_column=eval_name, 
        plot_metric=metric,
        plot_curve_column=curve_column, 
        # plot_curve_mask=[14e9],
        plot_x_label=config.DEFAULT_LABELS[x_column],
        plot_y_label=config.DEFAULT_LABELS[metric],
        plot_use_scatter=True,
        # y_smooth_use_line=True,
        plot_x_scale="log",
        # plot_y_scale=y_scale,
        # plot_title=config.TEST_EVALS[eval_name]['plot_str'],
        plot_use_legend=True,
        plot_legend_loc='upper right',
        plot_legend_bbox_to_anchor=(1, 0.5),
        # delta
        delta_base_step=0,
        # smooth
        # add_smooth=True,
        # add_std=True,
        # smooth_monotonic=True,
        # smooth_increasing=None,
        # smooth_strict=False,
        warmup_frac_raw=config.WARMUP_CLIPPING_FACTOR_FOR_RAW,
        warmup_frac_smooth=config.WARMUP_CLIPPING_FACTOR_FOR_SMOOTH,
        s_factor=1,
        k_spline=5,
        rolling_window=200,
        min_se=1e-6,
        x_inv_weight_power=0.3,
        ax=ax,
        save_to_dir=config.OUTPUT_BASE_DIR, 
        save_to_filename=f"fit_{model_type}_{eval_file_str}_{curve_column}_{x_column}_{metric}.pdf",
    )

    # plt.show()

if __name__ == "__main__":
    main()