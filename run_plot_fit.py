#!/usr/bin/env python3
"""
Scaling Law Pipeline - Multi-Eval Analysis
Processes multiple test evals from Experiment1 data and generates scaling law plots for each eval
"""

import data_proc
import fit_models
import plot_data
import config
import fit
import plot
import matplotlib.pyplot as plt
import numpy as np

def main():

    model_type = "instruct"
    df = data_proc.load_and_preprocess(config.CSV_INSTRUCT_RUNS if model_type == "instruct" else config.CSV_BASE_RUNS)
    
    # predicter = fit.FitLogErrRate(L=0.06333, r=1.73e-10, N0_k=4.95e9, r_e0=1e-9, N0_e0=3e9)
    # df["R_pred"] = predicter.predict_reward_df(df, "N", "E")
    # df["ErrRate_pred"] = predicter.predict_errrate_df(df, "N", "E")
    # df["DeltaReward_pred"] = data_proc.calc_delta_y(df, 'R_pred', base_step=1, curve_column="N")
    # df['DeltaErrRate_pred'] = data_proc.calc_delta_y(df, 'ErrRate_pred', base_step=1, curve_column="N")

    # eval_name = "val/test_score/openai/gsm8k" # must be one of config.TEST_EVALS.keys()
    eval_name = "holdout_score"
    x_column = "C" # key must be one of 'T', 'C', 'E'
    pred_metric = 'R' # key must be one of 'R', 'ErrRate', 'DeltaReward', 'DeltaErrRate'
    metric = 'R' # key must be one of 'R', 'ErrRate', 'DeltaReward', 'DeltaErrRate'
    curve_column = 'N' # key must be one of 'N', 'data_fator'
    # is_delta = True

    predicter = fit.fit_log_errrate(df, eval_name)
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
        plot_x_label=config.DEFAULT_X_LABELS[x_column],
        plot_y_label=config.DEFAULT_Y_LABELS[metric],
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
        save_to_filename=f"{config.TEST_EVALS[eval_name]['file_str']}_fit_{metric}_{x_column}_{model_type}.pdf",
    )

    # plt.show()

if __name__ == "__main__":
    main()