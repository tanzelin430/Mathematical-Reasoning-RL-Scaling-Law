#!/usr/bin/env python3
"""
Scaling Law Pipeline - Multi-Eval Analysis
Processes multiple test evals from Experiment1 data and generates scaling law plots for each eval
"""

import data_proc
import plot_data
import config
import fit

def main():
    model_type = "base"
    # model_type = "instruct"
    # model_type = "llama-instruct"
    # model_type = "llama-base"
    df = data_proc.load_and_preprocess(config.CSV_MAP[model_type])
    
    # eval_name = "val/test_score/openai/gsm8k" # must be one of config.TEST_EVALS.keys()
    eval_name = "holdout_score"
    curve_column = 'N' # key must be one of 'N', 'data_fator'
    for x_column in [ "C", "E", "T" ]: # "T", "C", "E"
        for metric in ["R", "ErrRate"]: # "R", "ErrRate", "DeltaReward", "DeltaErrRate"
            
            predicter = fit.fit_log_errrate(df, eval_name)
            # df_fit_plot = data_proc.apply_warmup_clipping(df, curve_column="N", warmup_frac=config.WARMUP_CLIPPING_FACTOR_FOR_RAW)
            # ax = plot.plot_curves(df_fit_plot, curve_column=curve_column, x_column=x_column, y_column=metric+"_pred", use_line=True, x_scale="log")

            ax = plot_data.predict_and_plot(
                df,
                predicter.predict_errrate_df,
                predict_x_column_list=["N", "E"],
                metric_column="ErrRate",
                plot_curve_column=curve_column,
                plot_x_column=x_column,
                plot_use_line=True,
                plot_y_lambda=(lambda y: 1 - y) if metric == "R" else None,
                # plot_use_delta=is_delta,
                # plot_delta_base_step=0,
                plot_x_scale="log",
                warmup_frac_raw=config.WARMUP_CLIPPING_FACTOR_FOR_RAW,
                # ax=ax,
            )
            
            plot_data.process_single_eval(
                df, 
                plot_x_column=x_column, 
                plot_eval_column=eval_name, 
                plot_metric=metric,
                plot_curve_column=curve_column, 
                plot_x_label=config.DEFAULT_X_LABELS[x_column],
                plot_y_label=config.DEFAULT_Y_LABELS[metric],
                plot_x_scale="log",
                # plot_y_scale=y_scale,
                plot_title=config.TEST_EVALS[eval_name]['plot_str'],
                plot_use_legend=True,
                # delta
                delta_base_step=1,
                # smooth
                # add_smooth=True,
                # add_std=True,
                smooth_monotonic=True,
                smooth_increasing=None,
                smooth_strict=False,
                warmup_frac_raw=config.WARMUP_CLIPPING_FACTOR_FOR_RAW,
                warmup_frac_smooth=config.WARMUP_CLIPPING_FACTOR_FOR_SMOOTH,
                s_factor=1,
                k_spline=5,
                rolling_window=200,
                min_se=1e-6,
                x_inv_weight_power=0.3,
                save_to_dir=config.OUTPUT_BASE_DIR, 
                save_to_filename_prefix='fit_'+model_type+'_',
                ax=ax,
            )

if __name__ == "__main__":
    main()