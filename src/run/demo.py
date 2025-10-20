#!/usr/bin/env python3
"""
Scaling Law Pipeline - Multi-Eval Analysis
Processes multiple test evals from Experiment1 data and generates scaling law plots for each eval
"""

from src.common import data_proc
from src.common import plot_data
from src.common import plot
from src.common import config

def main():
    model_type = "instruct"
    # model_type = "base"
    # model_type = "llama-base"
    # model_type = "llama-instruct"
    df = data_proc.load_and_preprocess(config.CSV_MAP[model_type])
    
    # eval_name = "val/test_score/openai/gsm8k" # must be one of config.TEST_EVALS.keys()
    eval_name = "holdout_score"
    curve_column = 'N' # key must be one of 'N', 'data_fator'
    for x_column in [ "C", "E", "T"]: # "T", "C", "E"
        for metric in ["ErrRate"]: # "R", "ErrRate", "DeltaReward", "DeltaErrRate"
            ax = plot_data.process_single_eval(
                df, 
                plot_x_column=x_column, 
                plot_eval_column=eval_name, 
                plot_metric=metric,
                plot_curve_column=curve_column, 
                plot_title=config.TEST_EVALS[eval_name]['plot_str'],
                plot_use_legend=True,
                # delta
                delta_base_step=1,
                # smooth
                add_smooth=True,
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
            )
            
            # Apply plot_basic_settings for styling and save
            plot.plot_basic_settings(
                ax=ax,
                x_scale="log",
                y_scale="log",
                x_label=config.DEFAULT_LABELS[x_column],
                y_label=config.DEFAULT_LABELS[metric],
                title=config.TEST_EVALS[eval_name]['plot_str'],
                use_legend=True,
                # Save configuration
                save_to_dir=config.OUTPUT_BASE_DIR, 
                save_to_filename_prefix=model_type+'_',
                plot_eval_column=eval_name,
                plot_curve_column=curve_column,
                plot_x_column=x_column,
                plot_metric=metric,
            )

if __name__ == "__main__":
    main()