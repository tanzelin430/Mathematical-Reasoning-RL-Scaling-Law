#!/usr/bin/env python3
"""
Scaling Law Pipeline - Multi-Eval Analysis
Processes multiple test evals from Experiment1 data and generates scaling law plots for each eval
"""

import data_proc
import plot_data
import config

def main():
    model_type = "base"
    model_type = "instruct"

    df = data_proc.load_and_preprocess(config.CSV_INSTRUCT_RUNS if model_type == "instruct" else config.CSV_BASE_RUNS)
    
    # eval_name = "val/test_score/openai/gsm8k" # must be one of config.TEST_EVALS.keys()
    eval_name = "holdout_score"
    x_column = "E" # key must be one of 'T', 'C', 'E'
    metric = 'R' # key must be one of 'R', 'ErrRate', 'DeltaReward', 'DeltaErrRate'
    curve_column = 'N' # key must be one of 'N', 'data_fator'
    plot_data.process_single_eval(
        df, 
        plot_x_column=x_column, 
        plot_eval_column=eval_name, 
        plot_metric=metric,
        plot_curve_column=curve_column, 
        # plot_curve_mask=[14e9],
        plot_x_label=config.DEFAULT_LABELS[x_column],
        plot_y_label=config.DEFAULT_LABELS[metric],
        plot_x_scale="log",
        # plot_y_scale="log",
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
        save_to_dir=config.OUTPUT_BASE_DIR, 
        save_to_filename_prefix=model_type+'_',
    )

if __name__ == "__main__":
    main()