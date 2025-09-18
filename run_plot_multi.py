#!/usr/bin/env python3
"""
Scaling Law Pipeline - Multi-Eval Analysis
Processes multiple test evals from Experiment1 data and generates scaling law plots for each eval
"""

import data_proc
import plot_data
import config

def main():
    df = data_proc.load_and_preprocess([
        config.SCRIPT_DIR / "csv" / "scaling_law_data_experiment1_base_run0.csv" ,
        # config.SCRIPT_DIR / "csv" / "scaling_law_data_experiment1_instruct_run0.csv" ,
        # config.SCRIPT_DIR / "csv" / "scaling_law_data_experiment1_instruct_run1.csv" ,
        # config.SCRIPT_DIR / "csv" / "scaling_law_data_experiment1_instruct_run2.csv" ,
    ])
    y_labels = {
        'R': "Reward", 
        'ErrRate': "Error Rate", 
        'DeltaReward': "Improvement"
    }

    x_labels = {
        "T": "Tokens",
        "C": "Compute (FLOPs)",
        "E": "Data Size"
    }
    # eval_name = "val/test_score/openai/gsm8k" # must be one of config.TEST_EVALS.keys()
    eval_name = "holdout_score"
    curve_column = 'N' # key must be one of 'N', 'data_fator'
    for x_column in ["T", "C", "E"]:
        for metric in ["R", "ErrRate", "DeltaReward", "DeltaErrRate"]:
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
                plot_title=eval_name,
                # delta
                delta_base_step=1,
                # smooth
                add_smooth=True,
                add_std=True,
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
            )

if __name__ == "__main__":
    main()