#!/usr/bin/env python3
"""
Scaling Law Pipeline - Multi-Eval Analysis
Processes multiple test evals from Experiment1 data and generates scaling law plots for each eval
"""

from src.common import data_proc
from src.common import plot
from src.common import config
import matplotlib.pyplot as plt

def main():
    data_source = "instruct"
    # data_source = "base"
    # data_source = "llama-base"
    # data_source = "llama-instruct"
    df = data_proc.load_and_preprocess(config.CSV_MAP[data_source])
    
    # Get physical dimensions for this data source
    physical_dimensions = config.get_physical_dimensions(data_source)
    physical_curve_column = physical_dimensions[0]  # N, slice_factor, or rollout_n
    physical_x_column = physical_dimensions[1]      # step
    
    # Remove step=0 data (because E=0 will cause log10(E)=-inf)
    df = df[df['step'] > 0].reset_index(drop=True)
    
    # eval_name = "val/test_score/openai/gsm8k" # must be one of config.TEST_EVALS.keys()
    eval_name = "holdout_score"
    curve_column = 'N' #
    for x_column in [ "C", "E", "T"]: # "T", "C", "E"
        for metric in ["ErrRate"]: # "R", "ErrRate", "DeltaReward", "DeltaErrRate"
            # Prepare eval data (using physical dimensions for merging)
            df_eval = data_proc.prepare_eval_data(
                df,
                eval_column=eval_name,
                curve_column=physical_curve_column,
                x_column=physical_x_column,
                calc_delta=metric.startswith('Delta'),
                delta_base_step=1
            )
            
            # Create figure
            fig, ax = plt.subplots(figsize=(6, 4))
            
            # Plot raw scatter points
            ax = plot.plot_curves(
                df_eval,
                curve_column=curve_column,
                x_column=x_column,
                y_column=metric,
                use_scatter=True,
                use_line=False,
                scatter_alpha=1.0,
                scatter_size=8.0,
                scatter_marker='o',
                ax=ax,
            )
            
            # Add smooth curves
            smooth_out_column = metric + "_smooth"
            df_smooth = data_proc.smooth_df(
                df_eval,
                curve_column=curve_column,
                col_x=x_column,
                col_y=metric,
                col_y_out=smooth_out_column,
                monotonic=True,
                increasing=None,
                strict=False,
                s_factor=1,
                k_spline=5,
                rolling_window=200,
                min_se=1e-6,
                x_inv_weight_power=0.3,
                use_linear=False
            )
            
            ax = plot.plot_curves(
                df_smooth,
                curve_column=curve_column,
                x_column=x_column,
                y_column=smooth_out_column,
                use_scatter=False,
                use_line=True,
                line_alpha=1.0,
                line_width=2.0,
                ax=ax
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
                save_to_filename_prefix=data_source+'_',
                plot_eval_column=eval_name,
                plot_curve=curve_column,
                plot_x_column=x_column,
                plot_metric=metric,
            )

if __name__ == "__main__":
    main()