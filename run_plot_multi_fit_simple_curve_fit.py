#!/usr/bin/env python3
"""
Scaling Law Pipeline - Multi-Eval Analysis
Processes multiple test evals from Experiment1 data and generates scaling law plots for each eval
"""

import data_proc
import fit_models_simple
import plot_data
import config
import fit
import plot
import matplotlib.pyplot as plt
import numpy as np

def main():
    data_source = "base"
    # data_source = "instruct"
    # data_source = "llama-instruct"
    # data_source = "llama-base"
    # data_source = "exp2-base"
    # data_source = "exp2-instruct"
    df = data_proc.load_and_preprocess(config.CSV_MAP[data_source])
    
    # Extract config values to local variables
    output_base_dir = config.OUTPUT_BASE_DIR
    warmup_clip_factor_raw = config.WARMUP_CLIPPING_FACTOR_FOR_RAW
    warmup_clip_factor_smooth = config.WARMUP_CLIPPING_FACTOR_FOR_SMOOTH
    test_evals = config.TEST_EVALS
    default_labels = config.DEFAULT_LABELS
    
    # eval_name = "val/test_score/openai/gsm8k" # must be one of config.TEST_EVALS.keys()
    eval_name = "holdout_score"
    
    # Fitting parameters - what we use for model fitting
    fit_curve_column = 'N'  # key must be one of 'N', 'Tau' - what we fit curves over
    fit_x_column = "E"       # What we use as X variable for fitting the scaling law
    fit_metrics = ["ErrRate"] # What metrics we fit
    
    # Plotting parameters - what we show in plots (can be different from fitting)
    plot_curve_column = 'N' # key must be one of 'N', 'Tau' - what curves to show
    plot_x_columns = ["E"]   # Available x columns for plotting
    plot_metrics = ["ErrRate"] # Available metrics for plotting
    
    # Fit load/save configuration
    fit_load_path = None  # Path to load pre-fitted model (None = fit from scratch)
    fit_save_path = None  # Path to save fitted model (None = don't save)
    # fit_load_path = "outputs/model_exp2-instruct_Tau_E.json"  # Example: load from file
    # fit_save_path = "outputs/model_exp2-instruct_Tau_E.json"  # Example: save to file
    fit_plot_k_E0 = True
    
    # Plot configuration
    plot_curve_mask = None #[1, 2, 5, 20, 25, 100]  # Which curves to include
    
    # Highlight configuration (shared parameters)
    highlight_curves = None #[1]                     # Which curves to highlight
    highlight_line_alpha = 1.0                 # Opacity for highlighted curves (1.0 = solid)
    highlight_line_width = 2.0                 # Line thickness for highlighted curves
    
    # Highlight control (separate for predict and plot phases)
    predict_highlight_enabled = False           # Enable highlight for predict_and_plot
    plot_highlight_enabled = False              # Enable highlight for process_single_eval

    # Plotting style configuration
    plot_use_scatter = True
    plot_use_line = True
    plot_x_scale = "log"
    plot_y_scale = "log"
    plot_use_legend = True
    plot_legend_loc = 'best'
    plot_legend_bbox_to_anchor = None
    
    # Extract lambda expressions to local variables
    plot_legend_lambda = lambda x: plot.legend_format(plot_curve_column, x)
    plot_y_lambda_r = lambda y: 1 - y  # For R metric transformation
    
    # Smoothing configuration
    add_smooth = False                          # Enable smoothing
    add_std = False                             # Add standard deviation bands
    smooth_monotonic = True                     # Force monotonic smoothing
    smooth_increasing = None                    # Force increasing trend (None=auto)
    smooth_strict = False                       # Strict monotonic constraints
    
    # Delta configuration
    delta_base_step = 1                         # Base step for delta calculation
    
    # Advanced smoothing parameters
    s_factor = 1                                # Smoothing factor
    k_spline = 5                                # Spline order
    rolling_window = 200                        # Rolling window size
    min_se = 1e-6                               # Minimum standard error
    x_inv_weight_power = 0.3                    # X inverse weight power
    
    # Fitting phase - using fit parameters
    print(f"\n=== Fitting for x_column: {fit_x_column}, curve_column: {fit_curve_column} ===")
    
    # Use simple lookup table fitting with specified fit parameters
    predicter = fit.fit_log_errrate_simple(df, eval_name, fit_curve_column, fit_x_column, 
                                          fit_load_path=fit_load_path,
                                          fit_save_path=fit_save_path,
                                          data_source=data_source)
    
    # Plotting phase - can iterate over different plot parameters
    for plot_x_column in plot_x_columns: # Available x columns for plotting
        print(f"\n=== Plotting for x_column: {plot_x_column} ===")
        
        # Note: We're using the fitted model but can plot with different x_column if needed
        
        if fit_plot_k_E0:
            # Get k(curve_column) and E0(curve_column) arrays for further analysis
            curve_values, k_values = predicter.get_k_array()
            curve_values_E0, E0_values = predicter.get_E0_array()
            
            print(f"\n=== k({fit_curve_column}) and E0({fit_curve_column}) Arrays for fitted x_column={fit_x_column} ===")
            print(f"{fit_curve_column}_values:", curve_values)
            print("k_values:", k_values)
            print("E0_values:", E0_values)
            
            # Plot k(curve_column) and E0(curve_column) scatter plots
            curve_billions = curve_values / 1e9  # Convert to billions for better readability
            
            # Plot k(curve_column) - Compact version
            fig1, ax1 = plt.subplots(figsize=(6, 4.5), dpi=300)  # Smaller figure size
            ax1 = plot.plot_basic(
                x=curve_billions,
                y=np.abs(k_values),  # Use absolute value since k is negative
                use_scatter=True,
                scatter_s=150,  # Larger scatter points
                color='blue',
                ax=ax1
            )
            ax1 = plot.plot_basic_settings(
                ax=ax1,
                x_scale="log",
                # y_scale="log", 
                x_label=f"Model Size {fit_curve_column} (Billions of Parameters)",
                y_label=f"k({fit_curve_column}) (Data Efficiency Slope for {fit_x_column})",
                title=f"Data Efficiency k({fit_curve_column}) vs Model Size (fitted on {fit_x_column})",
                use_legend=False,
                x_margin=0.1, y_margin=0.1  # 10% margins on both axes
            )
            
            plt.tight_layout(pad=1.0)  # Reduce padding
            eval_file_str = config.TEST_EVALS[eval_name]['file_str']
            plt.savefig(config.OUTPUT_BASE_DIR / f"fit_{data_source}_{eval_file_str}_{fit_curve_column}_{fit_x_column}_k_scatter.pdf", bbox_inches='tight', dpi=300)
            print(f"Saved k({fit_curve_column}) plot: {config.OUTPUT_BASE_DIR}/fit_{data_source}_{eval_file_str}_{fit_curve_column}_{fit_x_column}_k_scatter.pdf")
            
            # Plot E0(curve_column) - Compact version
            fig2, ax2 = plt.subplots(figsize=(6, 4.5), dpi=300)  # Smaller figure size
            ax2 = plot.plot_basic(
                x=curve_billions,
                y=E0_values,
                use_scatter=True,
                scatter_s=150,  # Larger scatter points
                color='red',
                ax=ax2
            )
            ax2 = plot.plot_basic_settings(
                ax=ax2,
                x_scale="log",
                y_scale=None,  # Linear scale for E0 since it can be negative
                x_label=f"{config.DEFAULT_LABELS[fit_curve_column]}",
                y_label=f"E0({fit_curve_column}) (Error Offset for {fit_x_column})",
                title=f"Error Offset E0({fit_curve_column}) vs Model Size (fitted on {fit_x_column})",
                use_legend=False,
                x_margin=0.1, y_margin=0.1  # 10% margins on both axes
            )
            
            plt.tight_layout(pad=1.0)  # Reduce padding
            plt.savefig(config.OUTPUT_BASE_DIR / f"fit_{data_source}_{eval_file_str}_{fit_curve_column}_{fit_x_column}_E0_scatter.pdf", bbox_inches='tight', dpi=300)
            print(f"Saved E0({fit_curve_column}) plot: {config.OUTPUT_BASE_DIR}/fit_{data_source}_{eval_file_str}_{fit_curve_column}_{fit_x_column}_E0_scatter.pdf")
        
        # fit_metric = "ErrRate"
        for plot_metric in plot_metrics: # "R", "ErrRate", "DeltaReward", "DeltaErrRate"
            # df_fit_plot = data_proc.apply_warmup_clip(df, curve_column="N", warmup_frac=config.WARMUP_CLIPPING_FACTOR_FOR_RAW)
            # ax = plot.plot_curves(df_fit_plot, curve_column=curve_column, x_column=x_column, y_column=metric+"_pred", use_line=True, x_scale="log")

            # Extract function parameters to local variables
            predict_x_column_list = [plot_curve_column, plot_x_column]
            predict_plot_highlight_curves = highlight_curves if predict_highlight_enabled else None
            predict_plot_use_scatter = False  # predict_and_plot usually shows lines only
            predict_plot_y_lambda = plot_y_lambda_r if plot_metric == "R" else None # TODO: support other than ErrRate

            ax = plot_data.predict_and_plot(
                df,
                predicter.predict_errrate_df,
                predict_x_column_list=predict_x_column_list,
                metric_column=plot_metric,
                plot_curve_column=plot_curve_column,
                plot_curve_mask=plot_curve_mask,

                # Highlight configuration (conditionally enabled)
                plot_highlight_curves=predict_plot_highlight_curves,
                plot_highlight_line_alpha=highlight_line_alpha,
                plot_highlight_line_width=highlight_line_width,

                # Plotting style
                plot_x_column=plot_x_column,  # Use plot x_column (which may differ from fit x_column)
                plot_use_line=plot_use_line,
                plot_use_scatter=predict_plot_use_scatter,
                plot_x_scale=plot_x_scale,
                plot_y_scale=plot_y_scale,
                plot_y_lambda=predict_plot_y_lambda,
                warmup_frac_raw=warmup_clip_factor_raw,
                # ax=ax,
            )
            
            # Extract more function parameters to local variables  
            process_plot_x_label = default_labels[plot_x_column]
            process_plot_y_label = default_labels[plot_metric]
            process_plot_highlight_curves = highlight_curves if plot_highlight_enabled else None
            process_plot_title = f"{test_evals[eval_name]['plot_str']} (fitted on {fit_x_column}, plotted on {plot_x_column})"
            process_save_to_filename_prefix = f"fit_{data_source}_"
            
            plot_data.process_single_eval(
                df, 
                plot_x_column=plot_x_column,
                plot_eval_column=eval_name, 
                plot_metric=plot_metric,
                plot_curve_column=plot_curve_column, 
                plot_x_label=process_plot_x_label,
                plot_y_label=process_plot_y_label,
                plot_curve_mask=plot_curve_mask,
                
                # Highlight configuration (conditionally enabled)
                plot_highlight_curves=process_plot_highlight_curves,
                plot_highlight_line_alpha=highlight_line_alpha,
                plot_highlight_line_width=highlight_line_width,
                
                # Plotting style
                plot_use_scatter=plot_use_scatter,
                plot_x_scale=plot_x_scale,
                plot_y_scale=plot_y_scale,
                plot_title=process_plot_title,
                plot_use_legend=plot_use_legend,
                plot_legend_loc=plot_legend_loc,
                plot_legend_bbox_to_anchor=plot_legend_bbox_to_anchor,
                plot_legend_lambda=plot_legend_lambda,
                
                # Delta configuration
                delta_base_step=delta_base_step,
                
                # Smoothing configuration
                add_smooth=add_smooth,
                add_std=add_std,
                smooth_monotonic=smooth_monotonic,
                smooth_increasing=smooth_increasing,
                smooth_strict=smooth_strict,
                
                # Advanced parameters
                warmup_frac_raw=warmup_clip_factor_raw,
                warmup_frac_smooth=warmup_clip_factor_smooth,
                s_factor=s_factor,
                k_spline=k_spline,
                rolling_window=rolling_window,
                min_se=min_se,
                x_inv_weight_power=x_inv_weight_power,
                ax=ax,
                save_to_dir=output_base_dir, 
                save_to_filename_prefix=process_save_to_filename_prefix,
            )

if __name__ == "__main__":
    main()