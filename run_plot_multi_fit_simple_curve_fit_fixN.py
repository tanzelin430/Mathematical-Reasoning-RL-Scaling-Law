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
    # Process both base and instruct models
    data_sources = ["base", "instruct"]
    
    # eval_name = "val/test_score/openai/gsm8k" # must be one of config.TEST_EVALS.keys()
    eval_name = "holdout_score"
    curve_column = 'N' # key must be one of 'N', 'data_fator'

    warmup_clip_num = 10
    plot_fit_title = f"L(D) (N=14B) - Base vs Instruct"
    plot_curve_mask = [14e9]

    for x_column in ["E"]: # Available x columns for fitting
        print(f"\n=== Fitting for x_column: {x_column} ===")
        
        # Store results for both model types
        all_results = {}
        
        for data_source in data_sources:
            print(f"\n--- Processing {data_source} ---")
            df = data_proc.load_and_preprocess(config.CSV_MAP[data_source])
            
            # Use simple lookup table fitting with selected x_column
            predicter = fit.fit_log_errrate_simple(df, eval_name, curve_column, x_column)
            
            # Get k(curve_column) and E0(curve_column) arrays for further analysis
            curve_values, k_values = predicter.get_k_array()
            curve_values_E0, E0_values = predicter.get_E0_array()
            
            print(f"\n=== k({curve_column}) and E0({curve_column}) Arrays for {data_source}, x_column={x_column} ===")
            print(f"{curve_column}_values:", curve_values)
            print("k_values:", k_values)
            print("E0_values:", E0_values)
            
            # Store results for combined plotting
            all_results[data_source] = {
                'curve_values': curve_values,
                'k_values': k_values,
                'E0_values': E0_values,
                'df': df,
                'predicter': predicter
            }
        
        # Create combined plots with both model types
        eval_file_str = config.TEST_EVALS[eval_name]['file_str']
        
        # Plot k(curve_column) for both model types
        fig1, ax1 = plt.subplots(figsize=(6, 4), dpi=300)
        
        for data_source in data_sources:
            curve_values = all_results[data_source]['curve_values']
            k_values = all_results[data_source]['k_values']
            # curve_values = curve_values / 1e9  # Convert to billions for better readability
            
            ax1 = plot.plot_basic(
                x=curve_values,
                y=np.abs(k_values),  # Use absolute value since k is negative
                use_scatter=True,
                scatter_s=50,
                scatter_alpha=0.5,
                scatter_marker="o",
                color=config.get_color_for_curve(data_source),
                ax=ax1
            )
            # Add label manually for legend
            ax1.scatter([], [], color=config.get_color_for_curve(data_source), s=10, alpha=0.5, marker="o", label=data_source.capitalize())
        
        ax1 = plot.plot_basic_settings(
            ax=ax1,
            x_scale="log",
            # y_scale="log", 
            x_label=f"Model Size {curve_column}",
            y_label=f"|k({curve_column})| (Data Efficiency Slope for {x_column})",
            title=f"Data Efficiency k({curve_column}) vs Model Size (fitted on {x_column})",
            use_legend=True,
            # Add custom formatting and grid settings
            # x_tick_spacing=5e4,
            # x_grid_spacing=1e4,
            y_tick_format="decimal",
            y_grid_spacing=0.1
        )
        plt.tight_layout()
        plt.savefig(config.OUTPUT_BASE_DIR / f"fit_both_{eval_file_str}_{curve_column}_{x_column}_k_scatter.pdf", bbox_inches='tight', dpi=300)
        print(f"Saved k({curve_column}) plot: {config.OUTPUT_BASE_DIR}/fit_both_{eval_file_str}_{curve_column}_{x_column}_k_scatter.pdf")
        
        # Plot E0(curve_column) for both model types
        fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=300)
        
        for data_source in data_sources:
            curve_values = all_results[data_source]['curve_values']
            E0_values = all_results[data_source]['E0_values']
            # curve_values = curve_values / 1e9  # Convert to billions for better readability
            
            ax2 = plot.plot_basic(
                x=curve_values,
                y=E0_values,
                use_scatter=True,
                scatter_s=50,
                scatter_alpha=0.5,
                scatter_marker="o",
                color=config.get_color_for_curve(data_source),
                ax=ax2
            )
            # Add label manually for legend
            ax2.scatter([], [], color=config.get_color_for_curve(data_source), s=10, alpha=0.5, marker="o", label=data_source.capitalize())
        
        ax2 = plot.plot_basic_settings(
            ax=ax2,
            x_scale="log",
            y_scale=None,  # Linear scale for E0 since it can be negative
            x_label=f"Model Size {curve_column}",
            y_label=f"E0({curve_column}) (Error Offset for {x_column})",
            title=f"Error Offset E0({curve_column}) vs Model Size (fitted on {x_column})",
            use_legend=True,
            # Add custom formatting and grid settings
            # x_tick_spacing=5e4,
            # x_grid_spacing=1e4,
            y_tick_format="decimal",
            # y_grid_spacing=0.1
        )
        plt.tight_layout()
        plt.savefig(config.OUTPUT_BASE_DIR / f"fit_both_{eval_file_str}_{curve_column}_{x_column}_E0_scatter.pdf", bbox_inches='tight', dpi=300)
        print(f"Saved E0({curve_column}) plot: {config.OUTPUT_BASE_DIR}/fit_both_{eval_file_str}_{curve_column}_{x_column}_E0_scatter.pdf")
        
        # Create combined plots for each metric with both model types
        for metric in ["ErrRate"]: # "R", "ErrRate", "DeltaReward", "DeltaErrRate"
            # Initialize combined plot
            fig3, ax3 = plt.subplots(figsize=(6, 4), dpi=300)
            
            for data_source in data_sources:
                df = all_results[data_source]['df']
                predicter = all_results[data_source]['predicter']
                
                # Create custom color mapping for this model type
                # Map 14e9 (the curve we're plotting) to the model_type color
                custom_color_mapping = {14e9: config.get_color_for_curve(data_source)}
                
                # Generate prediction and plot for this model type
                ax3 = plot_data.predict_and_plot(
                    df,
                    predicter.predict_errrate_df,
                    predict_x_column_list=[curve_column, x_column],  # Use the fitted x_column
                    metric_column=metric,
                    plot_curve_column=curve_column,
                    plot_curve_mask=plot_curve_mask,
                    plot_x_column=x_column,  # Only plot the fitted x_column
                    plot_use_line=True,
                    plot_y_lambda=(lambda y: 1 - y) if metric == "R" else None,
                    plot_x_scale="log",
                    plot_y_scale="log",
                    line_width=2,
                    line_alpha=1,
                    warmup_clip_raw=warmup_clip_num,
                    custom_color_mapping=custom_color_mapping,
                    ax=ax3,
                )
                
                # Add scatter plot for this model type with config colors
                plot_data.process_single_eval(
                    df, 
                    plot_x_column=x_column,  # Plot only the fitted x_column
                    plot_eval_column=eval_name, 
                    plot_metric=metric,
                    plot_curve_column=curve_column, 
                    plot_curve_mask=plot_curve_mask,
                    plot_use_scatter=True,
                    plot_use_legend=False,  # Disable individual legends, we'll add combined one later
                    scatter_alpha=0.5,
                    scatter_size=20,
                    scatter_marker="o",
                    warmup_clip_raw=warmup_clip_num,
                    s_factor=1,
                    k_spline=5,
                    rolling_window=200,
                    min_se=1e-6,
                    x_inv_weight_power=0.3,
                    custom_color_mapping=custom_color_mapping,
                    ax=ax3,
                    save_to_dir=None,  # Don't save individual plots, we'll save the combined one
                    save_to_filename_prefix=None,
                )
                
                # Apply plot_basic_settings after the last data source
                if data_source == data_sources[-1]:  # Only apply on the last data source
                    plot.plot_basic_settings(
                        ax=ax3,
                        x_scale="log",
                        y_scale="log",
                        x_label=config.DEFAULT_LABELS[x_column],
                        y_label=config.DEFAULT_LABELS[metric],
                        title=plot_fit_title,
                        x_tick_spacing=0.5,
                        y_tick_format="decimal",
                        y_tick_spacing=0.1,
                        use_legend=False
                    )
            
            # Add combined legend manually
            legend_handles = []
            legend_labels = []
            for data_source in data_sources:
                legend_handles.append(plt.Line2D([0], [0], color=config.get_color_for_curve(data_source), 
                                               linewidth=2, label=data_source.capitalize()))
                legend_labels.append(data_source.capitalize())
            
            ax3.legend(legend_handles, legend_labels, loc='best')
            
            # Save the combined plot
            plt.tight_layout()
            plt.savefig(config.OUTPUT_BASE_DIR / f"fit_both_{eval_file_str}_{curve_column}_{x_column}_{metric}.pdf", bbox_inches='tight', dpi=300)
            print(f"Saved combined {metric} plot: {config.OUTPUT_BASE_DIR}/fit_both_{eval_file_str}_{curve_column}_{x_column}_{metric}.pdf")

if __name__ == "__main__":
    main()