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
    model_types = ["base", "instruct"]
    
    # eval_name = "val/test_score/openai/gsm8k" # must be one of config.TEST_EVALS.keys()
    eval_name = "holdout_score"
    
    # 拟合参数：拟合 log_errrate = k(N) * log_E + E0(N)
    fit_curve_column = 'N'  # curve parameter for fitting
    fit_x_column = 'E'      # x variable for fitting
    
    # 绘图参数：固定E=E_max，以N为x轴绘图
    plot_curve_column = 'E' # curve parameter for plotting (will be fixed to E_max)
    plot_x_column = 'N'     # x variable for plotting
    
    warmup_clip = 10
    
    print(f"\n=== Fitting: log_errrate = k({fit_curve_column}) * log_{fit_x_column} + E0({fit_curve_column}) ===")
    print(f"=== Plotting: Fixed {plot_curve_column}=E_max, x-axis={plot_x_column} ===")
    
    # Store results for both model types
    all_results = {}
    
    for model_type in model_types:
        print(f"\n--- Processing {model_type} ---")
        df = data_proc.load_and_preprocess(config.CSV_MAP[model_type])
        
        E_max = df.groupby('N')['E'].max().min()
        print(f"E_max for {model_type}: {E_max}")
        
        # Use simple lookup table fitting
        predicter = fit.fit_log_errrate_simple(df, eval_name, fit_curve_column, fit_x_column)
        
        # Store results for combined plotting
        all_results[model_type] = {
            'df': df,
            'predicter': predicter,
            'E_max': E_max
        }
        
        # # Get k(curve_column) and E0(curve_column) arrays for further analysis
        # curve_values, k_values = predicter.get_k_array()
        # curve_values_E0, E0_values = predicter.get_E0_array()
        
        # print(f"\n=== k({curve_column}) and E0({curve_column}) Arrays for x_column={x_column} ===")
        # print(f"{curve_column}_values:", curve_values)
        # print("k_values:", k_values)
        # print("E0_values:", E0_values)
        
        # Plot k(curve_column) and E0(curve_column) scatter plots
        # curve_billions = curve_values / 1e9  # Convert to billions for better readability
        
        # # Plot k(curve_column)
        # fig1, ax1 = plt.subplots(figsize=(6, 4), dpi=300)
        # ax1 = plot.plot_basic(
        #     x=curve_billions,
        #     y=np.abs(k_values),  # Use absolute value since k is negative
        #     use_scatter=True,
        #     scatter_s=100,
        #     color='blue',
        #     ax=ax1
        # )
        # ax1 = plot.plot_basic_settings(
        #     ax=ax1,
        #     x_scale="log",
        #     # y_scale="log", 
        #     x_label=f"Model Size {curve_column} (Billions of Parameters)",
        #     y_label=f"|k({curve_column})| (Data Efficiency Slope for {x_column})",
        #     title=f"Data Efficiency k({curve_column}) vs Model Size (fitted on {x_column})",
        #     use_legend=False
        # )
        # plt.tight_layout()
        # eval_file_str = config.TEST_EVALS[eval_name]['file_str']
        # plt.savefig(config.OUTPUT_BASE_DIR / f"fit_{model_type}_{eval_file_str}_{curve_column}_{x_column}_k_scatter.pdf", bbox_inches='tight', dpi=300)
        # print(f"Saved k({curve_column}) plot: {config.OUTPUT_BASE_DIR}/fit_{model_type}_{eval_file_str}_{curve_column}_{x_column}_k_scatter.pdf")
        
        # # Plot E0(curve_column)
        # fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=300)
        # ax2 = plot.plot_basic(
        #     x=curve_billions,
        #     y=E0_values,
        #     use_scatter=True,
        #     scatter_s=100,
        #     color='red',
        #     ax=ax2
        # )
        # ax2 = plot.plot_basic_settings(
        #     ax=ax2,
        #     x_scale="log",
        #     y_scale=None,  # Linear scale for E0 since it can be negative
        #     x_label=f"Model Size {curve_column} (Billions of Parameters)",
        #     y_label=f"E0({curve_column}) (Error Offset for {x_column})",
        #     title=f"Error Offset E0({curve_column}) vs Model Size (fitted on {x_column})",
        #     use_legend=False
        # )
        # plt.tight_layout()
        # plt.savefig(config.OUTPUT_BASE_DIR / f"fit_{model_type}_{eval_file_str}_{curve_column}_{x_column}_E0_scatter.pdf", bbox_inches='tight', dpi=300)
        # print(f"Saved E0({curve_column}) plot: {config.OUTPUT_BASE_DIR}/fit_{model_type}_{eval_file_str}_{curve_column}_{x_column}_E0_scatter.pdf")
        
    # Create combined plots for each metric with both model types
    for metric in ["ErrRate"]: # "R", "ErrRate", "DeltaReward", "DeltaErrRate"
        # Initialize combined plot
        fig3, ax3 = plt.subplots(figsize=(6, 4), dpi=300)
        
        eval_file_str = config.TEST_EVALS[eval_name]['file_str']
        
        # 定义彩虹配色（从上到下：深红、深橙、深黄、深绿、深蓝）
        colors = ['#CC0000', '#CC6600', '#228B22', '#0066CC', '#663399']
        
        # Map all model types to rainbow colors
        all_colors = {
            "base": colors[0],        # Deep red
            "instruct": colors[1],    # Deep orange
            "Sota-Qwen3-(Reason, Dense)": colors[2],    # Deep green
            "Sota-Qwen3-(Reason, MOE)": colors[3],      # Deep blue
            "Sota-GPT-OSS-(Reason, MOE)": colors[4],    # Purple
        }
        
        for model_type in model_types:
            df = all_results[model_type]['df']
            predicter = all_results[model_type]['predicter']
            E_max = all_results[model_type]['E_max']
            
            # Create custom color mapping for this model type using rainbow colors
            # Map E_max (the curve we're plotting) to the rainbow color scheme
            custom_color_mapping = {E_max: all_colors[model_type]}
            
            # Generate prediction and plot for this model type
            ax3 = plot_data.predict_and_plot(
                df,
                predicter.predict_errrate_df,
                predict_x_column_list=[fit_curve_column, fit_x_column],  # Use fitting parameters for prediction
                metric_column=metric,
                plot_curve_column=plot_curve_column,  # Use plotting parameters for display
                plot_curve_mask=[E_max],
                plot_x_column=plot_x_column,  # Use plotting x_column
                plot_use_line=True,
                plot_y_lambda=(lambda y: 1 - y) if metric == "R" else None,
                plot_x_scale="log",
                plot_y_scale="log",
                warmup_clip=warmup_clip,
                custom_color_mapping=custom_color_mapping,
                ax=ax3,
            )
            
            # Add scatter plot for this model type with config colors
            ax3 = plot_data.process_single_eval(
                df, 
                plot_x_column=plot_x_column,  # Use plotting x_column
                plot_eval_column=eval_name, 
                plot_metric=metric,
                plot_curve_column=plot_curve_column,  # Use plotting curve_column
                plot_curve_mask=[E_max],
                plot_use_scatter=True,  
                plot_use_legend=False,  # Disable individual legends, we'll add combined one later
                plot_legend_loc='upper right',
                plot_legend_bbox_to_anchor=(1, 0.5),
                scatter_alpha=0.5,
                scatter_size=20,
                scatter_marker="o",
                line_width=2,
                line_alpha=1,
                warmup_clip=warmup_clip,
                s_factor=1,
                k_spline=5,
                rolling_window=200,
                min_se=1e-6,
                x_inv_weight_power=0.3,
                custom_color_mapping=custom_color_mapping,
                ax=ax3,
            )

        # Add SOTA comparison lines
        # Data from user requirements
        sota_model_sizes = {
            "Sota-Qwen3-(Reason, Dense)": [0.6, 1.7, 4, 8, 14, 32],
            "Sota-Qwen3-(Reason, MOE)": [30, 235],
            "Sota-GPT-OSS-(Reason, MOE)": [20, 120],
        }

        sota_scores = {
            "Sota-Qwen3-(Reason, Dense)": [0.178, 0.288, 0.418, 0.366, 0.388, 0.412],
            "Sota-Qwen3-(Reason, MOE)": [0.528, 0.602],
            "Sota-GPT-OSS-(Reason, MOE)": [0.556, 0.660],
        }

        # Define colors for SOTA lines using rainbow scheme
        sota_colors = {
            "Sota-Qwen3-(Reason, Dense)": colors[2],  # Deep green
            "Sota-Qwen3-(Reason, MOE)": colors[3],    # Deep blue
            "Sota-GPT-OSS-(Reason, MOE)": colors[4],  # Purple
        }

        # Define line styles (solid for dense, dashed for MOE)
        sota_line_styles = {
            "Sota-Qwen3-(Reason, Dense)": '-',    # Solid line
            "Sota-Qwen3-(Reason, MOE)": '--',     # Dashed line
            "Sota-GPT-OSS-(Reason, MOE)": '--',   # Dashed line
        }

        # Plot each SOTA line
        for label in sota_model_sizes:
            x = np.array(sota_model_sizes[label]) * 1e9  # Convert to same units (billions to actual numbers)
            y = 1 - np.array(sota_scores[label])  # Convert scores to error rates (1 - score)
            
            # Use plot_basic to add the line
            ax3 = plot.plot_basic(
                x=x,
                y=y,
                use_line=True,
                use_scatter=True,
                line_alpha=1.0,
                line_width=2,
                scatter_alpha=0.8,
                scatter_s=20,
                scatter_marker='o',
                color=sota_colors[label],
                ax=ax3
            )
            
            # Apply line style by modifying the last added line
            if sota_line_styles[label] == '--':
                lines = ax3.get_lines()
                if lines:
                    lines[-1].set_linestyle('--')
            
        plot.plot_basic_settings(
            ax=ax3,
            x_scale="log",
            y_scale="log",
            x_label=config.DEFAULT_LABELS[plot_x_column],
            y_label=config.DEFAULT_LABELS[metric],
            # title=f"L(N) (D=max) - Base vs Instruct",
            title="Performance Scaling of Post-RFT vs. SOTA",
            x_tick_format="auto",
            y_tick_format="decimal",
            y_tick_spacing=0.1,
            use_legend=False,
            save_to_dir=None,  # Don't save individual plots, we'll save the combined one
            save_to_filename_prefix=None,
        )
        
        # Add combined legend manually
        legend_handles = []
        legend_labels = []
        
        # Add existing model types (Base and Instruct) using rainbow colors
        model_type_labels = {
            "base": "Ours-Qwen2.5-(Base, Dense)",
            "instruct": "Ours-Qwen2.5-(Instruct, Dense)"
        }
        
        for model_type in model_types:
            legend_handles.append(plt.Line2D([0], [0], color=all_colors[model_type], 
                                           linewidth=2, label=model_type_labels[model_type]))
            legend_labels.append(model_type_labels[model_type])
        
        # Add SOTA lines to legend
        for label in sota_model_sizes:
            line_style = sota_line_styles[label]
            legend_handles.append(plt.Line2D([0], [0], color=sota_colors[label], 
                                           linewidth=2.5, linestyle=line_style, 
                                           marker='o', markersize=6, label=label))
            legend_labels.append(label)
        
        ax3.legend(legend_handles, legend_labels, loc='best', fontsize=8)
        
        # Save the combined plot
        plt.tight_layout()
        plt.savefig(config.OUTPUT_BASE_DIR / f"fit_both_{eval_file_str}_{plot_curve_column}_{plot_x_column}_{metric}.pdf", bbox_inches='tight', dpi=300)
        print(f"Saved combined {metric} plot: {config.OUTPUT_BASE_DIR}/fit_both_{eval_file_str}_{plot_curve_column}_{plot_x_column}_{metric}.pdf")

if __name__ == "__main__":
    main()