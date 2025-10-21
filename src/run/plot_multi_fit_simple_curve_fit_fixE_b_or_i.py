#!/usr/bin/env python3
"""
Scaling Law Pipeline - Multi-Eval Analysis
Processes multiple test evals from Experiment1 data and generates scaling law plots for each eval
"""

from src.common import data_proc
from src.common import fit_models_simple
from src.common import plot_data
from src.common import config
from src.common import fit
from src.common import plot
import matplotlib.pyplot as plt
import numpy as np

def main():
    model_type = "base"
    # model_type = "instruct"
    # model_type = "llama-instruct"
    # model_type = "llama-base"
    df = data_proc.load_and_preprocess(config.CSV_MAP[model_type])
    
    # eval_name = "val/test_score/openai/gsm8k" # must be one of config.TEST_EVALS.keys()
    eval_name = "holdout_score"
    
    # 拟合参数：拟合 log_errrate = k(N) * log_E + E0(N)
    fit_curve_column = 'N'  # curve parameter for fitting
    fit_x_column = 'E'      # x variable for fitting
    
    # 绘图参数：固定E=E_max，以N为x轴绘图
    plot_curve_column = 'E' # curve parameter for plotting (will be fixed to E_max)
    plot_x_column = 'N'     # x variable for plotting
    
    E_max = df.groupby('N')['E'].max().min()

    print(f"\n=== Fitting: log_errrate = k({fit_curve_column}) * log_{fit_x_column} + E0({fit_curve_column}) ===")
    print(f"=== Plotting: Fixed {plot_curve_column}={E_max}, x-axis={plot_x_column} ===")
    
    # Use simple lookup table fitting
    predicter = fit.fit_log_errrate_simple(df, eval_name, fit_curve_column, fit_x_column)
        
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
        
    for metric in ["ErrRate"]: # "R", "ErrRate", "DeltaReward", "DeltaErrRate"
        # df_fit_plot = data_proc.apply_warmup_clip(df, curve_column="N", warmup_frac=config.WARMUP_CLIPPING_FACTOR_FOR_RAW)
        # ax = plot.plot_curves(df_fit_plot, curve_column=curve_column, x_column=x_column, y_column=metric+"_pred", use_line=True, x_scale="log")

        ax = plot_data.predict_and_plot(
            df,
            predicter.predict_errrate_df,
            predict_x_column_list=[fit_curve_column, fit_x_column],  # Use fitting parameters for prediction
            metric_column=metric,
            plot_curve_column=plot_curve_column,  # Use plotting parameters for display
            # plot_curve_mask=[14e9],
            plot_curve_mask=[E_max],
            plot_x_column=plot_x_column,  # Use plotting x_column
            plot_use_line=True,
            plot_y_lambda=(lambda y: 1 - y) if metric == "R" else None,
            # plot_use_delta=is_delta,
            # plot_delta_base_step=0,
            plot_x_scale="log",
            plot_y_scale="log",
            # ax=ax,
        )
            
        plot_data.process_single_eval(
            df, 
            plot_x_column=plot_x_column,  # Use plotting x_column
            plot_eval_column=eval_name, 
            plot_metric=metric,
            plot_curve_column=plot_curve_column,  # Use plotting curve_column
            plot_x_label=config.DEFAULT_LABELS[plot_x_column],
            plot_y_label=config.DEFAULT_LABELS[metric],
            # plot_curve_mask=[14e9],
            plot_curve_mask=[E_max],
            plot_use_scatter=True,  
            plot_x_scale="log",
            plot_y_scale="log",
            plot_title=f"L(N) (D=max)",
            plot_use_legend=False,
            plot_legend_loc='upper right',
            plot_legend_bbox_to_anchor=(1, 0.5),
            line_width=2,
            line_alpha=1,
            scatter_alpha=1,
            scatter_size=10,
            scatter_marker="s",
            # delta
            # delta_base_step=1,
            # smooth
            # add_smooth=True,
            # add_std=True,
            # smooth_monotonic=True,
            # smooth_increasing=None,
            # smooth_strict=False,
            s_factor=1,
            k_spline=5,
            rolling_window=200,
            min_se=1e-6,
            x_inv_weight_power=0.3,
            ax=ax,
            save_to_dir=config.OUTPUT_BASE_DIR, 
            save_to_filename_prefix=f"fit_{model_type}_",
        )

if __name__ == "__main__":
    main()