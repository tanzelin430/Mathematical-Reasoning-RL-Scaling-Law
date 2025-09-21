#!/usr/bin/env python3
"""
Scaling Law Pipeline - Multi-Eval Analysis
Processes multiple test evals from Experiment1 data and generates scaling law plots for each eval
"""

import data_proc
import plot_data
import config
import plot

def main():
    data_source = "instruct"
    # data_source = "base"
    df = data_proc.load_and_preprocess(config.CSV_MAP[data_source])
    
    # ===========================
    # Plot Basic Curves
    # ===========================

    x_columns = ["C"]
    metrics = ["ErrRate"]


    # Filter out holdout_score from TEST_EVALS  
    test_evals_without_holdout = {k: v for k, v in config.TEST_EVALS.items() if k != 'holdout_score'}
    
    for x_column in x_columns:
        x_label = config.DEFAULT_LABELS[x_column]
        # Create metrics labels dictionary
        metrics_labels = {metric: config.DEFAULT_LABELS[metric] for metric in metrics}
        
        # Create subplots with getter function
        fig_axes, get_axes_for_eval = plot.create_multi_subplot_axes(
            metrics, len(test_evals_without_holdout), config.MULTI_FIGURE_COLUMNS, config.MULTI_FIGURE_SIZE
        )

        for i, (eval_name, eval_config) in enumerate(test_evals_without_holdout.items()):
            axes = get_axes_for_eval(i)
            
            # Plot each metric in the corresponding subplot
            for metric in metrics:
                metric_label = config.DEFAULT_LABELS[metric]
                if metric in axes:
                    # Check if delta calculation is needed
                    calc_delta = metric.startswith('Delta')
                    
                    plot_data.process_single_eval(
                        df.copy(), 
                        plot_x_column=x_column,
                        plot_eval_column=eval_name, 
                        plot_metric=metric,
                        plot_curve_column='N',
                        plot_x_label=x_label,
                        plot_y_label=metric_label,
                        plot_x_scale="log",
                        plot_title=eval_config['plot_str'],
                        plot_use_legend=True,
                        plot_use_scatter=True,
                        plot_use_line=False,
                        plot_smooth_use_scatter=False,
                        plot_smooth_use_line=True,
                        plot_legend_lambda=lambda x: plot.human_format_N(x),
                        ax=axes[metric],
                        delta_base_step=1,
                        calc_delta=calc_delta,
                        add_smooth=True,
                        smooth_monotonic=True,
                        warmup_frac_raw=config.WARMUP_CLIPPING_FACTOR_FOR_RAW,
                        warmup_frac_smooth=config.WARMUP_CLIPPING_FACTOR_FOR_SMOOTH,
                    )
            
        # Set figure labels
        plot.set_figure_labels(fig_axes, x_label, metrics_labels)
        
        # Apply full global legend layout for normal mode with formatted N labels
        plot.apply_global_legend_layout(fig_axes, sorted(df['N'].unique()), legend_lambda=lambda x: plot.human_format_N(x))
        
        # Save files - one file per metric
        for metric in metrics:
            metric_label = config.DEFAULT_LABELS[metric].replace(" ", "_").replace("(", "").replace(")", "")
            fig_axes[metric][0].savefig(config.OUTPUT_BASE_DIR / f"all_{data_source}_{x_column}_{metric}.pdf", dpi=300)
        print(f"\n saved {x_column} figures")

    print(f"\n Multi-subplot plots complete! Check {config.OUTPUT_BASE_DIR} for results")
    
if __name__ == "__main__":
    main()