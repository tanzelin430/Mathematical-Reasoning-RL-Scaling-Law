#!/usr/bin/env python3
"""
Scaling Law Pipeline - Multi-Eval Analysis
Processes multiple test evals from Experiment1 data and generates scaling law plots for each eval

Usage:
  python run_plot_multi-subplot.py --data-source base --x-columns E --metrics ErrRate --warmup-clip 10
  python run_plot_multi-subplot.py --data-source instruct --x-columns C,N --metrics ErrRate,Score --warmup-clip 5
"""

import argparse
from src.common import data_proc
from src.common import plot_data
from src.common import config
from src.common import plot

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Multi-Eval Analysis for Scaling Laws')
    parser.add_argument('--data-source', type=str, default='base',
                        choices=['base', 'instruct', 'llama-base', 'llama-instruct', 
                                'exp2-base', 'exp2-instruct', 'grpo-base'],
                        help='Data source to use (default: base)')
    parser.add_argument('--x-columns', type=str, default='E',
                        help='Comma-separated list of x-axis variables (default: E)')
    parser.add_argument('--metrics', type=str, default='ErrRate',
                        help='Comma-separated list of metrics (default: ErrRate)')
    parser.add_argument('--warmup-clip', type=int, default=10,
                        help='Number of warmup steps to clip (default: 10)')
    args = parser.parse_args()
    
    # Parse comma-separated arguments
    data_source = args.data_source
    x_columns = [x.strip() for x in args.x_columns.split(',')]
    metrics = [m.strip() for m in args.metrics.split(',')]
    warmup_clip = args.warmup_clip
    
    print(f"Configuration:")
    print(f"  Data source: {data_source}")
    print(f"  X columns: {x_columns}")
    print(f"  Metrics: {metrics}")
    print(f"  Warmup clip num: {warmup_clip}")
    print()
    
    df = data_proc.load_and_preprocess(config.CSV_MAP[data_source])
    
    # ===========================
    # Plot Basic Curves
    # ===========================

    # Filter out response_length from TEST_EVALS  
    exclude_keys = ['response_length']
    eval_map = {k: config.TEST_EVALS[k] for k in config.TEST_EVALS if k not in exclude_keys}
    
    for x_column in x_columns:
        x_label = config.DEFAULT_LABELS[x_column]
        # Create metrics labels dictionary
        metrics_labels = {metric: config.DEFAULT_LABELS[metric] for metric in metrics}
        
        # Create subplots with getter function
        fig_axes, get_axes_for_eval = plot.create_multi_subplot_axes(
            metrics, len(eval_map), config.MULTI_FIGURE_COLUMNS, config.MULTI_FIGURE_SIZE
        )

        for i, (eval_name, eval_config) in enumerate(eval_map.items()):
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
                        # plot_x_label=x_label,
                        # plot_y_label=metric_label,
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
                        warmup_clip=warmup_clip,
                    )
                    if x_column == "E":
                        plot.plot_basic_settings(
                            ax=axes[metric],
                            x_tick_subs=[5e3, 1e4, 5e4],
                        )
        
        # Set figure labels
        plot.set_figure_labels(fig_axes, x_label, metrics_labels)
        
        # Apply full global legend layout for normal mode with formatted N labels
        plot.apply_global_legend_layout(fig_axes, sorted(df['N'].unique()), legend_lambda=lambda x: plot.human_format_N(x))
        
        # Save files - one file per metric
        for metric in metrics:
            metric_label = config.DEFAULT_LABELS[metric].replace(" ", "_").replace("(", "").replace(")", "")
            filenpath = config.OUTPUT_BASE_DIR / f"all_{data_source}_{x_column}_{metric}.pdf"
            fig_axes[metric][0].savefig(filenpath, dpi=300)
            print(f"\n saved {x_column} {metric} figures: {filenpath}")

    print(f"\n Multi-subplot plots complete!")
    
if __name__ == "__main__":
    main()