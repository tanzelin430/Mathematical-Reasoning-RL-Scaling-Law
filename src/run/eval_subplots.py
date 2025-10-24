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
    
    # Load data
    df = data_proc.load_and_preprocess(config.CSV_MAP[data_source])
    
    # Remove step=0 data (because E=0 will cause log10(E)=-inf)
    df = df[df['step'] > 0].reset_index(drop=True)
    
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
                    # Prepare eval data for this specific evaluation
                    df_eval = df.copy()
                    df_eval = data_proc.prepare_eval_data(
                        df_eval,
                        eval_column=eval_name,
                        curve_column='N',
                        x_columns=[x_column],
                        calc_delta=metric.startswith('Delta'),
                        delta_base_step=1
                    )
                    
                    # Apply clipping if specified
                    if warmup_clip > 0:
                        df_eval = data_proc.apply_clip(
                            df_eval,
                            curve_column='N',
                            warmup_clip=warmup_clip,
                            warmup_clip_to=None,
                            ending_clip=0,
                            ending_clip_to=None
                        )
                    
                    # Plot raw scatter points
                    ax = plot.plot_curves(
                        df_eval,
                        curve_column='N',
                        x_column=x_column,
                        y_column=metric,
                        use_scatter=True,
                        use_line=False,
                        scatter_alpha=1.0,
                        scatter_size=8.0,
                        scatter_marker='o',
                        ax=axes[metric],
                    )
                    
                    # Add smooth curves
                    smooth_out_column = metric + "_smooth"
                    df_smooth = data_proc.smooth_df(
                        df_eval,
                        curve_column='N',
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
                        curve_column='N',
                        x_column=x_column,
                        y_column=smooth_out_column,
                        use_scatter=False,
                        use_line=True,
                        line_alpha=1.0,
                        line_width=2.0,
                        ax=axes[metric]
                    )
                    
                    # Apply basic settings and title
                    extra_settings = {}
                    if x_column == "E":
                        extra_settings['x_tick_subs'] = [5e3, 1e4, 5e4]
                    
                    plot.plot_basic_settings(
                        ax=axes[metric],
                        x_scale="log",
                        title=eval_config['plot_str'],
                        use_legend=True,
                        **extra_settings
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