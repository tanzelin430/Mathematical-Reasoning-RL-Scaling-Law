#!/usr/bin/env python3
"""
Scaling Law Pipeline - Multi-Eval Analysis (Out-of-Domain vs In-Domain)
Plots evaluation performance for a specified model size with each eval as a separate line

Usage:
  python dataset_curves.py --eval-group in_domain --x-columns C --metrics ErrRate --data-source base -N 14e9
  python dataset_curves.py --eval-group out_of_domain --x-columns C N --metrics ErrRate Score --data-source instruct -N 14e9

Options:
- eval_group: 'out_of_domain' or 'in_domain'
- x_columns: Comma-separated list of x-axis variables (e.g., 'C' or 'C,N')
- metrics: Comma-separated list of metrics (e.g., 'ErrRate' or 'ErrRate,Score')
- data_source: Data source to use (e.g., 'base', 'instruct', 'llama-base', 'llama-instruct', 'exp2-base', 'exp2-instruct', 'grpo-base')
- N: Model size to analyze (e.g., 14e9). Must be one of the available model sizes from COLOR_MAPPING.

- Out-of-domain: SuperGPQA, Code, Logic
- In-domain: Holdout, GSM8K, Math, AIME, AMC

The script uses config.TEST_EVALS for evaluation metadata and generates plots 
with one line per evaluation for the specified model size.
"""

import argparse
from src.common import data_proc
from src.common import config
from src.common import plot
import matplotlib.pyplot as plt

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Multi-Eval Analysis for Scaling Laws')
    parser.add_argument('--eval-group', type=str, default='in_domain',
                        choices=['out_of_domain', 'in_domain'],
                        help='Evaluation group to plot (default: in_domain)')
    parser.add_argument('--x-columns', type=str, default='C',
                        help='Comma-separated list of x-axis variables (default: C)')
    parser.add_argument('--metrics', type=str, default='ErrRate',
                        help='Comma-separated list of metrics (default: ErrRate)')
    parser.add_argument('--data-source', type=str, default='base',
                        choices=['base', 'instruct', 'llama-base', 'llama-instruct', 
                                'exp2-base', 'exp2-instruct', 'grpo-base'],
                        help='Data source to use (default: base)')
    parser.add_argument('--warmup-clip', type=int, default=None,
                        help='Number of warmup steps to clip (default: None)')
    parser.add_argument('--warmup-clip-to', type=int, default=None,
                        help='Warmup clip to step (default: None)')
    parser.add_argument('--ending-clip', type=int, default=None,
                        help='Number of ending steps to clip (default: None)')
    parser.add_argument('--ending-clip-to', type=int, default=None,
                        help='Ending clip to step (default: None)')
    parser.add_argument('-N', '--model-size', type=float, default=14e9,
                        choices=list(config.COLOR_MAPPING.keys()),
                        help='Model size N to analyze (default: 14e9). Must be one of the available model sizes.')
    args = parser.parse_args()
    
    # Parse comma-separated arguments
    eval_group = args.eval_group
    x_columns = [x.strip() for x in args.x_columns.split(',')]
    metrics = [m.strip() for m in args.metrics.split(',')]
    data_source = args.data_source
    warmup_clip = args.warmup_clip
    warmup_clip_to = args.warmup_clip_to
    ending_clip = args.ending_clip
    ending_clip_to = args.ending_clip_to
    N = args.model_size
    
    print(f"Configuration:")
    print(f"  Eval group: {eval_group}")
    print(f"  X columns: {x_columns}")
    print(f"  Metrics: {metrics}")
    print(f"  Data source: {data_source}")
    print(f"  Warmup clip num: {warmup_clip}")
    print(f"  Model size N: {N}")
    print()
    
    # Load data
    df = data_proc.load_and_preprocess(config.CSV_MAP[data_source])
    
    # Get physical dimensions for this data source
    physical_dimensions = config.get_physical_dimensions(data_source)
    physical_curve_column = physical_dimensions[0]
    physical_x_column = physical_dimensions[1]
    
    # Filter for specified model size only
    df = df[df['N'] == N].copy()
    if len(df) == 0:
        print(f"No data found for {plot.human_format_N(N)} model!")
        return
    
    # Remove step=0 data (because E=0 will cause log10(E)=-inf)
    # Note: merge_duplicate_steps is done inside prepare_eval_data
    df = df[df['step'] > 0].reset_index(drop=True)
    
    # Define evaluation group keys (using config.TEST_EVALS for metadata)
    eval_group_keys = {
        'out_of_domain': [
            'holdout_score',
            'val/test_score/stem__supergpqa',
            'val/test_score/codegen__humaneval', 
            'val/test_score/logic__zebra_puzzle_dataset'
        ],
        'in_domain': [
            'holdout_score',
            'val/test_score/openai/gsm8k',
            'val/test_score/math__math',
            'val/test_score/aime2024',
            'val/test_score/aimeamc2023'
        ]
    }
    
    # Get evaluation keys for eval group and build test_evals from config
    selected_keys = eval_group_keys[eval_group]
    test_evals = {k: config.TEST_EVALS[k] for k in selected_keys if k in config.TEST_EVALS}
    
    # Define evaluation order (based on plot._eval_order)
    eval_order_list = [
        "Logic - Zebra Puzzle",
        "SuperGPQA", 
        "AIME2024",
        "Holdout Validation",
        "AMC2023",
        "MATH-500",
        "GSM8K",
        "CodeGen - HumanEval"
    ]
    
    for x_column in x_columns:
        x_label = config.DEFAULT_LABELS[x_column]
        
        for metric in metrics:
            metric_label = config.DEFAULT_LABELS[metric]
            
            
            # Create single figure
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            
            # Collect legend handles and labels
            legend_handles = []
            legend_labels = []
            
            # Sort evaluations by the defined order
            ordered_evals = []
            for eval_label in eval_order_list:
                for eval_name, eval_config in test_evals.items():
                    if eval_config['plot_str'] == eval_label:
                        ordered_evals.append((eval_name, eval_config))
                        break
            
            # Add any remaining evals not in the order list
            for eval_name, eval_config in test_evals.items():
                if not any(eval_config['plot_str'] == item[1]['plot_str'] for item in ordered_evals):
                    ordered_evals.append((eval_name, eval_config))
            
            # Plot each evaluation as a separate line
            for i, (eval_name, eval_config) in enumerate(ordered_evals):
                # Check if evaluation column exists in data
                if eval_name not in df.columns:
                    print(f"Warning: {eval_name} not found in data, skipping...")
                    continue
                
                # Prepare eval data for this specific evaluation
                df_eval = df.copy()
                df_eval['eval_curve'] = eval_name  # Use eval name as curve identifier
                
                df_eval = data_proc.prepare_eval_data(
                    df_eval,
                    eval_column=eval_name,
                    curve_column='eval_curve',
                    x_column=x_column,
                    calc_delta=metric.startswith('Delta'),
                    delta_base_step=1
                )
                
                # Apply clipping if specified
                df_eval = data_proc.apply_clip(
                    df_eval,
                    curve_column='eval_curve',
                    warmup_clip=warmup_clip,
                    warmup_clip_to=warmup_clip_to,
                    ending_clip=ending_clip,
                    ending_clip_to=ending_clip_to,
                )
                
                # Create color mapping for this eval
                eval_color = config.get_color_for_curve(eval_name)
                temp_color_mapping = {eval_name: eval_color}
                
                # Plot raw scatter points
                ax = plot.plot_curves(
                    df_eval,
                    curve_column='eval_curve',
                    x_column=x_column,
                    y_column=metric,
                    use_scatter=True,
                    use_line=False,
                    scatter_alpha=1.0,
                    scatter_size=8.0,
                    scatter_marker='o',
                    line_alpha=1.0,
                    line_width=2.0,
                    custom_color_mapping=temp_color_mapping,
                    ax=ax,
                )
                
                # Add smooth curves
                smooth_out_column = metric + "_smooth"
                df_smooth = data_proc.smooth_df(
                    df_eval,
                    curve_column='eval_curve',
                    col_x=x_column,
                    col_y=metric,
                    col_y_out=smooth_out_column,
                    monotonic=True,
                    increasing=None,
                    strict=False,
                    s_factor=1,
                    k_spline=4,
                    rolling_window=200,
                    min_se=1e-7,
                    x_inv_weight_power=0,
                    use_linear=False
                )
                
                ax = plot.plot_curves(
                    df_smooth,
                    curve_column='eval_curve',
                    x_column=x_column,
                    y_column=smooth_out_column,
                    use_scatter=False,
                    use_line=True,
                    line_alpha=1.0,
                    line_width=2.0,
                    custom_color_mapping=temp_color_mapping,
                    ax=ax
                )
                
                # Collect legend handles and labels in order
                lines = ax.get_lines()
                if lines:
                    # Get the last line that was plotted (the smooth line)
                    handle = lines[-1]
                    label = eval_config['plot_str']
                    legend_handles.append(handle)
                    legend_labels.append(label)
            
            # Apply plot_basic_settings once after all plotting is complete
            plot.plot_basic_settings(
                ax=ax,
                x_scale="log",
                # y_scale="log",
                x_label=x_label,
                y_label=metric_label,
                title=f"{eval_group.replace('_', ' ').title()} Evaluations ({plot.human_format_N(N)} {data_source.title()} Model)",
                # x_tick_spacing=0.5,
                # y_tick_spacing=0.2,
                y_tick_subs=[0, 0.2, 0.4, 0.6, 0.8, 1],
                use_legend=True,
                legend_handles_labels=(legend_handles, legend_labels),
                # x_tick_on_data=True,
                # y_tick_on_data=True,
                x_tick_subs_log=[0.4, 1]
            )
            
            # Save figure
            metric_label_clean = metric_label.replace(" ", "_").replace("(", "").replace(")", "")
            filename = config.OUTPUT_BASE_DIR / f"{plot.human_format_N(N)}_{data_source}_{eval_group}_{x_column}_{metric_label_clean}.pdf"
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"\nSaved {eval_group} figure: {filename}")
            
            plt.close(fig)

    print(f"\nMulti-eval comparison plot complete for {eval_group}!")
    
if __name__ == "__main__":
    main()