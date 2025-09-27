#!/usr/bin/env python3
"""
Scaling Law Pipeline - Multi-Eval Analysis (CLI Version)
Processes multiple test evals from experiment data and generates scaling law plots for each eval
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Union

import data_proc
import fit_models_simple
import plot_data
import config
import fit
import plot
import matplotlib.pyplot as plt
import numpy as np


def parse_list_argument(value: str) -> List[str]:
    """Parse comma-separated string into list"""
    if not value:
        return []
    return [item.strip() for item in value.split(',') if item.strip()]


def parse_curve_mask(value_str: str) -> List[Union[int, float]]:
    """Parse curve mask supporting scientific notation like 7e9, 1.5e9"""
    values = []
    for item in value_str.split(','):
        item = item.strip()
        try:
            # Support scientific notation
            if 'e' in item.lower():
                values.append(float(item))
            else:
                values.append(int(item))
        except ValueError:
            raise ValueError(f"Invalid curve mask value: {item}")
    return values


def parse_highlight_curves(value_str: str) -> List[Union[int, float]]:
    """Parse highlight curves supporting scientific notation"""
    if not value_str:
        return []
    return parse_curve_mask(value_str)


def validate_args(args):
    """Validate command line arguments"""
    # Validate data_source
    if args.data_source not in config.CSV_MAP:
        raise ValueError(f"Invalid data-source: {args.data_source}. "
                        f"Must be one of: {list(config.CSV_MAP.keys())}")
    
    # Validate eval
    if args.eval not in config.TEST_EVALS:
        raise ValueError(f"Invalid eval: {args.eval}. "
                        f"Must be one of: {list(config.TEST_EVALS.keys())}")
    
    # Validate curve columns
    valid_curves = list(config.DEFAULT_LABELS.keys())
    if args.plot_curve_column not in valid_curves:
        raise ValueError(f"Invalid plot-curve: {args.plot_curve_column}. "
                        f"Must be one of: {valid_curves}, or add to config.DEFAULT_LABELS")
    
    if args.fit and args.fit_curve_column not in valid_curves:
        raise ValueError(f"Invalid fit-curve: {args.fit_curve_column}. "
                        f"Must be one of: {valid_curves}, or add to config.DEFAULT_LABELS")
    
    # Validate x columns (after processing) - use config.DEFAULT_LABELS for dynamic validation
    valid_x_columns = list(config.DEFAULT_LABELS.keys())
    if args.plot_x_columns:
        for x_col in args.plot_x_columns:
            if x_col not in valid_x_columns:
                raise ValueError(f"Invalid plot-x column: {x_col}. "
                                f"Must be one of: {valid_x_columns}, or add to config.DEFAULT_LABELS")
    
    if args.fit and args.fit_x_column not in valid_x_columns:
        raise ValueError(f"Invalid fit-x column: {args.fit_x_column}. "
                        f"Must be one of: {valid_x_columns}, or add to config.DEFAULT_LABELS")
    
    # Validate metrics - use config.DEFAULT_LABELS for dynamic validation
    valid_metrics = list(config.DEFAULT_LABELS.keys())
    for metric in args.plot_metrics:
        if metric not in valid_metrics:
            raise ValueError(f"Invalid metric: {metric}. "
                            f"Must be one of: {valid_metrics}, or add to config.DEFAULT_LABELS")
    
    if args.fit:
        for metric in args.fit_metrics:
            if metric not in valid_metrics:
                raise ValueError(f"Invalid fit metric: {metric}. "
                                f"Must be one of: {valid_metrics}, or add to config.DEFAULT_LABELS")


def create_argument_parser():
    """Create and configure argument parser"""
    parser = argparse.ArgumentParser(
        description="Scaling Law Pipeline - Multi-Eval Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  %(prog)s --data-source exp2-instruct --plot-curve Tau -x E --eval holdout_score --metric ErrRate

  # With fitting
  %(prog)s --data-source exp2-instruct --plot-curve Tau -x E --eval holdout_score --metric ErrRate \\
           --fit --fit-curve Tau --fit-x E --fit-metric ErrRate

  # Multiple metrics and x columns with highlighting
  %(prog)s --data-source exp2-base --plot-curve N -x C,E --eval holdout_score --metric ErrRate,R \\
           --plot-curve-mask 1e9,3e9,7e9 --highlight-curves-predict 1e9 --highlight-curves-plot 3e9

  # Load configuration from file
  %(prog)s --config-file config.json
        """
    )

    # Required arguments (except when using config file)
    parser.add_argument('--data-source', 
                       choices=list(config.CSV_MAP.keys()),
                       help='Data source to use')
    
    parser.add_argument('--plot-curve', dest='plot_curve_column',
                       choices=['N', 'Tau'],
                       help='Column to use for curve grouping in plots')
    
    parser.add_argument('-x', '--plot-x', dest='plot_x_columns',
                       action='append',
                       help='X columns for plotting (can be specified multiple times or comma-separated)')
    
    parser.add_argument('--eval',
                       choices=list(config.TEST_EVALS.keys()),
                       help='Evaluation metric to analyze')
    
    parser.add_argument('--metric', dest='plot_metrics',
                       action='append',
                       help='Metrics to plot (can be specified multiple times or comma-separated)')

    # Fitting arguments
    fit_group = parser.add_argument_group('fitting options')
    fit_group.add_argument('--fit', action='store_true',
                          help='Enable model fitting')
    fit_group.add_argument('--fit-curve', dest='fit_curve_column',
                          choices=['N', 'Tau'],
                          help='Column to use for curve grouping in fitting (default: same as --plot-curve)')
    fit_group.add_argument('--fit-x', dest='fit_x_column',
                          choices=list(config.DEFAULT_LABELS.keys()),
                          help='X column for fitting (default: first plot-x column)')
    fit_group.add_argument('--fit-metric', dest='fit_metrics',
                          action='append',
                          help='Metrics to fit (default: same as --metric)')
    fit_group.add_argument('--fit-load', dest='fit_load_path',
                          help='Path to load pre-fitted model')
    fit_group.add_argument('--fit-save', dest='fit_save_path',
                          help='Path to save fitted model')
    fit_group.add_argument('--fit-plot-k-e0', dest='fit_plot_k_E0',
                          action='store_true', default=False,
                          help='Plot k and E0 scatter plots (default: False)')

    # Plot configuration
    plot_group = parser.add_argument_group('plot configuration')
    plot_group.add_argument('--plot-curve-mask', dest='plot_curve_mask',
                           help='Curves to include in plots (comma-separated, supports scientific notation)')
    plot_group.add_argument('--highlight-curves-predict', dest='highlight_curves_predict',
                           help='Curves to highlight in predict_and_plot (comma-separated, supports scientific notation). If not set, no highlighting in predict phase.')
    plot_group.add_argument('--highlight-curves-plot', dest='highlight_curves_plot',
                           help='Curves to highlight in process_single_eval (comma-separated, supports scientific notation). If not set, no highlighting in plot phase.')
    plot_group.add_argument('--highlight-line-alpha', dest='highlight_line_alpha',
                           type=float, default=1.0,
                           help='Opacity for highlighted curves (default: 1.0)')
    plot_group.add_argument('--highlight-line-width', dest='highlight_line_width',
                           type=float, default=2.0,
                           help='Line width for highlighted curves (default: 2.0)')
    plot_group.add_argument('--line-alpha', dest='line_alpha',
                           type=float, default=1.0,
                           help='Line opacity (default: 1.0)')
    plot_group.add_argument('--line-width', dest='line_width',
                           type=float, default=2.0,
                           help='Line width (default: 2.0)')
    plot_group.add_argument('--scatter-alpha', dest='scatter_alpha',
                           type=float, default=0.3,
                           help='Scatter point opacity (default: 0.3)')
    plot_group.add_argument('--scatter-size', dest='scatter_size',
                           type=float, default=8.0,
                           help='Scatter point size (default: 8.0)')
    plot_group.add_argument('--scatter-marker', dest='scatter_marker',
                           default='o',
                           help='Scatter point marker style (default: o)')

    # Plot style
    style_group = parser.add_argument_group('plot style')
    style_group.add_argument('--plot-use-scatter', dest='plot_use_scatter',
                            type=lambda x: x.lower() in ['true', '1', 'yes'], default=True,
                            help='Use scatter plots (default: True)')
    style_group.add_argument('--plot-use-line', dest='plot_use_line',
                            type=lambda x: x.lower() in ['true', '1', 'yes'], default=True,
                            help='Use line plots (default: True)')
    style_group.add_argument('--no-scatter', dest='plot_use_scatter',
                            action='store_false',
                            help='Disable scatter plots')
    style_group.add_argument('--no-line', dest='plot_use_line',
                            action='store_false',
                            help='Disable line plots')
    style_group.add_argument('--plot-x-scale', dest='plot_x_scale',
                            choices=['linear', 'log'], default=None,
                            help='X-axis scale (default: linear)')
    style_group.add_argument('--plot-y-scale', dest='plot_y_scale',
                            choices=['linear', 'log'], default=None,
                            help='Y-axis scale (default: linear)')
    style_group.add_argument('--plot-use-legend', dest='plot_use_legend',
                            action='store_true', default=False,
                            help='Show legend (default: True)')
    style_group.add_argument('--plot-legend-loc', dest='plot_legend_loc',
                            default='best',
                            help='Legend location (default: best)')
    style_group.add_argument('--plot-legend-bbox-to-anchor', dest='plot_legend_bbox_to_anchor',
                            type=str, default=None,
                            help='Legend bbox_to_anchor as comma-separated values (e.g., "1.05,1" or "1,0,0.5,1")')
    style_group.add_argument('--plot-title', dest='plot_title',
                            help='Fixed title for all plots. If specified, overrides all other title options.')
    style_group.add_argument('--plot-title-template', dest='plot_title_template',
                            help='Title template. Variables: {eval}, {fit_x}, {plot_x}, {metric}. If not specified, auto-generated.')
    style_group.add_argument('--plot-titles', dest='plot_titles',
                            help='JSON object mapping x_column to custom titles, e.g. \'{"C": "Compute Scaling", "E": "Data Scaling"}\'')
    style_group.add_argument('--plot-x-label', dest='plot_x_label',
                            help='Custom x-axis label. If not specified, uses config.DEFAULT_LABELS')
    style_group.add_argument('--plot-y-label', dest='plot_y_label',
                            help='Custom y-axis label. If not specified, uses config.DEFAULT_LABELS')
    style_group.add_argument('--x-tick-spacing', dest='x_tick_spacing',
                            type=float, default=None,
                            help='X-axis tick spacing (e.g., 0.1, 0.2)')
    style_group.add_argument('--y-tick-spacing', dest='y_tick_spacing',
                            type=float, default=None,
                            help='Y-axis tick spacing (e.g., 0.1, 0.2)')
    style_group.add_argument('--x-grid-spacing', dest='x_grid_spacing',
                            type=float, default=None,
                            help='X-axis grid line spacing')
    style_group.add_argument('--y-grid-spacing', dest='y_grid_spacing',
                            type=float, default=None,
                            help='Y-axis grid line spacing')
    style_group.add_argument('--x-tick-format', dest='x_tick_format',
                            choices=['auto', 'decimal', 'sci', 'plain'], default=None,
                            help='X-axis tick format (default: None/plain)')
    style_group.add_argument('--y-tick-format', dest='y_tick_format',
                            choices=['auto', 'decimal', 'sci', 'plain'], default='auto',
                            help='Y-axis tick format (default: auto)')
    style_group.add_argument('--x-tick-subs', dest='x_tick_subs',
                            type=str, default=None,
                            help='Custom x-axis tick positions (comma-separated list, e.g., "1000,5000,10000")')
    style_group.add_argument('--y-tick-subs', dest='y_tick_subs',
                            type=str, default=None,
                            help='Custom y-axis tick positions (comma-separated list)')
    style_group.add_argument('--x-tick-subs-log', dest='x_tick_subs_log',
                            type=str, default=None,
                            help='Custom x-axis log tick multipliers (comma-separated list, e.g., "1,2,5")')
    style_group.add_argument('--y-tick-subs-log', dest='y_tick_subs_log',
                            type=str, default=None,
                            help='Custom y-axis log tick multipliers (comma-separated list)')

    # Smoothing configuration
    smooth_group = parser.add_argument_group('smoothing options')
    smooth_group.add_argument('--add-smooth', dest='add_smooth',
                             action='store_true', default=False,
                             help='Enable smoothing (default: False)')
    smooth_group.add_argument('--add-std', dest='add_std',
                             action='store_true', default=False,
                             help='Add standard deviation bands (default: False)')
    smooth_group.add_argument('--smooth-monotonic', dest='smooth_monotonic',
                             action='store_true', default=True,
                             help='Force monotonic smoothing (default: True)')
    smooth_group.add_argument('--smooth-increasing', dest='smooth_increasing',
                             type=str, default='None',
                             help='Force increasing trend (None/True/False, default: None)')
    smooth_group.add_argument('--smooth-strict', dest='smooth_strict',
                             action='store_true', default=False,
                             help='Strict monotonic constraints (default: False)')

    # Advanced parameters
    advanced_group = parser.add_argument_group('advanced parameters')
    advanced_group.add_argument('--warmup-clip-frac', dest='warmup_clip_factor_raw',
                               type=float, default=0.0,
                               help='Warmup clipping fraction for raw data (default: 0.1)')
    advanced_group.add_argument('--warmup-clip-frac-smooth', dest='warmup_clip_factor_smooth',
                               type=float, default=0.0,
                               help='Warmup clipping fraction for smooth data (default: 0.0)')
    advanced_group.add_argument('--warmup-clip', dest='warmup_clip_raw',
                               type=int, default=None,
                               help='Warmup clipping by absolute number of data points for raw data (clips first N points per curve, overrides --warmup-clip-frac)')
    advanced_group.add_argument('--warmup-clip-smooth', dest='warmup_clip_smooth',
                               type=int, default=None,
                               help='Warmup clipping by absolute number of data points for smooth data (clips first N points per curve, overrides --warmup-clip-frac-smooth)')
    advanced_group.add_argument('--ending-clip', dest='ending_clip_raw',
                               type=int, default=None,
                               help='Ending clipping by absolute step value for raw data (clips steps > ending_step per curve)')
    advanced_group.add_argument('--ending-clip-smooth', dest='ending_clip_smooth',
                               type=int, default=None,
                               help='Ending clipping by absolute step value for smooth data (clips steps > ending_step per curve)')
    advanced_group.add_argument('--delta-base-step', dest='delta_base_step',
                               type=int, default=1,
                               help='Base step for delta calculation (default: 1)')
    advanced_group.add_argument('--s-factor', dest='s_factor',
                               type=int, default=1,
                               help='Smoothing factor (default: 1)')
    advanced_group.add_argument('--k-spline', dest='k_spline',
                               type=int, default=5,
                               help='Spline order (default: 5)')
    advanced_group.add_argument('--rolling-window', dest='rolling_window',
                               type=int, default=200,
                               help='Rolling window size (default: 200)')
    advanced_group.add_argument('--min-se', dest='min_se',
                               type=float, default=1e-6,
                               help='Minimum standard error (default: 1e-6)')
    advanced_group.add_argument('--x-inv-weight-power', dest='x_inv_weight_power',
                               type=float, default=0.3,
                               help='X inverse weight power (default: 0.3)')

    # Output configuration
    output_group = parser.add_argument_group('output options')
    output_group.add_argument('--output-dir', dest='output_base_dir',
                             type=Path, default=config.OUTPUT_BASE_DIR,
                             help='Output directory (default: outputs/)')
    output_group.add_argument('--output-prefix', dest='output_prefix',
                             help='Output filename prefix (default: auto-generated)')

    # Configuration file
    parser.add_argument('--config-file', dest='config_file',
                       help='Load configuration from JSON file (for batch processing)')

    return parser


def process_parsed_args(args):
    """Process and normalize parsed arguments"""
    # Handle comma-separated values for list arguments
    if args.plot_x_columns:
        # If already a list (from config file) with no commas in individual items, keep as-is
        if isinstance(args.plot_x_columns, list) and all(isinstance(x, str) and ',' not in x for x in args.plot_x_columns):
            # Already a proper list from config file
            pass
        else:
            # Flatten list and parse comma-separated values from CLI
            flat_list = []
            for item in args.plot_x_columns:
                flat_list.extend(parse_list_argument(item))
            args.plot_x_columns = flat_list
    
    if args.plot_metrics:
        # If already a list (from config file) with no commas in individual items, keep as-is
        if isinstance(args.plot_metrics, list) and all(isinstance(x, str) and ',' not in x for x in args.plot_metrics):
            # Already a proper list from config file
            pass
        else:
            # Flatten list and parse comma-separated values from CLI
            flat_list = []
            for item in args.plot_metrics:
                flat_list.extend(parse_list_argument(item))
            args.plot_metrics = flat_list

    if args.fit_metrics:
        # If already a list (from config file) with no commas in individual items, keep as-is
        if isinstance(args.fit_metrics, list) and all(isinstance(x, str) and ',' not in x for x in args.fit_metrics):
            # Already a proper list from config file
            pass
        else:
            # Flatten list and parse comma-separated values from CLI
            flat_list = []
            for item in args.fit_metrics:
                flat_list.extend(parse_list_argument(item))
            args.fit_metrics = flat_list

    # Set defaults for fit parameters
    if args.fit:
        if not args.fit_curve_column:
            args.fit_curve_column = args.plot_curve_column
        if not args.fit_x_column:
            args.fit_x_column = args.plot_x_columns[0] if args.plot_x_columns else None
        if not args.fit_metrics:
            args.fit_metrics = args.plot_metrics

    # Parse curve masks and highlight curves
    if args.plot_curve_mask:
        if isinstance(args.plot_curve_mask, str):
            args.plot_curve_mask = parse_curve_mask(args.plot_curve_mask)
        # else: already a list from config file
    else:
        args.plot_curve_mask = None  # Default

    # Parse highlight curve arguments
    if args.highlight_curves_predict:
        if isinstance(args.highlight_curves_predict, str):
            args.highlight_curves_predict = parse_highlight_curves(args.highlight_curves_predict)
        # else: already a list from config file
    else:
        args.highlight_curves_predict = None  # Default
    
    if args.highlight_curves_plot:
        if isinstance(args.highlight_curves_plot, str):
            args.highlight_curves_plot = parse_highlight_curves(args.highlight_curves_plot)
        # else: already a list from config file
    else:
        args.highlight_curves_plot = None  # Default

    # Parse plot_titles JSON
    if args.plot_titles:
        if isinstance(args.plot_titles, str):
            try:
                args.plot_titles = json.loads(args.plot_titles)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in plot_titles: {e}")
        # else: already a dict from config file
    else:
        args.plot_titles = None  # Default

    # Handle smooth_increasing special case
    if args.smooth_increasing == 'None':
        args.smooth_increasing = None
    elif args.smooth_increasing == 'True':
        args.smooth_increasing = True
    elif args.smooth_increasing == 'False':
        args.smooth_increasing = False

    # Parse tick_subs parameters
    def parse_tick_subs(value_str):
        """Parse comma-separated tick positions supporting scientific notation"""
        if not value_str:
            return None
        values = []
        for item in value_str.split(','):
            item = item.strip()
            try:
                # Support scientific notation
                values.append(float(item))
            except ValueError:
                raise ValueError(f"Invalid tick position value: {item}")
        return values

    # Parse x_tick_subs
    if args.x_tick_subs:
        if isinstance(args.x_tick_subs, str):
            args.x_tick_subs = parse_tick_subs(args.x_tick_subs)
        # else: already a list from config file
    else:
        args.x_tick_subs = None

    # Parse y_tick_subs
    if args.y_tick_subs:
        if isinstance(args.y_tick_subs, str):
            args.y_tick_subs = parse_tick_subs(args.y_tick_subs)
        # else: already a list from config file
    else:
        args.y_tick_subs = None

    # Parse x_tick_subs_log
    if args.x_tick_subs_log:
        if isinstance(args.x_tick_subs_log, str):
            args.x_tick_subs_log = parse_tick_subs(args.x_tick_subs_log)
        # else: already a list from config file
    else:
        args.x_tick_subs_log = None

    # Parse y_tick_subs_log
    if args.y_tick_subs_log:
        if isinstance(args.y_tick_subs_log, str):
            args.y_tick_subs_log = parse_tick_subs(args.y_tick_subs_log)
        # else: already a list from config file
    else:
        args.y_tick_subs_log = None

    # Parse plot_legend_bbox_to_anchor
    if args.plot_legend_bbox_to_anchor:
        if isinstance(args.plot_legend_bbox_to_anchor, str):
            try:
                # Parse comma-separated bbox values
                bbox_values = [float(x.strip()) for x in args.plot_legend_bbox_to_anchor.split(',')]
                if len(bbox_values) == 2:
                    args.plot_legend_bbox_to_anchor = tuple(bbox_values)
                elif len(bbox_values) == 4:
                    args.plot_legend_bbox_to_anchor = tuple(bbox_values)
                else:
                    raise ValueError("bbox_to_anchor must have 2 or 4 values")
            except ValueError as e:
                raise ValueError(f"Invalid bbox_to_anchor format: {e}")
        # else: already a tuple from config file
    else:
        args.plot_legend_bbox_to_anchor = None

    # Set default output prefix
    if not args.output_prefix:
        args.output_prefix = f"fit_{args.data_source}_"

    return args


def convert_warmup_clip_to_factor(df, curve_column, warmup_clip_abs):
    """
    Convert absolute warmup_clip to relative warmup_clip_factor per curve.
    
    Args:
        df: DataFrame with curves
        curve_column: Column defining the curves 
        warmup_clip_abs: Absolute number of steps to clip
        
    Returns:
        Dictionary mapping curve_id to warmup_clip_factor
    """
    if warmup_clip_abs is None or warmup_clip_abs <= 0:
        return None
        
    curve_factors = {}
    for curve_id, group_df in df.groupby(curve_column):
        n_total = len(group_df)
        if n_total > 0:
            warmup_clip_factor = min(warmup_clip_abs / n_total, 1.0)  # Cap at 1.0
            curve_factors[curve_id] = warmup_clip_factor
        else:
            curve_factors[curve_id] = 0.0
    
    return curve_factors


def run_scaling_analysis(args):
    """Run the scaling law analysis with given arguments"""
    print(f"Loading data for data_source: {args.data_source}")
    df = data_proc.load_and_preprocess(config.CSV_MAP[args.data_source])
    
    # Handle absolute vs relative warmup clipping
    if args.warmup_clip_raw is not None:
        print(f"Using absolute warmup clipping: warmup_clip_raw={args.warmup_clip_raw} steps")
        # For fitting, we still need to convert to a reasonable factor
        curve_factors_raw = convert_warmup_clip_to_factor(df, args.plot_curve_column, args.warmup_clip_raw)
        if curve_factors_raw:
            args.warmup_clip_factor_raw = sum(curve_factors_raw.values()) / len(curve_factors_raw)
        else:
            args.warmup_clip_factor_raw = 0.0
    
    if args.warmup_clip_smooth is not None:
        print(f"Using absolute warmup clipping: warmup_clip_smooth={args.warmup_clip_smooth} steps")
        # For fitting, we still need to convert to a reasonable factor
        curve_factors_smooth = convert_warmup_clip_to_factor(df, args.plot_curve_column, args.warmup_clip_smooth)
        if curve_factors_smooth:
            args.warmup_clip_factor_smooth = sum(curve_factors_smooth.values()) / len(curve_factors_smooth)
        else:
            args.warmup_clip_factor_smooth = 0.0

    # Lambda functions (not CLI configurable)
    plot_legend_lambda = lambda x: plot.legend_format(args.plot_curve_column, x)
    plot_y_lambda_r = lambda y: 1 - y  # For R metric transformation

    predicter = None
    
    # Fitting phase
    if args.fit:
        print(f"\n=== Fitting for x_column: {args.fit_x_column}, curve_column: {args.fit_curve_column} ===")
        
        # Temporarily update config for fitting to use our converted warmup factor
        original_warmup_factor = config.WARMUP_CLIPPING_FACTOR_FOR_RAW
        config.WARMUP_CLIPPING_FACTOR_FOR_RAW = args.warmup_clip_factor_raw
        
        try:
            predicter = fit.fit_log_errrate_simple(
                df, args.eval, args.fit_curve_column, args.fit_x_column,
                fit_load_path=args.fit_load_path,
                fit_save_path=args.fit_save_path,
                data_source=args.data_source,
                warmup_step=args.warmup_clip_raw,
                ending_step=args.ending_clip_raw
            )
        finally:
            # Restore original config value
            config.WARMUP_CLIPPING_FACTOR_FOR_RAW = original_warmup_factor

    # Plotting phase
    for plot_x_column in args.plot_x_columns:
        print(f"\n=== Plotting for x_column: {plot_x_column} ===")
        
        # Plot k and E0 scatter plots if fitting was done
        if args.fit and args.fit_plot_k_E0 and predicter:
            # Get k(curve_column) and E0(curve_column) arrays for further analysis
            curve_values, k_values = predicter.get_k_array()
            curve_values_E0, E0_values = predicter.get_E0_array()
            
            print(f"\n=== k({args.fit_curve_column}) and E0({args.fit_curve_column}) Arrays for fitted x_column={args.fit_x_column} ===")
            print(f"{args.fit_curve_column}_values:", curve_values)
            print("k_values:", k_values)
            print("E0_values:", E0_values)
            
            # Plot k(curve_column) - Compact version
            fig1, ax1 = plt.subplots(figsize=(6, 4.5), dpi=300)
            ax1 = plot.plot_basic(
                x=curve_values,
                y=np.abs(k_values),  # Use absolute value since k is negative
                use_scatter=True,
                scatter_s=150,
                color='blue',
                ax=ax1
            )
            ax1 = plot.plot_basic_settings(
                ax=ax1,
                x_scale="log",
                x_label=f"Model Size {args.fit_curve_column}",
                y_label=f"k({args.fit_curve_column})",
                title=f"k({args.fit_curve_column}) vs Model Size",
                use_legend=False,
                x_margin=0.1, y_margin=0.1,
                x_tick_spacing=args.x_tick_spacing,
                y_tick_spacing=args.y_tick_spacing,
                x_grid_spacing=args.x_grid_spacing,
                y_grid_spacing=args.y_grid_spacing
            )
            
            
            plt.tight_layout(pad=1.0)
            eval_file_str = config.TEST_EVALS[args.eval]['file_str']
            
            # Ensure output directory exists
            args.output_base_dir.mkdir(parents=True, exist_ok=True)
            
            k_plot_path = args.output_base_dir / f"{args.output_prefix}{eval_file_str}_{args.fit_curve_column}_{args.fit_x_column}_k_scatter.pdf"
            plt.savefig(k_plot_path, bbox_inches='tight', dpi=300)
            print(f"Saved k({args.fit_curve_column}) plot: {k_plot_path}")
            
            # Plot E0(curve_column) - Compact version
            fig2, ax2 = plt.subplots(figsize=(6, 4.5), dpi=300)
            ax2 = plot.plot_basic(
                x=curve_values,
                y=E0_values,
                use_scatter=True,
                scatter_s=150,
                color='red',
                ax=ax2
            )
            ax2 = plot.plot_basic_settings(
                ax=ax2,
                x_scale="log",
                y_scale=None,
                x_label=f"{config.DEFAULT_LABELS[args.fit_curve_column]}",
                y_label=f"E0({args.fit_curve_column})",
                title=f"E({args.fit_curve_column}) vs Model Size",
                use_legend=False,
                x_margin=0.1, y_margin=0.1,
                x_tick_spacing=args.x_tick_spacing,
                y_tick_spacing=args.y_tick_spacing,
                x_grid_spacing=args.x_grid_spacing,
                y_grid_spacing=args.y_grid_spacing
            )
            
            
            plt.tight_layout(pad=1.0)
            
            # Output directory already ensured to exist above
            E0_plot_path = args.output_base_dir / f"{args.output_prefix}{eval_file_str}_{args.fit_curve_column}_{args.fit_x_column}_E0_scatter.pdf"
            plt.savefig(E0_plot_path, bbox_inches='tight', dpi=300)
            print(f"Saved E0({args.fit_curve_column}) plot: {E0_plot_path}")

        # Process each plot metric
        for plot_metric in args.plot_metrics:
            ax = None
            
            # Add fitted prediction curves if fitting was done
            if args.fit and predicter:
                predict_x_column_list = [args.fit_curve_column, args.fit_x_column]
                predict_plot_highlight_curves = args.highlight_curves_predict
                predict_plot_use_scatter = False  # predict_and_plot usually shows lines only
                predict_plot_y_lambda = plot_y_lambda_r if plot_metric == "R" else None

                ax = plot_data.predict_and_plot(
                    df,
                    predicter.predict_errrate_df,
                    predict_x_column_list=predict_x_column_list,
                    metric_column=plot_metric,
                    plot_curve_column=args.plot_curve_column,
                    plot_curve_mask=args.plot_curve_mask,

                    # Highlight configuration
                    plot_highlight_curves=predict_plot_highlight_curves,
                    plot_highlight_line_alpha=args.highlight_line_alpha,
                    plot_highlight_line_width=args.highlight_line_width,

                    # Plotting style
                    plot_x_column=plot_x_column,
                    plot_use_line=args.plot_use_line,
                    plot_use_scatter=predict_plot_use_scatter,
                    plot_x_scale=args.plot_x_scale,
                    plot_y_scale=args.plot_y_scale,
                    plot_y_lambda=predict_plot_y_lambda,
                    
                    # Line and scatter styling
                    line_alpha=args.line_alpha,
                    line_width=args.line_width,
                    scatter_alpha=args.scatter_alpha,
                    scatter_size=args.scatter_size,
                    scatter_marker=args.scatter_marker,
                    
                    # Axis formatting
                    x_tick_format=args.x_tick_format,
                    y_tick_format=args.y_tick_format,
                    
                    # Tick and grid spacing
                    x_tick_spacing=args.x_tick_spacing,
                    y_tick_spacing=args.y_tick_spacing,
                    x_grid_spacing=args.x_grid_spacing,
                    y_grid_spacing=args.y_grid_spacing,
                    
                    warmup_frac_raw=args.warmup_clip_factor_raw,
                    warmup_clip_raw=args.warmup_clip_raw,
                    ending_clip_raw=args.ending_clip_raw,
                )

            # Process the actual data plot
            process_plot_x_label = args.plot_x_label if args.plot_x_label else config.DEFAULT_LABELS[plot_x_column]
            process_plot_y_label = args.plot_y_label if args.plot_y_label else config.DEFAULT_LABELS[plot_metric]
            process_plot_highlight_curves = args.highlight_curves_plot
            
            # Generate plot title
            if args.plot_title:
                # Use fixed title for all plots (highest priority)
                process_plot_title = args.plot_title
            elif args.plot_titles and plot_x_column in args.plot_titles:
                # Use custom title for this x_column
                process_plot_title = args.plot_titles[plot_x_column]
            elif args.plot_title_template:
                # Use template with variable substitution
                template_vars = {
                    'eval': config.TEST_EVALS[args.eval]['plot_str'],
                    'fit_x': args.fit_x_column if args.fit else 'None',
                    'plot_x': plot_x_column,
                    'metric': plot_metric
                }
                process_plot_title = args.plot_title_template.format(**template_vars)
            else:
                # Use default auto-generated title
                if args.fit:
                    process_plot_title = f"{config.TEST_EVALS[args.eval]['plot_str']} (fitted on {args.fit_x_column}, plotted on {plot_x_column})"
                else:
                    process_plot_title = f"{config.TEST_EVALS[args.eval]['plot_str']} (plotted on {plot_x_column})"

            ax = plot_data.process_single_eval(
                df,
                plot_x_column=plot_x_column,
                plot_eval_column=args.eval,
                plot_metric=plot_metric,
                plot_curve_column=args.plot_curve_column,
                plot_curve_mask=args.plot_curve_mask,

                # Title configuration - IMPORTANT: Pass the title here!
                plot_title=process_plot_title,

                # Highlight configuration
                plot_highlight_curves=process_plot_highlight_curves,
                plot_highlight_line_alpha=args.highlight_line_alpha,
                plot_highlight_line_width=args.highlight_line_width,

                # Plotting style
                plot_use_scatter=args.plot_use_scatter,
                plot_use_legend=args.plot_use_legend,
                plot_legend_loc=args.plot_legend_loc,
                plot_legend_bbox_to_anchor=args.plot_legend_bbox_to_anchor,
                plot_legend_lambda=plot_legend_lambda,

                # Line and scatter styling
                line_alpha=args.line_alpha,
                line_width=args.line_width,
                scatter_alpha=args.scatter_alpha,
                scatter_size=args.scatter_size,
                scatter_marker=args.scatter_marker,
                
                # Delta configuration
                delta_base_step=args.delta_base_step,

                # Smoothing configuration
                add_smooth=args.add_smooth,
                add_std=args.add_std,
                smooth_monotonic=args.smooth_monotonic,
                smooth_increasing=args.smooth_increasing,
                smooth_strict=args.smooth_strict,

                # Advanced parameters
                warmup_frac_raw=args.warmup_clip_factor_raw,
                warmup_frac_smooth=args.warmup_clip_factor_smooth,
                warmup_clip_raw=args.warmup_clip_raw,
                warmup_clip_smooth=args.warmup_clip_smooth,
                ending_clip_raw=args.ending_clip_raw,
                ending_clip_smooth=args.ending_clip_smooth,
                s_factor=args.s_factor,
                k_spline=args.k_spline,
                rolling_window=args.rolling_window,
                min_se=args.min_se,
                x_inv_weight_power=args.x_inv_weight_power,
                ax=ax,
            )
            
            # Apply plot_basic_settings after process_single_eval (now includes save logic)
            plot.plot_basic_settings(
                ax=ax,
                x_scale=args.plot_x_scale,
                y_scale=args.plot_y_scale,
                x_label=process_plot_x_label,
                y_label=process_plot_y_label,
                title=process_plot_title,
                use_legend=args.plot_use_legend,
                legend_loc=args.plot_legend_loc,
                legend_bbox_to_anchor=args.plot_legend_bbox_to_anchor,
                x_tick_format=args.x_tick_format,
                y_tick_format=args.y_tick_format,
                x_tick_spacing=args.x_tick_spacing,
                y_tick_spacing=args.y_tick_spacing,
                x_grid_spacing=args.x_grid_spacing,
                y_grid_spacing=args.y_grid_spacing,
                # Custom tick positioning
                x_tick_subs=args.x_tick_subs,
                y_tick_subs=args.y_tick_subs,
                x_tick_subs_log=args.x_tick_subs_log,
                y_tick_subs_log=args.y_tick_subs_log,
                # Save configuration - moved from process_single_eval
                save_to_dir=args.output_base_dir,
                save_to_filename_prefix=args.output_prefix,
                plot_eval_column=args.eval,
                plot_curve_column=args.plot_curve_column,
                plot_x_column=plot_x_column,
                plot_metric=plot_metric,
                plot_curve_mask=args.plot_curve_mask,
            )


def load_config_file(config_file_path: str):
    """Load configuration from JSON file"""
    try:
        with open(config_file_path, 'r') as f:
            config_data = json.load(f)
        
        if 'runs' not in config_data:
            raise ValueError("Config file must contain 'runs' array")
        
        return config_data['runs']
    except Exception as e:
        raise ValueError(f"Error loading config file {config_file_path}: {e}")


def main():
    parser = create_argument_parser()
    args = parser.parse_args()

    # Handle config file batch processing
    if args.config_file:
        print(f"Loading configuration from: {args.config_file}")
        runs = load_config_file(args.config_file)
        
        for i, run_config in enumerate(runs):
            print(f"\n{'='*60}")
            print(f"Processing run {i+1}/{len(runs)}")
            print(f"{'='*60}")
            
            # Convert config dict to args object
            # Parse empty args to get all default values from argument parser
            run_args = parser.parse_args([])
            
            # Override with config values
            valid_args = vars(run_args).keys()  # Get all valid argument names
            invalid_args = []
            
            # Special keys that are allowed but not used as arguments
            ignored_keys = {'comment'}
            
            for key, value in run_config.items():
                if key in valid_args:
                    setattr(run_args, key, value)
                elif key not in ignored_keys:
                    invalid_args.append(key)
            
            # Warn about invalid arguments
            if invalid_args:
                print(f"Warning: Invalid parameters in config file: {', '.join(invalid_args)}")
                print("These parameters will be ignored.")
                
                # Provide suggestions for common mistakes
                suggestions = {
                    'use_legend': 'plot_use_legend',
                    'use_scatter': 'plot_use_scatter', 
                    'use_line': 'plot_use_line',
                    'x_scale': 'plot_x_scale',
                    'y_scale': 'plot_y_scale',
                    'legend_loc': 'plot_legend_loc',
                }
                
                for invalid_arg in invalid_args:
                    if invalid_arg in suggestions:
                        print(f"  Did you mean '{suggestions[invalid_arg]}' instead of '{invalid_arg}'?")
            
            
            # Apply defaults and process
            run_args = process_parsed_args(run_args)
            validate_args(run_args)
            run_scaling_analysis(run_args)
        
        print(f"\nCompleted {len(runs)} runs from config file.")
        return

    # Single run processing
    # Process arguments first
    args = process_parsed_args(args)
    
    # Check required arguments when not using config file
    required_args = ['data_source', 'plot_curve_column', 'plot_x_columns', 'eval', 'plot_metrics']
    missing_args = []
    for arg in required_args:
        if not getattr(args, arg, None):
            missing_args.append(arg.replace('_', '-'))
    
    if missing_args:
        parser.error(f"the following arguments are required when not using --config-file: {', '.join(['--' + arg for arg in missing_args])}")
    
    validate_args(args)
    run_scaling_analysis(args)


if __name__ == "__main__":
    main()
