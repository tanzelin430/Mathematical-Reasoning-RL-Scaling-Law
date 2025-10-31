#!/usr/bin/env python3
"""
Scaling Law Pipeline - Multi-Eval Analysis (CLI Version)
Processes multiple test evals from experiment data and generates scaling law plots for each eval
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Union
from src.common import config
from src.fit.models import list_available_models
from src.run.plot_fit_params import PLOT_PARAMS_SCHEMA


# Mapping from attribute name to CLI flag name
ARG_NAME_MAP = {
    'data_sources': '--data-sources',
    'eval': '--eval',
    'plot_curve': '--plot-curve',
    'plot_x_columns': '--plot-x',
    'plot_metrics': '--plot-metric',
    'fit_x': '--fit-x',
    'fit_curve': '--fit-curve',
    'fit_metric': '--fit-metric',
}

# Required arguments for each mode
REQUIRED_ARGS_BY_MODE = {
    'plot': ['data_sources', 'eval', 'plot_curve', 'plot_x_columns', 'plot_metrics'],
    'plot_fit': ['data_sources', 'fit_x', 'fit_curve', 'fit_metric'],
    'fit': ['data_sources', 'fit_x', 'fit_curve', 'fit_metric', 'eval'],
}


def parse_list_argument(value: str) -> List[str]:
    """Parse comma-separated string into list"""
    if not value:
        return []
    return [item.strip() for item in value.split(',') if item.strip()]


def normalize_list_arg(value: Union[str, List[Union[str, List[str]]], None]) -> List[str]:
    """Normalize CLI list-like arguments into a flat list of strings.

    Supports:
    - Single comma-separated string
    - List of strings (space-separated via nargs='+')
    - List of lists of strings (when flag used multiple times with nargs='+')
    """
    if value is None:
        return []
    tokens: List[str] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, list):
                for sub in item:
                    tokens.extend(parse_list_argument(sub) if isinstance(sub, str) else [str(sub)])
            else:
                tokens.extend(parse_list_argument(item) if isinstance(item, str) else [str(item)])
    elif isinstance(value, str):
        tokens = parse_list_argument(value)
    else:
        tokens = [str(value)]
    return tokens


def parse_curves(value: Union[str, List[str]]) -> List[Union[int, float]]:
    """Parse curve mask supporting scientific notation like 7e9, 1.5e9.

    Accepts comma-separated string or a list of tokens.
    """
    values: List[Union[int, float]] = []
    tokens = normalize_list_arg(value)
    for item in tokens:
        try:
            if 'e' in item.lower():
                values.append(float(item))
            else:
                values.append(int(item))
        except ValueError:
            raise ValueError(f"Invalid curve mask value: {item}")
    return values

def validate_args(args):
    """Validate command line arguments
    
    When --fit-load is used, most arguments are loaded from context and don't need validation.
    """
    # Check for deprecated argument usage and provide friendly error messages
    deprecated_args = []
    if hasattr(args, '_deprecated_curve') and args._deprecated_curve:
        deprecated_args.append("--curve")
    if hasattr(args, '_deprecated_x') and args._deprecated_x:
        deprecated_args.append("-x")
    if hasattr(args, '_deprecated_metric') and args._deprecated_metric:
        deprecated_args.append("--metric")
    
    if deprecated_args:
        error_msg = f"Error: The following arguments are deprecated: {', '.join(deprecated_args)}\n\n"
        error_msg += "Please use the explicit versions instead:\n"
        error_msg += "  --curve      → Use --plot-curve or --fit-curve\n"
        error_msg += "  -x           → Use --plot-x or --fit-x\n"
        error_msg += "  --metric     → Use --plot-metric or --fit-metric\n"
        raise ValueError(error_msg)
    
    # Validate data_sources (only if provided)
    if args.data_sources:
        for data_source in args.data_sources:
            if data_source not in config.CSV_MAP:
                raise ValueError(f"Invalid data-source: {data_source}. "
                                f"Must be one of: {list(config.CSV_MAP.keys())}")
    
    # Validate eval (only if provided)
    if args.eval and args.eval not in config.TEST_EVALS:
        raise ValueError(f"Invalid eval: {args.eval}. "
                        f"Must be one of: {list(config.TEST_EVALS.keys())}")
    
    # Validate columns (only if provided)
    valid_columns = list(config.DEFAULT_LABELS.keys())
    
    # Validate plot-curve
    if args.plot_curve and args.plot_curve not in valid_columns:
        raise ValueError(f"Invalid --plot-curve: {args.plot_curve}. "
                        f"Must be one of: {valid_columns}, or add to config.DEFAULT_LABELS")
    
    # Validate fit-curve
    if args.fit_curve and args.fit_curve not in valid_columns:
        raise ValueError(f"Invalid --fit-curve: {args.fit_curve}. "
                        f"Must be one of: {valid_columns}, or add to config.DEFAULT_LABELS")
    
    # Validate plot-x columns
    if args.plot_x_columns:
        for x_col in args.plot_x_columns:
            if x_col not in valid_columns:
                raise ValueError(f"Invalid --plot-x column: {x_col}. "
                                f"Must be one of: {valid_columns}, or add to config.DEFAULT_LABELS")
    
    # Validate fit-x
    if args.fit_x and args.fit_x not in valid_columns:
        raise ValueError(f"Invalid --fit-x column: {args.fit_x}. "
                        f"Must be one of: {valid_columns}, or add to config.DEFAULT_LABELS")
    
    # Validate plot-metric
    if args.plot_metrics:
        for metric in args.plot_metrics:
            if metric not in valid_columns:
                raise ValueError(f"Invalid --plot-metric: {metric}. "
                                f"Must be one of: {valid_columns}, or add to config.DEFAULT_LABELS")
    
    # Validate fit-metric
    if args.fit_metric and args.fit_metric not in valid_columns:
        raise ValueError(f"Invalid --fit-metric: {args.fit_metric}. "
                        f"Must be one of: {valid_columns}, or add to config.DEFAULT_LABELS")
    
    # Validate fit-model is provided when --fit is enabled
    if args.fit and not args.fit_model:
        raise ValueError("--fit-model is required when --fit is enabled. "
                       f"Available models: {', '.join(list_available_models())}")


def create_argument_parser():
    """Create and configure argument parser"""
    parser = argparse.ArgumentParser(
        description="Scaling Law Pipeline - Multi-Eval Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with plotting
  %(prog)s --data-sources exp2-instruct --plot-curve Tau --plot-x E --eval holdout_score --plot-metric ErrRate --plot

  # Multiple data sources
  %(prog)s --data-sources exp2-base exp2-instruct --plot-curve Tau --plot-x E --eval holdout_score --plot-metric ErrRate --plot

  # With fitting
  %(prog)s --data-sources exp2-instruct --fit-curve Tau --fit-x E --eval holdout_score --fit-metric ErrRate \
           --fit --fit-model InvExp

  # Multiple data sources, metrics and x columns with highlighting
  %(prog)s --data-sources exp2-base exp2-instruct --plot-curve N --plot-x C E --eval holdout_score --plot-metric ErrRate R \
           --plot-curve-mask 1e9 3e9 7e9 --highlight-curves-predict 1e9 --highlight-curves-plot 3e9 --plot

  # Load configuration from file
  %(prog)s --config-file config.json
        """
    )

    # Basic plot arguments
    parser.add_argument('--data-sources', dest='data_sources',
                       action='append', nargs='+',
                       help='Data source(s) to use (space- or comma-separated; flag can be repeated). Valid choices: ' + ', '.join(config.CSV_MAP.keys()))
    
    parser.add_argument('--plot-curve', dest='plot_curve',
                       help='Curve column for plotting, e.g.\'N\', \'Tau\', \'rollout_n\'')
    
    parser.add_argument('--plot-x', dest='plot_x_columns',
                       action='append', nargs='+',
                       help='X columns for plotting (space- or comma-separated; flag can be repeated)')
    
    parser.add_argument('--eval',
                       choices=list(config.TEST_EVALS.keys()),
                       help='Evaluation metric to analyze')
    
    parser.add_argument('--plot-metric', dest='plot_metrics',
                       action='append', nargs='+',
                       help='Metrics to plot (space- or comma-separated; flag can be repeated)')
    
    parser.add_argument('--plot', action='store_true',
                       help='Enable model plotting')
    parser.add_argument('--plot-fit', action='store_true',
                       help='Enable model plotting with fitted curves')
    
    # Deprecated argument aliases with friendly error messages
    deprecated_group = parser.add_argument_group('deprecated options (will show error)')
    deprecated_group.add_argument('--curve', dest='_deprecated_curve',
                       help='DEPRECATED: Use --plot-curve or --fit-curve instead')
    deprecated_group.add_argument('-x', dest='_deprecated_x',
                       help='DEPRECATED: Use --plot-x or --fit-x instead')
    deprecated_group.add_argument('--metric', dest='_deprecated_metric',
                       help='DEPRECATED: Use --plot-metric or --fit-metric instead')

    # Fitting arguments
    fit_group = parser.add_argument_group('fitting options')
    fit_group.add_argument('--fit', action='store_true',
                          help='Enable model fitting')
    fit_group.add_argument('--fit-curve',
                          help='Curve column for fitting, e.g.\'N\', \'Tau\', \'rollout_n\'')
    fit_group.add_argument('--fit-curve-mask', dest='fit_curve_mask', nargs='+',
                           help='Curves to include in fitting (space- or comma-separated, supports scientific notation)')
    fit_group.add_argument('--fit-x', dest='fit_x',
                          choices=list(config.DEFAULT_LABELS.keys()),
                          help='X column for fitting (default: first --plot-x column)')
    fit_group.add_argument('--fit-metric', dest='fit_metric',
                          help='Metric to fit (default: same as first --metric)')
    fit_group.add_argument('--fit-load', dest='fit_load',
                          help='Path to load batch fitters JSON (mutually exclusive with --fit)')
    fit_group.add_argument('--fit-save', dest='fit_save',
                          help='Path to save batch fitters JSON')
    fit_group.add_argument('--fit-save-append', dest='fit_save_append',
                          help='Path to append batch fitters JSON (loads existing file and appends new fits)')
    fit_group.add_argument('--fit-param-plot-schema', dest='fit_param_plot_schema',
                          type=str, default=None, 
                          choices=PLOT_PARAMS_SCHEMA.keys(),
                          help='Plot parameters scatter plots. Options: ' + ', '.join(PLOT_PARAMS_SCHEMA.keys()))
    fit_group.add_argument('--fit-model', dest='fit_model',
                          choices=list_available_models(),
                          help='Model to use for fitting (required when --fit is enabled)')
    fit_group.add_argument('--fit-cma-verbose-interval', dest='fit_cma_verbose_interval',
                          type=int, default=0,
                          help='CMA-ES verbose interval, 0 to disable verbose output (default: 0)')
    # Plot configuration
    plot_group = parser.add_argument_group('plot configuration')
    plot_group.add_argument('--plot-curve-mask', dest='plot_curve_mask', nargs='+',
                           help='Curves to include in plots (space- or comma-separated, supports scientific notation)')
    plot_group.add_argument('--highlight-curves-predict', dest='highlight_curves_predict', nargs='+',
                           help='Curves to highlight in predict_and_plot (space- or comma-separated, supports scientific notation). If not set, no highlighting in predict phase.')
    plot_group.add_argument('--highlight-curves-plot', dest='highlight_curves_plot', nargs='+',
                           help='Curves to highlight in plot phase (space- or comma-separated, supports scientific notation). If not set, no highlighting in plot phase.')
    plot_group.add_argument('--highlight-alpha', dest='highlight_alpha',
                           type=float, default=1.0,
                           help='Opacity for highlighted curves (default: 1.0)')
    plot_group.add_argument('--highlight-width', dest='highlight_width',
                           type=float, default=2.0,
                           help='Width for highlighted curves (applies to line width and scatter size, default: 2.0)')
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
                            action='store_true', default=True,
                            help='Use scatter plots (default: True)')
    style_group.add_argument('--plot-use-line', dest='plot_use_line',
                            action='store_true', default=False,
                            help='Use line plots (default: False)')
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
                            choices=['best', 'upper right', 'upper left', 'lower left', 'lower right', 
                                     'right', 'center left', 'center right', 'lower center', 'upper center', 'center'],
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
                            nargs='+', type=str, default=None,
                            help='Custom x-axis tick positions (space- or comma-separated list, e.g., "1000 5000 10000")')
    style_group.add_argument('--y-tick-subs', dest='y_tick_subs',
                            nargs='+', type=str, default=None,
                            help='Custom y-axis tick positions (space- or comma-separated list)')
    style_group.add_argument('--x-tick-subs-log', dest='x_tick_subs_log',
                            nargs='+', type=str, default=None,
                            help='Custom x-axis log tick multipliers (space- or comma-separated list, e.g., "1 2 5")')
    style_group.add_argument('--y-tick-subs-log', dest='y_tick_subs_log',
                            nargs='+', type=str, default=None,
                            help='Custom y-axis log tick multipliers (space- or comma-separated list)')

    # Smoothing configuration
    smooth_group = parser.add_argument_group('smoothing options')
    smooth_group.add_argument('--add-smooth', dest='add_smooth',
                             action='store_true', default=False,
                             help='Enable smoothing (default: False)')
    smooth_group.add_argument('--add-std', dest='add_std',
                             action='store_true', default=False,
                             help='Add standard deviation bands (default: False)')
    smooth_group.add_argument('--smooth-monotonic', dest='smooth_monotonic',
                             action='store_true', default=False,
                             help='Force monotonic smoothing (default: True)')
    smooth_group.add_argument('--smooth-increasing', dest='smooth_increasing',
                             type=str, default='None',
                             help='Force increasing trend (None/True/False, default: None)')
    smooth_group.add_argument('--smooth-strict', dest='smooth_strict',
                             action='store_true', default=False,
                             help='Strict monotonic constraints (default: False)')

    # Advanced parameters
    advanced_group = parser.add_argument_group('advanced parameters')
    advanced_group.add_argument('--warmup-clip', dest='warmup_clip',
                               type=int, default=None,
                               help='Warmup clipping: number of steps to remove from the beginning (0 means no clipping, applied once to all data)')
    advanced_group.add_argument('--warmup-clip-to', dest='warmup_clip_to',
                               type=int, default=None,
                               help='Warmup clipping: remove step < warmup_clip_to (keep step >= warmup_clip_to)')
    advanced_group.add_argument('--ending-clip', dest='ending_clip',
                               type=int, default=None,
                               help='Ending clipping: number of steps to remove from the end (0 means no clipping, applied once to all data)')
    advanced_group.add_argument('--ending-clip-to', dest='ending_clip_to',
                               type=int, default=None,
                               help='Ending clipping: remove step > ending_clip_to (keep step <= ending_clip_to)')
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
                               type=float, default=0,
                               help='X inverse weight power (default: 0), for both fit and smooth')

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
    # Normalize list-like arguments (support nargs and comma-separated tokens)
    if args.data_sources:
        args.data_sources = normalize_list_arg(args.data_sources)
    
    if args.plot_x_columns:
        args.plot_x_columns = normalize_list_arg(args.plot_x_columns)
    
    if args.plot_metrics:
        args.plot_metrics = normalize_list_arg(args.plot_metrics)

    # Set defaults for fit parameters
    if args.fit:
        if not args.fit_x:
            args.fit_x = args.plot_x_columns[0] if args.plot_x_columns else None
        if not args.fit_metric:
            args.fit_metric = args.plot_metrics[0] if args.plot_metrics else None

    # Parse curve masks and highlight curves
    args.fit_curve_mask = parse_curves(args.fit_curve_mask) if args.fit_curve_mask else None
    args.plot_curve_mask = parse_curves(args.plot_curve_mask) if args.plot_curve_mask else None
    args.highlight_curves_predict = parse_curves(args.highlight_curves_predict) if args.highlight_curves_predict else None
    args.highlight_curves_plot = parse_curves(args.highlight_curves_plot) if args.highlight_curves_plot else None

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
    def parse_tick_subs(value_any):
        """Parse tick positions supporting both space- and comma-separated inputs"""
        tokens = normalize_list_arg(value_any)
        if not tokens:
            return None
        values = []
        for item in tokens:
            try:
                values.append(float(item))
            except ValueError:
                raise ValueError(f"Invalid tick position value: {item}")
        return values

    # Parse x_tick_subs
    if args.x_tick_subs:
        args.x_tick_subs = parse_tick_subs(args.x_tick_subs)
    else:
        args.x_tick_subs = None

    # Parse y_tick_subs
    if args.y_tick_subs:
        args.y_tick_subs = parse_tick_subs(args.y_tick_subs)
    else:
        args.y_tick_subs = None

    # Parse x_tick_subs_log
    if args.x_tick_subs_log:
        args.x_tick_subs_log = parse_tick_subs(args.x_tick_subs_log)
    else:
        args.x_tick_subs_log = None

    # Parse y_tick_subs_log
    if args.y_tick_subs_log:
        args.y_tick_subs_log = parse_tick_subs(args.y_tick_subs_log)
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
        args.output_prefix = ""
        # if args.data_sources and len(args.data_sources) == 1:
        #     args.output_prefix = f"fit_{args.data_sources[0]}_"
        # else:
        #     args.output_prefix = "fit_"

    return args

def load_config_file(config_file_path: str):
    """Load configuration from JSON file"""
    try:
        # Convert relative path to absolute path based on current working directory
        if not os.path.isabs(config_file_path):
            config_file_path = os.path.join(os.getcwd(), config_file_path)
        
        with open(config_file_path, 'r') as f:
            config_data = json.load(f)
        
        if 'runs' not in config_data:
            raise ValueError("Config file must contain 'runs' array")
        
        return config_data['runs']
    except Exception as e:
        raise ValueError(f"Error loading config file {config_file_path}: {e}")

def validate_required_args(args):
    """Validate required arguments based on context.
    
    - When --fit-load is provided: no arguments are required (all loaded from file)
    - When --fit-load is NOT provided: specific arguments are required based on mode
    """
    # Validate: --fit and --fit-load are mutually exclusive
    if args.fit and args.fit_load:
        sys.exit("Error: --fit and --fit-load cannot be used together")
    
    # Validate: --fit-save and --fit-save-append are mutually exclusive
    if args.fit_save and args.fit_save_append:
        sys.exit("Error: --fit-save and --fit-save-append cannot be used together")
    
    missing_args = []
    
    # Check required arguments based on each active mode
    for mode in ['plot', 'plot_fit', 'fit']:
        if not getattr(args, mode, False):
            continue
            
        required = REQUIRED_ARGS_BY_MODE[mode]
        missing_args.extend([
            ARG_NAME_MAP[arg] for arg in required 
            if not getattr(args, arg, None)
        ])
    
    if missing_args:
        # Remove duplicates while preserving order
        missing_args = list(dict.fromkeys(missing_args))
        missing_str = ', '.join(missing_args)
        sys.exit(f"Error: the following arguments are required: {missing_str}")