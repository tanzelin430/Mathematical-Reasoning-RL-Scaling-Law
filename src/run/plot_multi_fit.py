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

from src.common import data_proc
from src.common import fit_models_simple
from src.common import plot_data
from src.common import config
from src.fit import fit
from src.common import plot
import matplotlib.pyplot as plt
import numpy as np
from src.fit.models import get_model_class, list_available_models
            

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


def parse_curve_mask(value: Union[str, List[str]]) -> List[Union[int, float]]:
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


def parse_highlight_curves(value: Union[str, List[str]]) -> List[Union[int, float]]:
    """Parse highlight curves supporting scientific notation"""
    if not value:
        return []
    return parse_curve_mask(value)


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
        
        # Validate fit-model is provided when --fit is enabled
        if not args.fit_model:
            raise ValueError("--fit-model is required when --fit is enabled. "
                           f"Available models: {', '.join(list_available_models())}")


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
  %(prog)s --data-source exp2-instruct --plot-curve Tau -x E --eval holdout_score --metric ErrRate \
           --fit --fit-model InvExp --fit-curve Tau --fit-x E --fit-metric ErrRate

  # Multiple metrics and x columns with highlighting
  %(prog)s --data-source exp2-base --plot-curve N -x C E --eval holdout_score --metric ErrRate R \
           --plot-curve-mask 1e9 3e9 7e9 --highlight-curves-predict 1e9 --highlight-curves-plot 3e9

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
                       action='append', nargs='+',
                       help='X columns for plotting (space- or comma-separated; flag can be repeated)')
    
    parser.add_argument('--eval',
                       choices=list(config.TEST_EVALS.keys()),
                       help='Evaluation metric to analyze')
    
    parser.add_argument('--metric', dest='plot_metrics',
                       action='append', nargs='+',
                       help='Metrics to plot (space- or comma-separated; flag can be repeated)')

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
                          action='append', nargs='+',
                          help='Metrics to fit (space- or comma-separated; default: same as --metric)')
    fit_group.add_argument('--fit-load', dest='fit_load_path',
                          help='Path to load pre-fitted model')
    fit_group.add_argument('--fit-save', dest='fit_save_path',
                          help='Path to save fitted model')
    fit_group.add_argument('--fit-plot-params', dest='fit_plot_params',
                          action='store_true', default=False,
                          help='Plot parameters scatter plots (default: False)')
    fit_group.add_argument('--fit-model', dest='fit_model',
                          choices=list_available_models(),
                          help='Model to use for fitting (required when --fit is enabled)')

    # Plot configuration
    plot_group = parser.add_argument_group('plot configuration')
    plot_group.add_argument('--plot-curve-mask', dest='plot_curve_mask', nargs='+',
                           help='Curves to include in plots (space- or comma-separated, supports scientific notation)')
    plot_group.add_argument('--highlight-curves-predict', dest='highlight_curves_predict', nargs='+',
                           help='Curves to highlight in predict_and_plot (space- or comma-separated, supports scientific notation). If not set, no highlighting in predict phase.')
    plot_group.add_argument('--highlight-curves-plot', dest='highlight_curves_plot', nargs='+',
                           help='Curves to highlight in process_single_eval (space- or comma-separated, supports scientific notation). If not set, no highlighting in plot phase.')
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
    advanced_group.add_argument('--warmup-clip', dest='warmup_clip',
                               type=int, default=None,
                               help='Warmup clipping: number of steps to remove from the beginning for raw data (0 means no clipping)')
    advanced_group.add_argument('--warmup-clip-smooth', dest='warmup_clip_smooth',
                               type=int, default=None,
                               help='Warmup clipping: number of steps to remove from the beginning for smooth data (0 means no clipping)')
    advanced_group.add_argument('--ending-clip', dest='ending_clip',
                               type=int, default=None,
                               help='Ending clipping: number of steps to remove from the end for raw data (0 means no clipping)')
    advanced_group.add_argument('--ending-clip-smooth', dest='ending_clip_smooth',
                               type=int, default=None,
                               help='Ending clipping: number of steps to remove from the end for smooth data (0 means no clipping)')
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
    # Normalize list-like arguments (support nargs and comma-separated tokens)
    if args.plot_x_columns:
        args.plot_x_columns = normalize_list_arg(args.plot_x_columns)
    
    if args.plot_metrics:
        args.plot_metrics = normalize_list_arg(args.plot_metrics)

    if args.fit_metrics:
        args.fit_metrics = normalize_list_arg(args.fit_metrics)

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
        args.plot_curve_mask = parse_curve_mask(args.plot_curve_mask)
    else:
        args.plot_curve_mask = None  # Default

    # Parse highlight curve arguments
    if args.highlight_curves_predict:
        args.highlight_curves_predict = parse_highlight_curves(args.highlight_curves_predict)
    else:
        args.highlight_curves_predict = None  # Default
    
    if args.highlight_curves_plot:
        args.highlight_curves_plot = parse_highlight_curves(args.highlight_curves_plot)
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
        args.output_prefix = f"fit_{args.data_source}_"

    return args



def plot_params(args, _curve_values, params_dict):
    _curve_label = config.DEFAULT_LABELS[args.fit_curve_column]
    for _param_key, _param_arr in params_dict.items():
        if len(_param_arr) != len(_curve_values):
            print(f"Skipping: {_param_key} has {len(_param_arr)} values, but {_curve_values} has {len(_curve_values)} values")
            continue
        print(f"{_param_key} values: {_param_arr}")
    
        # Plot param(curve_column) - Compact version
        fig1, ax1 = plt.subplots(figsize=(6, 4.5), dpi=300)
        ax1 = plot.plot_basic(
            x=_curve_values,
            y=_param_arr,
            use_scatter=True,
            scatter_s=150,
            color=config.COLOR_MAPPING[_param_key],
            ax=ax1
        )
        ax1 = plot.plot_basic_settings(
            ax=ax1,
            x_scale="log",
            x_label=f"{_curve_label}",
            y_label=f"{_param_key}({args.fit_curve_column})",
            title=f"{_param_key}({args.fit_curve_column}) vs {_curve_label}",
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
        
        param_plot_path = args.output_base_dir / f"{args.output_prefix}{eval_file_str}_{args.fit_curve_column}_{args.fit_x_column}_{_param_key}_scatter.pdf"
        plt.savefig(param_plot_path, bbox_inches='tight', dpi=300)
        print(f"Saved {_param_key}({args.fit_curve_column}) plot: {param_plot_path}")

def run_scaling_analysis(args):
    """Run the scaling law analysis with given arguments"""
    print(f"Loading data for data_source: {args.data_source}")
    df = data_proc.load_and_preprocess(config.CSV_MAP[args.data_source])
    
    # 移除step=0的数据（因为E=0会导致log10(E)=-inf）
    df = df[df['step'] > 0].reset_index(drop=True)
    
    group_columns = [args.plot_curve_column] + ['step']
    df = data_proc.merge_duplicate_steps(df, group_columns=group_columns, mode='mean')

    # Lambda functions (not CLI configurable)
    plot_legend_lambda = lambda x: plot.legend_format(args.plot_curve_column, x)
    plot_y_lambda_r = lambda y: 1 - y  # For R metric transformation

    fitter = None
    
    # Fitting phase
    if args.fit:
        print(f"\n=== Fitting for x_column: {args.fit_x_column}, curve_column: {args.fit_curve_column} ===")
        print(f"Using model: {args.fit_model}")
        
        # Get the fitter class dynamically from the model name
        FitterClass = get_model_class(args.fit_model)
        
        # [None, lambda x: np.log10(x)]
        fitter = fit.fit_on(
            FitterClass,
            df, eval_name = args.eval, x_column_list=[args.fit_curve_column, args.fit_x_column],
            fit_load_path=args.fit_load_path, fit_save_path=args.fit_save_path,
            warmup_clip=args.warmup_clip if args.warmup_clip is not None else 0,
            ending_clip=args.ending_clip if args.ending_clip is not None else 0
        )

        # Plot params (e.g. k, E0) scatter plots if fitting was done
        if args.fit and args.fit_plot_params and fitter:
            # Get k(curve_column) and E0(curve_column) arrays for further analysis
            # curve_values, k_values = predicter.get_k_array()
            # curve_values_E0, E0_values = predicter.get_E0_array()

            _curve_values, params_dict = fitter.get_params_array()
            print(f"\n=== params({args.fit_curve_column}) Arrays for fitted x_column={args.fit_x_column} ===")
            print(f"{args.fit_curve_column}_values:", _curve_values)
            plot_params(args, _curve_values, params_dict)

    # Plotting phase
    for plot_x_column in args.plot_x_columns:
        print(f"\n=== Plotting for x_column: {plot_x_column} ===")
        
        # Process each plot metric
        for plot_metric in args.plot_metrics:
            ax = None
            
            # Add fitted prediction curves if fitting was done
            if args.fit and fitter:
                predict_x_column_list = [args.fit_curve_column, args.fit_x_column]
                predict_plot_highlight_curves = args.highlight_curves_predict
                predict_plot_use_scatter = False  # predict_and_plot usually shows lines only
                predict_plot_y_lambda = plot_y_lambda_r if plot_metric == "R" else None
                
                ax = plot_data.predict_and_plot(
                    df,
                    fitter,
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
                    
                    warmup_clip=args.warmup_clip,
                    ending_clip=args.ending_clip,
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
                warmup_clip=args.warmup_clip,
                warmup_clip_smooth=args.warmup_clip_smooth,
                ending_clip=args.ending_clip,
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
