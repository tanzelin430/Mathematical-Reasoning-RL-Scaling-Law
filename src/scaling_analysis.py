#!/usr/bin/env python3
"""
Scaling Law Pipeline - Multi-Eval Analysis (CLI Version)
Processes multiple test evals from experiment data and generates scaling law plots for each eval
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from src.common import data_proc
from src.common import config
from src.common import source_curve
from src.fit import fit
from src.fit.fit import save_batch_fitters, load_batch_fitters
from src.common import plot
from src.common.cli_args import create_argument_parser, process_parsed_args, validate_args, validate_required_args
from src.fit.models import get_model_class
from src.run.plot_fit_params import plot_fit_params
# Lambda functions
R_TO = {
    'R': lambda r: r,
    'ErrRate': lambda r: 1 - r,
    'LogErrRate': lambda r: np.log10(np.clip(1 - r, 1e-12, None)),
}

R_FROM = {
    'R': lambda r: r,
    'ErrRate': lambda errrate: 1 - errrate,
    'LogErrRate': lambda logerrrate: 1 - 10**logerrrate,
}

def run_scaling_analysis(args):
    """Run the scaling law analysis with given arguments"""

    df_map = _data_prepare(args) if args.plot or args.fit else None

    fitters = []
    fitter_map = {}
    
    # Fitting Load
    if args.fit_load:
        print(f"\n=== Loading fitters from {args.fit_load} ===")
        fitters = load_batch_fitters(args.fit_load)
        
        # Build fitter_map and print info
        # Note: fitter_map maps data_source to the LAST fitter with that data_source
        # This is used for prediction plotting. Multiple fitters per data_source are preserved in fitters list.
        for fitter in fitters:
            context = fitter.get_context()
            data_source, fit_x = context["data_source"], context["fit_x"]
            fitter_map[(data_source, fit_x)] = fitter  # Last one wins for prediction
    
    # Fitting phase
    elif args.fit:
        fitters = _fit_multiple(args, df_map)
        
        # Build fitter_map
        # Note: fitter_map maps data_source to the LAST fitter with that data_source
        # This is used for prediction plotting. Multiple fitters per data_source are preserved in fitters list.
        for fitter in fitters:
            context = fitter.get_context()
            data_source, fit_x = context["data_source"], context["fit_x"]
            fitter_map[(data_source, fit_x)] = fitter  # Last one wins for prediction
        
        # Save or append if requested
        if args.fit_save:
            save_batch_fitters(args.fit_save, fitters)
        elif args.fit_save_append:
            save_batch_fitters(args.fit_save_append, fitters, append=True)
 
    # Plot params (e.g. k, E0) scatter plots if fitting was done
    if (args.fit or args.fit_load) and args.fit_param_plot_schema and fitters:
        plot_fit_params(args, fitters)

    # Plotting phase
    if args.plot:
        for plot_x_column in args.plot_x_columns:
            print(f"\n=== Plotting for x_column: {plot_x_column} ===")
            
            # Process each plot metric
            for plot_metric in args.plot_metrics:
                # Initialize shared ax
                ax = None
                
                # Unified loop: process each data source
                for data_source, df in df_map.items():
                    # Apply source-curve mask (works for both modes)
                    df = source_curve.apply_source_curve_mask(
                        df, args.plot_curve, data_source,
                        args.plot_source_curve_mask, args.plot_curve_mask
                    )
                    
                    if len(df) == 0:
                        continue
                    
                    # Update df_map with filtered data (needed for legend generation in merge mode)
                    df_map[data_source] = df
                    
                    # For separate mode: reset ax for each source
                    if not args.plot_merge_sources:
                        ax = None
                    
                    # Build custom color mapping for this data source
                    custom_color_mapping = _build_source_curve_color(args, df, data_source)
                    
                    # Add fitted prediction if available
                    if args.plot_fit:
                        fitter_key = (data_source, args.fit_x)
                        if fitter_key in fitter_map:
                            ax = _plot_fit_prediction(ax, args, df, fitter_map[fitter_key], 
                                                     plot_x_column, plot_metric, custom_color_mapping)
                        else:
                            # User requested --plot-fit but no fitter found
                            available = [f"({ds}, {fx})" for (ds, fx) in fitter_map.keys()] if fitter_map else ["none"]
                            raise ValueError(
                                f"No fitter found for data_source='{data_source}', fit_x='{args.fit_x}'. "
                                f"Available: {', '.join(available)}"
                            )
                    
                    print(f"----------- Unique values: {df[plot_x_column]}")
                    # Plot raw data
                    ax = _plot_raw_data(ax, args, df, plot_x_column, plot_metric, custom_color_mapping)
                    
                    # Add smooth curves if requested
                    if args.add_smooth:
                        ax = _plot_smooth_curve(ax, args, df, plot_x_column, plot_metric, custom_color_mapping)
                    
                    # Separate mode: finalize plot for each source
                    if not args.plot_merge_sources:
                        if args.plot_extra_lines:
                            ax = _plot_extra_lines(ax, args, plot_metric)
                        ax = _plot_settings(ax, args, df, plot_x_column, plot_metric, data_source)
                        plt.close(ax.figure)
                
                # Merge mode: finalize plot after all sources
                if args.plot_merge_sources and ax is not None:
                    if args.plot_extra_lines:
                        ax = _plot_extra_lines(ax, args, plot_metric)
                    ax = _plot_settings(ax, args, None, plot_x_column, plot_metric, df_map=df_map)
                    plt.close(ax.figure)

def _build_source_curve_color(args, df, data_source):
    """Build custom color mapping for this data source based on source-curve-color settings.
    
    Args:
        args: Command-line arguments
        df: DataFrame for the data source
        data_source: Name of the data source
    
    Returns:
        Dict mapping curve values to colors, or None if no custom coloring is requested
    """
    custom_color_mapping = None
    if args.plot_source_curve_color:
        custom_color_mapping = {}
        for curve_val in df[args.plot_curve].unique():
            color = source_curve.get_source_curve_color(
                data_source, curve_val, args.plot_source_curve_color
            )
            custom_color_mapping[curve_val] = color
    return custom_color_mapping

def _generate_merged_legend(df_map, args, plot_curve):
    """Generate legend for merged plot with data sources and extra lines.
    
    Args:
        df_map: Dict mapping data_source to DataFrame
        args: Command-line arguments
        plot_curve: The curve column name
    
    Returns:
        Tuple of (handles, labels) for legend
    """
    handles = []
    labels = []
    
    # Iterate through each data source and get unique curve values
    for data_source, df in df_map.items():
        for curve_val in df[plot_curve].unique():
            color = source_curve.get_source_curve_color(
                data_source, curve_val, args.plot_source_curve_color
            )
            label = source_curve.get_source_curve_label(
                data_source, plot_curve, curve_val, args.plot_source_curve_label
            )
            handles.append(Line2D([0], [0], color=color, linewidth=2))
            labels.append(label)
    
    # Add extra lines
    if args.plot_extra_lines:
        for line_name, line_config in args.plot_extra_lines.items():
            color = line_config.get('color', 'black')
            linestyle = line_config.get('linestyle', '-')
            marker = line_config.get('marker', 'o')
            label = line_config.get('label', line_name)
            handles.append(Line2D([0], [0], color=color, linestyle=linestyle,
                                 marker=marker, markersize=6, linewidth=2.5))
            labels.append(label)
    
    return handles, labels

def _data_prepare(args):
    """Prepare data for all sources.
    
    Returns a dict mapping data_source to DataFrame.
    """
    df_map = {}    
    for data_source in args.data_sources:
        df = _data_prepare_single_source(args, data_source)
        df_map[data_source] = df
    return df_map

def _data_prepare_single_source(args, data_source):
    print(f"Loading data for data_source: {data_source}")
    df = data_proc.load_and_preprocess(config.CSV_MAP[data_source])
    
    # Get physical dimensions for this data source (must be configured)
    physical_dimensions = config.get_physical_dimensions(data_source)
    physical_curve_column = physical_dimensions[0]  # N, slice_factor, or rollout_n
    physical_x_column = physical_dimensions[1]      # step
    
    # Collect all metrics that might need delta calculation
    all_metrics = list(set((args.plot_metrics or []) + ([args.fit_metric] if args.fit_metric else [])))
    
    # Prepare eval data
    df = data_proc.prepare_eval_data(
        df,
        eval_column=args.eval,
        curve_column=physical_curve_column,
        x_column=physical_x_column,
        calc_delta=any(metric is not None and metric.startswith('Delta') for metric in all_metrics),
        delta_base_step=args.delta_base_step
    )
    
    # Remove step=0 data (because E=0 will cause log10(E)=-inf)
    df = df[df['step'] > 0].reset_index(drop=True)
    
    # Apply clipping (use curve_column for curve grouping)
    if args.warmup_clip is not None or args.warmup_clip_to is not None or args.ending_clip is not None or args.ending_clip_to is not None:
        df = data_proc.apply_clip(
            df, 
            curve_column=physical_curve_column,
            warmup_clip=args.warmup_clip,
            warmup_clip_to=args.warmup_clip_to,
            ending_clip=args.ending_clip,
            ending_clip_to=args.ending_clip_to
        )
    return df

def _plot_fit_prediction(ax, args, df, fitter, plot_x_column, plot_metric, custom_color_mapping=None):
    """Plot fitted prediction curves.
    
    Args:
        custom_color_mapping: Optional dict mapping curve values to colors
    """
    fitter_context = fitter.get_context()
    fit_curve_column = fitter_context["fit_curve"]
    fit_x = fitter_context["fit_x"]
    fit_metric = fitter_context["metric"]

    predict_x_column_list = [fit_curve_column, fit_x]
    pred_column = plot_metric + "_pred"
    # predict R
    _pred_R = fit.predict_on(
        fitter, 
        df, 
        x_column_list=predict_x_column_list, 
        y_transform_recover=R_FROM[fit_metric],
    )
    
    df.loc[:, pred_column] = R_TO[plot_metric](_pred_R)  # Use .loc to avoid SettingWithCopyWarning
    
    ax = plot.plot_curves(
        df, 
        curve_column=args.plot_curve, # go with plot_curve
        x_column=plot_x_column, 
        y_column=pred_column, 
        use_line=True,
        use_scatter=False,
        highlight_curves=args.highlight_curves_predict,
        highlight_alpha=args.highlight_alpha,
        highlight_width=args.highlight_width,
        line_alpha=args.line_alpha,
        line_width=args.line_width,
        scatter_alpha=args.scatter_alpha,
        scatter_size=args.scatter_size,
        scatter_marker=args.scatter_marker,
        custom_color_mapping=custom_color_mapping,
        ax=ax,
    )
    return ax

def _plot_raw_data(ax, args, df, plot_x_column, plot_metric, custom_color_mapping=None):
    """Plot raw data points/lines.
    
    Args:
        custom_color_mapping: Optional dict mapping curve values to colors
    """
    ax = plot.plot_curves(
        df,
        curve_column=args.plot_curve,
        x_column=plot_x_column,
        y_column=plot_metric,
        y_std_column=plot_metric + '_std' if args.add_std else None,
        use_scatter=args.plot_use_scatter,
        use_line=args.plot_use_line,
        highlight_curves=args.highlight_curves_plot,
        highlight_alpha=args.highlight_alpha,
        highlight_width=args.highlight_width,
        line_alpha=args.line_alpha,
        line_width=args.line_width,
        scatter_alpha=args.scatter_alpha,
        scatter_size=args.scatter_size,
        scatter_marker=args.scatter_marker,
        custom_color_mapping=custom_color_mapping,
        ax=ax,
    )
    return ax

def _plot_smooth_curve(ax, args, df, plot_x_column, plot_metric, custom_color_mapping=None):
    """Plot smoothed curves.
    
    Args:
        custom_color_mapping: Optional dict mapping curve values to colors
    """
    smooth_out_column = plot_metric + "_smooth"
    
    df_smooth = data_proc.smooth_df(
        df,
        curve_column=args.plot_curve,
        col_x=plot_x_column,
        col_y=plot_metric,
        col_y_out=smooth_out_column,
        monotonic=args.smooth_monotonic,
        increasing=args.smooth_increasing,
        strict=args.smooth_strict,
        s_factor=args.s_factor, 
        k_spline=args.k_spline,
        rolling_window=args.rolling_window, 
        min_se=args.min_se, 
        x_inv_weight_power=args.x_inv_weight_power,
        use_linear=False
    )
    
    ax = plot.plot_curves(
        df_smooth,
        curve_column=args.plot_curve,
        x_column=plot_x_column,
        y_column=smooth_out_column,
        y_std_column=None,
        use_scatter=False,
        use_line=True,
        highlight_curves=args.highlight_curves_plot,
        highlight_alpha=args.highlight_alpha,
        highlight_width=args.highlight_width,
        line_alpha=args.line_alpha,
        line_width=args.line_width,
        scatter_alpha=args.scatter_alpha,
        scatter_size=args.scatter_size,
        scatter_marker=args.scatter_marker,
        custom_color_mapping=custom_color_mapping,
        ax=ax
    )
    return ax

def _plot_extra_lines(ax, args, plot_metric):
    
    extra_lines = args.plot_extra_lines
    
    for line_name, line_config in extra_lines.items():
        # Extract x and y data
        x_data = np.array(line_config['x'])
        y_data = np.array(line_config['y'])
        
        # Extract plotting parameters with unified naming
        color = line_config.get('color', 'black')
        label = line_config.get('label', line_name)
        
        # Line parameters
        line_style = line_config.get('line_style', '-')
        line_width = line_config.get('line_width', 2.5)
        line_alpha = line_config.get('line_alpha', 1.0)
        
        # Scatter parameters
        scatter_marker = line_config.get('scatter_marker', 'o')
        scatter_size = line_config.get('scatter_size', 25)
        scatter_alpha = line_config.get('scatter_alpha', 1.0)
        
        # Plot using plot_basic
        ax = plot.plot_basic(
            x=x_data,
            y=y_data,
            use_scatter=True,
            scatter_alpha=scatter_alpha,
            scatter_s=scatter_size,
            scatter_marker=scatter_marker,
            use_line=True,
            line_alpha=line_alpha,
            line_width=line_width,
            line_style=line_style,
            color=color,
            ax=ax
        )
    
    return ax

def _plot_settings(ax, args, df, plot_x_column, plot_metric, data_source=None, df_map=None):
    """Apply plot settings with optional merged legend support.
    
    Args:
        ax: Matplotlib axes
        args: Command-line arguments
        df: DataFrame (for separate mode)
        plot_x_column: X-axis column name
        plot_metric: Y-axis metric name
        data_source: Data source name (for separate mode, optional)
        df_map: Dict mapping data_source to DataFrame (for merge mode, optional)
    """
    process_plot_x_label = args.plot_x_label if args.plot_x_label else config.DEFAULT_LABELS[plot_x_column]
    process_plot_y_label = args.plot_y_label if args.plot_y_label else config.DEFAULT_LABELS[plot_metric]
    
    # Generate title
    if args.plot_title:
        process_plot_title = args.plot_title
    elif args.plot_titles and plot_x_column in args.plot_titles:
        process_plot_title = args.plot_titles[plot_x_column]
    elif df_map is not None:
        # Merge mode: use comparison title
        process_plot_title = f"{config.TEST_EVALS[args.eval]['plot_str']} - Comparison"
    elif args.plot_title_template:
        template_vars = {
            'eval': config.TEST_EVALS[args.eval]['plot_str'],
            'fit_x': args.fit_x if args.fit else 'None',
            'plot_x': plot_x_column,
            'metric': plot_metric
        }
        process_plot_title = args.plot_title_template.format(**template_vars)
    else:
        if args.fit:
            process_plot_title = f"{config.TEST_EVALS[args.eval]['plot_str']} (fitted on {args.fit_x}, plotted on {plot_x_column})"
        else:
            process_plot_title = f"{config.TEST_EVALS[args.eval]['plot_str']} (plotted on {plot_x_column})"
    
    # Generate legend
    if args.plot_use_legend:
        if df_map is not None:
            # Merge mode: use merged legend
            handles, labels = _generate_merged_legend(df_map, args, args.plot_curve)
            legend_handles_labels = (handles, labels)
        else:
            # Separate mode: use default legend
            legend_handles_labels = plot.prepare_legend(df, args.plot_curve)
    else:
        legend_handles_labels = None
    
    # Determine filename prefix
    if df_map is not None:
        # Merge mode: combine all sources
        filename_prefix = args.output_prefix + "_".join(args.data_sources) + "_"
    else:
        # Separate mode: single source
        filename_prefix = args.output_prefix + data_source + "_"
    
    plot.plot_basic_settings(
        ax=ax,
        x_scale=args.plot_x_scale,
        y_scale=args.plot_y_scale,
        x_label=process_plot_x_label,
        y_label=process_plot_y_label,
        title=process_plot_title,
        use_legend=args.plot_use_legend,
        legend_handles_labels=legend_handles_labels,
        legend_loc=args.plot_legend_loc,
        legend_bbox_to_anchor=args.plot_legend_bbox_to_anchor,
        x_tick_format=args.x_tick_format,
        y_tick_format=args.y_tick_format,
        x_tick_spacing=args.x_tick_spacing,
        y_tick_spacing=args.y_tick_spacing,
        x_grid_spacing=args.x_grid_spacing,
        y_grid_spacing=args.y_grid_spacing,
        x_tick_subs=args.x_tick_subs,
        y_tick_subs=args.y_tick_subs,
        x_tick_subs_log=args.x_tick_subs_log,
        y_tick_subs_log=args.y_tick_subs_log,
        # Save configuration
        save_to_dir=args.output_base_dir,
        save_to_filename_prefix=filename_prefix,
        plot_eval_column=args.eval,
        plot_curve=args.plot_curve,
        plot_x_column=plot_x_column,
        plot_metric=plot_metric,
    )
    return ax

def _fit_multiple(args, df_map):
    return [_fit_once(args, df, data_source) for data_source, df in df_map.items()]

def _fit_once(args, df, data_source):
    # Apply fit curve mask if provided
    if args.fit_curve_mask is not None:
        print(f"Filtering fit curves ({args.fit_curve}): {args.fit_curve_mask}")
        df = df[df[args.fit_curve].isin(args.fit_curve_mask)].copy()
    
    print(f"\n=== Fitting on data_source: {data_source}, L({args.fit_curve}, {args.fit_x}) ===")
    print(f"Using model: {args.fit_model}")
    
    FitterClass = get_model_class(args.fit_model)
    
    fitter = fit.fit_on(
        FitterClass,
        df, 
        eval_name=args.eval, 
        x_column_list=[args.fit_curve, args.fit_x],
        y_transform=R_TO[args.fit_metric],
        x_inv_weight_power=args.x_inv_weight_power,
        cma_verbose_interval=args.fit_cma_verbose_interval,
    )

    fitter.set_context({
        "data_source": data_source,
        "fit_model": args.fit_model,
        "fit_curve": args.fit_curve,
        "fit_x": args.fit_x,
        "eval": args.eval,
        "metric": args.fit_metric,
        "warmup_clip": args.warmup_clip if args.warmup_clip is not None else 0,
        "ending_clip": args.ending_clip if args.ending_clip is not None else 0,
        "x_inv_weight_power": args.x_inv_weight_power,
        "fit_curve_mask": args.fit_curve_mask,
    })
    
    info = fitter.get_info()
    print(f"Fit quality: R²={info['r2']:.6f}, Loss={info['loss']:.6f}, n_points={info['n_points']}")
    
    return fitter

        
def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    args = process_parsed_args(args)
    validate_required_args(args)
    validate_args(args)
    run_scaling_analysis(args)

if __name__ == "__main__":
    main()
