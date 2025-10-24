#!/usr/bin/env python3
"""
Scaling Law Pipeline - Multi-Eval Analysis (CLI Version)
Processes multiple test evals from experiment data and generates scaling law plots for each eval
"""

import numpy as np
import matplotlib.pyplot as plt
from src.common import data_proc
from src.common import config
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

def _set_args_from_context(args, context):
    args.fit_model = context["fit_model"]
    args.curve = context["fit_curve"]
    args.fit_x = context["fit_x"]
    args.eval = context["eval"]
    args.fit_metric = context["metric"]
    args.warmup_clip = context["warmup_clip"]
    args.ending_clip = context["ending_clip"]
    args.x_inv_weight_power = context["x_inv_weight_power"]
    args.curve_mask = context["curve_mask"]
    return args

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
                ax = None
                
                for data_source, df in df_map.items():
                    # Add fitted prediction curves if fitting was done
                    if args.plot_fit and (data_source, args.fit_x) in fitter_map:
                        ax = _plot_fit_prediction(ax, args, df, fitter_map[(data_source, args.fit_x)], plot_x_column, plot_metric)
                        
                    # Process the actual data plot
                    ax = _plot_raw_data(ax, args, df, plot_x_column, plot_metric)
                    
                    # Add smooth curves if requested
                    if args.add_smooth:
                        ax = _plot_smooth_curve(ax, args, df, plot_x_column, plot_metric)
                    
                    # Apply plot_basic_settings after plotting (now includes save logic)
                    ax = _plot_settings(ax, args, df, plot_x_column, plot_metric, data_source)
                    plt.close(ax.figure)

def _data_prepare_single_source(args, data_source):
    print(f"Loading data for data_source: {data_source}")
    df = data_proc.load_and_preprocess(config.CSV_MAP[data_source])
    
    # Prepare eval data once
    unique_x_columns = list(set((args.plot_x_columns or []) + ([args.fit_x] if args.fit_x else [])))
    unique_metrics = list(set((args.plot_metrics or []) + ([args.fit_metric] if args.fit_metric else [])))
    df = data_proc.prepare_eval_data(
        df,
        eval_column=args.eval,
        curve_column=args.curve,
        x_columns=unique_x_columns,
        calc_delta=any(metric is not None and metric.startswith('Delta') for metric in unique_metrics),
        delta_base_step=args.delta_base_step
    )
    
    # Remove step=0 data (because E=0 will cause log10(E)=-inf)
    df = df[df['step'] > 0].reset_index(drop=True)
    
    # Apply clipping
    if args.warmup_clip is not None or args.warmup_clip_to is not None or args.ending_clip is not None or args.ending_clip_to is not None:
        df = data_proc.apply_clip(
            df, 
            curve_column=args.curve,
            warmup_clip=args.warmup_clip if args.warmup_clip is not None else 0,
            warmup_clip_to=args.warmup_clip_to,
            ending_clip=args.ending_clip if args.ending_clip is not None else 0,
            ending_clip_to=args.ending_clip_to
        )
    
    # Apply curve mask filter
    if args.curve_mask is not None:
        print(f"Filtering curves: {args.curve_mask}")
        df = df[df[args.curve].isin(args.curve_mask)]
    return df

def _data_prepare(args):
    df_map = {}    
    for data_source in args.data_sources:
        df_map[data_source] = _data_prepare_single_source(args, data_source)
    return df_map

def _plot_fit_prediction(ax, args, df, fitter, plot_x_column, plot_metric):
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
    
    df[pred_column] = R_TO[plot_metric](_pred_R)
    
    ax = plot.plot_curves(
        df, 
        curve_column=fit_curve_column, 
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
        custom_color_mapping=None,
        ax=ax,
    )
    return ax

def _plot_raw_data(ax, args, df, plot_x_column, plot_metric):
    ax = plot.plot_curves(
        df,
        curve_column=args.curve,
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
        ax=ax,
    )
    return ax

def _plot_smooth_curve(ax, args, df, plot_x_column, plot_metric):
    smooth_out_column = plot_metric + "_smooth"
    
    df_smooth = data_proc.smooth_df(
        df,
        curve_column=args.curve,
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
        curve_column=args.curve,
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
        ax=ax
    )
    return ax
def _plot_settings(ax, args, df, plot_x_column, plot_metric, data_source):
    process_plot_x_label = args.plot_x_label if args.plot_x_label else config.DEFAULT_LABELS[plot_x_column]
    process_plot_y_label = args.plot_y_label if args.plot_y_label else config.DEFAULT_LABELS[plot_metric]
    
    # Generate title
    if args.plot_title:
        process_plot_title = args.plot_title
    elif args.plot_titles and plot_x_column in args.plot_titles:
        process_plot_title = args.plot_titles[plot_x_column]
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
    
    # Prepare legend if needed
    if args.plot_use_legend:
        legend_handles_labels = plot.prepare_legend(df, args.curve)
    else:
        legend_handles_labels = None
    
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
        save_to_filename_prefix=args.output_prefix+data_source+'_',
        plot_eval_column=args.eval,
        plot_curve=args.curve,
        plot_x_column=plot_x_column,
        plot_metric=plot_metric,
    )
    return ax

def _fit_multiple(args, df_map):
    return [_fit_once(args, df, data_source) for data_source, df in df_map.items()]

def _fit_once(args, df, data_source):
    print(f"\n=== Fitting for x_column: {args.fit_x}, curve_column: {args.curve} ===")
    print(f"Using model: {args.fit_model}")
    
    FitterClass = get_model_class(args.fit_model)
    
    fitter = fit.fit_on(
        FitterClass,
        df, 
        eval_name=args.eval, 
        x_column_list=[args.curve, args.fit_x],
        y_transform=R_TO[args.fit_metric],
        warmup_clip=0,
        warmup_clip_to=None,
        ending_clip=0,
        ending_clip_to=None,
        x_inv_weight_power=args.x_inv_weight_power,
        cma_verbose_interval=args.fit_cma_verbose_interval,
    )

    fitter.set_context({
        "data_source": data_source,
        "fit_model": args.fit_model,
        "fit_curve": args.curve,
        "fit_x": args.fit_x,
        "eval": args.eval,
        "metric": args.fit_metric,
        "warmup_clip": args.warmup_clip if args.warmup_clip is not None else 0,
        "ending_clip": args.ending_clip if args.ending_clip is not None else 0,
        "x_inv_weight_power": args.x_inv_weight_power,
        "curve_mask": args.curve_mask,
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
