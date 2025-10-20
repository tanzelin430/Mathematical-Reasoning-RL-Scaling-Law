#!/usr/bin/env python3


import os
import sys
from typing import Callable
import matplotlib.pyplot as plt

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import data_proc
import plot
import config
from pathlib import Path
import fit

def plot_curves_with_smooth(
    # plot params
    df,
    curve_column: str,
    x_column: str,
    y_column: str,  # column for raw scatter points
    y_std_column: str = None, 
    x_scale: str = None,
    y_scale: str = None,
    x_label: str = None,
    y_label: str = None,
    title: str = None,
    use_scatter: bool = False,
    use_line: bool = False,
    smooth_use_scatter: bool = False,
    smooth_use_line: bool = False,
    use_legend: bool = False,
    legend_lambda: Callable = None,
    legend_loc: str = 'best',
    legend_bbox_to_anchor: tuple = None,
    legend_fontsize: int = 8,
    # Highlight specific curves (backward compatible)
    highlight_curves: list = None,  # List of curve values to highlight
    highlight_line_alpha: float = 1.0,  # Alpha for highlighted curves
    highlight_line_width: float = None,  # Linewidth for highlighted curves
    # Line and scatter styling
    line_alpha: float = 1.0,
    line_width: float = 2.0,
    scatter_alpha: float = 0.3,
    scatter_size: float = 8.0,
    scatter_marker: str = 'o',
    # Axis formatting control
    x_tick_format: str = None,  # 'auto', 'decimal', 'sci', or 'plain'
    y_tick_format: str = 'auto',  # 'auto', 'decimal', 'sci', or 'plain'
    # Tick and grid spacing
    x_tick_spacing: float = None,
    y_tick_spacing: float = None,
    x_grid_spacing: float = None,
    y_grid_spacing: float = None,
    ax=None,
    # smooth params
    add_smooth: bool = False, 
    smooth_monotonic: bool = False,
    smooth_increasing: bool = True,
    smooth_strict: bool = False,
    warmup_frac_raw: float = 0,
    warmup_frac_smooth: float = 0,
    warmup_clip_raw: int = None,  # Absolute count clipping for raw data (clips first N points)
    warmup_clip_smooth: int = None,  # Absolute count clipping for smooth data (clips first N points)
    ending_clip_raw: int = None,  # Absolute ending clipping for raw data (clips last steps > ending_step)
    ending_clip_smooth: int = None,  # Absolute ending clipping for smooth data (clips last steps > ending_step)
    s_factor: float = 1,
    k_spline: int = 5,
    rolling_window: int = 200,
    min_se: float = 1e-6,
    x_inv_weight_power: float = 0.3,
    smooth_out_column: str = None,
    use_linear: bool = False,
    # Custom color mapping override
    custom_color_mapping: dict = None,  # Custom color mapping to override config colors
):

    # Apply warmup clipping for raw data
    if warmup_clip_raw is not None:
        # Use absolute count clipping (clips first N data points per curve)
        df = data_proc.apply_clip(df, curve_column=curve_column, warmup_step=warmup_clip_raw)
    elif warmup_frac_raw > 0:
        # Use relative clipping
        df = data_proc.apply_clip(df, curve_column=curve_column, warmup_frac=warmup_frac_raw)
    
    # Apply ending clipping for raw data
    if ending_clip_raw is not None:
        # Use absolute ending clipping (clips steps > ending_step per curve)
        df = data_proc.apply_clip(df, curve_column=curve_column, ending_step=ending_clip_raw)
    
    df_R_smooth = None
    if add_smooth:
        if not smooth_out_column:
            smooth_out_column = f"_{y_column}_smooth"
        # monotonic smoothing
        df_R_smooth = data_proc.smooth_df(
            df, 
            curve_column=curve_column,
            col_x=x_column, 
            col_y=y_column, 
            col_y_out=smooth_out_column,
            monotonic=smooth_monotonic,
            increasing=smooth_increasing,
            strict=smooth_strict,
            s_factor=s_factor, 
            k_spline=k_spline,
            rolling_window=rolling_window, 
            min_se=min_se, 
            x_inv_weight_power=x_inv_weight_power,
            use_linear=use_linear
        )
        # Apply warmup clipping for smooth data
        if warmup_clip_smooth is not None:
            # Use absolute count clipping (clips first N data points per curve)
            df_R_smooth = data_proc.apply_clip(df_R_smooth, curve_column=curve_column, warmup_step=warmup_clip_smooth)
        elif warmup_frac_smooth > 0:
            # Use relative clipping
            df_R_smooth = data_proc.apply_clip(df_R_smooth, curve_column=curve_column, warmup_frac=warmup_frac_smooth)
        
        # Apply ending clipping for smooth data
        if ending_clip_smooth is not None:
            # Use absolute ending clipping (clips steps > ending_step per curve)
            df_R_smooth = data_proc.apply_clip(df_R_smooth, curve_column=curve_column, ending_step=ending_clip_smooth)

    ax = plot.plot_curves(
        df,
        curve_column=curve_column,
        x_column=x_column,
        y_column=y_column,
        df_smooth=df_R_smooth,
        y_smooth_column=smooth_out_column,
        y_width_column=y_std_column,
        width_on_smooth=add_smooth,
        x_scale=x_scale,
        y_scale=y_scale,
        x_label=x_label,
        y_label=y_label,
        title=title,
        use_scatter=use_scatter,
        use_line=use_line,
        smooth_use_scatter=smooth_use_scatter,
        smooth_use_line=smooth_use_line,
        use_legend=use_legend,
        legend_lambda=legend_lambda,
        legend_loc=legend_loc,
        legend_bbox_to_anchor=legend_bbox_to_anchor,
        legend_fontsize=legend_fontsize,
        # Pass highlight parameters
        highlight_curves=highlight_curves,
        highlight_line_alpha=highlight_line_alpha,
        highlight_line_width=highlight_line_width,
        # Pass line and scatter styling - applies to raw data, smooth, and fitting
        line_alpha=line_alpha,
        line_width=line_width,
        scatter_alpha=scatter_alpha,
        scatter_s=scatter_size,
        scatter_marker=scatter_marker,
        # Pass axis formatting
        x_tick_format=x_tick_format,
        y_tick_format=y_tick_format,
        # Pass tick and grid spacing
        x_tick_spacing=x_tick_spacing,
        y_tick_spacing=y_tick_spacing,
        x_grid_spacing=x_grid_spacing,
        y_grid_spacing=y_grid_spacing,
        # Pass custom color mapping
        custom_color_mapping=custom_color_mapping,
        ax=ax
    )
    return ax

def process_single_eval(
    df, 
    plot_x_column: str, 
    plot_eval_column: str, 
    plot_metric: str, # 'R', 'ErrRate', 'DeltaReward', 'DeltaErrRate'
    plot_curve_column: str,
    plot_curve_mask: list[str]=None,
    plot_x_label: str=None,
    plot_y_label: str=None,
    plot_x_scale: str=None,
    plot_y_scale: str=None,
    plot_title: str=None, 
    plot_use_legend: bool = False,
    plot_use_scatter: bool = True,
    plot_use_line: bool = False,
    plot_smooth_use_scatter: bool = False,
    plot_smooth_use_line: bool = True,
    plot_legend_loc: str = 'best',
    plot_legend_bbox_to_anchor: tuple = None,
    plot_legend_lambda: Callable = None,
    plot_legend_fontsize: int = 8,
    # Highlight specific curves (backward compatible)
    plot_highlight_curves: list = None,  # List of curve values to highlight
    plot_highlight_line_alpha: float = 1.0,  # Alpha for highlighted curves
    plot_highlight_line_width: float = 3.0,  # Linewidth for highlighted curves
    # Line and scatter styling
    line_alpha: float = 1.0,
    line_width: float = 2.0,
    scatter_alpha: float = 0.3,
    scatter_size: float = 8.0,
    scatter_marker: str = 'o',
    # Axis formatting control
    x_tick_format: str = None,  # 'auto', 'decimal', 'sci', or 'plain'
    y_tick_format: str = 'auto',  # 'auto', 'decimal', 'sci', or 'plain'
    # Tick and grid spacing
    x_tick_spacing: float = None,
    y_tick_spacing: float = None,
    x_grid_spacing: float = None,
    y_grid_spacing: float = None,
    ax: plt.Axes=None,
    # smooth
    delta_base_step: int = 0,
    add_smooth=False,
    add_std=False,
    smooth_monotonic=False,
    smooth_increasing=None,
    smooth_strict=False,
    warmup_frac_raw: float=0,
    warmup_frac_smooth: float=0,
    warmup_clip_raw: int = None,  # Absolute count clipping for raw data
    warmup_clip_smooth: int = None,  # Absolute count clipping for smooth data
    ending_clip_raw: int = None,  # Absolute ending clipping for raw data
    ending_clip_smooth: int = None,  # Absolute ending clipping for smooth data
    calc_delta: bool=False, # in case there's no step 0 but don't care of it.
    s_factor=1,
    k_spline=5,
    rolling_window=200,
    min_se=1e-6,
    x_inv_weight_power=0.3,
    use_linear=False,
    # Custom color mapping override
    custom_color_mapping: dict = None,  # Custom color mapping to override config colors
    ):
    
    
    # =============================================================================
    # PHASE 1: DATA PREPROCESSING
    # =============================================================================
    
    df = df.rename(columns={plot_eval_column: 'R'})
    # Calculate error rate
    df['ErrRate'] = 1 - df['R']
    # Calculate std
    R_std = df.groupby([plot_curve_column, 'step'])['R'].std().to_frame('R_std')
    ErrRate_std = df.groupby([plot_curve_column, 'step'])['ErrRate'].std().to_frame('ErrRate_std')
    if calc_delta:
        df['DeltaReward'] = data_proc.calc_delta_y(df, 'R', base_step=delta_base_step, curve_column=plot_curve_column)
        df['DeltaErrRate'] = data_proc.calc_delta_y(df, 'ErrRate', base_step=delta_base_step, curve_column=plot_curve_column)
        # Calculate std
        DeltaReward_std = df.groupby([plot_curve_column, 'step'])['DeltaReward'].std().to_frame('DeltaReward_std')
        DeltaErrRate_std = df.groupby([plot_curve_column, 'step'])['DeltaErrRate'].std().to_frame('DeltaErrRate_std')
    # Merge multi rollout in same step
    # 注意：需要包含plot_x_column以避免不同x值的数据被错误合并
    merge_columns = [plot_curve_column, 'step']
    if plot_x_column not in merge_columns:
        merge_columns.append(plot_x_column)
    df = data_proc.merge_duplicate_steps(df, group_columns=merge_columns, mode='mean')

    # Add std cols back to df
    std_merge_columns = [plot_curve_column, 'step']
    df = df.merge(R_std, on=std_merge_columns)
    df = df.merge(ErrRate_std, on=std_merge_columns)
    if calc_delta:
        df = df.merge(DeltaReward_std, on=std_merge_columns)
        df = df.merge(DeltaErrRate_std, on=std_merge_columns)
    if plot_curve_mask is not None:
        print("unique values of plot_curve_column:", df[plot_curve_column].unique())
        print("plot_curve_mask:", plot_curve_mask)
        df = df[df[plot_curve_column].isin(plot_curve_mask)]
    df.sort_values(plot_x_column, inplace=True)
    
    ax = plot_curves_with_smooth(
        df,
        curve_column=plot_curve_column,
        x_column=plot_x_column,
        y_column=plot_metric,
        y_std_column=plot_metric+'_std' if add_std else None,
        x_scale=plot_x_scale,
        y_scale=plot_y_scale,
        x_label=plot_x_label,
        y_label=plot_y_label,
        title=plot_title,
        use_legend=plot_use_legend,
        use_scatter=plot_use_scatter,
        use_line=plot_use_line,
        smooth_use_scatter=plot_smooth_use_scatter,
        smooth_use_line=plot_smooth_use_line,
        legend_lambda=plot_legend_lambda,
        legend_loc=plot_legend_loc,
        legend_bbox_to_anchor=plot_legend_bbox_to_anchor,
        legend_fontsize=plot_legend_fontsize,
        # Pass highlight parameters
        highlight_curves=plot_highlight_curves,
        highlight_line_alpha=plot_highlight_line_alpha,
        highlight_line_width=plot_highlight_line_width,
        # Pass line and scatter styling - applies to raw data, smooth, and fitting
        line_alpha=line_alpha,
        line_width=line_width,
        scatter_alpha=scatter_alpha,
        scatter_size=scatter_size,
        scatter_marker=scatter_marker,
        # Pass axis formatting
        x_tick_format=x_tick_format,
        y_tick_format=y_tick_format,
        # Pass tick and grid spacing
        x_tick_spacing=x_tick_spacing,
        y_tick_spacing=y_tick_spacing,
        x_grid_spacing=x_grid_spacing,
        y_grid_spacing=y_grid_spacing,
        ax=ax,
        # smooth
        add_smooth=add_smooth,
        smooth_monotonic=smooth_monotonic,
        smooth_increasing=smooth_increasing,
        smooth_strict=smooth_strict,
        warmup_frac_raw=warmup_frac_raw,
        warmup_frac_smooth=warmup_frac_smooth,
        warmup_clip_raw=warmup_clip_raw,
        warmup_clip_smooth=warmup_clip_smooth,
        ending_clip_raw=ending_clip_raw,
        ending_clip_smooth=ending_clip_smooth,
        s_factor=s_factor,
        k_spline=k_spline,
        rolling_window=rolling_window,
        min_se=min_se,
        x_inv_weight_power=x_inv_weight_power,
        smooth_out_column=plot_metric+"_smooth" if add_smooth else None,
        use_linear=use_linear,
        # Pass custom color mapping
        custom_color_mapping=custom_color_mapping
    )

    # Save logic moved to plot_basic_settings - just return ax
    return ax


def process_single_eval_multi_metrics(
    df, plot_x_column: str, 
    plot_eval_column: str, 
    plot_curve_column: str,
    plot_x_label: str=None,
    plot_y_label: str=None,
    plot_x_scale: str=None,
    plot_y_scale: str=None,
    plot_title: str=None, 
    plot_use_legend: bool = False,
    plot_use_scatter: bool = False,
    plot_use_line: bool = False,
    plot_smooth_use_scatter: bool = False,
    plot_smooth_use_line: bool = False,
    plot_legend_lambda: Callable = None,
    plot_legend_loc: str = 'best',
    plot_legend_bbox_to_anchor: tuple = None,
    plot_legend_fontsize: int = 8,
    plot_reward: bool=False, plot_err_rate: bool=False, plot_delta_reward: bool=False, plot_delta_err_rate: bool=False, 
    ax_reward: plt.Axes=None, ax_err_rate: plt.Axes=None, ax_delta_reward: plt.Axes=None, ax_delta_err_rate: plt.Axes=None, 
    delta_base_step: int = 0,
    warmup_frac_raw: float = 0,
    warmup_frac_smooth: float = 0,
    warmup_clip_raw: int = None,  # Absolute count clipping for raw data
    warmup_clip_smooth: int = None,  # Absolute count clipping for smooth data
    ending_clip_raw: int = None,  # Absolute ending clipping for raw data
    ending_clip_smooth: int = None,  # Absolute ending clipping for smooth data
    output_dir: str = None,
    use_linear=False,
    ):
    
    # # Format eval names
    # eval_file_str, eval_plot_str = TEST_EVALS[eval_name]['file_str'], TEST_EVALS[eval_name]['plot_str']
    
    # Create eval-specific output directory
    # eval_output_dir = Path(output_dir) / eval_file_str
    # eval_output_dir.mkdir(parents=True, exist_ok=True)
    
    # =============================================================================
    # PHASE 1: DATA PREPROCESSING
    # =============================================================================
    
    df = df.rename(columns={plot_eval_column: 'R'})
    # Calculate error rate
    df['ErrRate'] = 1 - df['R']
    df['DeltaReward'] = data_proc.calc_delta_y(df, 'R', base_step=delta_base_step, curve_column=plot_curve_column)
    df['DeltaErrRate'] = data_proc.calc_delta_y(df, 'ErrRate', base_step=delta_base_step, curve_column=plot_curve_column)
    # Calculate std
    R_std = df.groupby([plot_curve_column, 'step'])['R'].std().to_frame('R_std')
    ErrRate_std = df.groupby([plot_curve_column, 'step'])['ErrRate'].std().to_frame('ErrRate_std')
    DeltaReward_std = df.groupby([plot_curve_column, 'step'])['DeltaReward'].std().to_frame('DeltaReward_std')
    DeltaErrRate_std = df.groupby([plot_curve_column, 'step'])['DeltaErrRate'].std().to_frame('DeltaErrRate_std')
    # Merge multi rollout in same step
    # 注意：需要包含plot_x_column以避免不同x值的数据被错误合并
    merge_columns = [plot_curve_column, 'step']
    if plot_x_column not in merge_columns:
        merge_columns.append(plot_x_column)
    df = data_proc.merge_duplicate_steps(df, group_columns=merge_columns, mode='mean')

    # Add std cols back to df
    std_merge_columns = [plot_curve_column, 'step']
    df = df.merge(R_std, on=std_merge_columns)
    df = df.merge(ErrRate_std, on=std_merge_columns)
    df = df.merge(DeltaReward_std, on=std_merge_columns)
    df = df.merge(DeltaErrRate_std, on=std_merge_columns)
    df.sort_values(plot_x_column, inplace=True)
    
    if plot_reward:
        ax = plot_curves_with_smooth(
            df,
            curve_column=plot_curve_column,
            x_column=plot_x_column,
            y_column="R",
            y_std_column='R_std',
            x_scale=plot_x_scale,
            y_scale=plot_y_scale,
            x_label=plot_x_label,
            y_label=plot_y_label,
            title=plot_title,
            use_scatter=plot_use_scatter,
            use_line=plot_use_line,
            smooth_use_scatter=plot_smooth_use_scatter,
            smooth_use_line=plot_smooth_use_line,
            use_legend=plot_use_legend,
            legend_lambda=plot_legend_lambda,
            legend_loc=plot_legend_loc,
            legend_bbox_to_anchor=plot_legend_bbox_to_anchor,
            legend_fontsize=plot_legend_fontsize,
            ax=ax_reward,
            # smooth
            add_smooth=True,
            smooth_monotonic=False,
            smooth_increasing=None,
            smooth_strict=False,
            warmup_frac_raw=warmup_frac_raw,
            warmup_frac_smooth=warmup_frac_smooth,
            warmup_clip_raw=warmup_clip_raw,
            warmup_clip_smooth=warmup_clip_smooth,
            ending_clip_raw=ending_clip_raw,
            ending_clip_smooth=ending_clip_smooth,
            s_factor=1,
            k_spline=5,
            rolling_window=200,
            min_se=1e-6,
            x_inv_weight_power=0.3,
            smooth_out_column="R_smooth",
            use_linear=use_linear
        )
        if ax_reward is None:
            plt.tight_layout()
            plt.savefig(output_dir / plot_eval_column / f"{plot_x_column}_reward.pdf", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved {output_dir / plot_eval_column / plot_x_column}_reward.pdf")
    if plot_err_rate:
        ax = plot_curves_with_smooth(
            df,
            curve_column=plot_curve_column,
            x_column=plot_x_column,
            y_column="ErrRate",
            y_std_column='ErrRate_std',
            x_scale=plot_x_scale,
            y_scale=plot_y_scale,
            x_label=plot_x_label,
            y_label=plot_y_label,
            title=plot_title,
            use_scatter=plot_use_scatter,
            use_line=plot_use_line,
            smooth_use_scatter=plot_smooth_use_scatter,
            smooth_use_line=plot_smooth_use_line,
            use_legend=plot_use_legend,
            legend_lambda=plot_legend_lambda,
            legend_loc=plot_legend_loc,
            legend_bbox_to_anchor=plot_legend_bbox_to_anchor,
            legend_fontsize=plot_legend_fontsize,
            ax=ax_err_rate,
            # smooth
            add_smooth=True,
            smooth_monotonic=False,
            smooth_increasing=None,
            smooth_strict=False,
            warmup_frac_raw=warmup_frac_raw,
            warmup_frac_smooth=warmup_frac_smooth,
            warmup_clip_raw=warmup_clip_raw,
            warmup_clip_smooth=warmup_clip_smooth,
            ending_clip_raw=ending_clip_raw,
            ending_clip_smooth=ending_clip_smooth,
            s_factor=1,
            k_spline=5,
            rolling_window=200,
            min_se=1e-6,
            x_inv_weight_power=0.3,
            smooth_out_column="ErrRate_smooth",
            use_linear=use_linear
        )
        if ax_err_rate is None:
            plt.tight_layout()
            plt.savefig(output_dir / plot_eval_column / f"{plot_x_column}_err_rate.pdf", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved {output_dir / plot_eval_column / plot_x_column}_err_rate.pdf")
    if plot_delta_reward:
        ax = plot_curves_with_smooth(
            df,
            curve_column=plot_curve_column,
            x_column=plot_x_column,
            y_column="DeltaReward",
            y_std_column='DeltaReward_std',
            x_scale=plot_x_scale,
            y_scale=plot_y_scale,
            x_label=plot_x_label,
            y_label=plot_y_label,
            title=plot_title,
            use_scatter=plot_use_scatter,
            use_line=plot_use_line,
            smooth_use_scatter=plot_smooth_use_scatter,
            smooth_use_line=plot_smooth_use_line,
            use_legend=plot_use_legend,
            legend_lambda=plot_legend_lambda,
            legend_loc=plot_legend_loc,
            legend_bbox_to_anchor=plot_legend_bbox_to_anchor,
            legend_fontsize=plot_legend_fontsize,
            ax=ax_delta_reward,
            # smooth
            add_smooth=True,
            smooth_monotonic=False,
            smooth_increasing=None,
            smooth_strict=False,
            warmup_frac_raw=warmup_frac_raw,
            warmup_frac_smooth=warmup_frac_smooth,
            warmup_clip_raw=warmup_clip_raw,
            warmup_clip_smooth=warmup_clip_smooth,
            ending_clip_raw=ending_clip_raw,
            ending_clip_smooth=ending_clip_smooth,
            s_factor=1,
            k_spline=5,
            rolling_window=200,
            min_se=1e-6,
            x_inv_weight_power=0.3,
            smooth_out_column="DeltaReward_smooth",
            use_linear=use_linear
        )
        if ax_delta_reward is None:
            plt.tight_layout()
            plt.savefig(output_dir / plot_eval_column / f"{plot_x_column}_delta_reward.pdf", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved {output_dir / plot_eval_column / plot_x_column}_delta_reward.pdf")
    if plot_delta_err_rate:
        ax = plot_curves_with_smooth(
            df,
            curve_column=plot_curve_column,
            x_column=plot_x_column,
            y_column="DeltaErrRate",
            y_std_column='DeltaErrRate_std',
            x_scale=plot_x_scale,
            y_scale=plot_y_scale,
            x_label=plot_x_label,
            y_label=plot_y_label,
            title=plot_title,
            use_scatter=plot_use_scatter,
            use_line=plot_use_line,
            smooth_use_scatter=plot_smooth_use_scatter,
            smooth_use_line=plot_smooth_use_line,
            use_legend=plot_use_legend,
            legend_lambda=plot_legend_lambda,
            legend_loc=plot_legend_loc,
            legend_bbox_to_anchor=plot_legend_bbox_to_anchor,
            legend_fontsize=plot_legend_fontsize,
            ax=ax_delta_err_rate,
            # smooth
            add_smooth=True,
            smooth_monotonic=False,
            smooth_increasing=None,
            smooth_strict=False,
            warmup_frac_raw=warmup_frac_raw,
            warmup_frac_smooth=warmup_frac_smooth,
            warmup_clip_raw=warmup_clip_raw,
            warmup_clip_smooth=warmup_clip_smooth,
            ending_clip_raw=ending_clip_raw,
            ending_clip_smooth=ending_clip_smooth,
            s_factor=1,
            k_spline=5,
            rolling_window=200,
            min_se=1e-6,
            x_inv_weight_power=0.3,
            smooth_out_column="DeltaErrRate_smooth",
            use_linear=use_linear
        )
        if ax_delta_err_rate is None:
            plt.tight_layout()
            plt.savefig(output_dir / plot_eval_column / f"{plot_x_column}_delta_err_rate.pdf", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved {output_dir / plot_eval_column / plot_x_column}_delta_err_rate.pdf")

def predict_and_plot(
    df,
    predict_func, # similar to predicter.predict_errrate_df
    predict_x_column_list,
    metric_column, # TODO: useless?
    plot_curve_column,
    plot_x_column,
    plot_curve_mask = None,
    plot_on_delta = False,
    plot_y_lambda = None,
    plot_delta_base_step = 0,
    plot_use_line = False,
    plot_use_scatter = False,
    plot_x_scale = None,
    plot_y_scale = None,
    # Highlight specific curves
    plot_highlight_curves: list = None,  # List of curve values to highlight
    plot_highlight_line_alpha: float = 1.0,  # Alpha for highlighted curves
    plot_highlight_line_width: float = 3.0,  # Linewidth for highlighted curves
    # Line and scatter styling
    line_alpha: float = 1.0,
    line_width: float = 2.0,
    scatter_alpha: float = 0.3,
    scatter_size: float = 8.0,
    scatter_marker: str = 'o',
    # Axis formatting control
    x_tick_format: str = None,  # 'auto', 'decimal', 'sci', or 'plain'
    y_tick_format: str = 'auto',  # 'auto', 'decimal', 'sci', or 'plain'
    # Tick control (added for completeness)
    x_tick_spacing: float = None,  # Spacing for x-axis ticks
    y_tick_spacing: float = None,  # Spacing for y-axis ticks
    # Grid control
    x_grid_spacing: float = None,  # Spacing for x-axis grid lines
    y_grid_spacing: float = None,  # Spacing for y-axis grid lines
    # Custom color mapping override
    custom_color_mapping: dict = None,  # Custom color mapping to override config colors
    warmup_frac_raw = 0,
    warmup_clip_raw: int = None,  # Absolute count clipping for prediction data
    ending_clip_raw: int = None,  # Absolute ending clipping for prediction data
    ax: plt.Axes=None,
):
    # predicter = fit.FitLogErrRate(L=0.06333, r=1.73e-10, N0_k=4.95e9, r_e0=1e-9, N0_e0=3e9)
    pred_column = metric_column + "_pred"
    df[pred_column] = predict_func(df, *predict_x_column_list)
    if plot_curve_mask is not None:
        df = df[df[plot_curve_column].isin(plot_curve_mask)]
    if plot_on_delta: # plot delta instead of raw pred
        # predict delta
        delta_pred_column = pred_column + "_delta"
        df[delta_pred_column] = data_proc.calc_delta_y(df, pred_column, base_step=plot_delta_base_step, curve_column=plot_curve_column)
        pred_column = delta_pred_column
    if plot_y_lambda is not None: # plot 1 - pred
        tmp_pred_column = pred_column + "_lambda"
        df[tmp_pred_column] = plot_y_lambda(df[pred_column])
        pred_column = tmp_pred_column
    # # eval_name = "val/test_score/openai/gsm8k" # must be one of config.TEST_EVALS.keys()
    # eval_name = "holdout_score"
    # x_column = "C" # key must be one of 'T', 'C', 'E'
    # metric = 'ErrRate' # key must be one of 'R', 'ErrRate', 'DeltaReward', 'DeltaErrRate'
    # curve_column = 'N' # key must be one of 'N', 'data_fator'

    # Apply warmup clipping for prediction data
    if warmup_clip_raw is not None:
        # Use absolute count clipping
        df_fit_plot = data_proc.apply_clip(df, curve_column=plot_curve_column, warmup_step=warmup_clip_raw)
    elif warmup_frac_raw > 0:
        # Use relative clipping
        df_fit_plot = data_proc.apply_clip(df, curve_column=plot_curve_column, warmup_frac=warmup_frac_raw)
    else:
        df_fit_plot = df
    
    # Apply ending clipping for prediction data
    if ending_clip_raw is not None:
        # Use absolute ending clipping
        df_fit_plot = data_proc.apply_clip(df_fit_plot, curve_column=plot_curve_column, ending_step=ending_clip_raw)
    ax = plot.plot_curves(
        df_fit_plot, 
        curve_column=plot_curve_column, x_column=plot_x_column, y_column=pred_column, 
        use_line=plot_use_line,
        use_scatter=plot_use_scatter,
        x_scale=plot_x_scale, y_scale=plot_y_scale, 
        # Pass highlight parameters
        highlight_curves=plot_highlight_curves,
        highlight_line_alpha=plot_highlight_line_alpha,
        highlight_line_width=plot_highlight_line_width,
        # Pass line and scatter styling
        line_alpha=line_alpha,
        line_width=line_width,
        scatter_alpha=scatter_alpha,
        scatter_s=scatter_size,
        scatter_marker=scatter_marker,
        # Pass axis formatting
        x_tick_format=x_tick_format,
        y_tick_format=y_tick_format,
        # Pass tick and grid spacing
        x_tick_spacing=x_tick_spacing,
        y_tick_spacing=y_tick_spacing,
        x_grid_spacing=x_grid_spacing,
        y_grid_spacing=y_grid_spacing,
        # Pass custom color mapping
        custom_color_mapping=custom_color_mapping,
        ax=ax,
    )
    return ax