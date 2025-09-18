#!/usr/bin/env python3


import os
import sys
import matplotlib.pyplot as plt

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import data_proc
import plot
import config
from pathlib import Path
import fit

def plot_generic_curve_allinone(
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
    ax=None,
    # smooth params
    add_smooth: bool = False, 
    smooth_monotonic: bool = False,
    smooth_increasing: bool = True,
    smooth_strict: bool = False,
    warmup_frac_raw: float = 0,
    warmup_frac_smooth: float = 0,
    s_factor: float = 1,
    k_spline: int = 5,
    rolling_window: int = 200,
    min_se: float = 1e-6,
    x_inv_weight_power: float = 0.3,
    smooth_out_column: str = None,
    # fit params
    fit_model: fit.FitLogErrRate = None,
    fit_out_column: str = None,
):

    if warmup_frac_raw > 0:
        df = data_proc.apply_warmup_clipping(df, curve_column=curve_column, warmup_frac=warmup_frac_raw)
    
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
            x_inv_weight_power=x_inv_weight_power
        )
        if warmup_frac_smooth > 0:
            df_R_smooth = data_proc.apply_warmup_clipping(df_R_smooth, curve_column=curve_column, warmup_frac=warmup_frac_smooth)

    ax = plot.plot_generic_curve(
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
        ax=ax
    )

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
    ax: plt.Axes=None,
    save_to_dir: str = None,
    # smooth
    delta_base_step: int = 0,
    add_smooth=False,
    add_std=False,
    smooth_monotonic=False,
    smooth_increasing=None,
    smooth_strict=False,
    warmup_frac_raw: float=0,
    warmup_frac_smooth: float=0,
    calc_delta: bool=True, # in case there's no step 0 but don't care of it.
    s_factor=1,
    k_spline=5,
    rolling_window=200,
    min_se=1e-6,
    x_inv_weight_power=0.3,
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
    df = data_proc.merge_duplicate_steps(df, group_columns=[plot_curve_column, 'step'], mode='mean')

    # Add std cols back to df
    df = df.merge(R_std, on=[plot_curve_column, 'step'])
    df = df.merge(ErrRate_std, on=[plot_curve_column, 'step'])
    if calc_delta:
        df = df.merge(DeltaReward_std, on=[plot_curve_column, 'step'])
        df = df.merge(DeltaErrRate_std, on=[plot_curve_column, 'step'])
    if plot_curve_mask is not None:
        print("unique values of plot_curve_column:", df[plot_curve_column].unique())
        print("plot_curve_mask:", plot_curve_mask)
        df = df[df[plot_curve_column].isin(plot_curve_mask)]
    df.sort_values(plot_x_column, inplace=True)
    
    ax = plot_generic_curve_allinone(
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
        ax=ax,
        # smooth
        add_smooth=add_smooth,
        smooth_monotonic=smooth_monotonic,
        smooth_increasing=smooth_increasing,
        smooth_strict=smooth_strict,
        warmup_frac_raw=warmup_frac_raw,
        warmup_frac_smooth=warmup_frac_smooth,
        s_factor=s_factor,
        k_spline=k_spline,
        rolling_window=rolling_window,
        min_se=min_se,
        x_inv_weight_power=x_inv_weight_power,
        smooth_out_column=plot_metric+"_smooth" if add_smooth else None
    )

    if save_to_dir is not None:
        plt.tight_layout()
        # Create eval-specific output directory
        eval_file_str = config.TEST_EVALS[plot_eval_column]['file_str']
        Path(save_to_dir).mkdir(parents=True, exist_ok=True)
        filename = f"{eval_file_str}_{plot_curve_column}_{plot_x_column}_{plot_metric}"
        if plot_curve_mask is not None:
            filename += f"_{str(plot_curve_mask)}"
        filename += ".pdf"
        save_to_path = save_to_dir / filename
        plt.savefig(save_to_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {save_to_path}")
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
    plot_reward: bool=False, plot_err_rate: bool=False, plot_delta_reward: bool=False, plot_delta_err_rate: bool=False, 
    ax_reward: plt.Axes=None, ax_err_rate: plt.Axes=None, ax_delta_reward: plt.Axes=None, ax_delta_err_rate: plt.Axes=None, 
    delta_base_step: int = 0,
    warmup_frac_raw: float = 0,
    warmup_frac_smooth: float = 0,
    output_dir: str = None,
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
    df = data_proc.merge_duplicate_steps(df, group_columns=[plot_curve_column, 'step'], mode='mean')

    # Add std cols back to df
    df = df.merge(R_std, on=[plot_curve_column, 'step'])
    df = df.merge(ErrRate_std, on=[plot_curve_column, 'step'])
    df = df.merge(DeltaReward_std, on=[plot_curve_column, 'step'])
    df = df.merge(DeltaErrRate_std, on=[plot_curve_column, 'step'])
    df.sort_values(plot_x_column, inplace=True)
    
    if plot_reward:
        ax = plot_generic_curve_allinone(
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
            ax=ax_reward,
            # smooth
            add_smooth=True,
            smooth_monotonic=False,
            smooth_increasing=None,
            smooth_strict=False,
            warmup_frac_raw=warmup_frac_raw,
            warmup_frac_smooth=warmup_frac_smooth,
            s_factor=1,
            k_spline=5,
            rolling_window=200,
            min_se=1e-6,
            x_inv_weight_power=0.3,
            smooth_out_column="R_smooth"
        )
        if ax_reward is None:
            plt.tight_layout()
            plt.savefig(output_dir / plot_eval_column / f"{plot_x_column}_reward.pdf", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved {output_dir / plot_eval_column / plot_x_column}_reward.pdf")
    if plot_err_rate:
        ax = plot_generic_curve_allinone(
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
            ax=ax_err_rate,
            # smooth
            add_smooth=True,
            smooth_monotonic=False,
            smooth_increasing=None,
            smooth_strict=False,
            warmup_frac_raw=warmup_frac_raw,
            warmup_frac_smooth=warmup_frac_smooth,
            s_factor=1,
            k_spline=5,
            rolling_window=200,
            min_se=1e-6,
            x_inv_weight_power=0.3,
            smooth_out_column="ErrRate_smooth"
        )
        if ax_err_rate is None:
            plt.tight_layout()
            plt.savefig(output_dir / plot_eval_column / f"{plot_x_column}_err_rate.pdf", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved {output_dir / plot_eval_column / plot_x_column}_err_rate.pdf")
    if plot_delta_reward:
        ax = plot_generic_curve_allinone(
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
            ax=ax_delta_reward,
            # smooth
            add_smooth=True,
            smooth_monotonic=False,
            smooth_increasing=None,
            smooth_strict=False,
            warmup_frac_raw=warmup_frac_raw,
            warmup_frac_smooth=warmup_frac_smooth,
            s_factor=1,
            k_spline=5,
            rolling_window=200,
            min_se=1e-6,
            x_inv_weight_power=0.3,
            smooth_out_column="DeltaReward_smooth"
        )
        if ax_delta_reward is None:
            plt.tight_layout()
            plt.savefig(output_dir / plot_eval_column / f"{plot_x_column}_delta_reward.pdf", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved {output_dir / plot_eval_column / plot_x_column}_delta_reward.pdf")
    if plot_delta_err_rate:
        ax = plot_generic_curve_allinone(
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
            ax=ax_delta_err_rate,
            # smooth
            add_smooth=True,
            smooth_monotonic=False,
            smooth_increasing=None,
            smooth_strict=False,
            warmup_frac_raw=warmup_frac_raw,
            warmup_frac_smooth=warmup_frac_smooth,
            s_factor=1,
            k_spline=5,
            rolling_window=200,
            min_se=1e-6,
            x_inv_weight_power=0.3,
            smooth_out_column="DeltaErrRate_smooth"
        )
        if ax_delta_err_rate is None:
            plt.tight_layout()
            plt.savefig(output_dir / plot_eval_column / f"{plot_x_column}_delta_err_rate.pdf", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved {output_dir / plot_eval_column / plot_x_column}_delta_err_rate.pdf")