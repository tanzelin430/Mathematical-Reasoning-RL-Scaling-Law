# -*- coding: utf-8 -*-
"""Visualization functions for RL scaling law analysis

This module provides functions for:
- Figure plotting (1a, 1b, 2a, 2b)
- Color mapping and formatting utilities
- Empirical frontier visualization
"""

import math
import data_proc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator  # 添加这行

__all__ = [
    # Figure plotting
    'plot_score_c_1a', 'plot_err_rate_1c', 'plot_ip_c_1b', 'plot_fit_score_c_2a', 'plot_fit_ip_2b',
    
    # Utility functions
    'human_format_N',
    
    # Empirical frontier visualization
    'vplot_empirical_f_of_R', 'plot_phi_over_steps',
    
    # Multi-subplot layout utilities
    'create_multi_subplot_axes', 'set_figure_labels', 'apply_tight_layout', 'apply_global_legend_layout'
]

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

COLOR_MAPPING = {
    # for model size
    0.5e9: '#1f77b4',
    1.5e9: '#ff7f0e',
    3e9: '#d62728',
    7e9: '#2ca02c',
    14e9: '#9467bd',
    # for data dup factor
    0: '#1f77b4',
    2: '#ff7f0e',
    4: '#d62728',
    5: '#2ca02c',
    10: '#9467bd',
    20: '#8c564b',
    25: '#7ce372', # need a new diff color
    50: '#e377c2',
    100: '#7f7f7f'
}

def human_format_N(N: float, mode: str = "eng", sig: int = 3, sci_coeff: bool = True) -> str:
    """
    Human-friendly N label:
      - mode="eng": 3B, 7.5B, 12M, 950K, 512
      - mode="sci": 3.2×10^9  (sci_coeff=True)  or  10^9 (sci_coeff=False)
    sig: significant digits for the leading coefficient (default 3 significant digits, won't transform 1.5B to 2B)
    """
    if N is None or not np.isfinite(N):
        return str(N)

    N = float(N)
    if N == 0:
        return "0"

    if mode == "eng":
        units = [("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3), ("", 1.0)]
        for u, f in units:
            if abs(N) >= f:
                val = N / f
                s = f"{val:.{sig}g}"  # Significant digit formatting, no excessive rounding
                return f"{s}{u}"
        return f"{N:.{sig}g}"

    elif mode == "sci":
        exp = int(math.floor(math.log10(abs(N))))
        if not sci_coeff:
            return f"10^{exp}"
        coeff = N / (10 ** exp)
        s = f"{coeff:.{sig}g}"
        return f"{s}×10^{exp}"

    else:
        return str(N)

# =============================================================================
# FIGURE PLOTTING
# =============================================================================

def _get_legends(_Ns: list[float]):
    sort_Ns = sorted(_Ns)
    handles = [Line2D([0], [0], color=COLOR_MAPPING[n], linewidth=2) for n in sort_Ns]
    labels = [f"N={human_format_N(n, mode='eng', sig=3, sci_coeff=True)}" for n in sort_Ns]
    return handles, labels, sort_Ns


def plot_basic(
    df,    # raw data to overlay as scatter points
    curve_column: str,
    x_column: str,
    y_column: str,  # column for raw scatter points
    x_scale: str = None,
    y_scale: str = None,
    x_label: str = None,
    y_label: str = None,
    title: str = None,
    use_scatter: bool = True,
    use_line: bool = True,
    scatter_alpha: float = 0.3,
    scatter_s: int = 8,
    line_alpha: float = 0.5,
    ax=None
):
    """Generic plotting function that can handle both score and error rate plots"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,4), dpi=140)
    
    # set legend
    unique_Ns = sorted(df[curve_column].unique())
    handles, labels, _ = _get_legends(unique_Ns)

    # Plot raw data as scatter points
    for g in data_proc.split_df(df, by_column=curve_column):
        curve_id = g[curve_column].iloc[0] if curve_column in g.columns else None
        x = g[x_column].to_numpy()
        y = g[y_column].to_numpy()
        # Scatter
        if use_scatter:
            ax.scatter(x, y, alpha=scatter_alpha, s=scatter_s, color=COLOR_MAPPING[curve_id], edgecolors='none')
        # Line
        if use_line:
            ax.plot(x, y, alpha=line_alpha, color=COLOR_MAPPING[curve_id])

    if x_scale:
        ax.set_xscale(x_scale)
        if x_scale == "log":
            # Add more tick marks for log scale - major ticks at 1, 2, 5 multiples
            ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            # Minor ticks for intermediate values
            ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
    if y_scale:
        ax.set_yscale(y_scale)
        if y_scale == "log":
            # Add more tick marks for log scale - major ticks at 1, 2, 5 multiples  
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            # Minor ticks for intermediate values
            ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    ax.legend(handles, labels, loc='best', fontsize=8)
    return ax


def plot_generic_curve(
    df,    # raw data to overlay as scatter points
    curve_column: str,
    x_column: str,
    y_column: str,  # column for raw scatter points
    df_smooth=None,  # smoothed data for curves, if None use df
    y_smooth_column: str = None,  # column for smooth curves, if None use y_column
    y_width_column: str = None, # e.g. std
    width_on_smooth: bool = False,
    x_scale: str = None,
    y_scale: str = None,
    x_label: str = None,
    y_label: str = None,
    title: str = None,
    ax=None
):
    """Generic plotting function that can handle both score and error rate plots"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,4), dpi=140)
    
    # set legend
    unique_Ns = sorted(df[curve_column].unique())
    handles, labels, _ = _get_legends(unique_Ns)

    # Plot raw data as scatter points
    for g in data_proc.split_df(df, by_column=curve_column):
        curve_id = g[curve_column].iloc[0] if curve_column in g.columns else None
        x = g[x_column].to_numpy()
        y = g[y_column].to_numpy()
        # Plot as scatter points with lighter color
        ax.scatter(x, y, alpha=0.3, s=8, color=COLOR_MAPPING[curve_id], edgecolors='none')
        # Plot width span
        if y_width_column and y_width_column in g.columns and not width_on_smooth:
            y_width = g[y_width_column].to_numpy()
            ax.fill_between(x, y - y_width, y + y_width, alpha=0.2, color=COLOR_MAPPING[curve_id])
    
    # Plot smooth curves
    if y_smooth_column is not None:
        if df_smooth is None:
            raise ValueError("df_smooth is required when y_smooth_column is not None")
        for g in data_proc.split_df(df_smooth, by_column=curve_column):
            curve_id = g[curve_column].iloc[0] if curve_column in g.columns else None
            x = g[x_column].to_numpy()
            y = g[y_smooth_column].to_numpy()
            # Line
            (ln,) = ax.plot(x, y, alpha=0.5, color=COLOR_MAPPING[curve_id])
            # Plot width span
            if y_width_column and y_width_column in g.columns and width_on_smooth:
                y_width = g[y_width_column].to_numpy()
                ax.fill_between(x, y - y_width, y + y_width, alpha=0.2, color=COLOR_MAPPING[curve_id])

    if x_scale:
        ax.set_xscale(x_scale)
        if x_scale == "log":
            # Add more tick marks for log scale - major ticks at 1, 2, 5 multiples
            ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            # Minor ticks for intermediate values
            ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
    if y_scale:
        ax.set_yscale(y_scale)
        if y_scale == "log":
            # Add more tick marks for log scale - major ticks at 1, 2, 5 multiples  
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            # Minor ticks for intermediate values
            ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    ax.legend(handles, labels, loc='best', fontsize=8)
    return ax


def vplot_empirical_f_of_R(
    df,
    I_of_R: pd.DataFrame,
    use_smooth: bool = True,
    label_by: str = "N",          # "N" | "runid" | None
    legend_max: int = 12,
    log_y: bool = True,
    alpha_runs: float = 0.35,
    ax=None
):
    """
    Empirical f(R) visualization:
      - For each run, plot R→C curve (i.e., y=C, x=R), showing "actual compute spent to achieve that R";
      - Overlay I_of_R envelope (R_level→I_minC), i.e., empirical f(R).

    Parameters
    ----------
    runs_dfs : List[pd.DataFrame]
        Each df must contain columns: ['runid','N','C','R']; if 'R_smooth' exists, can use use_smooth=True.
    I_of_R : pd.DataFrame
        Contains columns ['R_level', 'I_raw'], empirical frontier from suffix minimum.
    use_smooth : bool
        When True, prioritize 'R_smooth' as x-axis to avoid noise-caused reversals.
    label_by : str
        Legend labels by 'N' or 'runid'; None for no legend.
    legend_max : int
        Maximum number of runs to show in legend, avoid too long.
    log_y : bool
        Whether y-axis is logarithmic (usually log to see magnitude clearly).
    alpha_runs : float
        Opacity of each run curve.
    ax : matplotlib.axes.Axes or None
        Pass external axis; None creates new one.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """

    def _fmt_N(x):
        x = float(x)
        units = [("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3), ("", 1.0)]
        for u, f in units:
            if abs(x) >= f:
                return f"{x/f:.3g}{u}"
        return f"{x:.3g}"

    if ax is None:
        fig, ax = plt.subplots(figsize=(7,4.5), dpi=140)

    # 1) For each run: plot x=R, y=C
    handles, labels = [], []
    n_labeled = 0
    for g in data_proc.split_df(df, by_column='N'):
        # Check required columns: runid, N, C, and R or R_smooth
        required_cols = {'runid','N','C'}
        if not required_cols.issubset(g.columns) or (('R_smooth' not in g.columns) and ('R' not in g.columns)):
            raise ValueError("each run df must contain: runid, N, C, and R or R_smooth")

        r = g['R_smooth'] if (use_smooth and 'R_smooth' in g.columns) else g['R']
        x = r.to_numpy(dtype=float)
        y = g['C'].to_numpy(dtype=float)

        # Sort by R ascending to avoid reversals
        idx = np.argsort(x)
        x = x[idx]; y = y[idx]

        (ln,) = plt.plot(x, y, alpha=alpha_runs)
        # ax.scatter(x, y, s=2, marker="o", edgecolor="none")

        lbl = None
        if label_by == "N":
            lbl = f"N={_fmt_N(g['N'].iloc[0])}"
        elif label_by == "runid":
            lbl = str(g['runid'].iloc[0])

        if label_by and lbl and n_labeled < legend_max:
            handles.append(Line2D([0],[0], color=ln.get_color()))
            labels.append(lbl); n_labeled += 1

    # 2) Overlay empirical frontier (I_of_R): this is empirical f(R)
    R_env = I_of_R['R_level'].to_numpy(float)
    I_env = I_of_R['I_raw' ].to_numpy(float)
    # Ensure non-decreasing
    I_env = np.maximum.accumulate(np.maximum(I_env, 1e-300))
    # ax.plot(R_env, I_env, alpha=0.5, color='k', linewidth=2.5, label='envelope  I_of_R')
    ax.scatter(R_env, I_env, s=2, marker="o", edgecolor="k", color='k')

    # 3) Axes/title/legend
    ax.set_xlabel("Return R (critic_rewards_mean)")
    ax.set_ylabel("Compute / Intrinsic I (FLOPs)")
    if log_y:
        ax.set_yscale("log")
    ax.set_title("Empirical f(R): per-run C(R) with envelope I_of_R(R)")

    if label_by:
        # Also put envelope in legend (at the end)
        handles.append(Line2D([0],[0], color='k', linewidth=2.5))
        labels.append("I_of_R (envelope)")
        ax.legend(handles, labels, fontsize=8, ncol=2)

    return ax



def plot_phi_over_steps(
    _df,
    sample_size_per_step: float = 512.0,
    tail_fraction: float = 0.25,   # Use last fraction of samples to estimate steady state (e.g., last 1/4)
    logy: bool = True,
    label_by: str = "N",           # "N" | "runid" | None
    ax=None
):
    """
    Plot φ_i(step) = C / (N * (step * batch_size)) for each run;
    and return robust estimates for each run (using median/IQR of tail segment).

    Parameters
    ----------
    df : List[pd.DataFrame]
        List from make_run_curves(...) or monotone_smooth_points(...);
        each df must contain columns: ['runid','N','step','C_raw'].
    sample_size_per_step : float
        "Interaction sample size" per step (your 512).
    tail_fraction : float in (0,1]
        Take this fraction from the end of sequence for robust statistics (closer to "stable phase" φ).
    logy : bool
        Whether to use log scale for y-axis.
    label_by : str
        Legend labels: 'N' (model size) or 'runid' or None.
    ax : matplotlib.axes.Axes or None
        Pass existing subplot; None creates new figure.

    Returns
    -------
    ax : matplotlib.axes.Axes
    stats : pd.DataFrame
        Robust statistics for each run, columns:
        ['runid','N','phi_median','phi_q25','phi_q75','phi_iqr','n_tail']
    global_phi : float
        Global median of all runs' tails combined (can be used as κ estimate).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7,4), dpi=140)

    stats_rows = []
    all_tail_phi = []

    for g in data_proc.split_df(_df, by_column='N'):
        if not {'runid','N','step','C_raw'}.issubset(g.columns):
            raise ValueError("each run df must contain columns: runid, N, step, C")

        _df = g[['runid','N','step','C_raw']].dropna().copy()
        _df = _df.sort_values('step')

        N = float(_df['N'].iloc[0])
        runid = str(_df['runid'].iloc[0])

        E_est = _df['step'].astype(float) * float(sample_size_per_step)
        denom = np.maximum(N * E_est.to_numpy(float), 1e-30)
        phi = _df['C_raw'].to_numpy(float) / denom

        # Draw line
        lbl = None
        if label_by == "N":
            # Concise human-readable N
            def _fmt_N(x):
                units = [("T",1e12),("B",1e9),("M",1e6),("K",1e3),("",1.0)]
                for u,f in units:
                    if abs(x) >= f: return f"{x/f:.3g}{u}"
                return f"{x:.3g}"
            lbl = f"N={_fmt_N(N)}"
        elif label_by == "runid":
            lbl = runid

        ax.plot(_df['step'], phi, alpha=0.9, label=lbl)

        # Tail statistics
        m = len(phi)
        k0 = int(max(0, np.floor((1.0 - tail_fraction) * m)))
        tail = phi[k0:] if m > 0 else np.array([])
        tail = tail[np.isfinite(tail) & (tail > 0)]
        if len(tail) > 0:
            q25, med, q75 = np.quantile(tail, [0.25, 0.5, 0.75])
            iqr = q75 - q25
            all_tail_phi.append(tail)
            stats_rows.append({
                'runid': runid, 'N': N,
                'phi_median': float(med),
                'phi_q25': float(q25),
                'phi_q75': float(q75),
                'phi_iqr': float(iqr),
                'n_tail': int(len(tail))
            })

            # Visualize this run's tail median
            ax.hlines(med, _df['step'].min(), _df['step'].max(),
                      colors=ax.lines[-1].get_color(), linestyles='--', alpha=0.4)

    stats = pd.DataFrame(stats_rows).sort_values('N').reset_index(drop=True)
    global_phi = float(np.median(np.concatenate(all_tail_phi))) if all_tail_phi else np.nan

    ax.set_xlabel("step")
    ax.set_ylabel(r"$\phi = C / (N \cdot E)$  (with $E=step \times$ %.0f)" % sample_size_per_step)
    if logy:
        ax.set_yscale("log")
    ax.set_xscale("linear")  # Can also be changed to log as needed
    title = r"$\phi$ vs step (per run); tail-median global $\hat{\phi}=%.3g$" % global_phi
    ax.set_title(title)
    if label_by in ("N","runid"):
        ax.legend(fontsize=8, ncol=2)

    return ax, stats, global_phi

# ============================================================================
# Multi-Subplot Layout Utilities
# ============================================================================

def create_multi_subplot_axes(keys, total_evals, figure_columns, figure_size):
    """Create and return fig_axes dictionary with axes getter function"""
    fig_axes = {key: plt.subplots(
        (total_evals + figure_columns - 1) // figure_columns, figure_columns, 
        figsize=figure_size, constrained_layout=False
    ) for key in keys}
    
    def get_axes_for_eval(eval_index):
        if total_evals > figure_columns:
            row, col = eval_index // figure_columns, eval_index % figure_columns
            return {key: fig_axes[key][1][row, col] for key in keys}
        else:
            return {key: fig_axes[key][1] for key in keys}
    
    return fig_axes, get_axes_for_eval


def set_figure_labels(fig_axes, xlabel, y_labels):
    """Set supxlabel and supylabel for multi-subplot figures"""
    for key, ylabel in y_labels.items():
        if key in fig_axes:
            fig_axes[key][0].supxlabel(xlabel)
            fig_axes[key][0].supylabel(ylabel)


def apply_tight_layout(fig_axes, keep_legends=True):
    """Apply tight layout without global legend for multi-subplot figures"""
    # Optionally hide subplot legends to avoid clutter
    if not keep_legends:
        [legend.set_visible(False) for fig, _ in fig_axes.values() for ax in fig.get_axes() if (legend := ax.get_legend())]
    
    # Optimize layout to reduce left margin
    for fig, _ in fig_axes.values():
        for ax in fig.get_axes():
            # Use compact decimal format instead of scientific notation (0.7 instead of 7x10-1)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3g}'))
            ax.tick_params(axis='y', labelsize=8)  # Smaller y-axis labels
    
    # Auto-adjust layout with reduced left margin
    [fig.tight_layout() for fig, _ in fig_axes.values()]
    [fig.subplots_adjust(left=0.15) for fig, _ in fig_axes.values()]


def apply_global_legend_layout(fig_axes, unique_N, top=0.91):
    """Apply consistent global legend layout to multi-subplot figures"""
    # Hide all subplot legends
    [legend.set_visible(False) for fig, _ in fig_axes.values() for ax in fig.get_axes() if (legend := ax.get_legend())]
    # Auto-adjust layout first, then add global legend
    [fig.tight_layout() for fig, _ in fig_axes.values()]
    
    # create global legend
    handles, labels, _ = _get_legends(unique_N)
    [fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=len(labels)) for fig, _ in fig_axes.values()]
    
    # Adjust only the top to make room for legend
    [fig.subplots_adjust(top=top) for fig, _ in fig_axes.values()]
