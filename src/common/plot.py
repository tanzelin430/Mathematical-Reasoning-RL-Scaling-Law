# -*- coding: utf-8 -*-
"""Visualization functions for RL scaling law analysis

This module provides functions for:
- Figure plotting (1a, 1b, 2a, 2b)
- Color mapping and formatting utilities
- Empirical frontier visualization
"""

import math
from typing import Callable
import data_proc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator
import config

# 尝试导入seaborn，如果没有安装则使用matplotlib默认样式
try:
    import seaborn as sns
    # Set seaborn style for better looking plots with grid
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    SEABORN_AVAILABLE = True
except ImportError:
    # Fallback: 使用matplotlib实现类似seaborn的网格样式
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.color'] = '#b0b0b0'
    plt.rcParams['grid.linestyle'] = '-'
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['grid.alpha'] = 0.5
    plt.rcParams['axes.edgecolor'] = '#000000'
    plt.rcParams['axes.linewidth'] = 0.8
    SEABORN_AVAILABLE = False

# 全局字体加粗设置
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
# 注意: legend.fontweight 不是有效的rcParams参数，图例字体会继承font.weight设置

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

# 使用 config 中的颜色映射
COLOR_MAPPING = config.COLOR_MAPPING
get_color_for_curve = config.get_color_for_curve

def setup_axis_formatter(ax, axis, format_type='auto'):
    """
    Setup axis formatter with simple options.
    
    Args:
        ax: matplotlib axes object
        axis: 'x' or 'y'
        format_type: str, one of:
            - 'auto': Smart default (decimal for 0.01-100, individual 10^n for others) 
            - 'decimal': Always use decimal format
            - 'sci': Always use scientific notation  
            - 'plain': Matplotlib default
    """
    from matplotlib.ticker import ScalarFormatter, FuncFormatter
    import math

    def decimal_formatter(x, pos):
        if x == 0:
            return '0'
        
        import math
        abs_x = abs(x)
        
        # Calculate the number of significant figures needed (max 4)
        # Find the order of magnitude
        order_of_magnitude = math.floor(math.log10(abs_x))
        
        # Calculate decimal places needed for up to 4 significant figures
        max_sig_figs = 4
        if abs_x >= 1:
            # For numbers >= 1, decimal places = max_sig_figs - integer_digits
            integer_digits = len(str(int(abs_x)))
            decimal_places = max(0, max_sig_figs - integer_digits)
        else:
            # For numbers < 1, decimal places = -order_of_magnitude + max_sig_figs - 1
            decimal_places = -order_of_magnitude + max_sig_figs - 1
        
        # Cap decimal places to avoid excessive precision
        decimal_places = min(decimal_places, 6)
        
        # Format the number
        formatted = f'{x:.{decimal_places}f}'
        
        # Remove trailing zeros and decimal point if not needed
        if '.' in formatted:
            formatted = formatted.rstrip('0').rstrip('.')
        
        return formatted

    def sci_formatter(x, pos):
            if x == 0:
                return '0'
            abs_x = abs(x)
            exp = int(math.log10(abs_x))
            coeff = x / (10 ** exp)
            if abs(coeff - 1.0) < 0.01:
                return f'$10^{{{exp}}}$'
            else:
                return f'${decimal_formatter(coeff, pos)} \\times 10^{{{exp}}}$'
                # return f'${coeff:.1f} \\times 10^{{{exp}}}$'
            
    if format_type == 'plain':
        return
        
    if format_type is None or format_type == 'auto':
        def smart_formatter(x, pos):
            if x == 0:
                return '0'
            abs_x = abs(x)
            if 1e-2 <= abs_x <= 1e5:
                return decimal_formatter(x, pos)
            else:
                return sci_formatter(x, pos)
        _formatter = FuncFormatter(smart_formatter)
    elif format_type == 'decimal':
        _formatter = FuncFormatter(decimal_formatter)
    elif format_type == 'sci':
        _formatter = FuncFormatter(sci_formatter)
    if axis == 'y':
        ax.yaxis.set_major_formatter(_formatter)
        # ax.yaxis.set_minor_formatter(_formatter)
    else:
        ax.xaxis.set_major_formatter(_formatter)
        # ax.xaxis.set_minor_formatter(_formatter)

def setup_y_axis_formatter(ax, format_type='auto'):
    """Backward compatibility wrapper"""
    setup_axis_formatter(ax, 'y', format_type)

def legend_format(label_name: str, label_value: float) -> str:
    if label_name == "N":
        return human_format_N(label_value)
    elif label_name == "Tau":
        return f"τ={label_value}"
    elif label_name == "rollout_n":
        return f"ρ={label_value.split('rho')[1]}"
    else:
        return f"{label_name}={label_value}"
        
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
        # Always use B (Billion) as the unit
        val = N / 1e9
        s = f"{val:.{sig}g}"  # Significant digit formatting, no excessive rounding
        return f"{s}B"

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

# lambda x: f"N={human_format_N(x, mode='eng', sig=3, sci_coeff=True)}"
def _get_legends(_curves: list[float], legend_lambda = None):
    if legend_lambda is None:
        legend_lambda = lambda x: x
    def _flat(x):
        if isinstance(x, str):
            if x.startswith('rho'):
                # Handle rollout numbers like 'rho16' -> 16
                x = x.replace('rho', '')
                try:
                    x = int(x)  # Convert to integer directly
                except ValueError:
                    x = 1000000  # Fallback for invalid rollout numbers
        return float(x)
    sort_curves = sorted(_curves, key=_flat)
    handles = [Line2D([0], [0], color=get_color_for_curve(n), linewidth=2) for n in sort_curves]
    labels = [legend_lambda(n) for n in sort_curves]
    return handles, labels, sort_curves


def prepare_legend(df, curve_column: str, legend_lambda=None):
    """
    Prepare legend handles and labels for plotting.
    
    Args:
        df: DataFrame containing the curve data
        curve_column: Column name to group curves by
        legend_lambda: Optional function to format legend labels. If None, uses legend_format(curve_column, x)
    
    Returns:
        (handles, labels): Tuple of legend handles and labels for plot_basic_settings
    """
    unique_curves = df[curve_column].unique()
    if legend_lambda is None:
        legend_lambda = lambda x: legend_format(curve_column, x)
    handles, labels, _ = _get_legends(unique_curves, legend_lambda)
    return handles, labels


def plot_basic(
    x: np.ndarray,
    y: np.ndarray,
    # Scatter parameters
    use_scatter: bool = False,
    scatter_alpha: float = 0.3,
    scatter_s: int = 8,
    scatter_marker: str = 'o',
    # Line parameters
    use_line: bool = False,
    line_alpha: float = 0.5,
    line_width: float = None,
    line_style: str = '-',
    # Fill parameters
    fill_width: np.ndarray = None,
    fill_width_alpha: float = 0.2,
    # Common parameters
    color: str = 'k',
    ax: plt.Axes = None
):
    if not use_scatter and not use_line and fill_width is None:
        raise ValueError("At least one of use_scatter, use_line, or y_fill_width must be True or not None")
    """Generic plotting function that can handle both score and error rate plots"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,4), dpi=300)
    
    # Scatter
    if use_scatter:
        ax.scatter(x, y, alpha=scatter_alpha, s=scatter_s, marker=scatter_marker, color=color, edgecolors='none')
    # Line
    if use_line:
        plot_kwargs = {'alpha': line_alpha, 'color': color, 'linestyle': line_style}
        if line_width is not None:
            plot_kwargs['linewidth'] = line_width
        ax.plot(x, y, **plot_kwargs)
    if fill_width is not None:
        ax.fill_between(x, y - fill_width, y + fill_width, alpha=fill_width_alpha, color=color)
    
    return ax

def plot_basic_settings(
    ax: plt.Axes, 
    # x_scale=None, y_scale=None, x_label=None, y_label=None, title=None, use_legend: bool = False,
    # legend_handles_labels: tuple = None, legend_loc='best', legend_bbox_to_anchor=None, legend_fontsize=8
    x_scale: str = None,
    y_scale: str = None,
    x_label: str = None,
    y_label: str = None,
    title: str = None,
    use_legend: bool = False,
    legend_handles_labels: tuple = None,
    legend_loc: str = 'best',
    legend_bbox_to_anchor: tuple = None,
    legend_fontsize: int = 8,
    # Auto-scaling margins
    x_margin: float = None,
    y_margin: float = None,
    # Tick control
    x_tick_spacing: float = None,  # Spacing for x-axis ticks
    y_tick_spacing: float = None,  # Spacing for y-axis ticks
    # Grid control (simple spacing-based)
    x_grid_spacing: float = None,  # Spacing for x-axis grid lines
    y_grid_spacing: float = None,  # Spacing for y-axis grid lines
    # Axis formatting control
    x_tick_format: str = None,  # 'auto', 'decimal', 'sci', or 'plain'
    y_tick_format: str = 'auto',  # 'auto', 'decimal', 'sci', or 'plain'
    # Data-based tick positioning
    x_tick_on_data: bool = False,  # Use actual data points for x-axis ticks
    y_tick_on_data: bool = False,  # Use actual data points for y-axis ticks
    # Custom tick positioning
    x_tick_subs: list = None,       # Custom x-axis tick positions (direct positions, works in both linear and log scale)
    y_tick_subs: list = None,       # Custom y-axis tick positions (direct positions, works in both linear and log scale)
    x_tick_subs_log: list = None,   # Log-scale specific multipliers (e.g., [1,2,5] for 1x,2x,5x per decade, only for log scale)
    y_tick_subs_log: list = None,   # Log-scale specific multipliers (e.g., [1,2,5] for 1x,2x,5x per decade, only for log scale)
    # Save configuration - NEW ADDITION
    save_to_dir: str = None,
    save_to_filename: str = None,
    save_to_filename_prefix: str = None,
    # Additional save configuration for process_single_eval compatibility
    plot_eval_column: str = None,
    plot_curve: str = None, 
    plot_x_column: str = None,
    plot_metric: str = None,
):
    # Import required ticker classes
    from matplotlib.ticker import MultipleLocator, LogLocator, FixedLocator
    import numpy as np
    
    def _extract_data_points_from_ax(ax):
        """Extract x and y data points from all plotted elements in the axes"""
        all_x_data = []
        all_y_data = []
        
        # Extract from line plots
        for line in ax.get_lines():
            x_data = line.get_xdata()
            y_data = line.get_ydata()
            if len(x_data) > 0 and len(y_data) > 0:
                all_x_data.extend(x_data)
                all_y_data.extend(y_data)
        
        # Extract from scatter plots
        for collection in ax.collections:
            offsets = collection.get_offsets()
            if len(offsets) > 0:
                all_x_data.extend(offsets[:, 0])
                all_y_data.extend(offsets[:, 1])
        
        return np.array(all_x_data), np.array(all_y_data)
    
    def _get_ticks_from_data(data_values, max_ticks=10, scale='linear'):
        """Get tick positions from data values"""
        if len(data_values) == 0:
            return []
        
        # Remove duplicates and sort
        unique_values = sorted(set(data_values))
        
        # If too many points, select a subset
        if len(unique_values) > max_ticks:
            if scale == 'log':
                # For log scale, try to keep important powers of 10
                log_values = np.log10(unique_values)
                indices = np.linspace(0, len(unique_values)-1, max_ticks, dtype=int)
                unique_values = [unique_values[i] for i in indices]
            else:
                # For linear scale, use equal spacing
                indices = np.linspace(0, len(unique_values)-1, max_ticks, dtype=int)
                unique_values = [unique_values[i] for i in indices]
        
        return unique_values
    
    def _apply_custom_ticks(axis, tick_subs, tick_subs_log, scale='linear'):
        """Apply custom tick positions based on scale type and parameters"""
        # Validate input: only one type of tick_subs should be provided
        if tick_subs is not None and tick_subs_log is not None:
            raise ValueError("Cannot specify both tick_subs and tick_subs_log simultaneously")
        
        # Handle tick_subs_log parameter
        if tick_subs_log is not None:
            if scale != 'log':
                raise ValueError("tick_subs_log can only be used with log scale")
            if len(tick_subs_log) == 0:
                return
            
            # tick_subs_log are multipliers within each decade (e.g., [1, 2, 5])
            if not all(0 < sub <= 10 for sub in tick_subs_log):
                raise ValueError("tick_subs_log values must be between 0 and 10")
            
            # Convert to subs format for LogLocator (0 < sub <= 1)
            subs = []
            for sub in tick_subs_log:
                if sub == 10:
                    subs.append(1.0)
                else:
                    subs.append(sub / 10.0)
            
            axis.set_major_locator(LogLocator(base=10, subs=sorted(set(subs))))
            return
        
        # Handle regular tick_subs parameter (direct positions)
        if tick_subs is not None:
            if len(tick_subs) == 0:
                return
            # tick_subs are always direct tick positions, regardless of scale
            axis.set_major_locator(FixedLocator(tick_subs))
    
    if x_scale:
        ax.set_xscale(x_scale)
    if y_scale:
        ax.set_yscale(y_scale)
    
    
    if x_grid_spacing is not None:
        if x_scale == 'log':
            # For log scale, interpret spacing as fraction of decade
            # 0.1 means 10 ticks per decade (0.1, 0.2, ..., 1.0)
            if x_grid_spacing > 1:
                raise ValueError("x_grid_spacing must be less than 1 for log scale")
            subs = np.arange(x_grid_spacing, 1.0 + x_grid_spacing, x_grid_spacing).tolist()
            ax.xaxis.set_minor_locator(LogLocator(base=10, subs=subs))
            ax.grid(True, which='minor', axis='x', alpha=0.5)
        # Use MultipleLocator for grid spacing - works for both linear and log scales
        else:
            ax.xaxis.set_minor_locator(MultipleLocator(x_grid_spacing))
            ax.grid(True, which='minor', axis='x', alpha=0.5)
    if y_grid_spacing is not None:
        if y_scale == 'log':
            # For log scale, interpret spacing as fraction of decade
            # 0.1 means 10 ticks per decade (0.1, 0.2, ..., 1.0)
            if y_grid_spacing > 1:
                raise ValueError("y_grid_spacing must be less than 1 for log scale")
            subs = np.arange(y_grid_spacing, 1.0 + y_grid_spacing, y_grid_spacing).tolist()
            ax.yaxis.set_minor_locator(LogLocator(base=10, subs=subs))
            ax.grid(True, which='minor', axis='y', alpha=0.5)
        else:
            ax.yaxis.set_minor_locator(MultipleLocator(y_grid_spacing))
            ax.grid(True, which='minor', axis='y', alpha=0.5)
            
    # hide minor tick labels
    ax.tick_params(axis='x', which='minor', left=False, right=False,
       labelleft=False, labelright=False)
    # hide minor tick labels
    ax.tick_params(axis='y', which='minor', left=False, right=False,
       labelleft=False, labelright=False)

    if x_label:
        ax.set_xlabel(x_label, fontweight='bold')
    if y_label:
        ax.set_ylabel(y_label, fontweight='bold')
    if title:
        ax.set_title(title, fontweight='bold')
    if use_legend:
        _handles, _labels = ax.get_legend_handles_labels()
        handles, labels = legend_handles_labels if legend_handles_labels else ([], [])
        
        # Only set legend if we have handles to add, or if there are already handles in the plot
        total_handles = _handles + handles
        total_labels = _labels + labels
        
        if len(total_handles) > 0:
            ax.legend(total_handles, total_labels, loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor, fontsize=legend_fontsize)
    
    # Set margins for auto-scaling
    if x_margin is not None or y_margin is not None:
        ax.margins(x=x_margin if x_margin is not None else 0.05, 
                  y=y_margin if y_margin is not None else 0.05)
    
    # Apply tick positioning first, then formatting
    # This is important because setting a locator can reset the formatter
    
    # Determine tick positioning strategy for each axis
    # Priority: tick_on_data > (tick_subs or tick_subs_log) > tick_spacing
    
    # Handle X-axis ticks
    if x_tick_on_data:
        # Extract data points and use them for ticks
        all_x_data, _ = _extract_data_points_from_ax(ax)
        if len(all_x_data) > 0:
            x_ticks = _get_ticks_from_data(all_x_data, max_ticks=10, scale=x_scale)
            ax.xaxis.set_major_locator(FixedLocator(x_ticks))
    elif x_tick_subs is not None or x_tick_subs_log is not None:
        # Use custom tick positions
        _apply_custom_ticks(ax.xaxis, x_tick_subs, x_tick_subs_log, x_scale)
    elif x_tick_spacing is not None:
        # Use spacing-based ticks (existing logic)
        if x_scale == 'log':
            if x_tick_spacing > 1:
                raise ValueError("x_tick_spacing must be less than 1 for log scale")
            else:
                subs = np.arange(x_tick_spacing, 1.0 + x_tick_spacing, x_tick_spacing).tolist()
                ax.xaxis.set_major_locator(LogLocator(base=10, subs=subs))
        else:
            ax.xaxis.set_major_locator(MultipleLocator(x_tick_spacing))
    
    # Handle Y-axis ticks
    if y_tick_on_data:
        # Extract data points and use them for ticks
        _, all_y_data = _extract_data_points_from_ax(ax)
        if len(all_y_data) > 0:
            y_ticks = _get_ticks_from_data(all_y_data, max_ticks=10, scale=y_scale)
            ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    elif y_tick_subs is not None or y_tick_subs_log is not None:
        # Use custom tick positions
        _apply_custom_ticks(ax.yaxis, y_tick_subs, y_tick_subs_log, y_scale)
    elif y_tick_spacing is not None:
        # Use spacing-based ticks (existing logic)
        if y_scale == 'log':
            if y_tick_spacing > 1:
                raise ValueError("y_tick_spacing must be less than 1 for log scale")
            else:
                subs = np.arange(y_tick_spacing, 1.0 + y_tick_spacing, y_tick_spacing).tolist()
                ax.yaxis.set_major_locator(LogLocator(base=10, subs=subs))
        else:
            ax.yaxis.set_major_locator(MultipleLocator(y_tick_spacing))
    
    # Apply axis formatting AFTER setting tick spacing
    # This ensures formatting is applied to the custom tick positions
    setup_axis_formatter(ax, 'x', x_tick_format)
    setup_axis_formatter(ax, 'y', y_tick_format)
    
    # Save configuration - MOVED FROM process_single_eval
    if save_to_dir is not None or save_to_filename is not None:
        import matplotlib.pyplot as plt
        from pathlib import Path
        import config
        
        plt.tight_layout()
        
        if plot_eval_column and plot_curve and plot_x_column and plot_metric:
            # Create eval-specific output directory (same logic as process_single_eval)
            eval_file_str = config.TEST_EVALS[plot_eval_column]['file_str']
            Path(save_to_dir).mkdir(parents=True, exist_ok=True)
            filename = f"{eval_file_str}_{plot_curve}_{plot_x_column}_{plot_metric}"
            if save_to_filename_prefix is not None:
                filename = save_to_filename_prefix + filename
            # if curve_mask is not None:
            #     # Convert mask values to clean format for filename
            #     mask_str = "_".join([str(int(float(val))) if isinstance(val, (int, float)) or hasattr(val, 'item') 
            #                         else str(val) for val in curve_mask])
            #     filename += f"_{mask_str}"
            filename += ".pdf"
            if save_to_filename is not None:
                filename = save_to_filename
            save_to_path = Path(save_to_dir) / filename
        else:
            # Simple filename mode
            if save_to_filename is not None:
                save_to_path = Path(save_to_dir) / save_to_filename
            else:
                save_to_path = Path(save_to_dir) / "plot.pdf"
        
        plt.savefig(save_to_path, dpi=300, bbox_inches='tight')
        print(f"Saved {save_to_path}")
        # Note: We don't call plt.close() here since the caller might still need the ax
    
    return ax

def plot_curves(
    df,
    curve_column: str,
    x_column: str,
    y_column: str,
    y_std_column: str = None,
    use_scatter: bool = False,
    use_line: bool = False,
    # Highlight specific curves
    highlight_curves: list = None,  # List of curve values to highlight
    highlight_alpha: float = 1.0,  # Alpha for highlighted curves
    highlight_width: float = None,  # Width for highlighted curves (line width and scatter size)
    # Line and scatter styling
    line_alpha: float = 1.0,
    line_width: float = 2.0,
    scatter_alpha: float = 0.3,
    scatter_size: float = 8.0,
    scatter_marker: str = 'o',
    # Custom color mapping override
    custom_color_mapping: dict = None,
    ax=None
):
    """
    Generic plotting function for curves grouped by curve_column.
    Only handles data plotting; settings and legend should be handled by caller.
    """
    df = df.sort_values(x_column)
    # Plot curves by group
    for g in data_proc.split_df(df, by_column=curve_column):
        curve_id = g[curve_column].iloc[0] if curve_column in g.columns else None
        g = g.sort_values(x_column)
        x = g[x_column].to_numpy()
        y = g[y_column].to_numpy()
        y_width = g[y_std_column].to_numpy() if y_std_column and y_std_column in g.columns else None
        
        # Determine if this curve should be highlighted
        is_highlighted = highlight_curves is not None and curve_id in highlight_curves
        final_line_alpha = highlight_alpha if is_highlighted else line_alpha
        final_line_width = highlight_width if is_highlighted and highlight_width is not None else line_width
        final_scatter_size = highlight_width if is_highlighted and highlight_width is not None else scatter_size
        
        # Use custom color mapping if provided, otherwise use config color mapping
        if custom_color_mapping is not None and curve_id in custom_color_mapping:
            color = custom_color_mapping[curve_id]
        else:
            color = get_color_for_curve(curve_id)
            
        ax = plot_basic(
            x, y, 
            use_scatter=use_scatter, 
            scatter_alpha=scatter_alpha, 
            scatter_s=final_scatter_size,
            scatter_marker=scatter_marker,
            use_line=use_line, 
            line_alpha=final_line_alpha, 
            line_width=final_line_width,
            fill_width=y_width, 
            fill_width_alpha=0.2, 
            color=color, 
            ax=ax
        )
    
    return ax

def plot_empirical_f_of_R(
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
        fig, ax = plt.subplots(figsize=(7,4.5), dpi=300)

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
        fig, ax = plt.subplots(figsize=(7,4), dpi=300)

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
            fig_axes[key][0].supxlabel(xlabel, fontweight='bold')
            fig_axes[key][0].supylabel(ylabel, fontweight='bold')


def apply_tight_layout(fig_axes, keep_legends=True, y_tick_format='auto'):
    """Apply tight layout without global legend for multi-subplot figures"""
    # Optionally hide subplot legends to avoid clutter
    if not keep_legends:
        [legend.set_visible(False) for fig, _ in fig_axes.values() for ax in fig.get_axes() if (legend := ax.get_legend())]
    
    # Optimize layout to reduce left margin
    for fig, _ in fig_axes.values():
        for ax in fig.get_axes():
            # Apply Y-axis formatting
            setup_y_axis_formatter(ax, y_tick_format)
            ax.tick_params(axis='y', labelsize=8)  # Smaller y-axis labels
    
    # Auto-adjust layout with reduced left margin
    [fig.tight_layout() for fig, _ in fig_axes.values()]
    [fig.subplots_adjust(left=0.15) for fig, _ in fig_axes.values()]


def apply_global_legend_layout(fig_axes, unique_N, top=0.91, legend_lambda=None):
    """Apply consistent global legend layout to multi-subplot figures"""
    # Hide all subplot legends
    [legend.set_visible(False) for fig, _ in fig_axes.values() for ax in fig.get_axes() if (legend := ax.get_legend())]
    # Auto-adjust layout first, then add global legend
    [fig.tight_layout() for fig, _ in fig_axes.values()]
    
    # create global legend
    handles, labels, _ = _get_legends(unique_N, legend_lambda)
    [fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=len(labels)) for fig, _ in fig_axes.values()]
    
    # Adjust only the top to make room for legend
    [fig.subplots_adjust(top=top) for fig, _ in fig_axes.values()]
