# useless
import matplotlib.pyplot as plt
import data_proc
from plot import _get_legends
import config

def plot_ip_c_1b(
    intrinsic_points,
    y_column: str,
    y_smooth_column: str=None,
    xlabel: str=None,
    ylabel: str=None,
    title: str=None,
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,4), dpi=300)
    
    # set legend
    unique_Ns = sorted(intrinsic_points['N'].unique())
    handles, labels, _ = _get_legends(unique_Ns)

    # Group by runid and plot each run
    for g in data_proc.split_df(intrinsic_points, by_column='N'):
    # TODO for g in intrinsic_points.groupby('N'):
        current_N = g['N'].iloc[0]
        g = g.sort_values("C")

        # Draw line using pre-computed I_map values
        if y_smooth_column is not None:
            (ln,) = ax.plot(g["C"], g[y_smooth_column], alpha=0.5, color=config.COLOR_MAPPING[current_N])
        # test
        ax.scatter(g["C"], g[y_column], alpha=0.5, s=2, marker="o", edgecolor="none", color=config.COLOR_MAPPING[current_N])
        
    # Reference line y=x
    xs = np.geomspace(max(1e-12, intrinsic_points["C"].min()),
                      intrinsic_points["C"].max(), 200)
    ax.plot(xs, xs, linestyle="--", alpha=0.1, label="y = x")
    # Add reference line to legend
    handles.append(Line2D([0], [0], color='black', linestyle='--', linewidth=1))
    labels.append("y = x")
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(handles, labels, loc="best", fontsize=8)
    return ax

def plot_fit_score_c_2a(
    df,
    pred_return_curves,
    xlabel: str=None,
    ylabel: str=None,
    title: str=None,
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,4), dpi=300)

    dfs = data_proc.split_df(df, by_column='N')
    # set legend
    unique_Ns = sorted(set(g['N'].iloc[0] for g in dfs if 'N' in g.columns))
    handles, labels, _ = _get_legends(unique_Ns)

    # # Background: gray original/monotonized curves (not in legend)
    # for g in dfs:
    #     x = g["C"].to_numpy()
    #     y = (g["R_smooth"] if "R_smooth" in g.columns else g["R"]).to_numpy()
    #     ax.plot(x, y, alpha=0.25, color="gray")
    
    # Plot raw data as scatter points
    for g in dfs:
        current_N = g['N'].iloc[0] if 'N' in g.columns else None
        x = g["C"].to_numpy()
        y = g["R"].to_numpy()  # Use raw R values, not monotone
        # Plot as scatter points with lighter color
        ax.scatter(x, y, alpha=0.3, s=8, color=config.COLOR_MAPPING[current_N], edgecolors='none')
    
    
    # Prediction curves: group by N and add to legend, use consistent colors
    for N, sub in pred_return_curves.groupby("N"):
        # ax.plot(sub["C"], sub["R_pred"], linewidth=2, color=COLOR_MAPPING[N])
        ax.scatter(sub["C"], sub["R_pred"], s=2, alpha=0.8, marker="x", edgecolor="none", color=config.COLOR_MAPPING[N])

    ax.set_xscale("log")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(handles, labels, loc="best", fontsize=8)
    return ax

def plot_fit_ip_2b(
    intrinsic_points,
    pred_intrinsic_curves,
    tangent_points,
    xlabel: str=None,
    ylabel: str=None,
    title: str=None,
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,4), dpi=300)

    # set legend
    unique_Ns = sorted(pred_intrinsic_curves["N"].unique())
    handles, labels, _ = _get_legends(unique_Ns)

    # y=x
    xs = np.geomspace(max(1e-12, pred_intrinsic_curves["C"].min()), pred_intrinsic_curves["C"].max(), 200)
    ax.plot(xs, xs, linestyle="--", label="y = x")

    # Fitted curves grouped by N in legend, use consistent colors
    for N, sub in pred_intrinsic_curves.groupby("N"):
        ax.plot(sub["C"], sub["I_pred"], color=config.COLOR_MAPPING[N])
    # Tangent points
    if len(tangent_points):
        ax.scatter(tangent_points["C_tan"], tangent_points["I_tan"], s=24, marker="o", edgecolor="none", label="tangent")

    # Group by runid and plot each run
    for g in data_proc.split_df(intrinsic_points, by_column='N'):
    # TODO for g in intrinsic_points.groupby('N'):
        current_N = g['N'].iloc[0]
        g = g.sort_values("C")

        # Draw line using pre-computed I_map values
        # (ln,) = ax.plot(g["C"], g["I_map"], alpha=0.8, color=COLOR_MAPPING[current_N])
        # test
        ax.scatter(g["C"], g["I_map"], s=2, alpha=0.3, marker="o", edgecolor="none", color=config.COLOR_MAPPING[current_N])
        
    ax.set_xscale("log"); ax.set_yscale("log")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(handles, labels, loc="best", fontsize=8)
    return ax

# =============================================================================
# EMPIRICAL FRONTIER VISUALIZATION
# =============================================================================
