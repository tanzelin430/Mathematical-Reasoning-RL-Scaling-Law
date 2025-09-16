# -*- coding: utf-8 -*-
"""Data processing and inspection utilities for RL scaling law analysis"""

from typing import List
import numpy as np
import pandas as pd

import fit_utils

__all__ = [
    # Data processing
    'validate_data', 'normalize_data', 'merge_duplicate_steps', 'split_df', 
    'aggregate_runs_by_N', 'estimate_phi_from_runs', 'apply_warmup_clipping',
    'smooth_df_single_curve', 'smooth_df', 'sort_dfs',
    # Data inspection
    'print_data_statistics'
]

# =============================================================================
# DATA PROCESSING
# =============================================================================

def validate_data(df, metric_columns=[]) -> pd.DataFrame:
    if metric_columns is None or len(metric_columns) == 0:
        raise ValueError("provide metric_columns to validate")
    required_cols = ['model_params','runid','step','tokens', 'cumulative_flops']
    required_cols.extend(metric_columns)
    # Check which cols are missing
    missing_col = [col for col in required_cols if col not in df.columns]
    if missing_col:
        raise ValueError(f"Data must contain columns: {missing_col}")
    
    # validate if containing NaN
    nan_col = [col for col in required_cols if df[col].isna().any()]
    if nan_col:
        raise ValueError(f"Data must not contain NaN in columns: {nan_col}")
    
    return df

def normalize_data(df) -> pd.DataFrame:
    """Normalize column names and data types to standard format"""
    rename_map = {
        'model_params':'N',
        'cumulative_flops':'C_raw',
        'runid':'runid',
        'step':'step',
        'tokens':'tokens',
        'cumulative_tokens':'T',
    }
    for k,v in list(rename_map.items()):
        if k in df.columns and v != k:
            df[v] = df[k]

    def _ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        return df
        
    df = _ensure_numeric(df, ['N','C_raw','tokens','step'])
    return df

def merge_duplicate_steps(df, group_columns: list, mode: str = 'mean') -> pd.DataFrame:
    """Handle duplicate step values within the same group
    
    Args:
        df: Input data
        group_columns: List of columns to group by (e.g., ['N', 'step'] or ['runid', 'step'])
        mode: Strategy for handling duplicate step values
            - 'mean': Average points with same group values (default)
            - 'first'/'last': Keep first/last occurrence
    """
    # Check duplicate situations within each group
    duplicates = df.duplicated(subset=group_columns).sum()
    if duplicates > 0:
        print(f"Found {duplicates} duplicate rows for columns {group_columns}, using strategy: {mode}")
        
        if mode == 'mean':
            # Average points with same group values
            agg_dict = {c: "mean" for c in df.columns if c not in group_columns}
            except_dict = {
                'N': 'first',
                'model_params': 'first',
                'model_size': 'first',
                'experiment_name': 'first',
                'experiment_id': 'first',
                'runid': 'first',
            }
            # Remove group columns from except_dict if they exist
            for col in group_columns:
                except_dict.pop(col, None)

            df = df.groupby(group_columns).agg({**agg_dict, **except_dict}).reset_index()
        elif mode in ['first', 'last']:
            # Keep only the first/last occurrence
            df = df.drop_duplicates(subset=group_columns, keep=mode)
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    return df

def split_df(df: pd.DataFrame, by_column: str) -> List[pd.DataFrame]:
    return [g.copy().reset_index(drop=True) for _, g in df.groupby(by_column, sort=False)]

def estimate_phi_from_runs(
    df,
    sample_size_per_step: float = 512.0,
    tail_fraction: float = 1.0,
):
    """
    Estimate φ = C / (N * E), where E = step * sample_size_per_step.
    Returns:
      - phi_global: Global median of tail φ for all runs (recommended as κ)
      - phi_by_N: {N: tail median for that N} (optional finer granularity)
      - stats: Tail q25/median/q75 for each run, for inspection
    """
    rows = []
    byN = {}

    for g in split_df(df, by_column='N'):
        _df = g[['runid','N','step','C_raw']].dropna().sort_values('step')
        if _df.empty: continue
        N = float(_df['N'].iloc[0])
        step = _df['step'].to_numpy(float)
        E = step * float(sample_size_per_step)
        denom = np.maximum(N * E, 1e-30)
        phi = _df['C_raw'].to_numpy(float) / denom

        m = len(phi)
        k0 = int(max(0, np.floor((1.0 - tail_fraction) * m)))
        tail = phi[k0:]
        tail = tail[np.isfinite(tail) & (tail > 0)]
        if tail.size == 0: continue

        q25, med, q75 = np.quantile(tail, [0.25, 0.5, 0.75])
        rows.append(dict(runid=str(_df['runid'].iloc[0]), N=N,
                         phi_q25=float(q25), phi_median=float(med),
                         phi_q75=float(q75), n_tail=int(tail.size)))

        byN.setdefault(N, []).append(float(med))

    stats = pd.DataFrame(rows).sort_values('N').reset_index(drop=True)
    if not stats.empty:
        phi_global = float(np.median(stats['phi_median'].to_numpy()))
    else:
        phi_global = np.nan

    phi_by_N = {float(N): float(np.median(meds)) for N, meds in byN.items()}
    return phi_global, phi_by_N, stats

def apply_warmup_clipping(df, curve_column: str, warmup_frac: float = 1.0/64.0):
    return pd.concat([apply_warmup_clipping_single_curve(g, warmup_frac) for g in split_df(df, by_column=curve_column)], ignore_index=True)

def apply_warmup_clipping_single_curve(df, warmup_frac: float = 1.0/64.0):
    """
    Apply warmup clipping to remove early training steps.
    
    Args:
        df: DataFrame with columns ['runid', 'step', 'N', 'E', 'C', 'R', ...]
        warmup_frac: Fraction of data to clip from the beginning (default: 1/64)
    
    Returns:
        DataFrame with warmup steps removed
    """
    if len(df) == 0 or warmup_frac <= 0:
        return df.copy()
    
    # Sort by step to ensure proper ordering
    if 'step' in df.columns:
        df = df.sort_values('step').reset_index(drop=True)
    
    # Calculate number of steps to clip
    n_total = len(df)
    n_clip = int(np.floor(n_total * float(warmup_frac)))
    
    if n_clip >= n_total:
        # Clip everything - return empty DataFrame with same structure
        return df.iloc[0:0].copy()
    
    # Apply clipping
    df_clipped = df.iloc[n_clip:].copy().reset_index(drop=True)
    
    return df_clipped


def calc_improve(df, y_column: str, curve_column: str, debug=True):
    """Calculate improvement relative to step=0 for each N group"""
    # Calculate improvement rate relative to step=0 for each N
    def _calc_improve(group):
        BASE_STEP = 1
        group_df = df.loc[group.index]
        step_0_rows = group_df[group_df['step'] == BASE_STEP]
        if len(step_0_rows) == 0:
            raise ValueError(f"No step=0 found for {curve_column}={group_df[curve_column].iloc[0]}")
        baseline_y = step_0_rows[y_column].iloc[0]
        return group - baseline_y
    
    improve_column = df.groupby(curve_column)[y_column].transform(_calc_improve)
    
    # Print some sample data after calculation  
    if debug:
        print("\n=== AFTER Improve calculation ===")
        sample_steps = [0, 1, 20, 50, 100]
        for N in sorted(df[curve_column].unique())[:2]:  # Show first 2 N values
            print(f"\nN = {N}:")
            for step in sample_steps:
                mask = (df[curve_column] == N) & (df['step'] == step)
                if mask.any():
                    y_val = df.loc[mask, y_column].iloc[0]
                    improve_val = improve_column.loc[mask].iloc[0]
                    print(f"  step={step}: {y_column}={y_val:.4f}, Improve={improve_val:.4f} ({y_column}-{y_column}_step0={improve_val:.4f})")
    
    return improve_column

# =============================================================================
# SMOOTHING
# =============================================================================

def sort_dfs(runs_raw_dfs):
    return [g.sort_values('C').reset_index(drop=True) for g in runs_raw_dfs]

def smooth_df_single_curve(
    df,
    col_x: str,
    col_y: str,
    col_y_out: str,
    monotonic: bool = True,
    increasing: bool = True,
    strict: bool = True,
    s_factor: float = 0.1,
    k_spline: int = 3,

    rolling_window: int = 20,
    min_se: float = 1e-3,
    x_inv_weight_power: float = 0.2,
):
    x = df[col_x]
    y = df[col_y]
    w = fit_utils.get_weight(
        x, y, 
        rolling_window=rolling_window, 
        min_se=min_se, 
        x_inv_weight_power=x_inv_weight_power)
    
    y_smooth, _f = fit_utils.fit_smooth_monotonic(
        x, y, 
        monotonic=monotonic, 
        increasing=increasing, 
        strict=strict,
        s_factor=s_factor, 
        w=w, 
        k_spline=k_spline)
    df_out = df#.copy()
    df_out[col_y_out] = y_smooth
    return df_out

def smooth_df(
    df,
    curve_column: str,
    col_x: str,
    col_y: str,
    col_y_out: str,
    monotonic: bool = True,
    increasing: bool = True,
    strict: bool = True,
    s_factor: float = 0.1,
    k_spline: int = 3,
    rolling_window: int = 20,
    min_se: float = 1e-3,
    x_inv_weight_power: float = 0.2,
) -> pd.DataFrame:
    return pd.concat(
        [
            smooth_df_single_curve(g, col_x, col_y, col_y_out, monotonic, increasing, strict, s_factor, k_spline, rolling_window, min_se, x_inv_weight_power) 
            for g in split_df(df, by_column=curve_column)
        ], 
        ignore_index=True)

# =============================================================================
# DATA INSPECTION
# =============================================================================

def print_data_statistics(df):
    """Print detailed statistics about the loaded dataframe"""
    print(f"Loaded {len(df)} rows")
    
    # Display data preview
    print("\n=== Data Preview ===")
    print("Column names:", list(df.columns))
    print("Data types:")
    print(df.dtypes)
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Check if data is aggregated (by runid format: agg_N_xxx indicates aggregated data)
    is_aggregated = 'runid' in df.columns and df['runid'].str.startswith('agg_N_').all() if 'runid' in df.columns else 'runid' not in df.columns
    
    if is_aggregated:
        # Aggregated data statistics
        print("\n=== Aggregated Data Statistics (by N) ===")
        n_stats = df.groupby('N').agg({
            'C': 'count',        # Grid points per N
            'R': ['min', 'max', 'mean'],  # R statistics
            'count': 'sum'       # Total run count
        }).round(3)
        n_stats.columns = ['grid_points', 'R_min', 'R_max', 'R_mean', 'total_runs']
        print(n_stats)
        
        print(f"\nTotal {df['N'].nunique()} different N values")
        print(f"Total grid points: {len(df)}")
        print(f"Average grid points per N: {len(df) / df['N'].nunique():.1f}")
        
        if 'R_std' in df.columns:
            print(f"\nR_std statistics:")
            print(f"  Average std: {df['R_std'].mean():.4f}")
            print(f"  Max std: {df['R_std'].max():.4f}")
    else:
        # Raw data statistics
        print("\n=== <N, runid> Combination Statistics ===")
        group_stats = df.groupby(['N', 'runid']).agg({
            'step': ['count', 'min', 'max'],
        }).round(2)
        group_stats.columns = ['step_count', 'step_min', 'step_max']
        print(group_stats)
        
        print(f"\nTotal {len(group_stats)} <N, runid> combinations")
        print(f"Step range: {group_stats['step_count'].min()} - {group_stats['step_count'].max()}")
        print(f"Average steps per combination: {group_stats['step_count'].mean():.1f}")
        
        # Statistics by N
        print("\n=== Statistics by N ===")
        n_stats = df.groupby('N').agg({
            'runid': 'nunique',  # Number of unique runids
            'step': 'count',     # Total steps
        }).round(2)
        n_stats.columns = ['unique_runs', 'total_steps']
        print(n_stats)


def inspect_data(df):
    """Inspect data statistics including model_size, runid distribution and step continuity"""
    print("\n---- Inspection ----")
    print(f"Total records: {len(df)}")
    model_sizes = sorted(df['model_size'].unique())
    print(f"Model sizes: {model_sizes}")
    unique_runids = df['runid'].nunique()
    print(f"Unique runid: {unique_runids}")
    step_min, step_max = df['step'].min(), df['step'].max()
    print(f"Step range: {step_min} - {step_max}")
    
    print("\n---- Detailed ----")
    for model_size in sorted(model_sizes):
        model_data = df[df['model_size'] == model_size]
        runids = model_data['runid'].unique()
        print(f"  {model_size}: {len(runids)} runid", end="")
        
        # Check if step ranges are consistent across runids
        step_ranges = {}
        for runid in runids:
            run_data = model_data[model_data['runid'] == runid]
            steps = sorted(run_data['step'].unique())
            min_step, max_step = min(steps), max(steps)
            step_ranges[runid] = (min_step, max_step)
        
        if len(step_ranges) > 1:
            ranges = list(step_ranges.values())
            min_ranges = [r[0] for r in ranges]
            max_ranges = [r[1] for r in ranges]
            if not (len(set(min_ranges)) == 1 and len(set(max_ranges)) == 1):
                print("  ⚠️  step range inconsistent")
            else:
                print()
        else:
            print()
        
        # Individual run details
        for runid in sorted(runids):
            run_data = model_data[model_data['runid'] == runid]
            steps = sorted(run_data['step'].unique())
            min_step, max_step = min(steps), max(steps)
            unique_step_count = len(steps)
            
            print(f"  {runid}: {unique_step_count} steps (unique) (min: {min_step}, max: {max_step})", end="")
            
            # Check for missing steps
            expected_steps = list(range(min_step, max_step + 1))
            missing_steps = set(expected_steps) - set(steps)
            if missing_steps:
                missing_list = sorted(list(missing_steps))
                print(f",  ⚠️  Missing steps: {missing_list}", end="")
            
            # Check for duplicated steps
            step_counts = run_data['step'].value_counts()
            duplicated_steps = step_counts[step_counts > 1]
            if not duplicated_steps.empty:
                dup_dict = {k: int(v) for k, v in dict(duplicated_steps).items()}
                print(f",  ⚠️  Duplicated: {dup_dict}", end="")
            
            print()  # New line after each run
    