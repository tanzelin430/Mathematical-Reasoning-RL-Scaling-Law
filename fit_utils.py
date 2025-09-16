import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from scipy.interpolate import UnivariateSpline
from typing import Union

__all__ = [
    # Isotonic regression
    'isotonic_regression_pav', 'strictify_on_monotonic',
    
    # Spline fitting
    'fit_spline', 'fit_smooth_monotonic',
    
    # Weight calculation
    'get_weight'
]


def isotonic_regression_pav(
    x, y, 
    increasing=True, 
    sample_weight=None):

    iso_reg = IsotonicRegression(out_of_bounds='clip', increasing=increasing)
    iso_reg.fit(x, y, sample_weight=sample_weight)
    # iso_reg.predict(x)
    return iso_reg

def strictify_on_monotonic(
    y,
    *,
    increasing: bool = True,
    cap_fraction: float = 0.45,    # Maximum ratio of plateau tail point to right neighbor distance
    atol: float | None = None,     # Tolerance for plateau detection; None for adaptive based on data range
    eps_rel: float = 1e-9,         # Relative scale of total slope within plateau
    eps_abs: float = 1e-12,        # Absolute lower bound of total slope within plateau
):
    """
    Strictify "monotonic (non-decreasing/non-increasing)" sequences to "strictly monotonic",
    applying minimal linear slope only to plateau segments.
    - Do not modify non-plateau segments
    - Use right_gap to prevent boundary violations
    """
    y = np.asarray(y, dtype=float).copy()
    n = len(y)
    if n <= 1:
        return y


    # 1) Unify to "increasing" perspective (flip sign for decreasing, flip back at end)
    sign = 1.0 if increasing else -1.0
    z = sign * y  # Increasing perspective

    # 2) Plateau detection tolerance (adaptive to data range)
    rng = float(np.nanmax(z) - np.nanmin(z))
    tol = max(eps_abs, 1e-9 * max(1.0, rng)) if atol is None else float(atol)

    # 3) Assert input is already monotonic
    d = np.diff(z)
    if np.any(d < -tol):
        raise ValueError(f"Input is not non{'decreasing' if increasing else 'increasing'}.")
    # z = np.maximum.accumulate(z)


    delta_max_base = max(eps_abs, eps_rel * max(1.0, rng))
    i = 0
    while i < n - 1:
        j = i
        # Find plateau segment [i..j]: equal within tol
        while j + 1 < n and np.isclose(z[j + 1], z[i], rtol=0.0, atol=tol):
            j += 1
        if j > i:
            # "Safe distance" between plateau segment right side and next different value
            right_gap = z[j + 1] - z[j] if j < n - 1 else np.inf  # Tail plateau: no right neighbor constraint, use minimal amplitude
            if right_gap <= 0:
                raise ValueError(f"Impossible in strictify: Right gap is too small: {right_gap}")
            # Maximum total elevation amplitude allowed within plateau: must be minimal but not cross right neighbor
            delta_max = min(delta_max_base, cap_fraction * right_gap)

            # Distribute under linear slope, first point unchanged, tail point + delta_max
            z[i:j + 1] += delta_max * np.linspace(0.0, 1.0, j - i + 1)
        i = j + 1

    # 4) Restore direction
    z = sign * z
    return z

def fit_spline(
    x: np.ndarray, y: np.ndarray, 
    k_spline: int,
    s_factor: float,
    w: np.ndarray,
):

    if len(x) < 3:
        return y

    # Spline smoothing on log(x)
    # x = np.log(np.maximum(x.to_numpy(float), 1e-300))
    x = np.log(np.maximum(x, 1e-300))
    s_val = s_factor * len(x)
    spl = UnivariateSpline(x, y, w=w, s=s_val, k=k_spline)
    yhat = spl(x)

    return yhat, spl

def fit_smooth_monotonic(
    x: np.ndarray,
    y: np.ndarray,
    monotonic: bool,
    increasing: bool | None,
    strict: bool,
    s_factor: float,
    w: np.ndarray,
    k_spline: int,
):
    yhat, y_predict = fit_spline(x, y, k_spline, s_factor, w)

    if monotonic and increasing is None:
        k = min(5, len(y) // 2)
        mean_start = np.mean(y[:k])
        mean_end = np.mean(y[-k:])
        increasing = mean_start < mean_end  # Use average of first k and last k
    # yhat = y
    # Apply monotonic constraint if requested
    if monotonic:
        _f = isotonic_regression_pav(x, yhat, increasing=increasing, sample_weight=w)
        y_predict = _f.predict(x)
        yhat = _f.predict(x)
        # Attention: strict monotonic for smoother "Intrinsic - Compute" plot
        if strict:
            yhat = strictify_on_monotonic(yhat, increasing=increasing, cap_fraction=0.9, eps_rel=1e-9, eps_abs=1e-12)
    
    return yhat, y_predict


def get_weight(
    x: Union[np.ndarray, pd.Series],
    y: Union[np.ndarray, pd.Series],
    rolling_window: int = 20,
    min_se: float = 1e-3,
    x_inv_weight_power: float = 0.2,
):

    # Dynamically adjust min_periods to ensure it doesn't exceed data length
    data_length = len(y)
    min_periods = min(max(3, rolling_window//2), data_length)
    se = pd.Series(y).rolling(rolling_window, center=True, 
            min_periods=min_periods).sem()
    
    se = se.fillna(se.median()).clip(lower=min_se).to_numpy(float)
    

    # Combined weighting: (1/SE^2) * (1/x^power)
    w_reliability = 1.0 / (se**2)
    w_reliability = w_reliability / np.mean(w_reliability)  # normalize
    
    if x_inv_weight_power > 0:
        # Handle two types: pandas.Series and numpy.ndarray
        if isinstance(x, pd.Series):
            x_vals = x.to_numpy(float)
        else:
            x_vals = np.asarray(x, dtype=float)
        
        x_vals = np.maximum(x_vals, 1e-12)
        # normalize to 0-1
        w_xinv = 1.0 / (x_vals**x_inv_weight_power)
        w_xinv = w_xinv / np.mean(w_xinv)  # normalize
        
        w = w_reliability * w_xinv
    else:
        w = w_reliability

    w = w / np.mean(w)  # normalize

    return w