import numpy as np
import pandas as pd
import cma
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score
from scipy.interpolate import UnivariateSpline
from typing import Callable, List, Dict, Union, Optional, Tuple

'''
Smoothing
'''

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
        return y, None

    # Spline smoothing on log(x)
    x_log = np.log(np.maximum(x, 1e-300))
    
    # Check that x is increasing (should be sorted in smooth_df_single_curve)
    if not np.all(np.diff(x_log) >= 0):
        raise ValueError("x must be sorted in increasing order before calling fit_spline")
    
    s_val = s_factor * len(x)
    spl = UnivariateSpline(x_log, y, w=w, s=s_val, k=k_spline)
    yhat = spl(x_log)

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
        y_predict = _f.predict
        yhat = _f.predict(x)
        # Attention: strict monotonic for smoother "Intrinsic - Compute" plot
        if strict:
            yhat = strictify_on_monotonic(yhat, increasing=increasing, cap_fraction=0.9, eps_rel=1e-9, eps_abs=1e-12)
    
    return yhat, y_predict


'''
Utility functions
'''

def get_weight(
    x: Union[np.ndarray, pd.Series],
    y: Union[np.ndarray, pd.Series],
    rolling_window: int = 20,
    min_se: float = 1e-3,
    x_inv_weight_power: float = 0.2,
    x_inv_weight_alpha: float = 1.0,
):
    """
    Compute combined weights from reliability (based on local variance) and x-coordinate.
    
    Parameters
    ----------
    x : array-like or pd.Series
        The x coordinates of data points
    y : array-like or pd.Series  
        The y coordinates of data points
    rolling_window : int, default=20
        Window size for rolling standard error calculation
    min_se : float, default=1e-3
        Minimum standard error to avoid division by zero
    x_inv_weight_power : float, default=0.2
        Power for inverse x weighting (0 to disable)
    x_inv_weight_alpha : float, default=1.0
        Controls the strength of x-inverse weighting, in range [0, 1]
        - alpha=0: only use reliability weights
        - alpha=1: full combination (original behavior)
        - 0<alpha<1: interpolates between the two
        
    Returns
    -------
    weights : ndarray
        Normalized weights with mean=1.0
    """
    # Dynamically adjust min_periods to ensure it doesn't exceed data length
    data_length = len(y)
    min_periods = min(max(3, rolling_window//2), data_length)
    se = pd.Series(y).rolling(rolling_window, center=True, 
            min_periods=min_periods).sem()
    
    se = se.fillna(se.median()).clip(lower=min_se).to_numpy(float)
    
    # Reliability weighting: (1/SE^2)
    w_reliability = 1.0 / (se**2)
    w_reliability = w_reliability / np.mean(w_reliability)  # normalize
    
    if x_inv_weight_power > 0 and x_inv_weight_alpha > 0:
        # Use gen_inv_weights to compute x-based weights
        w_xinv = gen_inv_weights(x, power=x_inv_weight_power)
        
        # Modulation-style combination: w = w_reliability * (1 + alpha * (w_xinv - 1))
        # This is equivalent to: w = (1-alpha) * w_reliability + alpha * (w_reliability * w_xinv)
        w = w_reliability * (1 + x_inv_weight_alpha * (w_xinv - 1))
    else:
        w = w_reliability

    w = w / np.mean(w)  # normalize

    return w

def gen_inv_weights(
    x: Union[np.ndarray, pd.Series],
    power: float = 1.0
) -> np.ndarray:
    """
    Generate inverse weights based on x coordinate: weight ~ 1/x^power.
    
    The x values are first normalized to relative scale (x/x_min) to avoid
    numerical precision issues when x contains very large numbers (e.g., 1e16-1e21).
    
    Parameters
    ----------
    x : array-like or pd.Series
        The x coordinates of data points
    power : float, default=1.0
        The power for inverse weighting. weight = 1/(x^power)
        - power=0: uniform weights (all ones)
        - power=1: inverse proportional (1/x)
        - power>1: stronger emphasis on small x values (e.g., 1/x²)
        
    Returns
    -------
    weights : ndarray
        Normalized weights with mean=1.0, suitable for cma_curve_fit
        
    Notes
    -----
    For numerical stability, x is normalized to x/x_min before computing weights.
    This ensures the weight calculation operates on reasonable numerical ranges
    even when x contains very large values.
    """
    # Convert to numpy array
    if isinstance(x, pd.Series):
        x_vals = x.to_numpy(dtype=float)
    else:
        x_vals = np.asarray(x, dtype=float)
    
    # Ensure all x values are positive (avoid division by zero)
    x_vals = np.maximum(x_vals, 1e-12)
    
    if power <= 0:
        # Return uniform weights
        return np.ones_like(x_vals, dtype=float)
    
    # Normalize x to relative scale: x_rel = x / x_min
    # This avoids numerical precision issues when x contains very large numbers
    # The relative ratios are preserved: x_rel_max / x_rel_min = x_max / x_min
    x_min = np.min(x_vals)
    x_relative = x_vals / x_min  # Now x_relative ranges from 1.0 to (x_max/x_min)
    
    # Calculate inverse weights on the relative scale
    w_inv = 1.0 / (x_relative ** power)
    
    # Normalize to mean=1
    w_inv = w_inv / np.mean(w_inv)
    
    return w_inv

def gen_grouped_inv_weights(
    x_group: Union[np.ndarray, pd.Series],
    x: Union[np.ndarray, pd.Series],
    power: float = 1.0
) -> np.ndarray:
    """
    Generate inverse weights grouped by x_group, weighted by 1/x_weight^power.
    
    Within each group: weights ~ 1/x_weight^power
    Across groups: each group has equal total weight
    
    Parameters
    ----------
    x_group : array-like
        Grouping variable (e.g., N)
    x_weight : array-like
        Variable for weighting (e.g., C or E)
    power : float, default=1.0
        Power for inverse weighting
        
    Returns
    -------
    weights : ndarray
        Normalized weights with mean=1.0
    """
    # Convert to numpy
    if isinstance(x_group, pd.Series):
        x_group = x_group.to_numpy(dtype=float)
    else:
        x_group = np.asarray(x_group, dtype=float)
        
    if isinstance(x, pd.Series):
        x = x.to_numpy(dtype=float)
    else:
        x = np.asarray(x, dtype=float)
    
    weights = np.zeros_like(x_group, dtype=float)
    unique_groups = np.unique(x_group)
    
    for group_val in unique_groups:
        mask = (x_group == group_val)
        
        # Use gen_inv_weights for this group
        w_group = gen_inv_weights(x[mask], power=power)
        
        # Normalize group to sum=1 (equal total weight per group)
        w_group = w_group / np.sum(w_group)
        
        weights[mask] = w_group
    
    # Normalize to mean=1
    return weights / np.mean(weights)

def calculate_r2(y_true, y_pred):
    """
    Calculate R² value using sklearn.metrics.r2_score.
    Clean interface that only takes the observed and predicted values.
    
    Parameters:
    -----------
    y_true : array-like
        True/observed values
    y_pred : array-like
        Predicted values
        
    Returns:
    --------
    float : R² value
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Remove any NaN values
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(valid_mask):
        return 0.0
        
    y_true_clean = y_true[valid_mask]
    y_pred_clean = y_pred[valid_mask]
    
    if len(y_true_clean) == 0:
        return 0.0
    
    # Use sklearn's r2_score - clean and simple!
    return r2_score(y_true_clean, y_pred_clean)



'''
Fitting functions
'''

# ========= Numerical Constants =========
_EPS_TOLERANCE = 1e-15   # Close to zero tolerance
_EPS_POS  = 1e-30  # Minimum positive value to prevent log(0)
_EPS_MONO = 1e-12  # Minimum increment to ensure strict monotonicity
_PENALTY_LARGE = 1e12  # Large penalty value for infeasible solutions
_SUCC_LOSS_THRESHOLD = 3e-4  # Success threshold for optimization (~ r2>0.99)
_SUCC_R2_THRESHOLD = 0.9


def cma_curve_fit(
    func: Callable,
    xdata: Union[np.ndarray, Tuple[np.ndarray, ...]],
    ydata: np.ndarray,
    p0: Optional[np.ndarray] = None,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    weights: Optional[np.ndarray] = None,
    # CMA-ES specific parameters
    n_trials: int = 3,
    max_iters: int = 160,
    popsize: int = 20,
    sigma0: float = 0.3,
    seed: int = 0,
    verbose_interval: int = 200,
    # Advanced options
    normalize_params: bool = True,  # 归一化参数到[0,1]以提高CMA-ES稳定性
    custom_loss: Optional[Callable] = None,
    **cma_kwargs
) -> Tuple[np.ndarray, float, float]:
    """
    General-purpose curve fitting using CMA-ES optimization with multiple trials.
    
    This function provides a scipy.optimize.curve_fit-like interface but uses
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for optimization,
    which is more robust for complex, multi-modal optimization landscapes.
    
    Parameters
    ----------
    func : callable
        The model function f(x, *params) to fit. Should take xdata as first argument
        followed by parameters to fit.
    xdata : array_like or tuple of array_like
        Independent variable(s). If tuple, func should accept multiple arrays.
    ydata : array_like
        Dependent variable.
    p0 : array_like, optional
        Initial parameter guess. If None, defaults to ones.
    bounds : tuple of array_like, optional
        Lower and upper bounds for parameters as (lower, upper).
    weights : array_like, optional
        Weights for data points. If None, uniform weights are used.
    n_trials : int, default=3
        Number of independent optimization trials to run.
    max_iters : int, default=160
        Maximum iterations per CMA-ES trial.
    popsize : int, default=20
        Population size for CMA-ES.
    sigma0 : float, default=0.3
        Initial step size for CMA-ES.
    seed : int, default=0
        Random seed for reproducibility.
    verbose_interval : int, default=200
        Interval for printing progress. Set to 0 to disable periodic progress output.
    custom_loss : callable, optional
        Custom loss function. Should take (y_pred, y_true, weights) and return scalar.
    **cma_kwargs
        Additional keyword arguments passed to CMA-ES.
    
    Returns
    -------
    popt : ndarray
        Optimal parameters found by the optimizer.
    best_loss : float
        Final loss value achieved by the optimizer.
    r2 : float
        R-squared value indicating goodness of fit.
    
    Examples
    --------
    >>> def exponential(x, a, b, c):
    ...     return a * np.exp(-b * x) + c
    >>> x = np.linspace(0, 4, 50)
    >>> y = exponential(x, 2.5, 1.3, 0.5) + np.random.normal(0, 0.2, len(x))
    >>> popt, loss, r2 = cma_curve_fit(exponential, x, y, p0=[1, 1, 1])
    
    >>> # Multi-dimensional input example
    >>> def surface(data, a, b, c):
    ...     x, y = data
    ...     return a * x**2 + b * y**2 + c
    >>> x = np.linspace(-2, 2, 20)
    >>> y = np.linspace(-2, 2, 20)
    >>> X, Y = np.meshgrid(x, y)
    >>> Z = surface((X.ravel(), Y.ravel()), 1.5, 2.0, 1.0)
    >>> popt, loss, r2 = cma_curve_fit(surface, (X.ravel(), Y.ravel()), Z)
    """
    
    # Validate inputs
    xdata = np.asarray(xdata) if not isinstance(xdata, tuple) else tuple(np.asarray(x) for x in xdata)
    ydata = np.asarray(ydata, dtype=float)
    
    if isinstance(xdata, tuple):
        n_points = len(xdata[0])
        if not all(len(x) == n_points for x in xdata):
            raise ValueError("All xdata arrays must have the same length")
    else:
        n_points = len(xdata)
    
    if len(ydata) != n_points:
        raise ValueError("xdata and ydata must have the same length")
    
    # Set up weights
    if weights is None:
        weights = np.ones(n_points, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        if len(weights) != n_points:
            raise ValueError("weights must have the same length as data")
    
    # Determine number of parameters
    if p0 is None:
        # Try to infer parameter count by calling func
        for n in range(1, 10):
            try:
                test_params = np.ones(n)
                func(xdata, *test_params)
                n_params = n
                break
            except (TypeError, ValueError):
                continue
        else:
            raise ValueError("Could not determine number of parameters. Please provide p0.")
        p0 = np.ones(n_params)
    else:
        p0 = np.asarray(p0, dtype=float)
        n_params = len(p0)
    
    # Set up bounds
    if bounds is None:
        # Default bounds: very wide range
        lower_bounds = np.full(n_params, -1e6)
        upper_bounds = np.full(n_params, 1e6)
    else:
        lower_bounds, upper_bounds = bounds
        lower_bounds = np.asarray(lower_bounds, dtype=float)
        upper_bounds = np.asarray(upper_bounds, dtype=float)
    
    # Set up loss function
    if custom_loss is None:
        def loss_func(y_pred, y_true, w):
            return np.average((y_pred - y_true)**2, weights=w)
    else:
        loss_func = custom_loss
    
    # Define objective function for CMA-ES
    def objective(params):
        try:
            y_pred = func(xdata, *params)
            y_pred = np.asarray(y_pred, dtype=float)
            loss = loss_func(y_pred, ydata, weights)
            return loss
        except Exception as e:
            print(f"  Warning: objective evaluation failed: {type(e).__name__}: {e}")
            return _PENALTY_LARGE
    
    # === 可选：参数归一化包装器（消除CMA-ES警告，提高稳定性）===
    if normalize_params:
        scales, offsets = upper_bounds - lower_bounds, lower_bounds
        to_norm, from_norm = lambda p: (p - offsets) / scales, lambda p: p * scales + offsets
        obj_original, objective = objective, lambda p: obj_original(from_norm(p))
        p0, lower_bounds, upper_bounds = to_norm(p0), np.zeros(n_params), np.ones(n_params)
    else:
        from_norm = lambda p: p  # Identity function when normalization is disabled
    
    # Run multiple CMA-ES trials
    all_results = []
    best_loss = np.inf
    best_params = None
    best_es = None
    
    # Use better random seed strategy: if seed is 0, generate random seeds
    if seed == 0:
        rng_master = np.random.default_rng()
        trial_seeds = [rng_master.integers(0, 2**31) for _ in range(n_trials)]
    else:
        trial_seeds = [seed + trial for trial in range(n_trials)]
    
    for trial in range(n_trials):
        trial_seed = trial_seeds[trial]
        print(f"    --- CMA-ES Trial {trial + 1}/{n_trials} ---")
        
        # Set initial parameters for this trial
        if trial == 0:
            # First trial: use provided initial guess
            initial_params = p0.copy()
        else:
            # Subsequent trials: random initialization within bounds
            rng = np.random.default_rng(trial_seed)
            initial_params = rng.uniform(lower_bounds, upper_bounds, n_params)
        
        # Bounds check for init params 
        if np.any(initial_params < lower_bounds) or np.any(initial_params > upper_bounds):
            for i, (p, lb, ub) in enumerate(zip(initial_params, lower_bounds, upper_bounds)):
                if p < lb or p > ub:
                    raise ValueError(f"Parameter {i} is out of bounds: {p} not in [{lb}, {ub}]")
        
        # Run CMA-ES
        es = cma.CMAEvolutionStrategy(
            initial_params, 
            sigma0,
            {
                'popsize': popsize, 
                'seed': trial_seed,
                'bounds': [lower_bounds, upper_bounds],
                # 'verb_disp': 0,  # Suppress CMA-ES output
                **cma_kwargs
            }
        )
        
        for iteration in range(max_iters):
            xs = es.ask()
            vals = [objective(x) for x in xs]
            es.tell(xs, vals)
            
            current_best = es.best.f
            
            # Verbose output
            if verbose_interval > 0 and (iteration + 1) % verbose_interval == 0:
                params_display = from_norm(es.best.x)
                bound_flags = []
                for i, (p, lb, ub) in enumerate(zip(es.best.x, lower_bounds, upper_bounds)):
                    if abs(p - lb) < abs(ub - lb) * 1e-6:
                        bound_flags.append(f'p{i}↓')
                    elif abs(p - ub) < abs(ub - lb) * 1e-6:
                        bound_flags.append(f'p{i}↑')
                
                # Build boundary flag string
                bound_str = f" [{','.join(bound_flags)}]" if bound_flags else ""
                
                # Format params as single line string
                params_str = ' '.join([f'{p:.2e}' for p in params_display])
                print(f"    Iter {iteration+1:3d}: loss={current_best:.6e}, params=[{params_str}]{bound_str}")
            
            if es.stop():
                print(f"    CMA-ES stopping criteria met at iteration {iteration+1}")
                break
        
        # Store trial result
        final_loss = es.best.f
        final_params = from_norm(es.best.x)
        
        trial_result = {
            'params': final_params.copy(),
            'loss': final_loss,
            'iterations': iteration + 1,
            'success': final_loss < _PENALTY_LARGE / 2
        }
        all_results.append(trial_result)
        
        print(f"    trial loss: {final_loss:.6e}")
        print(f"    trial params: {final_params}")
        
        # Update best result
        if final_loss < best_loss:
            best_loss = final_loss
            best_params = final_params.copy()
            best_es = es
        
        # Early stopping
        if best_loss < _SUCC_LOSS_THRESHOLD:
            break
    
    # Prepare return values
    if best_params is None:
        raise RuntimeError("All optimization trials failed")
    
    # Calculate R² using the best parameters
    try:
        y_pred = func(xdata, *best_params)
        y_pred = np.asarray(y_pred, dtype=float)
        
        # Calculate R² similar to r2_local in run_xhyu_logistic.py
        ss_res = np.sum((ydata - y_pred) ** 2)
        ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    except:
        r2 = np.nan
        
    print(f"\n--- Final Results ---")
    # print(f"Best loss: {best_loss:.6e}")
    print(f"Best parameters: {best_params}")
    # print(f"R²: {r2:.4f}")
    if r2 < _SUCC_R2_THRESHOLD:
        print(f"\n[Warning] R² is too low: {r2:.4f}\n")
    
    return best_params, best_loss, r2
