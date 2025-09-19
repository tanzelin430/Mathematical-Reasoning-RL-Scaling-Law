import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, UnivariateSpline, interp1d
import data_proc
import fit_utils
import cma
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from scipy.interpolate import interp1d

from typing import Callable, List, Dict, Union, Optional, Tuple, Any
import fit_models
import config

# ========= Numerical Constants =========
_EPS_TOLERANCE = 1e-15   # Close to zero tolerance
_EPS_POS  = 1e-30  # Minimum positive value to prevent log(0)
_EPS_MONO = 1e-12  # Minimum increment to ensure strict monotonicity
_PENALTY_LARGE = 1e12  # Large penalty value for infeasible solutions

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
    verbose: bool = True,
    # Advanced options
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
    verbose : bool, default=True
        Whether to print optimization progress.
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
    def objective(params_opt):
        try:
            # Evaluate model directly with optimization parameters
            y_pred = func(xdata, *params_opt)
            y_pred = np.asarray(y_pred, dtype=float)
            
            # Calculate loss
            loss = loss_func(y_pred, ydata, weights)
            
            return loss
            
        except Exception as e:
            # Log the error for debugging but continue optimization
            if verbose:
                print(f"  Warning: objective evaluation failed with params {params_opt}: {type(e).__name__}: {e}")
            # Return large penalty to guide optimizer away from this region
            return _PENALTY_LARGE
    
    # Run multiple CMA-ES trials
    all_results = []
    best_loss = np.inf
    best_params = None
    best_es = None
    
    for trial in range(n_trials):
        if verbose:
            print(f"=== CMA-ES Trial {trial + 1}/{n_trials} ===")
        
        # Set initial parameters for this trial
        if trial == 0:
            # First trial: use provided initial guess
            initial_params = p0.copy()
        else:
            # Subsequent trials: random initialization within bounds
            rng = np.random.default_rng(seed + trial)
            initial_params = rng.uniform(lower_bounds, upper_bounds, n_params)
        
        # Bounds check for init params 
        if np.any(initial_params < lower_bounds) or np.any(initial_params > upper_bounds):
            for i, (p, lb, ub) in enumerate(zip(initial_params, lower_bounds, upper_bounds)):
                if p < lb:
                    raise ValueError(f"Parameter {i} is out of bounds: {p} < {lb}")
                elif p > ub:
                    raise ValueError(f"Parameter {i} is out of bounds: {p} > {ub}")
        
        # Run CMA-ES
        es = cma.CMAEvolutionStrategy(
            initial_params, 
            sigma0,
            {
                'popsize': popsize, 
                'seed': seed + trial,
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
            
            # Verbose output every 20 iterations
            if verbose and (iteration + 1) % 20 == 0:
                params_display = es.best.x
                
                # Check if parameters are stuck at boundaries
                bound_flags = []
                for i, (p, lb, ub) in enumerate(zip(params_display, lower_bounds, upper_bounds)):
                    if abs(p - lb) < abs(ub - lb) * 1e-6:  # Close to lower bound
                        bound_flags.append(f'p{i}↓')
                    elif abs(p - ub) < abs(ub - lb) * 1e-6:  # Close to upper bound
                        bound_flags.append(f'p{i}↑')
                
                # Build boundary flag string
                bound_str = f" [{','.join(bound_flags)}]" if bound_flags else ""
                
                # Format params as single line string
                params_str = ' '.join([f'{p:.2e}' for p in params_display])
                print(f"  Iter {iteration+1:3d}: loss={current_best:.6e}, params=[{params_str}]{bound_str}")
            
            if es.stop():
                if verbose:
                    print(f"  CMA-ES stopping criteria met at iteration {iteration+1}")
                break
        
        # Store trial result
        final_loss = es.best.f
        final_params = es.best.x
            
        trial_result = {
            'params': final_params.copy(),
            'loss': final_loss,
            'iterations': iteration + 1,
            'success': final_loss < _PENALTY_LARGE / 2
        }
        all_results.append(trial_result)
        
        if verbose:
            print(f"  Final loss: {final_loss:.6e}")
            print(f"  Final params: {final_params}")
        
        # Update best result
        if final_loss < best_loss:
            best_loss = final_loss
            best_params = final_params.copy()
            best_es = es
    
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
        
    if verbose:
        print(f"\n=== Final Results ===")
        print(f"Best loss: {best_loss:.6e}")
        print(f"Best parameters: {best_params}")
        print(f"R²: {r2:.4f}")
    
    return best_params, best_loss, r2


def get_x_y_data_from_df(
    df, 
    x_column_list, 
    y_column, 
    x_transform_list: list[Callable | None]=None, 
    y_transform: Callable=None
) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
    # 移除step=0的数据（因为E=0会导致log10(E)=-inf）
    df_fit = df[df['step'] > 0].reset_index(drop=True)
    
    # 对相同横坐标聚合：显示三个run的平均值
    # df_mean = (
    #     df_fit.groupby(['model_size', 'step'], as_index=False)
    #           .agg(N=('N', 'first'), C=('C', 'first'), E=('E', 'first'), ErrRate=('ErrRate', 'mean'))
    # )
    df_fit = data_proc.merge_duplicate_steps(df_fit, group_columns=['N', 'step'], mode='mean')
    
    # 准备拟合数据 - 返回tuple避免后续转置
    x_transform_list = [None] * len(x_column_list) if x_transform_list is None else x_transform_list
    x_data = tuple(x_transform(df_fit[c]) if x_transform is not None else df_fit[c] 
                     for c, x_transform in zip(x_column_list, x_transform_list))
    
    y_data = y_transform(df_fit[y_column]) if y_transform is not None else df_fit[y_column]
    y_data = y_data.to_numpy(dtype=float)
    
    # 确保所有x_arrays和y_data的长度相同
    for i, x_arr in enumerate(x_data):
        if len(x_arr) != len(y_data):
            raise ValueError(f"x_arrays[{i}] and y_data must have the same length, but got {len(x_arr)} and {len(y_data)}")
    
    return x_data, y_data



def fit_log_errrate(df, eval_name = "holdout_score"):
    # 准备拟合数据（参考run_xhyu_logistic.py的做法）
    df['ErrRate'] = 1 - df[eval_name]
    
    # clip for fitting
    df_fit = data_proc.apply_warmup_clipping(df, curve_column="N", warmup_frac=config.WARMUP_CLIPPING_FACTOR_FOR_RAW)

    x_data, y_data = get_x_y_data_from_df(
        df_fit, 
        x_column_list=["N", "E"], 
        y_column="ErrRate",
        x_transform_list=[None, lambda x: np.log10(x)], 
        y_transform=lambda x: np.log10(np.clip(x, 1e-12, None))
    )

    # log10_E_data = np.log10(E_data)
    # ErrRate_data = np.clip(df_fit['ErrRate'].to_numpy(dtype=float), 1e-12, None)
    # log10_ErrRate_data = np.log10(ErrRate_data)
    
    # 设置边界（与run_xhyu_logistic.py相同）
    bounds = (
        [0.001, 1e-12, 1e8, 1e-12, 1e8],     # 下界
        [0.5, 1e-8, 2e10, 1e-8, 2e10]        # 上界
    )

    # 初始参数猜测（与run_xhyu_logistic.py相同）
    p0 = [0.06, 1.7e-10, 5e9, 1e-9, 3e9]
    p0 = [0.3,1e-8, 2e10, 1e-8, 2e10]
    
    print("=== CMA-ES拟合 ===")
    print(f"数据点数量: {len(y_data)}")
    print(f"x_arrays: {len(x_data)} arrays with shapes {[arr.shape for arr in x_data]}")
    print(f"初始参数: L={p0[0]:.6f}, r={p0[1]:.2e}, N0_k={p0[2]:.2e}, r_e0={p0[3]:.2e}, N0_e0={p0[4]:.2e}")
    
    # 使用我们的CMA-ES curve_fit函数进行拟合
    fitted_params, best_loss, r2 = cma_curve_fit(
        fit_models.FitLogErrRate.model,
        x_data,  # 直接传递tuple，不需要转置
        y_data,
        p0=p0,
        bounds=bounds,
        n_trials=3,
        max_iters=1600,
        verbose=True
    )
    
    print(f"\n=== 拟合结果 ===")
    print(f"最终损失: {best_loss:.6e}")
    print(f"R²: {r2:.4f}")
    print(f"拟合参数: L={fitted_params[0]:.6f}, r={fitted_params[1]:.2e}, N0_k={fitted_params[2]:.2e}")
    print(f"           r_e0={fitted_params[3]:.2e}, N0_e0={fitted_params[4]:.2e}")
    
    # 创建预测器使用拟合得到的参数
    predicter = fit_models.FitLogErrRate(L=fitted_params[0], r=fitted_params[1], N0_k=fitted_params[2], 
                                 r_e0=fitted_params[3], N0_e0=fitted_params[4])

    return predicter