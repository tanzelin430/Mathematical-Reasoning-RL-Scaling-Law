# -*- coding: utf-8 -*-
"""Intrinsic capability mapping and scaling law fitting

This module provides functions for:
- Intrinsic capability mapping (R -> I)
- Scaling law fitting (joint alternating optimization)
- Prediction functions
"""

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, UnivariateSpline, interp1d
import fit_utils
import cma
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from scipy.interpolate import interp1d

from typing import Callable, List, Dict, Union
import data_proc

__all__ = [
    # Intrinsic mapping
    'make_I_interpolator', 'map_points_to_intrinsic', 'build_intrinsic_table',
    
    # Scaling law fitting
    'fit_scaling_law_joint_alternating',
    
    # Prediction
    'predict_intrinsic_curves', 'calc_tangent_points', 'predict_return_curves',
    
    # Utility functions
    'IntrinsicFunction', 'isotonic_regression_pav'
]

# ========= Numerical Constants =========
_EPS_TOLERANCE = 1e-15   # Close to zero tolerance
_EPS_POS  = 1e-30  # Minimum positive value to prevent log(0)
_EPS_MONO = 1e-12  # Minimum increment to ensure strict monotonicity
_PENALTY_LARGE = 1e12  # Large penalty value for infeasible solutions

# =============================================================================
# INTRINSIC MAPPING
# =============================================================================

def build_intrinsic_table(df: pd.DataFrame, column_R: str = "R_smooth") -> pd.DataFrame:
    """
    Build empirical frontier table I_min(R).
    
    This creates the mapping from performance R to minimum required compute I.
    I(R) represents the "efficiency frontier" - the best possible compute efficiency
    across all training runs for achieving performance level R.
    """
    pcs = []
    for g in data_proc.split_df(df, by_column='N'):
        if column_R not in g.columns:
            raise ValueError(f"runs_dfs must contain column_R: {column_R}")
        sub = g[[column_R, "C"]].copy() # todo: no need copy
        # sub = g[[column_R, "C"]].dropna().copy()
        sub.rename(columns={column_R: "R", "C": "C"}, inplace=True)
        pcs.append(sub)
    
    P = pd.concat(pcs, ignore_index=True)
    # Group by R and find minimum C for each R level to ensure uniqueness
    P = P.groupby("R", as_index=False)["C"].min().sort_values("R").reset_index(drop=True)
    
    R = P["R"].to_numpy(float)
    C = P["C"].to_numpy(float)
    
    # I(R) = min{C_i : R_i >= R} - minimum compute for achieving R or higher
    I = np.array([np.min(C[R >= r]) for r in R])

    out = pd.DataFrame({
        "R_level": R,
        "I_raw": I
    })

    def remove_duplicated_I_raw(_df):
        # Remove duplicated I_raw points, for a smoother envelope
        # Record the global minimum R point
        min_R_idx = _df["R_level"].idxmin()
        min_R_point = _df.loc[min_R_idx]
        
        # Keep the maximum R point for each I_raw
        _df = _df.groupby("I_raw", as_index=False).agg({
            "R_level": "max"
        })
        
        # Add back the global minimum R point if not exists
        if min_R_point["R_level"] not in _df["R_level"].values:
            _df = pd.concat([_df, min_R_point.to_frame().T], ignore_index=True)
        
        _df = _df.sort_values("R_level").reset_index(drop=True)
        return _df
    
    # out = remove_duplicated_I_raw(out)

    return out

def make_I_interpolator(I_of_R: pd.DataFrame, mode: str = "step") -> Callable:
    """
    Generate I(R) query function. Default 'step' is right-continuous step (conservative, ensures I(R) ≤ C),
    Optional 'loglinear' does linear interpolation on log(I) for smoother appearance.
    """
    R = I_of_R["R_level"].to_numpy(float)
    I = I_of_R["I_raw"].to_numpy(float)
    # Monotonic non-decreasing & positive value protection
    I = np.maximum.accumulate(np.maximum(I, 1e-300))
    # No interpolation
    if mode == "step":
        # right-continuous step: for queried R, find minimum I_raw among all nodes with R_level <= R
        def I_interp(r_query):
            q = np.atleast_1d(r_query).astype(float)
            q = np.clip(q, R[0], R[-1])
            result = np.zeros_like(q)
            for i, r in enumerate(q):
                # Find all nodes with R_level <= r
                valid_indices = np.where(r <= R)[0]
                if len(valid_indices) > 0:
                    result[i] = np.min(I[valid_indices])
                else:
                    result[i] = I[0]  # If not found, use minimum value
            return result
        return I_interp
    # Smooth interpolation
    elif mode == "loglinear":
        def I_interp(r_query):
            q = np.atleast_1d(r_query).astype(float)
            result = np.full_like(q, np.nan)
            
            # Handle NaN and inf values
            valid_mask = np.isfinite(q) & (q >= R[0]) & (q <= R[-1])
            
            if not np.any(valid_mask):
                print("here", q, R[0], R[-1])
                return result
            
            valid_q = q[valid_mask]
            valid_result = np.zeros_like(valid_q)
            
            for i, r in enumerate(valid_q):
                # Find the two adjacent points for linear interpolation
                # Find the index where R[idx] <= r < R[idx+1]
                idx = np.searchsorted(R, r, side='right') - 1
                
                # Handle edge cases
                if idx < 0:
                    # r is before the first point, use first point
                    valid_result[i] = I[0]
                elif idx >= len(R) - 1:
                    # r is after the last point, use last point
                    valid_result[i] = I[-1]
                else:
                    # Linear interpolation between adjacent points
                    # I(r) = I[idx] + (I[idx+1] - I[idx]) * (r - R[idx]) / (R[idx+1] - R[idx])
                    r1, r2 = R[idx], R[idx+1]
                    i1, i2 = I[idx], I[idx+1]
                    
                    if r2 > r1:  # Avoid division by zero
                        weight = (r - r1) / (r2 - r1)
                        valid_result[i] = i1 + weight * (i2 - i1)
                    else:
                        valid_result[i] = i1  # Use first point if R values are equal
            
            result[valid_mask] = valid_result
            return result
        return I_interp

    else:
        raise ValueError("mode must be 'step' or 'loglinear'")

def map_points_to_intrinsic(points_df: pd.DataFrame, I_interp: Callable, R_column: str = "R") -> pd.DataFrame:
    out = points_df.copy()

    r = out[R_column].to_numpy(float)
    out["I_map"] = I_interp(r)
    out["efficient_ratio"] = out["C"] / out["I_map"]
    return out

# =============================================================================
# SCALING LAW FITTING
# =============================================================================
class IntrinsicFunction:
    def __init__(self, aN, aE, Nc, Ec, beta):
        self.aN = aN
        self.aE = aE
        self.Nc = Nc
        self.Ec = Ec
        self.beta = beta
        self.logNc = np.log(Nc)
        self.logEc = np.log(Ec)

    def __str__(self):
        return f"IntrinsicFunction(aN={self.aN}, aE={self.aE}, Nc={self.Nc}, Ec={self.Ec}, beta={self.beta})"
        
    @staticmethod
    def predict_logI_by_params(N, E, aN, aE, Nc, Ec, beta):
        log_t = np.logaddexp(aN*(np.log(Nc) - np.log(N)),
                             aE*(np.log(Ec) - np.log(E)))
        log_I_pred = -(1.0 / beta) * log_t
        return log_I_pred
    
    @staticmethod
    def predict_logI_by_logparams(N, E, aN, aE, logNc, logEc, beta):
        log_t = np.logaddexp(aN*(logNc - np.log(N)),
                             aE*(logEc - np.log(E)))
        log_I_pred = -(1.0 / beta) * log_t
        return log_I_pred
    
    @staticmethod
    def predict_I_by_params(N, E, aN, aE, Nc, Ec, beta):
        I_pred = ((Nc / N) ** aN + (Ec / E) ** aE) ** (- 1.0 / beta)
        return I_pred
    
    @staticmethod
    def predict_I_by_logparams(N, E, aN, aE, logNc, logEc, beta):
        I_pred = ((np.exp(logNc) / N) ** aN + (np.exp(logEc) / E) ** aE) ** (- 1.0 / beta)
        return I_pred
    
    @staticmethod
    def calc_beta(aN, aE):
        beta = (aN * aE) / (aN + aE + _EPS_POS)
        return beta

    def predict_logI(self, N, E):
        log_I_pred = self.predict_logI_by_logparams(N, E, self.aN, self.aE, self.logNc, self.logEc, self.beta)
        return log_I_pred
    
    def predict_I(self, N, E):
        I_pred = self.predict_I_by_params(N, E, self.aN, self.aE, self.Nc, self.Ec, self.beta)
        return I_pred


def _get_data_driven_bounds(N_arr, E_arr,
                            alpha_range=(0.1, 10.0),
                            beta_min=0.3,
                            soft_log_margin_percentage=8.0):
    bounds = {}
    bounds['alpha_min'], bounds['alpha_max'] = alpha_range
    bounds['beta_min'] = beta_min

    logN = np.log(np.maximum(N_arr, 1e-300))
    logE = np.log(np.maximum(E_arr, 1e-300))
    
    # Take maximum and minimum values as scalars
    logN_max = float(np.max(logN))
    logE_max = float(np.max(logE))
    logN_min = float(np.min(logN))
    logE_min = float(np.min(logE))
    
    logN_perc = logN_max * (1-soft_log_margin_percentage/100)
    logE_perc = logE_max * (1-soft_log_margin_percentage/100)

    # Based on scaling law physics: Nc and Ec can be any positive values
    # If beyond -logN_max and logN_max, new N factors can be extracted, so a reasonable hard bound is 1/N to N
    bounds['Nc_hard_lo'] = np.exp(-logN_max) 
    bounds['Nc_hard_hi'] = np.exp(logN_max) 
    bounds['Ec_hard_lo'] = np.exp(-logE_max)
    bounds['Ec_hard_hi'] = np.exp(logE_max)

    # Set soft bounds based on hard bounds
    bounds['Nc_soft_lo'] = np.exp(-logN_perc)
    bounds['Nc_soft_hi'] = np.exp(logN_perc)
    bounds['Ec_soft_lo'] = np.exp(-logE_perc)
    bounds['Ec_soft_hi'] = np.exp(logE_perc)

    # For warm-start
    bounds['N_min'] = np.exp(logN_min)
    bounds['E_min'] = np.exp(logE_min)
    # bounds['N_median'] = float(np.exp(np.median(logN_max)))
    # bounds['E_median'] = float(np.exp(np.median(logE_max)))
    return bounds

def _wrap_f_safe(f_raw, floor=None):
    if floor is None:
        rp = np.linspace(0.0, 1.0, 256)
        ip = np.asarray(f_raw(rp), float)
        pos = ip[np.isfinite(ip) & (ip > 0)]
        floor = float(np.percentile(pos, 1)) if pos.size else 1e-12
    floor = max(float(floor), 1e-12)
    def f_safe(qR):
        y = np.asarray(f_raw(qR), float)
        y[~np.isfinite(y)] = floor
        y[y <= 0] = floor
        return y
    f_safe._floor = floor
    return f_safe

def isotonic_regression_pav(x, y, increasing=True):
    """PAV (Pool Adjacent Violators) algorithm for monotonic regression"""
    iso = IsotonicRegression(increasing=increasing, out_of_bounds='clip')
    return iso.fit_transform(x, y)

def _make_Iemp_interpolator(I_of_R):
    """Continuous query function for empirical frontier I_raw(R) (for scaling only, no shape change)"""
    
    R = np.asarray(I_of_R["R_level"], float)
    I = np.asarray(I_of_R["I_raw"], float)
    z = np.log(np.maximum(I, _EPS_POS))
    p = PchipInterpolator(R, z, extrapolate=False)  # Disable extrapolation
    
    def interpolator(rq):
        rq = np.asarray(rq, float)
        # Limit to empirical frontier R range
        rq = np.clip(rq, R.min(), R.max())
        return np.exp(p(rq))
    
    return interpolator


def _make_R_from_I_inverse(f_func, R_min=0.0, R_max=1.0, n=4096):
    """Construct continuous inverse function R=f^{-1}(I)"""
    # f_safe = _wrap_f_safe(f_func)

    # Sampling
    R = np.linspace(R_min, R_max, int(n))
    logI = f_func(R)
    I = np.exp(logI)

    # # Handle monotonicity and flat segments in logI domain
    # logI = np.log(I)
    # Non-decreasing (smooth occasional micro-reversals)
    logI = np.maximum.accumulate(logI)
    # Deduplication (required for strict monotonicity): keep first occurrence position
    logI_u, idx = np.unique(logI, return_index=True)
    R_u = R[idx]

    # Degenerate handling when insufficient valid points
    if len(logI_u) == 0:
        # Complete anomaly: return constant function
        def R_from_I(Iq):
            return np.full_like(np.asarray(Iq, float), R_min)
        return R_from_I
    elif len(logI_u) == 1:
        # All points fall on same logI: inverse function has only one value
        Ru = float(R_u[0])
        def R_from_I(Iq):
            return np.full_like(np.asarray(Iq, float), Ru)
        return R_from_I
    elif len(logI_u) == 2:
        # Two points: use linear interpolation
        Imin, Imax = np.exp(logI_u[0]), np.exp(logI_u[-1])
        def R_from_I(Iq):
            J = np.asarray(Iq, float)
            J[~np.isfinite(J)] = 1e-12
            J[J <= 0] = 1e-12
            J = np.clip(J, Imin, Imax)
            return np.interp(np.log(J), logI_u, R_u)
        return R_from_I
    else:
        # ≥3 points: PCHIP (logI_u strictly increasing, meets requirements)
        Imin, Imax = np.exp(logI_u[0]), np.exp(logI_u[-1])
        inv = PchipInterpolator(logI_u, R_u, extrapolate=True)
        def R_from_I(Iq):
            J = np.asarray(Iq, float)
            J[~np.isfinite(J)] = 1e-12
            J[J <= 0] = 1e-12
            J = np.clip(J, Imin, Imax)
            return inv(np.log(J))
        return R_from_I


def fit_scaling_law_joint_alternating(
    # data
    df_NERCI,
    R_column: str,
    I_column: str,
    fit_fR = True,
    # CMA-ES parameters
    seed=0, # random seed for exploration
    n_trials=3,
    max_iters=160, # inner loop max iterations
    popsize=20,
    sigma0=0.3, # initial step size
    early_stop_tol=1e-3,
    w_eq=1e-6, # weight for equality loss
    w_env=1e-7, # weight for envelope loss
    w_f_ratio=1.0,
    soft_penalty_ratio=0.01,
    # bounds
    alpha_range=(0.01, 3.0),
    beta_min=0.01,
    soft_log_margin_percentage=5.0
):
    """
    Joint fitting using alternating optimization approach.
    
    Simplified version following the plan:
    1) Data: <N, E, R, C, I>
    2) CMA iter: use IntrinsicFunction and fit_utils.isotonic_regression_pav
    3) Scale alignment: in outer_loop, outside CMA iter
    
    Returns: dict(αN, αE, Nc, Ec, β, f_func, fit_loss, ...)
    """

    # =============================================================================
    # STEP 1: PREPARE DATA FOR OPTIMIZATION
    # =============================================================================
    
    # validate df
    for col in ["N","E",R_column,"C",I_column]:
        if col not in df_NERCI.columns:
            raise ValueError(f"missing column {col} in df_NERCI")
    
    N_arr = df_NERCI["N"].to_numpy(float)
    E_arr = df_NERCI["E"].to_numpy(float)
    R_arr = df_NERCI[R_column].to_numpy(float)
    C_arr = df_NERCI["C"].to_numpy(float)
    I_arr = df_NERCI[I_column].to_numpy(float)

    # =============================================================================
    # STEP 2: SETUP OPTIMIZATION BOUNDS AND PARAMETERS
    # =============================================================================

    # Set bounds based on actual data range
    bounds = _get_data_driven_bounds(N_arr, E_arr, alpha_range=alpha_range, beta_min=beta_min, 
                                   soft_log_margin_percentage=soft_log_margin_percentage)
    
    # Set hard bounds in log space
    lb = np.array([np.log(bounds['alpha_min']), np.log(bounds['alpha_min']), 
                   np.log(bounds['Nc_hard_lo']), np.log(bounds['Ec_hard_lo'])], float)
    ub = np.array([np.log(bounds['alpha_max']), np.log(bounds['alpha_max']), 
                   np.log(bounds['Nc_hard_hi']), np.log(bounds['Ec_hard_hi'])], float)

    # =============================================================================
    # STEP 3: DEFINE OBJECTIVE FUNCTION FOR CMA-ES
    # =============================================================================
    
    def objective_loss(x_log, data):
        # Work in log space for numerical stability
        logaN, logaE, logNc, logEc = x_log
        aN, aE= np.exp(logaN), np.exp(logaE)
        
        # hard bounds for beta
        beta = IntrinsicFunction.calc_beta(aN, aE)
        
        # Outer loop: Fit aN, aE, Nc, Ec, beta with I target = I(N, E)
        logI_tgt = IntrinsicFunction.predict_logI_by_logparams(N_arr, E_arr, aN, aE, logNc, logEc, beta)
        logI_obs = np.log(np.maximum(I_arr, 1e-30))

        if fit_fR:
            # Inner loop: Fit f(R) with <R, I target> by isotonic regression
            # First sort R_arr in ascending order <R_arr, logI_tgt>
            sort_idx = np.argsort(R_arr)
            R_arr_sorted = R_arr[sort_idx]
            logI_tgt_sorted = logI_tgt[sort_idx]


            # ===============option1=======================

            logI_f, f_R_logI = fit_utils.fit_smooth_monotonic(
                R_arr_sorted, logI_tgt_sorted, 
                monotonic=True, increasing=True, strict=True, 
                s_factor=0, w=np.ones_like(R_arr_sorted), k_spline=3
            )

            # ===============option2=======================

            # # Use isotonic regression directly, completely bypass spline
            # iso_reg = fit_utils.isotonic_regression_pav(R_arr_sorted, logI_tgt_sorted, increasing=True)
            # logI_iso = iso_reg.predict(R_arr_sorted)
            # logI_strict = fit_utils.strictify_on_monotonic(logI_iso, increasing=True)
            # logI_f = logI_strict
            
            # # create interpolation function for f(R)
            # f_R_logI = interp1d(R_arr_sorted, logI_strict, kind='linear', bounds_error=False, fill_value='extrapolate')

            # ===============end=======================

            data["f_R_logI"] = f_R_logI

            # ==========================
            # L_f_env: Envelope Loss for f(R)
            # ==========================
            h = logI_f - np.log(C_arr[sort_idx]) # - np.log(1 + eps)
            mask = (h > 0)
            if np.any(mask):
                h_masked = h[mask]
                E_masked = E_arr[sort_idx][mask]
                # Use same weighting strategy for L_env
                weights_f_env = 1.0 / np.maximum(E_masked, 1e-30)
                L_f_env = float(np.average(h_masked**2, weights=weights_f_env))
            else:
                L_f_env = 0.0

            # ==========================
            # L_f_eq: Equivalent Loss for f(R)
            # ==========================
            logI_obs_sorted = logI_obs[sort_idx]
            
            # use 1/E as weight, give more weight to more efficient points
            E_sorted = E_arr[sort_idx]
            weights = 1.0 / np.maximum(E_sorted, 1e-30)  # Avoid division by zero

            L_f_eq = float(np.average((logI_f - logI_obs_sorted)**2, weights=weights))
        

        if beta < bounds['beta_min']:
            return _PENALTY_LARGE

        # ==========================
        # L_eq: Equivalent Loss for I(N,E)
        # ==========================
        weights = 1.0 / np.maximum(E_arr, 1e-30)  # Avoid division by zero
        L_eq = float(np.average((logI_tgt - logI_obs)**2, weights=weights))

        # ==========================
        # L_env: Intrinsic envelope constraint loss for I(N,E)
        # ==========================
        h = logI_tgt - np.log(C_arr) # - np.log(1 + eps)
        mask = (h > 0)
        if np.any(mask):
            h_masked = h[mask]
            E_masked = E_arr[mask]
            # Use same weighting strategy for L_env
            weights_env = 1.0 / np.maximum(E_masked, 1e-30)
            L_env = float(np.average(h_masked**2, weights=weights_env))
        else:
            L_env = 0.0
        
        # Joint loss
        total_loss = w_eq * L_eq + w_env * L_env
        if fit_fR:
            total_loss += (w_eq * L_f_eq + w_env * L_f_env) * w_f_ratio

        # Apply soft bound
        def qpen(x_log, lo, hi, m=2.0):
            x_actual = np.exp(x_log)
            if x_actual < lo: return ((lo - x_actual)/m)**2
            if x_actual > hi: return ((x_actual - hi)/m)**2
            return 0.0
        pen = qpen(logNc, bounds['Nc_soft_lo'], bounds['Nc_soft_hi']) + qpen(logEc, bounds['Ec_soft_lo'], bounds['Ec_soft_hi'])
        return total_loss + soft_penalty_ratio * pen
        

    # =============================================================================
    # STEP 4: CMA-ES OPTIMIZATION
    # =============================================================================
    
    def run_cma(initial_params_log, seed0):
        initial_params_log = np.clip(initial_params_log, lb, ub)

        es = cma.CMAEvolutionStrategy(
            initial_params_log, sigma0,
            {'popsize': popsize, 'seed': seed0, 'bounds': [lb, ub]}
        )
        bestf = np.inf
        for iter_count in range(max_iters):
            xs = es.ask()
            vals = [objective_loss(x, {}) for x in xs]
            es.tell(xs, vals)
            
            if es.best.f < bestf:
                bestf = es.best.f
            
            # Print parameters and loss every 10 iterations
            if (iter_count + 1) % 10 == 0:
                logaN, logaE, logNc, logEc = es.best.x
                aN, aE = np.exp(logaN), np.exp(logaE)
                Nc, Ec = np.exp(logNc), np.exp(logEc)
                beta = IntrinsicFunction.calc_beta(aN, aE)
                
                # Check if parameters are stuck at boundaries
                bound_flags = []
                
                # Check aN boundary
                if abs(aN - bounds['alpha_min']) < 1e-6:
                    bound_flags.append('aN↓')
                elif abs(aN - bounds['alpha_max']) < 1e-6:
                    bound_flags.append('aN↑')
                
                # Check aE boundary
                if abs(aE - bounds['alpha_min']) < 1e-6:
                    bound_flags.append('aE↓')
                elif abs(aE - bounds['alpha_max']) < 1e-6:
                    bound_flags.append('aE↑')
                
                # Check Nc boundary
                if abs(Nc - bounds['Nc_hard_lo']) < 1e-6:
                    bound_flags.append('Nc↓')
                elif abs(Nc - bounds['Nc_hard_hi']) < 1e-6:
                    bound_flags.append('Nc↑')
                
                # Check Ec boundary
                if abs(Ec - bounds['Ec_hard_lo']) < 1e-6:
                    bound_flags.append('Ec↓')
                elif abs(Ec - bounds['Ec_hard_hi']) < 1e-6:
                    bound_flags.append('Ec↑')
                
                # Check beta boundary
                if abs(beta - bounds['beta_min']) < 1e-6:
                    bound_flags.append('β↓')
                
                # Build boundary flag string
                bound_str = f" [{','.join(bound_flags)}]" if bound_flags else ""
                
                print(f"  iter {iter_count+1:3d}: aN={aN:.3f}, aE={aE:.3f}, Nc={Nc:.2e}, Ec={Ec:.2e}, beta={beta:.3f}, loss={es.best.f:.2e}{bound_str}")
            
            if es.stop():
                break
        return es

    # =============================================================================
    # STEP 5: MAIN OPTIMIZATION LOOP WITH SCALE ALIGNMENT
    # =============================================================================
    
    def run_optimization_loop(initial_params_log, seed_offset, name_prefix=""):
        """Run complete optimization loop including scale alignment"""
        current_params_log = initial_params_log.copy()
        loss_prev = None
        best_loss = np.inf
        best_result = None
        
        for loop in range(int(max(1, n_trials))):
            print(f"{name_prefix} outer loop {loop+1}")
            # CMA optimization
            es = run_cma(current_params_log, seed + seed_offset + loop)
            current_best_params_log = np.array(es.best.x, float)
            current_loss = float(es.best.f)

            # Recover params
            logaN, logaE, logNc, logEc = current_best_params_log
            aN, aE = np.exp(logaN), np.exp(logaE)
            beta = IntrinsicFunction.calc_beta(aN, aE)
            
            tmp={}

            current_best_result = {
                'alpha_N': aN, 'alpha_E': aE, 'N_c': np.exp(logNc), 'E_c': np.exp(logEc),
                'beta': beta, 'fit_loss': current_loss
            }
            if fit_fR:
                _ = objective_loss(es.best.x, tmp)   # Let objective recalculate with best.x and write corresponding f to tmp
                f_R_logI = tmp["f_R_logI"]
                current_best_result['f_R_logI'] = f_R_logI

            # Update best solution
            if current_loss < best_loss:
                best_loss = current_loss
                best_result = current_best_result

            loss_prev = current_loss
        
        return best_loss, best_result

    # =============================================================================
    # STEP 6: MAIN OPTIMIZATION EXECUTION
    # =============================================================================
    
    # Main optimization with warm-start
    print("=== Starting main optimization ===")
    x_best_log = np.array([np.log(1.5), np.log(1.5), np.log(bounds['N_min']), np.log(bounds['E_min'])], float)
    best_loss, best_result = run_optimization_loop(x_best_log, 101, "main")

    # # Optional: Random restart for comparison
    # print("\n=== Starting random restart ===")

    # rng = np.random.default_rng(seed)
    # x_rand_log = np.array([
    #     rng.uniform(np.log(bounds['alpha_min']), np.log(bounds['alpha_max'])),
    #     rng.uniform(np.log(bounds['alpha_min']), np.log(bounds['alpha_max'])),
    #     rng.uniform(np.log(bounds['Nc_hard_lo']), np.log(bounds['Nc_hard_hi'])),
    #     rng.uniform(np.log(bounds['Ec_hard_lo']), np.log(bounds['Ec_hard_hi']))
    # ], float)
    
    # Use selected best result
    assert best_result is not None, "Impossible: best_result is None"

    # Final feasibility check
    if fit_fR:
        f_R_logI = best_result['f_R_logI']
        logI_chk = f_R_logI(R_arr)
        if np.any(logI_chk - np.log(np.maximum(C_arr, _EPS_POS)) > _EPS_TOLERANCE):
            print("[WARN] Feasibility violated at return; consider increasing outer_loops or adjusting bounds.")
    
    return {
        'intrinsic': IntrinsicFunction(best_result['alpha_N'], best_result['alpha_E'], best_result['N_c'], best_result['E_c'], best_result['beta']),
        'f_R_logI': best_result['f_R_logI'] if fit_fR else None,
    }

# =============================================================================
# PREDICTION
# =============================================================================


def sample_x(x_range: tuple[float, float], on_log: bool = True, sample_num: int = 200) -> np.ndarray:
    if on_log:
        log_x = np.linspace(np.log(x_range[0]), np.log(x_range[1]), sample_num)
        x = np.exp(log_x)
    else:
        log_x = np.geomspace(np.log(x_range[0]), np.log(x_range[1]), sample_num)  # Match actual data range
        x = np.exp(log_x)
    return x

def predict_intrinsic_curves(df_NEC: pd.DataFrame, intrinsic: IntrinsicFunction, phi: float, warmup_clipping_factor: float, sample_size_per_step: float) -> pd.DataFrame:
    """Predict intrinsic curves and tangent points"""

    rows = []
    for N, sub in df_NEC.groupby('N'):
        # sample X axis by E to plot fitting curves
        min_step = sub["step"].min()
        max_step = sub["step"].max()
        # apply warmup clipping to range (min, max)
        # clipping = np.round((max_step - min_step + 1) * warmup_clipping_factor)
        # min_step = min_step + clipping
        # if clipping > 0:
        #     print(f"clipping: {clipping} steps. (min_step: {min_step}, max_step: {max_step}, warmup_clipping_factor: {warmup_clipping_factor})")

        E_use = sample_x(x_range=(min_step * sample_size_per_step, max_step * sample_size_per_step), on_log=False, sample_num=max_step-min_step+1)
        
        I_pred = intrinsic.predict_I(N, E_use)
        C = phi * N * E_use
        rows.append(pd.DataFrame({'C': C, 'I_pred': I_pred, 'N': N, 'E': E_use}))
    return pd.concat(rows, ignore_index=True)


def calc_tangent_points(df_I_pred_CNE) -> pd.DataFrame:
    # Tangent points: find points where dI/dC ≈ 1 on (log C, log I) (numerical approximation)
    tang_rows = []
    for N, sub in df_I_pred_CNE.groupby('N'):
        if len(sub) < 3: 
            continue
        sub = sub.sort_values('C')
        C_vals = sub['C'].to_numpy(); I_vals = sub['I_pred'].to_numpy()
        dI = np.gradient(np.log(I_vals), np.log(C_vals))  # Take derivative on log-log scale
        idx = int(np.argmin(np.abs(dI - 1.0)))            # Slope≈1 means I~C
        tang_rows.append({'N': float(N), 'C_tan': float(C_vals[idx]), 'I_tan': float(I_vals[idx])})
    tangent_points = pd.DataFrame(tang_rows)

    return tangent_points

def predict_return_curves(df_I_pred_CNE, f_R_logI) -> pd.DataFrame:
    """Use scaling law formula to predict return curves"""

    R_from_I = _make_R_from_I_inverse(f_R_logI)
    # apply R_from_I to df_I_pred_NE['I_pred'] and set to "R_pred"
    df_I_pred_CNE['R_pred'] = R_from_I(df_I_pred_CNE['I_pred'])
    
    df_R_pred_I_pred_CNE = df_I_pred_CNE
    return df_R_pred_I_pred_CNE
