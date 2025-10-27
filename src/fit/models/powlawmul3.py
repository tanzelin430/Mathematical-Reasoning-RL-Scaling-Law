import numpy as np
from dataclasses import dataclass
from typing import ClassVar
from src.fit.base import BaseFitter


@dataclass
class PowLawMul3(BaseFitter):
    f"""
    Power law model: L(N, C) = (N/N_p)^r * (C0 / C )^( k_{max} * N / (N + N0) )
    
    where:
        C0    : reference compute constant (~1e16–1e20)
        N_p   : pivot model parameter (~1e7–1e11)
        r     : right-shift exponent controlling pivot location (0–1.2)
        N0    : saturation scale for N (controls curvature of slope-vs-N, 1–20)
        k_max : maximum slope at N=N0
    
    Parameters:
        n: N (model parameters, e.g., 1e9, 3e9, 7e9)
        x: C (compute)
    """
    MODEL_NAME = 'powlawmul3'
    # Instance fields - dataclass automatically generates __init__
    C0: float
    N_p: float
    r: float
    N0: float
    k_max: float
    # Class variables (not part of __init__)
    # PARAM_NAMES is auto-generated from dataclass fields
    
    DEFAULT_BOUNDS = (
        # C0,    N_p,    r,     N0,    k_max
        [1e5,    1e5,   0.03,   5e5,      1e-4],    # Lower   放宽但不许到 0
        [1e20,   1e11,  2,   8e15,     1],    # Upper   给足自由找“右移”与交点
    )

    # —— Heuristic P0 adapted from your previous best:
    # Old best: r_old ≈ 1.07, k_max ≈ 0.136  →  effective N-exponent ≈ r_old * ⟨k⟩ ≈ O(0.07~0.1)
    # So set r_new around ~0.08
    DEFAULT_P0 = [
        1e6,     # C0
        1e7,     # N_p
        1,    # r
        1e10,  # N0
        0.18,    # kmax

    ]
    
    @classmethod
    def model(cls, data, *args):
        """
        Args:
            data: (n, x) tuple of arrays
            *args: C0, N_p, r, N0, k_max
        """
        n, x = data
        C0, N_p, r, N0, k_max = args
        
        # Convert to arrays for vectorized operations
        n = np.atleast_1d(n)
        x = np.atleast_1d(x)
        
        # Compute the exponent: k_max * N / (N + N0)
        exponent = k_max * n / (n + N0)
        
        # Apply formula: L = (N/N_p)^r * (C0/C)^exponent
        # Add safety checks for numerical stability
        with np.errstate(over='raise', invalid='raise'):
            try:
                result = np.power(n / N_p, r) * np.power(C0 / x, exponent)
            except (FloatingPointError, RuntimeWarning):
                # Handle potential overflow/underflow
                # Use log space for numerical stability
                log_result = r * np.log(n / N_p) + exponent * np.log(C0 / x)
                result = np.exp(log_result)
        
        return result

