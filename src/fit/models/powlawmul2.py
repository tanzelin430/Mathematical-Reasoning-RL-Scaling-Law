import numpy as np
from dataclasses import dataclass
from typing import ClassVar
from src.fit.base import BaseFitter


@dataclass
class PowLawMul2(BaseFitter):
    f"""
    Power law model: L(N, C) = N^r * (C0 / C )^( k_{max} * N / (N + N0) )
    
    where:
        C0    : reference compute constant (~1e16–1e20)
        r     : right-shift exponent controlling pivot location (0–1.2)
        N0    : saturation scale for N (controls curvature of slope-vs-N, 1–20)
        k_max : maximum slope at N=N0
    
    Parameters:
        n: N (model parameters, e.g., 1e9, 3e9, 7e9)
        x: C (compute)
    """
    MODEL_NAME = 'powlawmul2'
    # Instance fields - dataclass automatically generates __init__
    C0: float
    r: float
    N0: float
    k_max: float
    # Class variables (not part of __init__)
    # PARAM_NAMES is auto-generated from dataclass fields
    
    DEFAULT_BOUNDS = (
        [1e9,  -2,    1e8,    1e-4],    # Lower
        [5e20,   2,    5e12,   1.0],     # Upper
    )

    # —— Heuristic P0 adapted from your previous best:
    # Old best: r_old ≈ 1.07, k_max ≈ 0.136  →  effective N-exponent ≈ r_old * ⟨k⟩ ≈ O(0.07~0.1)
    # So set r_new around ~0.08
    DEFAULT_P0 = [
        1e16,    # C0   —— 先沿用旧解的量级，便于收敛
        0,     # r    —— 约等于 r_old * 典型 k(N) 的量级
        1e11,     # N0   —— 与旧值同量级，控制 k(N) 的拐点
        0.18,     # k_max —— 略高于旧解 0.136，给局部搜索一点头部空间

    ]
    
    @classmethod
    def model(cls, data, *args):
        """
        Args:
            data: (n, x) tuple of arrays
            *args: C0, r, N0, k_max
        """
        n, x = data
        C0, r, N0, k_max = args
        
        # Convert to arrays for vectorized operations
        n = np.atleast_1d(n)
        x = np.atleast_1d(x)
        
        # Compute the exponent: k_max * N / (N + N0)
        exponent = k_max * n / (n + N0)
        
        # Apply formula: L = N^r * (C0/C)^exponent
        # Add safety checks for numerical stability
        with np.errstate(over='raise', invalid='raise'):
            try:
                result = np.power(n, r) * np.power(C0 / x, exponent)
            except (FloatingPointError, RuntimeWarning):
                # Handle potential overflow/underflow
                # Use log space for numerical stability
                log_result = r * np.log(n) + exponent * np.log(C0 / x)
                result = np.exp(log_result)
        
        return result

