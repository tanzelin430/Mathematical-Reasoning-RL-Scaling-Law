import numpy as np
from dataclasses import dataclass
from typing import ClassVar
from src.fit.base import BaseFitter


@dataclass
class PowLawMul(BaseFitter):
    f"""
    Power law model: L(N, C) = ((C0 * N^r) / C )^( k_{max} * N / (N + N0) )
    
    where:
        C0    : reference compute constant (~1e16–1e20)
        r     : right-shift exponent controlling pivot location (0–1.2)
        N0    : saturation scale for N (controls curvature of slope-vs-N, 1–20)
        k_max : maximum slope at N=N0
    
    Parameters:
        n: N (model parameters, e.g., 1e9, 3e9, 7e9)
        x: C (compute)
    """
    MODEL_NAME = 'powlawmul'
    # Instance fields - dataclass automatically generates __init__
    C0: float
    r: float
    N0: float
    k_max: float
    # Class variables (not part of __init__)
    # PARAM_NAMES is auto-generated from dataclass fields
    
    DEFAULT_BOUNDS = (
        # C0,    r,     N0,    k_max
        [1e-13,   0.3,   1e9, 1e-3],      # Lower
        [5e11,  4,   5e12, 1.0],     # Upper
    )

    DEFAULT_P0 = [
        1e8,            # C0       ⬅ 更自由地选交点位置
        1,           # r        ⬅ 控制右移速度
        1e11,           # N0       ⬅ tighter：幂指数收敛越快(early flattening)，同时变化越快
        0.5,            # k_max    ⬅ 最大渐进斜率/k(N)整体系数
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
        
        # Compute the exponent: N / (N + N0)
        exponent = k_max * n / (n + N0)
        
        # Compute the base: (C0 * N^r) / C
        base = (C0 * np.power(n, r)) / x
        
        # Apply formula: L = base^exponent
        # Add safety checks for numerical stability
        with np.errstate(over='raise', invalid='raise'):
            try:
                result = np.power(base, exponent)
            except (FloatingPointError, RuntimeWarning):
                # Handle potential overflow/underflow
                # Use log space for numerical stability
                log_result = exponent * np.log(base)
                result = np.exp(log_result)
        
        return result

