import numpy as np
from dataclasses import dataclass
from typing import ClassVar
from src.fit.base import BaseFitter


@dataclass
class PowLawMul(BaseFitter):
    f"""
    Power law model: L(N, C) = L_* * ( (C0 * N^r) / C )^( k_{max} * N / (N + N0) )
    
    where:
        L_*   : baseline loss at the reference compute (dimensionless, ~0.4–1.0)
        C0    : reference compute constant (~1e16–1e20)
        r     : right-shift exponent controlling pivot location (0–1.2)
        N0    : saturation scale for N (controls curvature of slope-vs-N, 1–20)
        k_max : maximum slope at N=N0
    
    Interpretation:
        • For each fixed N, log L ∝ - (N / (N + N0)) * log C   → pure power law in C
        • As N increases, slope magnitude grows smoothly (S-shaped in log N)
        • Curves intersect near C ≈ C0 * N^r  → intersection shifts right with N
        • At large N >> N0, exponent → 1  (slope saturates)
    
    Parameters:
        n: N (model parameters, e.g., 1e9, 3e9, 7e9)
        x: C (compute)
    """
    MODEL_NAME = 'powlawmul'
    # Instance fields - dataclass automatically generates __init__
    L_star: float
    C0: float
    r: float
    N0: float
    k_max: float
    # Class variables (not part of __init__)
    # PARAM_NAMES is auto-generated from dataclass fields
    
    # DEFAULT_BOUNDS = (
    #     [0.60,  5e10,  0.20,   8e9],    # Lower: L_star, C0,   r,     N0
    #     [2,  2e17,  1,  1e13],    # Upper
    # )

    # DEFAULT_P0 = [
    #     0.70,         # L_star
    #     1e17,         # C0  ⬅⬅ 控制交点x轴位置
    #     0.40,         # r   ⬅⬅ 适度控制随 N 右移速度
    #     15e9          # N0    ⬅⬅ 变大，缓和幂指数变化
    # ]

    # R2=0.7945
    # DEFAULT_BOUNDS: ClassVar[tuple] = (
    #     [0.65,   1e7,   0.30,   3e10],     # Lower: L_star, C0,    r,     N0
    #     [0.95,   5e12,  0.80,   5e11],     # Upper
    # )

    # DEFAULT_P0: ClassVar[list] = [
    #     0.80,            # L_star
    #     1e8,            # C0       ⬅ 更自由地选交点位置
    #     0.55,            # r        ⬅ 控制右移速度
    #     1e11             # N0       ⬅ 幂指数变化更稳
    # ]

    DEFAULT_BOUNDS = (
        [0.65,   1e7,   0.30,   1e9, 0.1],      # Lower: L_star, C0,    r,     N0,    k_max
        [2,   5e15,  0.80,   5e11, 1.0],     # Upper: tighter N0 to force early flattening
    )

    DEFAULT_P0 = [
        0.80,           # L_star
        1e8,            # C0       ⬅ 更自由地选交点位置
        0.55,            # r        ⬅ 控制右移速度
        1e11,             # N0       ⬅ 幂指数变化更稳
        0.6
    ]

    @classmethod
    def model(cls, data, *args):
        """
        Args:
            data: (n, x) tuple of arrays
            *args: L_star, C0, r, N0, k_max
        """
        n, x = data
        L_star, C0, r, N0, k_max = args
        
        # Convert to arrays for vectorized operations
        n = np.atleast_1d(n)
        x = np.atleast_1d(x)
        
        # Compute the exponent: N / (N + N0)
        exponent = k_max * n / (n + N0)
        
        # Compute the base: (C0 * N^r) / C
        base = (C0 * np.power(n, r)) / x
        
        # Apply formula: L = L_* * base^exponent
        # Add safety checks for numerical stability
        with np.errstate(over='raise', invalid='raise'):
            try:
                result = L_star * np.power(base, exponent)
            except (FloatingPointError, RuntimeWarning):
                # Handle potential overflow/underflow
                # Use log space for numerical stability
                log_result = np.log(L_star) + exponent * np.log(base)
                result = np.exp(log_result)
        
        return result

