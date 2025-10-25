import numpy as np
from dataclasses import dataclass
from typing import ClassVar
from src.fit.base import BaseFitter


@dataclass
class PowLawPlus(BaseFitter):
    f"""
    Power law model: L(N, C) = (C0 / C )^k(N) + N^r, k(N) = k_{max} * N / (N + N0)
    
    where:
        C0    : reference compute constant (~1e16–1e20)
        N0    : reference model parameter (~1e7–1e11)
        r     : right-shift exponent controlling pivot location (0–1.2)
        k_max : maximum slope at N=N0
    
    Parameters:
        n: N (model parameters, e.g., 1e9, 3e9, 7e9)
        x: C (compute)
    """
    MODEL_NAME = 'powlawplus'
    # Instance fields - dataclass automatically generates __init__
    C0: float
    N0: float
    r: float
    k_max: float
    # Class variables (not part of __init__)
    # PARAM_NAMES is auto-generated from dataclass fields
    
    # DEFAULT_BOUNDS = (
    #     # L_star, C0,    r,     N0,    k_max
    #     [0.65,   1e-13,   0.30,   1e9, 0.01],      # Lower
    #     [2,      5e11,  4,   5e11, 1.0],     # Upper
    # )

    # DEFAULT_P0 = [
    #     0.80,           # L_star
    #     1e8,            # C0       ⬅ 更自由地选交点位置
    #     0.55,           # r        ⬅ 控制右移速度
    #     1e11,           # N0       ⬅ tighter：幂指数收敛越快(early flattening)，同时变化越快
    #     0.6,            # k_max    ⬅ 最大渐进斜率/k(N)整体系数
    # ]

    DEFAULT_BOUNDS = (
        # C0,    N0,    r,    k_max
        [1e-13,  1e7,  -4, 1e-5],      # Lower
        [5e29,   5e11,  4, 1.0],     # Upper
    )

    DEFAULT_P0 = [
        1e8,            # C0       ⬅ 更自由地选交点位置
        1e11,           # N0       
        0.5,           # r       ⬅ tighter：幂指数收敛越快(early flattening)，同时变化越快
        0.6,            # k_max    ⬅ 最大渐进斜率/k(N)整体系数
    ]

    @classmethod
    def model(cls, data, *args):
        """
        Args:
            data: (n, x) tuple of arrays
            *args: C0, N0, r, k_max
        """
        n, x = data
        C0, N0, r, k_max = args
        
        # Convert to arrays for vectorized operations
        n = np.atleast_1d(n)
        x = np.atleast_1d(x)
        
        # Compute the exponent: N / (N + N0)
        exponent = k_max * n / (n + N0)
        
        return (C0 / x) ** exponent + (n ** r)

