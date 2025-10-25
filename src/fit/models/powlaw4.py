import numpy as np
from dataclasses import dataclass
from typing import ClassVar
from src.fit.base import BaseFitter


@dataclass
class PowLaw4(BaseFitter):
    """
    Power law model: y = E + (A/n)^alpha + (B/(x+Z))^beta
    
    where:
    - n: N (model parameters)
    - x: E or C (compute/energy)
    - Z: constant baseline for compute/energy
    - E: constant baseline
    - A, alpha: power law coefficients for N
    - B, beta: power law coefficients for compute/energy
    """
    MODEL_NAME = 'powlaw4'
    # Instance fields - dataclass automatically generates __init__
    E: float
    A: float
    B: float
    alpha: float
    beta: float
    Z: float
    
    # # Default fitting parameters (5 params: E, A, B, alpha, beta)
    # DEFAULT_BOUNDS = (
    #     [1e-6,   1e1,  1e12,   0.3,  5e-3],   # Lower: E, A, B, alpha, beta
    #     [0.5,   1e15,  1e25,  3,  0.2]    # Upper bounds
    # )
    
    # # Initial guess based on fitted values
    # DEFAULT_P0 = [1e-3, 2e9, 2e16, 1, 0.095]

    # L = E + (A/n)^alpha + (B/(x+Z))^beta
    DEFAULT_BOUNDS: ClassVar[tuple] = (
        [1e-4,   1e9,   1e12,  1.2,  0.07, 0.9e10],   # Lower:  E,     A,     B,   alpha, beta, Z
        [1e-2,   1e12,  3e15,  4.0,  0.11, 1.1e16]    # Upper:  E,     A,     B,   alpha, beta, Z
    )

    DEFAULT_P0: ClassVar[list] = [
        3e-3,    # E_init   : 让平台项可见但不主导
        1e10,    # A_init   : 小N时显著抬高 (A/N)^alpha
        3e15,    # B_init   : 强行把 (B/C)^beta 压下去一些
        1.8,     # alpha_init: 加强 N 依赖
        0.095,    # beta_init : 对齐你观测的上限斜率
        1e16     # Z_init    : 参考模型大小
    ]

    @classmethod
    def model(cls, data, *args):
        """
        Args:
            data: (n, x) tuple of arrays
            *args: E, A, B, alpha, beta, Z
        """
        n, x = data
        E, A, B, alpha, beta, Z = args
        
        # Convert to arrays for vectorized operations
        n = np.atleast_1d(n)
        x = np.atleast_1d(x)
        
        # Apply power law formula: y = E + A/n^alpha + B/(x+Z)^beta
        result = E + (A / n) ** alpha + (B / (x + Z)) ** beta
        
        return result
