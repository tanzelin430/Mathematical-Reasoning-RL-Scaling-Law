import numpy as np
from dataclasses import dataclass
from typing import ClassVar
from src.fit.base import BaseFitter


@dataclass
class PowLaw2(BaseFitter):
    """
    Power law model: y = E + (A/n)^alpha + (B*n^gamma/x)^beta
    
    where:
    - n: N (model parameters)
    - x: E or C (compute/energy)
    - E: constant baseline
    - A, alpha: power law coefficients for N
    - B, beta: power law coefficients for compute/energy
    - gamma: power law coefficient for N
    """
    MODEL_NAME = 'powlaw2'
    # Instance fields - dataclass automatically generates __init__
    E: float
    A: float
    B: float
    alpha: float
    beta: float
    gamma: float

    DEFAULT_BOUNDS: ClassVar[tuple] = (
        [0,   1e9,   1e12,  1.2,  0.07,  0.8],   # Lower:  E,     A,     B,   alpha, beta, gamma
        [0.5,   1e12,  3e15,  4.0,  0.11,  1]    # Upper:  E,     A,     B,   alpha, beta, gamma
    )

    DEFAULT_P0: ClassVar[list] = [
        3e-3,     # E_init    : 小地板，防数值问题但不主导
        1e10,     # A_init    : 保持小N有明显偏移，支撑斜率差异
        3e15,     # B_init    : 压低 (B/C)^beta 的统治力，留给 N 维度空间
        1.8,      # alpha_init: 稍强的 N 依赖（已验证能分离斜率）
        0.095,    # beta_init : 对齐观测上限斜率 ~0.09
        0.8      # gamma_init: 先温和抬高大N、压低小N，主要用于纵向对齐
    ]

    @classmethod
    def model(cls, data, *args):
        """
        Args:
            data: (n, x) tuple of arrays
            *args: E, A, B, alpha, beta, gamma
        """
        n, x = data
        E, A, B, alpha, beta, gamma = args
        
        # Convert to arrays for vectorized operations
        n = np.atleast_1d(n)
        x = np.atleast_1d(x)
        
        # Apply power law formula: y = E + (A/n)^alpha + (B*n^gamma/x)^beta
        result = E + (A / n) ** alpha + (B * n ** gamma / x) ** beta
        
        return result
