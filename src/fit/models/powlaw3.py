import numpy as np
from dataclasses import dataclass
from typing import ClassVar
from src.fit.base import BaseFitter


@dataclass
class PowLaw3(BaseFitter):
    """
    Power law model: L = E + (A/n)^alpha + (B/x)^beta * (n/N0)^eta
    
    where:
    - n: N (model parameters)
    - x: E or C (compute/energy)
    - E: constant baseline
    - A, alpha: power law coefficients for N
    - B, beta: power law coefficients for compute/energy
    - eta: power law coefficient for N
    """
    MODEL_NAME = 'powlaw3'
    # Instance fields - dataclass automatically generates __init__
    E: float
    A: float
    B: float
    alpha: float
    beta: float
    eta: float
    N0: float

    # E,         A,        B,        alpha,  beta,   eta          # N0 固定为 1e10
    DEFAULT_BOUNDS: ClassVar[tuple] = (
        [1e-5,    2e8,     5e13,     0.8,    0.07,  -0.40, 0.9e10],       # Lower
        [5e-3,    2e9,     2e15,     1.6,    0.11,   0.05, 1.1e10]        # Upper
    )

    DEFAULT_P0: ClassVar[list] = [
        1e-3,     # E_init  小地板，避免整体>1
        6e8,      # A_init  明显低于之前，压低 0.5B
        8e14,     # B_init  把曲线量级拉回 0.5–1，不让 C 项过强或过弱
        1.1,      # alpha_init  更平缓，保证 0.5B ~ 1.5B 更接近
        0.09,     # beta_init   与上限斜率一致
        -0.15,     # eta_init    轻微下压大N，避免 7B 以上上移
        1e10     # N0_init    参考模型大小
    ]

    @classmethod
    def model(cls, data, *args):
        """
        Args:
            data: (n, x) tuple of arrays
            *args: E, A, B, alpha, beta, eta, N0
        """
        n, x = data
        E, A, B, alpha, beta, eta, N0 = args
        
        # Convert to arrays for vectorized operations
        n = np.atleast_1d(n)
        x = np.atleast_1d(x)
        
        # Apply power law formula: y = E + (A/n)^alpha + (B/x)^beta * (n/N0)^eta
        result = E + (A / n) ** alpha + (B / x) ** beta * (n / N0) ** eta
        
        return result
