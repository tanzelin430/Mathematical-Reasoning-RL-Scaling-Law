import numpy as np
from dataclasses import dataclass
from typing import ClassVar
from src.fit.base import BaseFitter

@dataclass
class InvExpKExp(BaseFitter):
    """
    Inverse exponential model with power-law k(N): y = k(N) * log10(S/x)
    where k(N) = K * N^alpha (power-law relationship with N)
    
    Parameters:
    - S: scaling constant
    - K: power-law coefficient
    - alpha: power-law exponent for k(N) = K * N^alpha
    """
    MODEL_NAME = 'invexp_kexp'
    # Instance fields - dataclass automatically generates __init__
    S: float
    K: float
    alpha: float
    
    # Class variables (not part of __init__)
    # PARAM_NAMES is auto-generated from dataclass fields
    
    DEFAULT_BOUNDS: ClassVar[tuple] = (
        [1e10, 2e-6, 0.15],      # Lower: S, K, alpha
        [1e20, 3e-4, 0.55]       # Upper: S, K, alpha (留约10x空间)
    )
    
    DEFAULT_P0: ClassVar[list] = [1e15, 2.45e-5, 0.33]  # Initial guess based on power-law fit
    
    @classmethod
    def model(cls, data, *args):
        """
        Args:
            data: (n, x) tuple of arrays
            *args: S, K, alpha
        """
        n, x = data
        S, K, alpha = args
        
        # Convert to arrays for vectorized operations
        N = np.atleast_1d(n)
        x = np.atleast_1d(x)
        
        # Calculate k(N) = K * N^alpha for each data point
        k_N = K * (N ** alpha)
        
        # Apply formula: y = k(N) * log10(S / x)
        result = k_N * np.log10(S / x)
        
        return result

