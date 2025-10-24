import numpy as np
from dataclasses import dataclass
from typing import ClassVar
from src.fit.base import BaseFitter

@dataclass
class InvExpKLinear(BaseFitter):
    """
    Inverse exponential model with affine-logarithmic k(N): y = k(N) * log10(S/x)
    where k(N) = K * log10(N) + b (logarithmic with intercept)
    
    Parameters:
    - S: scaling constant
    - K: logarithmic coefficient
    - b: intercept for k(N) = K * log10(N) + b
    """
    MODEL_NAME = 'invexp_klinear'
    # Instance fields - dataclass automatically generates __init__
    S: float
    K: float
    b: float
    
    DEFAULT_BOUNDS: ClassVar[tuple] = (
        [1e14, 0.005, -0.8],      # Lower: S, K, b (更宽松的范围)
        [1e18, 0.15, 0.2]         # Upper: S, K, b
    )
    
    DEFAULT_P0: ClassVar[list] = [1.56e16, 0.041, -0.35]  # Initial guess based on CMA-ES fit
    
    @staticmethod
    def _build_lookup(S, K, b):
        """Build lookup table from parameters."""
        N_values = [0.5e9, 1.5e9, 3e9, 7e9, 14e9, 32e9, 72e9]
        return {
            'k': {N: K * np.log10(N) + b for N in N_values}
        }
    
    @classmethod
    def model(cls, data, *args):
        """
        Args:
            data: (n, x) tuple of arrays
            *args: S, K, b
        """
        n, x = data
        S, K, b = args
        
        N = np.atleast_1d(n)
        x = np.atleast_1d(x)
        
        # Calculate k(N) = K * log10(N) + b
        k_N = K * np.log10(N) + b
        
        # Apply formula: y = k(N) * log10(S / x)
        result = k_N * np.log10(S / x)
        return result


