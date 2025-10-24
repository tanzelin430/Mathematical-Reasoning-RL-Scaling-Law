import numpy as np
from dataclasses import dataclass
from typing import ClassVar
from src.fit.base import BaseFitter

@dataclass
class InvExpKQuadLog(BaseFitter):
    """
    Inverse exponential model with quadratic-logarithmic k(N): y = k(N) * log10(S/x)
    where k(N) = a*log10(N)^2 + b*log10(N) + c (quadratic in log10(N))
    
    Parameters:
    - S: scaling constant
    - a: quadratic coefficient
    - b: linear coefficient  
    - c: constant term

    Note: fitting result shows a->0, so it is equivalent to linear model.
    """
    MODEL_NAME = 'invexp_kquadlog'
    # Instance fields - dataclass automatically generates __init__
    S: float
    a: float
    b: float
    c: float
    
    DEFAULT_BOUNDS: ClassVar[tuple] = (
        [1e14, 0.0, -0.1, -0.5],      # Lower: S, a, b, c
        [1e18, 0.01, 0.1, 0.5]        # Upper: S, a, b, c
    )
    
    DEFAULT_P0: ClassVar[list] = [1.56e16, 0.0029, -0.016, -0.077]  # Initial guess from analysis
    
    @staticmethod
    def _build_lookup(S, a, b, c):
        """Build lookup table from parameters."""
        N_values = [0.5e9, 1.5e9, 3e9, 7e9, 14e9, 32e9, 72e9]
        return {
            'k': {N: a * np.log10(N)**2 + b * np.log10(N) + c 
                  for N in N_values}
        }
    
    @classmethod
    def model(cls, data, *args):
        """
        Args:
            data: (n, x) tuple of arrays
            *args: S, a, b, c
        """
        n, x = data
        S, a, b, c = args
        
        N = np.atleast_1d(n)
        x = np.atleast_1d(x)
        
        # Calculate k(N) = a*log10(N)^2 + b*log10(N) + c
        log10_N = np.log10(N)
        k_N = a * log10_N**2 + b * log10_N + c
        
        # Apply formula: y = k(N) * log10(S / x)
        result = k_N * np.log10(S / x)
        return result



