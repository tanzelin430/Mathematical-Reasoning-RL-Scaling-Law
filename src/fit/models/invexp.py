import numpy as np
from dataclasses import dataclass
from typing import ClassVar
from src.fit.base import BaseFitter


@dataclass
class InvExp(BaseFitter):
    """
    Inverse exponential model: y=(S/x)^k(n)
    where k(n) is a lookup table for each n value and S is a constant.
    """
    MODEL_NAME = 'invexp'
    # Instance fields - dataclass automatically generates __init__
    S: float
    k0_5: float
    k1_5: float
    k3: float
    k7: float
    k14: float
    k32: float
    k72: float
    
    DEFAULT_BOUNDS: ClassVar[tuple] = (
        [1e0] + [1e-6] * 7,   # Lower
        [1e20] + [1] * 7   # Upper
    )
    
    DEFAULT_P0: ClassVar[list] = [1e15] + [0.1] * 7  # Initial guess: S=1e15, k=0.1
    
    @staticmethod
    def _build_lookup(S, k0_5, k1_5, k3, k7, k14, k32, k72):
        """Build lookup table from parameters."""
        return {
            'k': {
                0.5e9: k0_5, 1.5e9: k1_5, 3e9: k3, 7e9: k7,
                14e9: k14, 32e9: k32, 72e9: k72
            }
        }
    
    @classmethod
    def model(cls, data, *args):
        """
        Args:
            data: (n, x) tuple of arrays
            *args: S, k0_5, k1_5, k3, k7, k14, k32, k72
        """
        n, x = data
        S = args[0]
        
        # Build lookup table
        lookup = cls._build_lookup(*args)
        
        # Convert to arrays for vectorized operations
        n = np.atleast_1d(n)
        x = np.atleast_1d(x)
        
        # Initialize output array
        result = np.zeros_like(n, dtype=float)
        
        # Group by n
        unique_n_values = np.unique(n)
        for n_val in unique_n_values:
            # Find closest n in lookup table
            closest_n = min(lookup['k'].keys(), key=lambda n: abs(n - n_val))
            if abs(closest_n - n_val) > 1:  # Tolerance: 1
                raise ValueError(f"n={n_val/1e9:.1f}B not in lookup table (closest: {closest_n/1e9:.1f}B)")
            
            # Get parameters for this n
            k = lookup['k'][closest_n]
            
            # Apply formula to all points with this N value: y = (S/x)^k(N)
            mask = (n == n_val)
            result[mask] = (S / x[mask]) ** k
        
        return result

