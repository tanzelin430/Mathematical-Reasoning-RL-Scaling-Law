import numpy as np
from dataclasses import dataclass
from typing import ClassVar
from src.fit.base import BaseFitter

@dataclass
class LogLinear(BaseFitter):
    """
    Log-linear lookup table model:
    y = log(L) = -k(n) * log(x) + E0(n)
    where k(n) and E0(n) are lookup tables for each n value.
    """
    MODEL_NAME = 'loglinear'

    # Instance fields - dataclass automatically generates __init__
    k0_5: float
    k1_5: float
    k3: float
    k7: float
    k14: float
    k32: float
    k72: float
    E0_0_5: float
    E0_1_5: float
    E0_3: float
    E0_7: float
    E0_14: float
    E0_32: float
    E0_72: float
    
    DEFAULT_BOUNDS: ClassVar[tuple] = (
        [0.0] * 7 + [-10.0] * 7,  # Lower: k >= 0, E0 >= -10
        [1.0] * 7 + [10.0] * 7      # Upper: k <= 1, E0 <= 0
    )
    
    DEFAULT_P0: ClassVar[list] = [0.1] * 7 + [0.1] * 7  # Initial guess: k=0.1, E0=-1.0
    
    @staticmethod
    def _build_lookup(k0_5, k1_5, k3, k7, k14, k32, k72,
                      E0_0_5, E0_1_5, E0_3, E0_7, E0_14, E0_32, E0_72):
        """Build lookup table from parameters. Reused by both __post_init__ and model()."""
        return {
            'k': {
                0.5e9: k0_5, 1.5e9: k1_5, 3e9: k3, 7e9: k7,
                14e9: k14, 32e9: k32, 72e9: k72
            },
            'E0': {
                0.5e9: E0_0_5, 1.5e9: E0_1_5, 3e9: E0_3, 7e9: E0_7,
                14e9: E0_14, 32e9: E0_32, 72e9: E0_72
            }
        }
    
    @classmethod
    def model(cls, data, *args):
        """
        Args:
            data: (n, x) tuple of arrays
            *args: k0_5, k1_5, k3, k7, k14, k32, k72, 
                   E0_0_5, E0_1_5, E0_3, E0_7, E0_14, E0_32, E0_72
        """
        n, x = data
        
        # Build lookup table using shared method
        lookup = cls._build_lookup(*args)
        
        # Convert to arrays for vectorized operations
        n = np.atleast_1d(n)
        x = np.atleast_1d(x)
        
        # Initialize output array
        result = np.zeros_like(n, dtype=float)
        
        # Process each unique N value
        unique_N_values = np.unique(n)
        for N_val in unique_N_values:
            # Find closest N in lookup table
            closest_N = min(lookup['k'].keys(), key=lambda n: abs(n - N_val))
            if abs(closest_N - N_val) > 1:  # Tolerance: 1
                raise ValueError(f"N={N_val/1e9:.1f}B not in lookup table (closest: {closest_N/1e9:.1f}B)")
            
            # Get parameters for this N
            k = lookup['k'][closest_N]
            E0 = lookup['E0'][closest_N]
            
            # Apply formula to all points with this N value
            mask = (n == N_val)
            result[mask] = np.power(10.0, -k * np.log10(x[mask]) + E0)
        
        return result

