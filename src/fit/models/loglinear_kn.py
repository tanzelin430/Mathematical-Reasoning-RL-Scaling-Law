import numpy as np
from dataclasses import dataclass
from typing import ClassVar
from src.fit.base import BaseFitter

@dataclass
class LogLinearKn(BaseFitter):
    """
    Log-linear model with k(n) function:
    log(y) = -k(n) * log(x) + E0(n)
    where k(n) = k_max * N / (N + N0)
    and E0(n) are lookup tables for each n value.
    """
    MODEL_NAME = 'loglinear_kn'

    # Parameters for k(n) function
    k_max: float
    N0: float
    
    # E0 lookup table parameters
    E0_0_5: float
    E0_1_5: float
    E0_3: float
    E0_7: float
    E0_14: float
    E0_32: float
    E0_72: float
    
    DEFAULT_BOUNDS: ClassVar[tuple] = (
        [0.0, 0.1e9] + [-10.0] * 7,  # Lower: k_max >= 0, N0 >= 0.1B, E0 >= -10
        [2.0, 100e9] + [10.0] * 7     # Upper: k_max <= 2, N0 <= 100B, E0 <= 10
    )
    
    DEFAULT_P0: ClassVar[list] = [0.5, 7e9] + [0.1] * 7  # Initial guess: k_max=0.5, N0=7B, E0=0.1
    
    @staticmethod
    def _build_lookup(k_max, N0, E0_0_5, E0_1_5, E0_3, E0_7, E0_14, E0_32, E0_72):
        """Build lookup table from parameters. Reused by both __post_init__ and model()."""
        return {
            'k_max': k_max,
            'N0': N0,
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
            *args: k_max, N0, E0_0_5, E0_1_5, E0_3, E0_7, E0_14, E0_32, E0_72
        """
        n, x = data
        
        # Build lookup table using shared method
        lookup = cls._build_lookup(*args)
        k_max = lookup['k_max']
        N0 = lookup['N0']
        
        # Convert to arrays for vectorized operations
        n = np.atleast_1d(n)
        x = np.atleast_1d(x)
        
        # Initialize output array
        result = np.zeros_like(n, dtype=float)
        
        # Process each unique N value
        unique_N_values = np.unique(n)
        for N_val in unique_N_values:
            # Find closest N in lookup table for E0
            closest_N = min(lookup['E0'].keys(), key=lambda n: abs(n - N_val))
            if abs(closest_N - N_val) > 1:  # Tolerance: 1
                raise ValueError(f"N={N_val/1e9:.1f}B not in lookup table (closest: {closest_N/1e9:.1f}B)")
            
            # Calculate k(n) using the function k_max * N / (N + N0)
            k_n = k_max * N_val / (N_val + N0)
            
            # Get E0 for this N
            E0 = lookup['E0'][closest_N]
            
            # Apply formula to all points with this N value
            mask = (n == N_val)
            result[mask] = np.power(10.0, -k_n * np.log10(x[mask]) + E0)
        
        return result

    def get_k_function_params(self):
        """Return k(n) function parameters for analysis."""
        return {
            'k_max': self.k_max,
            'N0': self.N0
        }
    
    def get_lookup_params(self) -> dict:
        """
        Override to return parameters in compatible format for plotting.
        
        For loglinear_kn, we compute k(n) values for each model size and 
        return both k and E0 as lookup tables like the original loglinear model.
        """
        if not hasattr(self, 'lookup') or self.lookup is None:
            return None
        
        # Model sizes from E0 lookup table
        model_sizes = sorted(self.lookup['E0'].keys())
        
        # Compute k(n) for each model size using k(n) = k_max * N / (N + N0)
        k_computed = {}
        for N in model_sizes:
            k_n = self.k_max * N / (N + self.N0)
            k_computed[N] = k_n
        
        # Return in the format expected by plotting code
        return {
            'k': k_computed,
            'E0': self.lookup['E0']
        }