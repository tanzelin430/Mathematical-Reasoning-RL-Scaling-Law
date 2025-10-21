import numpy as np
from src.fit.base import BaseFitter

class LogLinear(BaseFitter):
    """
    Log-linear lookup table model:
    y = log(L) = -k(x0) * log(x1) + E0(x0)
    where k(x0) and E0(x0) are lookup tables for each x0 value.
    x0: N; x1: E, C
    """
    
    PARAM_NAMES = ['k0_5', 'k1_5', 'k3', 'k7', 'k14', 'k32', 'k72', 
                   'E0_0_5', 'E0_1_5', 'E0_3', 'E0_7', 'E0_14', 'E0_32', 'E0_72']
    
    N_VALUES = [0.5e9, 1.5e9, 3e9, 7e9, 14e9, 32e9, 72e9]
    
    # Configuration for get_params_array (for plotting/analysis)
    PARAM_ARRAY_CONFIG = {
        'curve_values': [0.5e9, 1.5e9, 3e9, 7e9, 14e9, 32e9, 72e9],
        'param_groups': {
            'k': ['k0_5', 'k1_5', 'k3', 'k7', 'k14', 'k32', 'k72'],
            'E0': ['E0_0_5', 'E0_1_5', 'E0_3', 'E0_7', 'E0_14', 'E0_32', 'E0_72']
        }
    }
    
    # Default fitting parameters (14 params: 7 k's + 7 E0's)
    DEFAULT_BOUNDS = (
        [0.0] * 7 + [-10.0] * 7,  # Lower: k >= 0, E0 >= -10
        [1.0] * 7 + [10.0] * 7      # Upper: k <= 1, E0 <= 0
    )
    
    DEFAULT_P0 = [0.1] * 7 + [0.1] * 7  # Initial guess: k=0.1, E0=-1.0
    
    def __init__(self, k0_5, k1_5, k3, k7, k14, k32, k72, 
                 E0_0_5, E0_1_5, E0_3, E0_7, E0_14, E0_32, E0_72):
        # Store as lookup tables for efficient prediction with tolerance
        self.k_lookup = {
            0.5e9: k0_5, 1.5e9: k1_5, 3e9: k3, 7e9: k7,
            14e9: k14, 32e9: k32, 72e9: k72
        }
        self.E0_lookup = {
            0.5e9: E0_0_5, 1.5e9: E0_1_5, 3e9: E0_3, 7e9: E0_7,
            14e9: E0_14, 32e9: E0_32, 72e9: E0_72
        }
        
        # Also store as individual attributes for JSON serialization (via PARAM_NAMES)
        self.k0_5, self.k1_5, self.k3, self.k7 = k0_5, k1_5, k3, k7
        self.k14, self.k32, self.k72 = k14, k32, k72
        self.E0_0_5, self.E0_1_5, self.E0_3, self.E0_7 = E0_0_5, E0_1_5, E0_3, E0_7
        self.E0_14, self.E0_32, self.E0_72 = E0_14, E0_32, E0_72
    
    @staticmethod
    def model(data, k0_5, k1_5, k3, k7, k14, k32, k72, 
              E0_0_5, E0_1_5, E0_3, E0_7, E0_14, E0_32, E0_72):
        """Model function for CMA-ES curve fitting."""
        fitter = LogLinear(k0_5, k1_5, k3, k7, k14, k32, k72,
                              E0_0_5, E0_1_5, E0_3, E0_7, E0_14, E0_32, E0_72)
        return fitter.predict(data)

    def predict(self, data):
        """
        Predict log10(errrate) for given (N, E) values.
        Supports both single values and arrays.
        
        Formula: log_errrate = -k(N) * log10(E) + E0(N)
        """
        x0, x1 = data
        
        # Convert to arrays for vectorized operations
        x0 = np.atleast_1d(x0)
        x1 = np.atleast_1d(x1)
        
        # Initialize output array
        result = np.zeros_like(x0, dtype=float)
        
        # Process each unique N value
        unique_N_values = np.unique(x0)
        for N_val in unique_N_values:
            # Find closest N in lookup table
            closest_N = min(self.k_lookup.keys(), key=lambda n: abs(n - N_val))
            if abs(closest_N - N_val) > 1:  # Tolerance: 1
                raise ValueError(f"N={N_val/1e9:.1f}B not in lookup table (closest: {closest_N/1e9:.1f}B)")
            
            # Get parameters for this N
            k = self.k_lookup[closest_N]
            E0 = self.E0_lookup[closest_N]
            
            # Apply formula to all points with this N value
            mask = (x0 == N_val)
            result[mask] = -k * np.log10(x1[mask]) + E0
        
        return result

