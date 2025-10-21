import numpy as np
from src.fit.base import BaseFitter


class InvExp(BaseFitter):
    """
    Inverse exponential model: L=(X_0/C)^k(N)
    y = log(L) = k(x0) * log(S/x1)
    where k(x0) is a lookup table for each x0 value and S is a constant.
    x0: N; x1: E, C
    """
    
    PARAM_NAMES = ['S', 'k0_5', 'k1_5', 'k3', 'k7', 'k14', 'k32', 'k72']
    
    N_VALUES = [0.5e9, 1.5e9, 3e9, 7e9, 14e9, 32e9, 72e9]
    
    # Configuration for get_params_array (for plotting/analysis)
    PARAM_ARRAY_CONFIG = {
        'curve_values': [0.5e9, 1.5e9, 3e9, 7e9, 14e9, 32e9, 72e9],
        'param_groups': {
            'k': ['k0_5', 'k1_5', 'k3', 'k7', 'k14', 'k32', 'k72'],
            'S': ['S']
        }
    }
    
    # Default fitting parameters (8 params: 1 S + 7 k's)
    DEFAULT_BOUNDS = (
        [1e0] + [1e-6] * 7,   # Lower
        [1e20] + [1] * 7   # Upper
    )
    
    DEFAULT_P0 = [1e15] + [0.1] * 7  # Initial guess: S=1e6, k=0.5
    
    def __init__(self, S, k0_5, k1_5, k3, k7, k14, k32, k72):
        # Store S constant
        self.S = S
        
        # Store as lookup tables for efficient prediction with tolerance
        self.k_lookup = {
            0.5e9: k0_5, 1.5e9: k1_5, 3e9: k3, 7e9: k7,
            14e9: k14, 32e9: k32, 72e9: k72
        }
        
        # Also store as individual attributes for JSON serialization (via PARAM_NAMES)
        self.k0_5, self.k1_5, self.k3, self.k7 = k0_5, k1_5, k3, k7
        self.k14, self.k32, self.k72 = k14, k32, k72
    
    @staticmethod
    def model(data, S, k0_5, k1_5, k3, k7, k14, k32, k72):
        """Model function for CMA-ES curve fitting."""
        fitter = InvExp(S, k0_5, k1_5, k3, k7, k14, k32, k72)
        return fitter.predict(data)

    def predict(self, data):
        """
        Predict y for given (N, x) values.
        Supports both single values and arrays.
        
        Formula: y = (S/x)^k(N)
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
            
            # Apply formula to all points with this N value: y = (S/x)^k(N)
            mask = (x0 == N_val)
            result[mask] = k * np.log10(self.S / x1[mask])
        
        return result

