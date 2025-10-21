import numpy as np
from src.fit.base import BaseFitter

class InvExpKLinear(BaseFitter):
    """
    Inverse exponential model with affine-logarithmic k(N): y = k(N) * log10(S/x)
    where k(N) = K * log10(N) + b (logarithmic with intercept)
    
    Parameters:
    - S: scaling constant
    - K: logarithmic coefficient
    - b: intercept for k(N) = K * log10(N) + b
    """
    
    PARAM_NAMES = ['S', 'K', 'b']
    
    N_VALUES = [0.5e9, 1.5e9, 3e9, 7e9, 14e9, 32e9, 72e9]
    
    # Configuration for get_params_array (for plotting/analysis)
    PARAM_ARRAY_CONFIG = {
        'curve_values': [0.5e9, 1.5e9, 3e9, 7e9, 14e9, 32e9, 72e9],
        'param_groups': {
            'k': ['K', 'b'],  # k(N) = K * log10(N) + b
            'S': ['S']
        }
    }
    
    # Default fitting parameters (3 params: S, K, b)
    # Based on CMA-ES fit: S=1.56e16, K=0.0411, b=-0.354, R²=0.971
    DEFAULT_BOUNDS = (
        [1e14, 0.005, -0.8],      # Lower: S, K, b (更宽松的范围)
        [1e18, 0.15, 0.2]         # Upper: S, K, b
    )
    
    DEFAULT_P0 = [1.56e16, 0.041, -0.35]  # Initial guess based on CMA-ES fit
    
    def __init__(self, S, K, b):
        """
        Initialize the model.
        
        Args:
            S: scaling constant
            K: logarithmic coefficient
            b: intercept for k(N) = K * log10(N) + b
        """
        self.S = S
        self.K = K
        self.b = b
    
    @staticmethod
    def model(data, S, K, b):
        """Model function for CMA-ES curve fitting."""
        fitter = InvExpKLinear(S, K, b)
        return fitter.predict(data)

    def predict(self, data):
        """
        Predict y for given (N, x) values.
        Supports both single values and arrays.
        
        Formula: y = k(N) * log10(S/x), where k(N) = K * log10(N) + b
        """
        x0, x1 = data
        
        # Convert to arrays for vectorized operations
        N = np.atleast_1d(x0)
        x = np.atleast_1d(x1)
        
        # Calculate k(N) using the helper method
        k_N = self.get_k_at_N(N)
        
        # Apply formula: y = k(N) * log10(S / x)
        result = k_N * np.log10(self.S / x)
        
        return result
    
    def get_k_at_N(self, N):
        """
        Helper method to get k value at specific N.
        
        Args:
            N: model size (can be scalar or array)
            
        Returns:
            k(N) = K * log10(N) + b
        """
        return self.K * np.log10(np.atleast_1d(N)) + self.b


