import numpy as np
from src.fit.base import BaseFitter

class InvExpKExp(BaseFitter):
    """
    Inverse exponential model with power-law k(N): y = k(N) * log10(S/x)
    where k(N) = K * N^alpha (power-law relationship with N)
    
    Parameters:
    - S: scaling constant
    - K: power-law coefficient
    - alpha: power-law exponent for k(N) = K * N^alpha
    """
    
    PARAM_NAMES = ['S', 'K', 'alpha']
    
    N_VALUES = [0.5e9, 1.5e9, 3e9, 7e9, 14e9, 32e9, 72e9]
    
    # Configuration for get_params_array (for plotting/analysis)
    PARAM_ARRAY_CONFIG = {
        'curve_values': [0.5e9, 1.5e9, 3e9, 7e9, 14e9, 32e9, 72e9],
        'param_groups': {
            'k': ['K', 'alpha'],  # k(N) = K * N^alpha
            'S': ['S']
        }
    }
    
    # Default fitting parameters (3 params: S, K, alpha)
    # Based on power-law fit: k(N) = 2.45e-05 * N^0.3324, R² = 0.909
    DEFAULT_BOUNDS = (
        [1e10, 2e-6, 0.15],      # Lower: S, K, alpha
        [1e20, 3e-4, 0.55]       # Upper: S, K, alpha (留约10x空间)
    )
    
    DEFAULT_P0 = [1e15, 2.45e-5, 0.33]  # Initial guess based on power-law fit
    
    def __init__(self, S, K, alpha):
        """
        Initialize the model.
        
        Args:
            S: scaling constant
            K: power-law coefficient
            alpha: power-law exponent for k(N) = K * N^alpha
        """
        self.S = S
        self.K = K
        self.alpha = alpha
    
    @staticmethod
    def model(data, S, K, alpha):
        """Model function for CMA-ES curve fitting."""
        fitter = InvExpKExp(S, K, alpha)
        return fitter.predict(data)

    def predict(self, data):
        """
        Predict y for given (N, x) values.
        Supports both single values and arrays.
        
        Formula: y = k(N) * log10(S/x), where k(N) = K * N^alpha
        """
        x0, x1 = data
        
        # Convert to arrays for vectorized operations
        N = np.atleast_1d(x0)
        x = np.atleast_1d(x1)
        
        # Calculate k(N) = K * N^alpha for each data point
        k_N = self.K * (N ** self.alpha)
        
        # Apply formula: y = k(N) * log10(S / x)
        result = k_N * np.log10(self.S / x)
        
        return result
    
    def get_k_at_N(self, N):
        """
        Helper method to get k value at specific N.
        
        Args:
            N: model size (can be scalar or array)
            
        Returns:
            k(N) = K * N^alpha
        """
        return self.K * (np.atleast_1d(N) ** self.alpha)


