import numpy as np
from src.fit.base import BaseFitter

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
    
    PARAM_NAMES = ['S', 'a', 'b', 'c']
    
    N_VALUES = [0.5e9, 1.5e9, 3e9, 7e9, 14e9, 32e9, 72e9]
    
    # Configuration for get_params_array (for plotting/analysis)
    PARAM_ARRAY_CONFIG = {
        'curve_values': [0.5e9, 1.5e9, 3e9, 7e9, 14e9, 32e9, 72e9],
        'param_groups': {
            'k': ['a', 'b', 'c'],  # k(N) = a*log10(N)^2 + b*log10(N) + c
            'S': ['S']
        }
    }
    
    # Default fitting parameters (4 params: S, a, b, c)
    # Based on quadratic-log fit to lookup table k values: R² = 0.960
    DEFAULT_BOUNDS = (
        [1e14, 0.0, -0.1, -0.5],      # Lower: S, a, b, c
        [1e18, 0.01, 0.1, 0.5]        # Upper: S, a, b, c
    )
    
    DEFAULT_P0 = [1.56e16, 0.0029, -0.016, -0.077]  # Initial guess from analysis
    
    def __init__(self, S, a, b, c):
        """
        Initialize the model.
        
        Args:
            S: scaling constant
            a: quadratic coefficient for log10(N)^2
            b: linear coefficient for log10(N)
            c: constant term
        """
        self.S = S
        self.a = a
        self.b = b
        self.c = c
    
    @staticmethod
    def model(data, S, a, b, c):
        """Model function for CMA-ES curve fitting."""
        fitter = InvExpKQuadLog(S, a, b, c)
        return fitter.predict(data)

    def predict(self, data):
        """
        Predict y for given (N, x) values.
        Supports both single values and arrays.
        
        Formula: y = k(N) * log10(S/x), where k(N) = a*log10(N)^2 + b*log10(N) + c
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
            k(N) = a*log10(N)^2 + b*log10(N) + c
        """
        log10_N = np.log10(np.atleast_1d(N))
        return self.a * log10_N**2 + self.b * log10_N + self.c



