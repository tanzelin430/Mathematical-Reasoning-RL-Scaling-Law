import numpy as np
from dataclasses import dataclass
from typing import ClassVar
from src.fit.base import BaseFitter

@dataclass
class LogLinearTau(BaseFitter):
    """
    Log-linear lookup table model for tau (training steps):
    y = log(L) = -k(tau) * log(x) + E0(tau)
    where k(tau) and E0(tau) are lookup tables for each tau value.
    tau values: [1, 2, 5, 20, 25, 50, 100]
    """
    MODEL_NAME = 'loglinear-tau'

    # Instance fields - dataclass automatically generates __init__
    k1: float
    k2: float
    k5: float
    k20: float
    k25: float
    k50: float
    k100: float
    E0_1: float
    E0_2: float
    E0_5: float
    E0_20: float
    E0_25: float
    E0_50: float
    E0_100: float
    
    DEFAULT_BOUNDS: ClassVar[tuple] = (
        [0.0] * 7 + [-10.0] * 7,  # Lower: k >= 0, E0 >= -10
        [1.0] * 7 + [10.0] * 7      # Upper: k <= 1, E0 <= 10
    )
    
    DEFAULT_P0: ClassVar[list] = [0.1] * 7 + [0.1] * 7  # Initial guess: k=0.1, E0=0.1
    
    @staticmethod
    def _build_lookup(k1, k2, k5, k20, k25, k50, k100,
                      E0_1, E0_2, E0_5, E0_20, E0_25, E0_50, E0_100):
        """Build lookup table from parameters. Reused by both __post_init__ and model()."""
        return {
            'k': {
                1: k1, 2: k2, 5: k5, 20: k20,
                25: k25, 50: k50, 100: k100
            },
            'E0': {
                1: E0_1, 2: E0_2, 5: E0_5, 20: E0_20,
                25: E0_25, 50: E0_50, 100: E0_100
            }
        }
    
    @classmethod
    def model(cls, data, *args):
        """
        Args:
            data: (tau, x) tuple of arrays
            *args: k1, k2, k5, k20, k25, k50, k100, 
                   E0_1, E0_2, E0_5, E0_20, E0_25, E0_50, E0_100
        """
        tau, x = data
        
        # Build lookup table using shared method
        lookup = cls._build_lookup(*args)
        
        # Convert to arrays for vectorized operations
        tau = np.atleast_1d(tau)
        x = np.atleast_1d(x)
        
        # Initialize output array
        result = np.zeros_like(tau, dtype=float)
        
        # Process each unique tau value
        unique_tau_values = np.unique(tau)
        for tau_val in unique_tau_values:
            # Find closest tau in lookup table
            closest_tau = min(lookup['k'].keys(), key=lambda t: abs(t - tau_val))
            if abs(closest_tau - tau_val) > 0.5:  # Tolerance: 0.5
                raise ValueError(f"tau={tau_val} not in lookup table (closest: {closest_tau})")
            
            # Get parameters for this tau
            k = lookup['k'][closest_tau]
            E0 = lookup['E0'][closest_tau]
            
            # Apply formula to all points with this tau value
            mask = (tau == tau_val)
            result[mask] = np.power(10.0, -k * np.log10(x[mask]) + E0)
        
        return result

