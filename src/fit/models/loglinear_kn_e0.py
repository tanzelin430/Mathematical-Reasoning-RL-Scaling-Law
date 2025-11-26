import numpy as np
from dataclasses import dataclass
from typing import ClassVar
from src.fit.base import BaseFitter

@dataclass
class LogLinearKnE0(BaseFitter):
    """
    Log-linear model with both k(N) and E0(N) parameterized:

    log(y) = -k(N) * log(x) + E0(N)

    where:
        k(N) = k_max * N / (N + N0)
        E0(N) = E0_max * N / (N + N1)

    This reduces parameters from 9 (k_max, N0, E0_0.5, ..., E0_72)
    to just 4 (k_max, N0, E0_max, N1).
    """
    MODEL_NAME = 'loglinear_kn_e0'

    # Parameters for k(N) function
    k_max: float
    N0: float

    # Parameters for E0(N) function
    E0_max: float
    N1: float

    DEFAULT_BOUNDS: ClassVar[tuple] = (
        [0.0, 0.1e9, -10.0, 0.1e9],   # Lower: k_max>=0, N0>=0.1B, E0_max>=-10, N1>=0.1B
        [2.0, 100e9, 10.0, 100e9]      # Upper: k_max<=2, N0<=100B, E0_max<=10, N1<=100B
    )

    DEFAULT_P0: ClassVar[list] = [0.5, 7e9, 2.0, 15e9]  # k_max, N0, E0_max, N1

    @classmethod
    def model(cls, data, k_max, N0, E0_max, N1):
        """
        Args:
            data: (n, x) tuple of arrays
            k_max: Maximum k value
            N0: Half-saturation point for k(N)
            E0_max: Maximum E0 value
            N1: Half-saturation point for E0(N)
        """
        n, x = data

        # Convert to arrays for vectorized operations
        n = np.atleast_1d(n)
        x = np.atleast_1d(x)

        # Calculate k(N) and E0(N) for each data point
        k_N = k_max * n / (n + N0)
        E0_N = E0_max * n / (n + N1)

        # Apply formula: L = 10^(-k(N) * log10(x) + E0(N))
        result = 10 ** (-k_N * np.log10(x) + E0_N)

        return result

    def get_k_function_params(self):
        """Return k(N) function parameters for analysis."""
        return {
            'k_max': self.k_max,
            'N0': self.N0
        }

    def get_e0_function_params(self):
        """Return E0(N) function parameters for analysis."""
        return {
            'E0_max': self.E0_max,
            'N1': self.N1
        }

    def get_lookup_params(self) -> dict:
        """
        Generate lookup table values from parametric functions.
        This allows compatibility with plotting code expecting lookup tables.
        """
        # Standard model sizes
        model_sizes = [0.5e9, 1.5e9, 3e9, 7e9, 14e9, 32e9, 72e9]

        # Compute k(N) and E0(N) for each model size
        k_computed = {}
        E0_computed = {}

        for N in model_sizes:
            k_N = self.k_max * N / (N + self.N0)
            E0_N = self.E0_max * N / (N + self.N1)

            k_computed[N] = k_N
            E0_computed[N] = E0_N

        return {
            'k': k_computed,
            'E0': E0_computed
        }
