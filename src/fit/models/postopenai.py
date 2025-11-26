import numpy as np
from dataclasses import dataclass
from typing import ClassVar
from src.fit.base import BaseFitter


@dataclass
class PostOpenAINRef(BaseFitter):
    f"""
    Post-OpenAI scaling law model with functional B0 and C0:
    L(N, C) = [(N0/N)^β + B0(N)/(C+C0(N))]^α
    
    where:
        N0      : saturation scale for N (global parameter)
        β       : power exponent for the N-dependent term (global parameter)
        α       : outer power exponent (global parameter)
        N_ref   : reference N value for B0 and C0 scaling
        B0(N)   : B_max + B0 * (N_ref/N)^p_B
        C0(N)   : C_max + C0 * (N_ref/N)^p_C
    
    Parameters:
        n: N (model parameters, e.g., 1e9, 3e9, 7e9)
        x: C (compute)
    """
    MODEL_NAME = 'postopenai_Nref'
    # Instance fields - dataclass automatically generates __init__
    N0: float
    beta: float
    alpha: float
    N_ref: float
    B_max: float
    B0: float
    p_B: float
    C_max: float
    C0: float
    p_C: float
    
    DEFAULT_BOUNDS: ClassVar[tuple] = (
        # N0,     β,      α,      N_ref,  B_max,  B0,     p_B,    C_max,  C0,     p_C
       [2e8, 0.8, 0.1, 1e8,   0,   1e3, 0.0, 0,   1e3,  0.0],      # Lower
       [5e8, 1.8, 0.3, 1e10, 1e4,  1e6, 3.0, 1e5,  5e6, 3.0]       # Upper
    )

    DEFAULT_P0: ClassVar[list] = [3.7e8, 1.02, 0.198, 3e9, 224, 4.62e4, 2.1, 3.42e3, 1.54e5, 1.72]
    
    @staticmethod
    def _compute_B0(N, N_ref, B_max, B0, p_B):
        """
        Compute B0(N) = B_max + B0 * (N_ref/N)^p_B
        """
        return B_max + B0 * np.power(N_ref / N, p_B)
    
    @staticmethod
    def _compute_C0(N, N_ref, C_max, C0, p_C):
        """
        Compute C0(N) = C_max + C0 * (N_ref/N)^p_C
        """
        return C_max + C0 * np.power(N_ref / N, p_C)

    @classmethod
    def model(cls, data, *args):
        """
        Args:
            data: (n, x) tuple of arrays
            *args: N0, beta, alpha, N_ref, B_max, B0, p_B, C_max, C0, p_C
        """
        n, x = data
        
        # Unpack parameters
        N0, beta, alpha, N_ref, B_max, B0, p_B, C_max, C0, p_C = args
        
        # Convert to arrays for vectorized operations
        n = np.atleast_1d(n)
        x = np.atleast_1d(x)
        
        # Compute B0(N) and C0(N) for all N values
        B0_N = cls._compute_B0(n, N_ref, B_max, B0, p_B)
        C0_N = cls._compute_C0(n, N_ref, C_max, C0, p_C)
        
        # Compute first term: (N0/N)^β
        term1 = np.power(N0 / n, beta)
        
        # Compute second term: B0(N)/(C+C0(N))
        term2 = B0_N / (x + C0_N)
        
        # Compute inner sum: [(N0/N)^β + B0(N)/(C+C0(N))]
        inner_sum = term1 + term2
        
        # Apply outer power: [...]^α
        # Add safety checks for numerical stability
        with np.errstate(over='raise', invalid='raise'):
            try:
                result = np.power(inner_sum, alpha)
            except (FloatingPointError, RuntimeWarning):
                # Handle potential overflow/underflow
                # Use log space for numerical stability
                log_result = alpha * np.log(inner_sum)
                result = np.exp(log_result)
        
        return result