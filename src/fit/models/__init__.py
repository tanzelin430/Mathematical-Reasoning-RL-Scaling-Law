"""
Fit Models Registry

This module provides a registry of all available fit models and utilities
to dynamically access them by name.
"""

import inspect
from typing import Type
from src.fit.base import BaseFitter
# don't delete, used in MODEL_REGISTRY
from .loglinear import LogLinear
from .invexp import InvExp
from .invexp_klinear import InvExpKLinear
from .invexp_kexp import InvExpKExp
from .invexp_kquadlog import InvExpKQuadLog
from .powlaw import PowLaw
from .powlaw2 import PowLaw2
from .powlaw3 import PowLaw3
from .powlaw4 import PowLaw4
from .powlawmul import PowLawMul
from .powlawmul_lstar import PowLawMulLStar
from .powlawplus import PowLawPlus
from .loglinear_tau import LogLinearTau

MODEL_REGISTRY = {
    cls.MODEL_NAME: cls
    for name, cls in globals().items()
    if (inspect.isclass(cls) and 
        issubclass(cls, BaseFitter) and 
        cls is not BaseFitter)
}

def get_model_class(model_name: str) -> Type:
    """
    Get a fit model class by name.
    
    Args:
        model_name: Name of the model (e.g., 'InvExp')
        
    Returns:
        The model class
        
    Raises:
        ValueError: If model_name is not found in the registry
        
    Example:
        >>> FitterClass = get_model_class('InvExp')
        >>> fitter = FitterClass(...)
    """
    if model_name not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Model '{model_name}' not found. "
            f"Available models: {available}"
        )
    return MODEL_REGISTRY[model_name]


def list_available_models():
    """Return a list of all available model names."""
    return list(MODEL_REGISTRY.keys())
