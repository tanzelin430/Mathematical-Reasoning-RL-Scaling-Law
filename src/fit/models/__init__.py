"""
Fit Models Registry

This module provides a registry of all available fit models and utilities
to dynamically access them by name.
"""

from typing import Type
from .loglinear import LogLinear
from .invexp import InvExp
from .invexp_klinear import InvExpKLinear
from .invexp_kexp import InvExpKExp
from .invexp_kquadlog import InvExpKQuadLog


# Model Registry - maps model names to their classes
MODEL_REGISTRY = {
    'LogLinear': LogLinear,
    'InvExp': InvExp,
    'InvExpKLinear': InvExpKLinear,
    'InvExpKQuadLog': InvExpKQuadLog,
    'InvExpKExp': InvExpKExp,
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
