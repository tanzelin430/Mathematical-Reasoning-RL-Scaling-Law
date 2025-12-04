import numpy as np
from typing import Dict, Tuple, List
from datetime import datetime
from abc import ABC, abstractmethod
from dataclasses import fields, is_dataclass
import json
import inspect
# ============================================================================
# Base Fitter Class - Provides common functionality for all fitters
# ============================================================================

class BaseFitter(ABC):
    """
    Base class for all fitting models. Provides core functionality:
    - save_to_json / load_from_json: automatic serialization based on PARAM_NAMES
    - get_lookup_params: generic method for extracting parameters for plotting/analysis
    - model: generic classmethod for CMA-ES curve fitting (subclasses don't need to override)
    
    Subclasses should define (as dataclass):
    - Use @dataclass decorator and define instance fields with type annotations
    - predict: core prediction logic (model-specific)
    - get_lookup_params: (optional) return lookup table parameters for plotting/analysis
    
    PARAM_NAMES is automatically extracted from dataclass fields.
    
    Note: DataFrame prediction is handled by business layer functions in fit.py
    """
    
    def __init_subclass__(cls, **kwargs):
        """Enforce MODEL_NAME definition in concrete subclasses."""
        super().__init_subclass__(**kwargs)
        if not inspect.isabstract(cls) and (not hasattr(cls, 'MODEL_NAME') or cls.MODEL_NAME is None):
                raise TypeError(f"{cls.__name__} must define MODEL_NAME class attribute.\n")

    @classmethod
    def get_param_names(cls) -> List[str]:
        """
        Get parameter names for this model.
        
        Automatically extracts field names from dataclass.
        
        Returns:
            List of parameter names in order
        """
        if not is_dataclass(cls):
            raise ValueError(f"{cls.__name__} must be a dataclass (use @dataclass decorator)")
        return [f.name for f in fields(cls)]
    
    @classmethod
    @abstractmethod
    def model(cls, data, *args):
        """
        Core model function for CMA-ES curve fitting.
        
        This method must be implemented by all subclasses. It should directly
        compute predictions from the given parameters without creating an instance.
        This avoids instance creation overhead during optimization.
        
        Args:
            data: Input data (typically tuple of arrays like (N, C))
            *args: Model parameters in the order defined by PARAM_NAMES
            
        Returns:
            np.ndarray: Predicted values
            
        Example:
            @classmethod
            def model(cls, data, E, A, B):
                n, x = data
                return E - A * np.log(x) + B
        """
        pass
    
    def predict(self, data):
        """
        Predict output for given input data using this instance's parameters.
        
        This method is generic and works for all subclasses by extracting
        parameters from the instance and calling the model method.
        
        Args:
            data: Input data, typically a tuple of arrays (e.g., (N, C) or (N, E))
                  where the first element is the curve parameter and the second
                  is the x-axis variable.
                  
        Returns:
            np.ndarray: Predicted values. Supports both single values and arrays.
        """
        # Extract parameters from instance in the correct order
        param_values = [getattr(self, name) for name in self.__class__.get_param_names()]
        return self.__class__.model(data, *param_values)
    
    def __post_init__(self):
        """
        Generic post-initialization for dataclasses.
        
        If the subclass implements _build_lookup(), this method will automatically
        extract parameters and build the lookup table. Otherwise, sets lookup to None.
        
        Also initializes CONTEXT and INFO (uppercase to avoid conflicts with model params).
        
        Subclasses can override this method if they need custom initialization.
        """
        if hasattr(self.__class__, '_build_lookup'):
            # Auto-extract parameters and build lookup
            param_values = [getattr(self, name) for name in self.get_param_names()]
            self.lookup = self.__class__._build_lookup(*param_values)
        else:
            self.lookup = None
        
        # Initialize context and info (uppercase to avoid param conflicts)
        self.CONTEXT = {}
        self.INFO = {}
    
    def get_lookup_params(self) -> dict:
        if not hasattr(self, 'lookup') or self.lookup is None:
            return None
        return self.lookup
    
    def set_context(self, context: dict):
        """Set fit context (configuration)."""
        self.CONTEXT = context
    
    def get_context(self) -> dict:
        """Get fit context (configuration)."""
        return self.CONTEXT
    
    def set_info(self, info: dict):
        """Set fit info (quality metrics)."""
        self.INFO = info
    
    def get_info(self) -> dict:
        """Get fit info (quality metrics)."""
        return self.INFO
    
    def _get_params_for_json(self) -> dict:
        """
        Get parameters as dict for JSON serialization.
        
        Automatically extracts parameters based on PARAM_NAMES.
        Handles dict with numeric keys by converting them to strings.
        """
        params = {}
        param_names = self.__class__.get_param_names()
        for name in param_names:
            value = getattr(self, name)
            # Handle dict with numeric keys (e.g., {1e9: 0.5} -> {"1e9": 0.5})
            if isinstance(value, dict) and value:
                first_key = next(iter(value.keys()))
                if isinstance(first_key, (int, float)):
                    value = {str(k): v for k, v in value.items()}
            params[name] = value
        return params
    
    @classmethod
    def _create_from_params(cls, params: dict):
        """
        Create instance from params dict.
        
        Automatically handles dict with string keys that represent numbers,
        converting them back to numeric keys.
        """
        processed_params = {}
        param_names = cls.get_param_names()
        for name in param_names:
            value = params[name]
            # Handle dict with string keys that look like numbers
            if isinstance(value, dict) and value:
                first_key = next(iter(value.keys()))
                if isinstance(first_key, str):
                    # Try to convert string keys to float
                    try:
                        float(first_key)
                        value = {float(k): v for k, v in value.items()}
                    except ValueError:
                        # Not numeric strings, keep as is
                        pass
            processed_params[name] = value
        
        return cls(**processed_params)

