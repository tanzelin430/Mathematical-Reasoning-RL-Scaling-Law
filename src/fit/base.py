import numpy as np
from typing import Dict, Tuple
from datetime import datetime
import json
# ============================================================================
# Base Fitter Class - Provides common functionality for all fitters
# ============================================================================

class BaseFitter:
    """
    Base class for all fitting models. Provides core functionality:
    - save_to_json / load_from_json: automatic serialization based on PARAM_NAMES
    - get_params_array: generic method for extracting parameters for plotting/analysis
    
    Subclasses should define:
    - PARAM_NAMES: list of parameter names (required for serialization)
    - PARAM_ARRAY_CONFIG: dict for get_params_array (optional, for plotting/analysis)
      Format: {'curve_values': [...], 'param_groups': {'group1': [...], ...}}
    - __init__: constructor accepting all parameters in PARAM_NAMES
    - predict: core prediction logic (model-specific)
    
    Note: DataFrame prediction is handled by business layer functions in fit.py
    """
    
    PARAM_NAMES = []  # Subclass should override this
    
    def get_params_array(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Get parameters as arrays for analysis and plotting.
        
        This is a generic method that works with declarative configuration.
        Subclasses should define PARAM_ARRAY_CONFIG with:
        - 'curve_values': list/array of x-axis values for plotting
        - 'param_groups': dict mapping group names to lists of parameter names
        
        Returns:
        --------
        tuple : (curve_values, params_dict)
            - curve_values: np.ndarray of x-axis values
            - params_dict: dict mapping group names to parameter value arrays
        
        Example:
        --------
        class MyModel(BaseFitter):
            PARAM_NAMES = ['k0_5', 'k1_5', 'k3', 'E0_0_5', 'E0_1_5', 'E0_3']
            PARAM_ARRAY_CONFIG = {
                'curve_values': [0.5e9, 1.5e9, 3e9],
                'param_groups': {
                    'k': ['k0_5', 'k1_5', 'k3'],
                    'E0': ['E0_0_5', 'E0_1_5', 'E0_3']
                }
            }
        
        Then get_params_array() returns:
        (array([0.5e9, 1.5e9, 3e9]), {'k': array([...]), 'E0': array([...])})
        """
        if not hasattr(self.__class__, 'PARAM_ARRAY_CONFIG'):
            raise NotImplementedError(
                f"{self.__class__.__name__} must define PARAM_ARRAY_CONFIG class attribute "
                "for get_params_array to work. See BaseFitter.get_params_array docstring for format."
            )
        
        config = self.__class__.PARAM_ARRAY_CONFIG
        
        # Extract curve values
        curve_values = np.array(config['curve_values'])
        
        # Extract parameter groups
        params_dict = {}
        for group_name, param_names in config['param_groups'].items():
            param_values = np.array([getattr(self, name) for name in param_names])
            params_dict[group_name] = param_values
        
        return curve_values, params_dict
    
    def save_to_json(self, filepath: str, 
                     metadata: dict = None):
        """Save model to JSON file."""
        # Get parameters as dict
        params_dict = self._get_params_for_json()
        
        result = {
            "model": self.__class__.__name__,
            "params": params_dict,
            "datetime": datetime.now().isoformat(),
        }
        
        # Add optional metadata
        if metadata:
            result["metadata"] = metadata
        
        from pathlib import Path
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Model saved to: {filepath}")
    
    @classmethod
    def load_from_json(cls, filepath: str):
        """Load model from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if data["model"] != cls.__name__:
            raise ValueError(f"Expected {cls.__name__} model, got {data['model']}")
        
        instance = cls._create_from_params(data["params"])
        
        print(f"Model loaded from: {filepath}")
        if 'metadata' in data:
            print(f"Fit metadata: {data['metadata']}")
        
        return instance
    
    def _get_params_for_json(self) -> dict:
        """
        Get parameters as dict for JSON serialization.
        
        Automatically extracts parameters based on PARAM_NAMES.
        Handles dict with numeric keys by converting them to strings.
        """
        params = {}
        for name in self.PARAM_NAMES:
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
        for name in cls.PARAM_NAMES:
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

