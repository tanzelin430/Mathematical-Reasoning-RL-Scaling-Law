"""
Utilities for handling source-curve specific configurations.

This module provides helper functions for working with nested source-curve maps
used in plot customization (masks, labels, colors).
"""

import numpy as np
from src.common import config


def normalize_curve_key(curve_val):
    """Normalize curve value to string key for matching in JSON maps.
    
    Handles int, float, and scientific notation (e.g., 5e8).
    
    Args:
        curve_val: The curve value to normalize (int, float, or string)
    
    Returns:
        List of possible string representations to try for matching
    """
    if curve_val is None:
        return []
    
    # Try multiple representations
    keys = []
    
    # 1. Direct string
    keys.append(str(curve_val))
    
    # 2. As integer if it's a whole number
    if isinstance(curve_val, (int, float)) and curve_val == int(curve_val):
        keys.append(str(int(curve_val)))
    
    # 3. Scientific notation with 1 decimal
    if isinstance(curve_val, (int, float)):
        keys.append(f"{curve_val:.1e}")
        # Also try without decimal if it's a clean power
        if curve_val >= 1e6:
            exp = int(np.log10(curve_val))
            mantissa = curve_val / (10 ** exp)
            if abs(mantissa - round(mantissa)) < 0.01:
                keys.append(f"{int(round(mantissa))}e{exp}")
    
    return keys


def get_from_source_curve_map(source_curve_map, data_source, curve_val, default=None):
    """Get value from a nested source-curve map.
    
    Args:
        source_curve_map: Dict like {"base": {"53760": value, ...}, ...}
        data_source: Source key (e.g., "base", "instruct")
        curve_val: Curve value to match (e.g., 53760, 5e8)
        default: Default value if not found
    
    Returns:
        The value from the map, or default if not found
    """
    if not source_curve_map or data_source not in source_curve_map:
        return default
    
    curve_map = source_curve_map[data_source]
    keys = normalize_curve_key(curve_val)
    
    for key in keys:
        if key in curve_map:
            return curve_map[key]
    
    return default


def get_source_curve_color(data_source, curve_val, plot_source_curve_color):
    """Get color for a specific source-curve combination.
    
    Args:
        data_source: The data source name
        curve_val: The curve value
        plot_source_curve_color: Nested dict from args (source -> curve -> color)
    
    Returns:
        Color string
    """
    # Try custom color first
    color = get_from_source_curve_map(plot_source_curve_color, data_source, curve_val)
    if color:
        return color
    
    # Fallback to data_source default color
    return config.get_color_for_curve(data_source)


def get_source_curve_label(data_source, curve_column, curve_val, plot_source_curve_label):
    """Get label for a specific source-curve combination.
    
    Args:
        data_source: The data source name
        curve_column: The curve column name (e.g., "E", "N")
        curve_val: The curve value
        plot_source_curve_label: Nested dict from args (source -> curve -> label)
    
    Returns:
        Label string
    """
    # Try custom label first
    label = get_from_source_curve_map(plot_source_curve_label, data_source, curve_val)
    if label:
        return label
    
    # Fallback to default format
    from src.common.plot import legend_format
    return f"{data_source.capitalize()}-{legend_format(curve_column, curve_val)}"


def apply_source_curve_mask(df, curve_column, data_source, plot_source_curve_mask, plot_curve_mask):
    """Apply source-curve specific mask to a single DataFrame.
    
    Args:
        df: DataFrame to filter
        curve_column: The curve column name
        data_source: The data source for this DataFrame
        plot_source_curve_mask: Nested dict (source -> list of curve values)
        plot_curve_mask: Fallback list of curve values
    
    Returns:
        Filtered DataFrame
    """
    # Try source-specific mask first
    if plot_source_curve_mask and data_source in plot_source_curve_mask:
        curve_values = plot_source_curve_mask[data_source]
        print(f"Filtering curves ({curve_column}) for {data_source}: {curve_values}")
        return df[df[curve_column].isin(curve_values)].copy()
    
    # Fallback to general curve mask
    if plot_curve_mask is not None:
        print(f"Filtering curves ({curve_column}): {plot_curve_mask}")
        return df[df[curve_column].isin(plot_curve_mask)].copy()
    
    return df

