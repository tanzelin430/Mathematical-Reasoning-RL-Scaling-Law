import numpy as np
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from src.common import data_proc
from src.fit.base import BaseFitter
from typing import Callable, List, Dict, Union, Optional, Tuple
from src.common.fit_utils import cma_curve_fit, gen_inv_weights, gen_grouped_inv_weights
from src.fit.models import get_model_class

def get_x_y_data_from_df(
    df, 
    x_column_list, 
    y_column, 
    x_transform_list: list[Callable | None]=None, 
    y_transform: Callable=None,
) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
    """
    Extract and preprocess x, y data from dataframe for fitting.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    x_column_list : list
        List of column names for x variables (curve column should be included here)
    y_column : str
        Column name for y variable  
    x_transform_list : list of callable, optional
        List of transform functions for each x column
    y_transform : callable, optional
        Transform function for y column (e.g., lambda x: 1-x for ErrRate)
        
    Returns:
    --------
    tuple : (x_data_tuple, y_data_array)
    """
    df_fit = df.copy()
    
    # 准备拟合数据 - 返回tuple避免后续转置
    x_transform_list = [None] * len(x_column_list) if x_transform_list is None else x_transform_list
    x_data = tuple(x_transform(df_fit[c]) if x_transform is not None else df_fit[c] 
                     for c, x_transform in zip(x_column_list, x_transform_list))
    
    y_data = y_transform(df_fit[y_column]) if y_transform is not None else df_fit[y_column]
    y_data = y_data.to_numpy(dtype=float)
    
    # 确保所有x_arrays和y_data的长度相同
    for i, x_arr in enumerate(x_data):
        if len(x_arr) != len(y_data):
            raise ValueError(f"x_arrays[{i}] and y_data must have the same length, but got {len(x_arr)} and {len(y_data)}")
    
    return x_data, y_data

def fit_on(
    FitterClass, 
    df, 
    eval_name = "holdout_score", 
    x_column_list=["N", "E"], 
    x_transform_list=[None, None], 
    y_transform: Callable=None, 
    bounds=None, 
    p0=None, 
    x_inv_weight_power=0, 
    cma_verbose_interval=0,
) -> BaseFitter:
    """
    Fit a model to data using CMA-ES optimization.
    
    Returns:
    --------
    fitter : BaseFitter
        Fitted model with INFO set (access via fitter.get_info())
    """
    # 准备拟合数据（使用通用函数）
    x_data, y_data = get_x_y_data_from_df(
        df, 
        x_column_list=x_column_list, 
        y_column=eval_name,
        x_transform_list=x_transform_list,
        y_transform=y_transform,
    )
    
    # Get bounds and p0: caller's args > FitterClass defaults
    if bounds is None:
        bounds = getattr(FitterClass, 'DEFAULT_BOUNDS', None)
    if p0 is None:
        p0 = getattr(FitterClass, 'DEFAULT_P0', None)
    
    
    # Generate grouped inverse weights: group by x_data[0] (N), weight by x_data[1] (C/E)
    # This ensures each N curve has equal total weight, while points within each curve are weighted by 1/C
    inv_weights = gen_grouped_inv_weights(
        x_group=x_data[0],   # N for grouping
        x=x_data[1],  # C/E for weighting
        power=x_inv_weight_power
    )
    # 使用CMA-ES curve_fit函数进行拟合
    fitted_params, best_loss, r2 = cma_curve_fit(
        FitterClass.model,
        x_data,  # 直接传递tuple，不需要转置
        y_data,
        p0=p0,
        bounds=bounds,
        n_trials=5,  # 增加以提高稳定性
        max_iters=1600,
        verbose=True,
        verbose_interval=cma_verbose_interval,
        weights=inv_weights
    )
    
    # 创建预测器使用拟合得到的参数
    fitter = FitterClass(*fitted_params)
    
    # Build and set info dict
    info = {
        "r2": r2,
        "loss": best_loss,
        "n_points": len(y_data)
    }
    fitter.set_info(info)

    return fitter

def predict_on(fitter: "BaseFitter", df: pd.DataFrame, x_column_list=["N", "E"], y_transform_recover: Callable=None) -> np.ndarray:
    """
    Business layer prediction function.
    
    Returns:
    --------
    np.ndarray : Predicted error rates (converted from log10)
    """
    
    # Extract x data as arrays (more efficient than row-by-row)
    x_data = tuple(df[col].values for col in x_column_list)
    
    # Predict
    predictions = fitter.predict(x_data)

    # Recover predictions
    if y_transform_recover is not None:
        predictions = y_transform_recover(predictions)
    
    return predictions

'''
Save and load
'''

def save_batch_fitters(filepath: str, fitters: List[BaseFitter], append: bool = False):
    """
    Internal function to write fitters to JSON file.
    
    Parameters:
    -----------
    filepath : str
        Path to the JSON file
    fitters : list of BaseFitter
        Each fitter should have CONTEXT and INFO set via set_context() and set_info()
    append : bool
        If True, append to existing file; if False, overwrite
    """
    # Load existing data if appending and file exists
    if append and Path(filepath).exists():
        with open(filepath, 'r') as f:
            result = json.load(f)
        existing_count = len(result.get('fits', []))
        print(f"Loading existing file: {filepath}")
        print(f"  - Existing fits: {existing_count}")
    else:
        result = {
            "datetime": datetime.now().isoformat(),
            "fits": []
        }
        existing_count = 0
        if not append:
            action = "Saving" if Path(filepath).exists() else "Creating"
            print(f"{action} file: {filepath}")
        else:
            print(f"Creating new file: {filepath}")
    
    # Append new fits
    for fitter in fitters:
        fit_entry = {
            "context": fitter.get_context(),
            "params": fitter._get_params_for_json(),
            "info": fitter.get_info()
        }
        result["fits"].append(fit_entry)
    
    # Update timestamp
    if append and existing_count > 0:
        result["last_updated"] = datetime.now().isoformat()
    
    # Ensure output directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Write to file
    with open(filepath, 'w') as f:
        json.dump(result, f, indent=2)
    
    # Print summary
    if append and existing_count > 0:
        print(f"Batch fitters appended to: {filepath}")
        print(f"  - New fits added: {len(fitters)}")
        print(f"  - Total fits: {len(result['fits'])}")
    else:
        print(f"Batch fitters saved to: {filepath}")
        print(f"  - Total fits: {len(fitters)}")




def load_batch_fitters(filepath: str, filter_context: Optional[Dict] = None) -> List[BaseFitter]:
    """
    Load multiple fitters from a JSON file.
    
    Parameters:
    -----------
    filepath : str
        Path to the JSON file
    filter_context : dict, optional
        Filter fits by context fields (e.g., {"data_source": "base"})
    
    Returns:
    --------
    list of BaseFitter instances
        Each fitter has CONTEXT and INFO accessible via get_context() and get_info()
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    fitters = []
    for fit_data in data["fits"]:
        # Filter if needed
        if filter_context:
            if not all(fit_data["context"].get(k) == v for k, v in filter_context.items()):
                continue
        
        # Get model class
        model_name = fit_data["context"]["fit_model"]
        FitterClass = get_model_class(model_name)
        
        # Reconstruct fitter
        fitter = FitterClass._create_from_params(fit_data["params"])
        
        # Set context and info
        fitter.set_context(fit_data["context"])
        fitter.set_info(fit_data.get("info", {}))
        
        fitters.append(fitter)
    
    print(f"Batch fitters loaded from: {filepath}")
    print(f"  - Total fits loaded: {len(fitters)}")
    if filter_context:
        print(f"  - Filter applied: {filter_context}")
    
    return fitters