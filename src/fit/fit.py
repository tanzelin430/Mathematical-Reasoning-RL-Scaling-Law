import numpy as np
import pandas as pd
from src.common import data_proc
from src.fit.base import BaseFitter
from typing import Callable, List, Dict, Union, Optional, Tuple
from src.common.fit_utils import cma_curve_fit

def get_x_y_data_from_df(
    df, 
    x_column_list, 
    y_column, 
    x_transform_list: list[Callable | None]=None, 
    y_transform: Callable=None,
    warmup_clip: int = 0,
    ending_clip: int = 0
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
    warmup_clip : int, optional
        Number of steps to remove from the beginning (0 means no clipping)
    ending_clip : int, optional
        Number of steps to remove from the end (0 means no clipping)
        
    Returns:
    --------
    tuple : (x_data_tuple, y_data_array)
    """
    df_fit = df.copy()
    
    # Apply clipping
    if warmup_clip > 0 or ending_clip > 0:
        curve_column = x_column_list[0] if x_column_list else "N"
        df_fit = data_proc.apply_clip(
            df_fit, 
            curve_column=curve_column, 
            warmup_clip=warmup_clip,
            ending_clip=ending_clip
        )
    
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

def fit_on(FitterClass, df, eval_name = "holdout_score", x_column_list=["N", "E"], x_transform_list=[None, None], fit_load_path=None, fit_save_path=None, warmup_clip=0, ending_clip=0, bounds=None, p0=None):
    """
    Fit a model to data using CMA-ES optimization.
    """
    # Load from file if requested
    if fit_load_path is not None:
        print(f"=== Loading Model from {fit_load_path} ===")
        return FitterClass.load_from_json(fit_load_path)
    
    logErrRate = lambda x: np.log10(np.clip(1 - x, 1e-12, None))
    
    # 准备拟合数据（使用通用函数）
    x_data, y_data = get_x_y_data_from_df(
        df, 
        x_column_list=x_column_list, 
        y_column=eval_name,
        x_transform_list=x_transform_list,
        y_transform=logErrRate,
        warmup_clip=warmup_clip,
        ending_clip=ending_clip
    )
    
    # Get bounds and p0: caller's args > FitterClass defaults
    if bounds is None:
        bounds = getattr(FitterClass, 'DEFAULT_BOUNDS', None)
    if p0 is None:
        p0 = getattr(FitterClass, 'DEFAULT_P0', None)
    
    print("=== CMA-ES拟合 ===")
    print(f"数据点数量: {len(y_data)}")
    print(f"x_arrays: {len(x_data)} arrays with shapes {[arr.shape for arr in x_data]}")
    print(f"模型: {FitterClass.__name__}, 参数数量: {len(p0)}")
    
    # 使用CMA-ES curve_fit函数进行拟合
    fitted_params, best_loss, r2 = cma_curve_fit(
        FitterClass.model,
        x_data,  # 直接传递tuple，不需要转置
        y_data,
        p0=p0,
        bounds=bounds,
        n_trials=2,
        max_iters=1600,
        verbose=True
    )
    
    print(f"\n=== 拟合结果 ===")
    print(f"最终损失: {best_loss:.6e}")
    print(f"R²: {r2:.4f}")
    
    # 创建预测器使用拟合得到的参数
    fitter = FitterClass(*fitted_params)

    # Save if requested
    if fit_save_path is not None:
        fitter.save_to_json(fit_save_path)

    return fitter

def predict_on(fitter: "BaseFitter", df: pd.DataFrame, x_column_list=["N", "E"]) -> np.ndarray:
    """
    Business layer prediction function.
    
    Returns:
    --------
    np.ndarray : Predicted error rates (converted from log10)
    """
    y_transform_recover = lambda x: 10 ** x
    
    # Extract x data as arrays (more efficient than row-by-row)
    x_data = tuple(df[col].values for col in x_column_list)
    
    # Get predictions and convert back from log10
    predictions = y_transform_recover(fitter.predict(x_data))
    
    return predictions