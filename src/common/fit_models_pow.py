#!/usr/bin/env python3
"""
Simple Lookup Table Fitting for Scaling Laws
log_errrate = k(N) * log_E + E0(N)
where k(N) and E0(N) are lookup tables for the 5 N values.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.linear_model import LinearRegression
import data_proc
import config
import json
from datetime import datetime

class SimpleLinearLookupFit:
    """
    Simple lookup table fitting: log_errrate = k(N) * log_E + E0(N)
    where k(N) and E0(N) are stored as lookup tables for each N value.
    """
    
    def __init__(self):
        # Lookup tables for k(N) and E0(N)
        self.k_lookup: Dict[float, float] = {}
        self.E0_lookup: Dict[float, float] = {}
        self.fit_x_column = "E"  # Default x column used for fitting
    
    def fit(self, df: pd.DataFrame, eval_name: str = "holdout_score") -> Dict:
        """
        Fit k(N) and E0(N) lookup tables using scipy curve_fit for each N.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with columns N, E, and eval_name
        eval_name : str
            Name of the evaluation column to fit
            
        Returns:
        --------
        Dict with fit results and R² values
        """
        print("=== Simple Lookup Table Fitting ===")
        
        # Prepare data - reuse existing data processing
        df['ErrRate'] = 1 - df[eval_name]
        
        # Apply warmup clipping using existing function
        df_fit = data_proc.apply_clip(
            df, 
            curve_column="N", 
            warmup_frac=config.WARMUP_CLIPPING_FACTOR_FOR_RAW
        )
        
        # Remove step=0 and merge duplicates - reuse existing function
        df_fit = df_fit[df_fit['step'] > 0].reset_index(drop=True)
        df_fit = data_proc.merge_duplicate_steps(
            df_fit, 
            group_columns=['N', 'step'], 
            mode='mean'
        )
        
        # Get unique N values
        N_values = sorted(df_fit['N'].unique())
        print(f"Found N values: {[f'{N/1e9:.1f}B' for N in N_values]}")
        
        # Fit for each N value
        fit_results = {}
        all_r2 = []
        
        for N in N_values:
            # Get data for this N
            df_N = df_fit[df_fit['N'] == N].copy()
            
            if len(df_N) < 2:
                print(f"Warning: Not enough data points for N={N/1e9:.1f}B")
                continue
                
            # Prepare X and y
            log_E = np.log10(df_N['E'].values)
            log_errrate = np.log10(np.clip(df_N['ErrRate'].values, 1e-12, None))
            
            # Remove any infinite values
            valid_mask = np.isfinite(log_E) & np.isfinite(log_errrate)
            log_E = log_E[valid_mask]
            log_errrate = log_errrate[valid_mask]
            
            if len(log_E) < 2:
                print(f"Warning: Not enough valid data points for N={N/1e9:.1f}B")
                continue
            
            # Linear regression using sklearn: log_errrate = k * log_E + E0
            X = log_E.reshape(-1, 1)
            y = log_errrate
            
            reg = LinearRegression()
            reg.fit(X, y)
            
            k = -reg.coef_[0]  # Take negative to store positive k (physics convention)
            E0 = reg.intercept_
            r2 = reg.score(X, y)
            
            # Store in lookup tables
            self.k_lookup[N] = k
            self.E0_lookup[N] = E0
            
            fit_results[N] = {
                'k': k,
                'E0': E0,
                'r2': r2,
                'n_points': len(log_E)
            }
            all_r2.append(r2)
            
            print(f"N={N/1e9:>4.1f}B: k={k:>8.4f}, E0={E0:>8.4f}, R²={r2:>6.4f}, n={len(log_E):>3d}")
        
        overall_r2 = np.mean(all_r2) if all_r2 else 0.0
        print(f"\nOverall mean R²: {overall_r2:.4f}")
        
        return {
            'k_lookup': self.k_lookup.copy(),
            'E0_lookup': self.E0_lookup.copy(), 
            'N_values': N_values.copy(),
            'fit_results': fit_results,
            'overall_r2': overall_r2
        }
    
    def predict_errrate_single(self, N: float, E: float) -> float:
        """Predict error rate for a single (N, E) point."""
        if N not in self.k_lookup or N not in self.E0_lookup:
            raise ValueError(f"N={N/1e9:.1f}B not found in lookup tables")
        
        k = self.k_lookup[N]
        E0 = self.E0_lookup[N]
        
        log_E = np.log10(E)
        log_errrate = -k * log_E + E0  # k is stored as positive, so use -k in prediction
        errrate = 10 ** log_errrate
        
        return errrate
    
    def predict_errrate_df(self, df: pd.DataFrame, N_col: str = "N", E_col: str = None) -> np.ndarray:
        """Predict error rates for a dataframe."""
        if E_col is None:
            E_col = self.fit_x_column  # Use the column that was used for fitting
            
        predictions = []
        
        for _, row in df.iterrows():
            N = row[N_col]
            X = row[E_col]
            
            if N in self.k_lookup and N in self.E0_lookup and X > 0:
                pred = self.predict_errrate_single(N, X)
                predictions.append(pred)
            else:
                predictions.append(np.nan)
        
        return np.array(predictions)
    
    def predict_reward_df(self, df: pd.DataFrame, N_col: str = "N", E_col: str = None) -> np.ndarray:
        """Predict rewards (1 - error_rate) for a dataframe."""
        errrates = self.predict_errrate_df(df, N_col, E_col)
        return 1 - errrates
    
    def get_k_array(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get k(N) as arrays for analysis."""
        N_values = sorted(self.k_lookup.keys())
        N_arr = np.array(N_values)
        k_arr = np.array([self.k_lookup[N] for N in N_values])
        return N_arr, k_arr
    
    def get_E0_array(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get E0(N) as arrays for analysis."""
        N_values = sorted(self.E0_lookup.keys())
        N_arr = np.array(N_values)
        E0_arr = np.array([self.E0_lookup[N] for N in N_values])
        return N_arr, E0_arr
    
    def save_to_json(self, filepath: str, data_source: str, eval_name: str = "holdout_score", 
                     curve_column: str = "N", df: pd.DataFrame = None):
        """Save model and results to JSON file."""
        # Convert numpy keys to strings for JSON serialization
        k_lookup_str = {str(k): v for k, v in self.k_lookup.items()}
        E0_lookup_str = {str(k): v for k, v in self.E0_lookup.items()}
        
        # Calculate R² if dataframe is provided
        r2 = 0.0
        if df is not None:
            from fit_utils import calculate_r2
            import data_proc
            import config
            
            # Prepare data the same way as in fitting
            df_temp = df.copy()
            df_temp['ErrRate'] = 1 - df_temp[eval_name]
            
            # Apply same preprocessing as in fitting
            df_fit = data_proc.apply_clip(
                df_temp, 
                curve_column=curve_column, 
                warmup_frac=config.WARMUP_CLIPPING_FACTOR_FOR_RAW
            )
            df_fit = df_fit[df_fit['step'] > 0].reset_index(drop=True)
            df_fit = data_proc.merge_duplicate_steps(
                df_fit, 
                group_columns=[curve_column, 'step'], 
                mode='mean'
            )
            
            # Get predictions and calculate R²
            y_pred = self.predict_errrate_df(df_fit)
            y_true = df_fit['ErrRate'].values
            r2 = calculate_r2(y_true, y_pred)
        
        # Determine x variables
        x_vars = [curve_column, self.fit_x_column]
        
        result = {
            "model": "SimpleLinearLookupFit",
            "params": {
                "k_lookup": k_lookup_str,
                "E0_lookup": E0_lookup_str,
                "fit_x_column": self.fit_x_column
            },
            "R2": r2,
            "data_source": data_source if data_source is not None else "None",
            "datetime": datetime.now().isoformat(),
            "fit_config": {
                "x": x_vars,
                "y": "ErrRate",
                "eval_name": eval_name
            }
        }
        
        # Ensure output directory exists
        from pathlib import Path
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Fitted model saved to: {filepath}")
    
    @classmethod
    def load_from_json(cls, filepath: str) -> 'SimpleLinearLookupFit':
        """Load model from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if data["model"] != "SimpleLinearLookupFit":
            raise ValueError(f"Expected SimpleLinearLookupFit model, got {data['model']}")
        
        # Create new instance
        instance = cls()
        
        # Load parameters
        params = data["params"]
        instance.k_lookup = {float(k): v for k, v in params["k_lookup"].items()}
        instance.E0_lookup = {float(k): v for k, v in params["E0_lookup"].items()}
        instance.fit_x_column = params["fit_x_column"]
        
        print(f"Model loaded from: {filepath}")
        print(f"Data source: {data['data_source']}")
        print(f"R²: {data['R2']:.4f}")
        print(f"Fit config: {data['fit_config']}")
        
        return instance