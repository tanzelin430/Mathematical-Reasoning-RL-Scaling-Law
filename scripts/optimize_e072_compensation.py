#!/usr/bin/env python3
"""
Optimize E0_72 as a compensation parameter for hybrid extrapolation.

Strategy:
1. Load k_max, N0 from 32B fit (0.5B-32B data)
2. Load E0_{0.5-32} from 32B fit
3. Calculate k(72B) using k_max * 72e9 / (72e9 + N0)
4. Optimize ONLY E0_72 on 72B data to minimize prediction error
5. Save hybrid parameters with optimized E0_72

This approach preserves k(N) smoothness while correcting intercept with minimal 72B data.
"""

import json
import numpy as np
from scipy.optimize import minimize_scalar
from src.common import data_proc, config

def compute_loss_for_e0_72(E0_72, k_72B, df_72B, fit_x, eval_col='holdout_score'):
    """
    Compute loss for a given E0_72 value on 72B data.

    Uses log-linear model: log10(y) = -k * log10(x) + E0
    or equivalently: y = 10^E0 * x^(-k)

    Loss is computed in log space (as in the original fitting).
    """
    # Get x and y data
    x_data = df_72B[fit_x].values
    y_data_raw = df_72B[eval_col].values  # Raw score (reward)

    # Convert to ErrRate = 1 - score
    y_true = 1 - y_data_raw

    # Filter out invalid data
    valid_mask = (x_data > 0) & (y_true > 0) & np.isfinite(x_data) & np.isfinite(y_true)
    x_data = x_data[valid_mask]
    y_true = y_true[valid_mask]

    if len(x_data) == 0:
        return np.inf

    # Compute predictions: y = 10^(-k * log10(x) + E0)
    y_pred = np.power(10.0, -k_72B * np.log10(x_data) + E0_72)

    # Check for invalid predictions
    if not np.all(np.isfinite(y_pred)):
        return np.inf

    # Compute loss in LOG SPACE (same as original fitting)
    log_y_true = np.log10(y_true)
    log_y_pred = np.log10(y_pred)

    # MSE in log space
    loss = np.mean((log_y_true - log_y_pred) ** 2)

    return loss

def compute_r2(y_true, y_pred):
    """Compute R² in original space"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def optimize_e0_72_for_config(fit_32B, data_source, warmup_clip):
    """
    Optimize E0_72 for a single configuration.

    Args:
        fit_32B: Dictionary with 32B fit parameters and context
        data_source: 'base' or 'instruct'
        warmup_clip: Number of warmup steps to clip

    Returns:
        Dictionary with hybrid parameters and optimization info
    """
    context = fit_32B['context']
    params_32B = fit_32B['params']
    fit_x = context['fit_x']

    print(f"\n{'='*80}")
    print(f"Processing: {data_source} - L(N, {fit_x})")
    print(f"{'='*80}")

    # Extract k parameters from 32B fit
    k_max = params_32B['k_max']
    N0 = params_32B['N0']

    # Calculate k(72B) from extrapolation
    N_72B = 72e9
    k_72B = k_max * N_72B / (N_72B + N0)

    print(f"k_max = {k_max:.6f}")
    print(f"N0 = {N0/1e9:.3f}B")
    print(f"k(72B) extrapolated = {k_72B:.6f}")

    # Load 72B data
    df = data_proc.load_and_preprocess(config.CSV_MAP[data_source])

    # Apply warmup clip
    if warmup_clip > 0:
        df = df[df['step'] > warmup_clip].copy()

    # Filter for 72B only
    df_72B = df[df['N'] == N_72B].copy()

    if len(df_72B) == 0:
        raise ValueError(f"No 72B data found for {data_source}")

    print(f"72B data points: {len(df_72B)}")

    # Initial guess: use linear extrapolation from 14B and 32B
    E0_14 = params_32B['E0_14']
    E0_32 = params_32B['E0_32']

    # Linear extrapolation in log(N) space
    log_N_14 = np.log10(14e9)
    log_N_32 = np.log10(32e9)
    log_N_72 = np.log10(72e9)

    # Slope: (E0_32 - E0_14) / (log_N_32 - log_N_14)
    slope = (E0_32 - E0_14) / (log_N_32 - log_N_14)
    E0_72_init = E0_32 + slope * (log_N_72 - log_N_32)

    print(f"E0_72 initial guess (linear extrapolation): {E0_72_init:.6f}")

    # Define objective function
    def objective(E0_72):
        return compute_loss_for_e0_72(E0_72, k_72B, df_72B, fit_x)

    # Optimize E0_72
    # Use reasonable bounds around initial guess
    bounds = (E0_72_init - 2.0, E0_72_init + 2.0)

    result = minimize_scalar(objective, bounds=bounds, method='bounded')

    E0_72_opt = result.x
    loss_72B = result.fun

    print(f"E0_72 optimized = {E0_72_opt:.6f}")
    print(f"Loss (log space) = {loss_72B:.6e}")

    # Compute R² for 72B in original space
    x_data = df_72B[fit_x].values
    y_data_raw = df_72B['holdout_score'].values
    y_true = 1 - y_data_raw

    valid_mask = (x_data > 0) & (y_true > 0) & np.isfinite(x_data) & np.isfinite(y_true)
    x_data = x_data[valid_mask]
    y_true = y_true[valid_mask]

    y_pred = np.power(10.0, -k_72B * np.log10(x_data) + E0_72_opt)
    r2_72B = compute_r2(y_true, y_pred)

    print(f"R² (72B) = {r2_72B:.6f}")

    # Create hybrid parameters
    hybrid_params = {
        'k_max': k_max,
        'N0': N0,
        'E0_0_5': params_32B['E0_0_5'],
        'E0_1_5': params_32B['E0_1_5'],
        'E0_3': params_32B['E0_3'],
        'E0_7': params_32B['E0_7'],
        'E0_14': params_32B['E0_14'],
        'E0_32': params_32B['E0_32'],
        'E0_72': E0_72_opt,
    }

    # Create hybrid fit entry
    hybrid_fit = {
        'context': context.copy(),
        'params': hybrid_params,
        'info': {
            'r2': fit_32B['info']['r2'],  # R² from 32B fit
            'r2_72B': r2_72B,  # R² for 72B extrapolation
            'loss_72B': loss_72B,  # Loss for 72B
            'k_72B': k_72B,  # Extrapolated k value
        }
    }

    return hybrid_fit

def main():
    # Load 32B fit results
    with open('outputs/fits_exp1_up32B.json', 'r') as f:
        data_32B = json.load(f)

    fits_32B = data_32B['fits']

    print("="*80)
    print("Optimizing E0_72 as compensation parameter")
    print("="*80)
    print(f"Loaded {len(fits_32B)} fits from 32B data")

    # Process each fit configuration
    hybrid_fits = []

    for fit_32B in fits_32B:
        context = fit_32B['context']
        data_source = context['data_source']
        warmup_clip = context['warmup_clip']

        try:
            hybrid_fit = optimize_e0_72_for_config(fit_32B, data_source, warmup_clip)
            hybrid_fits.append(hybrid_fit)
        except Exception as e:
            print(f"ERROR processing {data_source} L(N,{context['fit_x']}): {e}")
            continue

    # Save results
    output_data = {
        'description': 'Hybrid: k from 32B, E0_72 as compensation',
        'fits': hybrid_fits,
    }

    output_path = 'outputs/fits_hybrid_kn_e072_new.json'
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Saved {len(hybrid_fits)} hybrid fits to: {output_path}")
    print(f"{'='*80}")

    # Compare with existing fits_hybrid_kn_e072.json if it exists
    try:
        with open('outputs/fits_hybrid_kn_e072.json', 'r') as f:
            existing_data = json.load(f)

        print("\n" + "="*80)
        print("Comparison with existing fits_hybrid_kn_e072.json:")
        print("="*80)

        for i, (new_fit, old_fit) in enumerate(zip(hybrid_fits, existing_data['fits'])):
            context = new_fit['context']
            data_source = context['data_source']
            fit_x = context['fit_x']

            E0_72_new = new_fit['params']['E0_72']
            E0_72_old = old_fit['params']['E0_72']
            diff = E0_72_new - E0_72_old
            rel_diff_pct = (diff / E0_72_old) * 100 if E0_72_old != 0 else float('inf')

            print(f"\n{data_source} L(N,{fit_x}):")
            print(f"  E0_72 (existing): {E0_72_old:.6f}")
            print(f"  E0_72 (new):      {E0_72_new:.6f}")
            print(f"  Difference:       {diff:+.6f}  ({rel_diff_pct:+.2f}%)")
            print(f"  R²_72B (existing): {old_fit['info']['r2_72B']:.6f}")
            print(f"  R²_72B (new):      {new_fit['info']['r2_72B']:.6f}")
    except FileNotFoundError:
        print("\nNo existing fits_hybrid_kn_e072.json found for comparison")

if __name__ == '__main__':
    main()
