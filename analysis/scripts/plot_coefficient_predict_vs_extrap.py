#!/usr/bin/env python3
"""
Plot k(N) coefficient comparison: Intra-model vs Inter-model prediction strategies.

Two PDFs:
1. coefficient_k_predict_vs_extrap_C.pdf - L(N,C) for Base and Instruct
2. coefficient_k_predict_vs_extrap_D.pdf - L(N,D) for Base and Instruct

Each PDF contains 2 subplots (Base and Instruct), showing:
- Intra-model prediction: k(N) from early training data (dashed line) + all scatter points
- Inter-model prediction: k(N) from 0.5B-32B data (solid line) + all scatter points + 72B diamond
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

# Configuration
FITS_DIR = Path("outputs")
OUTPUT_DIR = Path("outputs")

# Load fit results
FITS_PREDICT_BASE = FITS_DIR / "fits_predict_base25.json"
FITS_PREDICT_INSTRUCT = FITS_DIR / "fits_predict_instruct40.json"
FITS_EXTRAP = FITS_DIR / "fits_hybrid_kn_e072.json"

# Model sizes (in billions)
MODEL_SIZES = [0.5, 1.5, 3, 7, 14, 32, 72]
MODEL_SIZES_NUMERIC = [x * 1e9 for x in MODEL_SIZES]

# Colors - matching the original coefficient comparison plot
COLOR_CURVE = '#2E86AB'  # Blue for curves
COLOR_INTRA_SCATTER = '#A23B72'  # Purple for intra-model prediction points
COLOR_INTER_SCATTER = '#F18F01'  # Orange for inter-model prediction points


def load_fit_params(fit_file, source, x_var):
    """Load k_max and N0 from fit file for given source and x_var."""
    with open(fit_file, 'r') as f:
        data = json.load(f)

    # Extract fits array
    fits = data.get('fits', data) if isinstance(data, dict) else data

    # Map x_var to fit_x in context
    fit_x_map = {'C': 'C_raw', 'E': 'E'}
    target_fit_x = fit_x_map.get(x_var, x_var)

    # Find the fit for this source and x_var
    for fit in fits:
        context = fit.get('context', {})
        if context.get('data_source') == source and context.get('fit_x') == target_fit_x:
            params = fit['params']
            k_max = params['k_max']
            N0 = params['N0']
            return k_max, N0

    raise ValueError(f"Fit not found for source={source}, x_var={x_var} (fit_x={target_fit_x}) in {fit_file}")


def compute_k_curve(k_max, N0, N_values):
    """Compute k(N) = k_max * N / (N + N0)."""
    return k_max * N_values / (N_values + N0)


def plot_comparison(x_var, output_file):
    """
    Plot k(N) comparison for given x_var (C or E).

    Args:
        x_var: 'C' or 'E'
        output_file: Output PDF path
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    sources = ['base', 'instruct']
    titles = ['Base', 'Instruct']
    x_label_map = {'C': 'C', 'E': 'D'}
    x_label = x_label_map.get(x_var, x_var)

    for idx, (ax, source, title) in enumerate(zip(axes, sources, titles)):
        # Load Intra-model prediction fit params (from early training data)
        if source == 'base':
            k_max_intra, N0_intra = load_fit_params(FITS_PREDICT_BASE, source, x_var)
        else:
            k_max_intra, N0_intra = load_fit_params(FITS_PREDICT_INSTRUCT, source, x_var)

        # Load Inter-model prediction fit params (from 0.5B-32B)
        k_max_inter, N0_inter = load_fit_params(FITS_EXTRAP, source, x_var)

        # Generate smooth N values for curves
        N_smooth = np.logspace(np.log10(0.5e9), np.log10(72e9), 200)

        # Compute k(N) curves
        k_intra = compute_k_curve(k_max_intra, N0_intra, N_smooth)
        k_inter = compute_k_curve(k_max_inter, N0_inter, N_smooth)

        # 1. Intra-model prediction: dashed line + purple scatter points
        ax.plot(N_smooth / 1e9, k_intra, '--', color=COLOR_CURVE, linewidth=2.5,
                alpha=0.7, zorder=1)

        # Scatter points for all model sizes (intra-model)
        k_intra_points = compute_k_curve(k_max_intra, N0_intra, np.array(MODEL_SIZES_NUMERIC))
        ax.scatter(MODEL_SIZES, k_intra_points, s=120, marker='o',
                  color=COLOR_INTRA_SCATTER, edgecolors='white', linewidths=1.5,
                  alpha=0.9, zorder=4)

        # 2. Inter-model prediction: solid line + orange scatter points
        ax.plot(N_smooth / 1e9, k_inter, '-', color=COLOR_CURVE, linewidth=2.5,
                alpha=0.9, zorder=2)

        # Scatter points for all model sizes (inter-model)
        k_inter_points = compute_k_curve(k_max_inter, N0_inter, np.array(MODEL_SIZES_NUMERIC))

        # Plot 0.5B-32B as circles
        ax.scatter(MODEL_SIZES[:-1], k_inter_points[:-1], s=120, marker='o',
                  color=COLOR_INTER_SCATTER, edgecolors='white', linewidths=1.5,
                  alpha=0.9, zorder=5)

        # Plot 72B as diamond
        ax.scatter([MODEL_SIZES[-1]], [k_inter_points[-1]], s=150, marker='D',
                  color=COLOR_INTER_SCATTER, edgecolors='white', linewidths=2,
                  alpha=0.95, zorder=6)

        # Formatting
        ax.set_xscale('log')
        ax.set_xlabel('N (Model Size)', fontsize=16, fontweight='bold')
        ax.set_ylabel('K(N)', fontsize=16, fontweight='bold')
        ax.set_title(f'{title}: L(N,{x_label})', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
        ax.tick_params(axis='both', which='major', labelsize=14)

        # Set x-axis ticks to model sizes
        ax.set_xticks(MODEL_SIZES)
        ax.set_xticklabels([f'{s:.1f}' if s < 1 else f'{int(s)}' for s in MODEL_SIZES])

        # Legend - show on every subplot
        legend_elements = [
            Line2D([0], [0], linestyle='--', color=COLOR_CURVE, linewidth=2.5,
                   label='Intra-model prediction'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_INTRA_SCATTER,
                   markersize=9, markeredgecolor='white', markeredgewidth=1.5,
                   linestyle='', label='Intra-model points'),
            Line2D([0], [0], linestyle='-', color=COLOR_CURVE, linewidth=2.5,
                   label='Inter-model prediction'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_INTER_SCATTER,
                   markersize=9, markeredgecolor='white', markeredgewidth=1.5,
                   linestyle='', label='Inter-model points (0.5B-32B)'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor=COLOR_INTER_SCATTER,
                   markersize=9, markeredgecolor='white', markeredgewidth=1.5,
                   linestyle='', label='Inter-model 72B (extrapolated)')
        ]
        ax.legend(handles=legend_elements, loc='best', fontsize=13, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Generate L(N,C) comparison
    plot_comparison('C', OUTPUT_DIR / 'coefficient_k_predict_vs_extrap_C.pdf')

    # Generate L(N,D) comparison
    plot_comparison('E', OUTPUT_DIR / 'coefficient_k_predict_vs_extrap_D.pdf')

    print("\nGenerated 2 PDFs:")
    print("  - coefficient_k_predict_vs_extrap_C.pdf (L(N,C) comparison)")
    print("  - coefficient_k_predict_vs_extrap_D.pdf (L(N,D) comparison)")


if __name__ == '__main__':
    main()
