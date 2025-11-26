#!/bin/bash
# Generate k(N) coefficient comparison plots: Predict vs Extrapolate

set -e

echo "Generating k(N) coefficient comparison: Predict vs Extrapolate..."
uv run python scripts/plot_coefficient_predict_vs_extrap.py

echo "Done! Check outputs/coefficient_k_predict_vs_extrap_C.pdf and coefficient_k_predict_vs_extrap_D.pdf"
