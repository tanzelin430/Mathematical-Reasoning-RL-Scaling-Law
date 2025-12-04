#!/bin/bash

# Plot hybrid extrapolation results with E0_72 compensation
# Shows fitted curves with 0.5B-32B data (solid) and 72B extrapolation (dashed)

echo "========================================="
echo "Plotting Hybrid Extrapolation Results"
echo "========================================="

echo ""
echo "=== 1/4: Base L(N, C) ==="
uv run -m src.scaling_analysis \
  --plot \
  --plot-fit \
  --fit-load outputs/fits_hybrid_kn_e072.json \
  --fit-x C_raw \
  --data-sources base \
  --warmup-clip 0 \
  --fit-curve N \
  --fit-metric ErrRate \
  --plot-curve N \
  --plot-x C_raw \
  --eval holdout_score \
  --plot-metric ErrRate \
  --plot-x-scale log \
  --plot-y-scale log \
  --y-tick-spacing 0.1 \
  --y-tick-format auto \
  --y-grid-spacing 0.1 \
  --line-alpha 1 \
  --line-width 2.0 \
  --plot-use-legend \
  --scatter-alpha 0.5 \
  --scatter-size 20 \
  --scatter-marker o \
  --plot-title "Fitted L(N,C) on Base Model with Inter-model Prediction" \
  --output-prefix extrap_

echo ""
echo "=== 2/4: Base L(N, E) ==="
uv run -m src.scaling_analysis \
  --plot \
  --plot-fit \
  --fit-load outputs/fits_hybrid_kn_e072.json \
  --fit-x E \
  --data-sources base \
  --warmup-clip 0 \
  --fit-curve N \
  --fit-metric ErrRate \
  --plot-curve N \
  --plot-x E \
  --eval holdout_score \
  --plot-metric ErrRate \
  --plot-x-scale log \
  --plot-y-scale log \
  --y-tick-spacing 0.1 \
  --y-tick-format auto \
  --y-grid-spacing 0.1 \
  --line-alpha 1 \
  --line-width 2.0 \
  --plot-use-legend \
  --scatter-alpha 0.5 \
  --scatter-size 20 \
  --scatter-marker o \
  --plot-title "Fitted L(N,D) on Base Model with Inter-model Prediction" \
  --output-prefix extrap_

echo ""
echo "=== 3/4: Instruct L(N, C) ==="
uv run -m src.scaling_analysis \
  --plot \
  --plot-fit \
  --fit-load outputs/fits_hybrid_kn_e072.json \
  --fit-x C_raw \
  --data-sources instruct \
  --warmup-clip 3 \
  --fit-curve N \
  --fit-metric ErrRate \
  --plot-curve N \
  --plot-x C_raw \
  --eval holdout_score \
  --plot-metric ErrRate \
  --plot-x-scale log \
  --plot-y-scale log \
  --y-tick-spacing 0.1 \
  --y-tick-format auto \
  --y-grid-spacing 0.1 \
  --line-alpha 1 \
  --line-width 2.0 \
  --plot-use-legend \
  --scatter-alpha 0.5 \
  --scatter-size 20 \
  --scatter-marker o \
  --plot-title "Fitted L(N,C) on Instruct Model with Inter-model Prediction" \
  --output-prefix extrap_

echo ""
echo "=== 4/4: Instruct L(N, E) ==="
uv run -m src.scaling_analysis \
  --plot \
  --plot-fit \
  --fit-load outputs/fits_hybrid_kn_e072.json \
  --fit-x E \
  --data-sources instruct \
  --warmup-clip 3 \
  --fit-curve N \
  --fit-metric ErrRate \
  --plot-curve N \
  --plot-x E \
  --eval holdout_score \
  --plot-metric ErrRate \
  --plot-x-scale log \
  --plot-y-scale log \
  --y-tick-spacing 0.1 \
  --y-tick-format auto \
  --y-grid-spacing 0.1 \
  --line-alpha 1 \
  --line-width 2.0 \
  --plot-use-legend \
  --scatter-alpha 0.5 \
  --scatter-size 20 \
  --scatter-marker o \
  --plot-title "Fitted L(N,D) on Instruct Model with Inter-model Prediction" \
  --output-prefix extrap_

echo ""
echo "========================================="
echo "Complete! Generated plots:"
echo "  - outputs/extrap_base_holdout_N_C_raw_ErrRate.pdf"
echo "  - outputs/extrap_base_holdout_N_E_ErrRate.pdf"
echo "  - outputs/extrap_instruct_holdout_N_C_raw_ErrRate.pdf"
echo "  - outputs/extrap_instruct_holdout_N_E_ErrRate.pdf"
echo "========================================="
