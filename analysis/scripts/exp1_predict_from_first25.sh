#!/bin/bash

# Prediction workflow for Base model: Fit on first 25 points, predict the rest
# Configuration:
# - warmup-clip: 0 (no warmup removal)
# - ending-clip: 80 (use first 25 points: 105 - 80 = 25)
# - Fitted segment shown with solid line, predicted segment with dashed line

# Step 1: Fit on first 25 points (all model sizes: 0.5B-72B)
echo "=== Step 1/4: Fitting L(N, C) on Base (first 25 points, warmup-clip 0) ==="
uv run -m src.scaling_analysis \
  --warmup-clip 0 \
  --ending-clip 80 \
  --data-sources base \
  --eval holdout_score \
  --fit \
  --fit-model loglinear_kn \
  --fit-curve N \
  --fit-x C_raw \
  --fit-metric ErrRate \
  --fit-save outputs/fits_predict_base25.json \
  --x-inv-weight-power 0

echo ""
echo "=== Step 2/4: Fitting L(N, E) on Base (first 25 points, warmup-clip 0) ==="
uv run -m src.scaling_analysis \
  --warmup-clip 0 \
  --ending-clip 80 \
  --data-sources base \
  --eval holdout_score \
  --fit \
  --fit-model loglinear_kn \
  --fit-curve N \
  --fit-x E \
  --fit-metric ErrRate \
  --fit-save-append outputs/fits_predict_base25.json \
  --x-inv-weight-power 0

echo ""
echo "=== Step 3/4: Plotting L(N, C) prediction ==="
uv run -m src.scaling_analysis \
  --plot \
  --plot-fit \
  --fit-load outputs/fits_predict_base25.json \
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
  --plot-title "Fitted L(N,C) on Base Model with Intra-model Prediction" \
  --output-prefix predict_base25_

echo ""
echo "=== Step 4/4: Plotting L(N, E) prediction ==="
uv run -m src.scaling_analysis \
  --plot \
  --plot-fit \
  --fit-load outputs/fits_predict_base25.json \
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
  --plot-title "Fitted L(N,D) on Base Model with Intra-model Prediction" \
  --output-prefix predict_base25_

echo ""
echo "========================================="
echo "Prediction workflow complete for Base model!"
echo "Generated files:"
echo "  - outputs/fits_predict_base25.json"
echo "  - outputs/predict_base25_base_holdout_N_C_raw_ErrRate.pdf"
echo "  - outputs/predict_base25_base_holdout_N_E_ErrRate.pdf"
echo ""
echo "Note: Solid lines = fitted region (first 25 points, no warmup removal)"
echo "      Dashed lines = predicted region (remaining points)"
echo "========================================="
