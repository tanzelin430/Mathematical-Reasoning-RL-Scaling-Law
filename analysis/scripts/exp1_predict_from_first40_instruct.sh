#!/bin/bash

# Prediction workflow for Instruct model: Fit on first 40 points (after warmup), predict the rest
# Configuration:
# - warmup-clip: 3 (remove first 3 warmup points)
# - ending-clip: 65 (use first 40 points: 105 - 65 = 40)
# - Fitted segment shown with solid line, predicted segment with dashed line
# - Plotting also uses warmup-clip 3 to hide first 3 points

# Step 1: Fit on first 40 points (all model sizes: 0.5B-72B, after removing first 3)
echo "=== Step 1/4: Fitting L(N, C) on Instruct (first 40 points, warmup-clip 3) ==="
uv run -m src.scaling_analysis \
  --warmup-clip 3 \
  --ending-clip 65 \
  --data-sources instruct \
  --eval holdout_score \
  --fit \
  --fit-model loglinear_kn \
  --fit-curve N \
  --fit-x C_raw \
  --fit-metric ErrRate \
  --fit-save outputs/fits_predict_instruct40.json \
  --x-inv-weight-power 0

echo ""
echo "=== Step 2/4: Fitting L(N, E) on Instruct (first 40 points, warmup-clip 3) ==="
uv run -m src.scaling_analysis \
  --warmup-clip 3 \
  --ending-clip 65 \
  --data-sources instruct \
  --eval holdout_score \
  --fit \
  --fit-model loglinear_kn \
  --fit-curve N \
  --fit-x E \
  --fit-metric ErrRate \
  --fit-save-append outputs/fits_predict_instruct40.json \
  --x-inv-weight-power 0

echo ""
echo "=== Step 3/4: Plotting L(N, C) prediction ==="
uv run -m src.scaling_analysis \
  --plot \
  --plot-fit \
  --fit-load outputs/fits_predict_instruct40.json \
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
  --plot-title "Fitted L(N,C) on Instruct Model with Intra-model Prediction" \
  --output-prefix predict_instruct40_

echo ""
echo "=== Step 4/4: Plotting L(N, E) prediction ==="
uv run -m src.scaling_analysis \
  --plot \
  --plot-fit \
  --fit-load outputs/fits_predict_instruct40.json \
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
  --plot-title "Fitted L(N,D) on Instruct Model with Intra-model Prediction" \
  --output-prefix predict_instruct40_

echo ""
echo "========================================="
echo "Prediction workflow complete for Instruct model!"
echo "Generated files:"
echo "  - outputs/fits_predict_instruct40.json"
echo "  - outputs/predict_instruct40_instruct_holdout_N_C_raw_ErrRate.pdf"
echo "  - outputs/predict_instruct40_instruct_holdout_N_E_ErrRate.pdf"
echo ""
echo "Note: Solid lines = fitted region (first 40 points after removing first 3)"
echo "      Dashed lines = predicted region (remaining points)"
echo "      First 3 warmup points are hidden in plots"
echo "========================================="
