#!/bin/bash

# Generated from params/exp1_fit.json
# This script contains 4 runs

echo "=== Run 1/4: figure 1a & 2a ==="
uv run -m src.scaling_analysis \
  --plot \
  --data-sources base \
  --plot-curve N \
  --plot-x C_raw \
  --eval holdout_score \
  --plot-metric ErrRate \
  --add-smooth \
  --smooth-monotonic \
  --s-factor 1 \
  --k-spline 3 \
  --rolling-window 200 \
  --min-se 1e-3 \
  --x-inv-weight-power 0.2 \
  --plot-x-scale log \
  --plot-y-scale log \
  --y-tick-spacing 0.1 \
  --y-tick-format auto \
  --y-grid-spacing 0.1 \
  --line-alpha 1 \
  --line-width 2.0 \
  --plot-use-legend \
  --plot-title "Compute vs TestLoss on Base models" \
  --scatter-alpha 0.3 \
  --scatter-size 15 \
  --scatter-marker o \
  --output-prefix smooth_

echo ""
echo "=== Run 2/4: figure 1b & 3a ==="

uv run -m src.scaling_analysis \
  --plot \
  --data-sources base \
  --plot-curve N \
  --plot-x E \
  --eval holdout_score \
  --plot-metric ErrRate \
  --add-smooth \
  --smooth-monotonic \
  --s-factor 1 \
  --k-spline 3 \
  --rolling-window 200 \
  --min-se 1e-3 \
  --x-inv-weight-power 0.2 \
  --plot-x-scale log \
  --plot-y-scale log \
  --y-tick-spacing 0.1 \
  --y-tick-format auto \
  --y-grid-spacing 0.1 \
  --line-alpha 1 \
  --line-width 2.0 \
  --plot-use-legend \
  --plot-title "Datasize vs TestLoss on Base models" \
  --scatter-alpha 0.3 \
  --scatter-size 15 \
  --scatter-marker o \
  --output-prefix smooth_

echo ""
echo "=== Run 3/4: figure 2a ==="
uv run -m src.scaling_analysis \
  --plot \
  --data-sources instruct \
  --plot-curve N \
  --plot-x C_raw \
  --eval holdout_score \
  --plot-metric ErrRate \
  --add-smooth \
  --smooth-monotonic \
  --s-factor 1 \
  --k-spline 3 \
  --rolling-window 200 \
  --min-se 1e-3 \
  --x-inv-weight-power 0.2 \
  --plot-x-scale log \
  --plot-y-scale log \
  --y-tick-spacing 0.1 \
  --y-tick-format auto \
  --y-grid-spacing 0.1 \
  --line-alpha 1 \
  --line-width 2.0 \
  --plot-use-legend \
  --plot-title "Compute vs TestLoss on Instruct models" \
  --scatter-alpha 0.3 \
  --scatter-size 15 \
  --scatter-marker o \
  --output-prefix smooth_

echo ""
echo "=== Run 4/4: figure 3b ==="
uv run -m src.scaling_analysis \
  --plot \
  --data-sources instruct \
  --plot-curve N \
  --plot-x E \
  --eval holdout_score \
  --plot-metric ErrRate \
  --add-smooth \
  --smooth-monotonic \
  --s-factor 1 \
  --k-spline 3 \
  --rolling-window 200 \
  --min-se 1e-3 \
  --x-inv-weight-power 0.2 \
  --plot-x-scale log \
  --plot-y-scale log \
  --y-tick-spacing 0.1 \
  --y-tick-format auto \
  --y-grid-spacing 0.1 \
  --line-alpha 1 \
  --line-width 2.0 \
  --plot-use-legend \
  --plot-title "Datasize vs TestLoss on Instruct models" \
  --scatter-alpha 0.3 \
  --scatter-size 15 \
  --scatter-marker o \
  --output-prefix smooth_


echo ""
echo "All runs completed!"

