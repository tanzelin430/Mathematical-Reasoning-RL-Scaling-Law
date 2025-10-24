#!/bin/bash

# Generated from params/exp2_smooth.json
# This script contains 2 runs (smoothing without fitting)

echo "=== Run 1/2: figure 7a ==="
uv run -m src.scaling_analysis \
  --plot \
  --data-sources exp2-base \
  --curve Tau \
  -x step \
  --eval holdout_score \
  --metric ErrRate \
  --add-smooth \
  --smooth-monotonic \
  --s-factor 1 \
  --k-spline 3 \
  --rolling-window 200 \
  --min-se 1e-3 \
  --x-inv-weight-power 0.2 \
  --warmup-clip-to 10 \
  --ending-clip-to 100 \
  --plot-x-scale log \
  --plot-y-scale log \
  --x-tick-spacing 0.2 \
  --x-tick-format decimal \
  --x-grid-spacing 0.2 \
  --y-tick-spacing 0.1 \
  --y-tick-format auto \
  --line-alpha 1 \
  --line-width 2 \
  --scatter-alpha 0.5 \
  --scatter-size 15 \
  --scatter-marker o \
  --plot-use-legend \
  --plot-title "Data Reuse on Base model (7B)" \
  --curve-mask 1 2 5 20 25 50 100 \
  --highlight-curves-predict 1 \
  --highlight-width 3.0 \

echo ""
echo "=== Run 2/2: figure 7b ==="
uv run -m src.scaling_analysis \
  --plot \
  --data-sources exp2-instruct \
  --curve Tau \
  -x step \
  --eval holdout_score \
  --metric ErrRate \
  --add-smooth \
  --s-factor 1 \
  --k-spline 3 \
  --rolling-window 200 \
  --min-se 1e-3 \
  --x-inv-weight-power 0.2 \
  --warmup-clip-to 10 \
  --ending-clip-to 100 \
  --plot-x-scale log \
  --plot-y-scale log \
  --x-grid-spacing 0.2 \
  --x-tick-spacing 0.2 \
  --x-tick-format decimal \
  --y-tick-spacing 0.1 \
  --y-tick-format auto \
  --line-alpha 1 \
  --line-width 2 \
  --scatter-alpha 0.3 \
  --scatter-size 15 \
  --scatter-marker o \
  --plot-use-legend \
  --plot-title "Data Reuse on Instruct model (7B)" \
  --curve-mask 1 2 5 20 25 50 100 \
  --highlight-curves-predict 1 \
  --highlight-width 3.0 \

echo ""
echo "All runs completed!"

