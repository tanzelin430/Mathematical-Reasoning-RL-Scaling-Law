#!/bin/bash

# Generated from params/exp1_smooth.json
# This script contains 4 runs (smoothing without fitting)

echo "=== Run 1/4: figure 1a ==="
uv run -m src.run.plot_multi_fit \
  --data-source base \
  --plot-curve N \
  -x C_raw \
  --eval holdout_score \
  --metric ErrRate \
  --warmup-clip-frac 0.1 \
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
  --output-prefix smooth_base_

echo ""
echo "=== Run 2/4: figure 2a ==="
uv run -m src.run.plot_multi_fit \
  --data-source base \
  --plot-curve N \
  -x E \
  --eval holdout_score \
  --metric ErrRate \
  --warmup-clip-frac 0.1 \
  --add-smooth \
  --smooth-monotonic \
  --s-factor 1 \
  --k-spline 3 \
  --rolling-window 200 \
  --min-se 1e-3 \
  --x-inv-weight-power 0.2 \
  --plot-x-scale log \
  --plot-y-scale log \
  --x-tick-spacing 0.5 \
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
  --output-prefix smooth_base_

echo ""
echo "=== Run 3/4: figure 1b ==="
uv run -m src.run.plot_multi_fit \
  --data-source instruct \
  --plot-curve N \
  -x C_raw \
  --eval holdout_score \
  --metric ErrRate \
  --warmup-clip-frac 0.1 \
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
  --scatter-alpha 0.3 \
  --scatter-size 15 \
  --scatter-marker o \
  --plot-use-legend \
  --plot-title "Compute vs TestLoss on Instruct models" \
  --output-prefix smooth_instruct_

echo ""
echo "=== Run 4/4: figure 2b ==="
uv run -m src.run.plot_multi_fit \
  --data-source instruct \
  --plot-curve N \
  -x E \
  --eval holdout_score \
  --metric ErrRate \
  --warmup-clip-frac 0.1 \
  --add-smooth \
  --smooth-monotonic \
  --s-factor 1 \
  --k-spline 3 \
  --rolling-window 200 \
  --min-se 1e-3 \
  --x-inv-weight-power 0.2 \
  --plot-x-scale log \
  --plot-y-scale log \
  --x-tick-spacing 0.5 \
  --y-tick-spacing 0.1 \
  --y-tick-format auto \
  --y-grid-spacing 0.1 \
  --line-alpha 1 \
  --line-width 2.0 \
  --scatter-alpha 0.3 \
  --scatter-size 15 \
  --scatter-marker o \
  --plot-use-legend \
  --plot-title "Datasize vs TestLoss on Instruct models" \
  --output-prefix smooth_instruct_

echo ""
echo "All runs completed!"

