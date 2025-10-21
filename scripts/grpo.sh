#!/bin/bash

# Generated from params/grpo.json
# This script contains 4 runs (GRPO experiments)

echo "=== Run 1/4: GRPO Rollout - Base (E) ==="
uv run -m src.run.plot_multi_fit \
  --data-source grpo-base \
  --plot-curve rollout_n \
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
  --x-tick-spacing 0.5 \
  --x-grid-spacing 0.1 \
  --x-tick-format auto \
  --y-tick-spacing 0.05 \
  --y-tick-format decimal \
  --line-alpha 1 \
  --line-width 2.0 \
  --plot-use-legend \
  --plot-title " " \
  --scatter-alpha 0.8 \
  --scatter-size 10 \
  --scatter-marker o \
  --output-prefix grpo_base_

echo ""
echo "=== Run 2/4: GRPO Rollout - Base (C_raw) ==="
uv run -m src.run.plot_multi_fit \
  --data-source grpo-base \
  --plot-curve rollout_n \
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
  --x-grid-spacing 0.2 \
  --y-tick-spacing 0.05 \
  --line-alpha 1 \
  --line-width 2.0 \
  --plot-use-legend \
  --plot-title " " \
  --scatter-alpha 0.8 \
  --scatter-size 10 \
  --scatter-marker o \
  --output-prefix grpo_base_

echo ""
echo "=== Run 3/4: GRPO Rollout - Instruct (E) ==="
uv run -m src.run.plot_multi_fit \
  --data-source grpo-instruct \
  --plot-curve rollout_n \
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
  --x-tick-spacing 0.5 \
  --x-grid-spacing 0.1 \
  --x-tick-format auto \
  --y-tick-spacing 0.05 \
  --y-tick-format decimal \
  --line-alpha 1 \
  --line-width 2.0 \
  --plot-use-legend \
  --plot-title " " \
  --scatter-alpha 0.8 \
  --scatter-size 10 \
  --scatter-marker o \
  --output-prefix grpo_instruct_

echo ""
echo "=== Run 4/4: GRPO Rollout - Instruct (C_raw) ==="
uv run -m src.run.plot_multi_fit \
  --data-source grpo-instruct \
  --plot-curve rollout_n \
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
  --x-grid-spacing 0.2 \
  --y-tick-spacing 0.05 \
  --line-alpha 1 \
  --line-width 2.0 \
  --plot-use-legend \
  --plot-title " " \
  --scatter-alpha 0.8 \
  --scatter-size 10 \
  --scatter-marker o \
  --output-prefix grpo_instruct_

echo ""
echo "All runs completed!"

