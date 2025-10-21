#!/bin/bash

# Generated from params/test_saveload.json
# This script contains 2 runs

echo "=== Run 1/2: Fitting C - Test Save ==="
uv run -m src.run.plot_multi_fit \
  --data-source base \
  --plot-curve N \
  -x C_raw \
  --eval holdout_score \
  --metric ErrRate \
  --warmup-clip-frac 0.1 \
  --fit \
  --fit-plot-params \
  --fit-curve N \
  --plot-curve-mask 0.5e9 1.5e9 3e9 7e9 14e9 32e9 72e9 \
  --fit-x C_raw \
  --fit-metric ErrRate \
  --fit-save outputs/test_fitter_save.json \
  --plot-x-scale log \
  --plot-y-scale log \
  --y-tick-spacing 0.1 \
  --y-tick-format auto \
  --y-grid-spacing 0.1 \
  --line-alpha 1 \
  --line-width 2.0 \
  --plot-use-legend \
  --plot-title "Fitted L(N, C) on Base models" \
  --scatter-alpha 0.3 \
  --scatter-size 15 \
  --scatter-marker o \
  --output-prefix fit_base_save_

echo ""
echo "=== Run 2/2: Load saved model - Test Load ==="
uv run -m src.run.plot_multi_fit \
  --data-source base \
  --plot-curve N \
  -x C_raw \
  --eval holdout_score \
  --metric ErrRate \
  --warmup-clip-frac 0.1 \
  --fit \
  --fit-plot-params \
  --fit-curve N \
  --plot-curve-mask 0.5e9 1.5e9 3e9 7e9 14e9 32e9 72e9 \
  --fit-x C_raw \
  --fit-metric ErrRate \
  --fit-load outputs/test_fitter_save.json \
  --plot-x-scale log \
  --plot-y-scale log \
  --y-tick-spacing 0.1 \
  --y-tick-format auto \
  --y-grid-spacing 0.1 \
  --line-alpha 1 \
  --line-width 2.0 \
  --plot-use-legend \
  --plot-title "Loaded Model L(N, C) on Base models" \
  --scatter-alpha 0.3 \
  --scatter-size 15 \
  --scatter-marker o \
  --output-prefix fit_base_load_

echo ""
echo "All runs completed!"

