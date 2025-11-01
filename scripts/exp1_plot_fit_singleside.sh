#!/bin/bash

# load fitting results and plot fitting curves
# always use L(N, C) to plot x = Compute & x = Data Size

echo "=== Run 1/1: fixE ==="
uv run -m src.scaling_analysis \
  --plot \
  --plot-fit \
  --plot-extra-lines scripts/sota_lines.json \
  --fit-load outputs/fits_exp1.json \
  --fit-x C_raw \
  --data-sources instruct \
  --fit-curve N \
  --plot-curve E \
  --plot-curve-mask 53760 \
  --plot-x N \
  --eval holdout_score \
  --plot-metric ErrRate \
  --fit-metric ErrRate \
  --plot-use-scatter \
  --plot-x-scale log \
  --plot-y-scale log \
  --y-tick-spacing 0.1 \
  --y-tick-format auto \
  --y-grid-spacing 0.1 \
  --line-alpha 1 \
  --line-width 2.0 \
  --plot-use-legend \
  --plot-title "Single Side Fitting" \
  --scatter-alpha 1 \
  --scatter-size 15 \
  --scatter-marker o \
  --output-prefix fit_singleside_
