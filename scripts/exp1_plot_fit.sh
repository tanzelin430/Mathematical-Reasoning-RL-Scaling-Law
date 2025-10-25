#!/bin/bash

# load fitting results and plot fitting curves
# always use L(N, C) to plot x = Compute & x = Data Size

echo "=== Run 1/4: figure 1a & 2a ==="
uv run -m src.scaling_analysis \
  --plot \
  --plot-fit \
  --fit-load outputs/fits_exp1.json \
  --fit-x C_raw \
  --data-sources base \
  --curve N \
  -x C_raw \
  --eval holdout_score \
  --metric ErrRate \
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
  --output-prefix fit_

echo ""
echo "=== Run 2/4: figure 1b & 3a ==="

uv run -m src.scaling_analysis \
  --plot \
  --plot-fit \
  --fit-load outputs/fits_exp1.json \
  --fit-x C_raw \
  --data-sources base \
  --curve N \
  -x E \
  --eval holdout_score \
  --metric ErrRate \
  --plot-x-scale log \
  --plot-y-scale log \
  --y-tick-spacing 0.1 \
  --y-tick-format auto \
  --y-grid-spacing 0.1 \
  --line-alpha 1 \
  --line-width 2.0 \
  --plot-use-legend \
  --plot-title "Fitted L(N, D) on Base models" \
  --scatter-alpha 0.3 \
  --scatter-size 15 \
  --scatter-marker o \
  --output-prefix fit_

echo ""
echo "=== Run 3/4: figure 2a ==="
uv run -m src.scaling_analysis \
  --plot \
  --plot-fit \
  --fit-load outputs/fits_exp1.json \
  --fit-x C_raw \
  --data-sources instruct \
  --curve N \
  -x C_raw \
  --eval holdout_score \
  --metric ErrRate \
  --plot-x-scale log \
  --plot-y-scale log \
  --y-tick-spacing 0.1 \
  --y-tick-format auto \
  --y-grid-spacing 0.1 \
  --line-alpha 1 \
  --line-width 2.0 \
  --plot-use-legend \
  --plot-title "Fitted L(N, C) on Instruct models" \
  --scatter-alpha 0.3 \
  --scatter-size 15 \
  --scatter-marker o \
  --output-prefix fit_

echo ""
echo "=== Run 4/4: figure 3b ==="
uv run -m src.scaling_analysis \
  --plot \
  --plot-fit \
  --fit-load outputs/fits_exp1.json \
  --fit-x C_raw \
  --data-sources instruct \
  --curve N \
  -x E \
  --eval holdout_score \
  --metric ErrRate \
  --plot-x-scale log \
  --plot-y-scale log \
  --y-tick-spacing 0.1 \
  --y-tick-format auto \
  --y-grid-spacing 0.1 \
  --line-alpha 1 \
  --line-width 2.0 \
  --plot-use-legend \
  --plot-title "Fitted L(N, D) on Instruct models" \
  --scatter-alpha 0.3 \
  --scatter-size 15 \
  --scatter-marker o \
  --output-prefix fit_


echo ""
echo "All runs completed!"

