#!/bin/bash

# Generated from params/exp2_fit_loglinear.json
# This script contains 2 runs

echo "=== Run 1/2: figure 7a ==="
uv run -m src.run.plot_multi_fit \
  --data-source exp2-base \
  --plot-curve Tau \
  -x step \
  --eval holdout_score \
  --metric ErrRate \
  --warmup-clip 10 \
  --ending-clip 100 \
  --fit \
  --fit-curve Tau \
  --fit-x step \
  --fit-metric ErrRate \
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
  --plot-curve-mask "1,2,5,20,25,50,100" \
  --highlight-curves-predict "1" \
  --highlight-line-width 3.0 \
  --output-prefix fit_exp2-base_

echo ""
echo "=== Run 2/2: figure 7b ==="
uv run -m src.run.plot_multi_fit \
  --data-source exp2-instruct \
  --plot-curve Tau \
  -x step \
  --eval holdout_score \
  --metric ErrRate \
  --warmup-clip 10 \
  --ending-clip 100 \
  --fit \
  --fit-curve Tau \
  --fit-x step \
  --fit-metric ErrRate \
  --fit-plot-params \
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
  --plot-curve-mask "1,2,5,20,25,50,100" \
  --highlight-curves-predict "1" \
  --highlight-line-width 3.0 \
  --output-prefix fit_exp2-instruct_

echo ""
echo "All runs completed!"

