#!/bin/bash

echo "=== Run 1/2: figure 7a ==="
uv run -m src.scaling_analysis \
  --plot \
  --plot-fit \
  --data-sources exp2-base \
  --plot-curve Tau \
  --plot-x step \
  --eval holdout_score \
  --plot-metric ErrRate \
  --warmup-clip-to 10 \
  --ending-clip-to 100 \
  --fit \
  --fit-model loglinear-tau \
  --fit-curve Tau \
  --fit-x step \
  --fit-metric ErrRate \
  --fit-curve-mask 1 2 5 20 25 50 100 \
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
  --plot-curve-mask 1 2 5 20 25 50 100 \
  --highlight-curves-predict 1 \
  --highlight-width 3.0 \
  --output-prefix fit_

echo ""
echo "=== Run 2/2: figure 7b ==="
uv run -m src.scaling_analysis \
  --plot \
  --plot-fit \
  --data-sources exp2-instruct \
  --plot-curve Tau \
  --plot-x step \
  --eval holdout_score \
  --plot-metric ErrRate \
  --warmup-clip-to 10 \
  --ending-clip-to 100 \
  --fit \
  --fit-model loglinear-tau \
  --fit-curve Tau \
  --fit-x step \
  --fit-metric ErrRate \
  --fit-curve-mask 1 2 5 20 25 50 100 \
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
  --plot-curve-mask 1 2 5 20 25 50 100 \
  --highlight-curves-predict "1" \
  --highlight-width 3.0 \
  --output-prefix fit_

echo ""
echo "All runs completed!"

