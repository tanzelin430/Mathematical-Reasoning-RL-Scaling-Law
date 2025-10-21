#!/bin/bash

# Generated from params/response_length_k_e0.json
# This script contains 1 run

echo "=== Run 1/1: Response Length - loss ==="
uv run -m src.run.plot_multi_fit \
  --data-source instruct \
  --plot-curve N \
  -x response_length \
  --eval holdout_score \
  --metric ErrRate \
  --warmup-clip 0 \
  --fit \
  --fit-curve N \
  --fit-x response_length \
  --fit-metric ErrRate \
  --fit-plot-params \
  --plot-x-scale log \
  --plot-y-scale log \
  --x-tick-spacing 0.1 \
  --x-tick-format decimal \
  --y-tick-spacing 0.1 \
  --y-tick-format auto \
  --y-grid-spacing 0.1 \
  --line-alpha 1 \
  --line-width 2.0 \
  --plot-use-legend \
  --plot-title "Fitted Test Loss vs Response Length (Instruct)" \
  --scatter-alpha 1 \
  --scatter-size 10 \
  --scatter-marker o \
  --output-prefix fit_instruct_

echo ""
echo "All runs completed!"

