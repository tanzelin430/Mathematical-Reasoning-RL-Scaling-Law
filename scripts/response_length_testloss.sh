#!/bin/bash

# Generated from params/response_length_testloss.json
# This script contains 2 runs

echo "=== Run 1/2: Response Length - loss - base ==="
uv run -m src.scaling_analysis \
  --plot \
  --data-sources base \
  --curve N \
  -x response_length \
  --eval holdout_score \
  --metric ErrRate \
  --warmup-clip 0 \
  --fit \
  --fit-model loglinear \
  --fit-x response_length \
  --fit-metric ErrRate \
  --plot-x-label "Response Length" \
  --plot-y-label "Test Loss" \
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
  --plot-title "Fitted Test Loss vs Response Length (Base)" \
  --scatter-alpha 1 \
  --scatter-size 10 \
  --scatter-marker o \
  --output-prefix response_fit_

echo ""
echo "=== Run 2/2: Response Length - loss - instruct ==="
uv run -m src.scaling_analysis \
  --plot \
  --data-sources instruct \
  --curve N \
  -x response_length \
  --eval holdout_score \
  --metric ErrRate \
  --warmup-clip 0 \
  --fit \
  --fit-model loglinear \
  --fit-x response_length \
  --fit-metric ErrRate \
  --plot-x-label "Response Length" \
  --plot-y-label "Test Loss" \
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
  --output-prefix response_fit_

echo ""
echo "All runs completed!"

