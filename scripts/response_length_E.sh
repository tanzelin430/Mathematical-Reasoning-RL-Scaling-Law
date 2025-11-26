#!/bin/bash

# Generated from params/response_length_E.json
# This script contains 2 runs

echo "=== Run 1/2: Response Length - base ==="
uv run -m src.scaling_analysis \
  --plot \
  --data-sources base \
  --plot-curve N \
  --plot-x E \
  --eval response_length \
  --plot-metric R \
  --add-smooth \
  --s-factor 500 \
  --k-spline 3 \
  --rolling-window 70 \
  --min-se 1e-6 \
  --x-inv-weight-power 2 \
  --warmup-clip 10 \
  --ending-clip 5 \
  --plot-x-scale log \
  --x-tick-spacing 0.5 \
  --x-tick-format sci \
  --plot-y-label "Response Length" \
  --line-alpha 1 \
  --line-width 2.0 \
  --plot-use-legend \
  --plot-title "Response Length vs D (Base)" \
  --scatter-alpha 0.8 \
  --scatter-size 15 \
  --scatter-marker o \
  --output-prefix response_

echo ""
echo "=== Run 2/2: Response Length - instruct ==="
uv run -m src.scaling_analysis \
  --plot \
  --data-sources instruct \
  --plot-curve N \
  --plot-x E \
  --eval response_length \
  --plot-metric R \
  --add-smooth \
  --s-factor 400 \
  --k-spline 3 \
  --rolling-window 30 \
  --min-se 1e-6 \
  --x-inv-weight-power 2 \
  --warmup-clip 10 \
  --ending-clip 5 \
  --plot-x-scale log \
  --x-tick-spacing 0.5 \
  --x-tick-format sci \
  --plot-y-label "Response Length" \
  --line-alpha 1 \
  --line-width 2.0 \
  --plot-use-legend \
  --plot-title "Response Length vs D (Instruct)" \
  --scatter-alpha 0.8 \
  --scatter-size 15 \
  --scatter-marker o \
  --output-prefix response_

echo ""
echo "All runs completed!"

