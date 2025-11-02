#!/bin/bash

# load fitting results and plot fitting curves
# Plot base and instruct on the same figure with different E_max values
# Matching the style of plot_multi_fit_simple_curve_fit_fixE.py

echo "=== Run 1/1: fixE - Base vs Instruct Comparison ==="
uv run -m src.scaling_analysis \
  --plot \
  --plot-fit \
  --plot-merge-sources \
  --plot-extra-lines scripts/sota_lines.json \
  --fit-load outputs/fits_exp1.json \
  --fit-x C_raw \
  --data-sources base instruct \
  --fit-curve N \
  --plot-curve E \
  --plot-source-curve-mask '{"base": [52736], "instruct": [52736]}' \
  --plot-source-curve-label '{"base": {"52736": "Ours-Qwen2.5-(Base, Dense)"}, "instruct": {"52736": "Ours-Qwen2.5-(Instruct, Dense)"}}' \
  --plot-source-curve-color '{"base": {"52736": "#CC0000"}, "instruct": {"52736": "#CC6600"}}' \
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
  --line-width 2.5 \
  --plot-use-legend \
  --plot-title "Performance Scaling of Post-RFT vs. SOTA" \
  --scatter-alpha 1 \
  --scatter-size 30 \
  --scatter-marker o \
  --output-prefix fit_singleside_
