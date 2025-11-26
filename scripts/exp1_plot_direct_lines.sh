#!/bin/bash

# Plot Qwen2.5 base and instruct curves by directly connecting scatter points
# No fitting curves - just raw data points connected with lines

echo "=== Plotting Direct Lines: Base vs Instruct Comparison ==="
uv run -m src.scaling_analysis \
  --plot \
  --plot-merge-sources \
  --plot-extra-lines scripts/sota_lines.json \
  --data-sources base instruct \
  --plot-curve E \
  --plot-source-curve-mask '{"base": [52736], "instruct": [52736]}' \
  --plot-source-curve-label '{"base": {"52736": "Ours-Qwen2.5-(Base, Dense)"}, "instruct": {"52736": "Ours-Qwen2.5-(Instruct, Dense)"}}' \
  --plot-source-curve-color '{"base": {"52736": "#CC0000"}, "instruct": {"52736": "#CC6600"}}' \
  --plot-x N \
  --eval holdout_score \
  --plot-metric ErrRate \
  --plot-use-scatter \
  --plot-use-line \
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
  --output-prefix direct_lines_

echo ""
echo "Plot saved with prefix: direct_lines_"
