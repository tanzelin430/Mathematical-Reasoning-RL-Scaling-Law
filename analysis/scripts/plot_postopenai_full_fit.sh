#!/bin/bash

# Plot PostOpenAI full fit results (all data 0.5B-72B)

echo "========================================="
echo "Plotting PostOpenAI Full Fit Results"
echo "========================================="

echo ""
echo "=== 1/2: Base L(N, C) ==="
uv run -m src.scaling_analysis \
  --plot \
  --plot-fit \
  --fit-load outputs/fits_postopenai_base.json \
  --fit-x C_raw \
  --data-sources base \
  --warmup-clip 0 \
  --fit-curve N \
  --fit-metric ErrRate \
  --plot-curve N \
  --plot-x C_raw \
  --eval holdout_score \
  --plot-metric ErrRate \
  --plot-x-scale log \
  --plot-y-scale log \
  --y-tick-spacing 0.1 \
  --y-tick-format auto \
  --y-grid-spacing 0.1 \
  --line-alpha 1 \
  --line-width 2.0 \
  --plot-use-legend \
  --scatter-alpha 0.5 \
  --scatter-size 20 \
  --scatter-marker o \
  --plot-title "PostOpenAI Full Fit: Base" \
  --output-prefix postopenai_full_

# echo ""
# echo "=== 2/2: Instruct L(N, C) ==="
# uv run -m src.scaling_analysis \
#   --plot \
#   --plot-fit \
#   --fit-load outputs/fits_postopenai_full.json \
#   --fit-x C_raw \
#   --data-sources instruct \
#   --warmup-clip 0 \
#   --fit-curve N \
#   --fit-metric ErrRate \
#   --plot-curve N \
#   --plot-x C_raw \
#   --eval holdout_score \
#   --plot-metric ErrRate \
#   --plot-x-scale log \
#   --plot-y-scale log \
#   --y-tick-spacing 0.1 \
#   --y-tick-format auto \
#   --y-grid-spacing 0.1 \
#   --line-alpha 1 \
#   --line-width 2.0 \
#   --plot-use-legend \
#   --scatter-alpha 0.5 \
#   --scatter-size 20 \
#   --scatter-marker o \
#   --plot-title "PostOpenAI Full Fit: Instruct" \
#   --output-prefix postopenai_full_

echo ""
echo "========================================="
echo "Complete! Generated plots:"
echo "  - outputs/postopenai_full_base_holdout_N_C_raw_ErrRate.pdf"
# echo "  - outputs/postopenai_full_instruct_holdout_N_C_raw_ErrRate.pdf"
echo "========================================="
