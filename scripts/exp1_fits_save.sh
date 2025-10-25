#!/bin/bash

# Fitting L(N, C) & L(N, D) on Base & Instruct models, save fit result to json.
# Use exp1_plot_fit.sh to load and plot the fitting curves.

# Available -fit-model options: 
#   loglinear, invexp, powlaw, powlawmul, invexp_klinear, invexp_kquadlog, invexp_kexp

echo "=== Run 1/2: Fitting L(N, C) on Base & Instruct models ==="
uv run -m src.scaling_analysis \
  --warmup-clip 0 \
  --ending-clip 0 \
  --data-sources base instruct \
  --eval holdout_score \
  --fit \
  --fit-model loglinear \
  --curve N \
  --fit-x C_raw \
  --fit-metric ErrRate \
  --fit-save outputs/fits_exp1.json \
  --x-inv-weight-power 0 \
  --curve-mask 0.5e9 1.5e9 3e9 7e9 14e9 32e9 72e9 \

echo ""
echo "=== Run 2/2: Fitting L(N, D) on Base & Instruct models ==="

uv run -m src.scaling_analysis \
  --warmup-clip 0 \
  --ending-clip 0 \
  --data-sources base instruct \
  --eval holdout_score \
  --fit \
  --fit-model loglinear \
  --curve N \
  --fit-x E \
  --fit-metric ErrRate \
  --fit-save-append outputs/fits_exp1.json \
  --x-inv-weight-power 0 \
  --curve-mask 0.5e9 1.5e9 3e9 7e9 14e9 32e9 72e9 \
