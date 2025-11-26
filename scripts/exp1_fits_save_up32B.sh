#!/bin/bash

# Fitting L(N, C) & L(N, D) on Base & Instruct models using only 0.5B-32B data
# Compare results with full dataset (0.5B-72B)

# Available -fit-model options:
#   loglinear, loglinear_kn, invexp, powlaw, powlawmul, invexp_klinear, invexp_kquadlog, invexp_kexp

echo "=== Run 1/4: Fitting L(N, C) on Base models (0.5B-32B only) ==="
uv run -m src.scaling_analysis \
  --warmup-clip 0 \
  --ending-clip 0 \
  --data-sources base \
  --eval holdout_score \
  --fit \
  --fit-model loglinear_kn \
  --fit-curve N \
  --fit-x C_raw \
  --fit-metric ErrRate \
  --fit-save outputs/fits_exp1_up32B.json \
  --x-inv-weight-power 0 \
  --fit-curve-mask 0.5e9 1.5e9 3e9 7e9 14e9 32e9 \

echo ""
echo "=== Run 2/4: Fitting L(N, D) on Base models (0.5B-32B only) ==="

uv run -m src.scaling_analysis \
  --warmup-clip 0 \
  --ending-clip 0 \
  --data-sources base \
  --eval holdout_score \
  --fit \
  --fit-model loglinear_kn \
  --fit-curve N \
  --fit-x E \
  --fit-metric ErrRate \
  --fit-save-append outputs/fits_exp1_up32B.json \
  --x-inv-weight-power 0 \
  --fit-curve-mask 0.5e9 1.5e9 3e9 7e9 14e9 32e9 \

echo ""
echo "=== Run 3/4: Fitting L(N, C) on Instruct models (0.5B-32B only) ==="

uv run -m src.scaling_analysis \
  --warmup-clip 0 \
  --ending-clip 0 \
  --data-sources instruct \
  --eval holdout_score \
  --fit \
  --fit-model loglinear_kn \
  --fit-curve N \
  --fit-x C_raw \
  --fit-metric ErrRate \
  --fit-save-append outputs/fits_exp1_up32B.json \
  --x-inv-weight-power 0 \
  --fit-curve-mask 0.5e9 1.5e9 3e9 7e9 14e9 32e9 \

echo ""
echo "=== Run 4/4: Fitting L(N, D) on Instruct models (0.5B-32B only) ==="

uv run -m src.scaling_analysis \
  --warmup-clip 0 \
  --ending-clip 0 \
  --data-sources instruct \
  --eval holdout_score \
  --fit \
  --fit-model loglinear_kn \
  --fit-curve N \
  --fit-x E \
  --fit-metric ErrRate \
  --fit-save-append outputs/fits_exp1_up32B.json \
  --x-inv-weight-power 0 \
  --fit-curve-mask 0.5e9 1.5e9 3e9 7e9 14e9 32e9 \
