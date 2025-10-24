#!/bin/bash

# Fitting C
# Generated from params/test.json
# 
# Available fit models: LogLinear, InvExp, InvExpKLinear, InvExpKQuadLog, InvExpKExp
# Note: fit_model is required when --fit is enabled, but not specified in test.json
# Using InvExp as default here

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
