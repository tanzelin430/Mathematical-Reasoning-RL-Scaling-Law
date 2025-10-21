#!/bin/bash

# Fitting C
# Generated from params/test.json
# 
# Available fit models: LogLinear, InvExp, InvExpKLinear, InvExpKQuadLog, InvExpKExp
# Note: fit_model is required when --fit is enabled, but not specified in test.json
# Using InvExp as default here

uv run -m src.run.plot_multi_fit \
  --data-source base \
  --plot-curve N \
  -x C_raw \
  --eval holdout_score \
  --metric ErrRate \
  --warmup-clip-frac 0.1 \
  --fit \
  --fit-model InvExp \
  --fit-plot-params \
  --fit-curve N \
  --fit-x C_raw \
  --fit-metric ErrRate \
  --plot-curve-mask "0.5e9,1.5e9,3e9,7e9,14e9,32e9,72e9" \
  --plot-x-scale log \
  --plot-y-scale log \
  --y-tick-spacing 0.1 \
  --y-tick-format auto \
  --y-grid-spacing 0.1 \
  --line-alpha 1 \
  --line-width 2.0 \
  --plot-use-legend \
  --plot-title "Fitted L(N, C) on Base models" \
  --scatter-alpha 0.3 \
  --scatter-size 15 \
  --scatter-marker o \
  --output-prefix fit_base_

