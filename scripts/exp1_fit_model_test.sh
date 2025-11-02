#!/bin/bash

# test fitting model on L(N, C)

echo "=== Run 1/1: Test fitting model ==="

args=(
  # Data clip
  --warmup-clip 4
  --ending-clip 0
  
  # Data sources and evaluation
  # --data-sources base
  --data-sources instruct
  --eval holdout_score
  
  # Fit configuration
  --fit
  # --fit-model loglinear # change model here
  # --fit-model invexp # change model here
  --fit-model loglinear_kn # change model here
  # --fit-model powlawplus # change model here
  --fit-curve N
  --fit-x C_raw
  --fit-metric ErrRate
  --fit-save outputs/fits_exp1_test.json
  --x-inv-weight-power 0
  --fit-curve-mask 0.5e9 1.5e9 3e9 7e9 14e9 32e9 72e9
  
  # Plot configuration
  --plot
  --plot-fit
  --plot-x C_raw
  --plot-curve N
  --plot-metric ErrRate
  
  # Plot styling
  --plot-x-scale log
  --plot-y-scale log
  --y-tick-spacing 0.1
  --y-tick-format auto
  --y-grid-spacing 0.1
  
  # Line and scatter styling
  --line-alpha 1
  --line-width 2.0
  --plot-use-legend
  --plot-title "Fitted L(N, C)"
  --scatter-alpha 0.3
  --scatter-size 15
  --scatter-marker o
  
  # Output
  --output-prefix fit_
)

# Run the command with all arguments
uv run -m src.scaling_analysis "${args[@]}"