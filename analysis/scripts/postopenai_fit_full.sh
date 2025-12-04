#!/bin/bash

# Fit PostOpenAI model on full dataset (0.5B-72B) as baseline

echo "=== Fitting PostOpenAI on full dataset (0.5B-72B) ==="

for data_source in base; do
  echo ""
  echo "--- Fitting $data_source ---"

  # Set warmup-clip to 0 for all data sources (no clipping)
  warmup_clip=0

  args=(
    # Data clip
    --warmup-clip $warmup_clip
    --ending-clip 0

    # Data sources and evaluation
    --data-sources $data_source
    --eval holdout_score

    # Fit configuration
    --fit
    --fit-model postopenai
    --fit-curve N
    --fit-x C_raw
    --fit-metric ErrRate
    --fit-save outputs/fits_postopenai_base.json
    --x-inv-weight-power 0.6
    --fit-curve-mask 0.5e9 1.5e9 3e9 7e9 14e9 32e9 72e9  # Include all
  )

  # Run the command
  uv run -m src.scaling_analysis "${args[@]}"
done

echo ""
echo "=== All fits saved to: outputs/fits_postopenai_full.json ==="
