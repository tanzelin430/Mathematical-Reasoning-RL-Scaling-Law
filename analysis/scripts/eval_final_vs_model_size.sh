#!/bin/bash

# Plot final test loss vs model size for cross-domain evaluations
# Creates 2x4 subplot grids with all 8 evaluations
# Generates separate PDFs for base and instruct models

echo "=== Run 1/2: Base Model (8 subplots) ==="
uv run -m src.run.eval_final_vs_model_size \
  --metric ErrRate \
  --data-source base \
  --warmup-clip 0

echo ""
echo "=== Run 2/2: Instruct Model (8 subplots) ==="
uv run -m src.run.eval_final_vs_model_size \
  --metric ErrRate \
  --data-source instruct \
  --warmup-clip 4

echo ""
echo "All runs completed!"
echo "Generated:"
echo "  - crossdomain_base_ErrRate_vs_N.pdf (8 subplots)"
echo "  - crossdomain_instruct_ErrRate_vs_N.pdf (8 subplots)"
