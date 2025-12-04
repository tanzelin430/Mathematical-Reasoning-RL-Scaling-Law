#!/bin/bash

# Analyze uncertainty in scaling law fits
# Generates bar plots with error bars and LaTeX tables

echo "=== Analyzing Uncertainty Metrics ==="
uv run python scripts/analyze_uncertainty.py

echo ""
echo "Outputs generated:"
echo "  - outputs/uncertainty_base_holdout_score.pdf"
echo "  - outputs/uncertainty_instruct_holdout_score.pdf"
echo "  - outputs/uncertainty_tables.tex"
