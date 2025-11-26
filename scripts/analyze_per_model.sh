#!/bin/bash

# Automated Per-Model K(N), E0(N), and R² Analysis
#
# This script analyzes the hybrid fit results (fits_hybrid_kn_e072.json)
# and generates LaTeX tables with per-model K(N), E0(N), and R² values.
#
# Prerequisites:
#   - outputs/fits_hybrid_kn_e072.json must exist
#   - Run ./scripts/exp1_fits_save_up32B.sh first if needed
#   - Run python scripts/optimize_e072_compensation.py if needed
#
# Output files:
#   - outputs/per_model_compact_tables.tex (main output for papers)
#   - outputs/per_model_r2_latex.txt (detailed tables)

echo "==================================================================="
echo "Per-Model K(N), E0(N), and R² Analysis"
echo "==================================================================="
echo ""

# Check prerequisites
if [ ! -f "outputs/fits_hybrid_kn_e072.json" ]; then
    echo "Error: outputs/fits_hybrid_kn_e072.json not found!"
    echo "Please run the following commands first:"
    echo "  1. ./scripts/exp1_fits_save_up32B.sh"
    echo "  2. uv run python scripts/optimize_e072_compensation.py"
    exit 1
fi

echo "Running per-model analysis..."
uv run python scripts/analyze_per_model_fit.py > outputs/per_model_analysis_output.txt 2>&1

if [ $? -eq 0 ]; then
    echo "✓ Analysis completed successfully!"
    echo ""
    echo "Output files generated:"
    echo "  - outputs/per_model_compact_tables.tex"
    echo "  - outputs/per_model_r2_latex.txt"
    echo "  - outputs/per_model_analysis_output.txt (full output)"
    echo ""

    # Extract compact tables to separate file
    echo "Extracting compact tables..."
    sed -n '/^\\begin{table}\[H\]/,/^\\end{table}$/p' outputs/per_model_analysis_output.txt | \
        sed '/^$/d' > outputs/per_model_compact_tables.tex

    echo "✓ Compact tables extracted to outputs/per_model_compact_tables.tex"
    echo ""
    echo "==================================================================="
    echo "Summary:"
    echo "==================================================================="

    # Show key statistics
    grep "^base\|^instruct" outputs/per_model_analysis_output.txt | head -14

else
    echo "✗ Error: Analysis failed!"
    echo "Check outputs/per_model_analysis_output.txt for details"
    exit 1
fi
