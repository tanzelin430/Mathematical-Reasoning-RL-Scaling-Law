#!/bin/bash

# Convert all PDF files in outputs/ directory to PNG format
# Uses PyMuPDF for high-quality conversion

set -e

echo "Converting PDFs to PNG format..."
echo

uv run python scripts/convert_pdfs_to_png.py

echo
echo "Conversion complete!"
