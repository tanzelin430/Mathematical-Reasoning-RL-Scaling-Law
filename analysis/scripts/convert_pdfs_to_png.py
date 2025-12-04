#!/usr/bin/env python3
"""
Convert all PDF files in outputs/ directory to PNG format.

This script uses PyMuPDF (fitz) to convert PDFs to high-resolution PNGs
suitable for presentations and web viewing.
"""

import os
from pathlib import Path
import fitz  # PyMuPDF

def convert_pdf_to_png(pdf_path: Path, dpi: int = 300) -> None:
    """
    Convert a single PDF file to PNG format using PyMuPDF.

    Args:
        pdf_path: Path to the PDF file
        dpi: Resolution for the output PNG (default: 300 for high quality)
    """
    print(f"Converting {pdf_path.name}...")

    try:
        # Open the PDF
        pdf_document = fitz.open(pdf_path)
        num_pages = pdf_document.page_count

        # Calculate zoom factor from DPI (default PDF is 72 DPI)
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)

        # For single-page PDFs, save directly
        if num_pages == 1:
            page = pdf_document[0]
            pix = page.get_pixmap(matrix=mat)
            png_path = pdf_path.with_suffix('.png')
            pix.save(png_path)
            print(f"  ✓ Saved to {png_path.name}")
        else:
            # For multi-page PDFs, save each page separately
            for page_num in range(num_pages):
                page = pdf_document[page_num]
                pix = page.get_pixmap(matrix=mat)
                png_path = pdf_path.with_suffix(f'_page{page_num+1}.png')
                pix.save(png_path)
                print(f"  ✓ Saved page {page_num+1} to {png_path.name}")

        pdf_document.close()

    except Exception as e:
        print(f"  ✗ Error: {e}")

def main():
    """Convert all PDFs in outputs/ directory to PNG."""
    outputs_dir = Path("outputs")

    if not outputs_dir.exists():
        print(f"Error: {outputs_dir} directory not found!")
        return

    # Find all PDF files
    pdf_files = sorted(outputs_dir.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in outputs/ directory")
        return

    print(f"Found {len(pdf_files)} PDF files to convert\n")

    # Convert each PDF
    for pdf_path in pdf_files:
        convert_pdf_to_png(pdf_path)

    print(f"\n✓ Conversion complete! Generated PNG files in {outputs_dir}/")

if __name__ == "__main__":
    main()
