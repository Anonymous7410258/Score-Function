#!/bin/bash
# =============================================================================
# Meta-TTA-TSM: Dataset Download Script
# =============================================================================
# This script prepares dataset directories.
#
# The primary experiments use synthetic datasets generated programmatically
# (see data/datasets.py). For real-world datasets (financial, biological),
# place your data files in data/raw/ and run the preprocessing pipeline.
#
# Usage:
#   bash scripts/download_dataset.sh
# =============================================================================

set -e

echo "============================================="
echo "  Meta-TTA-TSM: Dataset Preparation"
echo "============================================="

# Create data directories
mkdir -p data/raw
mkdir -p data/processed

echo "Data directories created:"
echo "  data/raw/       — Place raw data files here"
echo "  data/processed/ — Processed data will be stored here"
echo ""
echo "Synthetic datasets (Gaussian, ICA, GGM) are generated"
echo "automatically during training. No download needed."
echo ""
echo "For real-world datasets:"
echo "  1. Place files in data/raw/"
echo "  2. Run: python -m utils.preprocessing --input data/raw/ --output data/processed/"
echo ""
echo "Done!"
