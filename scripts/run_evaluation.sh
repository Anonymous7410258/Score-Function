#!/bin/bash
# =============================================================================
# Meta-TTA-TSM: Evaluation Script
# =============================================================================
# Evaluate trained model across missingness regimes.
#
# Usage:
#   bash scripts/run_evaluation.sh
# =============================================================================

set -e

CONFIG="configs/config.yaml"
CHECKPOINT="checkpoints/best_model.pt"
DATASET="gaussian"
DEVICE="auto"
OUTPUT_DIR="results"
N_SEEDS=5

echo "============================================="
echo "  Meta-TTA-TSM: Evaluation"
echo "============================================="
echo "  Dataset:    ${DATASET}"
echo "  Checkpoint: ${CHECKPOINT}"
echo "  Seeds:      ${N_SEEDS}"
echo "============================================="

python -m evaluation.evaluate \
    --config ${CONFIG} \
    --checkpoint ${CHECKPOINT} \
    --dataset ${DATASET} \
    --missing-rates 0.2 0.4 0.6 0.8 \
    --n-seeds ${N_SEEDS} \
    --device ${DEVICE} \
    --output-dir ${OUTPUT_DIR}

echo ""
echo "Evaluation complete! Results saved to: ${OUTPUT_DIR}/"
