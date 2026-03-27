#!/bin/bash
# =============================================================================
# Meta-TTA-TSM: Training Script
# =============================================================================
# Run meta-training for topology-conditioned score matching.
#
# Usage:
#   bash scripts/run_training.sh
#
# Modify the variables below to customize your training run.
# =============================================================================

set -e

# --- Configuration ---
CONFIG="configs/config.yaml"
TRAINING_CONFIG="configs/training.yaml"
DATASET="gaussian"
MISSING_RATE=0.4
DEVICE="auto"
SEED=42
CHECKPOINT_DIR="checkpoints"

echo "============================================="
echo "  Meta-TTA-TSM: Meta-Training"
echo "============================================="
echo "  Dataset:      ${DATASET}"
echo "  Missing Rate: ${MISSING_RATE}"
echo "  Device:       ${DEVICE}"
echo "  Seed:         ${SEED}"
echo "============================================="

python -m training.train \
    --config ${CONFIG} \
    --training-config ${TRAINING_CONFIG} \
    --dataset ${DATASET} \
    --missing-rate ${MISSING_RATE} \
    --device ${DEVICE} \
    --seed ${SEED} \
    --checkpoint-dir ${CHECKPOINT_DIR}

echo ""
echo "Training complete! Checkpoints saved to: ${CHECKPOINT_DIR}/"
echo "View TensorBoard logs: tensorboard --logdir runs/"
