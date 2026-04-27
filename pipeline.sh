#!/bin/bash

# TGSAM-2 Complete Pipeline Script
# Trains model and evaluates on test set in a single run
# Usage: ./pipeline.sh [GPU_ID] [DATASET] [NUM_EPOCHS]
# Example: ./pipeline.sh 0 acdc 50
#          ./pipeline.sh 1 spleen 30

set -e  # Exit on any error

# Parse arguments
GPU_ID=${1:-0}
DATASET=${2:-acdc}
NUM_EPOCHS=${3:-50}

# Set CUDA device
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "=========================================="
echo "TGSAM-2 Pipeline"
echo "=========================================="
echo "GPU_ID: $GPU_ID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Dataset: $DATASET"
echo "Num Epochs: $NUM_EPOCHS"
echo "=========================================="
echo ""

# Validate dataset
if [[ ! "$DATASET" =~ ^(acdc|spleen|prostate|cvc)$ ]]; then
    echo "Error: Invalid dataset. Must be one of: acdc, spleen, prostate, cvc"
    exit 1
fi

CONFIG_PATH="configs/${DATASET}.yaml"
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Error: Config file not found at $CONFIG_PATH"
    exit 1
fi

# Create output directory
CHECKPOINT_DIR="outputs/${DATASET}"
EVAL_DIR="outputs/eval/${DATASET}"
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$EVAL_DIR"

# ==================== TRAINING ====================
echo ""
echo "======== PHASE 1: TRAINING ========"
echo "Starting training on $DATASET dataset..."
echo "Config: $CONFIG_PATH"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo ""

python train.py \
    --config "$CONFIG_PATH" \
    --device "cuda:0"

# Find best checkpoint
echo ""
echo "======== Finding best checkpoint ========"
BEST_CHECKPOINT=$(ls -t "$CHECKPOINT_DIR"/best_dsc_*.pth 2>/dev/null | head -1)

if [[ -z "$BEST_CHECKPOINT" ]]; then
    echo "Error: No checkpoint found in $CHECKPOINT_DIR"
    exit 1
fi

echo "Found best checkpoint: $BEST_CHECKPOINT"
echo ""

# ==================== EVALUATION ====================
echo ""
echo "======== PHASE 2: EVALUATION ========"
echo "Starting evaluation..."
echo "Checkpoint: $BEST_CHECKPOINT"
echo "Eval dir: $EVAL_DIR"
echo ""

python evaluate.py \
    --config "$CONFIG_PATH" \
    --checkpoint "$BEST_CHECKPOINT" \
    --device "cuda:0" \
    --output_dir "$EVAL_DIR"

# ==================== RESULTS ====================
echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Results Summary:"
echo "  Training checkpoint: $BEST_CHECKPOINT"
echo "  Evaluation results: $EVAL_DIR"
echo ""
echo "View detailed results:"
echo "  cat $EVAL_DIR/summary.txt"
echo "  cat $EVAL_DIR/results.json"
echo ""
