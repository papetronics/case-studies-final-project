#!/usr/bin/env bash
set -e

# Build and run the dice maximizer scenario
./build_image.sh A_dice_maximizer \
  --epochs 3 \
  --episodes-per-batch 16 \
  --log-dir /workspace/logs