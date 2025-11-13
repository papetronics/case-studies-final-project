#!/usr/bin/env bash
set -e

# Build and run the single turn RL scenario
./build_image.sh C_single_turn_score_maximizer \
  --epochs 500 \
  --games-per-epoch 100 \
  --games-per-batch 4 \
  --log-dir /workspace/logs \
  --hidden-size 384 \
  --num-hidden 3 \
  --activation-function Swish \
  --learning-rate 0.00075 \
  --min-lr-ratio 0.0001 \
  --gamma-min 0.9 \
  --gamma-max 1.0
