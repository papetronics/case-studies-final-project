#!/usr/bin/env bash
set -e

# Build and run the single turn RL scenario
./build_image.sh yahtzee_agent \
  --game-scenario single_turn \
  --epochs 500 \
  --total-train-games 260000 \
  --games-per-batch 26 \
  --log-dir /workspace/logs \
  --hidden-size 384 \
  --num-hidden 4 \
  --activation-function Swish \
  --learning-rate 0.0005 \
  --min-lr-ratio 0.01 \
  --gamma-min 0.9 \
  --gamma-max 1.0 \
  --dropout-rate 0.1 \
  --entropy-coeff-rolling-max 0.15 \
  --entropy-coeff-rolling-min 0.045 \
  --entropy-coeff-scoring-max 0.3 \
  --entropy-coeff-scoring-min 0.006 \
  --entropy-hold-period 0.4 \
  --entropy-anneal-period 0.35 \
  --critic-coeff 0.05
