#!/usr/bin/env bash
set -e

# Build and run the full game RL scenario
./build_image.sh C_single_turn_score_maximizer \
  --game-scenario full_game \
  --epochs 500 \
  --total-train-games 260000 \
  --games-per-batch 26 \
  --log-dir /workspace/logs \
  --hidden-size 384 \
  --num-hidden 4 \
  --activation-function Swish \
  --learning-rate 0.0005 \
  --min-lr-ratio 0.01 \
  --gamma-min 1.0 \
  --gamma-max 1.0 \
  --dropout-rate 0.1 \
  --phi-features potential_scoring_opportunities \
  --entropy-coeff-start 0.05 \
  --entropy-coeff-end 0.0 \
  --entropy-anneal-percentage 0.4 \
  --critic-coeff 0.05
