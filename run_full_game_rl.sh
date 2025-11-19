#!/usr/bin/env bash
set -e

# Build and run the full game RL scenario
./build_image.sh yahtzee_agent \
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
  --phi-features dice_onehot,dice_counts,rolls_used,phase,has_earned_yahtzee,available_categories,percent_progress_towards_bonus,potential_scoring_opportunities,game_progress \
  --entropy-coeff-rolling-max 0.05 \
  --entropy-coeff-rolling-min 0.0 \
  --entropy-coeff-scoring-max 0.05 \
  --entropy-coeff-scoring-min 0.0 \
  --entropy-hold-period 0.4 \
  --entropy-anneal-period 0.4 \
  --critic-coeff 0.05
