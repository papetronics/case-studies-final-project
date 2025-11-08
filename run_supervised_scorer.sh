#!/usr/bin/env bash
set -e

# Build and run the supervised scorer scenario
./build_image.sh B_supervised_scorer \
  --epochs 50 \
  --episodes-per-batch 32 \
  --log-dir /workspace/logs \
  --hidden-size 256 \
  --num-hidden 4 \
  --dataset-size 5000
