#!/usr/bin/env bash
set -e

IMAGE_NAME="csml-final-project"

# Change to the project root directory
cd ..

echo "🔨 Building local dev image..."
docker build -f Dockerfile.wandb -t $IMAGE_NAME .

echo "🚀 Running container..."

# Ensure logs directory exists with correct permissions
mkdir -p "$(pwd)/logs"

# Run container with user mapping and proper volume permissions
docker run --rm \
  --gpus all \
  -u "$(id -u):$(id -g)" \
  -v "$(pwd)/logs":/workspace/logs \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -e WANDB_MODE=offline \
  -e HOME=/tmp \
  $IMAGE_NAME \
  --epochs 500 \
  --episodes-per-batch 52 \
  --log-dir /workspace/logs \
  --scenario single_turn_score_maximizer \
  --hidden-size 384 \
  --num-hidden 3 \
  --dataset-size 1300 \
  --activation-function Swish \
  --learning-rate 0.00075 \
  --min-lr-ratio 0.0001 \
  --gamma-min 0.0 \
  --gamma-max 1.0

echo "✅ Done."
