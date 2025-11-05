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
  python /workspace/src/B_supervised_scorer/main.py \
  --epochs 50 --episodes-per-batch 32 --log-dir /workspace/logs --hidden-size 256 --num-hidden 4 --dataset-size 5000

echo "✅ Done."
