#!/usr/bin/env bash
set -e

IMAGE_NAME="pape-lab-local-dev"

echo "🔨 Building local dev image..."
docker build -f Dockerfile.wandb -t $IMAGE_NAME .

echo "🚀 Running container..."
docker run --rm \
  --gpus all \
  -v "$(pwd)":/workspace \
  -w /workspace \
  -e WANDB_MODE=offline \
  $IMAGE_NAME \
  python main.py

echo "✅ Done."
