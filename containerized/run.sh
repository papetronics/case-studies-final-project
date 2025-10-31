#!/usr/bin/env bash
set -e

IMAGE_NAME="pape-lab-local-dev"

# Change to the project root directory
cd ..

echo "🔨 Building local dev image..."
docker build -f containerized/Dockerfile.wandb -t $IMAGE_NAME .

echo "🚀 Running container..."
docker run --rm \
  --gpus all \
  -v "$(pwd)/containerized":/workspace \
  -w /workspace \
  -e WANDB_MODE=offline \
  $IMAGE_NAME \
  python main.py

echo "✅ Done."
