#!/usr/bin/env bash
set -e

IMAGE_NAME="csml-final-project"

# Check if we got the required arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <python_module> [additional_args...]"
    echo "Example: $0 yahtzee_agent --epochs 3 --log-dir /workspace/logs"
    exit 1
fi

PYTHON_MODULE="$1"
shift  # Remove the first argument, leaving the rest as additional args

echo "🔨 Building local dev image..."
docker build -f Dockerfile.wandb -t $IMAGE_NAME .
echo "✅ Image built successfully: $IMAGE_NAME"

echo "🚀 Running container..."

# Ensure logs and checkpoints directories exist with correct permissions
mkdir -p "$(pwd)/logs"
mkdir -p "$(pwd)/checkpoints"

# Determine if we need interactive mode (for test scripts)
DOCKER_FLAGS="--rm --gpus all"
if [[ "$PYTHON_MODULE" == *"test"* ]] || [[ "$*" == *"--mode test"* ]]; then
    DOCKER_FLAGS="$DOCKER_FLAGS -it"
fi

# Run container with user mapping and proper volume permissions
docker run $DOCKER_FLAGS \
  -u "$(id -u):$(id -g)" \
  -v "$(pwd)/logs":/workspace/logs \
  -v "$(pwd)/checkpoints":/opt/ml/checkpoints \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -e WANDB_MODE=offline \
  -e HOME=/tmp \
  $IMAGE_NAME \
  python -m "$PYTHON_MODULE" "$@"