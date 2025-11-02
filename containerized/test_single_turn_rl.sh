#!/usr/bin/env bash
set -e

IMAGE_NAME="csml-final-project"

# Parse checkpoint path argument
CHECKPOINT_PATH=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --checkpoint-path)
      CHECKPOINT_PATH="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

# Extract path after /workspace/logs and combine with workspace prefix
if [[ -n "$CHECKPOINT_PATH" ]]; then
  # Remove /workspace/logs prefix if it exists, otherwise use the full path
  if [[ "$CHECKPOINT_PATH" == /workspace/logs/* ]]; then
    RELATIVE_PATH="${CHECKPOINT_PATH#/workspace/logs/}"
  else
    RELATIVE_PATH="$CHECKPOINT_PATH"
  fi
  WORKSPACE_CHECKPOINT_PATH="/workspace/$RELATIVE_PATH"
else
  WORKSPACE_CHECKPOINT_PATH="/workspace/single_turn_model_checkpoint.pth"
fi

# Change to the project root directory
cd ..

echo "🔨 Building local dev image..."
docker build -f containerized/Dockerfile.wandb -t $IMAGE_NAME .

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
  --scenario test_single_turn_rl --checkpoint-path "$WORKSPACE_CHECKPOINT_PATH"

echo "✅ Done."
