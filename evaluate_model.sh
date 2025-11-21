#!/usr/bin/env bash
set -e

# Parse arguments
CHECKPOINT_PATH=""
NUM_GAMES=10000
BASELINE_PATH=""
ALPHA=0.05

while [[ $# -gt 0 ]]; do
  case $1 in
    --checkpoint-path)
      CHECKPOINT_PATH="$2"
      shift 2
      ;;
    --num-games)
      NUM_GAMES="$2"
      shift 2
      ;;
    --baseline)
      BASELINE_PATH="$2"
      shift 2
      ;;
    --alpha)
      ALPHA="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check required arguments
if [[ -z "$CHECKPOINT_PATH" ]]; then
  echo "Error: --checkpoint-path is required"
  exit 1
fi

if [[ -z "$BASELINE_PATH" ]]; then
  echo "Error: --baseline is required"
  exit 1
fi

# Convert local path to container path
# If the path starts with "checkpoints/", prepend /opt/ml/
# If the path starts with "logs/", prepend /workspace/
if [[ "$CHECKPOINT_PATH" == checkpoints/* ]]; then
  CONTAINER_CHECKPOINT_PATH="/opt/ml/$CHECKPOINT_PATH"
elif [[ "$CHECKPOINT_PATH" == logs/* ]]; then
  CONTAINER_CHECKPOINT_PATH="/workspace/$CHECKPOINT_PATH"
elif [[ "$CHECKPOINT_PATH" == /workspace/* ]] || [[ "$CHECKPOINT_PATH" == /opt/ml/* ]]; then
  # Already a container path
  CONTAINER_CHECKPOINT_PATH="$CHECKPOINT_PATH"
else
  # Assume it's a relative path from checkpoints
  CONTAINER_CHECKPOINT_PATH="/opt/ml/checkpoints/$CHECKPOINT_PATH"
fi

./build_image.sh yahtzee_agent.evaluate_model \
  --checkpoint-path "$CONTAINER_CHECKPOINT_PATH" \
  --num-games "$NUM_GAMES" \
  --baseline "/workspace/$BASELINE_PATH" \
  --alpha "$ALPHA"
