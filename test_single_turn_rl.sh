#!/usr/bin/env bash
set -e

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

# Build and run the test scenario
./build_image.sh C_single_turn_score_maximizer \
  --mode test \
  --checkpoint-path "$WORKSPACE_CHECKPOINT_PATH"
