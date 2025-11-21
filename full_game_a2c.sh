#!/usr/bin/env bash
set -e

# Load config from JSON and build arguments
CONFIG_FILE="wandb/full_game_a2c.json"

# Parse JSON and build argument list
ARGS=()
while IFS='=' read -r key value; do
  # Convert snake_case to kebab-case and add to args
  flag=$(echo "$key" | sed 's/_/-/g')
  ARGS+=("--${flag}" "${value}")
done < <(python3 -c "
import json
import sys
with open('${CONFIG_FILE}') as f:
    config = json.load(f)
    for key, value in config['run_config'].items():
        print(f'{key}={value}')
")

# Build and run the full game RL scenario
./build_image.sh yahtzee_agent \
  "${ARGS[@]}" \
  --log-dir /workspace/logs
