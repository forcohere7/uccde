#!/bin/bash

set -e

# Set COMPOSE_BAKE and UV_HTTP_TIMEOUT environment variable
export COMPOSE_BAKE=true
export UV_HTTP_TIMEOUT=60

# Function to install yq if not present
install_yq() {
  if ! command -v yq >/dev/null || ! yq --version | grep -q "mikefarah"; then
    echo "Installing Go-based yq (mikefarah/yq)..."
    if [[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]]; then
      wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O /tmp/yq || { echo "Error: Failed to download yq"; exit 1; }
      sudo mv /tmp/yq /usr/local/bin/yq || { echo "Error: Failed to move yq to /usr/local/bin"; exit 1; }
      sudo chmod +x /usr/local/bin/yq || { echo "Error: Failed to make yq executable"; exit 1; }
      echo "yq (Go-based) installed successfully"
    else
      echo "Unsupported platform. Please install yq manually from https://github.com/mikefarah/yq"
      exit 1
    fi
  else
    echo "Correct yq (Go-based) is already installed"
  fi
}

# Install yq if not present
install_yq

# Check for yq (redundant check for safety)
command -v yq >/dev/null || { echo "Error: yq not found after installation attempt"; exit 1; }

CONFIG_FILE="./config.yml"
[ -f "$CONFIG_FILE" ] || { echo "Error: Config file missing at $CONFIG_FILE"; exit 1; }

# Parse config
IP_ENABLED=$(yq e '.pipeline.services.ip' "$CONFIG_FILE")
REALESRGAN_ENABLED=$(yq e '.pipeline.services.realesrgan' "$CONFIG_FILE")
INPUT_DIR=$(yq e '.pipeline.directories.input_folder' "$CONFIG_FILE")
OUTPUT_DIR=$(yq e '.pipeline.directories.output_folder' "$CONFIG_FILE")
REBUILD=$(yq e '.pipeline.rebuild // false' "$CONFIG_FILE")  # Default to false if not set
TEMP_DIR="./temp"

# Validate directories
[ "$(realpath "$INPUT_DIR")" = "$(realpath "$OUTPUT_DIR")" ] || [ "$(realpath "$INPUT_DIR")" = "$(realpath "$TEMP_DIR")" ] || [ "$(realpath "$OUTPUT_DIR")" = "$(realpath "$TEMP_DIR")" ] && { echo "Error: Input, output, and temp directories must be different"; exit 1; }

# Create directories
mkdir -p "$INPUT_DIR" "$OUTPUT_DIR" "$TEMP_DIR"
chmod 755 "$INPUT_DIR" "$OUTPUT_DIR" "$TEMP_DIR" 2>/dev/null || true

# Cleanup function
cleanup() {
  [ -d "$TEMP_DIR" ] && [ "$TEMP_DIR" != "/" ] && [ "$TEMP_DIR" != "/tmp" ] && rm -rf "$TEMP_DIR" 2>/dev/null
}

trap cleanup EXIT INT TERM

# Build images if rebuild is enabled
if [ "$REBUILD" = "true" ]; then
  echo "Rebuilding Docker Compose services..."
  docker-compose build
else
  echo "Skipping Docker Compose build (rebuild=false)"
fi

# Run services
if [ "$IP_ENABLED" = "true" ] && [ "$REALESRGAN_ENABLED" = "true" ]; then
  echo "Running both image-processing and Real-ESRGAN in sequence"
  docker-compose run --rm -v "$(realpath "$INPUT_DIR"):/app/input" -v "$(realpath "$TEMP_DIR"):/app/output" image-processing python ip_inference.py --input_dir /app/input --output_dir /app/output
  [ -z "$(ls -A "$TEMP_DIR")" ] && { echo "Error: No files in $TEMP_DIR for Real-ESRGAN"; exit 1; }
  docker-compose run --rm -v "$(realpath "$TEMP_DIR"):/app/input" -v "$(realpath "$OUTPUT_DIR"):/app/output" real-esrgan /app/input /app/output
elif [ "$IP_ENABLED" = "true" ]; then
  echo "Running only image-processing"
  docker-compose run --rm -v "$(realpath "$INPUT_DIR"):/app/input" -v "$(realpath "$OUTPUT_DIR"):/app/output" image-processing python ip_inference.py --input_dir /app/input --output_dir /app/output
elif [ "$REALESRGAN_ENABLED" = "true" ]; then
  echo "Running only Real-ESRGAN"
  [ -z "$(ls -A "$INPUT_DIR")" ] && { echo "Error: No files in $INPUT_DIR for Real-ESRGAN"; exit 1; }
  docker-compose run --rm -v "$(realpath "$INPUT_DIR"):/app/input" -v "$(realpath "$OUTPUT_DIR"):/app/output" real-esrgan /app/input /app/output
else
  echo "Error: No services enabled"
  exit 1
fi

echo "Pipeline completed"