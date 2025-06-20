#!/bin/bash

set -e

# Check for yq
command -v yq >/dev/null || { echo "Error: yq not found. Install with: apt install yq (Ubuntu) or brew install yq (macOS)"; exit 1; }

CONFIG_FILE="./config.yml"
[ -f "$CONFIG_FILE" ] || { echo "Error: Config file missing at $CONFIG_FILE"; exit 1; }

# Parse config
IP_ENABLED=$(yq e '.pipeline.services.ip' "$CONFIG_FILE")
REALESRGAN_ENABLED=$(yq e '.pipeline.services.realesrgan' "$CONFIG_FILE")
INPUT_DIR=$(yq e '.pipeline.directories.input_folder' "$CONFIG_FILE")
OUTPUT_DIR=$(yq e '.pipeline.directories.output_folder' "$CONFIG_FILE")
TEMP_DIR="./temp"

# Validate directories
[ "$(realpath $INPUT_DIR)" = "$(realpath $OUTPUT_DIR)" ] || [ "$(realpath $INPUT_DIR)" = "$(realpath $TEMP_DIR)" ] || [ "$(realpath $OUTPUT_DIR)" = "$(realpath $TEMP_DIR)" ] && { echo "Error: Input, output, and temp directories must be different"; exit 1; }

# Create directories
mkdir -p "$INPUT_DIR" "$OUTPUT_DIR" "$TEMP_DIR"
chmod 755 "$INPUT_DIR" "$OUTPUT_DIR" "$TEMP_DIR" 2>/dev/null || true

# Cleanup function
cleanup() {
  [ -d "$TEMP_DIR" ] && [ "$TEMP_DIR" != "/" ] && [ "$TEMP_DIR" != "/tmp" ] && rm -rf "$TEMP_DIR" 2>/dev/null
}

trap cleanup EXIT INT TERM

# Build images
docker-compose build

# Run services
if [ "$IP_ENABLED" = "true" ] && [ "$REALESRGAN_ENABLED" = "true" ]; then
  echo "Running both image-processing and Real-ESRGAN in sequence"
  docker-compose run --rm -v "$(realpath $INPUT_DIR):/app/input" -v "$(realpath $TEMP_DIR):/app/output" image-processing python ip_inference.py --input_dir /app/input --output_dir /app/output
  [ -z "$(ls -A "$TEMP_DIR")" ] && { echo "Error: No files in $TEMP_DIR for Real-ESRGAN"; exit 1; }
  docker-compose run --rm -v "$(realpath $TEMP_DIR):/app/input" -v "$(realpath $OUTPUT_DIR):/app/output" real-esrgan /app/input /app/output
elif [ "$IP_ENABLED" = "true" ]; then
  echo "Running only image-processing"
  docker-compose run --rm -v "$(realpath $INPUT_DIR):/app/input" -v "$(realpath $OUTPUT_DIR):/app/output" image-processing python ip_inference.py --input_dir /app/input --output_dir /app/output
elif [ "$REALESRGAN_ENABLED" = "true" ]; then
  echo "Running only Real-ESRGAN"
  [ -z "$(ls -A "$INPUT_DIR")" ] && { echo "Error: No files in $INPUT_DIR for Real-ESRGAN"; exit 1; }
  docker-compose run --rm -v "$(realpath $INPUT_DIR):/app/input" -v "$(realpath $OUTPUT_DIR):/app/output" real-esrgan /app/input /app/output
else
  echo "Error: No services enabled"
  exit 1
fi

echo "Pipeline completed"