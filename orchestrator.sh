#!/bin/bash

# Exit on any error
set -e

# Check if yq is installed
if ! command -v yq &> /dev/null; then
  echo "Error: yq command not found. Please install yq first."
  echo "On Ubuntu/Debian: sudo apt install yq"
  echo "On macOS: brew install yq"
  exit 1
fi

CONFIG_FILE="./config.yml"
if [ ! -f "$CONFIG_FILE" ]; then
  echo "Error: Config file not found at $CONFIG_FILE"
  exit 1
fi

# Parse config using yq
IP_ENABLED=$(yq e '.pipeline.services.ip' "$CONFIG_FILE")
REALESRGAN_ENABLED=$(yq e '.pipeline.services.realesrgan' "$CONFIG_FILE")
INPUT_DIR=$(yq e '.pipeline.directories.input_folder' "$CONFIG_FILE")
OUTPUT_DIR=$(yq e '.pipeline.directories.output_folder' "$CONFIG_FILE")
TEMP_DIR="./temp" # Keep temp as a fixed directory for simplicity, or add it to config.yml

# Validate that input and output directories are not the same as TEMP_DIR
if [ "$(realpath $INPUT_DIR)" = "$(realpath $TEMP_DIR)" ] || [ "$(realpath $OUTPUT_DIR)" = "$(realpath $TEMP_DIR)" ]; then
  echo "Error: Input or output directory cannot be set to $TEMP_DIR (temporary folder)"
  exit 1
fi

# Validate that input and output directories are not the same
if [ "$(realpath $INPUT_DIR)" = "$(realpath $OUTPUT_DIR)" ]; then
  echo "Error: Input and output directories cannot be the same"
  exit 1
fi

# Ensure input, output, and temp directories exist
echo "Setting up directories: $INPUT_DIR, $OUTPUT_DIR, $TEMP_DIR..."
mkdir -p "$INPUT_DIR" "$OUTPUT_DIR" "$TEMP_DIR"

# Only set permissions on directories we just created, ignore failures
echo "Setting permissions (ignoring any errors)..."
chmod 755 "$INPUT_DIR" "$OUTPUT_DIR" "$TEMP_DIR" 2>/dev/null || echo "Warning: Could not set permissions on some directories"

# Function to clean up temp folder
cleanup() {
  if [ -n "$TEMP_DIR" ] && [ -d "$TEMP_DIR" ] && [ "$TEMP_DIR" != "/" ] && [ "$TEMP_DIR" != "/tmp" ]; then
    # Force remove without caring about permissions
    rm -rf "$TEMP_DIR" 2>/dev/null || {
      echo "Warning: Could not remove temp folder normally, trying with sudo..."
      sudo rm -rf "$TEMP_DIR" 2>/dev/null || echo "Warning: Could not remove temp folder"
    }
    echo "Removed temporary folder $TEMP_DIR"
  else
    echo "Skipping removal of temporary folder: invalid or unsafe path"
  fi
}

# Trap errors and interrupts to ensure cleanup
trap cleanup EXIT INT TERM

echo "Starting pipeline with IP_ENABLED=$IP_ENABLED, REALESRGAN_ENABLED=$REALESRGAN_ENABLED"

# Build all images first
echo "Building Docker images..."
docker-compose build

# Run services based on config
if [ "$IP_ENABLED" = "true" ] && [ "$REALESRGAN_ENABLED" = "true" ]; then
  echo "Running both image-processing and Real-ESRGAN in sequence"
  
  echo "Step 1: Running image-processing service..."
  docker-compose run --rm -v "$(realpath $INPUT_DIR):/app/input" -v "$(realpath $TEMP_DIR):/app/output" image-processing python ip_inference.py --input_dir /app/input --output_dir /app/output
  
  echo "Debug: Checking contents of $TEMP_DIR..."
  docker-compose run --rm -v "$(realpath $TEMP_DIR):/app/temp" image-processing ls -l /app/temp || echo "Warning: No files found in /app/temp"
  
  echo "Step 2: Running Real-ESRGAN service..."
  if [ -z "$(ls -A "$TEMP_DIR")" ]; then
    echo "Error: No files found in $TEMP_DIR for Real-ESRGAN to process"
    exit 1
  fi
  docker-compose run --rm -v "$(realpath $TEMP_DIR):/app/input" -v "$(realpath $OUTPUT_DIR):/app/output" real-esrgan /app/input /app/output

elif [ "$IP_ENABLED" = "true" ]; then
  echo "Running only image-processing"
  docker-compose run --rm -v "$(realpath $INPUT_DIR):/app/input" -v "$(realpath $OUTPUT_DIR):/app/output" image-processing python ip_inference.py --input_dir /app/input --output_dir /app/output

elif [ "$REALESRGAN_ENABLED" = "true" ]; then
  echo "Running only Real-ESRGAN"
  if [ -z "$(ls -A "$INPUT_DIR")" ]; then
    echo "Error: No files found in $INPUT_DIR for Real-ESRGAN to process"
    exit 1
  fi
  docker-compose run --rm -v "$(realpath $INPUT_DIR):/app/input" -v "$(realpath $OUTPUT_DIR):/app/output" real-esrgan /app/input /app/output

else
  echo "Error: No services enabled in config"
  exit 1
fi

echo "Pipeline completed successfully"