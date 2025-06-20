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

# Ensure input, output, and temp directories exist with correct permissions
echo "Setting up directories..."
mkdir -p ./input ./output ./temp
chmod -R 777 ./input ./output ./temp || { echo "Error: Failed to set permissions on directories"; exit 1; }

# Function to clean up temp folder
cleanup() {
  rm -rf ./temp/*
  echo "Cleaned up temporary folder"
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
  docker-compose run --rm image-processing python ip_inference.py --input_dir /app/input --output_dir /app/temp
  
  echo "Debug: Checking contents of /app/temp..."
  docker-compose run --rm image-processing ls -l /app/temp || echo "Warning: No files found in /app/temp"
  
  echo "Step 2: Running Real-ESRGAN service..."
  if [ -z "$(ls -A ./temp)" ]; then
    echo "Error: No files found in ./temp for Real-ESRGAN to process"
    exit 1
  fi
  docker-compose run --rm real-esrgan /app/temp /app/output

elif [ "$IP_ENABLED" = "true" ]; then
  echo "Running only image-processing"
  docker-compose run --rm image-processing python ip_inference.py --input_dir /app/input --output_dir /app/output

elif [ "$REALESRGAN_ENABLED" = "true" ]; then
  echo "Running only Real-ESRGAN"
  if [ -z "$(ls -A ./input)" ]; then
    echo "Error: No files found in ./input for Real-ESRGAN to process"
    exit 1
  fi
  docker-compose run --rm real-esrgan /app/input /app/output

else
  echo "Error: No services enabled in config"
  exit 1
fi

echo "Pipeline completed successfully"