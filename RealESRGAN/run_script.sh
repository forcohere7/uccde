#!/bin/bash

# Check if an input argument is provided
if [ -z "$1" ]; then
  echo "Error: Please provide an input path (either a .mp4 file or a folder with images)"
  exit 1
fi

INPUT_PATH="$1"
MODEL_PATH="/app/model/net_g_5000.pth"
OUTPUT_DIR="/app/output"
TILE=400
OUTSCALE=4

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Check if the input is a file with .mp4 extension
if [ -f "$INPUT_PATH" ] && [[ "$INPUT_PATH" == *.mp4 ]]; then
  echo "Detected .mp4 file. Running video processing script..."
  python /app/src/inference_realesrgan_video.py -i "$INPUT_PATH" -o "$OUTPUT_DIR" --model_path "$MODEL_PATH" --outscale "$OUTSCALE" --tile "$TILE"
elif [ -d "$INPUT_PATH" ]; then
  echo "Detected directory. Removing non-image files..."
  # Remove non-image files (keep only .png, .jpg, .jpeg, .bmp, .tiff)
  find "$INPUT_PATH" -type f ! -regex ".*\.\(png\|jpg\|jpeg\|bmp\|tiff\)$" -delete
  # Check if there are any image files left
  if [ -z "$(find "$INPUT_PATH" -type f -regex ".*\.\(png\|jpg\|jpeg\|bmp\|tiff\)$")" ]; then
    echo "Error: No image files (.png, .jpg, .jpeg, .bmp, .tiff) found in $INPUT_PATH"
    exit 1
  fi
  echo "Running image processing script..."
  python /app/src/inference_realesrgan_image.py -n RealESRGAN_x4plus -i "$INPUT_PATH" -o "$OUTPUT_DIR" --model_path "$MODEL_PATH" --outscale "$OUTSCALE" --tile "$TILE"
else
  echo "Error: Input must be a .mp4 file or a directory containing images"
  exit 1
fi