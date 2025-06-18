# Use Python 3.12 slim base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV, FFmpeg, and other required libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libopencv-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv globally
RUN pip install --no-cache-dir uv

# Copy pyproject.toml and uv.lock
COPY pyproject.toml uv.lock ./

# Install Python dependencies using uv and ensure virtual environment
RUN uv sync --frozen

COPY dependency_fix.sh .
RUN chmod +x dependency_fix.sh
RUN ./dependency_fix.sh

# Copy model file
COPY model/net_g_5000.pth model/net_g_5000.pth

# Copy source scripts
COPY src/inference_realesrgan_image.py src/inference_realesrgan_image.py
COPY src/inference_realesrgan_video.py src/inference_realesrgan_video.py

# Create input and output directories
RUN mkdir -p input output video

# Copy shell script to determine and run the appropriate Python script
COPY run_script.sh .

# Set environment variable to activate uv virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Make the shell script executable
RUN chmod +x run_script.sh

# Set the shell script as the entrypoint
ENTRYPOINT ["./run_script.sh"]