# Use Python 3.12 slim base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv globally
RUN pip install --no-cache-dir uv

# Copy pyproject.toml and uv.lock
COPY pyproject.toml uv.lock .

# Install Python dependencies using uv and ensure virtual environment
RUN uv sync --frozen

# Copy application code
COPY . .

# Set environment variable to activate uv virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Command to run the inference script
CMD ["python", "ip_inference.py", "--input_dir", "/app/input", "--output_dir", "/app/output"]