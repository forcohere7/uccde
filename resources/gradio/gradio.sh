#!/bin/bash

set -e

# Set COMPOSE_BAKE environment variable
export COMPOSE_BAKE=true
echo "COMPOSE_BAKE is set to $COMPOSE_BAKE"

# Step 1: Build Docker containers
echo "Building Docker containers..."
docker-compose build || { echo "Error: Docker Compose build failed"; exit 1; }

# Step 2: Function to set up virtual environment and install Gradio
setup_venv() {
  VENV_DIR="./venv"
  if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR" || { echo "Error: Failed to create virtual environment"; exit 1; }
  fi
  source "$VENV_DIR/bin/activate" || { echo "Error: Failed to activate virtual environment"; exit 1; }

  # Check if Gradio is installed
  if ! python -c "import gradio" 2>/dev/null; then
    echo "Installing Gradio..."
    pip install --upgrade pip || { echo "Error: Failed to upgrade pip"; exit 1; }
    pip install gradio || { echo "Error: Failed to install Gradio"; exit 1; }
    echo "Gradio installed successfully"
  else
    echo "Gradio is already installed"
  fi
}

# Step 3: Set up virtual environment and install Gradio
setup_venv

# Step 4: Run the Gradio UI
echo "Starting Gradio UI..."
python gradio_ui.py