#!/bin/bash

set -e

# Function to set up virtual environment and install Gradio
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

# Set up virtual environment and install Gradio
setup_venv

# Run the Gradio UI
echo "Starting Gradio UI..."
python gradio_ui.py