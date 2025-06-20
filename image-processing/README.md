# Underwater Colour Correction and Image Enhancement

This project provides a Python-based solution for enhancing underwater images and videos by applying color correction and image enhancement techniques. It processes images and videos in a specified input directory and saves the results to a specified output directory. The project supports both local execution and containerized execution using Docker.

## Project Structure

```
underwater_image_enhancement/
├── Dockerfile
├── pyproject.toml
├── uv.lock
├── requirements.txt
├── ip_inference.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── guided_filter.py
│   ├── colour_correction.py
│   ├── image_enhancement.py
│   └── video_processing.py
├── input/
└── output/
```

* `ip_inference.py`: Main script to process images and videos using `argparse` for input/output directory specification.
* `src/`: Contains modular Python code for color correction, image enhancement, and video processing.
* `input/`: Default directory for input images and videos.
* `output/`: Default directory for processed images and videos.
* `Dockerfile`: Defines the Docker image for running the project.
* `pyproject.toml`, `uv.lock`, and `requirements.txt`: Define Python dependencies using `uv` or `pip`.

## Prerequisites

* **Python 3.12** (for local execution).
* **Docker** (for containerized execution).
* Input files in supported formats:

  * Images: `.jpg`, `.png`, etc.
  * Videos: `.mp4`, `.avi`, `.mov`, etc. (defined in `src/config.py` as `CONFIG['video_extensions']`).

## Setup and Running Locally

### 1. Install Dependencies

Create a virtual environment and install dependencies using `uv` (recommended) or `pip`.

#### Using `uv` (Recommended)

1. Install `uv`:

   ```bash
   pip install uv
   ```

2. Sync dependencies:

   ```bash
   uv sync
   ```

3. Activate the virtual environment:

   ```bash
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

#### Using `pip`

Dependencies are listed in `requirements.txt`:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Install System Dependencies

For video processing with OpenCV, install FFmpeg:

* **Ubuntu/Debian**:

  ```bash
  sudo apt-get update
  sudo apt-get install -y ffmpeg
  ```

* **macOS** (using Homebrew):

  ```bash
  brew install ffmpeg
  ```

* **Windows**:
  Install FFmpeg manually and add it to your PATH, or use a package manager like Chocolatey:

  ```bash
  choco install ffmpeg
  ```

### 3. Prepare Input Files

Place images or videos in the desired input directory (default: `input`). Ensure files have supported extensions as defined in `src/config.py`.

### 4. Run the Script

Execute the main script with optional input and output directory arguments:

```bash
python ip_inference.py --input_dir input --output_dir output
```

* `--input_dir`: Specify the input directory (default: `input`).
* `--output_dir`: Specify the output directory (default: `output`).

**Example with custom directories**:

```bash
mkdir -p my_input my_output
python ip_inference.py --input_dir my_input --output_dir my_output
```

This processes all files in the specified input directory and saves results to the specified output directory.

## Setup and Running with Docker

### 1. Build the Docker Image

Build the Docker image using the provided `Dockerfile`:

```bash
docker build -t ip .
```

### 2. Run the Docker Container

Run the container, mounting local input and output directories to `/app/input` and `/app/output` in the container:

```bash
mkdir -p input output
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output ip
```

* `--rm`: Removes the container after execution.
* `-v $(pwd)/input:/app/input`: Mounts the local `input` directory to `/app/input` in the container.
* `-v $(pwd)/output:/app/output`: Mounts the local `output` directory to `/app/output` in the container.

**Example with custom directories**:

```bash
mkdir -p my_input my_output
docker run --rm -v $(pwd)/my_input:/app/input -v $(pwd)/my_output:/app/output ip
```

### Notes

* Ensure the local input and output directories exist before running the container.
* Place input files (images/videos) in the local input directory before running.
* Processed files will appear in the local output directory after execution.
* Avoid using absolute paths like `/input` or `/output` on the host unless they exist and have appropriate permissions (e.g., `sudo mkdir -p /input && sudo chmod -R 777 /input`).
* The `Dockerfile` creates `/app/input` and `/app/output` with `777` permissions to ensure writability.

## Troubleshooting

* **Missing Dependencies**:

  * Ensure all Python dependencies are installed (`opencv-python-headless`, `natsort`, etc.).
  * For Docker, the `Dockerfile` includes `ffmpeg`. For local runs, install it manually.
* **File Not Found**:

  * Confirm input files are in the specified input directory and have supported extensions (check `CONFIG['video_extensions']` in `src/config.py`).
* **Permission Errors**:

  * For local runs, ensure the output directory is writable.
  * For Docker, ensure mounted directories (`input`, `output`) exist and are accessible. Create them with:

    ```bash
    mkdir -p input output
    chmod -R u+rwx input output
    ```
  * If using absolute paths (e.g., `/input`, `/output`), create them with appropriate permissions:

    ```bash
    sudo mkdir -p /input /output
    sudo chmod -R 777 /input /output
    ```
* **Docker Issues**:

  * Verify Docker is running and you have permission to execute commands. Use `sudo` if needed or adjust Docker permissions.
  * Check the image built successfully (`docker images`) and the container runs without errors.
* **FFmpeg Errors**:

  * Ensure FFmpeg is installed and accessible. For local runs, verify it’s in your PATH. For Docker, the `Dockerfile` includes FFmpeg.
* **Script Errors**:

  * If `CONFIG['video_extensions']` is undefined, ensure `src/config.py` defines it (e.g., `CONFIG = {'video_extensions': ['.mp4', '.avi', '.mov']}`).
  * Check for valid input files and correct paths in `ip_inference.py`.

## Contributing

For contributions, issues, or improvements, please open an issue or pull request on the project repository.