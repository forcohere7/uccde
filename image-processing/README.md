# Underwater Colour Correction and Image Enhancement

This project provides a Python-based solution for enhancing underwater images and videos by applying color correction and image enhancement techniques. It processes images and videos in the `input` directory and saves the results to the `output` directory.

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

* `ip_inference.py`: Main script to process images and videos.
* `src/`: Contains modular Python code for color correction, image enhancement, and video processing.
* `input/`: Directory for input images and videos.
* `output/`: Directory for processed images and videos.
* `Dockerfile`: Defines the Docker image for running the project.
* `pyproject.toml`, `uv.lock`, and `requirements.txt`: Define Python dependencies using `uv` or `pip`.

## Prerequisites

* **Python 3.12** (for local execution).
* **Docker** (for containerized execution).
* Input files (images or videos) in supported formats: images (e.g., `.jpg`, `.png`) and videos (`.mp4`, `.avi`, `.mov`).

## Setup and Running Locally

### 1. Install Dependencies

Create a virtual environment and install dependencies using `uv` or `pip`.

#### Using `uv`

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

For video processing and OpenCV, install FFmpeg and OpenCV system dependencies:

* **Ubuntu/Debian**:

  ```bash
  sudo apt-get update
  sudo apt-get install -y ffmpeg
  ```
* **macOS** (using Homebrew):

  ```bash
  brew install ffmpeg
  ```
* **Windows**: Install FFmpeg manually and add it to your PATH. Or use a package manager:

  ```bash
  choco install ffmpeg
  ```

### 3. Prepare Input Files

Place images or videos in the `input` directory.

### 4. Run the Script

Execute the main script:

```bash
python ip_inference.py
```

This processes all files in `input/` and saves results to `output/`.

## Setup and Running with Docker

### 1. Build the Docker Image

```bash
docker build -t ip .
```

### 2. Run the Docker Container

Run the container, mounting `input` and `output` directories:

```bash
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output ip
```

* `-v $(pwd)/input:/app/input`: Mounts the local `input` directory to `/app/input` in the container.
* `-v $(pwd)/output:/app/output`: Mounts the local `output` directory to `/app/output`.
* `--rm`: Removes the container after execution.
* `ip`: The image name specified during the build.

### Notes

* Ensure `input` and `output` directories exist before running the container.
* Place input files in the `input` directory.
* Processed files will appear in the `output` directory.

## Troubleshooting

* **Missing Dependencies**: Ensure all Python and system dependencies are installed. For Docker, verify `ffmpeg` is included in the image.
* **File Not Found**: Confirm input files are in the `input` directory and have supported extensions (images: `.jpg`, `.png`, etc.; videos: `.mp4`, `.avi`, `.mov`).
* **Docker Issues**: Check Docker is running and you have permission to execute commands. For permission errors, prepend `sudo` or adjust Docker permissions.
* **FFmpeg Errors**: Ensure FFmpeg is installed and accessible. In Docker, the `Dockerfile` includes FFmpeg; for local runs, install it manually.

---

For contributions, issues, or improvements, feel free to open an issue or pull request.