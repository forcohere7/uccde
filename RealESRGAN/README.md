# Underwater Image Enhancement using Fine-tuned Real-ESRGAN

This project provides a flexible application for enhancing images and videos using the fine-tuned Real-ESRGAN model. It can be run either using Docker or locally with `uv`.

## Prerequisites

### For Docker:

* **Docker**: Ensure Docker is installed and running. [Download Docker](https://www.docker.com/)
* **Input Files**:

  * Image Processing: A directory containing images (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`)
  * Video Processing: A single `.mp4` video file
* **Output Directory**: A local directory to store the enhanced results

### For Local Execution with `uv`:

* **Python 3.12**: [Install Python 3.12](https://www.python.org/)
* **uv**: Install globally using `pip install uv`
* **FFmpeg**:

  * Ubuntu: `sudo apt-get install ffmpeg`
  * macOS: `brew install ffmpeg`
  * Windows: Follow instructions on FFmpeg's website
* **System Dependencies for OpenCV**:

  * Ubuntu: `sudo apt-get install libopencv-dev`
  * macOS: `brew install opencv`
  * Windows: Install via pip or follow specific guides
* **Model File**: Download `net_g_5000.pth` and place it in a `model/` directory

## Setup and Usage

### Option 1: Running with Docker

#### Build the Docker Image:

```bash
docker build -t real-esrgan .
```

#### Prepare Input and Output Directories:

* Create local directories: `input/`, `output/`, `video/`
* Place image files in `input/`, or video file in `video/`

#### Run the Docker Container:

* **Image Processing**:

```bash
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output real-esrgan /app/input
```

* **Video Processing**:

```bash
docker run --rm -v $(pwd)/video:/app/video -v $(pwd)/output:/app/output real-esrgan /app/video/test1.mp4
```

#### Output:

* Enhanced results saved in `output/` with `_out` suffix (e.g., `image_out.png`, `test1_out.mp4`)

### Option 2: Running Locally with `uv`

#### Clone and Setup the Project:

```bash
cd path/to/project
uv sync --frozen
```

#### Fix Dependency Issue:

If you encounter the following error:

```
ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'
```

This is due to Real-ESRGAN using an outdated import:

```python
from torchvision.transforms.functional_tensor import rgb_to_grayscale
```

This breaks in newer versions of torchvision. To resolve this while retaining the ESRGAN-compatible versions, a patch is applied to the `basicsr` package. Run:

```bash
chmod +x dependency_fix.sh
./dependency_fix.sh
```

#### Prepare Input and Model Files:

* Create `input/`, `output/`, `video/`, and `model/` directories
* Place media files and `net_g_5000.pth` appropriately

#### Activate Virtual Environment:

```bash
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

#### Run Scripts:

* **Image Processing**:

```bash
python src/inference_realesrgan_image.py -n RealESRGAN_x4plus -i input -o output --model_path model/net_g_5000.pth --outscale 4 --tile 400
```

* **Video Processing**:

```bash
python src/inference_realesrgan_video.py -i video/test1.mp4 -o output --model_path model/net_g_5000.pth --outscale 4 --tile 400
```

#### Output:

* Enhanced results saved in `output/` with `_out` suffix

## Notes

* **Tile Size**: Adjust `--tile` (e.g., `200`) if CUDA OOM occurs
* **Outscale**: Default is `4` for 4x enhancement
* **Model**: Only `RealESRGAN_x4plus` with `net_g_5000.pth` is supported
* **Permissions**: Ensure input/output directories are accessible by Docker
* **FFmpeg**: Required for video processing

## Troubleshooting

* **CUDA Memory Errors**: Reduce `--tile` value
* **FFmpeg Errors**: Ensure it's in `PATH`
* **No Images Found**: Check supported formats
* **Model Not Found**: Verify model path and filename