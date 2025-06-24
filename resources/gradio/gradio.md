# Gradio UI for Underwater Image Enhancement

This project provides a Gradio-based web interface for enhancing underwater images using color correction and super-resolution techniques. The application is containerized using Docker and orchestrated with Docker Compose, ensuring a consistent environment for processing.

## Prerequisites

* **Docker**: Ensure Docker is installed and running.
* **Docker Compose**: Required for orchestrating the containers.
* **Python 3.8+**: Needed for setting up the virtual environment and running the Gradio UI.
* **Gradio**: Installed automatically within the virtual environment by the setup script.
* **Git LFS**: Required for pulling large model files.

## Setup Instructions

### Set Up Git LFS and Pull Model Files:

```bash
git lfs install
git lfs pull
```

* The `git lfs install` command sets up Git Large File Storage (LFS) to handle large model files.
* The `git lfs pull` command ensures all model files are downloaded if they are missing.

### Run the Setup Script:

Execute the provided bash script (`gradio.sh`) to configure the environment and start the Gradio UI:

```bash
chmod +x gradio.sh
./gradio.sh
```

The script performs the following:

* Sets the `COMPOSE_BAKE` environment variable.
* Builds Docker containers using `docker-compose build`.
* Creates and activates a Python virtual environment.
* Installs Gradio if not already installed.
* Launches the Gradio UI.

### Access the Gradio Interface:

Once the script runs, the Gradio UI will be available at:

```
http://0.0.0.0:7860
```

Open this URL in a web browser to interact with the interface.

## Using the Gradio Interface

### Upload an Image:

* Use the file upload component to select an underwater image (supported formats: PNG, JPG, etc.).

### Select Enhancement Options:

* **Color Correction**: Enable to apply color correction using the image-processing service.
* **Super-Resolution**: Enable to apply super-resolution using the real-esrgan service.
* Both options can be used together or individually. At least one must be selected.

### Run the Enhancement:

* Click the "Enhance Image" button to process the uploaded image.
* The original and enhanced images will be displayed side-by-side, along with a status message indicating success or any errors.

### View Results:

* The input image is shown in the **Original** section.
* The processed image appears in the **Enhanced** section.
* Status messages provide feedback on the processing outcome.

## Directory Structure

* **Input Directory** (`./input`): Stores uploaded images for processing.
* **Output Directory** (`./output`): Stores enhanced images after processing.
* **Temporary Directory**: Used as intermediate storage for sequential processing (e.g., color correction followed by super-resolution).
* **Virtual Environment** (`./venv`): Contains the Python environment with Gradio and dependencies.

## Notes

### Docker Services:

* The `image-processing` service handles color correction via `ip_inference.py`.
* The `real-esrgan` service performs super-resolution.
* Both services are invoked via Docker Compose with appropriate volume mounts for input/output directories.

### Error Handling:

* The script and Gradio UI include error checks for missing directories, invalid inputs, and Docker execution failures.
* Temporary directories are cleaned up automatically to avoid clutter.

### Permissions:

* The script sets appropriate permissions (`0o755`) for input, output, and temporary directories to ensure accessibility.

### Gradio Configuration:

* The UI is configured to run on `0.0.0.0:7860` for local access.
* Adjust `server_name` and `server_port` in `gradio_ui.py` if needed.

## Troubleshooting

### Docker Build Fails:

* Ensure Docker and Docker Compose are installed and up-to-date.
* Check the `docker-compose.yml` file for correct service definitions.

### Gradio Not Accessible:

* Verify the server is running and the port (7860) is not blocked by a firewall.
* Check the console for error messages from the `gradio.sh` script or Gradio.

### No Output Files:

* Ensure at least one enhancement option (Color Correction or Super-Resolution) is selected.
* Verify that the input image is valid and accessible.

### Missing Model Files:

* If model files are missing, ensure `git lfs pull` was executed after setup.
* Verify that Git LFS is installed (`git lfs install`).

## Dependencies

* **Gradio**: Provides the web-based UI.
* **Docker Compose**: Manages the containerized services.
* **Python Libraries**:

  * `os`, `subprocess`, `shutil`, `tempfile`: Used for file and process management in `gradio_ui.py`.
* **Git LFS**: Manages large model files used by the image processing services.

For further details on the Docker services or image processing algorithms, refer to the respective service documentation or source code.