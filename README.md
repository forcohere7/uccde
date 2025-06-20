**Colour Correction and Detail Enhancement in Underwater Images using Hybrid Real-ESRGAN**

This project delivers a comprehensive pipeline for enhancing underwater images and videos by integrating underwater color correction techniques with a fine-tuned Real-ESRGAN model for super-resolution and detail refinement. The process consists of two primary stages:

1. Underwater color correction.
2. Real-ESRGAN-based detail enhancement.

The pipeline supports both local execution with `uv` and containerized execution using Docker and Docker Compose. Designed to be modular, scalable, and user-friendly, the pipeline is suitable for both researchers and practical deployments.

---

### **Key Features**

* Modular Docker architecture for each processing stage.
* Fine-tuned Real-ESRGAN model for underwater image enhancement.
* Guided filtering and histogram-based color correction.
* Compatible with various image and video formats.
* Scripted pipeline for easy automation.

---

### **Setup Instructions**

#### **Docker Execution (Recommended)**

1. **Edit Configuration**

Update `config.yml`:

```yaml
pipeline:
  services:
    ip: true
    realesrgan: true
  directories:
    input_folder: input
    output_folder: output
```

2. **Prepare Directories**

```bash
mkdir -p input output temp
chmod -R u+rwx input output temp
```

Place your files in the `input/` directory.

3. **Run the Pipeline**

```bash
chmod +x orchestrator.sh
./orchestrator.sh
```

This script will:

* Validate configuration.
* Build Docker images.
* Run the color correction service and then Real-ESRGAN.

#### **Local Execution (Advanced)**

1. **Color Correction Module**

```bash
cd image-processing
uv sync --frozen
source .venv/bin/activate
python ip_inference.py --input_dir ../input --output_dir ../temp
```

2. **Real-ESRGAN Module**

```bash
cd ../RealESRGAN
uv sync --frozen
chmod +x dependency_fix.sh
./dependency_fix.sh
source .venv/bin/activate
python src/inference_realesrgan_image.py -n RealESRGAN_x4plus -i ../temp -o ../output --model_path model/net_g_5000.pth --outscale 4 --tile 400
```

Use `inference_realesrgan_video.py` for video enhancement.

---

### **Troubleshooting**

* **CUDA OOM**: Reduce `--tile` value.
* **No Output**: Verify input paths and extensions.
* **FFmpeg Issues**: Confirm FFmpeg installation.
* **Permission Issues**: Run `chmod -R u+rwx input output temp`.
* **Dependency Errors**: Use `dependency_fix.sh`.

---

### **References**

* Song, W., Wang, Y., Huang, D., & Tjondronegoro, D. (2018). A rapid scene depth estimation model based on underwater light attenuation prior. *PCM 2018*, Springer.
* Huang, D., Wang, Y., Song, W., Sequeira, J., & Mavromatis, S. (2018). Shallow-water image enhancement using relative global histogram stretching. *MMM 2018*, Springer.
* Aghelan, A. (2022). Underwater Images Super-Resolution Using GAN-based Model. *arXiv:2211.03550*.