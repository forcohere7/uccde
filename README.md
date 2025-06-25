# Colour Correction and Detail Enhancement in Underwater Images using Hybrid Real-ESRGAN

This project delivers a comprehensive pipeline for enhancing underwater images and videos by integrating underwater color correction techniques with a fine-tuned Real-ESRGAN model for super-resolution and detail refinement. The process consists of two primary stages:

1. Underwater color correction.
2. Real-ESRGAN-based detail enhancement.

The pipeline supports both local execution with `uv` and containerized execution using Docker and Docker Compose. Designed to be modular, scalable, and user-friendly, the pipeline is suitable for both researchers and practical deployments.

## Key Features

* Modular Docker architecture for each processing stage.
* Fine-tuned Real-ESRGAN model for underwater image enhancement.
* Guided filtering and histogram-based color correction.
* Compatible with various image and video formats.
* Scripted pipeline for easy automation.

## Model Fine-tuning

The Real-ESRGAN model was fine-tuned using the publicly available **USR-248 dataset**, which consists of 248 paired underwater low-resolution and high-resolution images. Training was conducted for approximately 4,000 iterations using a free Google Colab GPU environment due to computational constraints. While the dataset size and training time were limited, the model demonstrated noticeable qualitative improvements over the pretrained version.

For evaluation, a separate benchmark dataset such as **UIEB (Underwater Image Enhancement Benchmark)** was used to assess the enhancement quality using metrics like **PSNR** and **MSE**.

## Setup Instructions

### Docker Execution (Recommended)

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
  rebuild: false  # Set to true to rebuild Docker images
```

* The `rebuild` field controls whether Docker images are rebuilt. Set to `true` to force a rebuild, or keep as `false` (default) to skip rebuilding.
* The pipeline sets `COMPOSE_BAKE=true` to enable specific Docker Compose features.

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
* Build Docker images (if `rebuild: true` in `config.yml`).
* Run the color correction service and then Real-ESRGAN.

### Local Execution (Advanced)

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

## Troubleshooting

* **CUDA OOM**: Reduce `--tile` value.
* **No Output**: Verify input paths and extensions.
* **FFmpeg Issues**: Confirm FFmpeg installation.
* **Permission Issues**: Run `chmod -R u+rwx input output temp`.
* **Dependency Errors**: Use `dependency_fix.sh`.

## Primary Paper References

* Song, W., Wang, Y., Huang, D., & Tjondronegoro, D. (2018). A rapid scene depth estimation model based on underwater light attenuation prior. *PCM 2018*, Springer.
* Huang, D., Wang, Y., Song, W., Sequeira, J., & Mavromatis, S. (2018). Shallow-water image enhancement using relative global histogram stretching. *MMM 2018*, Springer.
* Aghelan, A. (2022). Underwater Images Super-Resolution Using GAN-based Model. *arXiv:2211.03550*.

## License

This project is licensed under the [MIT License](LICENSE).

It uses a fine-tuned model based on [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), which is released under the BSD 3-Clause License. Portions of the source code, including the inference script, have been reused or modified from the original Real-ESRGAN repository and are redistributed under the terms of the BSD 3-Clause License.
Credit to the original authors is retained as per the license.

The model was fine-tuned on the USR-248 dataset.

---

## Acknowledgements

- Real-ESRGAN © 2021 Xintao Wang  
  License: BSD 3-Clause  
  Repository: https://github.com/xinntao/Real-ESRGAN

Implementations of the following methods were developed from scratch based on their original papers. Unofficial GitHub implementations were referred to only for conceptual understanding:

- ULAP (Underwater Light Attenuation Prior)
- RGHS (Relative Global Histogram Stretching)
