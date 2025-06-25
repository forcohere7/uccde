# Real-ESRGAN: PyTorch to ONNX Conversion and Inference

This repository provides scripts and utilities to convert a pretrained PyTorch RRDBNet model (used in Real-ESRGAN) to ONNX format and perform inference using ONNXRuntime.

## Overview

* **Conversion**: `pytorch2onnx.py` converts a PyTorch `.pth` model to ONNX format.
* **Inference**: `onnx_inference.py` performs image upscaling using ONNXRuntime.
* **Notebook**: `onnx_conversion_inference.ipynb` demonstrates conversion and inference interactively.
* **Dependencies**: Listed in `requirements.txt`.

---

## Requirements

* Python 3.12+
* PyTorch
* ONNX
* ONNXRuntime (CPU or GPU)
* basicsr (for RRDBNet architecture)
* OpenCV
* NumPy
* PyYAML

Install dependencies:

```bash
pip install -r requirements.txt
```

### `requirements.txt` Content

```
onnx
onnxruntime-gpu
torch
basicsr
opencv-python
numpy
pyyaml
```

> Replace `onnxruntime-gpu` with `onnxruntime` for CPU-only inference.

---

## Scripts

### 1. PyTorch to ONNX Conversion (`pytorch2onnx.py`)

This script converts a pretrained PyTorch RRDBNet model to ONNX format.

#### Usage

```bash
python pytorch2onnx.py \
  --input ../../RealESRGAN/model/net_g_5000.pth \
  --output ../../RealESRGAN/model/net_g_5000.onnx \
  --config config.yml
```

#### Arguments

| Argument          | Type | Default                                  | Description                                  |
| ----------------- | ---- | ---------------------------------------- | -------------------------------------------- |
| `--input`         | str  | ../../RealESRGAN/model/net\_g\_5000.pth  | Path to input PyTorch model (.pth)           |
| `--output`        | str  | ../../RealESRGAN/model/net\_g\_5000.onnx | Path to output ONNX model                    |
| `--config`        | str  | config.yml                               | Path to model configuration YAML file        |
| `--opset_version` | int  | 11                                       | ONNX opset version                           |
| `--params`        | flag | (disabled)                               | Use `params` instead of `params_ema` weights |

#### Configuration File (`config.yml`)

```yaml
network_g:
  num_in_ch: 3
  num_out_ch: 3
  num_feat: 64
  num_block: 23
  num_grow_ch: 32
scale: 4
```

#### Output

```
Done! Exported to ../../RealESRGAN/model/net_g_5000.onnx
```

#### Notes

* Dummy input: `torch.rand(1, 3, 64, 64)`
* Dynamic axes enabled for batch size, height, and width.
* Use `--params` for training weights; default uses `params_ema` for inference.

#### Difference between `params` and `params_ema`

* **params**: Raw weights for continued training/fine-tuning.
* **params\_ema**: EMA-smoothed weights, ideal for inference.

---

### 2. ONNX Inference (`onnx_inference.py`)

Performs image upscaling using the ONNX model.

#### Usage

```bash
python onnx_inference.py \
  --model_path ../../RealESRGAN/model/net_g_5000.onnx \
  --input_path input.png \
  --output_path output.png \
  --output_scale 4
```

#### Arguments

| Argument         | Type | Default                                  | Description                                                                      |
| ---------------- | ---- | ---------------------------------------- | -------------------------------------------------------------------------------- |
| `--model_path`   | str  | ../../RealESRGAN/model/net\_g\_5000.onnx | Path to ONNX model                                                               |
| `--input_path`   | str  | input.png                                | Path to input image                                                              |
| `--output_path`  | str  | output.png                               | Path to output image                                                             |
| `--output_scale` | int  | 4                                        | Upsampling scale factor (usually 2 or 4)                                         |
| `--tile`         | int  | 0                                        | Tile size for tiled inference (0 to disable)                                     |
| `--num_threads`  | int  | -1                                       | Number of threads (-1 = default)                                                 |
| `--providers`    | str  | CPUExecutionProvider                     | Comma-separated ONNX providers (e.g. CUDAExecutionProvider,CPUExecutionProvider) |

#### Notes

* Supports RGB, RGBA, and grayscale inputs.
* Use `--tile` for large images to avoid OOM.
* RGBA alpha channels handled via upscaling or OpenCV resizing.
* Output saved as 8-bit or 16-bit depending on input.

#### Output

```
Starting enhancement...
Inference time: <time>
Enhanced image saved to output.png
```

---

### 3. Jupyter Notebook (`onnx_conversion_inference.ipynb`)

Provides an interactive guide for:

* PyTorch to ONNX conversion
* ONNX inference on sample images

#### Notes

* Assumes same file structure as scripts.
* Helpful for testing and debugging.

---

## Additional Notes

* Ensure `.pth` model contains either `params_ema` (default) or `params`.
* ONNX export uses `opset_version=11`.
* Model supports dynamic shape inputs but only 3 or 4 channels.
* Use `onnxruntime-gpu` and `--providers CUDAExecutionProvider` for GPU inference.