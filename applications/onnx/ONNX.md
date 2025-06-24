# PyTorch to ONNX Export Script and Inference

This script converts a pretrained PyTorch RRDBNet model (used in Real-ESRGAN) to ONNX format.

## Requirements

* Python 3.12+
* PyTorch
* basicsr (for RRDBNet architecture)

Install dependencies:

```bash
pip install torch
pip install basicsr
```

## Usage

```bash
python pytorch2onnx.py \
  --input ../../RealESRGAN/model/net_g_5000.pth \
  --output ../../RealESRGAN/model/net_g_5000.onnx
```

### Optional Argument

To use `params` instead of the default `params_ema` when loading the model:

```bash
python pytorch2onnx.py --params
```

## Notes

* The dummy input tensor used during export is `torch.rand(1, 3, 64, 64)` (a 64x64 RGB image).
* The ONNX export uses `opset_version=11`.
* Ensure the input model file (`.pth`) contains either `params_ema` (default) or `params` keys.

### Difference between `params` and `params_ema`

* `params`: These are the raw weights of the model as updated directly by backpropagation. They are ideal for **further training or fine-tuning**, since they reflect the most recent learning steps.

* `params_ema`: These are the **exponential moving average** of the weights collected during training. They tend to produce smoother and more stable results, making them better suited for **inference**.

In summary:

* Use `params` → when continuing training.
* Use `params_ema` → for final inference (default).

## Output

After successful conversion, you should see:

```
ONNX model exported to: ../../RealESRGAN/model/net_g_5000.onnx
```