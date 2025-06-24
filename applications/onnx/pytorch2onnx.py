import torch
import torch.onnx
from basicsr.archs.rrdbnet_arch import RRDBNet

def main():
    input_model_path = '../../RealESRGAN/model/net_g_5000.pth'
    output_onnx_path = '../../RealESRGAN/model/net_g_5000.onnx'
    use_params_ema = True  # Set to False if you want to use 'params' instead of 'params_ema'

    # Initialize model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

    # Load model weights
    keyname = 'params_ema' if use_params_ema else 'params'
    model.load_state_dict(torch.load(input_model_path)[keyname])

    # Set model to evaluation mode on CPU
    model.eval()
    model.cpu()

    # Dummy input for ONNX export
    x = torch.rand(1, 3, 64, 64)

    # Export the model
    with torch.no_grad():
        torch_out = torch.onnx.export(model, x, output_onnx_path, opset_version=11, export_params=True)
    print("ONNX export completed.")

if __name__ == '__main__':
    main()