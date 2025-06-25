import argparse
import yaml
import torch
import torch.onnx
from basicsr.archs.rrdbnet_arch import RRDBNet

def main(args):
    # Load model configuration
    with open(args.config, 'r') as reader:
        config = yaml.load(reader, Loader=yaml.FullLoader)
    print('network_g config:', config['network_g'])

    # Initialize model
    model = RRDBNet(
        num_in_ch=config['network_g']['num_in_ch'],
        num_out_ch=config['network_g']['num_out_ch'],
        num_feat=config['network_g']['num_feat'],
        num_block=config['network_g']['num_block'],
        num_grow_ch=config['network_g']['num_grow_ch'],
        scale=config['scale']
    )

    # Load model weights
    keyname = 'params_ema' if args.params else 'params'
    model.load_state_dict(torch.load(args.input)[keyname])
    model.cpu().eval()

    # Create example input
    x = torch.rand(1, 3, config['network_g']['num_feat'], config['network_g']['num_feat'])

    # Export to ONNX
    with torch.no_grad():
        torch.onnx.export(
            model, x, args.output,
            opset_version=args.opset_version,
            export_params=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                'output': {0: 'batch_size', 2: 'height', 3: 'width'}
            }
        )
    print(f"Done! Exported to {args.output}")


if __name__ == '__main__':
    """Convert PyTorch model to ONNX"""
    parser = argparse.ArgumentParser(description='Convert PyTorch model to ONNX')
    parser.add_argument(
        '--input', type=str, default='../../RealESRGAN/model/net_g_5000.pth',
        help='Input model path (.pth), defaults to ../../RealESRGAN/model/net_g_5000.pth'
    )
    parser.add_argument(
        '--output', type=str, default='../../RealESRGAN/model/net_g_5000.onnx',
        help='Output ONNX model path, defaults to ../../RealESRGAN/model/net_g_5000.onnx'
    )
    parser.add_argument(
        '--config', type=str, default='config.yml',
        help='Path to config YAML file, defaults to config.yml'
    )
    parser.add_argument(
        '--opset_version', type=int, default=11,
        help='ONNX opset version, defaults to 11'
    )
    parser.add_argument(
        '--params', action='store_false',
        help='Use params instead of params_ema'
    )
    args = parser.parse_args()
    main(args)