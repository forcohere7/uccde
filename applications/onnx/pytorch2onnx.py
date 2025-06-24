import argparse
import torch
import torch.onnx
from basicsr.archs.rrdbnet_arch import RRDBNet

def main(args):
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

    keyname = 'params' if args.params else 'params_ema'
    model.load_state_dict(torch.load(args.input)[keyname])
    model.eval()

    x = torch.rand(1, 3, 64, 64)

    with torch.no_grad():
        torch.onnx.export(
            model, x, args.output, opset_version=11, export_params=True
        )

    print(f"ONNX model exported to: {args.output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert PyTorch model to ONNX")
    parser.add_argument(
        '--input',
        type=str,
        default='../../RealESRGAN/model/net_g_5000.pth',
        help='Input model path'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../../RealESRGAN/model/net_g_5000.onnx',
        help='Output ONNX path'
    )
    parser.add_argument(
        '--params',
        action='store_true',
        help='Use params instead of params_ema (default is params_ema)'
    )

    args = parser.parse_args()
    main(args)