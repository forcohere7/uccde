import argparse
import cv2
import glob
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='input', help='Input image or folder')
    parser.add_argument('-o', '--output', type=str, default='output', help='Output folder')
    parser.add_argument('-n', '--model_name', type=str, default='RealESRGAN_x4plus', help='Model name (only RealESRGAN_x4plus supported)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pre-trained model')
    parser.add_argument('--outscale', type=float, default=4, help='The final upsampling scale of the image')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored image')
    parser.add_argument('--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    
    args = parser.parse_args()

    # Ensure the model name is RealESRGAN_x4plus
    if args.model_name != 'RealESRGAN_x4plus':
        raise ValueError('This script only supports RealESRGAN_x4plus model')

    # Define the RealESRGAN_x4plus model (x4 RRDBNet)
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    netscale = 4

    # Initialize the upsampler
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=args.model_path,
        model=model,
        tile=args.tile,
        tile_pad=10,
        pre_pad=0,
        half=False  # Use fp32 precision
    )

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Handle input: single file or directory
    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))

    # Process each image
    for idx, path in enumerate(paths):
        imgname, extension = os.path.splitext(os.path.basename(path))
        print(f'Processing {idx}: {imgname}')

        # Read image
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f'Failed to load image: {path}')
            continue

        # Determine image mode (RGB or RGBA)
        img_mode = 'RGBA' if len(img.shape) == 3 and img.shape[2] == 4 else None

        # Enhance image
        try:
            output, _ = upsampler.enhance(img, outscale=args.outscale)
        except RuntimeError as error:
            print(f'Error processing {imgname}: {error}')
            print('Try reducing --tile size if you encounter CUDA out of memory.')
            continue

        # Determine output extension
        extension = extension[1:] if img_mode != 'RGBA' else 'png'

        # Save output
        save_path = os.path.join(args.output, f'{imgname}_{args.suffix}.{extension}')
        cv2.imwrite(save_path, output)
        print(f'Saved: {save_path}')

if __name__ == '__main__':
    main()