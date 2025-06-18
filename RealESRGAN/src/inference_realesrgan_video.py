import argparse
import cv2
import os
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from tqdm import tqdm
import ffmpeg

def get_video_meta_info(video_path):
    """Extract metadata from video using ffmpeg-python."""
    try:
        probe = ffmpeg.probe(video_path)
        video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        has_audio = any(s['codec_type'] == 'audio' for s in probe['streams'])
        return {
            'width': video_stream['width'],
            'height': video_stream['height'],
            'fps': eval(video_stream['avg_frame_rate']),
            'audio': has_audio,
            'nb_frames': int(video_stream.get('nb_frames', 0))
        }
    except Exception as e:
        raise RuntimeError(f"Failed to probe video: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Simple Real-ESRGAN video upscaling')
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input video file')
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument('-n', '--model_name', type=str, default='RealESRGAN_x4plus', help='Model name (only RealESRGAN_x4plus supported)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pre-trained model')
    parser.add_argument('--outscale', type=float, default=4, help='The final upsampling scale')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored video')
    parser.add_argument('--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--fp32', action='store_true', help='Use FP32 precision during inference (default: FP16)')
    parser.add_argument('--ffmpeg_bin', type=str, default='ffmpeg', help='Path to ffmpeg executable')
    args = parser.parse_args()

    # Validate model name
    if args.model_name != 'RealESRGAN_x4plus':
        raise ValueError('This script only supports RealESRGAN_x4plus model')

    # Validate input video
    if not os.path.isfile(args.input):
        raise ValueError(f'Input video {args.input} does not exist')

    # Validate model path
    if not os.path.isfile(args.model_path):
        raise ValueError(f'Model path {args.model_path} does not exist')

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Initialize Real-ESRGAN model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    netscale = 4
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=args.model_path,
        model=model,
        tile=args.tile,
        tile_pad=10,
        pre_pad=0,
        half=not args.fp32  # FP16 by default, FP32 if --fp32 is provided
    )

    # Get video metadata
    meta = get_video_meta_info(args.input)
    width, height = meta['width'], meta['height']
    fps = meta['fps']
    has_audio = meta['audio']
    nb_frames = meta['nb_frames']

    # Open input video
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.input}")

    # Prepare output video path
    video_name = os.path.splitext(os.path.basename(args.input))[0]
    video_save_path = os.path.join(args.output, f'{video_name}_{args.suffix}.mp4')

    # Initialize output video writer using ffmpeg-python
    out_width, out_height = int(width * args.outscale), int(height * args.outscale)
    try:
        process = (
            ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{out_width}x{out_height}', framerate=fps)
            .output(
                video_save_path,
                pix_fmt='yuv420p',
                vcodec='libx264',
                loglevel='error',
                **({'acodec': 'copy'} if has_audio else {})
            )
            .overwrite_output()
            .run_async(pipe_stdin=True, cmd=args.ffmpeg_bin)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize video writer: {str(e)}")

    # Process video frames
    pbar = tqdm(total=nb_frames if nb_frames > 0 else None, unit='frame', desc='Processing')
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Enhance frame
        try:
            output, _ = upsampler.enhance(frame, outscale=args.outscale)
        except RuntimeError as error:
            print(f'Error processing frame: {error}')
            print('Try reducing --tile size if you encounter CUDA out of memory.')
            continue

        # Write frame to output
        try:
            process.stdin.write(output.astype(np.uint8).tobytes())
        except Exception as e:
            print(f'Error writing frame: {str(e)}')
            break

        pbar.update(1)

    # Clean up
    cap.release()
    try:
        process.stdin.close()
        process.wait()
    except Exception as e:
        print(f"Error closing video writer: {str(e)}")
    pbar.close()

    # If video has audio, re-mux audio from input to output
    if has_audio:
        temp_path = os.path.join(args.output, f'temp_{video_name}.mp4')
        try:
            (
                ffmpeg.input(args.input).audio
                .output(
                    ffmpeg.input(video_save_path).video,
                    temp_path,
                    vcodec='copy',
                    acodec='copy',
                    loglevel='error'
                )
                .overwrite_output()
                .run(cmd=args.ffmpeg_bin)
            )
            os.replace(temp_path, video_save_path)
        except Exception as e:
            print(f"Error re-muxing audio: {str(e)}")

    print(f'Saved: {video_save_path}')

if __name__ == '__main__':
    main()