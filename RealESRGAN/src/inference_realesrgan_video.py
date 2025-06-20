import argparse
import cv2
import os
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from tqdm import tqdm
import ffmpeg
import mimetypes
import numpy as np

class VideoReader:
    def __init__(self, video_path, ffmpeg_bin='ffmpeg'):
        self.ffmpeg_bin = ffmpeg_bin
        meta = self.get_video_meta_info(video_path)
        self.width = meta['width']
        self.height = meta['height']
        self.fps = meta['fps']
        self.audio = meta['audio']
        self.nb_frames = meta['nb_frames']
        self.stream_reader = (
            ffmpeg.input(video_path).output('pipe:', format='rawvideo', pix_fmt='bgr24', loglevel='error')
            .run_async(pipe_stdin=True, pipe_stdout=True, cmd=ffmpeg_bin)
        )
        self.idx = 0

    def get_video_meta_info(self, video_path):
        probe = ffmpeg.probe(video_path)
        video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
        has_audio = any(stream['codec_type'] == 'audio' for stream in probe['streams'])
        return {
            'width': video_streams[0]['width'],
            'height': video_streams[0]['height'],
            'fps': eval(video_streams[0]['avg_frame_rate']),
            'audio': ffmpeg.input(video_path).audio if has_audio else None,
            'nb_frames': int(video_streams[0]['nb_frames'])
        }

    def get_frame(self):
        if self.idx >= self.nb_frames:
            return None
        img_bytes = self.stream_reader.stdout.read(self.width * self.height * 3)
        if not img_bytes:
            return None
        img = np.frombuffer(img_bytes, np.uint8).reshape([self.height, self.width, 3])
        self.idx += 1
        return img

    def get_resolution(self):
        return self.height, self.width

    def get_fps(self):
        return self.fps

    def get_audio(self):
        return self.audio

    def __len__(self):
        return self.nb_frames

    def close(self):
        self.stream_reader.stdin.close()
        self.stream_reader.wait()

class VideoWriter:
    def __init__(self, video_save_path, audio, height, width, fps, outscale, ffmpeg_bin='ffmpeg'):
        self.ffmpeg_bin = ffmpeg_bin
        out_width, out_height = int(width * outscale), int(height * outscale)
        if out_height > 2160:
            print('Warning: Output video exceeds 4K resolution, which may be slow due to I/O. Consider reducing outscale.')
        input_args = {
            'format': 'rawvideo',
            'pix_fmt': 'bgr24',
            's': f'{out_width}x{out_height}',
            'framerate': fps
        }
        output_args = {
            'pix_fmt': 'yuv420p',
            'vcodec': 'libx264',
            'loglevel': 'error'
        }
        if audio is not None:
            output_args['acodec'] = 'copy'
            self.stream_writer = (
                ffmpeg.input('pipe:', **input_args)
                .output(audio, video_save_path, **output_args)
                .overwrite_output()
                .run_async(pipe_stdin=True, pipe_stdout=True, cmd=ffmpeg_bin)
            )
        else:
            self.stream_writer = (
                ffmpeg.input('pipe:', **input_args)
                .output(video_save_path, **output_args)
                .overwrite_output()
                .run_async(pipe_stdin=True, pipe_stdout=True, cmd=ffmpeg_bin)
            )

    def write_frame(self, frame):
        frame = frame.astype(np.uint8).tobytes()
        self.stream_writer.stdin.write(frame)

    def close(self):
        self.stream_writer.stdin.close()
        self.stream_writer.wait()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='input', help='Input video')
    parser.add_argument('-o', '--output', type=str, default='output', help='Output folder')
    parser.add_argument('-n', '--model_name', type=str, default='RealESRGAN_x4plus', help='Model name (only RealESRGAN_x4plus supported)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pre-trained model')
    parser.add_argument('--outscale', type=float, default=4, help='The final upsampling scale of the video')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored video')
    parser.add_argument('--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--ffmpeg_bin', type=str, default='ffmpeg', help='Path to ffmpeg')
    parser.add_argument('--fp32', action='store_true', help='Use fp32 precision during inference. Default: fp16 (half precision).')
    args = parser.parse_args()

    # Ensure model name is RealESRGAN_x4plus
    if args.model_name != 'RealESRGAN_x4plus':
        raise ValueError('This script only supports RealESRGAN_x4plus model')

    # Validate input and model path
    args.input = args.input.rstrip('/').rstrip('\\')
    if not os.path.isfile(args.input) or not mimetypes.guess_type(args.input)[0].startswith('video'):
        raise ValueError('Input must be a video file')
    if not os.path.isfile(args.model_path):
        raise ValueError(f'Model path {args.model_path} does not exist')

    # Convert .flv to .mp4 if necessary
    if args.input.endswith('.flv'):
        mp4_path = args.input.replace('.flv', '.mp4')
        os.system(f'{args.ffmpeg_bin} -i {args.input} -codec copy {mp4_path}')
        args.input = mp4_path

    # Initialize model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path=args.model_path,
        model=model,
        tile=args.tile,
        tile_pad=10,
        pre_pad=0,
        half=not args.fp32  # Use FP16 by default, FP32 if --fp32 is specified
    )

    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(args.input))[0]
    video_save_path = os.path.join(args.output, f'{video_name}_{args.suffix}.mp4')

    # Process video
    reader = VideoReader(args.input, args.ffmpeg_bin)
    audio = reader.get_audio()
    height, width = reader.get_resolution()
    fps = reader.get_fps()
    writer = VideoWriter(video_save_path, audio, height, width, fps, args.outscale, args.ffmpeg_bin)

    pbar = tqdm(total=len(reader), unit='frame', desc=f'Processing {video_name}')
    while True:
        img = reader.get_frame()
        if img is None:
            break
        try:
            output, _ = upsampler.enhance(img, outscale=args.outscale)
            writer.write_frame(output)
        except RuntimeError as error:
            print(f'Error processing frame: {error}')
            print('Try reducing --tile size if you encounter CUDA out of memory.')
        pbar.update(1)

    reader.close()
    writer.close()
    print(f'Saved: {video_save_path}')

if __name__ == '__main__':
    main()