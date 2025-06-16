import os
import cv2
import datetime
import numpy as np
import natsort
import math
from scipy import stats
from skimage.color import rgb2hsv, hsv2rgb, rgb2lab, lab2rgb
from multiprocessing import Pool, cpu_count
from functools import partial
import ffmpeg

# Configuration settings
CONFIG = {
    'input_dir': 'input',
    'output_dir': 'output',
    'block_size': 9,
    'gimfilt_radius': 50,
    'eps': 1e-3,
    'rb_compensation_flag': 0,  # 0: Compensate both Red and Blue, 1: Compensate only Red
    'video_extensions': ['.mp4', '.avi', '.mov'],  # Supported video formats
    'output_video_fps': 30,  # Default output video frame rate
    'output_video_codec': 'mp4v',  # Codec for output video
    'temp_video_path': 'temp_output.mp4',  # Temporary video file without audio
}

# Set numpy to ignore overflow warnings
np.seterr(over='ignore')

# ==================== OPTIMIZED GUIDED FILTER CLASS ====================

class GuidedFilter:
    """Optimized guided filter for image processing."""
    def __init__(self, input_image, radius=5, epsilon=0.4):
        self._radius = 2 * radius + 1
        self._epsilon = epsilon
        self._input_image = self._to_float_img(input_image)
        self._init_filter()

    def _to_float_img(self, img):
        """Converts image to float32 format, normalizing if necessary."""
        if img.dtype == np.float32:
            return img
        return img.astype(np.float32) / 255.0

    def _init_filter(self):
        """Initializes filter parameters for guided filtering."""
        img = self._input_image
        r = self._radius
        eps = self._epsilon
        ir, ig, ib = img[:, :, 0], img[:, :, 1], img[:, :, 2]

        # Precompute means
        ksize = (r, r)
        self._ir_mean = cv2.blur(ir, ksize)
        self._ig_mean = cv2.blur(ig, ksize)
        self._ib_mean = cv2.blur(ib, ksize)

        # Compute variances and covariances in one pass
        irr = cv2.blur(ir * ir, ksize) - self._ir_mean ** 2 + eps
        irg = cv2.blur(ir * ig, ksize) - self._ir_mean * self._ig_mean
        irb = cv2.blur(ir * ib, ksize) - self._ir_mean * self._ib_mean
        igg = cv2.blur(ig * ig, ksize) - self._ig_mean ** 2 + eps
        igb = cv2.blur(ig * ib, ksize) - self._ig_mean * self._ib_mean
        ibb = cv2.blur(ib * ib, ksize) - self._ib_mean ** 2 + eps

        # Compute inverse covariance matrix
        det = irr * (igg * ibb - igb * igb) - irg * (irg * ibb - igb * irb) + irb * (irg * igb - igg * irb)
        self._irr_inv = (igg * ibb - igb * igb) / det
        self._irg_inv = -(irg * ibb - igb * irb) / det
        self._irb_inv = (irg * igb - igg * irb) / det
        self._igg_inv = (irr * ibb - irb * irb) / det
        self._igb_inv = -(irr * igb - irb * irg) / det
        self._ibb_inv = (irr * igg - irg * irg) / det

    def _compute_coefficients(self, input_p):
        """Computes filter coefficients for the input image."""
        r = self._radius
        ksize = (r, r)
        ir, ig, ib = self._input_image[:, :, 0], self._input_image[:, :, 1], self._input_image[:, :, 2]
        p_mean = cv2.blur(input_p, ksize)
        ipr_cov = cv2.blur(ir * input_p, ksize) - self._ir_mean * p_mean
        ipg_cov = cv2.blur(ig * input_p, ksize) - self._ig_mean * p_mean
        ipb_cov = cv2.blur(ib * input_p, ksize) - self._ib_mean * p_mean
        ar = self._irr_inv * ipr_cov + self._irg_inv * ipg_cov + self._irb_inv * ipb_cov
        ag = self._irg_inv * ipr_cov + self._igg_inv * ipg_cov + self._igb_inv * ipb_cov
        ab = self._irb_inv * ipr_cov + self._igb_inv * ipg_cov + self._ibb_inv * ipb_cov
        b = p_mean - ar * self._ir_mean - ag * self._ig_mean - ab * self._ib_mean
        return cv2.blur(ar, ksize), cv2.blur(ag, ksize), cv2.blur(ab, ksize), cv2.blur(b, ksize)

    def _compute_output(self, ab):
        """Computes the output of the guided filter."""
        ar_mean, ag_mean, ab_mean, b_mean = ab
        ir, ig, ib = self._input_image[:, :, 0], self._input_image[:, :, 1], self._input_image[:, :, 2]
        return ar_mean * ir + ag_mean * ig + ab_mean * ib + b_mean

    def filter(self, input_p):
        """Applies the guided filter to the input."""
        p_32f = self._to_float_img(input_p)
        ab = self._compute_coefficients(p_32f)
        return self._compute_output(ab)

# ==================== OPTIMIZED COLOUR CORRECTION CLASS ====================

class ColourCorrection:
    """Optimized color correction for underwater images."""
    def __init__(self, block_size=CONFIG['block_size'], gimfilt_radius=CONFIG['gimfilt_radius'], eps=CONFIG['eps']):
        self.block_size = block_size
        self.gimfilt_radius = gimfilt_radius
        self.eps = eps

    def _compensate_rb(self, image, flag):
        """Compensates Red and/or Blue channels using Green channel."""
        b, g, r = cv2.split(image.astype(np.float64))
        min_r, max_r = np.min(r), np.max(r)
        min_g, max_g = np.min(g), np.max(g)
        min_b, max_b = np.min(b), np.max(b)
        r = np.where(max_r != min_r, (r - min_r) / (max_r - min_r), r - min_r)
        g = np.where(max_g != min_g, (g - min_g) / (max_g - min_g), g - min_g)
        b = np.where(max_b != min_b, (b - min_b) / (max_b - min_b), b - min_b)
        mean_r, mean_g, mean_b = np.mean(r), np.mean(g), np.mean(b)
        if flag == 0:
            r = (r + (mean_g - mean_r) * (1 - r) * g) * max_r
            b = (b + (mean_g - mean_b) * (1 - b) * g) * max_b
            g = g * max_g
        elif flag == 1:
            r = (r + (mean_g - mean_r) * (1 - r) * g) * max_r
            g = g * max_g
            b = b * max_b
        return cv2.merge([np.clip(b, 0, 255).astype(np.uint8),
                         np.clip(g, 0, 255).astype(np.uint8),
                         np.clip(r, 0, 255).astype(np.uint8)])

    def _estimate_background_light(self, img, depth_map):
        """Estimates the background light for the image."""
        height, width = img.shape[:2]
        img = img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img
        n_bright = int(np.ceil(0.001 * height * width))
        indices = np.argpartition(depth_map.ravel(), -n_bright)[-n_bright:]
        candidates = img.reshape(-1, 3)[indices]
        magnitudes = np.linalg.norm(candidates, axis=1)
        max_idx = np.argmax(magnitudes)
        return candidates[max_idx] * 255.0

    def _compute_depth_map(self, img):
        """Computes the depth map for the image."""
        img = img.astype(np.float32) / 255.0
        x_1 = np.maximum(img[:, :, 0], img[:, :, 1])
        x_2 = img[:, :, 2]
        return 0.51157954 + 0.50516165 * x_1 - 0.90511117 * x_2

    def _compute_min_depth(self, img, background_light):
        """Computes the minimum depth for the image."""
        img = img.astype(np.float32) / 255.0
        background_light = background_light / 255.0
        max_values = np.max(np.abs(img - background_light), axis=(0, 1)) / np.maximum(background_light, 1 - background_light)
        return 1 - np.max(max_values)

    def _global_stretching_depth(self, img_l):
        """Applies global histogram stretching to the depth map."""
        flat = img_l.ravel()
        indices = np.argsort(flat)
        i_min, i_max = flat[indices[len(flat)//2000]], flat[indices[-len(flat)//2000]]
        return np.clip((img_l - i_min) / (i_max - i_min + 1e-10), 0, 1)

    def _get_rgb_transmission(self, depth_map):
        """Computes RGB transmission maps."""
        return 0.97 ** depth_map, 0.95 ** depth_map, 0.83 ** depth_map

    def _refine_transmission_map(self, transmission_b, transmission_g, transmission_r, img):
        """Refines transmission maps using guided filter."""
        guided_filter = GuidedFilter(img, self.gimfilt_radius, self.eps)
        transmission = np.stack([
            guided_filter.filter(transmission_b),
            guided_filter.filter(transmission_g),
            guided_filter.filter(transmission_r)
        ], axis=-1)
        return transmission

    def _compute_scene_radiance(self, img, transmission, atmospheric_light):
        """Computes the final scene radiance."""
        img = img.astype(np.float16)
        scene_radiance = (img - atmospheric_light) / np.maximum(transmission, 1e-10) + atmospheric_light
        return np.clip(scene_radiance, 0, 255).astype(np.uint8)

    def process(self, img, rb_compensation_flag=CONFIG['rb_compensation_flag']):
        """Applies color correction to the input image."""
        img_compensated = self._compensate_rb(img, rb_compensation_flag)
        depth_map = self._compute_depth_map(img_compensated)
        depth_map = self._global_stretching_depth(depth_map)
        guided_filter = GuidedFilter(img_compensated, self.gimfilt_radius, self.eps)
        refined_depth_map = guided_filter.filter(depth_map)
        refined_depth_map = np.clip(refined_depth_map, 0, 1)
        atmospheric_light = self._estimate_background_light(img_compensated, depth_map)
        d_0 = self._compute_min_depth(img_compensated, atmospheric_light)
        d_f = 8 * (depth_map + d_0)
        transmission_b, transmission_g, transmission_r = self._get_rgb_transmission(d_f)
        transmission = self._refine_transmission_map(transmission_b, transmission_g, transmission_r, img_compensated)
        return self._compute_scene_radiance(img_compensated, transmission, atmospheric_light)

# ==================== OPTIMIZED IMAGE ENHANCEMENT FUNCTIONS ====================

def cal_equalisation(img, ratio):
    """Applies equalization to the image with given ratio."""
    return np.clip(img * ratio, 0, 255)

def rgb_equalisation(img):
    """Equalizes RGB channels of the image."""
    img = img.astype(np.float32)
    ratio = 128 / (np.mean(img, axis=(0, 1)) + 1e-10)
    return cal_equalisation(img, ratio[:, None, None])

def stretch_range(r_array, height, width):
    """Computes stretching range for histogram equalization."""
    flat = r_array.ravel()
    mode = stats.mode(flat, keepdims=True).mode[0] if flat.size > 0 else np.median(flat)
    mode_indices = np.where(flat == mode)[0]
    mode_index_before = mode_indices[0] if mode_indices.size > 0 else len(flat) // 2
    dr_min = (1 - 0.655) * mode
    max_index = min(len(flat) - 1, len(flat) - int((len(flat) - mode_index_before) * 0.005))
    sr_max = np.sort(flat)[max_index]
    return dr_min, sr_max, mode

def global_stretching_ab(a, height, width):
    """Applies global stretching to a or b channel in LAB color space."""
    return a * (1.3 ** (1 - np.abs(a / 128)))

def basic_stretching(img):
    """Applies basic stretching to each channel of the image."""
    img = img.astype(np.float64)
    min_vals = np.min(img, axis=(0, 1), keepdims=True)
    max_vals = np.max(img, axis=(0, 1), keepdims=True)
    img = np.where(max_vals != min_vals, (img - min_vals) * 255 / (max_vals - min_vals), img)
    return np.clip(img, 0, 255).astype(np.uint8)

def global_stretching_luminance(img_l, height, width):
    """Applies global histogram stretching to luminance channel."""
    flat = img_l.ravel()
    indices = np.argsort(flat)
    i_min, i_max = flat[indices[len(flat)//100]], flat[indices[-len(flat)//100]]
    if i_max == i_min:
        i_min, i_max = flat.min(), flat.max()
        if i_max == i_min:
            return img_l
    return np.clip((img_l - i_min) * 100 / (i_max - i_min + 1e-10), 0, 100)

def lab_stretching(scene_radiance):
    """Applies stretching in LAB color space."""
    scene_radiance = np.clip(scene_radiance, 0, 255).astype(np.uint8)
    img_lab = rgb2lab(scene_radiance)
    l, a, b = img_lab[:, :, 0], img_lab[:, :, 1], img_lab[:, :, 2]
    img_lab[:, :, 0] = global_stretching_luminance(l, *scene_radiance.shape[:2])
    img_lab[:, :, 1] = global_stretching_ab(a, *scene_radiance.shape[:2])
    img_lab[:, :, 2] = global_stretching_ab(b, *scene_radiance.shape[:2])
    return lab2rgb(img_lab) * 255

def global_stretching_advanced(r_array, height, width, lambda_val, k_val):
    """Applies advanced global stretching to an image channel."""
    flat = r_array.ravel()
    indices = np.argsort(flat)
    i_min, i_max = flat[indices[len(flat)//200]], flat[indices[-len(flat)//200]]
    dr_min, sr_max, mode = stretch_range(r_array, height, width)
    t_n = lambda_val ** 4
    o_max_left = sr_max * t_n * k_val / mode
    o_max_right = 255 * t_n * k_val / mode
    dif = o_max_right - o_max_left
    if dif >= 1:
        indices = np.arange(1, int(dif) + 1)
        sum_val = np.sum((1.526 + indices) * mode / (t_n * k_val))
        dr_max = sum_val / int(dif)
        p_out = np.where(r_array < i_min, (r_array - i_min) * (dr_min / i_min) + i_min,
                         np.where(r_array > i_max, (r_array - dr_max) * (dr_max / i_max) + i_max,
                                  ((r_array - i_min) * (255 - i_min) / (i_max - i_min) + i_min)))
    else:
        p_out = np.where(r_array < i_min, (r_array - r_array.min()) * (dr_min / r_array.min()) + r_array.min(),
                         ((r_array - i_min) * (255 - dr_min) / (i_max - i_min) + dr_min))
    return p_out

def relative_stretching(scene_radiance, height, width):
    """Applies relative stretching to RGB channels."""
    scene_radiance = scene_radiance.astype(np.float64)
    scene_radiance[:, :, 0] = global_stretching_advanced(scene_radiance[:, :, 0], height, width, 0.97, 1.25)
    scene_radiance[:, :, 1] = global_stretching_advanced(scene_radiance[:, :, 1], height, width, 0.95, 1.25)
    scene_radiance[:, :, 2] = global_stretching_advanced(scene_radiance[:, :, 2], height, width, 0.83, 0.85)
    return scene_radiance

def image_enhancement(scene_radiance):
    """Enhances the input image using various stretching techniques."""
    if scene_radiance.shape[2] == 3:
        scene_radiance = cv2.cvtColor(scene_radiance, cv2.COLOR_BGR2RGB)
    if np.max(scene_radiance) == np.min(scene_radiance):
        return scene_radiance
    scene_radiance = scene_radiance.astype(np.float64)
    scene_radiance = basic_stretching(scene_radiance)
    scene_radiance = lab_stretching(scene_radiance)
    return np.clip(scene_radiance, 0, 255).astype(np.uint8)

# ==================== OPTIMIZED PROCESSING FUNCTIONS ====================

def process_image(file, input_dir=CONFIG['input_dir'], output_dir=CONFIG['output_dir']):
    """Processes a single image."""
    file_path = os.path.join(input_dir, file)
    prefix = file.split('.')[0]
    print(f'Processing image: {file}')
    img = cv2.imread(file_path)
    if img is None:
        print(f"Could not read image: {file}")
        return
    colour_corrector = ColourCorrection()
    print("Applying color correction...")
    corrected_img = colour_corrector.process(img)
    print("Applying image enhancement...")
    final_result = image_enhancement(corrected_img)
    final_result_bgr = cv2.cvtColor(final_result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_dir, f'{prefix}_ColourCorrected.jpg'), final_result_bgr)
    print(f"Completed processing image: {file}")

def process_video_frame(frame, colour_corrector):
    """Processes a single video frame."""
    corrected_frame = colour_corrector.process(frame)
    enhanced_frame = image_enhancement(corrected_frame)
    return cv2.cvtColor(enhanced_frame, cv2.COLOR_RGB2BGR)

def process_video(file, input_dir=CONFIG['input_dir'], output_dir=CONFIG['output_dir']):
    """Processes a single video and preserves original audio."""
    file_path = os.path.join(input_dir, file)
    prefix = file.split('.')[0]
    print(f'Processing video: {file}')
    
    # Open video file
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f"Could not open video: {file}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or CONFIG['output_video_fps']
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Define output video (temporary, without audio)
    temp_output_path = os.path.join(output_dir, CONFIG['temp_video_path'])
    final_output_path = os.path.join(output_dir, f'{prefix}_ColourCorrected.mp4')
    fourcc = cv2.VideoWriter_fourcc(*CONFIG['output_video_codec'])
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Could not create output video: {temp_output_path}")
        cap.release()
        return
    
    # Initialize colour corrector
    colour_corrector = ColourCorrection()
    
    # Process frames
    print(f"Processing {frame_count} frames...")
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        print(f"Processing frame {frame_idx + 1}/{frame_count}")
        processed_frame = process_video_frame(frame, colour_corrector)
        out.write(processed_frame)
        frame_idx += 1
    
    # Release resources
    cap.release()
    out.release()
    
    # Merge original audio with processed video using ffmpeg
    try:
        print(f"Merging original audio into: {final_output_path}")
        input_video = ffmpeg.input(temp_output_path)
        input_audio = ffmpeg.input(file_path).audio
        output = ffmpeg.output(input_video.video, input_audio, final_output_path, vcodec='copy', acodec='copy', strict='experimental')
        ffmpeg.run(output, overwrite_output=True)
        print(f"Completed processing video with audio: {file}")
        
        # Clean up temporary file
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
    except ffmpeg.Error as e:
        print(f"Error merging audio: {e.stderr.decode()}")
        return
    
def main():
    """Main function to process images and videos in the input directory."""
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    if not os.access(CONFIG['output_dir'], os.W_OK):
        print(f"No write permissions for directory {CONFIG['output_dir']}")
        exit(1)

    start_time = datetime.datetime.now()
    files = natsort.natsorted(os.listdir(CONFIG['input_dir']))
    files = [f for f in files if os.path.isfile(os.path.join(CONFIG['input_dir'], f))]

    # Separate images and videos
    image_files = [f for f in files if os.path.splitext(f)[1].lower() not in CONFIG['video_extensions']]
    video_files = [f for f in files if os.path.splitext(f)[1].lower() in CONFIG['video_extensions']]

    # Process images using multiprocessing
    if image_files:
        print(f"Processing {len(image_files)} images...")
        with Pool(processes=cpu_count()) as pool:
            pool.map(process_image, image_files)

    # Process videos sequentially
    if video_files:
        print(f"Processing {len(video_files)} videos...")
        for video_file in video_files:
            process_video(video_file)

    print(f'Total processing time: {datetime.datetime.now() - start_time}')

if __name__ == "__main__":
    main()