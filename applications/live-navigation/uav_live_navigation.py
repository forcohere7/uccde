import cv2
import numpy as np
from scipy import stats
from skimage.color import rgb2hsv, hsv2rgb, rgb2lab, lab2rgb

# Configuration settings
CONFIG = {
    'block_size': 9,
    'gimfilt_radius': 30,
    'eps': 1e-2,
    'rb_compensation_flag': 0,  # 0: Compensate both Red and Blue, 1: Compensate only Red
    'enhancement_strength': 0.6,  # Control enhancement intensity
}

# Set numpy to ignore overflow warnings
np.seterr(over='ignore')

# Guided Filter Class
class GuidedFilter:
    """Guided filter for image processing to refine transmission maps while preserving edges."""
    def __init__(self, input_image, radius=5, epsilon=0.4):
        self._radius = 2 * radius + 1
        self._epsilon = epsilon
        self._input_image = self._to_float_img(input_image)
        self._init_filter()

    def _to_float_img(self, img):
        if img.dtype == np.float32:
            return img
        return img.astype(np.float32) / 255.0

    def _init_filter(self):
        img = self._input_image
        r = self._radius
        eps = self._epsilon
        ir, ig, ib = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        ksize = (r, r)
        self._ir_mean = cv2.blur(ir, ksize)
        self._ig_mean = cv2.blur(ig, ksize)
        self._ib_mean = cv2.blur(ib, ksize)
        irr = cv2.blur(ir * ir, ksize) - self._ir_mean ** 2 + eps
        irg = cv2.blur(ir * ig, ksize) - self._ir_mean * self._ig_mean
        irb = cv2.blur(ir * ib, ksize) - self._ir_mean * self._ib_mean
        igg = cv2.blur(ig * ig, ksize) - self._ig_mean ** 2 + eps
        igb = cv2.blur(ig * ib, ksize) - self._ig_mean * self._ib_mean
        ibb = cv2.blur(ib * ib, ksize) - self._ib_mean ** 2 + eps
        det = irr * (igg * ibb - igb * igb) - irg * (irg * ibb - igb * irb) + irb * (irg * igb - igg * irb)
        self._irr_inv = (igg * ibb - igb * igb) / det
        self._irg_inv = -(irg * ibb - igb * irb) / det
        self._irb_inv = (irg * igb - igg * irb) / det
        self._igg_inv = (irr * ibb - irb * irb) / det
        self._igb_inv = -(irr * igb - irb * irg) / det
        self._ibb_inv = (irr * igg - irg * irg) / det

    def _compute_coefficients(self, input_p):
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
        ar_mean, ag_mean, ab_mean, b_mean = ab
        ir, ig, ib = self._input_image[:, :, 0], self._input_image[:, :, 1], self._input_image[:, :, 2]
        return ar_mean * ir + ag_mean * ig + ab_mean * ib + b_mean

    def filter(self, input_p):
        p_32f = self._to_float_img(input_p)
        ab = self._compute_coefficients(p_32f)
        return self._compute_output(ab)

# Colour Correction Class
class ColourCorrection:
    """Handles underwater image color correction by compensating for color casts and estimating depth."""
    def __init__(self, block_size=CONFIG['block_size'], gimfilt_radius=CONFIG['gimfilt_radius'], eps=CONFIG['eps']):
        self.block_size = block_size
        self.gimfilt_radius = gimfilt_radius
        self.eps = eps

    def _compensate_rb(self, image, flag):
        b, g, r = cv2.split(image.astype(np.float64))
        min_r, max_r = np.min(r), np.max(r)
        min_g, max_g = np.min(g), np.max(g)
        min_b, max_b = np.min(b), np.max(b)
        if max_r == min_r or max_g == min_g or max_b == min_b:
            return image
        r = (r - min_r) / (max_r - min_r)
        g = (g - min_g) / (max_g - min_g)
        b = (b - min_b) / (max_b - min_b)
        mean_r, mean_g, mean_b = np.mean(r), np.mean(g), np.mean(b)
        compensation_strength = 0.4
        if flag == 0:
            r = (r + compensation_strength * (mean_g - mean_r) * (1 - r) * g) * max_r
            b = (b + compensation_strength * (mean_g - mean_b) * (1 - b) * g) * max_b
            g = g * max_g
        elif flag == 1:
            r = (r + compensation_strength * (mean_g - mean_r) * (1 - r) * g) * max_r
            g = g * max_g
            b = b * max_b
        return cv2.merge([np.clip(b, 0, 255).astype(np.uint8),
                         np.clip(g, 0, 255).astype(np.uint8),
                         np.clip(r, 0, 255).astype(np.uint8)])

    def _estimate_background_light(self, img, depth_map):
        img = img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img
        height, width = img.shape[:2]
        n_bright = int(np.ceil(0.001 * height * width))
        indices = np.argpartition(depth_map.ravel(), -n_bright)[-n_bright:]
        candidates = img.reshape(-1, 3)[indices]
        magnitudes = np.linalg.norm(candidates, axis=1)
        sorted_indices = np.argsort(magnitudes)[::-1]
        top_n = 10
        top_candidates = candidates[sorted_indices[:top_n]]
        atmospheric_light = np.mean(top_candidates, axis=0) * 255.0
        return atmospheric_light

    def _compute_depth_map(self, img):
        img = img.astype(np.float32) / 255.0
        x_1 = np.maximum(img[:, :, 0], img[:, :, 1])
        x_2 = img[:, :, 2]
        return 0.51157954 + 0.50516165 * x_1 - 0.90511117 * x_2

    def _compute_min_depth(self, img, background_light):
        img = img.astype(np.float32) / 255.0
        background_light = background_light / 255.0
        max_values = np.max(np.abs(img - background_light), axis=(0, 1)) / np.maximum(background_light, 1 - background_light)
        return 1 - np.max(max_values)

    def _global_stretching_depth(self, img_l):
        flat = img_l.ravel()
        indices = np.argsort(flat)
        i_min, i_max = flat[indices[len(flat)//1000]], flat[indices[-len(flat)//1000]]
        result = np.clip((img_l - i_min) / (i_max - i_min + 1e-10), 0, 1)
        return cv2.GaussianBlur(result, (3, 3), 0.5)

    def _get_rgb_transmission(self, depth_map):
        return 0.98 ** depth_map, 0.97 ** depth_map, 0.88 ** depth_map

    def _refine_transmission_map(self, transmission_b, transmission_g, transmission_r, img):
        guided_filter = GuidedFilter(img, self.gimfilt_radius, self.eps)
        transmission = np.stack([
            guided_filter.filter(transmission_b),
            guided_filter.filter(transmission_g),
            guided_filter.filter(transmission_r)
        ], axis=-1)
        return transmission

    def _compute_scene_radiance(self, img, transmission, atmospheric_light):
        img = img.astype(np.float32)
        min_transmission = 0.2
        transmission = np.maximum(transmission, min_transmission)
        scene_radiance = (img - atmospheric_light) / transmission + atmospheric_light
        return np.clip(scene_radiance, 0, 255).astype(np.uint8)

    def process(self, img, rb_compensation_flag=CONFIG['rb_compensation_flag']):
        img_compensated = self._compensate_rb(img, rb_compensation_flag)
        depth_map = self._compute_depth_map(img_compensated)
        depth_map = self._global_stretching_depth(depth_map)
        guided_filter = GuidedFilter(img_compensated, self.gimfilt_radius, self.eps)
        refined_depth_map = guided_filter.filter(depth_map)
        refined_depth_map = np.clip(refined_depth_map, 0, 1)
        atmospheric_light = self._estimate_background_light(img_compensated, depth_map)
        d_0 = self._compute_min_depth(img_compensated, atmospheric_light)
        d_f = 6 * (depth_map + d_0)
        transmission_b, transmission_g, transmission_r = self._get_rgb_transmission(d_f)
        transmission = self._refine_transmission_map(transmission_b, transmission_g, transmission_r, img_compensated)
        return self._compute_scene_radiance(img_compensated, transmission, atmospheric_light)

# Image Enhancement Functions
def cal_equalisation(img, ratio):
    return np.clip(img * ratio, 0, 255)

def rgb_equalisation(img):
    img = img.astype(np.float32)
    current_mean = np.mean(img, axis=(0, 1))
    target_mean = 140
    ratio = target_mean / (current_mean + 1e-10)
    ratio = np.clip(ratio, 0.8, 1.2)
    return cal_equalisation(img, ratio)

def stretch_range(r_array, height, width):
    flat = r_array.ravel()
    mode = stats.mode(flat, keepdims=True).mode[0] if flat.size > 0 else np.median(flat)
    mode_indices = np.where(flat == mode)[0]
    mode_index_before = mode_indices[0] if mode_indices.size > 0 else len(flat) // 2
    dr_min = (1 - 0.755) * mode
    max_index = min(len(flat) - 1, len(flat) - int((len(flat) - mode_index_before) * 0.01))
    sr_max = np.sort(flat)[max_index]
    return dr_min, sr_max, mode

def global_stretching_ab(a, height, width):
    return a * (1.05 ** (1 - np.abs(a / 128)))

def basic_stretching(img):
    img = img.astype(np.float64)
    min_vals = np.percentile(img, 2, axis=(0,1))
    max_vals = np.percentile(img, 98, axis=(0,1))
    range_vals = max_vals - min_vals
    min_range = 50
    mask = range_vals < min_range
    max_vals[mask] = min_vals[mask] + min_range
    img = np.clip((img - min_vals) * 255 / (max_vals - min_vals + 1e-10), 0, 255)
    return img.astype(np.uint8)

def global_stretching_luminance(img_l, height, width):
    flat = img_l.ravel()
    indices = np.argsort(flat)
    i_min, i_max = flat[indices[len(flat)//50]], flat[indices[-len(flat)//50]]
    if i_max == i_min:
        i_min, i_max = flat.min(), flat.max()
        if i_max == i_min:
            return img_l
    return np.clip((img_l - i_min) * 95 / (i_max - i_min + 1e-10), 0, 100)

def lab_stretching(scene_radiance):
    scene_radiance = np.clip(scene_radiance, 0, 255).astype(np.uint8)
    original = scene_radiance.copy()
    img_lab = rgb2lab(scene_radiance)
    l, a, b = img_lab[:, :, 0], img_lab[:, :, 1], img_lab[:, :, 2]
    img_lab[:, :, 0] = global_stretching_luminance(l, *scene_radiance.shape[:2])
    img_lab[:, :, 1] = global_stretching_ab(a, *scene_radiance.shape[:2])
    img_lab[:, :, 2] = global_stretching_ab(b, *scene_radiance.shape[:2])
    enhanced = lab2rgb(img_lab) * 255
    blend_factor = CONFIG['enhancement_strength']
    result = blend_factor * enhanced + (1 - blend_factor) * original
    return result

def global_stretching_advanced(r_array, height, width, lambda_val, k_val):
    flat = r_array.ravel()
    indices = np.argsort(flat)
    i_min, i_max = flat[indices[len(flat)//100]], flat[indices[-len(flat)//100]]
    dr_min, sr_max, mode = stretch_range(r_array, height, width)
    t_n = lambda_val ** 2
    o_max_left = sr_max * t_n * k_val / mode
    o_max_right = 255 * t_n * k_val / mode
    dif = o_max_right - o_max_left
    if dif >= 1:
        indices = np.arange(1, int(dif) + 1)
        sum_val = np.sum((1.326 + indices) * mode / (t_n * k_val))
        dr_max = sum_val / int(dif)
        p_out = np.where(r_array < i_min, (r_array - i_min) * (dr_min / i_min) + i_min,
                         np.where(r_array > i_max, (r_array - dr_max) * (dr_max / i_max) + i_max,
                                  ((r_array - i_min) * (255 - i_min) / (i_max - i_min) + i_min)))
    else:
        p_out = np.where(r_array < i_min, (r_array - r_array.min()) * (dr_min / r_array.min()) + r_array.min(),
                         ((r_array - i_min) * (255 - dr_min) / (i_max - i_min) + dr_min))
    return p_out

def relative_stretching(scene_radiance, height, width):
    scene_radiance = scene_radiance.astype(np.float64)
    scene_radiance[:, :, 0] = global_stretching_advanced(scene_radiance[:, :, 0], height, width, 0.98, 1.1)
    scene_radiance[:, :, 1] = global_stretching_advanced(scene_radiance[:, :, 1], height, width, 0.97, 1.1)
    scene_radiance[:, :, 2] = global_stretching_advanced(scene_radiance[:, :, 2], height, width, 0.88, 0.9)
    return scene_radiance

def image_enhancement(scene_radiance):
    if scene_radiance.shape[2] == 3:
        scene_radiance = cv2.cvtColor(scene_radiance, cv2.COLOR_BGR2RGB)
    if np.max(scene_radiance) == np.min(scene_radiance):
        return scene_radiance
    original = scene_radiance.copy()
    scene_radiance = scene_radiance.astype(np.float64)
    scene_radiance = basic_stretching(scene_radiance)
    scene_radiance = lab_stretching(scene_radiance)
    final_blend = 0.8
    result = final_blend * scene_radiance + (1 - final_blend) * original
    return np.clip(result, 0, 255).astype(np.uint8)

# Process frame function
def process_frame(frame, colour_corrector):
    corrected_frame = colour_corrector.process(frame)
    enhanced_frame = image_enhancement(corrected_frame)
    return cv2.cvtColor(enhanced_frame, cv2.COLOR_RGB2BGR)

# Main function for live feed processing
def main_live_feed():
    """Main function to process live camera feed for underwater navigation."""
    # Initialize video capture (0 for default camera, adjust if needed)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Initialize colour corrector
    colour_corrector = ColourCorrection()

    print("Starting live feed processing. Press Ctrl+C to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Process the frame
        processed_frame = process_frame(frame, colour_corrector)

        # Display the processed frame
        cv2.imshow('Underwater Navigation - Live Feed', processed_frame)

        # Exit on Ctrl+C
        cv2.waitKey(1)

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Live feed processing stopped.")

if __name__ == '__main__':
    main_live_feed()