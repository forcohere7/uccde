import numpy as np
import cv2
from .guided_filter import GuidedFilter
from .config import CONFIG

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

    def _global_stretching_depth(self, img_l):
        """Applies global histogram stretching to the depth map."""
        flat = img_l.ravel()
        indices = np.argsort(flat)
        i_min, i_max = flat[indices[len(flat)//2000]], flat[indices[-len(flat)//2000]]
        return np.clip((img_l - i_min) / (i_max - i_min + 1e-10), 0, 1)