import numpy as np
import cv2
from .guided_filter import GuidedFilter
from .config import CONFIG

class ColourCorrection:
    """
    Optimized color correction for underwater images.
    
    Attributes:
        block_size (int): Size of the block for processing
        gimfilt_radius (int): Radius for guided filter
        eps (float): Regularization parameter for guided filter
    """
    
    def __init__(self, block_size=CONFIG['block_size'], gimfilt_radius=CONFIG['gimfilt_radius'], eps=CONFIG['eps']):
        """
        Initializes the ColourCorrection with specified parameters.
        
        Args:
            block_size (int): Size of the processing block
            gimfilt_radius (int): Radius for guided filtering
            eps (float): Regularization parameter for guided filter
        """
        self.block_size = int(block_size)
        self.gimfilt_radius = int(gimfilt_radius)
        self.eps = float(eps)
    
    def _compensate_rgb(self, image, flag):
        """
        Compensates Red and/or Blue channels using Green channel.
        
        Args:
            image (ndarray): Input RGB image
            flag (int): 0 for both Red and Blue, 1 for Red only
            
        Returns:
            ndarray: Compensated image as uint8
        """
        b, g, r = cv2.split(image.astype(np.float32))
        min_r, max_r = np.min(r), np.max(r)
        min_g, max_g = np.min(g), np.max(g)
        min_b, max_b = np.min(b), np.max(b)
        
        # Handle zero range cases
        if max_r == min_r or max_g == min_g or max_b == min_b:
            return image
            
        # Normalize channels
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
            
        return cv2.merge([
            np.clip(b, 0, 255).astype(np.uint8),
            np.clip(g, 0, 255).astype(np.uint8),
            np.clip(r, 0, 255).astype(np.uint8)
        ])
    
    def _estimate_background_light(self, img, depth_map):
        """
        Estimates background light by averaging top 10 brightest pixels.
        
        Args:
            img (ndarray): Input image
            depth_map (ndarray): Depth map of the image
            
        Returns:
            ndarray: Estimated atmospheric light
        """
        height, width = img.shape[:2]
        img = img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img
        n_bright = int(np.ceil(0.001 * height * width))
        indices = np.argpartition(depth_map.ravel(), -n_bright)[-n_bright:]
        candidates = img.reshape(-1, 3)[indices]
        magnitudes = np.linalg.norm(candidates, axis=1)
        sorted_indices = np.argsort(magnitudes)[::-1]
        top_n = 10
        top_candidates = candidates[sorted_indices[:top_n]]
        return np.mean(top_candidates, axis=0) * 255.0
    
    def _compute_depth_map(self, img):
        """
        Computes the depth map for the image.
        
        Args:
            img (ndarray): Input image
            
        Returns:
            ndarray: Computed depth map
        """
        img = img.astype(np.float32) / 255.0
        x_1 = np.maximum(img[:, :, 0], img[:, :, 1])
        x_2 = img[:, :, 2]
        return 0.51157954 + 0.50516165 * x_1 - 0.90511117 * x_2
    
    def _compute_min_depth(self, img, background_light):
        """
        Computes the minimum depth for the image.
        
        Args:
            img (ndarray): Input image
            background_light (ndarray): Estimated background light
            
        Returns:
            float: Minimum depth value
        """
        img = img.astype(np.float32) / 255.0
        background_light = background_light / 255.0
        max_values = np.max(np.abs(img - background_light), axis=(0, 1)) / np.maximum(background_light, 1 - background_light)
        return 1 - np.max(max_values)
    
    def _global_stretching_depth(self, img_l):
        """
        Applies global histogram stretching to the depth map.
        
        Args:
            img_l (ndarray): Input depth map
            
        Returns:
            ndarray: Stretched and smoothed depth map
        """
        flat = img_l.ravel()
        indices = np.argsort(flat)
        i_min, i_max = flat[indices[len(flat)//1000]], flat[indices[-len(flat)//1000]]
        result = np.clip((img_l - i_min) / (i_max - i_min + 1e-10), 0, 1)
        return cv2.GaussianBlur(result, (3, 3), 0.5)
    
    def _get_rgb_transmission(self, depth_map):
        """
        Computes RGB transmission maps.
        
        Args:
            depth_map (ndarray): Depth map
            
        Returns:
            tuple: Blue, Green, Red transmission maps
        """
        return 0.98 ** depth_map, 0.97 ** depth_map, 0.88 ** depth_map
    
    def _refine_transmission_map(self, transmission_b, transmission_g, transmission_r, img):
        """
        Refines transmission maps using guided filter.
        
        Args:
            transmission_b (ndarray): Blue transmission map
            transmission_g (ndarray): Green transmission map
            transmission_r (ndarray): Red transmission map
            img (ndarray): Input image
            
        Returns:
            ndarray: Refined transmission maps
        """
        guided_filter = GuidedFilter(img, self.gimfilt_radius, self.eps)
        transmission = np.stack([
            guided_filter.filter(transmission_b),
            guided_filter.filter(transmission_g),
            guided_filter.filter(transmission_r)
        ], axis=-1)
        return transmission
    
    def _compute_scene_radiance(self, img, transmission, atmospheric_light):
        """
        Computes the final scene radiance.
        
        Args:
            img (ndarray): Input image
            transmission (ndarray): Transmission maps
            atmospheric_light (ndarray): Estimated background light
            
        Returns:
            ndarray: Computed scene radiance as uint8
        """
        img = img.astype(np.float32)
        min_transmission = 0.2
        transmission = np.maximum(transmission, min_transmission)
        scene_radiance = (img - atmospheric_light) / transmission + atmospheric_light
        return np.clip(scene_radiance, 0, 255).astype(np.uint8)
    
    def process(self, img, rb_compensation_flag=CONFIG['rb_compensation_flag']):
        """
        Applies color correction to preserve image quality.
        
        Args:
            img (ndarray): Input image
            rb_compensation_flag (int): Flag for RB compensation
            
        Returns:
            ndarray: Color-corrected image
        """
        img_compensated = self._compensate_rgb(img, rb_compensation_flag)
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