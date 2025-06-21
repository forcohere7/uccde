import numpy as np
import cv2
from scipy import stats
from skimage.color import rgb2lab, lab2rgb
from .config import CONFIG

def cal_equalisation(img, ratio):
    """
    Applies equalization to the image with given ratio.
    
    Args:
        img (ndarray): Input image array
        ratio (float): Equalization ratio
    
    Returns:
        ndarray: Equalized image clipped to [0, 255]
    """
    return np.clip(img * ratio, 0, 255)

def rgb_equalisation(img):
    """
    Equalizes RGB channels of the image.
    
    Args:
        img (ndarray): Input RGB image
    
    Returns:
        ndarray: Equalized RGB image
    """
    img = img.astype(np.float32)
    current_mean = np.mean(img, axis=(0, 1))
    target_mean = 140
    ratio = target_mean / (current_mean + 1e-10)
    ratio = np.clip(ratio, 0.8, 1.2)
    return cal_equalisation(img, ratio)

def stretch_range(r_array, height, width):
    """
    Computes stretching range for histogram equalization.
    
    Args:
        r_array (ndarray): Input channel array
        height (int): Image height
        width (int): Image width
    
    Returns:
        tuple: (dr_min, sr_max, mode) stretching parameters
    """
    flat = r_array.ravel()
    mode = stats.mode(flat, keepdims=True).mode[0] if flat.size > 0 else np.median(flat)
    mode_indices = np.where(flat == mode)[0]
    mode_index_before = mode_indices[0] if mode_indices.size > 0 else len(flat) // 2
    dr_min = (1 - 0.755) * mode
    max_index = min(len(flat) - 1, len(flat) - int((len(flat) - mode_index_before) * 0.01))
    sr_max = np.sort(flat)[max_index]
    return dr_min, sr_max, mode

def global_stretching_ab(a, height, width):
    """
    Applies global stretching to a or b channel in LAB color space with reduced enhancement.
    
    Args:
        a (ndarray): a or b channel array
        height (int): Image height
        width (int): Image width
    
    Returns:
        ndarray: Stretched channel
    """
    return a * (1.05 ** (1 - np.abs(a / 128)))

def basic_stretching(img):
    """
    Applies basic stretching to each channel using percentiles.
    
    Args:
        img (ndarray): Input image
    
    Returns:
        ndarray: Stretched image as uint8
    """
    img = img.astype(np.float64)
    min_vals = np.percentile(img, 2, axis=(0, 1))
    max_vals = np.percentile(img, 98, axis=(0, 1))
    
    range_vals = max_vals - min_vals
    min_range = 50
    mask = range_vals < min_range
    max_vals[mask] = min_vals[mask] + min_range
    
    img = np.clip((img - min_vals) * 255 / (max_vals - min_vals + 1e-10), 0, 255)
    return img.astype(np.uint8)

def global_stretching_luminance(img_l, height, width):
    """
    Applies global histogram stretching to luminance channel.
    
    Args:
        img_l (ndarray): Luminance channel
        height (int): Image height
        width (int): Image width
    
    Returns:
        ndarray: Stretched luminance channel
    """
    flat = img_l.ravel()
    indices = np.argsort(flat)
    i_min, i_max = flat[indices[len(flat)//50]], flat[indices[-len(flat)//50]]
    if i_max == i_min:
        i_min, i_max = flat.min(), flat.max()
        if i_max == i_min:
            return img_l
    return np.clip((img_l - i_min) * 95 / (i_max - i_min + 1e-10), 0, 100)

def lab_stretching(scene_radiance):
    """
    Applies stretching in LAB color space.
    
    Args:
        scene_radiance (ndarray): Input RGB image
    
    Returns:
        ndarray: Enhanced image
    """
    scene_radiance = np.clip(scene_radiance, 0, 255).astype(np.uint8)
    original = scene_radiance.copy()
    
    img_lab = rgb2lab(scene_radiance)
    l, a, b = img_lab[:, :, 0], img_lab[:, :, 1], img_lab[:, :, 2]
    img_lab[:, :, 0] = global_stretching_luminance(l, *scene_radiance.shape[:2])
    img_lab[:, :, 1] = global_stretching_ab(a, *scene_radiance.shape[:2])
    img_lab[:, :, 2] = global_stretching_ab(b, *scene_radiance.shape[:2])
    enhanced = lab2rgb(img_lab) * 255
    
    blend_factor = CONFIG['enhancement_strength']  # Default: 0.6
    result = blend_factor * enhanced + (1 - blend_factor) * original
    return result

def global_stretching_advanced(r_array, height, width, lambda_val, k_val):
    """
    Applies advanced global stretching to an image channel.
    
    Args:
        r_array (ndarray): Input channel array
        height (int): Image height
        width (int): Image width
        lambda_val (float): Lambda parameter
        k_val (float): K parameter
    
    Returns:
        ndarray: Stretched channel
    """
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
    """
    Applies relative stretching to RGB channels.
    
    Args:
        scene_radiance (ndarray): Input RGB image
        height (int): Image height
        width (int): Image width
    
    Returns:
        ndarray: Stretched image
    """
    scene_radiance = scene_radiance.astype(np.float64)
    scene_radiance[:, :, 0] = global_stretching_advanced(scene_radiance[:, :, 0], height, width, 0.98, 1.1)
    scene_radiance[:, :, 1] = global_stretching_advanced(scene_radiance[:, :, 1], height, width, 0.97, 1.1)
    scene_radiance[:, :, 2] = global_stretching_advanced(scene_radiance[:, :, 2], height, width, 0.88, 0.9)
    return scene_radiance

def image_enhancement(scene_radiance):
    """
    Enhances the input image using various stretching techniques.
    
    Args:
        scene_radiance (ndarray): Input BGR or RGB image
    
    Returns:
        ndarray: Enhanced image as uint8
    """
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