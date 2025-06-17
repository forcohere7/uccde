import numpy as np
import cv2
from scipy import stats
from skimage.color import rgb2lab, lab2rgb

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