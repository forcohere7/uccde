import numpy as np
import cv2

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

        # Compute variances and covariances
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