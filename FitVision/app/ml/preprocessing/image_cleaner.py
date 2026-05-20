"""
FitVision — Image Cleaning Module
===================================
CS389 Image Processing — Phase 2

Implements noise-reduction filters applied BEFORE pose estimation.
Cleaning goal: reduce noise/artifacts while preserving body contour edges
               that MediaPipe Pose needs for accurate landmark detection.

Filters implemented:
  1. Gaussian Filter     — remove high-frequency noise & compression artifacts
  2. Median Filter       — remove salt-and-pepper (impulse) noise
  3. Bilateral Filter    — edge-preserving smoothing (PREFERRED for pose input)

Recommended order: Gaussian first → then Bilateral
  Gaussian removes random noise that would confuse the bilateral filter's
  range kernel; bilateral then preserves the now-cleaner edges.
"""

import cv2
import numpy as np


# ─── 1. GAUSSIAN FILTER ───────────────────────────────────────────────────────

def apply_gaussian_filter(
    frame: np.ndarray,
    ksize: tuple = (5, 5),
    sigma_x: float = 1.0,
) -> np.ndarray:
    """
    Apply Gaussian blur to reduce high-frequency noise.

    The Gaussian kernel assigns higher weights to pixels closer to the center,
    producing smooth, isotropic noise reduction.

    Formula:
        G(x, y) = (1 / 2πσ²) × exp(-(x² + y²) / 2σ²)

    Parameters
    ----------
    frame   : np.ndarray  Input BGR frame (H, W, 3)
    ksize   : tuple       Kernel size — must be odd (default: 5×5)
    sigma_x : float       Gaussian sigma (default: 1.0)

    Returns
    -------
    np.ndarray  Filtered frame, same shape as input

    Use-case in FitVision
    ---------------------
    Reduces MPEG-4 block artifacts and sensor noise before bilateral filtering.
    Slight edge softening is an acceptable trade-off for cleaner input.
    """
    return cv2.GaussianBlur(frame, ksize=ksize, sigmaX=sigma_x)


# ─── 2. MEDIAN FILTER ─────────────────────────────────────────────────────────

def apply_median_filter(
    frame: np.ndarray,
    ksize: int = 3,
) -> np.ndarray:
    """
    Apply median filter to remove impulse (salt-and-pepper) noise.

    As a non-linear operation, the median completely removes extreme outlier
    pixels without the blurring side-effect of averaging filters.

    Parameters
    ----------
    frame : np.ndarray  Input BGR frame (H, W, 3)
    ksize : int         Kernel size — must be odd (default: 3)

    Returns
    -------
    np.ndarray  Filtered frame, same shape as input

    Use-case in FitVision
    ---------------------
    Eliminates bright/dark pixel spikes from video transmission errors.
    Better edge preservation than Gaussian for impulse-type noise.
    """
    return cv2.medianBlur(frame, ksize=ksize)


# ─── 3. BILATERAL FILTER ──────────────────────────────────────────────────────

def apply_bilateral_filter(
    frame: np.ndarray,
    d: int = 9,
    sigma_color: float = 75,
    sigma_space: float = 75,
) -> np.ndarray:
    """
    Apply edge-preserving bilateral filter.

    Extends Gaussian filtering with a range kernel: pixels with very different
    intensities from the center receive low weight, preventing edge blurring.

    Formula:
        BF[p] = (1/Wp) × Σ G_s(||p-q||) × G_r(|I_p - I_q|) × I_q
        where G_s = spatial kernel, G_r = intensity-range kernel

    Parameters
    ----------
    frame       : np.ndarray  Input BGR frame (H, W, 3)
    d           : int         Pixel neighborhood diameter (default: 9)
    sigma_color : float       Intensity range sensitivity (default: 75)
                              Higher = smoother regions are broader
    sigma_space : float       Spatial Gaussian sigma (default: 75)
                              Higher = farther pixels influence each other

    Returns
    -------
    np.ndarray  Filtered frame, same shape as input

    Use-case in FitVision
    ---------------------
    Smooths background texture WITHOUT blurring body joint edges.
    MediaPipe Pose landmark detection improves because body contours are cleaner.
    Trade-off: slower than Gaussian (~18ms/frame vs ~1ms for Gaussian).
    """
    return cv2.bilateralFilter(frame, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


# ─── COMBINED CLEANING PIPELINE ───────────────────────────────────────────────

def clean_frame(
    frame: np.ndarray,
    gaussian_ksize: tuple = (5, 5),
    gaussian_sigma: float = 1.0,
    bilateral_d: int = 9,
    bilateral_sigma_color: float = 75,
    bilateral_sigma_space: float = 75,
) -> np.ndarray:
    """
    Full cleaning pipeline: Gaussian → Bilateral.

    Step 1 (Gaussian): Remove random noise that would confuse the bilateral
                       filter's range kernel.
    Step 2 (Bilateral): Preserve the now-cleaner body contour edges while
                        smoothing remaining background texture variations.

    Parameters
    ----------
    frame                  : np.ndarray  Input BGR frame (H, W, 3)
    gaussian_ksize         : tuple       Gaussian kernel size (default: 5×5)
    gaussian_sigma         : float       Gaussian sigma (default: 1.0)
    bilateral_d            : int         Bilateral neighborhood diameter (default: 9)
    bilateral_sigma_color  : float       Bilateral intensity range (default: 75)
    bilateral_sigma_space  : float       Bilateral spatial range (default: 75)

    Returns
    -------
    np.ndarray  Cleaned frame, same shape as input
    """
    frame = apply_gaussian_filter(frame, ksize=gaussian_ksize, sigma_x=gaussian_sigma)
    frame = apply_bilateral_filter(
        frame, d=bilateral_d,
        sigma_color=bilateral_sigma_color,
        sigma_space=bilateral_sigma_space,
    )
    return frame
