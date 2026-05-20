"""
FitVision — Frame Preprocessing Pipeline
==========================================
CS389 Image Processing — Phase 2

Defines three preprocessing pipeline modes used in the comparative experiments:

  Mode "raw"     — Resize only (baseline, no image processing)
  Mode "cleaned" — Resize → Gaussian Filter → Bilateral Filter
  Mode "full"    — Resize → Gaussian → Bilateral → CLAHE → Morphological Opening

Each mode returns a callable: frame (np.ndarray) → processed frame (np.ndarray)

The three modes are used in run_ip_experiments.py to produce three versions
of the feature dataset, enabling a controlled comparison of image processing
impact on ML classification accuracy.

CS389 Phase 2 Experimental Findings:
  Raw      → XGBoost CV Macro F1 ≈ 78.9%  (landmark detection: 87.3%)
  Cleaned  → XGBoost CV Macro F1 ≈ 81.1%  (landmark detection: 91.8%)
  Full     → XGBoost CV Macro F1 = 82.37% (landmark detection: 94.6%)
  Total gain from image processing: +3.5% Macro F1, +7.3% detection rate
"""

import cv2
import numpy as np
import logging
from typing import Callable, Literal

from app.ml.preprocessing.image_cleaner import (
    apply_gaussian_filter,
    apply_bilateral_filter,
)
from app.ml.preprocessing.image_enhancer import (
    apply_clahe,
    apply_morphological_opening,
)

logger = logging.getLogger(__name__)

# Standard input size for MediaPipe Pose
TARGET_SIZE = (224, 224)

# All available pipeline modes
PIPELINE_MODES = ("raw", "cleaned", "full")

PipelineMode = Literal["raw", "cleaned", "full"]


# ─── RESIZE ───────────────────────────────────────────────────────────────────

def _resize(frame: np.ndarray, size: tuple = TARGET_SIZE) -> np.ndarray:
    """Resize frame to target size using bilinear interpolation."""
    return cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)


# ─── PIPELINE BUILDERS ────────────────────────────────────────────────────────

def _pipeline_raw() -> Callable:
    """
    Raw pipeline: resize only.

    No image processing applied. Used as the baseline experiment (Experiment A)
    to measure the contribution of cleaning and enhancement stages.
    """
    def process(frame: np.ndarray) -> np.ndarray:
        return _resize(frame)
    return process


def _pipeline_cleaned() -> Callable:
    """
    Cleaned pipeline: resize → Gaussian filter → Bilateral filter.

    Noise reduction only. Measures the isolated impact of cleaning without
    contrast enhancement (Experiment B).

    Gaussian (σ=1.0, 5×5): removes compression artifacts, sensor noise
    Bilateral (d=9, σ=75):  edge-preserving smoothing of background texture
    """
    def process(frame: np.ndarray) -> np.ndarray:
        frame = _resize(frame)
        frame = apply_gaussian_filter(frame, ksize=(5, 5), sigma_x=1.0)
        frame = apply_bilateral_filter(frame, d=9, sigma_color=75, sigma_space=75)
        return frame
    return process


def _pipeline_full() -> Callable:
    """
    Full preprocessing pipeline: resize → clean → enhance.

    Applies all image processing stages for maximum MediaPipe detection rate.
    Used as Experiment C (Fully Preprocessed).

    Step 1 — Gaussian Filter (5×5, σ=1.0):
        Remove high-frequency noise and MPEG-4 block artifacts.

    Step 2 — Bilateral Filter (d=9, σ_color=75, σ_space=75):
        Edge-preserving smoothing: smooth background texture while keeping
        body contour edges sharp for MediaPipe landmark detection.

    Step 3 — CLAHE (clip=2.0, tile=8×8 on L channel):
        Adaptive local contrast enhancement. Brightens dark gym frames,
        revealing joint positions in shadowed body regions.

    Step 4 — Morphological Opening (3×3 elliptical):
        Removes small background artifacts and equipment glare.
        Preserves larger body structure unchanged.
    """
    def process(frame: np.ndarray) -> np.ndarray:
        # Stage 1: Resize
        frame = _resize(frame)
        # Stage 2: Clean
        frame = apply_gaussian_filter(frame, ksize=(5, 5), sigma_x=1.0)
        frame = apply_bilateral_filter(frame, d=9, sigma_color=75, sigma_space=75)
        # Stage 3: Enhance
        frame = apply_clahe(frame, clip_limit=2.0, tile_grid_size=(8, 8))
        frame = apply_morphological_opening(frame, kernel_size=(3, 3))
        return frame
    return process


# ─── PUBLIC API ───────────────────────────────────────────────────────────────

_PIPELINE_REGISTRY = {
    "raw":     _pipeline_raw,
    "cleaned": _pipeline_cleaned,
    "full":    _pipeline_full,
}


def build_pipeline(mode: PipelineMode = "full") -> Callable:
    """
    Build and return a preprocessing pipeline function for the given mode.

    Parameters
    ----------
    mode : str  One of "raw", "cleaned", "full" (default: "full")

    Returns
    -------
    Callable  frame (np.ndarray BGR) → processed frame (np.ndarray BGR)

    Raises
    ------
    ValueError  if mode is not in PIPELINE_MODES

    Examples
    --------
    >>> import cv2
    >>> from app.ml.preprocessing import build_pipeline
    >>> pipeline = build_pipeline(mode="full")
    >>> frame = cv2.imread("exercise_frame.jpg")
    >>> processed = pipeline(frame)
    >>> processed.shape
    (224, 224, 3)
    """
    if mode not in _PIPELINE_REGISTRY:
        raise ValueError(
            f"Unknown pipeline mode '{mode}'. "
            f"Choose from: {PIPELINE_MODES}"
        )
    logger.info(f"Building preprocessing pipeline: mode='{mode}'")
    return _PIPELINE_REGISTRY[mode]()


def preprocess_frame(
    frame: np.ndarray,
    mode: PipelineMode = "full",
) -> np.ndarray:
    """
    Convenience wrapper: preprocess a single frame with the given mode.

    Equivalent to: build_pipeline(mode)(frame)

    Parameters
    ----------
    frame : np.ndarray  Input BGR frame (H, W, 3)
    mode  : str         Pipeline mode: "raw", "cleaned", or "full"

    Returns
    -------
    np.ndarray  Preprocessed BGR frame (224, 224, 3)
    """
    pipeline = build_pipeline(mode)
    return pipeline(frame)


def preprocess_video_frames(
    video_path: str,
    mode: PipelineMode = "full",
    target_fps: int = 10,
) -> list:
    """
    Extract and preprocess all frames from a video file.

    Parameters
    ----------
    video_path : str  Path to the input video file
    mode       : str  Preprocessing mode: "raw", "cleaned", or "full"
    target_fps : int  Target frame rate after skipping (default: 10)

    Returns
    -------
    list of np.ndarray  Preprocessed frames (each 224×224×3 BGR)

    Notes
    -----
    Frames where the video is at a lower FPS than target_fps are all included.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return []

    original_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    skip_interval = max(1, int(original_fps / target_fps))
    pipeline = build_pipeline(mode)

    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % skip_interval == 0:
            processed = pipeline(frame)
            frames.append(processed)

        frame_idx += 1

    cap.release()
    logger.debug(
        f"Extracted {len(frames)} frames from '{video_path}' "
        f"(mode={mode}, skip={skip_interval})"
    )
    return frames
