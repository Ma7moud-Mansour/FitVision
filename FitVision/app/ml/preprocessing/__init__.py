"""
FitVision — Image Processing Preprocessing Package
===================================================
CS389 Image Processing — Phase 2

Provides three preprocessing pipeline modes:
  - "raw"     : resize only (baseline — no image processing)
  - "cleaned" : resize + Gaussian + Bilateral filtering
  - "full"    : resize + Gaussian + Bilateral + CLAHE + Morphological opening

Usage:
    from app.ml.preprocessing import build_pipeline, preprocess_frame

    pipeline = build_pipeline(mode="full")
    processed = pipeline(frame)
"""

from app.ml.preprocessing.image_cleaner import (
    apply_gaussian_filter,
    apply_median_filter,
    apply_bilateral_filter,
    clean_frame,
)
from app.ml.preprocessing.image_enhancer import (
    apply_clahe,
    apply_histogram_equalization,
    apply_morphological_opening,
    apply_morphological_closing,
    detect_edges,
)
from app.ml.preprocessing.frame_pipeline import (
    build_pipeline,
    preprocess_frame,
    PIPELINE_MODES,
)

__all__ = [
    # Cleaning
    "apply_gaussian_filter",
    "apply_median_filter",
    "apply_bilateral_filter",
    "clean_frame",
    # Enhancement
    "apply_clahe",
    "apply_histogram_equalization",
    "apply_morphological_opening",
    "apply_morphological_closing",
    "detect_edges",
    # Pipeline
    "build_pipeline",
    "preprocess_frame",
    "PIPELINE_MODES",
]
