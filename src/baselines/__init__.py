"""Experimental baselines kept separate from the proposed authentication workflow."""

from src.baselines.opencv_phash import (
    FrameHash,
    OpenCVPHash,
    PHashError,
    compare_video_frame_hashes,
    fit_robust_threshold,
    phash_hamming_distance,
    video_decision,
)

__all__ = [
    "FrameHash",
    "OpenCVPHash",
    "PHashError",
    "compare_video_frame_hashes",
    "fit_robust_threshold",
    "phash_hamming_distance",
    "video_decision",
]
