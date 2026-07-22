"""Stream-preserving normalized feature fusion helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.features.alignment import (
    RESNET_SEGMENT_DIMENSION,
    TEMPORAL_SEGMENT_DIMENSION,
    AlignedFeatureSet,
)
from src.features.normalization import RobustNormalizer


COMBINED_FEATURE_DIMENSION = RESNET_SEGMENT_DIMENSION + TEMPORAL_SEGMENT_DIMENSION
STREAM_BOUNDARIES = {
    "resnet": {
        "start_index": 0,
        "end_index": RESNET_SEGMENT_DIMENSION - 1,
        "dimension": RESNET_SEGMENT_DIMENSION,
    },
    "temporal": {
        "start_index": RESNET_SEGMENT_DIMENSION,
        "end_index": COMBINED_FEATURE_DIMENSION - 1,
        "dimension": TEMPORAL_SEGMENT_DIMENSION,
    },
}


class FeatureFusionError(RuntimeError):
    """Raised when normalized feature streams cannot be combined safely."""


@dataclass(frozen=True)
class NormalizedFeatureBundle:
    """Raw, normalized, and combined segment-level feature arrays."""

    video_id: str
    segment_ids: np.ndarray
    segment_start_times: np.ndarray
    segment_end_times: np.ndarray
    resnet_raw_features: np.ndarray
    temporal_raw_features: np.ndarray
    resnet_normalized_features: np.ndarray
    temporal_normalized_features: np.ndarray
    combined_normalized_features: np.ndarray
    stream_boundaries: dict[str, dict[str, int]]

    def value_range(self) -> tuple[float, float]:
        """Return min and max over all normalized feature values."""

        if self.combined_normalized_features.size == 0:
            return 0.0, 0.0
        return (
            float(np.min(self.combined_normalized_features)),
            float(np.max(self.combined_normalized_features)),
        )

    def finite(self) -> bool:
        """Return whether all normalized feature arrays contain finite values."""

        return bool(
            np.all(np.isfinite(self.resnet_normalized_features))
            and np.all(np.isfinite(self.temporal_normalized_features))
            and np.all(np.isfinite(self.combined_normalized_features))
        )


def combine_normalized_streams(
    aligned: AlignedFeatureSet,
    resnet_normalizer: RobustNormalizer,
    temporal_normalizer: RobustNormalizer,
) -> NormalizedFeatureBundle:
    """Normalize ResNet and temporal streams separately, then concatenate them."""

    resnet_normalized = resnet_normalizer.transform(aligned.resnet_features)
    temporal_normalized = temporal_normalizer.transform(aligned.temporal_features)
    if resnet_normalized.shape != (aligned.segment_count, RESNET_SEGMENT_DIMENSION):
        raise FeatureFusionError(f"Unexpected normalized ResNet shape: {resnet_normalized.shape}.")
    if temporal_normalized.shape != (aligned.segment_count, TEMPORAL_SEGMENT_DIMENSION):
        raise FeatureFusionError(f"Unexpected normalized temporal shape: {temporal_normalized.shape}.")
    combined = np.concatenate([resnet_normalized, temporal_normalized], axis=1).astype(np.float32)
    if combined.shape != (aligned.segment_count, COMBINED_FEATURE_DIMENSION):
        raise FeatureFusionError(f"Unexpected combined normalized shape: {combined.shape}.")
    if not np.all(np.isfinite(combined)):
        raise FeatureFusionError("Combined normalized features contain non-finite values.")
    return NormalizedFeatureBundle(
        video_id=aligned.video_id,
        segment_ids=aligned.segment_ids.astype(np.int64, copy=True),
        segment_start_times=aligned.segment_start_times.astype(np.float64, copy=True),
        segment_end_times=aligned.segment_end_times.astype(np.float64, copy=True),
        resnet_raw_features=aligned.resnet_features.astype(np.float32, copy=True),
        temporal_raw_features=aligned.temporal_features.astype(np.float32, copy=True),
        resnet_normalized_features=resnet_normalized,
        temporal_normalized_features=temporal_normalized,
        combined_normalized_features=combined,
        stream_boundaries={name: dict(values) for name, values in STREAM_BOUNDARIES.items()},
    )


def stream_boundaries_for_manifest() -> dict[str, dict[str, Any]]:
    """Return JSON-serializable stream boundary metadata."""

    return {name: dict(values) for name, values in STREAM_BOUNDARIES.items()}
