"""Tests for stream-preserving normalized feature fusion."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.features.alignment import AlignedFeatureSet
from src.features.fusion import COMBINED_FEATURE_DIMENSION, combine_normalized_streams
from src.features.normalization import RobustNormalizer


def _aligned() -> AlignedFeatureSet:
    resnet = np.vstack([np.zeros(1024, dtype=np.float32), np.ones(1024, dtype=np.float32)])
    temporal = np.vstack([np.zeros(18, dtype=np.float32), np.ones(18, dtype=np.float32)])
    return AlignedFeatureSet(
        video_id="T001",
        segment_ids=np.asarray([0, 1], dtype=np.int64),
        segment_start_times=np.asarray([0.0, 5.0], dtype=np.float64),
        segment_end_times=np.asarray([5.0, 10.0], dtype=np.float64),
        resnet_features=resnet,
        temporal_features=temporal,
        resnet_source_path=Path("resnet.npz"),
        temporal_source_path=Path("temporal.npz"),
        resnet_source_sha256="resnet-sha",
        temporal_source_sha256="temporal-sha",
        warnings=[],
    )


def test_fusion_shapes_boundaries_and_ordering() -> None:
    """Fusion preserves stream boundaries and deterministic concatenation."""

    aligned = _aligned()
    resnet_normalizer = RobustNormalizer.fit(aligned.resnet_features)
    temporal_normalizer = RobustNormalizer.fit(aligned.temporal_features)
    bundle = combine_normalized_streams(aligned, resnet_normalizer, temporal_normalizer)

    assert bundle.resnet_normalized_features.shape == (2, 1024)
    assert bundle.temporal_normalized_features.shape == (2, 18)
    assert bundle.combined_normalized_features.shape == (2, COMBINED_FEATURE_DIMENSION)
    assert bundle.stream_boundaries["resnet"] == {"start_index": 0, "end_index": 1023, "dimension": 1024}
    assert bundle.stream_boundaries["temporal"] == {"start_index": 1024, "end_index": 1041, "dimension": 18}
    assert np.allclose(bundle.combined_normalized_features[:, :1024], bundle.resnet_normalized_features)
    assert np.allclose(bundle.combined_normalized_features[:, 1024:], bundle.temporal_normalized_features)
    assert bundle.segment_ids.tolist() == [0, 1]
    assert bundle.finite()
