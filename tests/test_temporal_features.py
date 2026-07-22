"""Tests for interpretable temporal frame-difference features."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.features.temporal_features import (
    PAIR_FEATURE_NAMES,
    SEGMENT_FEATURE_NAMES,
    TemporalFeatureError,
    aggregate_temporal_segment_features,
    calculate_pair_features,
    extract_temporal_features_from_frames,
)
from src.features.temporal_sampling import TemporalFrameRecord, TemporalSamplingConfig
from src.video.segmentation import load_segment_manifest


def _frame(segment_id: int, frame_index: int, value: float, success: bool = True) -> TemporalFrameRecord:
    timestamp = segment_id * 5 + 0.125 + frame_index * 0.25
    return TemporalFrameRecord(
        video_id="T001",
        segment_id=segment_id,
        frame_index=frame_index,
        requested_timestamp_seconds=timestamp,
        actual_timestamp_seconds=timestamp,
        success=success,
        error_message=None if success else "decode failed",
        image=np.full((8, 8), value, dtype=np.float32) if success else None,
    )


def _segment_manifest(path: Path) -> Path:
    manifest = {
        "video_id": "T001",
        "source_video_path": "/tmp/source.mp4",
        "video_metadata_reference": None,
        "segment_duration_seconds": 5.0,
        "sample_frames_per_second": 1.0,
        "incomplete_segment_policy": "discard",
        "number_complete_segments": 1,
        "processed_duration_seconds": 5.0,
        "discarded_duration_seconds": 0.0,
        "segments": [
            {
                "video_id": "T001",
                "segment_id": 0,
                "start_time_seconds": 0.0,
                "end_time_seconds": 5.0,
                "duration_seconds": 5.0,
                "is_complete": True,
                "expected_sample_count": 5,
                "source_video_path": "/tmp/source.mp4",
            }
        ],
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _empty_segment_manifest(path: Path) -> Path:
    manifest = {
        "video_id": "T001",
        "source_video_path": "/tmp/source.mp4",
        "video_metadata_reference": None,
        "segment_duration_seconds": 5.0,
        "sample_frames_per_second": 1.0,
        "incomplete_segment_policy": "discard",
        "number_complete_segments": 0,
        "processed_duration_seconds": 0.0,
        "discarded_duration_seconds": 4.9,
        "segments": [],
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_identical_frames_have_near_zero_features() -> None:
    """Identical frames produce near-zero temporal differences."""

    first = np.full((16, 16), 0.5, dtype=np.float32)
    second = np.full((16, 16), 0.5, dtype=np.float32)

    features = calculate_pair_features(first, second)

    assert features.shape == (6,)
    assert np.allclose(features, 0.0)


def test_different_frames_have_high_changed_ratio() -> None:
    """Abrupt full-frame changes produce large difference values."""

    first = np.zeros((16, 16), dtype=np.float32)
    second = np.ones((16, 16), dtype=np.float32)

    features = calculate_pair_features(first, second)

    assert features[0] > 0.9
    assert features[2] > 0.9
    assert features[3] == pytest.approx(1.0)


def test_small_noise_is_lower_than_abrupt_change() -> None:
    """Small compression-like noise is lower than a full-frame abrupt change."""

    base = np.full((16, 16), 0.5, dtype=np.float32)
    noise = base + np.full((16, 16), 2 / 255.0, dtype=np.float32)
    abrupt = np.ones((16, 16), dtype=np.float32)

    noisy_features = calculate_pair_features(base, noise)
    abrupt_features = calculate_pair_features(base, abrupt)

    assert noisy_features[0] < abrupt_features[0]
    assert noisy_features[2] < abrupt_features[2]
    assert noisy_features[3] < abrupt_features[3]


def test_segment_feature_vector_order_and_aggregation() -> None:
    """Segment aggregation produces 18 ordered mean/std/max features."""

    pair_features = np.asarray(
        [
            [1.0, 2.0, 3.0, 0.1, 4.0, 0.2],
            [3.0, 6.0, 9.0, 0.5, 8.0, 0.6],
        ],
        dtype=np.float32,
    )

    segment_vector = aggregate_temporal_segment_features(pair_features)

    assert PAIR_FEATURE_NAMES[0] == "mean_absolute_difference"
    assert SEGMENT_FEATURE_NAMES[:6] == [
        "mean_mad",
        "mean_absdiff_std",
        "mean_rmse",
        "mean_changed_ratio",
        "mean_p90",
        "mean_edge_change",
    ]
    assert segment_vector.shape == (18,)
    assert np.allclose(segment_vector[:6], np.mean(pair_features, axis=0))
    assert np.allclose(segment_vector[6:12], np.std(pair_features, axis=0, ddof=0))
    assert np.allclose(segment_vector[12:], np.max(pair_features, axis=0))


def test_missing_frame_handling_and_zero_pair_failure(tmp_path: Path) -> None:
    """Missing frames create failed pair records; zero valid pairs fail the segment."""

    manifest = load_segment_manifest(_segment_manifest(tmp_path / "segments.json"))
    config = TemporalSamplingConfig(sample_fps=4)

    frames = [
        _frame(0, 0, 0.0),
        _frame(0, 1, 0.1),
        _frame(0, 2, 0.2, success=False),
    ]
    result = extract_temporal_features_from_frames("T001", manifest, frames, config)
    assert result.pair_features.shape == (1, 6)
    assert sum(1 for record in result.pair_records if not record.success) == 1
    assert result.warnings

    with pytest.raises(TemporalFeatureError, match="zero valid"):
        extract_temporal_features_from_frames(
            "T001",
            manifest,
            [_frame(0, 0, 0.0), _frame(0, 1, 0.1, success=False)],
            config,
        )


def test_no_complete_segments_returns_empty_feature_arrays(tmp_path: Path) -> None:
    """Videos with no complete segments produce empty deterministic arrays."""

    manifest = load_segment_manifest(_empty_segment_manifest(tmp_path / "empty_segments.json"))
    result = extract_temporal_features_from_frames(
        "T001",
        manifest,
        [],
        TemporalSamplingConfig(sample_fps=4),
    )

    assert result.pair_features.shape == (0, 6)
    assert result.segment_features.shape == (0, 18)
    assert result.segment_ids.shape == (0,)
    assert result.pair_records == []
    assert result.segment_records == []
