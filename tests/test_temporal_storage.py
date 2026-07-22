"""Tests for temporal feature storage and cache helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.features.temporal_features import (
    SEGMENT_FEATURE_NAMES,
    TemporalFeatureResult,
    TemporalPairRecord,
    TemporalSegmentRecord,
)
from src.features.temporal_sampling import TemporalSamplingConfig
from src.features.temporal_storage import (
    build_temporal_cache_key,
    build_temporal_manifest,
    load_temporal_npz,
    save_temporal_manifest,
    save_temporal_npz,
    temporal_manifest_matches,
    temporal_output_paths,
)
from src.features.feature_storage import sha256_file


def _result() -> TemporalFeatureResult:
    return TemporalFeatureResult(
        pair_features=np.asarray([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], dtype=np.float32),
        pair_records=[
            TemporalPairRecord(
                video_id="T001",
                segment_id=0,
                pair_index=0,
                first_requested_timestamp_seconds=0.125,
                second_requested_timestamp_seconds=0.375,
                first_actual_timestamp_seconds=0.125,
                second_actual_timestamp_seconds=0.375,
                temporal_gap_seconds=0.25,
                feature_row_index=0,
                feature_values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                success=True,
            )
        ],
        pair_segment_ids=np.asarray([0], dtype=np.int64),
        pair_indices=np.asarray([0], dtype=np.int64),
        pair_start_timestamps=np.asarray([0.125], dtype=np.float64),
        pair_end_timestamps=np.asarray([0.375], dtype=np.float64),
        segment_ids=np.asarray([0], dtype=np.int64),
        segment_features=np.ones((1, 18), dtype=np.float32),
        segment_records=[
            TemporalSegmentRecord(
                video_id="T001",
                segment_id=0,
                start_time_seconds=0.0,
                end_time_seconds=5.0,
                expected_temporal_frame_count=20,
                decoded_temporal_frame_count=20,
                failed_temporal_frame_count=0,
                expected_pair_count=19,
                successful_pair_count=19,
                failed_pair_count=0,
                missing_pair_count=0,
                segment_feature_row_index=0,
                segment_feature_dimension=18,
                maximum_discontinuity_pair_index=0,
                maximum_discontinuity_timestamp_seconds=0.25,
                success=True,
            )
        ],
        segment_successful_pair_counts=np.asarray([19], dtype=np.int64),
        segment_max_discontinuity_pair_indices=np.asarray([0], dtype=np.int64),
        segment_max_discontinuity_timestamps=np.asarray([0.25], dtype=np.float64),
        warnings=[],
        failures=[],
    )


def test_temporal_npz_and_manifest_serialization(tmp_path: Path) -> None:
    """Temporal feature arrays and manifests are serializable."""

    paths = temporal_output_paths(tmp_path / "temporal", "T001")
    result = _result()
    source_video = tmp_path / "source.mp4"
    segment_manifest = tmp_path / "segments.json"
    source_video.write_bytes(b"video")
    segment_manifest.write_text('{"segments": []}', encoding="utf-8")
    config = TemporalSamplingConfig()

    save_temporal_npz(paths, result, overwrite=True)
    arrays = load_temporal_npz(paths.npz_path)
    assert arrays["pair_features"].shape == (1, 6)
    assert arrays["segment_features"].shape == (1, 18)
    assert arrays["feature_names"].tolist() == SEGMENT_FEATURE_NAMES

    manifest = build_temporal_manifest(
        video_id="T001",
        source_video_path=source_video,
        source_video_sha256=sha256_file(source_video),
        segment_manifest_path=segment_manifest,
        segment_manifest_sha256=sha256_file(segment_manifest),
        config=config,
        result=result,
        paths=paths,
        npz_sha256=sha256_file(paths.npz_path),
        total_processing_time_seconds=1.0,
    )
    save_temporal_manifest(manifest, paths, overwrite=True)
    assert paths.manifest_path.exists()


def test_temporal_cache_matching() -> None:
    """Temporal cache matching uses source and preprocessing configuration."""

    config = TemporalSamplingConfig(sample_fps=4, frame_width=224, frame_height=224)
    cache_key = build_temporal_cache_key("video-sha", "segment-sha", config)
    assert temporal_manifest_matches(dict(cache_key), cache_key)
    changed = dict(cache_key)
    changed["frame_width"] = 112
    assert not temporal_manifest_matches(changed, cache_key)
