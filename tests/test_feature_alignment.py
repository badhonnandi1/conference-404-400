
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.features.alignment import FeatureAlignmentError, load_aligned_features


def _write_alignment_fixture(
    root: Path,
    video_id: str = "T001",
    resnet_ids: list[int] | None = None,
    temporal_ids: list[int] | None = None,
    resnet_dim: int = 1024,
    temporal_dim: int = 18,
    resnet_manifest_video_id: str | None = None,
    temporal_manifest_video_id: str | None = None,
    nonfinite: bool = False,
) -> tuple[Path, Path, Path]:
    resnet_ids = [2, 0, 1] if resnet_ids is None else resnet_ids
    temporal_ids = [1, 2, 0] if temporal_ids is None else temporal_ids
    resnet_dir = root / "resnet" / video_id
    temporal_dir = root / "temporal" / video_id
    resnet_dir.mkdir(parents=True)
    temporal_dir.mkdir(parents=True)
    resnet_path = resnet_dir / f"{video_id}_resnet_features.npz"
    temporal_path = temporal_dir / f"{video_id}_temporal_features.npz"

    resnet_features = np.asarray(
        [np.full(resnet_dim, segment_id, dtype=np.float32) for segment_id in resnet_ids]
    )
    temporal_features = np.asarray(
        [np.full(temporal_dim, segment_id + 10, dtype=np.float32) for segment_id in temporal_ids]
    )
    if nonfinite:
        resnet_features[0, 0] = np.nan
    np.savez_compressed(
        resnet_path,
        segment_ids=np.asarray(resnet_ids, dtype=np.int64),
        segment_combined_embeddings=resnet_features,
    )
    np.savez_compressed(
        temporal_path,
        segment_ids=np.asarray(temporal_ids, dtype=np.int64),
        segment_features=temporal_features,
    )
    (resnet_dir / f"{video_id}_resnet_manifest.json").write_text(
        json.dumps({"video_id": resnet_manifest_video_id or video_id, "warnings": []}),
        encoding="utf-8",
    )
    (temporal_dir / f"{video_id}_temporal_manifest.json").write_text(
        json.dumps(
            {
                "video_id": temporal_manifest_video_id or video_id,
                "segment_records": [
                    {
                        "segment_id": segment_id,
                        "start_time_seconds": float(segment_id * 5),
                        "end_time_seconds": float(segment_id * 5 + 5),
                        "success": True,
                    }
                    for segment_id in sorted(set(temporal_ids))
                ],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    segment_manifest = root / f"{video_id}_segments.json"
    segment_manifest.write_text(
        json.dumps(
            {
                "video_id": video_id,
                "source_video_path": "/tmp/video.mp4",
                "video_metadata_reference": None,
                "segment_duration_seconds": 5,
                "sample_frames_per_second": 1,
                "incomplete_segment_policy": "discard",
                "number_complete_segments": 3,
                "processed_duration_seconds": 15,
                "discarded_duration_seconds": 0,
                "segments": [
                    {
                        "video_id": video_id,
                        "segment_id": segment_id,
                        "start_time_seconds": float(segment_id * 5),
                        "end_time_seconds": float(segment_id * 5 + 5),
                        "duration_seconds": 5,
                        "is_complete": True,
                        "expected_sample_count": 5,
                        "source_video_path": "/tmp/video.mp4",
                    }
                    for segment_id in [0, 1, 2]
                ],
            }
        ),
        encoding="utf-8",
    )
    return resnet_path, temporal_path, segment_manifest


def test_correct_alignment_and_deterministic_ordering(tmp_path: Path) -> None:
    """Matching segment IDs are sorted and aligned by segment ID."""

    resnet_path, temporal_path, segment_manifest = _write_alignment_fixture(tmp_path)
    aligned = load_aligned_features("T001", resnet_path, temporal_path, segment_manifest)

    assert aligned.segment_ids.tolist() == [0, 1, 2]
    assert aligned.resnet_features.shape == (3, 1024)
    assert aligned.temporal_features.shape == (3, 18)
    assert np.all(aligned.resnet_features[:, 0] == [0, 1, 2])
    assert np.all(aligned.temporal_features[:, 0] == [10, 11, 12])
    assert aligned.segment_start_times.tolist() == [0.0, 5.0, 10.0]


def test_missing_temporal_segment_is_rejected(tmp_path: Path) -> None:
    """A ResNet segment without a temporal match is rejected."""

    resnet_path, temporal_path, segment_manifest = _write_alignment_fixture(
        tmp_path,
        resnet_ids=[0, 1, 2],
        temporal_ids=[0, 1],
    )
    with pytest.raises(FeatureAlignmentError, match="Temporal features.*missing"):
        load_aligned_features("T001", resnet_path, temporal_path, segment_manifest)


def test_missing_resnet_segment_is_rejected(tmp_path: Path) -> None:
    """A temporal segment without a ResNet match is rejected."""

    resnet_path, temporal_path, segment_manifest = _write_alignment_fixture(
        tmp_path,
        resnet_ids=[0, 1],
        temporal_ids=[0, 1, 2],
    )
    with pytest.raises(FeatureAlignmentError, match="ResNet features.*missing"):
        load_aligned_features("T001", resnet_path, temporal_path, segment_manifest)


def test_duplicate_segment_ids_are_rejected(tmp_path: Path) -> None:
    """Duplicate segment IDs are rejected before alignment."""

    resnet_path, temporal_path, segment_manifest = _write_alignment_fixture(
        tmp_path,
        resnet_ids=[0, 1, 1],
        temporal_ids=[0, 1, 2],
    )
    with pytest.raises(FeatureAlignmentError, match="duplicate segment IDs"):
        load_aligned_features("T001", resnet_path, temporal_path, segment_manifest)


def test_mismatched_feature_dimensions_are_rejected(tmp_path: Path) -> None:
    """Incorrect feature dimensions are rejected with a clear error."""

    resnet_path, temporal_path, segment_manifest = _write_alignment_fixture(tmp_path, temporal_dim=17)
    with pytest.raises(FeatureAlignmentError, match="Temporal feature dimension"):
        load_aligned_features("T001", resnet_path, temporal_path, segment_manifest)


def test_nonfinite_feature_values_are_rejected(tmp_path: Path) -> None:
    """NaN or infinity values are rejected before normalization."""

    resnet_path, temporal_path, segment_manifest = _write_alignment_fixture(tmp_path, nonfinite=True)
    with pytest.raises(FeatureAlignmentError, match="non-finite"):
        load_aligned_features("T001", resnet_path, temporal_path, segment_manifest)


def test_mismatched_video_ids_are_rejected(tmp_path: Path) -> None:
    """Feature manifests must belong to the requested video ID."""

    resnet_path, temporal_path, segment_manifest = _write_alignment_fixture(
        tmp_path,
        resnet_manifest_video_id="OTHER",
    )
    with pytest.raises(FeatureAlignmentError, match="does not match requested"):
        load_aligned_features("T001", resnet_path, temporal_path, segment_manifest)
