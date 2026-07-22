"""Tests for normalization artifact and normalized feature storage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.features.normalization_storage import (
    DEVELOPMENT_NORMALIZATION_WARNING,
    fit_and_store_normalization_artifact,
    load_normalization_artifact,
    load_normalized_npz,
    normalize_and_store_features,
)
from src.video.metadata import ExistingOutputError


def _write_video_features(root: Path, video_id: str, offset: float) -> None:
    resnet_dir = root / "features" / "resnet" / video_id
    temporal_dir = root / "features" / "temporal" / video_id
    manifests_dir = root / "manifests"
    resnet_dir.mkdir(parents=True)
    temporal_dir.mkdir(parents=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    segment_ids = np.asarray([1, 0], dtype=np.int64)
    resnet = np.vstack(
        [
            np.full(1024, offset + 1, dtype=np.float32),
            np.full(1024, offset + 0, dtype=np.float32),
        ]
    )
    temporal = np.vstack(
        [
            np.full(18, offset + 2, dtype=np.float32),
            np.full(18, offset + 1, dtype=np.float32),
        ]
    )
    np.savez_compressed(
        resnet_dir / f"{video_id}_resnet_features.npz",
        segment_ids=segment_ids,
        segment_combined_embeddings=resnet,
    )
    np.savez_compressed(
        temporal_dir / f"{video_id}_temporal_features.npz",
        segment_ids=segment_ids,
        segment_features=temporal,
    )
    (resnet_dir / f"{video_id}_resnet_manifest.json").write_text(
        json.dumps({"video_id": video_id, "warnings": []}),
        encoding="utf-8",
    )
    (temporal_dir / f"{video_id}_temporal_manifest.json").write_text(
        json.dumps(
            {
                "video_id": video_id,
                "segment_records": [
                    {
                        "segment_id": int(segment_id),
                        "start_time_seconds": float(segment_id * 5),
                        "end_time_seconds": float(segment_id * 5 + 5),
                        "success": True,
                    }
                    for segment_id in segment_ids
                ],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    (manifests_dir / f"{video_id}_segments.json").write_text(
        json.dumps(
            {
                "video_id": video_id,
                "source_video_path": "/tmp/video.mp4",
                "video_metadata_reference": None,
                "segment_duration_seconds": 5,
                "sample_frames_per_second": 1,
                "incomplete_segment_policy": "discard",
                "number_complete_segments": 2,
                "processed_duration_seconds": 10,
                "discarded_duration_seconds": 0,
                "segments": [
                    {
                        "video_id": video_id,
                        "segment_id": 0,
                        "start_time_seconds": 0.0,
                        "end_time_seconds": 5.0,
                        "duration_seconds": 5,
                        "is_complete": True,
                        "expected_sample_count": 5,
                        "source_video_path": "/tmp/video.mp4",
                    },
                    {
                        "video_id": video_id,
                        "segment_id": 1,
                        "start_time_seconds": 5.0,
                        "end_time_seconds": 10.0,
                        "duration_seconds": 5,
                        "is_complete": True,
                        "expected_sample_count": 5,
                        "source_video_path": "/tmp/video.mp4",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_calibration_artifact_and_normalized_storage(tmp_path: Path) -> None:
    """Calibration and normalized outputs are stored, loaded, and cached safely."""

    _write_video_features(tmp_path, "V001", 0.0)
    _write_video_features(tmp_path, "V002", 2.0)
    artifact, aligned_sets = fit_and_store_normalization_artifact(
        video_ids=["V001", "V002"],
        resnet_root=tmp_path / "features" / "resnet",
        temporal_root=tmp_path / "features" / "temporal",
        manifests_root=tmp_path / "manifests",
        calibration_root=tmp_path / "calibration",
        overwrite=True,
    )

    assert artifact.paths.npz_path.exists()
    assert artifact.paths.manifest_path.exists()
    assert artifact.manifest["total_calibration_segments"] == 4
    assert DEVELOPMENT_NORMALIZATION_WARNING in artifact.manifest["warnings"]
    assert len(aligned_sets) == 2
    loaded = load_normalization_artifact(tmp_path / "calibration", artifact.calibration_id)
    assert loaded.resnet_normalizer.feature_dimension == 1024
    assert loaded.temporal_normalizer.feature_dimension == 18

    bundle, manifest, paths, reused = normalize_and_store_features(
        video_id="V001",
        resnet_root=tmp_path / "features" / "resnet",
        temporal_root=tmp_path / "features" / "temporal",
        manifests_root=tmp_path / "manifests",
        calibration_root=tmp_path / "calibration",
        normalized_root=tmp_path / "features" / "normalized",
        overwrite=True,
    )
    assert not reused
    assert bundle is not None
    assert paths.npz_path.exists()
    arrays = load_normalized_npz(paths.npz_path)
    assert arrays["resnet_raw_features"].shape == (2, 1024)
    assert arrays["temporal_raw_features"].shape == (2, 18)
    assert arrays["combined_normalized_features"].shape == (2, 1042)
    assert manifest["npz_sha256"]
    assert manifest["source_resnet_sha256"]
    assert manifest["source_temporal_sha256"]
    assert manifest["development_warning"] == DEVELOPMENT_NORMALIZATION_WARNING

    cached_bundle, cached_manifest, _, cached = normalize_and_store_features(
        video_id="V001",
        resnet_root=tmp_path / "features" / "resnet",
        temporal_root=tmp_path / "features" / "temporal",
        manifests_root=tmp_path / "manifests",
        calibration_root=tmp_path / "calibration",
        normalized_root=tmp_path / "features" / "normalized",
        overwrite=False,
    )
    assert cached
    assert cached_bundle is None
    assert cached_manifest["npz_sha256"] == manifest["npz_sha256"]

    broken_manifest = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
    broken_manifest["source_resnet_sha256"] = "changed"
    paths.manifest_path.write_text(json.dumps(broken_manifest), encoding="utf-8")
    with pytest.raises(ExistingOutputError, match="cache metadata"):
        normalize_and_store_features(
            video_id="V001",
            resnet_root=tmp_path / "features" / "resnet",
            temporal_root=tmp_path / "features" / "temporal",
            manifests_root=tmp_path / "manifests",
            calibration_root=tmp_path / "calibration",
            normalized_root=tmp_path / "features" / "normalized",
            overwrite=False,
        )
