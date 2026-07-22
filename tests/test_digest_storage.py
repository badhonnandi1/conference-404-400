"""Tests for quantization artifact and digest storage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.authentication.digest_storage import (
    QUANTIZATION_WARNING,
    build_and_store_digest,
    create_and_store_quantizer,
    load_digest_npz,
    load_quantization_artifact,
)
from src.features.feature_storage import sha256_file
from src.features.normalization import RobustNormalizer
from src.features.normalization_storage import (
    normalization_artifact_paths,
    save_normalization_manifest,
    save_normalization_parameters_npz,
)
from src.video.metadata import ExistingOutputError


def _write_normalization_artifact(root: Path) -> None:
    paths = normalization_artifact_paths(root / "calibration", "DEV_NORMALIZATION_V1")
    resnet_values = np.vstack(
        [
            np.zeros(1024, dtype=np.float32),
            np.ones(1024, dtype=np.float32),
            np.full(1024, 2.0, dtype=np.float32),
        ]
    )
    temporal_values = np.vstack(
        [
            np.zeros(18, dtype=np.float32),
            np.ones(18, dtype=np.float32),
            np.full(18, 2.0, dtype=np.float32),
        ]
    )
    resnet = RobustNormalizer.fit(resnet_values)
    temporal = RobustNormalizer.fit(temporal_values)
    save_normalization_parameters_npz(paths, resnet, temporal, overwrite=True)
    manifest = {
        "calibration_id": "DEV_NORMALIZATION_V1",
        "status": "development",
        "development_only": True,
        "source_video_ids": ["T001"],
        "total_calibration_segments": 3,
        "resnet_dimension": 1024,
        "temporal_dimension": 18,
        "normalization_method": "median_iqr",
        "epsilon": 1e-8,
        "clipping_range": [-5.0, 5.0],
        "npz_output_path": str(paths.npz_path),
        "npz_sha256": sha256_file(paths.npz_path),
        "warnings": [],
        "limitations": [],
    }
    save_normalization_manifest(manifest, paths, overwrite=True)


def _write_normalized_features(root: Path, video_id: str = "T001") -> None:
    output_dir = root / "features" / "normalized" / video_id
    output_dir.mkdir(parents=True)
    segment_ids = np.asarray([0, 1], dtype=np.int64)
    resnet = np.vstack([np.full(1024, -0.5, dtype=np.float32), np.full(1024, 0.5, dtype=np.float32)])
    temporal = np.vstack([np.full(18, -0.5, dtype=np.float32), np.full(18, 0.5, dtype=np.float32)])
    combined = np.concatenate([resnet, temporal], axis=1)
    npz_path = output_dir / f"{video_id}_normalized_features.npz"
    np.savez_compressed(
        npz_path,
        segment_ids=segment_ids,
        segment_start_times=np.asarray([0.0, 5.0], dtype=np.float64),
        segment_end_times=np.asarray([5.0, 10.0], dtype=np.float64),
        resnet_normalized_features=resnet,
        temporal_normalized_features=temporal,
        combined_normalized_features=combined,
    )
    calibration_npz = root / "calibration" / "DEV_NORMALIZATION_V1" / "normalization_parameters.npz"
    manifest = {
        "video_id": video_id,
        "calibration_id": "DEV_NORMALIZATION_V1",
        "development_only": True,
        "source_resnet_path": str(root / "missing_resnet.npz"),
        "source_temporal_path": str(root / "missing_temporal.npz"),
        "calibration_npz_path": str(calibration_npz),
        "calibration_npz_sha256": sha256_file(calibration_npz),
        "normalization_settings": {"clipping_range": [-5.0, 5.0]},
        "npz_sha256": sha256_file(npz_path),
    }
    (output_dir / f"{video_id}_normalized_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


def test_quantizer_digest_storage_cache_and_reload(tmp_path: Path) -> None:
    """Quantizer and digest outputs serialize, reload, and cache safely."""

    _write_normalization_artifact(tmp_path)
    _write_normalized_features(tmp_path)

    quantizer = create_and_store_quantizer(
        normalization_root=tmp_path / "calibration",
        quantization_root=tmp_path / "calibration",
        normalization_id="DEV_NORMALIZATION_V1",
        quantization_id="DEV_QUANTIZATION_V1",
        overwrite=True,
    )
    loaded_quantizer = load_quantization_artifact(tmp_path / "calibration", "DEV_QUANTIZATION_V1")
    assert loaded_quantizer.npz_sha256 == quantizer.npz_sha256
    assert QUANTIZATION_WARNING in quantizer.manifest["warnings"]

    bundle, manifest, paths, reused = build_and_store_digest(
        video_id="T001",
        normalized_root=tmp_path / "features" / "normalized",
        quantization_root=tmp_path / "calibration",
        digest_root=tmp_path / "digests",
        quantization_id="DEV_QUANTIZATION_V1",
        overwrite=True,
    )
    assert not reused
    assert bundle is not None
    assert bundle.resnet_binary_digests.shape == (2, 1024)
    assert bundle.temporal_bin_indices.shape == (2, 18)
    assert bundle.temporal_binary_digests.shape == (2, 36)
    assert bundle.hybrid_binary_digests.shape == (2, 1060)
    assert bundle.resnet_packed_digests.shape == (2, 128)
    assert bundle.temporal_packed_digests.shape == (2, 5)
    assert bundle.hybrid_packed_digests.shape == (2, 133)
    assert bundle.validate_round_trips()
    assert manifest["development_warning"] == QUANTIZATION_WARNING
    assert manifest["npz_sha256"] == sha256_file(paths.npz_path)
    assert "clipping_statistics" in manifest

    arrays = load_digest_npz(paths.npz_path)
    assert arrays["hybrid_binary_digests"].shape == (2, 1060)

    cached_bundle, cached_manifest, _, cached = build_and_store_digest(
        video_id="T001",
        normalized_root=tmp_path / "features" / "normalized",
        quantization_root=tmp_path / "calibration",
        digest_root=tmp_path / "digests",
        quantization_id="DEV_QUANTIZATION_V1",
        overwrite=False,
    )
    assert cached
    assert cached_bundle is None
    assert cached_manifest["npz_sha256"] == manifest["npz_sha256"]

    broken = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
    broken["source_normalized_feature_sha256"] = "changed"
    paths.manifest_path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ExistingOutputError, match="cache metadata"):
        build_and_store_digest(
            video_id="T001",
            normalized_root=tmp_path / "features" / "normalized",
            quantization_root=tmp_path / "calibration",
            digest_root=tmp_path / "digests",
            quantization_id="DEV_QUANTIZATION_V1",
            overwrite=False,
        )
