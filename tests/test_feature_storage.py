"""Tests for feature storage and cache helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.features.aggregation import SegmentAggregationResult, SegmentFeatureRecord
from src.features.device import DeviceInfo
from src.features.feature_storage import (
    build_feature_cache_key,
    build_feature_manifest,
    feature_manifest_matches,
    feature_output_paths,
    load_feature_manifest,
    load_feature_npz,
    save_feature_manifest,
    save_feature_npz,
    sha256_file,
)
from src.features.resnet_features import FrameExtractionResult, FrameFeatureRecord, ResNetModelBundle


def _frame_result() -> FrameExtractionResult:
    return FrameExtractionResult(
        embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        records=[
            FrameFeatureRecord(
                video_id="T001",
                segment_id=0,
                frame_index=0,
                requested_timestamp_seconds=0.5,
                actual_timestamp_seconds=0.5,
                frame_path="/tmp/a.jpg",
                embedding_row_index=0,
                embedding_dimension=2,
                original_embedding_norm=2.0,
                normalized_embedding_norm=1.0,
                extraction_success=True,
            ),
            FrameFeatureRecord(
                video_id="T001",
                segment_id=0,
                frame_index=1,
                requested_timestamp_seconds=1.5,
                actual_timestamp_seconds=1.5,
                frame_path="/tmp/b.jpg",
                embedding_row_index=1,
                embedding_dimension=2,
                original_embedding_norm=3.0,
                normalized_embedding_norm=1.0,
                extraction_success=True,
            ),
        ],
        failures=[],
        warnings=[],
    )


def _segment_result() -> SegmentAggregationResult:
    return SegmentAggregationResult(
        segment_ids=np.asarray([0], dtype=np.int64),
        mean_embeddings=np.asarray([[0.5, 0.5]], dtype=np.float32),
        std_embeddings=np.asarray([[0.5, 0.5]], dtype=np.float32),
        combined_embeddings=np.asarray([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32),
        records=[
            SegmentFeatureRecord(
                segment_id=0,
                start_time_seconds=0.0,
                end_time_seconds=5.0,
                expected_sampled_frames=2,
                successfully_used_frames=2,
                failed_or_missing_frames=0,
                mean_embedding_row_index=0,
                standard_deviation_embedding_row_index=0,
                combined_embedding_row_index=0,
                mean_embedding_dimension=2,
                standard_deviation_embedding_dimension=2,
                combined_embedding_dimension=4,
            )
        ],
        warnings=[],
        failures=[],
    )


def test_sha256_npz_and_manifest_serialization(tmp_path: Path) -> None:
    """Feature arrays and manifests are serializable and reloadable."""

    source_manifest = tmp_path / "T001_frames.json"
    source_manifest.write_text('{"video_id": "T001"}', encoding="utf-8")
    outputs = feature_output_paths(tmp_path / "features", "T001")
    frame_result = _frame_result()
    segment_result = _segment_result()

    save_feature_npz(outputs, frame_result, segment_result, overwrite=True)
    arrays = load_feature_npz(outputs.npz_path)

    assert arrays["frame_embeddings"].shape == (2, 2)
    assert arrays["segment_combined_embeddings"].shape == (1, 4)
    assert sha256_file(outputs.npz_path)

    bundle = ResNetModelBundle(
        model=None,  # type: ignore[arg-type]
        transform=lambda image: image,  # type: ignore[arg-type]
        architecture="resnet18",
        weight_identifier="DEFAULT",
        preprocessing_description="test preprocessing",
        embedding_dimension=2,
        model_loading_time_seconds=0.1,
    )
    device_info = DeviceInfo(
        requested_device="cpu",
        selected_device="cpu",
        mps_built=True,
        mps_available=False,
        architecture="arm64",
    )
    manifest = build_feature_manifest(
        video_id="T001",
        source_frame_manifest_path=source_manifest,
        source_frame_manifest_sha256=sha256_file(source_manifest),
        npz_sha256=sha256_file(outputs.npz_path),
        paths=outputs,
        bundle=bundle,
        device_info=device_info,
        batch_size=2,
        normalize_frame_embeddings=True,
        frame_result=frame_result,
        segment_result=segment_result,
        total_processing_time_seconds=1.0,
        source_frame_failures=[],
    )
    save_feature_manifest(manifest, outputs, overwrite=True)

    loaded = load_feature_manifest(outputs.manifest_path)
    assert loaded["video_id"] == "T001"
    assert loaded["npz_sha256"] == sha256_file(outputs.npz_path)
    assert loaded["frame_extraction_records"][0]["embedding_row_index"] == 0


def test_cache_key_matching() -> None:
    """Cache matching uses the required reproducibility fields."""

    cache_key = build_feature_cache_key(
        source_frame_manifest_sha256="abc",
        architecture="resnet18",
        weight_identifier="DEFAULT",
        preprocessing_description="preprocess",
        normalize_frame_embeddings=True,
        embedding_dimension=512,
    )
    assert feature_manifest_matches(dict(cache_key), cache_key)

    changed = dict(cache_key)
    changed["embedding_dimension"] = 256
    assert not feature_manifest_matches(changed, cache_key)
