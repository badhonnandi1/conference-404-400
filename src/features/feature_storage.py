"""Storage and cache helpers for extracted feature arrays."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.features.aggregation import SegmentAggregationResult
from src.features.device import DeviceInfo
from src.features.resnet_features import FrameExtractionResult, ResNetModelBundle
from src.video.metadata import ExistingOutputError


@dataclass(frozen=True)
class FeatureOutputPaths:
    """Output paths for one video's ResNet feature artifacts."""

    output_dir: Path
    npz_path: Path
    manifest_path: Path


def sha256_file(path: str | Path) -> str:
    """Compute a SHA-256 checksum for a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_output_paths(output_root: str | Path, video_id: str) -> FeatureOutputPaths:
    """Return deterministic output paths for a video's ResNet features."""

    output_dir = Path(output_root) / video_id
    return FeatureOutputPaths(
        output_dir=output_dir,
        npz_path=output_dir / f"{video_id}_resnet_features.npz",
        manifest_path=output_dir / f"{video_id}_resnet_manifest.json",
    )


def build_feature_cache_key(
    source_frame_manifest_sha256: str,
    architecture: str,
    weight_identifier: str,
    preprocessing_description: str,
    normalize_frame_embeddings: bool,
    embedding_dimension: int,
) -> dict[str, Any]:
    """Build the fields used to decide whether a feature cache is reusable."""

    return {
        "source_frame_manifest_sha256": source_frame_manifest_sha256,
        "model_architecture": architecture,
        "weight_identifier": weight_identifier,
        "preprocessing_description": preprocessing_description,
        "normalize_frame_embeddings": normalize_frame_embeddings,
        "embedding_dimension": embedding_dimension,
    }


def feature_manifest_matches(manifest: dict[str, Any], cache_key: dict[str, Any]) -> bool:
    """Return whether a stored feature manifest matches the requested cache key."""

    return all(manifest.get(key) == value for key, value in cache_key.items())


def load_feature_manifest(path: str | Path) -> dict[str, Any]:
    """Load a feature manifest from JSON."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_can_write_features(
    paths: FeatureOutputPaths,
    overwrite: bool,
    cache_key: dict[str, Any],
) -> bool:
    """Return True when an existing feature result can be reused."""

    if not paths.npz_path.exists() and not paths.manifest_path.exists():
        return False
    if overwrite:
        return False
    if paths.npz_path.exists() and paths.manifest_path.exists():
        manifest = load_feature_manifest(paths.manifest_path)
        if feature_manifest_matches(manifest, cache_key):
            return True
    raise ExistingOutputError(
        f"Feature outputs already exist under {paths.output_dir}. "
        "Use --overwrite to regenerate because the cache metadata did not match."
    )


def save_feature_npz(
    paths: FeatureOutputPaths,
    frame_result: FrameExtractionResult,
    segment_result: SegmentAggregationResult,
    overwrite: bool = False,
) -> Path:
    """Save frame and segment feature arrays in compressed NumPy format."""

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    if paths.npz_path.exists() and not overwrite:
        raise ExistingOutputError(
            f"Feature NPZ already exists: {paths.npz_path}. Use --overwrite to replace it."
        )

    success_records = [record for record in frame_result.records if record.extraction_success]
    success_records.sort(
        key=lambda record: (
            record.segment_id,
            record.requested_timestamp_seconds,
            record.frame_index,
        )
    )
    np.savez_compressed(
        paths.npz_path,
        frame_embeddings=frame_result.embeddings.astype(np.float32),
        frame_segment_ids=np.asarray([record.segment_id for record in success_records], dtype=np.int64),
        frame_indices=np.asarray([record.frame_index for record in success_records], dtype=np.int64),
        frame_requested_timestamps=np.asarray(
            [record.requested_timestamp_seconds for record in success_records],
            dtype=np.float64,
        ),
        frame_actual_timestamps=np.asarray(
            [
                np.nan if record.actual_timestamp_seconds is None else record.actual_timestamp_seconds
                for record in success_records
            ],
            dtype=np.float64,
        ),
        segment_ids=segment_result.segment_ids.astype(np.int64),
        segment_mean_embeddings=segment_result.mean_embeddings.astype(np.float32),
        segment_std_embeddings=segment_result.std_embeddings.astype(np.float32),
        segment_combined_embeddings=segment_result.combined_embeddings.astype(np.float32),
    )
    return paths.npz_path


def build_feature_manifest(
    video_id: str,
    source_frame_manifest_path: str | Path,
    source_frame_manifest_sha256: str,
    npz_sha256: str,
    paths: FeatureOutputPaths,
    bundle: ResNetModelBundle,
    device_info: DeviceInfo,
    batch_size: int,
    normalize_frame_embeddings: bool,
    frame_result: FrameExtractionResult,
    segment_result: SegmentAggregationResult,
    total_processing_time_seconds: float,
    source_frame_failures: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the JSON-serializable feature manifest."""

    frame_successes = sum(1 for record in frame_result.records if record.extraction_success)
    average_time = (
        total_processing_time_seconds / frame_successes if frame_successes else None
    )
    warnings = list(frame_result.warnings) + list(segment_result.warnings)
    failures = list(frame_result.failures) + list(segment_result.failures)
    if source_frame_failures:
        warnings.append(
            f"Source frame manifest contains {len(source_frame_failures)} failed Phase 1 frame records."
        )

    return {
        "video_id": video_id,
        "source_frame_manifest_path": str(Path(source_frame_manifest_path).resolve()),
        "source_frame_manifest_sha256": source_frame_manifest_sha256,
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        "torch_version": __import__("torch").__version__,
        "torchvision_version": __import__("torchvision").__version__,
        "model_architecture": bundle.architecture,
        "weight_identifier": bundle.weight_identifier,
        "preprocessing_description": bundle.preprocessing_description,
        "selected_device": device_info.selected_device,
        "device_info": device_info.to_dict(),
        "batch_size": batch_size,
        "feature_dimensions": {
            "frame_embedding": bundle.embedding_dimension,
            "segment_mean": bundle.embedding_dimension,
            "segment_standard_deviation": bundle.embedding_dimension,
            "segment_combined": bundle.embedding_dimension * 2,
        },
        "normalize_frame_embeddings": normalize_frame_embeddings,
        "normalization_epsilon": 1e-12,
        "frame_extraction_records": [record.to_dict() for record in frame_result.records],
        "segment_aggregation_records": [record.to_dict() for record in segment_result.records],
        "output_npz_path": str(paths.npz_path.resolve()),
        "npz_sha256": npz_sha256,
        "model_loading_time_seconds": bundle.model_loading_time_seconds,
        "total_processing_time_seconds": total_processing_time_seconds,
        "average_processing_time_per_frame_seconds": average_time,
        "warnings": warnings,
        "failures": failures,
    }


def save_feature_manifest(
    manifest: dict[str, Any],
    paths: FeatureOutputPaths,
    overwrite: bool = False,
) -> Path:
    """Save a feature manifest as formatted JSON."""

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    if paths.manifest_path.exists() and not overwrite:
        raise ExistingOutputError(
            f"Feature manifest already exists: {paths.manifest_path}. Use --overwrite to replace it."
        )
    with paths.manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return paths.manifest_path


def load_feature_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load feature arrays from a compressed NPZ file."""

    with np.load(Path(path)) as payload:
        return {name: payload[name] for name in payload.files}
