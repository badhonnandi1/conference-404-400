"""Storage and manifest helpers for temporal consistency features."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from src.features.feature_storage import sha256_file
from src.features.temporal_features import (
    PAIR_FEATURE_NAMES,
    SEGMENT_AGGREGATIONS,
    SEGMENT_FEATURE_NAMES,
    TemporalFeatureResult,
)
from src.features.temporal_sampling import TemporalSamplingConfig, decode_temporal_frames
from src.video.metadata import ExistingOutputError
from src.video.segmentation import SegmentManifest, load_segment_manifest


@dataclass(frozen=True)
class TemporalOutputPaths:
    """Output paths for one video's temporal feature artifacts."""

    output_dir: Path
    npz_path: Path
    manifest_path: Path


def temporal_output_paths(output_root: str | Path, video_id: str) -> TemporalOutputPaths:
    """Return deterministic output paths for a video's temporal features."""

    output_dir = Path(output_root) / video_id
    return TemporalOutputPaths(
        output_dir=output_dir,
        npz_path=output_dir / f"{video_id}_temporal_features.npz",
        manifest_path=output_dir / f"{video_id}_temporal_manifest.json",
    )


def build_temporal_cache_key(
    source_video_sha256: str,
    segment_manifest_sha256: str,
    config: TemporalSamplingConfig,
) -> dict[str, Any]:
    """Build cache metadata for temporal feature reuse."""

    return {
        "source_video_sha256": source_video_sha256,
        "segment_manifest_sha256": segment_manifest_sha256,
        "temporal_sample_fps": config.sample_fps,
        "frame_width": config.frame_width,
        "frame_height": config.frame_height,
        "grayscale": config.grayscale,
        "gaussian_blur_kernel": config.gaussian_blur_kernel,
        "changed_pixel_threshold": config.changed_pixel_threshold,
        "pair_feature_names": PAIR_FEATURE_NAMES,
        "segment_aggregation": SEGMENT_AGGREGATIONS,
        "segment_feature_names": SEGMENT_FEATURE_NAMES,
    }


def temporal_manifest_matches(manifest: dict[str, Any], cache_key: dict[str, Any]) -> bool:
    """Return whether a temporal manifest matches a requested cache key."""

    return all(manifest.get(key) == value for key, value in cache_key.items())


def ensure_can_write_temporal(
    paths: TemporalOutputPaths,
    overwrite: bool,
    cache_key: dict[str, Any],
) -> bool:
    """Return True when existing temporal outputs can be reused."""

    if not paths.npz_path.exists() and not paths.manifest_path.exists():
        return False
    if overwrite:
        return False
    if paths.npz_path.exists() and paths.manifest_path.exists():
        with paths.manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if temporal_manifest_matches(manifest, cache_key):
            return True
    raise ExistingOutputError(
        f"Temporal outputs already exist under {paths.output_dir}. "
        "Use --overwrite to regenerate because cache metadata did not match."
    )


def save_temporal_npz(
    paths: TemporalOutputPaths,
    result: TemporalFeatureResult,
    overwrite: bool = False,
) -> Path:
    """Save temporal pair and segment arrays in compressed NumPy format."""

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    if paths.npz_path.exists() and not overwrite:
        raise ExistingOutputError(
            f"Temporal feature NPZ already exists: {paths.npz_path}. Use --overwrite to replace it."
        )
    np.savez_compressed(
        paths.npz_path,
        pair_features=result.pair_features.astype(np.float32),
        pair_segment_ids=result.pair_segment_ids.astype(np.int64),
        pair_indices=result.pair_indices.astype(np.int64),
        pair_start_timestamps=result.pair_start_timestamps.astype(np.float64),
        pair_end_timestamps=result.pair_end_timestamps.astype(np.float64),
        segment_ids=result.segment_ids.astype(np.int64),
        segment_features=result.segment_features.astype(np.float32),
        segment_successful_pair_counts=result.segment_successful_pair_counts.astype(np.int64),
        segment_max_discontinuity_pair_indices=result.segment_max_discontinuity_pair_indices.astype(np.int64),
        segment_max_discontinuity_timestamps=result.segment_max_discontinuity_timestamps.astype(np.float64),
        feature_names=np.asarray(SEGMENT_FEATURE_NAMES),
    )
    return paths.npz_path


def build_temporal_manifest(
    video_id: str,
    source_video_path: str | Path,
    source_video_sha256: str,
    segment_manifest_path: str | Path,
    segment_manifest_sha256: str,
    config: TemporalSamplingConfig,
    result: TemporalFeatureResult,
    paths: TemporalOutputPaths,
    npz_sha256: str,
    total_processing_time_seconds: float,
) -> dict[str, Any]:
    """Build a human-readable temporal feature manifest."""

    segment_count = len(result.segment_records)
    return {
        "video_id": video_id,
        "source_video_path": str(Path(source_video_path).resolve()),
        "source_video_sha256": source_video_sha256,
        "segment_manifest_path": str(Path(segment_manifest_path).resolve()),
        "segment_manifest_sha256": segment_manifest_sha256,
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        "temporal_sample_fps": config.sample_fps,
        "frame_preprocessing": config.to_dict(),
        "pair_feature_definitions": {
            "mean_absolute_difference": "mean(abs(frame_b - frame_a))",
            "absdiff_standard_deviation": "population standard deviation of absolute differences",
            "normalized_rmse": "sqrt(mean((frame_b - frame_a)^2))",
            "changed_pixel_ratio": "ratio of pixels where absdiff exceeds threshold / 255",
            "p90_absolute_difference": "90th percentile of absolute differences",
            "edge_change_ratio": "ratio of pixels with meaningful Sobel edge-magnitude change",
        },
        "pair_feature_names": PAIR_FEATURE_NAMES,
        "segment_aggregation": SEGMENT_AGGREGATIONS,
        "segment_feature_names": SEGMENT_FEATURE_NAMES,
        "segment_records": [record.to_dict() for record in result.segment_records],
        "pair_records": [record.to_dict() for record in result.pair_records],
        "output_npz_path": str(paths.npz_path.resolve()),
        "npz_sha256": npz_sha256,
        "total_processing_time_seconds": total_processing_time_seconds,
        "average_processing_time_per_segment_seconds": (
            total_processing_time_seconds / segment_count if segment_count else None
        ),
        "warnings": result.warnings,
        "failures": result.failures,
    }


def save_temporal_manifest(
    manifest: dict[str, Any],
    paths: TemporalOutputPaths,
    overwrite: bool = False,
) -> Path:
    """Save a temporal feature manifest as formatted JSON."""

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    if paths.manifest_path.exists() and not overwrite:
        raise ExistingOutputError(
            f"Temporal feature manifest already exists: {paths.manifest_path}. Use --overwrite to replace it."
        )
    with paths.manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return paths.manifest_path


def load_temporal_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load temporal feature arrays from an NPZ file."""

    with np.load(Path(path)) as payload:
        return {name: payload[name] for name in payload.files}


def extract_and_store_temporal_features(
    video_id: str,
    source_video_path: str | Path,
    segment_manifest_path: str | Path,
    output_root: str | Path,
    config: TemporalSamplingConfig,
    overwrite: bool = False,
) -> tuple[TemporalFeatureResult | None, dict[str, Any] | None, TemporalOutputPaths, bool]:
    """Extract temporal features and store NPZ/JSON outputs.

    Returns the result, manifest, output paths, and whether a cache was reused.
    """

    source_video_sha256 = sha256_file(source_video_path)
    segment_manifest_sha256 = sha256_file(segment_manifest_path)
    cache_key = build_temporal_cache_key(source_video_sha256, segment_manifest_sha256, config)
    paths = temporal_output_paths(output_root, video_id)
    if ensure_can_write_temporal(paths, overwrite, cache_key):
        with paths.manifest_path.open("r", encoding="utf-8") as handle:
            return None, json.load(handle), paths, True

    started = perf_counter()
    segment_manifest = load_segment_manifest(segment_manifest_path)
    frames = decode_temporal_frames(source_video_path, segment_manifest, config)
    from src.features.temporal_features import extract_temporal_features_from_frames

    result = extract_temporal_features_from_frames(video_id, segment_manifest, frames, config)
    save_temporal_npz(paths, result, overwrite=overwrite)
    npz_checksum = sha256_file(paths.npz_path)
    total_time = perf_counter() - started
    manifest = build_temporal_manifest(
        video_id=video_id,
        source_video_path=source_video_path,
        source_video_sha256=source_video_sha256,
        segment_manifest_path=segment_manifest_path,
        segment_manifest_sha256=segment_manifest_sha256,
        config=config,
        result=result,
        paths=paths,
        npz_sha256=npz_checksum,
        total_processing_time_seconds=total_time,
    )
    manifest.update(cache_key)
    save_temporal_manifest(manifest, paths, overwrite=overwrite)
    return result, manifest, paths, False
