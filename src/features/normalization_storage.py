"""Storage helpers for calibration and normalized feature artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from time import perf_counter
from typing import Any

import numpy as np

from src.features.alignment import (
    RESNET_SEGMENT_DIMENSION,
    TEMPORAL_SEGMENT_DIMENSION,
    AlignedFeatureSet,
    default_resnet_feature_path,
    default_temporal_feature_path,
    load_aligned_features,
)
from src.features.feature_storage import sha256_file
from src.features.fusion import (
    COMBINED_FEATURE_DIMENSION,
    NormalizedFeatureBundle,
    combine_normalized_streams,
    stream_boundaries_for_manifest,
)
from src.features.normalization import (
    DEFAULT_CLIP_MAX,
    DEFAULT_CLIP_MIN,
    DEFAULT_EPSILON,
    NORMALIZATION_METHOD,
    RobustNormalizer,
)
from src.video.metadata import ExistingOutputError


DEFAULT_CALIBRATION_ID = "DEV_NORMALIZATION_V1"
DEVELOPMENT_NORMALIZATION_WARNING = (
    "This normalization artifact was fitted using only three original development "
    "videos. It is intended for pipeline validation and must be replaced before final "
    "compression-resilience and tamper-detection experiments."
)


@dataclass(frozen=True)
class NormalizationArtifactPaths:
    """Output paths for a calibration artifact."""

    output_dir: Path
    npz_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class LoadedNormalizationArtifact:
    """Loaded stream-specific normalizers and calibration metadata."""

    calibration_id: str
    resnet_normalizer: RobustNormalizer
    temporal_normalizer: RobustNormalizer
    manifest: dict[str, Any]
    paths: NormalizationArtifactPaths
    npz_sha256: str


@dataclass(frozen=True)
class NormalizedOutputPaths:
    """Output paths for normalized features for one video."""

    output_dir: Path
    npz_path: Path
    manifest_path: Path


def normalization_artifact_paths(calibration_root: str | Path, calibration_id: str) -> NormalizationArtifactPaths:
    """Return deterministic output paths for a calibration artifact."""

    output_dir = Path(calibration_root) / calibration_id
    return NormalizationArtifactPaths(
        output_dir=output_dir,
        npz_path=output_dir / "normalization_parameters.npz",
        manifest_path=output_dir / "normalization_manifest.json",
    )


def normalized_output_paths(output_root: str | Path, video_id: str) -> NormalizedOutputPaths:
    """Return deterministic output paths for one video's normalized features."""

    output_dir = Path(output_root) / video_id
    return NormalizedOutputPaths(
        output_dir=output_dir,
        npz_path=output_dir / f"{video_id}_normalized_features.npz",
        manifest_path=output_dir / f"{video_id}_normalized_manifest.json",
    )


def save_normalization_parameters_npz(
    paths: NormalizationArtifactPaths,
    resnet_normalizer: RobustNormalizer,
    temporal_normalizer: RobustNormalizer,
    overwrite: bool = False,
) -> Path:
    """Save stream-specific normalization parameters in one NPZ file."""

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    if paths.npz_path.exists() and not overwrite:
        raise ExistingOutputError(
            f"Normalization parameters already exist: {paths.npz_path}. Use --overwrite to replace them."
        )
    arrays: dict[str, np.ndarray] = {}
    arrays.update(resnet_normalizer.to_arrays(prefix="resnet_"))
    arrays.update(temporal_normalizer.to_arrays(prefix="temporal_"))
    np.savez_compressed(paths.npz_path, **arrays)
    return paths.npz_path


def save_normalization_manifest(
    manifest: dict[str, Any],
    paths: NormalizationArtifactPaths,
    overwrite: bool = False,
) -> Path:
    """Save a calibration manifest as formatted JSON."""

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    if paths.manifest_path.exists() and not overwrite:
        raise ExistingOutputError(
            f"Normalization manifest already exists: {paths.manifest_path}. Use --overwrite to replace it."
        )
    with paths.manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return paths.manifest_path


def load_normalization_artifact(
    calibration_root: str | Path,
    calibration_id: str,
) -> LoadedNormalizationArtifact:
    """Load a fitted calibration artifact and its manifest."""

    paths = normalization_artifact_paths(calibration_root, calibration_id)
    if not paths.npz_path.exists() or not paths.manifest_path.exists():
        raise FileNotFoundError(f"Normalization artifact not found for calibration ID {calibration_id}.")
    with np.load(paths.npz_path, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    with paths.manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("calibration_id") != calibration_id:
        raise ValueError(
            f"Normalization manifest calibration_id '{manifest.get('calibration_id')}' "
            f"does not match requested '{calibration_id}'."
        )
    return LoadedNormalizationArtifact(
        calibration_id=calibration_id,
        resnet_normalizer=RobustNormalizer.from_arrays(arrays, prefix="resnet_"),
        temporal_normalizer=RobustNormalizer.from_arrays(arrays, prefix="temporal_"),
        manifest=manifest,
        paths=paths,
        npz_sha256=sha256_file(paths.npz_path),
    )


def _default_segment_manifest_path(manifests_root: str | Path, video_id: str) -> Path:
    return Path(manifests_root) / f"{video_id}_segments.json"


def load_aligned_features_from_roots(
    video_id: str,
    resnet_root: str | Path,
    temporal_root: str | Path,
    manifests_root: str | Path,
) -> AlignedFeatureSet:
    """Load aligned streams using the repository's default feature locations."""

    return load_aligned_features(
        video_id=video_id,
        resnet_feature_path=default_resnet_feature_path(resnet_root, video_id),
        temporal_feature_path=default_temporal_feature_path(temporal_root, video_id),
        segment_manifest_path=_default_segment_manifest_path(manifests_root, video_id),
    )


def fit_and_store_normalization_artifact(
    video_ids: list[str],
    resnet_root: str | Path,
    temporal_root: str | Path,
    manifests_root: str | Path,
    calibration_root: str | Path,
    calibration_id: str = DEFAULT_CALIBRATION_ID,
    status: str = "development",
    overwrite: bool = False,
    epsilon: float = DEFAULT_EPSILON,
    clip_min: float = DEFAULT_CLIP_MIN,
    clip_max: float = DEFAULT_CLIP_MAX,
) -> tuple[LoadedNormalizationArtifact, list[AlignedFeatureSet]]:
    """Fit and save a development stream-normalization artifact."""

    if not video_ids:
        raise ValueError("At least one video ID is required to fit normalization.")
    paths = normalization_artifact_paths(calibration_root, calibration_id)
    if (paths.npz_path.exists() or paths.manifest_path.exists()) and not overwrite:
        raise ExistingOutputError(
            f"Normalization artifact already exists under {paths.output_dir}. Use --overwrite to refit it."
        )

    aligned_sets = [
        load_aligned_features_from_roots(video_id, resnet_root, temporal_root, manifests_root)
        for video_id in video_ids
    ]
    resnet_matrix = np.vstack([aligned.resnet_features for aligned in aligned_sets])
    temporal_matrix = np.vstack([aligned.temporal_features for aligned in aligned_sets])
    resnet_normalizer = RobustNormalizer.fit(
        resnet_matrix,
        epsilon=epsilon,
        clip_min=clip_min,
        clip_max=clip_max,
    )
    temporal_normalizer = RobustNormalizer.fit(
        temporal_matrix,
        epsilon=epsilon,
        clip_min=clip_min,
        clip_max=clip_max,
    )
    save_normalization_parameters_npz(paths, resnet_normalizer, temporal_normalizer, overwrite=overwrite)
    npz_checksum = sha256_file(paths.npz_path)

    source_paths = []
    source_checksums = []
    source_segment_ids = []
    for aligned in aligned_sets:
        source_paths.append(
            {
                "video_id": aligned.video_id,
                "resnet_path": str(aligned.resnet_source_path),
                "temporal_path": str(aligned.temporal_source_path),
            }
        )
        source_checksums.append(
            {
                "video_id": aligned.video_id,
                "resnet_sha256": aligned.resnet_source_sha256,
                "temporal_sha256": aligned.temporal_source_sha256,
            }
        )
        source_segment_ids.append(
            {
                "video_id": aligned.video_id,
                "segment_ids": aligned.segment_ids.astype(int).tolist(),
            }
        )

    manifest = {
        "calibration_id": calibration_id,
        "status": status,
        "development_only": True,
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        "source_video_ids": list(video_ids),
        "source_segment_ids": source_segment_ids,
        "total_calibration_segments": int(resnet_matrix.shape[0]),
        "resnet_dimension": RESNET_SEGMENT_DIMENSION,
        "temporal_dimension": TEMPORAL_SEGMENT_DIMENSION,
        "normalization_method": NORMALIZATION_METHOD,
        "epsilon": float(epsilon),
        "clipping_range": [float(clip_min), float(clip_max)],
        "source_feature_paths": source_paths,
        "source_feature_checksums": source_checksums,
        "npz_output_path": str(paths.npz_path.resolve()),
        "npz_sha256": npz_checksum,
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "resnet_normalizer": resnet_normalizer.summary(),
        "temporal_normalizer": temporal_normalizer.summary(),
        "warnings": [DEVELOPMENT_NORMALIZATION_WARNING]
        + [warning for aligned in aligned_sets for warning in aligned.warnings],
        "limitations": [
            "Development calibration uses only original V001-V003 videos.",
            "This artifact is not final or optimal.",
            "Quantization, HMAC, thresholding, and verification are not implemented in Phase 4.",
        ],
    }
    save_normalization_manifest(manifest, paths, overwrite=overwrite)
    return load_normalization_artifact(calibration_root, calibration_id), aligned_sets


def build_normalized_cache_key(
    aligned: AlignedFeatureSet,
    artifact: LoadedNormalizationArtifact,
) -> dict[str, Any]:
    """Build cache fields for normalized feature reuse."""

    return {
        "source_resnet_sha256": aligned.resnet_source_sha256,
        "source_temporal_sha256": aligned.temporal_source_sha256,
        "calibration_npz_sha256": artifact.npz_sha256,
        "normalization_method": NORMALIZATION_METHOD,
        "epsilon": artifact.resnet_normalizer.epsilon,
        "clipping_range": [
            artifact.resnet_normalizer.clip_min,
            artifact.resnet_normalizer.clip_max,
        ],
        "resnet_dimension": RESNET_SEGMENT_DIMENSION,
        "temporal_dimension": TEMPORAL_SEGMENT_DIMENSION,
        "combined_dimension": COMBINED_FEATURE_DIMENSION,
    }


def normalized_manifest_matches(manifest: dict[str, Any], cache_key: dict[str, Any]) -> bool:
    """Return whether a normalized manifest matches a requested cache key."""

    return all(manifest.get(key) == value for key, value in cache_key.items())


def ensure_can_write_normalized(
    paths: NormalizedOutputPaths,
    overwrite: bool,
    cache_key: dict[str, Any],
) -> bool:
    """Return True when an existing normalized output can be reused."""

    if not paths.npz_path.exists() and not paths.manifest_path.exists():
        return False
    if overwrite:
        return False
    if paths.npz_path.exists() and paths.manifest_path.exists():
        with paths.manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if normalized_manifest_matches(manifest, cache_key):
            return True
    raise ExistingOutputError(
        f"Normalized feature outputs already exist under {paths.output_dir}. "
        "Use --overwrite to regenerate because cache metadata did not match."
    )


def save_normalized_npz(
    paths: NormalizedOutputPaths,
    bundle: NormalizedFeatureBundle,
    overwrite: bool = False,
) -> Path:
    """Save normalized feature arrays in compressed NumPy format."""

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    if paths.npz_path.exists() and not overwrite:
        raise ExistingOutputError(
            f"Normalized feature NPZ already exists: {paths.npz_path}. Use --overwrite to replace it."
        )
    np.savez_compressed(
        paths.npz_path,
        segment_ids=bundle.segment_ids.astype(np.int64),
        segment_start_times=bundle.segment_start_times.astype(np.float64),
        segment_end_times=bundle.segment_end_times.astype(np.float64),
        resnet_raw_features=bundle.resnet_raw_features.astype(np.float32),
        temporal_raw_features=bundle.temporal_raw_features.astype(np.float32),
        resnet_normalized_features=bundle.resnet_normalized_features.astype(np.float32),
        temporal_normalized_features=bundle.temporal_normalized_features.astype(np.float32),
        combined_normalized_features=bundle.combined_normalized_features.astype(np.float32),
    )
    return paths.npz_path


def save_normalized_manifest(
    manifest: dict[str, Any],
    paths: NormalizedOutputPaths,
    overwrite: bool = False,
) -> Path:
    """Save normalized feature metadata as formatted JSON."""

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    if paths.manifest_path.exists() and not overwrite:
        raise ExistingOutputError(
            f"Normalized feature manifest already exists: {paths.manifest_path}. Use --overwrite to replace it."
        )
    with paths.manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return paths.manifest_path


def build_normalized_manifest(
    video_id: str,
    artifact: LoadedNormalizationArtifact,
    aligned: AlignedFeatureSet,
    bundle: NormalizedFeatureBundle,
    paths: NormalizedOutputPaths,
    npz_sha256: str,
    processing_time_seconds: float,
) -> dict[str, Any]:
    """Build a JSON manifest for normalized feature outputs."""

    min_value, max_value = bundle.value_range()
    return {
        "video_id": video_id,
        "calibration_id": artifact.calibration_id,
        "status": artifact.manifest.get("status"),
        "development_only": bool(artifact.manifest.get("development_only", True)),
        "development_warning": DEVELOPMENT_NORMALIZATION_WARNING,
        "source_resnet_path": str(aligned.resnet_source_path),
        "source_temporal_path": str(aligned.temporal_source_path),
        "source_resnet_sha256": aligned.resnet_source_sha256,
        "source_temporal_sha256": aligned.temporal_source_sha256,
        "calibration_npz_path": str(artifact.paths.npz_path.resolve()),
        "calibration_npz_sha256": artifact.npz_sha256,
        "segment_count": bundle.segment_ids.shape[0],
        "segment_ids": bundle.segment_ids.astype(int).tolist(),
        "feature_dimensions": {
            "resnet_raw": RESNET_SEGMENT_DIMENSION,
            "temporal_raw": TEMPORAL_SEGMENT_DIMENSION,
            "resnet_normalized": RESNET_SEGMENT_DIMENSION,
            "temporal_normalized": TEMPORAL_SEGMENT_DIMENSION,
            "combined_normalized": COMBINED_FEATURE_DIMENSION,
        },
        "stream_boundaries": stream_boundaries_for_manifest(),
        "normalization_settings": {
            "method": NORMALIZATION_METHOD,
            "epsilon": artifact.resnet_normalizer.epsilon,
            "clipping_range": [
                artifact.resnet_normalizer.clip_min,
                artifact.resnet_normalizer.clip_max,
            ],
            "resnet_zero_iqr_dimension_count": int(np.count_nonzero(artifact.resnet_normalizer.zero_iqr_mask)),
            "temporal_zero_iqr_dimension_count": int(
                np.count_nonzero(artifact.temporal_normalizer.zero_iqr_mask)
            ),
        },
        "normalized_value_range": [min_value, max_value],
        "finite_values": bundle.finite(),
        "output_npz_path": str(paths.npz_path.resolve()),
        "npz_sha256": npz_sha256,
        "processing_time_seconds": processing_time_seconds,
        "warnings": list(aligned.warnings) + [DEVELOPMENT_NORMALIZATION_WARNING],
        "failures": [],
    }


def load_normalized_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load a normalized feature NPZ file."""

    with np.load(Path(path), allow_pickle=False) as payload:
        return {name: payload[name] for name in payload.files}


def normalize_and_store_features(
    video_id: str,
    resnet_root: str | Path,
    temporal_root: str | Path,
    manifests_root: str | Path,
    calibration_root: str | Path,
    normalized_root: str | Path,
    calibration_id: str = DEFAULT_CALIBRATION_ID,
    overwrite: bool = False,
) -> tuple[NormalizedFeatureBundle | None, dict[str, Any], NormalizedOutputPaths, bool]:
    """Normalize one video's aligned feature streams and store outputs."""

    artifact = load_normalization_artifact(calibration_root, calibration_id)
    aligned = load_aligned_features_from_roots(video_id, resnet_root, temporal_root, manifests_root)
    paths = normalized_output_paths(normalized_root, video_id)
    cache_key = build_normalized_cache_key(aligned, artifact)
    if ensure_can_write_normalized(paths, overwrite, cache_key):
        with paths.manifest_path.open("r", encoding="utf-8") as handle:
            return None, json.load(handle), paths, True

    started = perf_counter()
    bundle = combine_normalized_streams(
        aligned,
        artifact.resnet_normalizer,
        artifact.temporal_normalizer,
    )
    save_normalized_npz(paths, bundle, overwrite=overwrite)
    npz_checksum = sha256_file(paths.npz_path)
    processing_time = perf_counter() - started
    manifest = build_normalized_manifest(
        video_id=video_id,
        artifact=artifact,
        aligned=aligned,
        bundle=bundle,
        paths=paths,
        npz_sha256=npz_checksum,
        processing_time_seconds=processing_time,
    )
    manifest.update(cache_key)
    save_normalized_manifest(manifest, paths, overwrite=overwrite)
    return bundle, manifest, paths, False
