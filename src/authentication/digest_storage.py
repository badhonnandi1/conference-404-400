"""Storage and cache helpers for quantization artifacts and binary digests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from time import perf_counter
from typing import Any

import numpy as np

from src.authentication.digest import DigestBundle, build_digest_bundle, digest_summary
from src.authentication.quantization import (
    DEFAULT_QUANTIZATION_ID,
    DEFAULT_QUANTIZATION_VERSION,
    DIGEST_LENGTHS,
    QUANTIZATION_WARNING,
    QuantizationParameters,
    derive_quantization_parameters,
    digest_lengths_for_manifest,
    gray_code_mapping_for_manifest,
    quantization_metadata,
    stream_boundaries_for_manifest,
)
from src.features.alignment import RESNET_SEGMENT_DIMENSION, TEMPORAL_SEGMENT_DIMENSION
from src.features.feature_storage import sha256_file
from src.features.normalization_storage import load_normalization_artifact
from src.video.metadata import ExistingOutputError


PADDING_POLICY = "zero_pad_to_full_byte"


@dataclass(frozen=True)
class QuantizationArtifactPaths:
    """Output paths for a quantization artifact."""

    output_dir: Path
    npz_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class LoadedQuantizationArtifact:
    """Loaded quantizer parameters and metadata."""

    quantization_id: str
    parameters: QuantizationParameters
    manifest: dict[str, Any]
    paths: QuantizationArtifactPaths
    npz_sha256: str


@dataclass(frozen=True)
class NormalizedFeatureInput:
    """Validated normalized feature arrays for one video."""

    video_id: str
    calibration_id: str
    development_only: bool
    segment_ids: np.ndarray
    segment_start_times: np.ndarray
    segment_end_times: np.ndarray
    resnet_normalized_features: np.ndarray
    temporal_normalized_features: np.ndarray
    source_path: Path
    source_sha256: str
    manifest_path: Path
    manifest: dict[str, Any]
    calibration_npz_sha256: str


@dataclass(frozen=True)
class DigestOutputPaths:
    """Output paths for one video's digests."""

    output_dir: Path
    npz_path: Path
    manifest_path: Path


def quantization_artifact_paths(calibration_root: str | Path, quantization_id: str) -> QuantizationArtifactPaths:
    """Return deterministic quantization artifact paths."""

    output_dir = Path(calibration_root) / quantization_id
    return QuantizationArtifactPaths(
        output_dir=output_dir,
        npz_path=output_dir / "quantization_parameters.npz",
        manifest_path=output_dir / "quantization_manifest.json",
    )


def digest_output_paths(digest_root: str | Path, video_id: str) -> DigestOutputPaths:
    """Return deterministic digest output paths for a video."""

    output_dir = Path(digest_root) / video_id
    return DigestOutputPaths(
        output_dir=output_dir,
        npz_path=output_dir / f"{video_id}_digests.npz",
        manifest_path=output_dir / f"{video_id}_digest_manifest.json",
    )


def normalized_feature_paths(normalized_root: str | Path, video_id: str) -> tuple[Path, Path]:
    """Return normalized NPZ and manifest paths for a video."""

    output_dir = Path(normalized_root) / video_id
    return (
        output_dir / f"{video_id}_normalized_features.npz",
        output_dir / f"{video_id}_normalized_manifest.json",
    )


def _validate_segment_ids(segment_ids: np.ndarray, video_id: str) -> np.ndarray:
    ids = np.asarray(segment_ids, dtype=np.int64)
    if ids.ndim != 1:
        raise ValueError(f"Digest source segment IDs for {video_id} must be one-dimensional.")
    unique_ids, counts = np.unique(ids, return_counts=True)
    duplicates = unique_ids[counts > 1]
    if duplicates.size:
        raise ValueError(f"Digest source for {video_id} has duplicate segment IDs: {duplicates.tolist()}.")
    if ids.tolist() != sorted(ids.tolist()):
        raise ValueError(f"Digest source segment IDs for {video_id} are not deterministically sorted.")
    return ids.copy()


def load_normalized_feature_input(
    normalized_root: str | Path,
    video_id: str,
    expected_calibration_id: str | None = None,
    clipping_tolerance: float = 1e-6,
) -> NormalizedFeatureInput:
    """Load and validate Phase 4 normalized features for digest construction."""

    npz_path, manifest_path = normalized_feature_paths(normalized_root, video_id)
    if not npz_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"Normalized features not found for {video_id}: {npz_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("video_id") != video_id:
        raise ValueError(f"Normalized manifest video_id '{manifest.get('video_id')}' does not match '{video_id}'.")
    calibration_id = str(manifest.get("calibration_id", ""))
    if not calibration_id:
        raise ValueError(f"Normalized manifest for {video_id} is missing calibration_id.")
    if expected_calibration_id and calibration_id != expected_calibration_id:
        raise ValueError(
            f"Normalized manifest calibration_id '{calibration_id}' does not match quantizer "
            f"normalization_id '{expected_calibration_id}'."
        )
    source_sha = sha256_file(npz_path)
    if manifest.get("npz_sha256") and manifest["npz_sha256"] != source_sha:
        raise ValueError(f"Normalized NPZ checksum mismatch for {video_id}: {npz_path}")
    calibration_path = Path(str(manifest.get("calibration_npz_path", "")))
    calibration_sha = str(manifest.get("calibration_npz_sha256", ""))
    if calibration_path.exists() and calibration_sha and sha256_file(calibration_path) != calibration_sha:
        raise ValueError(f"Calibration checksum mismatch for normalized source {video_id}.")
    for path_key, checksum_key in (
        ("source_resnet_path", "source_resnet_sha256"),
        ("source_temporal_path", "source_temporal_sha256"),
    ):
        source_path = Path(str(manifest.get(path_key, "")))
        source_checksum = manifest.get(checksum_key)
        if source_path.exists() and source_checksum and sha256_file(source_path) != source_checksum:
            raise ValueError(f"{path_key} checksum mismatch for {video_id}.")

    with np.load(npz_path, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    required = [
        "segment_ids",
        "segment_start_times",
        "segment_end_times",
        "resnet_normalized_features",
        "temporal_normalized_features",
        "combined_normalized_features",
    ]
    missing = [name for name in required if name not in arrays]
    if missing:
        raise ValueError(f"Normalized NPZ for {video_id} is missing arrays: {missing}.")
    segment_ids = _validate_segment_ids(arrays["segment_ids"], video_id)
    resnet = np.asarray(arrays["resnet_normalized_features"], dtype=np.float32)
    temporal = np.asarray(arrays["temporal_normalized_features"], dtype=np.float32)
    combined = np.asarray(arrays["combined_normalized_features"], dtype=np.float32)
    if resnet.shape != (segment_ids.shape[0], RESNET_SEGMENT_DIMENSION):
        raise ValueError(f"Unexpected ResNet normalized shape for {video_id}: {resnet.shape}.")
    if temporal.shape != (segment_ids.shape[0], TEMPORAL_SEGMENT_DIMENSION):
        raise ValueError(f"Unexpected temporal normalized shape for {video_id}: {temporal.shape}.")
    if combined.shape != (segment_ids.shape[0], RESNET_SEGMENT_DIMENSION + TEMPORAL_SEGMENT_DIMENSION):
        raise ValueError(f"Unexpected combined normalized shape for {video_id}: {combined.shape}.")
    if not np.all(np.isfinite(resnet)) or not np.all(np.isfinite(temporal)) or not np.all(np.isfinite(combined)):
        raise ValueError(f"Normalized features for {video_id} contain non-finite values.")
    clipping_range = manifest.get("normalization_settings", {}).get("clipping_range", [-5.0, 5.0])
    clip_min, clip_max = float(clipping_range[0]), float(clipping_range[1])
    if float(np.min(combined)) < clip_min - clipping_tolerance or float(np.max(combined)) > clip_max + clipping_tolerance:
        raise ValueError(f"Normalized features for {video_id} exceed the configured clipping range.")
    return NormalizedFeatureInput(
        video_id=video_id,
        calibration_id=calibration_id,
        development_only=bool(manifest.get("development_only", True)),
        segment_ids=segment_ids,
        segment_start_times=np.asarray(arrays["segment_start_times"], dtype=np.float64).copy(),
        segment_end_times=np.asarray(arrays["segment_end_times"], dtype=np.float64).copy(),
        resnet_normalized_features=resnet.copy(),
        temporal_normalized_features=temporal.copy(),
        source_path=npz_path.resolve(),
        source_sha256=source_sha,
        manifest_path=manifest_path.resolve(),
        manifest=manifest,
        calibration_npz_sha256=calibration_sha,
    )


def create_and_store_quantizer(
    normalization_root: str | Path,
    quantization_root: str | Path,
    normalization_id: str,
    quantization_id: str = DEFAULT_QUANTIZATION_ID,
    version: str = DEFAULT_QUANTIZATION_VERSION,
    status: str = "development",
    bit_order: str = "big",
    overwrite: bool = False,
) -> LoadedQuantizationArtifact:
    """Create a development quantizer from an existing normalization artifact."""

    paths = quantization_artifact_paths(quantization_root, quantization_id)
    if (paths.npz_path.exists() or paths.manifest_path.exists()) and not overwrite:
        raise ExistingOutputError(
            f"Quantization artifact already exists under {paths.output_dir}. Use --overwrite to regenerate it."
        )
    normalization = load_normalization_artifact(normalization_root, normalization_id)
    params = derive_quantization_parameters(
        normalization,
        quantization_id=quantization_id,
        version=version,
        status=status,
        bit_order=bit_order,
    )
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(paths.npz_path, **params.to_arrays())
    npz_checksum = sha256_file(paths.npz_path)
    manifest = {
        **quantization_metadata(params),
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        "source_videos_used_by_normalization": normalization.manifest.get("source_video_ids", []),
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "npz_output_path": str(paths.npz_path.resolve()),
        "npz_sha256": npz_checksum,
        "warnings": [QUANTIZATION_WARNING],
        "limitations": [
            "Development quantizer depends on DEV_NORMALIZATION_V1.",
            "HMAC, Hamming-distance comparison, thresholds, verification, compression testing, and tamper testing are not implemented in Phase 5.",
        ],
    }
    with paths.manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return load_quantization_artifact(quantization_root, quantization_id)


def load_quantization_artifact(
    quantization_root: str | Path,
    quantization_id: str,
) -> LoadedQuantizationArtifact:
    """Load a quantization artifact and verify its manifest ID."""

    paths = quantization_artifact_paths(quantization_root, quantization_id)
    if not paths.npz_path.exists() or not paths.manifest_path.exists():
        raise FileNotFoundError(f"Quantization artifact not found for ID {quantization_id}: {paths.output_dir}")
    with paths.manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("quantization_id") != quantization_id:
        raise ValueError(
            f"Quantization manifest ID '{manifest.get('quantization_id')}' does not match '{quantization_id}'."
        )
    npz_sha = sha256_file(paths.npz_path)
    if manifest.get("npz_sha256") and manifest["npz_sha256"] != npz_sha:
        raise ValueError(f"Quantization NPZ checksum mismatch: {paths.npz_path}")
    with np.load(paths.npz_path, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    params = QuantizationParameters.from_arrays(
        arrays,
        quantization_id=quantization_id,
        version=str(manifest["quantization_version"]),
        normalization_id=str(manifest["normalization_calibration_id"]),
        normalization_npz_sha256=str(manifest["normalization_artifact_checksum"]),
        bit_order=str(manifest["bit_order"]),
        status=str(manifest["status"]),
        development_only=bool(manifest["development_only"]),
    )
    return LoadedQuantizationArtifact(
        quantization_id=quantization_id,
        parameters=params,
        manifest=manifest,
        paths=paths,
        npz_sha256=npz_sha,
    )


def save_digest_npz(paths: DigestOutputPaths, bundle: DigestBundle, overwrite: bool = False) -> Path:
    """Save digest arrays in compressed NumPy format."""

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    if paths.npz_path.exists() and not overwrite:
        raise ExistingOutputError(f"Digest NPZ already exists: {paths.npz_path}. Use --overwrite to replace it.")
    np.savez_compressed(
        paths.npz_path,
        segment_ids=bundle.segment_ids.astype(np.int64),
        segment_start_times=bundle.segment_start_times.astype(np.float64),
        segment_end_times=bundle.segment_end_times.astype(np.float64),
        resnet_binary_digests=bundle.resnet_binary_digests.astype(np.uint8),
        temporal_bin_indices=bundle.temporal_bin_indices.astype(np.uint8),
        temporal_binary_digests=bundle.temporal_binary_digests.astype(np.uint8),
        hybrid_binary_digests=bundle.hybrid_binary_digests.astype(np.uint8),
        resnet_packed_digests=bundle.resnet_packed_digests.astype(np.uint8),
        temporal_packed_digests=bundle.temporal_packed_digests.astype(np.uint8),
        hybrid_packed_digests=bundle.hybrid_packed_digests.astype(np.uint8),
        resnet_bit_length=np.asarray(bundle.resnet_bit_length, dtype=np.int64),
        temporal_bit_length=np.asarray(bundle.temporal_bit_length, dtype=np.int64),
        hybrid_bit_length=np.asarray(bundle.hybrid_bit_length, dtype=np.int64),
    )
    return paths.npz_path


def load_digest_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load digest arrays from NPZ."""

    with np.load(Path(path), allow_pickle=False) as payload:
        return {name: payload[name] for name in payload.files}


def build_digest_cache_key(
    normalized: NormalizedFeatureInput,
    quantizer: LoadedQuantizationArtifact,
) -> dict[str, Any]:
    """Build digest cache matching fields."""

    return {
        "source_normalized_feature_sha256": normalized.source_sha256,
        "source_calibration_sha256": normalized.calibration_npz_sha256,
        "quantization_artifact_sha256": quantizer.npz_sha256,
        "quantization_version": quantizer.parameters.version,
        "feature_dimensions": {
            "resnet": RESNET_SEGMENT_DIMENSION,
            "temporal": TEMPORAL_SEGMENT_DIMENSION,
        },
        "bit_order": quantizer.parameters.bit_order,
        "stream_boundaries": stream_boundaries_for_manifest(),
        "padding_policy": PADDING_POLICY,
    }


def digest_manifest_matches(manifest: dict[str, Any], cache_key: dict[str, Any]) -> bool:
    """Return whether digest cache metadata matches."""

    return all(manifest.get(key) == value for key, value in cache_key.items())


def ensure_can_write_digest(
    paths: DigestOutputPaths,
    overwrite: bool,
    cache_key: dict[str, Any],
) -> bool:
    """Return True when an existing digest output can be reused."""

    if not paths.npz_path.exists() and not paths.manifest_path.exists():
        return False
    if overwrite:
        return False
    if paths.npz_path.exists() and paths.manifest_path.exists():
        with paths.manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if digest_manifest_matches(manifest, cache_key):
            return True
    raise ExistingOutputError(
        f"Digest outputs already exist under {paths.output_dir}. "
        "Use --overwrite to regenerate because cache metadata did not match."
    )


def build_digest_manifest(
    normalized: NormalizedFeatureInput,
    quantizer: LoadedQuantizationArtifact,
    bundle: DigestBundle,
    paths: DigestOutputPaths,
    npz_sha256: str,
    processing_time_seconds: float,
) -> dict[str, Any]:
    """Build JSON manifest for one video's digests."""

    summary = digest_summary(bundle)
    return {
        "video_id": normalized.video_id,
        "normalization_calibration_id": normalized.calibration_id,
        "quantization_id": quantizer.quantization_id,
        "development_only": bool(normalized.development_only or quantizer.parameters.development_only),
        "development_warning": QUANTIZATION_WARNING,
        "source_normalized_feature_path": str(normalized.source_path),
        "source_normalized_feature_sha256": normalized.source_sha256,
        "source_calibration_sha256": normalized.calibration_npz_sha256,
        "quantization_artifact_path": str(quantizer.paths.npz_path.resolve()),
        "quantization_artifact_sha256": quantizer.npz_sha256,
        "segment_count": int(bundle.segment_ids.shape[0]),
        "segments": [
            {
                "segment_id": int(segment_id),
                "start_time_seconds": float(start),
                "end_time_seconds": float(end),
            }
            for segment_id, start, end in zip(
                bundle.segment_ids,
                bundle.segment_start_times,
                bundle.segment_end_times,
                strict=True,
            )
        ],
        "digest_dimensions": digest_lengths_for_manifest(),
        "stream_boundaries": stream_boundaries_for_manifest(),
        "bit_order": bundle.bit_order,
        "padding_bit_counts": {
            "resnet": bundle.resnet_padding_bits,
            "temporal": bundle.temporal_padding_bits,
            "hybrid": bundle.hybrid_padding_bits,
        },
        "padding_policy": PADDING_POLICY,
        "bit_statistics": summary,
        "clipping_statistics": bundle.clipping_statistics,
        "pack_unpack_round_trip": bundle.validate_round_trips(),
        "output_npz_path": str(paths.npz_path.resolve()),
        "npz_sha256": npz_sha256,
        "processing_time_seconds": processing_time_seconds,
        "warnings": [QUANTIZATION_WARNING],
        "failures": [],
    }


def save_digest_manifest(manifest: dict[str, Any], paths: DigestOutputPaths, overwrite: bool = False) -> Path:
    """Save a digest manifest as formatted JSON."""

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    if paths.manifest_path.exists() and not overwrite:
        raise ExistingOutputError(
            f"Digest manifest already exists: {paths.manifest_path}. Use --overwrite to replace it."
        )
    with paths.manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return paths.manifest_path


def build_and_store_digest(
    video_id: str,
    normalized_root: str | Path,
    quantization_root: str | Path,
    digest_root: str | Path,
    quantization_id: str = DEFAULT_QUANTIZATION_ID,
    overwrite: bool = False,
) -> tuple[DigestBundle | None, dict[str, Any], DigestOutputPaths, bool]:
    """Build and store one video's binary digests."""

    quantizer = load_quantization_artifact(quantization_root, quantization_id)
    normalized = load_normalized_feature_input(
        normalized_root,
        video_id,
        expected_calibration_id=quantizer.parameters.normalization_id,
    )
    paths = digest_output_paths(digest_root, video_id)
    cache_key = build_digest_cache_key(normalized, quantizer)
    if ensure_can_write_digest(paths, overwrite, cache_key):
        with paths.manifest_path.open("r", encoding="utf-8") as handle:
            return None, json.load(handle), paths, True

    started = perf_counter()
    clipping_range = normalized.manifest.get("normalization_settings", {}).get("clipping_range", [-5.0, 5.0])
    bundle = build_digest_bundle(
        video_id=video_id,
        segment_ids=normalized.segment_ids,
        segment_start_times=normalized.segment_start_times,
        segment_end_times=normalized.segment_end_times,
        resnet_normalized_features=normalized.resnet_normalized_features,
        temporal_normalized_features=normalized.temporal_normalized_features,
        parameters=quantizer.parameters,
        clip_min=float(clipping_range[0]),
        clip_max=float(clipping_range[1]),
    )
    save_digest_npz(paths, bundle, overwrite=overwrite)
    npz_sha = sha256_file(paths.npz_path)
    processing_time = perf_counter() - started
    manifest = build_digest_manifest(
        normalized=normalized,
        quantizer=quantizer,
        bundle=bundle,
        paths=paths,
        npz_sha256=npz_sha,
        processing_time_seconds=processing_time,
    )
    manifest.update(cache_key)
    save_digest_manifest(manifest, paths, overwrite=overwrite)
    return bundle, manifest, paths, False
