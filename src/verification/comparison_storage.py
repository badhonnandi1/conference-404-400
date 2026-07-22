"""Storage and cache helpers for Phase 7 digest comparison results."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from src.authentication.hmac_auth import HMACKeyInfo
from src.features.feature_storage import sha256_file
from src.verification.comparison import (
    ATTRIBUTION_LABELS,
    ComparisonConfig,
    ComparisonResult,
    DigestComparisonError,
    compare_digests,
)
from src.video.metadata import ExistingOutputError


class ComparisonStorageError(RuntimeError):
    """Raised when comparison results cannot be stored or loaded."""


@dataclass(frozen=True)
class ComparisonOutputPaths:
    """Output paths for one reference/query comparison."""

    output_dir: Path
    npz_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class StoredComparison:
    """Saved or cached comparison result."""

    result: ComparisonResult
    manifest: dict[str, Any]
    paths: ComparisonOutputPaths
    cache_reused: bool


def comparison_output_paths(root: str | Path, reference_id: str, query_id: str) -> ComparisonOutputPaths:
    """Return deterministic output paths for a reference/query comparison."""

    output_dir = Path(root) / f"{reference_id}__vs__{query_id}"
    return ComparisonOutputPaths(
        output_dir=output_dir,
        npz_path=output_dir / "comparison_results.npz",
        manifest_path=output_dir / "comparison_manifest.json",
    )


def comparison_npz_arrays(result: ComparisonResult) -> dict[str, np.ndarray]:
    """Build NPZ arrays for matched-segment comparison results."""

    return {
        "matched_segment_ids": np.asarray([item.segment_id for item in result.segment_results], dtype=np.int64),
        "resnet_raw_distances": np.asarray(
            [item.resnet.raw_distance for item in result.segment_results], dtype=np.int64
        ),
        "resnet_normalized_distances": np.asarray(
            [item.resnet.normalized_distance for item in result.segment_results], dtype=np.float64
        ),
        "temporal_raw_distances": np.asarray(
            [item.temporal.raw_distance for item in result.segment_results], dtype=np.int64
        ),
        "temporal_normalized_distances": np.asarray(
            [item.temporal.normalized_distance for item in result.segment_results], dtype=np.float64
        ),
        "hybrid_raw_distances": np.asarray(
            [item.hybrid.raw_distance for item in result.segment_results], dtype=np.int64
        ),
        "hybrid_normalized_distances": np.asarray(
            [item.hybrid.normalized_distance for item in result.segment_results], dtype=np.float64
        ),
        "balanced_diagnostic_scores": np.asarray(
            [item.development_diagnostic_score for item in result.segment_results], dtype=np.float64
        ),
        "attribution_codes": np.asarray([item.attribution_code for item in result.segment_results], dtype=np.int64),
    }


def save_comparison_npz(result: ComparisonResult, paths: ComparisonOutputPaths, overwrite: bool = False) -> Path:
    """Save comparison arrays in compressed NumPy format."""

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    if paths.npz_path.exists() and not overwrite:
        raise ExistingOutputError(f"Comparison NPZ already exists: {paths.npz_path}. Use --overwrite to replace it.")
    np.savez_compressed(paths.npz_path, **comparison_npz_arrays(result))
    return paths.npz_path


def load_comparison_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load comparison result arrays from NPZ."""

    with np.load(Path(path), allow_pickle=False) as payload:
        return {name: payload[name] for name in payload.files}


def build_comparison_manifest(
    result: ComparisonResult,
    paths: ComparisonOutputPaths,
    npz_sha256: str,
    processing_time_seconds: float,
) -> dict[str, Any]:
    """Build a JSON manifest for one comparison result."""

    return {
        "comparison_id": result.comparison_id,
        "reference_video_id": result.reference_video_id,
        "query_video_id": result.query_video_id,
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        "reference_authentication_record_path": str(result.reference_record_path),
        "reference_record_checksum": result.reference_record_checksum,
        "reference_hmac_verification_result": result.reference_hmac_verification_result,
        "query_digest_path": str(result.query_digest_path),
        "query_digest_checksum": result.query_digest_checksum,
        "query_digest_manifest_path": str(result.query_manifest_path),
        "query_digest_manifest_checksum": result.query_manifest_checksum,
        "normalization_id": result.normalization_id,
        "quantization_id": result.quantization_id,
        "comparison_configuration": result.config.to_dict(),
        "diagnostic_weights": result.config.diagnostic_weights.to_dict(),
        "alignment_results": result.alignment.to_manifest(),
        "per_segment_results": [segment.to_dict() for segment in result.segment_results],
        "video_level_summary": result.video_summary,
        "output_npz_path": str(paths.npz_path.resolve()),
        "output_npz_checksum": npz_sha256,
        "processing_time_seconds": processing_time_seconds,
        "warnings": list(result.warnings),
        "failures": list(result.failures),
        "no_threshold_warning": result.warnings[0],
        "attribution_code_legend": {str(code): label for code, label in ATTRIBUTION_LABELS.items()},
    }


def save_comparison_manifest(
    manifest: dict[str, Any],
    paths: ComparisonOutputPaths,
    overwrite: bool = False,
) -> Path:
    """Save comparison manifest JSON."""

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    if paths.manifest_path.exists() and not overwrite:
        raise ExistingOutputError(
            f"Comparison manifest already exists: {paths.manifest_path}. Use --overwrite to replace it."
        )
    with paths.manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return paths.manifest_path


def load_comparison_manifest(path: str | Path) -> dict[str, Any]:
    """Load a comparison manifest."""

    manifest_path = Path(path)
    if not manifest_path.exists():
        raise ComparisonStorageError(f"Comparison manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ComparisonStorageError(f"Comparison manifest root must be an object: {manifest_path}")
    return manifest


def _cache_matches(manifest: dict[str, Any], result: ComparisonResult, paths: ComparisonOutputPaths) -> bool:
    if not paths.npz_path.exists():
        return False
    if manifest.get("reference_record_checksum") != result.reference_record_checksum:
        return False
    if not bool(manifest.get("reference_hmac_verification_result", {}).get("record_valid")):
        return False
    if not bool(result.reference_hmac_verification_result.get("record_valid")):
        return False
    if manifest.get("query_digest_checksum") != result.query_digest_checksum:
        return False
    if manifest.get("normalization_id") != result.normalization_id:
        return False
    if manifest.get("quantization_id") != result.quantization_id:
        return False
    if manifest.get("comparison_configuration") != result.config.to_dict():
        return False
    if manifest.get("diagnostic_weights") != result.config.diagnostic_weights.to_dict():
        return False
    if manifest.get("output_npz_checksum") != sha256_file(paths.npz_path):
        return False
    return True


def compare_and_store_digests(
    reference_id: str,
    query_id: str,
    authentication_record_root: str | Path,
    digest_root: str | Path,
    comparison_root: str | Path,
    key_info: HMACKeyInfo,
    config: ComparisonConfig,
    algorithm: str = "HMAC-SHA-256",
    overwrite: bool = False,
) -> StoredComparison:
    """Compare two digest records and store or reuse the comparison result."""

    started = perf_counter()
    result = compare_digests(
        reference_id=reference_id,
        query_id=query_id,
        authentication_record_root=authentication_record_root,
        digest_root=digest_root,
        key_info=key_info,
        config=config,
        algorithm=algorithm,
    )
    paths = comparison_output_paths(comparison_root, reference_id, query_id)
    if paths.npz_path.exists() or paths.manifest_path.exists():
        if overwrite:
            pass
        elif paths.npz_path.exists() and paths.manifest_path.exists():
            manifest = load_comparison_manifest(paths.manifest_path)
            if _cache_matches(manifest, result, paths):
                return StoredComparison(result=result, manifest=manifest, paths=paths, cache_reused=True)
            raise ExistingOutputError(
                f"Comparison outputs already exist under {paths.output_dir}. "
                "Use --overwrite to regenerate because cache metadata did not match."
            )
        else:
            raise ExistingOutputError(
                f"Incomplete comparison outputs exist under {paths.output_dir}. Use --overwrite to regenerate."
            )

    save_comparison_npz(result, paths, overwrite=overwrite)
    npz_sha = sha256_file(paths.npz_path)
    processing_time = perf_counter() - started
    manifest = build_comparison_manifest(result, paths, npz_sha, processing_time)
    save_comparison_manifest(manifest, paths, overwrite=overwrite)
    return StoredComparison(result=result, manifest=manifest, paths=paths, cache_reused=False)


def inspect_comparison(comparison_root: str | Path, reference_id: str, query_id: str) -> dict[str, Any]:
    """Load and return a comparison manifest for inspection."""

    return load_comparison_manifest(comparison_output_paths(comparison_root, reference_id, query_id).manifest_path)
