
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.authentication.digest import DigestError, unpack_packed_bits
from src.authentication.quantization import DIGEST_LENGTHS, STREAM_BOUNDARIES
from src.authentication.digest_storage import digest_output_paths, load_digest_npz
from src.features.feature_storage import sha256_file


SCHEMA_VERSION = 1
TIMESTAMP_SCALE = 1_000_000


class CanonicalizationError(RuntimeError):
    """Raised when a digest cannot be converted to a canonical payload."""


def seconds_to_microseconds(value: float | int) -> int:
    """Convert a floating timestamp in seconds to integer microseconds."""

    numeric = float(value)
    if not np.isfinite(numeric):
        raise CanonicalizationError(f"Timestamp is not finite: {value!r}.")
    return int(round(numeric * TIMESTAMP_SCALE))


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    """Serialize a payload as deterministic canonical JSON bytes."""

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_payload_sha256(payload: dict[str, Any]) -> str:
    """Return the SHA-256 checksum of a canonical payload."""

    import hashlib

    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _as_int_scalar(arrays: dict[str, np.ndarray], name: str) -> int:
    if name not in arrays:
        raise CanonicalizationError(f"Digest NPZ is missing array: {name}.")
    value = np.asarray(arrays[name])
    if value.shape not in {(), (1,)}:
        raise CanonicalizationError(f"{name} must be a scalar digest length, got shape {value.shape}.")
    return int(value.reshape(-1)[0])


def _require_array(arrays: dict[str, np.ndarray], name: str) -> np.ndarray:
    if name not in arrays:
        raise CanonicalizationError(f"Digest NPZ is missing array: {name}.")
    return np.asarray(arrays[name])


def _validate_binary_matrix(values: np.ndarray, expected_width: int, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.uint8)
    if matrix.ndim != 2 or matrix.shape[1] != expected_width:
        raise CanonicalizationError(f"{name} must have shape (segments, {expected_width}), got {matrix.shape}.")
    if not np.all(np.isin(matrix, [0, 1])):
        raise CanonicalizationError(f"{name} contains values other than 0 and 1.")
    return matrix


def _validate_segment_ids(values: np.ndarray, video_id: str) -> np.ndarray:
    ids = np.asarray(values, dtype=np.int64)
    if ids.ndim != 1:
        raise CanonicalizationError(f"Segment IDs for {video_id} must be one-dimensional.")
    if ids.tolist() != sorted(ids.tolist()):
        raise CanonicalizationError(f"Segment IDs for {video_id} are not ordered.")
    unique_ids, counts = np.unique(ids, return_counts=True)
    duplicates = unique_ids[counts > 1]
    if duplicates.size:
        raise CanonicalizationError(f"Segment IDs for {video_id} contain duplicates: {duplicates.tolist()}.")
    return ids


def _validate_packed_round_trip(
    packed: np.ndarray,
    unpacked: np.ndarray,
    bit_length: int,
    bit_order: str,
    name: str,
) -> np.ndarray:
    packed_matrix = np.asarray(packed, dtype=np.uint8)
    if packed_matrix.ndim != 2:
        raise CanonicalizationError(f"{name} packed digest must be a 2D byte matrix.")
    try:
        restored = unpack_packed_bits(packed_matrix, bit_length, bit_order)
    except DigestError as exc:
        raise CanonicalizationError(str(exc)) from exc
    if not np.array_equal(restored, unpacked):
        raise CanonicalizationError(f"{name} packed and unpacked digests do not agree.")
    return packed_matrix


def _manifest_digest_lengths(manifest: dict[str, Any]) -> dict[str, int]:
    lengths = manifest.get("digest_dimensions")
    if not isinstance(lengths, dict):
        raise CanonicalizationError("Digest manifest is missing digest_dimensions.")
    result = {str(key): int(value) for key, value in lengths.items()}
    if result != DIGEST_LENGTHS:
        raise CanonicalizationError(f"Unexpected digest dimensions: {result}.")
    return result


def _manifest_stream_boundaries(manifest: dict[str, Any]) -> dict[str, dict[str, int]]:
    boundaries = manifest.get("stream_boundaries")
    if not isinstance(boundaries, dict):
        raise CanonicalizationError("Digest manifest is missing stream_boundaries.")
    result = {
        str(stream): {
            "start": int(values["start"]),
            "end_exclusive": int(values["end_exclusive"]),
        }
        for stream, values in boundaries.items()
    }
    if result != STREAM_BOUNDARIES:
        raise CanonicalizationError(f"Unexpected stream boundaries: {result}.")
    return result


def _hex_row(matrix: np.ndarray, row_index: int) -> str:
    return np.asarray(matrix[row_index], dtype=np.uint8).tobytes().hex()


def build_canonical_payload_from_digest_files(
    video_id: str,
    digest_root: str | Path,
    schema_version: int = SCHEMA_VERSION,
) -> dict[str, Any]:
    """Build a canonical authentication payload from a stored digest NPZ and manifest."""

    paths = digest_output_paths(digest_root, video_id)
    if not paths.npz_path.exists():
        raise CanonicalizationError(f"Digest NPZ not found for {video_id}: {paths.npz_path}")
    if not paths.manifest_path.exists():
        raise CanonicalizationError(f"Digest manifest not found for {video_id}: {paths.manifest_path}")

    with paths.manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("video_id") != video_id:
        raise CanonicalizationError(
            f"Digest manifest video_id '{manifest.get('video_id')}' does not match '{video_id}'."
        )

    digest_npz_checksum = sha256_file(paths.npz_path)
    if manifest.get("npz_sha256") != digest_npz_checksum:
        raise CanonicalizationError(f"Digest NPZ checksum mismatch for {video_id}: {paths.npz_path}")
    digest_manifest_checksum = sha256_file(paths.manifest_path)
    normalization_id = str(manifest.get("normalization_calibration_id", ""))
    quantization_id = str(manifest.get("quantization_id", ""))
    if not normalization_id:
        raise CanonicalizationError(f"Digest manifest for {video_id} is missing normalization_calibration_id.")
    if not quantization_id:
        raise CanonicalizationError(f"Digest manifest for {video_id} is missing quantization_id.")

    bit_order = str(manifest.get("bit_order", ""))
    if bit_order not in {"big", "little"}:
        raise CanonicalizationError(f"Unsupported bit order for {video_id}: {bit_order!r}.")
    digest_lengths = _manifest_digest_lengths(manifest)
    stream_boundaries = _manifest_stream_boundaries(manifest)
    arrays = load_digest_npz(paths.npz_path)

    segment_ids = _validate_segment_ids(_require_array(arrays, "segment_ids"), video_id)
    start_times = np.asarray(_require_array(arrays, "segment_start_times"), dtype=np.float64)
    end_times = np.asarray(_require_array(arrays, "segment_end_times"), dtype=np.float64)
    if start_times.shape != segment_ids.shape or end_times.shape != segment_ids.shape:
        raise CanonicalizationError("Segment timestamp arrays must match segment_ids shape.")
    if not np.all(np.isfinite(start_times)) or not np.all(np.isfinite(end_times)):
        raise CanonicalizationError("Segment timestamps contain non-finite values.")

    resnet_length = _as_int_scalar(arrays, "resnet_bit_length")
    temporal_length = _as_int_scalar(arrays, "temporal_bit_length")
    hybrid_length = _as_int_scalar(arrays, "hybrid_bit_length")
    expected_lengths = (
        digest_lengths["resnet"],
        digest_lengths["temporal"],
        digest_lengths["hybrid"],
    )
    if (resnet_length, temporal_length, hybrid_length) != expected_lengths:
        raise CanonicalizationError(
            f"Digest bit lengths {(resnet_length, temporal_length, hybrid_length)} do not match {expected_lengths}."
        )

    resnet_bits = _validate_binary_matrix(
        _require_array(arrays, "resnet_binary_digests"),
        resnet_length,
        "ResNet binary digests",
    )
    temporal_bits = _validate_binary_matrix(
        _require_array(arrays, "temporal_binary_digests"),
        temporal_length,
        "Temporal binary digests",
    )
    hybrid_bits = _validate_binary_matrix(
        _require_array(arrays, "hybrid_binary_digests"),
        hybrid_length,
        "Hybrid binary digests",
    )
    segment_count = int(segment_ids.shape[0])
    for name, matrix in (
        ("ResNet binary digests", resnet_bits),
        ("Temporal binary digests", temporal_bits),
        ("Hybrid binary digests", hybrid_bits),
    ):
        if matrix.shape[0] != segment_count:
            raise CanonicalizationError(f"{name} segment count does not match segment_ids.")

    resnet_packed = _validate_packed_round_trip(
        _require_array(arrays, "resnet_packed_digests"),
        resnet_bits,
        resnet_length,
        bit_order,
        "ResNet",
    )
    temporal_packed = _validate_packed_round_trip(
        _require_array(arrays, "temporal_packed_digests"),
        temporal_bits,
        temporal_length,
        bit_order,
        "Temporal",
    )
    hybrid_packed = _validate_packed_round_trip(
        _require_array(arrays, "hybrid_packed_digests"),
        hybrid_bits,
        hybrid_length,
        bit_order,
        "Hybrid",
    )
    if any(matrix.shape[0] != segment_count for matrix in (resnet_packed, temporal_packed, hybrid_packed)):
        raise CanonicalizationError("Packed digest segment count does not match segment_ids.")

    manifest_segments = manifest.get("segments", [])
    if not isinstance(manifest_segments, list) or len(manifest_segments) != segment_count:
        raise CanonicalizationError("Digest manifest segment records do not match digest segment count.")

    segments: list[dict[str, Any]] = []
    for index, segment_id in enumerate(segment_ids.tolist()):
        manifest_segment = manifest_segments[index]
        if int(manifest_segment.get("segment_id")) != int(segment_id):
            raise CanonicalizationError("Digest manifest segment order does not match NPZ segment order.")
        start_us = seconds_to_microseconds(start_times[index])
        end_us = seconds_to_microseconds(end_times[index])
        if end_us <= start_us:
            raise CanonicalizationError(f"Segment {segment_id} has a non-positive duration.")
        segments.append(
            {
                "segment_id": int(segment_id),
                "start_time_microseconds": start_us,
                "end_time_microseconds": end_us,
                "resnet_bit_length": resnet_length,
                "temporal_bit_length": temporal_length,
                "hybrid_bit_length": hybrid_length,
                "resnet_packed_digest_hex": _hex_row(resnet_packed, index),
                "temporal_packed_digest_hex": _hex_row(temporal_packed, index),
                "hybrid_packed_digest_hex": _hex_row(hybrid_packed, index),
            }
        )

    payload = {
        "schema_version": int(schema_version),
        "video_id": video_id,
        "normalization_id": normalization_id,
        "quantization_id": quantization_id,
        "development_only": bool(manifest.get("development_only", False)),
        "digest_npz_checksum": digest_npz_checksum,
        "digest_manifest_checksum": digest_manifest_checksum,
        "bit_order": bit_order,
        "stream_boundaries": deepcopy(stream_boundaries),
        "digest_lengths": deepcopy(digest_lengths),
        "segment_count": segment_count,
        "segments": segments,
    }
    canonical_json_bytes(payload)
    return payload
