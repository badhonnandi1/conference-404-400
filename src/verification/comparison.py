"""Segment-level digest comparison using Hamming distance."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.authentication.auth_record_storage import (
    authentication_record_paths,
    load_authentication_record,
    verify_authentication_record,
)
from src.authentication.digest import DigestError, unpack_packed_bits
from src.authentication.digest_storage import digest_output_paths, load_digest_npz
from src.authentication.hmac_auth import HMACKeyInfo
from src.authentication.quantization import DIGEST_LENGTHS, HYBRID_DIGEST_LENGTH, STREAM_BOUNDARIES
from src.features.feature_storage import sha256_file
from src.verification.hamming import HammingDistanceResult, hamming_distance
from src.verification.segment_alignment import SegmentAlignmentResult, SegmentDescriptor, align_segments


NO_THRESHOLD_WARNING = (
    "This comparison contains raw and normalized digest distances only. No acceptance threshold has been "
    "applied, so it must not be interpreted as an authenticity or tamper-classification result."
)
ATTRIBUTION_CODES = {
    "no_difference": 0,
    "resnet_dominant": 1,
    "temporal_dominant": 2,
    "approximately_equal": 3,
}
ATTRIBUTION_LABELS = {value: key for key, value in ATTRIBUTION_CODES.items()}


class DigestComparisonError(RuntimeError):
    """Raised when a digest comparison cannot be performed safely."""


@dataclass(frozen=True)
class DiagnosticWeights:
    """Temporary stream-balanced diagnostic weights."""

    resnet: float = 0.5
    temporal: float = 0.5

    def validate(self, tolerance: float = 1.0e-9) -> None:
        """Validate non-negative weights that sum to one."""

        if self.resnet < 0 or self.temporal < 0:
            raise DigestComparisonError("Diagnostic weights must be non-negative.")
        if abs((self.resnet + self.temporal) - 1.0) > tolerance:
            raise DigestComparisonError("Diagnostic weights must sum to 1.")

    def to_dict(self) -> dict[str, float | str]:
        """Return JSON-friendly weights with an explicit development label."""

        return {
            "resnet": float(self.resnet),
            "temporal": float(self.temporal),
            "label": "development_diagnostic_weights_not_final",
        }


@dataclass(frozen=True)
class ComparisonConfig:
    """Configuration for Phase 7 distance comparison."""

    resnet_bit_length: int = DIGEST_LENGTHS["resnet"]
    temporal_bit_length: int = DIGEST_LENGTHS["temporal"]
    hybrid_bit_length: int = DIGEST_LENGTHS["hybrid"]
    diagnostic_weights: DiagnosticWeights = DiagnosticWeights()
    tie_tolerance: float = 1.0e-9
    segment_alignment: str = "strict"
    timestamp_tolerance_microseconds: int = 1000

    def validate(self) -> None:
        """Validate comparison settings."""

        if (
            self.resnet_bit_length != DIGEST_LENGTHS["resnet"]
            or self.temporal_bit_length != DIGEST_LENGTHS["temporal"]
            or self.hybrid_bit_length != DIGEST_LENGTHS["hybrid"]
        ):
            raise DigestComparisonError("Comparison bit lengths must match Phase 5 digest lengths.")
        if self.tie_tolerance < 0:
            raise DigestComparisonError("tie_tolerance must be non-negative.")
        if self.timestamp_tolerance_microseconds < 0:
            raise DigestComparisonError("timestamp_tolerance_microseconds must be non-negative.")
        if self.segment_alignment not in {"strict", "partial"}:
            raise DigestComparisonError("segment_alignment must be 'strict' or 'partial'.")
        self.diagnostic_weights.validate()

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-friendly configuration."""

        return {
            "resnet_bit_length": self.resnet_bit_length,
            "temporal_bit_length": self.temporal_bit_length,
            "hybrid_bit_length": self.hybrid_bit_length,
            "diagnostic_weights": self.diagnostic_weights.to_dict(),
            "tie_tolerance": self.tie_tolerance,
            "segment_alignment": self.segment_alignment,
            "timestamp_tolerance_microseconds": self.timestamp_tolerance_microseconds,
        }


@dataclass(frozen=True)
class LoadedDigestRecord:
    """Validated digest bits and metadata for reference or query video."""

    video_id: str
    normalization_id: str
    quantization_id: str
    digest_lengths: dict[str, int]
    stream_boundaries: dict[str, dict[str, int]]
    bit_order: str
    segments: tuple[SegmentDescriptor, ...]
    resnet_bits: np.ndarray
    temporal_bits: np.ndarray
    hybrid_bits: np.ndarray
    resnet_packed: np.ndarray
    temporal_packed: np.ndarray
    hybrid_packed: np.ndarray
    source_path: Path
    source_checksum: str
    manifest_path: Path | None = None
    manifest_checksum: str | None = None


@dataclass(frozen=True)
class SegmentComparisonResult:
    """Distance results for one matched segment."""

    segment_id: int
    reference_index: int
    query_index: int
    start_time_microseconds: int
    end_time_microseconds: int
    resnet: HammingDistanceResult
    temporal: HammingDistanceResult
    hybrid: HammingDistanceResult
    development_diagnostic_score: float
    flat_hybrid_normalized_distance: float
    attribution_label: str
    attribution_code: int
    distance_difference_resnet_minus_temporal: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly segment comparison result."""

        return {
            "segment_id": self.segment_id,
            "reference_index": self.reference_index,
            "query_index": self.query_index,
            "start_time_microseconds": self.start_time_microseconds,
            "end_time_microseconds": self.end_time_microseconds,
            "resnet_raw_distance": self.resnet.raw_distance,
            "resnet_normalized_distance": self.resnet.normalized_distance,
            "temporal_raw_distance": self.temporal.raw_distance,
            "temporal_normalized_distance": self.temporal.normalized_distance,
            "hybrid_raw_distance": self.hybrid.raw_distance,
            "hybrid_normalized_distance": self.hybrid.normalized_distance,
            "flat_hybrid_normalized_distance": self.flat_hybrid_normalized_distance,
            "development_diagnostic_score": self.development_diagnostic_score,
            "relative_stream_attribution": self.attribution_label,
            "attribution_code": self.attribution_code,
            "distance_difference_resnet_minus_temporal": self.distance_difference_resnet_minus_temporal,
        }


@dataclass(frozen=True)
class ComparisonResult:
    """Complete digest comparison result."""

    comparison_id: str
    reference_video_id: str
    query_video_id: str
    reference_record_path: Path
    reference_record_checksum: str
    reference_hmac_verification_result: dict[str, Any]
    query_digest_path: Path
    query_digest_checksum: str
    query_manifest_path: Path
    query_manifest_checksum: str
    normalization_id: str
    quantization_id: str
    config: ComparisonConfig
    alignment: SegmentAlignmentResult
    segment_results: tuple[SegmentComparisonResult, ...]
    video_summary: dict[str, Any]
    warnings: tuple[str, ...]
    failures: tuple[str, ...]


def balanced_diagnostic_score(
    resnet_normalized_distance: float,
    temporal_normalized_distance: float,
    weights: DiagnosticWeights = DiagnosticWeights(),
) -> float:
    """Calculate the temporary stream-balanced diagnostic score."""

    weights.validate()
    return float(weights.resnet * resnet_normalized_distance + weights.temporal * temporal_normalized_distance)


def relative_stream_attribution(
    resnet_normalized_distance: float,
    temporal_normalized_distance: float,
    tie_tolerance: float = 1.0e-9,
) -> tuple[str, float]:
    """Return descriptive relative distance attribution without threshold decisions."""

    if tie_tolerance < 0:
        raise DigestComparisonError("tie_tolerance must be non-negative.")
    difference = float(resnet_normalized_distance - temporal_normalized_distance)
    if abs(resnet_normalized_distance) <= tie_tolerance and abs(temporal_normalized_distance) <= tie_tolerance:
        return "no_difference", difference
    if difference > tie_tolerance:
        return "resnet_dominant", difference
    if difference < -tie_tolerance:
        return "temporal_dominant", difference
    return "approximately_equal", difference


def _decode_packed_hex_to_bits(hex_value: str, bit_length: int, bit_order: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        packed_bytes = bytes.fromhex(hex_value)
    except ValueError as exc:
        raise DigestComparisonError("Packed digest hex is malformed.") from exc
    packed = np.frombuffer(packed_bytes, dtype=np.uint8).copy()
    try:
        bits = unpack_packed_bits(packed.reshape(1, -1), bit_length, bit_order)[0]
    except DigestError as exc:
        raise DigestComparisonError(str(exc)) from exc
    return packed, bits.astype(np.uint8)


def _validate_hybrid(resnet_bits: np.ndarray, temporal_bits: np.ndarray, hybrid_bits: np.ndarray, label: str) -> None:
    expected = np.concatenate([resnet_bits, temporal_bits], axis=-1)
    if not np.array_equal(expected, hybrid_bits):
        raise DigestComparisonError(f"{label} hybrid digest does not equal ResNet bits followed by temporal bits.")


def _validate_digest_lengths(lengths: dict[str, int], config: ComparisonConfig) -> None:
    expected = {
        "resnet": config.resnet_bit_length,
        "temporal": config.temporal_bit_length,
        "hybrid": config.hybrid_bit_length,
    }
    if lengths != expected:
        raise DigestComparisonError(f"Digest lengths {lengths} do not match expected {expected}.")


def load_verified_reference_digest(
    reference_id: str,
    authentication_record_root: str | Path,
    key_info: HMACKeyInfo,
    config: ComparisonConfig,
    algorithm: str = "HMAC-SHA-256",
) -> tuple[LoadedDigestRecord, dict[str, Any]]:
    """Load and decode a trusted reference authentication record after HMAC verification."""

    config.validate()
    paths = authentication_record_paths(authentication_record_root, reference_id)
    if not paths.record_path.exists():
        raise DigestComparisonError(f"Reference authentication record not found: {paths.record_path}")
    record = load_authentication_record(paths.record_path)
    verification = verify_authentication_record(record, key_info, algorithm=algorithm)
    verification_dict = verification.to_dict()
    required_ok = (
        verification.schema_valid
        and verification.algorithm_supported
        and verification.payload_checksum_valid
        and verification.key_fingerprint_match
        and verification.hmac_valid
        and verification.record_valid
    )
    if not required_ok:
        raise DigestComparisonError(
            f"Reference authentication record verification failed for {reference_id}: {verification.failure_reason}"
        )
    payload = record["payload"]
    if payload.get("video_id") != reference_id:
        raise DigestComparisonError("Reference payload video_id does not match requested reference ID.")
    digest_lengths = {str(key): int(value) for key, value in payload["digest_lengths"].items()}
    _validate_digest_lengths(digest_lengths, config)
    stream_boundaries = {
        str(stream): {"start": int(values["start"]), "end_exclusive": int(values["end_exclusive"])}
        for stream, values in payload["stream_boundaries"].items()
    }
    if stream_boundaries != STREAM_BOUNDARIES:
        raise DigestComparisonError("Reference stream boundaries do not match Phase 5 boundaries.")
    bit_order = str(payload["bit_order"])
    if bit_order not in {"big", "little"}:
        raise DigestComparisonError(f"Unsupported reference bit order: {bit_order}.")

    segments: list[SegmentDescriptor] = []
    resnet_bits: list[np.ndarray] = []
    temporal_bits: list[np.ndarray] = []
    hybrid_bits: list[np.ndarray] = []
    resnet_packed: list[np.ndarray] = []
    temporal_packed: list[np.ndarray] = []
    hybrid_packed: list[np.ndarray] = []
    for segment in payload["segments"]:
        segments.append(
            SegmentDescriptor(
                segment_id=int(segment["segment_id"]),
                start_time_microseconds=int(segment["start_time_microseconds"]),
                end_time_microseconds=int(segment["end_time_microseconds"]),
            )
        )
        rp, rb = _decode_packed_hex_to_bits(
            str(segment["resnet_packed_digest_hex"]), config.resnet_bit_length, bit_order
        )
        tp, tb = _decode_packed_hex_to_bits(
            str(segment["temporal_packed_digest_hex"]), config.temporal_bit_length, bit_order
        )
        hp, hb = _decode_packed_hex_to_bits(
            str(segment["hybrid_packed_digest_hex"]), config.hybrid_bit_length, bit_order
        )
        _validate_hybrid(rb, tb, hb, "Reference")
        resnet_packed.append(rp)
        temporal_packed.append(tp)
        hybrid_packed.append(hp)
        resnet_bits.append(rb)
        temporal_bits.append(tb)
        hybrid_bits.append(hb)
    if len(segments) != int(payload["segment_count"]):
        raise DigestComparisonError("Reference payload segment_count does not match segment list.")

    return (
        LoadedDigestRecord(
            video_id=reference_id,
            normalization_id=str(payload["normalization_id"]),
            quantization_id=str(payload["quantization_id"]),
            digest_lengths=digest_lengths,
            stream_boundaries=stream_boundaries,
            bit_order=bit_order,
            segments=tuple(segments),
            resnet_bits=np.vstack(resnet_bits).astype(np.uint8) if resnet_bits else np.empty((0, 1024), dtype=np.uint8),
            temporal_bits=np.vstack(temporal_bits).astype(np.uint8)
            if temporal_bits
            else np.empty((0, 36), dtype=np.uint8),
            hybrid_bits=np.vstack(hybrid_bits).astype(np.uint8) if hybrid_bits else np.empty((0, 1060), dtype=np.uint8),
            resnet_packed=np.vstack(resnet_packed).astype(np.uint8)
            if resnet_packed
            else np.empty((0, 128), dtype=np.uint8),
            temporal_packed=np.vstack(temporal_packed).astype(np.uint8)
            if temporal_packed
            else np.empty((0, 5), dtype=np.uint8),
            hybrid_packed=np.vstack(hybrid_packed).astype(np.uint8)
            if hybrid_packed
            else np.empty((0, 133), dtype=np.uint8),
            source_path=paths.record_path.resolve(),
            source_checksum=sha256_file(paths.record_path),
        ),
        verification_dict,
    )


def _seconds_to_microseconds(value: float | int) -> int:
    numeric = float(value)
    if not np.isfinite(numeric):
        raise DigestComparisonError("Query segment timestamp is not finite.")
    return int(round(numeric * 1_000_000))


def _require_binary_matrix(matrix: np.ndarray, expected_width: int, name: str) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.uint8)
    if values.ndim != 2 or values.shape[1] != expected_width:
        raise DigestComparisonError(f"{name} must have shape (segments, {expected_width}), got {values.shape}.")
    if not np.all(np.isin(values, [0, 1])):
        raise DigestComparisonError(f"{name} contains non-binary values.")
    return values.copy()


def _validate_segment_ids(segment_ids: np.ndarray, label: str) -> np.ndarray:
    ids = np.asarray(segment_ids, dtype=np.int64)
    if ids.ndim != 1:
        raise DigestComparisonError(f"{label} segment IDs must be one-dimensional.")
    unique, counts = np.unique(ids, return_counts=True)
    duplicates = unique[counts > 1]
    if duplicates.size:
        raise DigestComparisonError(f"{label} contains duplicate segment IDs: {duplicates.tolist()}.")
    return ids


def load_query_digest(
    query_id: str,
    digest_root: str | Path,
    reference: LoadedDigestRecord,
    config: ComparisonConfig,
) -> LoadedDigestRecord:
    """Load and validate a query digest NPZ/manifest against the trusted reference metadata."""

    config.validate()
    paths = digest_output_paths(digest_root, query_id)
    if not paths.npz_path.exists() or not paths.manifest_path.exists():
        raise DigestComparisonError(f"Query digest outputs not found for {query_id}: {paths.output_dir}")
    with paths.manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("video_id") != query_id:
        raise DigestComparisonError("Query digest manifest video_id does not match requested query ID.")
    npz_checksum = sha256_file(paths.npz_path)
    if manifest.get("npz_sha256") != npz_checksum:
        raise DigestComparisonError(f"Query digest NPZ checksum mismatch for {query_id}.")
    if manifest.get("quantization_id") != reference.quantization_id:
        raise DigestComparisonError("Query quantization ID does not match reference.")
    if manifest.get("normalization_calibration_id") != reference.normalization_id:
        raise DigestComparisonError("Query normalization ID does not match reference.")
    digest_lengths = {str(key): int(value) for key, value in manifest["digest_dimensions"].items()}
    if digest_lengths != reference.digest_lengths:
        raise DigestComparisonError("Query digest dimensions do not match reference.")
    _validate_digest_lengths(digest_lengths, config)
    stream_boundaries = {
        str(stream): {"start": int(values["start"]), "end_exclusive": int(values["end_exclusive"])}
        for stream, values in manifest["stream_boundaries"].items()
    }
    if stream_boundaries != reference.stream_boundaries:
        raise DigestComparisonError("Query stream boundaries do not match reference.")
    bit_order = str(manifest["bit_order"])
    if bit_order != reference.bit_order:
        raise DigestComparisonError("Query bit order does not match reference.")

    arrays = load_digest_npz(paths.npz_path)
    segment_ids = _validate_segment_ids(arrays["segment_ids"], "Query")
    start_times = np.asarray(arrays["segment_start_times"], dtype=np.float64)
    end_times = np.asarray(arrays["segment_end_times"], dtype=np.float64)
    if start_times.shape != segment_ids.shape or end_times.shape != segment_ids.shape:
        raise DigestComparisonError("Query segment timestamps must match segment_ids shape.")
    resnet_bits = _require_binary_matrix(arrays["resnet_binary_digests"], config.resnet_bit_length, "Query ResNet")
    temporal_bits = _require_binary_matrix(
        arrays["temporal_binary_digests"], config.temporal_bit_length, "Query temporal"
    )
    hybrid_bits = _require_binary_matrix(arrays["hybrid_binary_digests"], config.hybrid_bit_length, "Query hybrid")
    _validate_hybrid(resnet_bits, temporal_bits, hybrid_bits, "Query")
    if resnet_bits.shape[0] != segment_ids.shape[0] or temporal_bits.shape[0] != segment_ids.shape[0]:
        raise DigestComparisonError("Query digest row counts do not match segment IDs.")

    resnet_length = int(np.asarray(arrays["resnet_bit_length"]).reshape(-1)[0])
    temporal_length = int(np.asarray(arrays["temporal_bit_length"]).reshape(-1)[0])
    hybrid_length = int(np.asarray(arrays["hybrid_bit_length"]).reshape(-1)[0])
    if (resnet_length, temporal_length, hybrid_length) != (
        config.resnet_bit_length,
        config.temporal_bit_length,
        config.hybrid_bit_length,
    ):
        raise DigestComparisonError("Query stored bit lengths do not match comparison configuration.")
    try:
        if not np.array_equal(
            unpack_packed_bits(arrays["resnet_packed_digests"], resnet_length, bit_order),
            resnet_bits,
        ):
            raise DigestComparisonError("Query packed ResNet digest does not match unpacked bits.")
        if not np.array_equal(
            unpack_packed_bits(arrays["temporal_packed_digests"], temporal_length, bit_order),
            temporal_bits,
        ):
            raise DigestComparisonError("Query packed temporal digest does not match unpacked bits.")
        if not np.array_equal(
            unpack_packed_bits(arrays["hybrid_packed_digests"], hybrid_length, bit_order),
            hybrid_bits,
        ):
            raise DigestComparisonError("Query packed hybrid digest does not match unpacked bits.")
    except DigestError as exc:
        raise DigestComparisonError(str(exc)) from exc

    segments = tuple(
        SegmentDescriptor(
            segment_id=int(segment_id),
            start_time_microseconds=_seconds_to_microseconds(start),
            end_time_microseconds=_seconds_to_microseconds(end),
        )
        for segment_id, start, end in zip(segment_ids, start_times, end_times, strict=True)
    )
    return LoadedDigestRecord(
        video_id=query_id,
        normalization_id=str(manifest["normalization_calibration_id"]),
        quantization_id=str(manifest["quantization_id"]),
        digest_lengths=digest_lengths,
        stream_boundaries=stream_boundaries,
        bit_order=bit_order,
        segments=segments,
        resnet_bits=resnet_bits,
        temporal_bits=temporal_bits,
        hybrid_bits=hybrid_bits,
        resnet_packed=np.asarray(arrays["resnet_packed_digests"], dtype=np.uint8).copy(),
        temporal_packed=np.asarray(arrays["temporal_packed_digests"], dtype=np.uint8).copy(),
        hybrid_packed=np.asarray(arrays["hybrid_packed_digests"], dtype=np.uint8).copy(),
        source_path=paths.npz_path.resolve(),
        source_checksum=npz_checksum,
        manifest_path=paths.manifest_path.resolve(),
        manifest_checksum=sha256_file(paths.manifest_path),
    )


def _summary_value(values: list[float], fn: str) -> float | None:
    if not values:
        return None
    return float(np.mean(values)) if fn == "mean" else float(np.max(values))


def _segment_id_for_max_distance(results: tuple[SegmentComparisonResult, ...], stream: str) -> int | None:
    if not results:
        return None
    if stream == "resnet":
        return int(max(results, key=lambda result: result.resnet.normalized_distance).segment_id)
    if stream == "temporal":
        return int(max(results, key=lambda result: result.temporal.normalized_distance).segment_id)
    raise DigestComparisonError(f"Unknown stream for maximum segment lookup: {stream}")


def _segment_id_for_max_score(results: tuple[SegmentComparisonResult, ...]) -> int | None:
    if not results:
        return None
    return int(max(results, key=lambda result: result.development_diagnostic_score).segment_id)


def _build_video_summary(
    alignment: SegmentAlignmentResult,
    segment_results: tuple[SegmentComparisonResult, ...],
) -> dict[str, Any]:
    resnet = [result.resnet.normalized_distance for result in segment_results]
    temporal = [result.temporal.normalized_distance for result in segment_results]
    hybrid = [result.hybrid.normalized_distance for result in segment_results]
    balanced = [result.development_diagnostic_score for result in segment_results]
    attribution_counts = Counter(result.attribution_label for result in segment_results)
    return {
        "reference_segment_count": alignment.reference_segment_count,
        "query_segment_count": alignment.query_segment_count,
        "matched_segment_count": alignment.matched_segment_count,
        "missing_segment_count": alignment.missing_segment_count,
        "extra_segment_count": alignment.extra_segment_count,
        "timestamp_mismatch_count": alignment.timestamp_mismatch_count,
        "mean_resnet_normalized_distance": _summary_value(resnet, "mean"),
        "maximum_resnet_normalized_distance": _summary_value(resnet, "max"),
        "mean_temporal_normalized_distance": _summary_value(temporal, "mean"),
        "maximum_temporal_normalized_distance": _summary_value(temporal, "max"),
        "mean_flat_hybrid_normalized_distance": _summary_value(hybrid, "mean"),
        "maximum_flat_hybrid_normalized_distance": _summary_value(hybrid, "max"),
        "mean_balanced_diagnostic_score": _summary_value(balanced, "mean"),
        "maximum_balanced_diagnostic_score": _summary_value(balanced, "max"),
        "segment_id_with_maximum_resnet_distance": _segment_id_for_max_distance(segment_results, "resnet"),
        "segment_id_with_maximum_temporal_distance": _segment_id_for_max_distance(segment_results, "temporal"),
        "segment_id_with_maximum_balanced_diagnostic_score": _segment_id_for_max_score(segment_results),
        "attribution_counts": {label: int(attribution_counts.get(label, 0)) for label in ATTRIBUTION_CODES},
        "comparison_complete": alignment.comparison_complete,
        "alignment_valid": alignment.alignment_valid,
        "distances_computed": bool(segment_results),
    }


def compare_digests(
    reference_id: str,
    query_id: str,
    authentication_record_root: str | Path,
    digest_root: str | Path,
    key_info: HMACKeyInfo,
    config: ComparisonConfig,
    algorithm: str = "HMAC-SHA-256",
) -> ComparisonResult:
    """Verify the reference record, load the query digest, align segments, and compute distances."""

    config.validate()
    reference, verification = load_verified_reference_digest(
        reference_id,
        authentication_record_root,
        key_info,
        config,
        algorithm=algorithm,
    )
    query = load_query_digest(query_id, digest_root, reference, config)
    alignment = align_segments(
        list(reference.segments),
        list(query.segments),
        timestamp_tolerance_microseconds=config.timestamp_tolerance_microseconds,
        alignment_mode=config.segment_alignment,
    )
    segment_results: list[SegmentComparisonResult] = []
    failures: list[str] = []
    for segment_id, ref_index, query_index in alignment.matched_pairs:
        resnet = hamming_distance(reference.resnet_bits[ref_index], query.resnet_bits[query_index])
        temporal = hamming_distance(reference.temporal_bits[ref_index], query.temporal_bits[query_index])
        hybrid = hamming_distance(reference.hybrid_bits[ref_index], query.hybrid_bits[query_index])
        if hybrid.raw_distance != resnet.raw_distance + temporal.raw_distance:
            raise DigestComparisonError("Hybrid raw distance does not equal ResNet plus temporal distance.")
        score = balanced_diagnostic_score(
            resnet.normalized_distance,
            temporal.normalized_distance,
            config.diagnostic_weights,
        )
        label, difference = relative_stream_attribution(
            resnet.normalized_distance,
            temporal.normalized_distance,
            tie_tolerance=config.tie_tolerance,
        )
        ref_segment = reference.segments[ref_index]
        segment_results.append(
            SegmentComparisonResult(
                segment_id=segment_id,
                reference_index=ref_index,
                query_index=query_index,
                start_time_microseconds=ref_segment.start_time_microseconds,
                end_time_microseconds=ref_segment.end_time_microseconds,
                resnet=resnet,
                temporal=temporal,
                hybrid=hybrid,
                development_diagnostic_score=score,
                flat_hybrid_normalized_distance=hybrid.normalized_distance,
                attribution_label=label,
                attribution_code=ATTRIBUTION_CODES[label],
                distance_difference_resnet_minus_temporal=difference,
            )
        )
    if not alignment.alignment_valid:
        failures.append("Structural alignment findings were recorded; strict comparisons are incomplete.")
    segment_tuple = tuple(segment_results)
    return ComparisonResult(
        comparison_id=f"{reference_id}__vs__{query_id}",
        reference_video_id=reference_id,
        query_video_id=query_id,
        reference_record_path=reference.source_path,
        reference_record_checksum=reference.source_checksum,
        reference_hmac_verification_result=verification,
        query_digest_path=query.source_path,
        query_digest_checksum=query.source_checksum,
        query_manifest_path=query.manifest_path or Path(),
        query_manifest_checksum=query.manifest_checksum or "",
        normalization_id=reference.normalization_id,
        quantization_id=reference.quantization_id,
        config=config,
        alignment=alignment,
        segment_results=segment_tuple,
        video_summary=_build_video_summary(alignment, segment_tuple),
        warnings=(NO_THRESHOLD_WARNING,),
        failures=tuple(failures),
    )
