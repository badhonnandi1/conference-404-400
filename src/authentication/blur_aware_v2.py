
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.authentication.canonicalization import canonical_json_bytes, canonical_payload_sha256
from src.authentication.digest import pack_bit_matrix, unpack_packed_bits
from src.authentication.hmac_auth import (
    DEFAULT_ALGORITHM,
    HMACKeyInfo,
    compute_hmac_sha256_hex,
    verify_hmac_sha256_hex,
)
from src.authentication.quantization import GRAY_CODE_MAPPING, QuantizationError, validate_normalized_matrix
from src.features.alignment import RESNET_SEGMENT_DIMENSION, TEMPORAL_SEGMENT_DIMENSION
from src.features.normalization import RobustNormalizer
from src.features.spatial_quality import SPATIAL_SEGMENT_DIMENSION
from src.verification.hamming import hamming_distance


PIPELINE_V2_ID = "PIPELINE_V2_BLUR_AWARE"
V2_SCHEMA_VERSION = 2
V2_QUANTIZATION_VERSION = "blur_aware_quantizer_v2"
V2_NORMALIZATION_VERSION = "three_stream_normalization_v2"
V2_RESNET_DIGEST_LENGTH = RESNET_SEGMENT_DIMENSION
V2_TEMPORAL_BITS_PER_FEATURE = 2
V2_SPATIAL_BITS_PER_FEATURE = 2
V2_TEMPORAL_DIGEST_LENGTH = TEMPORAL_SEGMENT_DIMENSION * V2_TEMPORAL_BITS_PER_FEATURE
V2_SPATIAL_DIGEST_LENGTH = SPATIAL_SEGMENT_DIMENSION * V2_SPATIAL_BITS_PER_FEATURE
V2_HYBRID_DIGEST_LENGTH = V2_RESNET_DIGEST_LENGTH + V2_TEMPORAL_DIGEST_LENGTH + V2_SPATIAL_DIGEST_LENGTH
V2_CONTINUOUS_DIMENSIONS = {
    "resnet": V2_RESNET_DIGEST_LENGTH,
    "temporal": TEMPORAL_SEGMENT_DIMENSION,
    "spatial": SPATIAL_SEGMENT_DIMENSION,
    "combined": V2_RESNET_DIGEST_LENGTH + TEMPORAL_SEGMENT_DIMENSION + SPATIAL_SEGMENT_DIMENSION,
}
V2_DIGEST_LENGTHS = {
    "resnet": V2_RESNET_DIGEST_LENGTH,
    "temporal": V2_TEMPORAL_DIGEST_LENGTH,
    "spatial": V2_SPATIAL_DIGEST_LENGTH,
    "hybrid": V2_HYBRID_DIGEST_LENGTH,
}
V2_STREAM_BOUNDARIES = {
    "resnet": {"start": 0, "end_exclusive": V2_RESNET_DIGEST_LENGTH},
    "temporal": {
        "start": V2_RESNET_DIGEST_LENGTH,
        "end_exclusive": V2_RESNET_DIGEST_LENGTH + V2_TEMPORAL_DIGEST_LENGTH,
    },
    "spatial": {
        "start": V2_RESNET_DIGEST_LENGTH + V2_TEMPORAL_DIGEST_LENGTH,
        "end_exclusive": V2_HYBRID_DIGEST_LENGTH,
    },
}
DEFAULT_V2_WEIGHTS = {"resnet": 0.4, "temporal": 0.3, "spatial": 0.3}
SPATIAL_SHARPNESS_COLUMNS = tuple(range(SPATIAL_SEGMENT_DIMENSION))


class BlurAwareV2Error(RuntimeError):
    """Raised when V2 digest or comparison helpers fail."""


@dataclass(frozen=True)
class ThreeStreamNormalizers:
    """Fold-specific robust normalizers for V2 streams."""

    normalization_id: str
    resnet: RobustNormalizer
    temporal: RobustNormalizer
    spatial: RobustNormalizer
    training_video_ids: tuple[str, ...]
    training_source_ids: tuple[str, ...]

    def to_manifest(self) -> dict[str, Any]:
        """Return JSON-friendly fold normalization metadata."""

        return {
            "normalization_id": self.normalization_id,
            "version": V2_NORMALIZATION_VERSION,
            "pipeline_id": PIPELINE_V2_ID,
            "training_video_ids": list(self.training_video_ids),
            "training_source_ids": list(self.training_source_ids),
            "feature_dimensions": V2_CONTINUOUS_DIMENSIONS,
            "normalizers": {
                "resnet": self.resnet.summary(),
                "temporal": self.temporal.summary(),
                "spatial": self.spatial.summary(),
            },
        }


@dataclass(frozen=True)
class V2QuantizationParameters:
    """Fold-specific V2 thresholds for all three normalized streams."""

    quantization_id: str
    normalization_id: str
    resnet_thresholds: np.ndarray
    temporal_q1_thresholds: np.ndarray
    temporal_median_thresholds: np.ndarray
    temporal_q3_thresholds: np.ndarray
    spatial_q1_thresholds: np.ndarray
    spatial_median_thresholds: np.ndarray
    spatial_q3_thresholds: np.ndarray
    gray_code_table: np.ndarray
    bit_order: str = "big"
    version: str = V2_QUANTIZATION_VERSION

    def validate(self) -> None:
        """Validate threshold dimensions and binary coding table."""

        if self.resnet_thresholds.shape != (V2_RESNET_DIGEST_LENGTH,):
            raise BlurAwareV2Error(f"ResNet thresholds must be {V2_RESNET_DIGEST_LENGTH}-D.")
        for name, values, dimension in (
            ("temporal_q1_thresholds", self.temporal_q1_thresholds, TEMPORAL_SEGMENT_DIMENSION),
            ("temporal_median_thresholds", self.temporal_median_thresholds, TEMPORAL_SEGMENT_DIMENSION),
            ("temporal_q3_thresholds", self.temporal_q3_thresholds, TEMPORAL_SEGMENT_DIMENSION),
            ("spatial_q1_thresholds", self.spatial_q1_thresholds, SPATIAL_SEGMENT_DIMENSION),
            ("spatial_median_thresholds", self.spatial_median_thresholds, SPATIAL_SEGMENT_DIMENSION),
            ("spatial_q3_thresholds", self.spatial_q3_thresholds, SPATIAL_SEGMENT_DIMENSION),
        ):
            if values.shape != (dimension,):
                raise BlurAwareV2Error(f"{name} must have shape ({dimension},), got {values.shape}.")
            if not np.all(np.isfinite(values)):
                raise BlurAwareV2Error(f"{name} contains non-finite values.")
        if not np.all(np.isfinite(self.resnet_thresholds)):
            raise BlurAwareV2Error("ResNet thresholds contain non-finite values.")
        if self.gray_code_table.shape != (4, 2) or not np.all(np.isin(self.gray_code_table, [0, 1])):
            raise BlurAwareV2Error("Gray-code table must have shape (4, 2) and binary values.")
        if self.bit_order not in {"big", "little"}:
            raise BlurAwareV2Error("bit_order must be 'big' or 'little'.")

    def to_manifest(self) -> dict[str, Any]:
        """Return JSON-friendly V2 quantization metadata."""

        self.validate()
        return {
            "quantization_id": self.quantization_id,
            "normalization_id": self.normalization_id,
            "version": self.version,
            "pipeline_id": PIPELINE_V2_ID,
            "continuous_dimensions": V2_CONTINUOUS_DIMENSIONS,
            "digest_lengths": V2_DIGEST_LENGTHS,
            "stream_boundaries": V2_STREAM_BOUNDARIES,
            "resnet_quantization_method": "median_binary",
            "temporal_quantization_method": "quartile_gray_code",
            "spatial_quantization_method": "quartile_gray_code",
            "gray_code_mapping": {
                f"bin_{index}": "".join(str(bit) for bit in self.gray_code_table[index])
                for index in range(4)
            },
            "bit_order": self.bit_order,
        }


@dataclass(frozen=True)
class V2DigestBundle:
    """Unpacked and packed V2 digests for one video."""

    video_id: str
    segment_ids: np.ndarray
    segment_start_times: np.ndarray
    segment_end_times: np.ndarray
    resnet_binary_digests: np.ndarray
    temporal_bin_indices: np.ndarray
    temporal_binary_digests: np.ndarray
    spatial_bin_indices: np.ndarray
    spatial_binary_digests: np.ndarray
    hybrid_binary_digests: np.ndarray
    resnet_packed_digests: np.ndarray
    temporal_packed_digests: np.ndarray
    spatial_packed_digests: np.ndarray
    hybrid_packed_digests: np.ndarray
    bit_order: str
    resnet_bit_length: int = V2_RESNET_DIGEST_LENGTH
    temporal_bit_length: int = V2_TEMPORAL_DIGEST_LENGTH
    spatial_bit_length: int = V2_SPATIAL_DIGEST_LENGTH
    hybrid_bit_length: int = V2_HYBRID_DIGEST_LENGTH

    def validate_round_trips(self) -> bool:
        """Return whether packed digests unpack exactly."""

        return (
            np.array_equal(unpack_packed_bits(self.resnet_packed_digests, self.resnet_bit_length, self.bit_order), self.resnet_binary_digests)
            and np.array_equal(unpack_packed_bits(self.temporal_packed_digests, self.temporal_bit_length, self.bit_order), self.temporal_binary_digests)
            and np.array_equal(unpack_packed_bits(self.spatial_packed_digests, self.spatial_bit_length, self.bit_order), self.spatial_binary_digests)
            and np.array_equal(unpack_packed_bits(self.hybrid_packed_digests, self.hybrid_bit_length, self.bit_order), self.hybrid_binary_digests)
        )


@dataclass(frozen=True)
class V2SegmentComparison:
    """V2 segment-level distance and attribution values."""

    segment_id: int
    reference_index: int
    query_index: int
    resnet_raw_distance: int
    temporal_raw_distance: int
    spatial_raw_distance: int
    hybrid_raw_distance: int
    resnet_normalized_distance: float
    temporal_normalized_distance: float
    spatial_normalized_distance: float
    hybrid_normalized_distance: float
    weighted_score: float
    blur_loss: float
    stream_attribution: str


def fit_three_stream_normalizers(
    *,
    normalization_id: str,
    training_video_ids: list[str],
    training_source_ids: list[str],
    resnet_features: np.ndarray,
    temporal_features: np.ndarray,
    spatial_features: np.ndarray,
) -> ThreeStreamNormalizers:
    """Fit V2 robust normalizers from training benign/reference samples."""

    return ThreeStreamNormalizers(
        normalization_id=normalization_id,
        resnet=RobustNormalizer.fit(resnet_features),
        temporal=RobustNormalizer.fit(temporal_features),
        spatial=RobustNormalizer.fit(spatial_features),
        training_video_ids=tuple(training_video_ids),
        training_source_ids=tuple(training_source_ids),
    )


def transform_three_streams(
    normalizers: ThreeStreamNormalizers,
    resnet_features: np.ndarray,
    temporal_features: np.ndarray,
    spatial_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalize all V2 streams and return combined continuous features."""

    resnet = normalizers.resnet.transform(resnet_features)
    temporal = normalizers.temporal.transform(temporal_features)
    spatial = normalizers.spatial.transform(spatial_features)
    combined = np.concatenate([resnet, temporal, spatial], axis=1).astype(np.float32)
    expected = V2_CONTINUOUS_DIMENSIONS["combined"]
    if combined.ndim != 2 or combined.shape[1] != expected:
        raise BlurAwareV2Error(f"Combined V2 feature dimension must be {expected}, got {combined.shape}.")
    return resnet, temporal, spatial, combined


def derive_v2_quantization_parameters(
    normalizers: ThreeStreamNormalizers,
    quantization_id: str,
    bit_order: str = "big",
) -> V2QuantizationParameters:
    """Derive V2 quantizer thresholds from fold normalizers."""

    gray_table = np.asarray([GRAY_CODE_MAPPING[index] for index in range(4)], dtype=np.uint8)

    def normalized_quantiles(normalizer: RobustNormalizer) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            ((normalizer.q1 - normalizer.median) / normalizer.safe_scale).astype(np.float64),
            np.zeros(normalizer.feature_dimension, dtype=np.float64),
            ((normalizer.q3 - normalizer.median) / normalizer.safe_scale).astype(np.float64),
        )

    temporal_q1, temporal_median, temporal_q3 = normalized_quantiles(normalizers.temporal)
    spatial_q1, spatial_median, spatial_q3 = normalized_quantiles(normalizers.spatial)
    params = V2QuantizationParameters(
        quantization_id=quantization_id,
        normalization_id=normalizers.normalization_id,
        resnet_thresholds=np.zeros(V2_RESNET_DIGEST_LENGTH, dtype=np.float64),
        temporal_q1_thresholds=temporal_q1,
        temporal_median_thresholds=temporal_median,
        temporal_q3_thresholds=temporal_q3,
        spatial_q1_thresholds=spatial_q1,
        spatial_median_thresholds=spatial_median,
        spatial_q3_thresholds=spatial_q3,
        gray_code_table=gray_table,
        bit_order=bit_order,
    )
    params.validate()
    return params


def quantize_resnet_v2(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Quantize normalized ResNet values to one bit per feature."""

    matrix = validate_normalized_matrix(values, V2_RESNET_DIGEST_LENGTH, "V2 ResNet")
    threshold = np.asarray(thresholds, dtype=np.float32).reshape(1, -1)
    if threshold.shape != (1, V2_RESNET_DIGEST_LENGTH):
        raise QuantizationError("V2 ResNet threshold shape is invalid.")
    return (matrix >= threshold).astype(np.uint8)


def assign_quartile_bins(
    values: np.ndarray,
    q1_thresholds: np.ndarray,
    median_thresholds: np.ndarray,
    q3_thresholds: np.ndarray,
    expected_dimension: int,
    stream_name: str,
) -> np.ndarray:
    """Assign normalized stream features to quartile bins 0 through 3."""

    matrix = validate_normalized_matrix(values, expected_dimension, stream_name)
    q1 = np.asarray(q1_thresholds, dtype=np.float32).reshape(1, -1)
    median = np.asarray(median_thresholds, dtype=np.float32).reshape(1, -1)
    q3 = np.asarray(q3_thresholds, dtype=np.float32).reshape(1, -1)
    for name, threshold in (("q1", q1), ("median", median), ("q3", q3)):
        if threshold.shape != (1, expected_dimension):
            raise QuantizationError(f"{stream_name} {name} threshold shape is invalid: {threshold.shape}.")
    bins = np.zeros(matrix.shape, dtype=np.uint8)
    bins[matrix >= q1] = 1
    bins[matrix >= median] = 2
    bins[matrix >= q3] = 3
    return bins


def gray_encode_bins(bin_indices: np.ndarray, gray_code_table: np.ndarray) -> np.ndarray:
    """Gray-code encode any 2D quartile-bin matrix."""

    bins = np.asarray(bin_indices, dtype=np.uint8)
    if bins.ndim != 2 or not np.all(np.isin(bins, [0, 1, 2, 3])):
        raise QuantizationError("Quartile bin indices must be a 2D matrix containing only 0, 1, 2, or 3.")
    table = np.asarray(gray_code_table, dtype=np.uint8)
    if table.shape != (4, 2) or not np.all(np.isin(table, [0, 1])):
        raise QuantizationError("Gray-code table must have shape (4, 2) and binary values.")
    return table[bins.astype(np.int64)].reshape(bins.shape[0], bins.shape[1] * 2).astype(np.uint8)


def build_v2_hybrid_digest(resnet_bits: np.ndarray, temporal_bits: np.ndarray, spatial_bits: np.ndarray) -> np.ndarray:
    """Concatenate V2 streams in ResNet, temporal, spatial order."""

    resnet = np.asarray(resnet_bits, dtype=np.uint8)
    temporal = np.asarray(temporal_bits, dtype=np.uint8)
    spatial = np.asarray(spatial_bits, dtype=np.uint8)
    if resnet.ndim != 2 or resnet.shape[1] != V2_RESNET_DIGEST_LENGTH:
        raise BlurAwareV2Error("V2 ResNet bits have invalid shape.")
    if temporal.ndim != 2 or temporal.shape[1] != V2_TEMPORAL_DIGEST_LENGTH:
        raise BlurAwareV2Error("V2 temporal bits have invalid shape.")
    if spatial.ndim != 2 or spatial.shape[1] != V2_SPATIAL_DIGEST_LENGTH:
        raise BlurAwareV2Error("V2 spatial bits have invalid shape.")
    if len({resnet.shape[0], temporal.shape[0], spatial.shape[0]}) != 1:
        raise BlurAwareV2Error("V2 digest streams must have matching segment counts.")
    hybrid = np.concatenate([resnet, temporal, spatial], axis=1).astype(np.uint8)
    if hybrid.shape[1] != V2_HYBRID_DIGEST_LENGTH:
        raise BlurAwareV2Error("Unexpected V2 hybrid digest length.")
    return hybrid


def build_v2_digest_bundle(
    *,
    video_id: str,
    segment_ids: np.ndarray,
    segment_start_times: np.ndarray,
    segment_end_times: np.ndarray,
    resnet_normalized_features: np.ndarray,
    temporal_normalized_features: np.ndarray,
    spatial_normalized_features: np.ndarray,
    parameters: V2QuantizationParameters,
) -> V2DigestBundle:
    """Build unpacked and packed V2 digests for one video."""

    parameters.validate()
    resnet_bits = quantize_resnet_v2(resnet_normalized_features, parameters.resnet_thresholds)
    temporal_bins = assign_quartile_bins(
        temporal_normalized_features,
        parameters.temporal_q1_thresholds,
        parameters.temporal_median_thresholds,
        parameters.temporal_q3_thresholds,
        TEMPORAL_SEGMENT_DIMENSION,
        "V2 temporal",
    )
    spatial_bins = assign_quartile_bins(
        spatial_normalized_features,
        parameters.spatial_q1_thresholds,
        parameters.spatial_median_thresholds,
        parameters.spatial_q3_thresholds,
        SPATIAL_SEGMENT_DIMENSION,
        "V2 spatial",
    )
    temporal_bits = gray_encode_bins(temporal_bins, parameters.gray_code_table)
    spatial_bits = gray_encode_bins(spatial_bins, parameters.gray_code_table)
    hybrid_bits = build_v2_hybrid_digest(resnet_bits, temporal_bits, spatial_bits)
    resnet_packed, _, _ = pack_bit_matrix(resnet_bits, parameters.bit_order)
    temporal_packed, _, _ = pack_bit_matrix(temporal_bits, parameters.bit_order)
    spatial_packed, _, _ = pack_bit_matrix(spatial_bits, parameters.bit_order)
    hybrid_packed, _, _ = pack_bit_matrix(hybrid_bits, parameters.bit_order)
    bundle = V2DigestBundle(
        video_id=video_id,
        segment_ids=np.asarray(segment_ids, dtype=np.int64).copy(),
        segment_start_times=np.asarray(segment_start_times, dtype=np.float64).copy(),
        segment_end_times=np.asarray(segment_end_times, dtype=np.float64).copy(),
        resnet_binary_digests=resnet_bits,
        temporal_bin_indices=temporal_bins,
        temporal_binary_digests=temporal_bits,
        spatial_bin_indices=spatial_bins,
        spatial_binary_digests=spatial_bits,
        hybrid_binary_digests=hybrid_bits,
        resnet_packed_digests=resnet_packed,
        temporal_packed_digests=temporal_packed,
        spatial_packed_digests=spatial_packed,
        hybrid_packed_digests=hybrid_packed,
        bit_order=parameters.bit_order,
    )
    if not bundle.validate_round_trips():
        raise BlurAwareV2Error("V2 digest pack/unpack round-trip failed.")
    return bundle


def spatial_blur_loss(reference_spatial_features: np.ndarray, query_spatial_features: np.ndarray, epsilon: float = 1.0e-8) -> np.ndarray:
    """Return per-segment positive sharpness loss from reference to query."""

    reference = np.asarray(reference_spatial_features, dtype=np.float32)
    query = np.asarray(query_spatial_features, dtype=np.float32)
    if reference.shape != query.shape or reference.ndim != 2 or reference.shape[1] != SPATIAL_SEGMENT_DIMENSION:
        raise BlurAwareV2Error(f"Spatial blur-loss arrays must both be (segments, {SPATIAL_SEGMENT_DIMENSION}).")
    columns = np.asarray(SPATIAL_SHARPNESS_COLUMNS, dtype=np.int64)
    ref = np.maximum(reference[:, columns], 0.0)
    qry = np.maximum(query[:, columns], 0.0)
    loss = np.maximum(ref - qry, 0.0) / np.maximum(np.abs(ref), epsilon)
    values = np.mean(loss, axis=1).astype(np.float32)
    if not np.all(np.isfinite(values)):
        raise BlurAwareV2Error("Spatial blur-loss produced non-finite values.")
    return values


def weighted_v2_score(
    resnet_normalized_distance: float,
    temporal_normalized_distance: float,
    spatial_normalized_distance: float,
    weights: dict[str, float] | None = None,
) -> float:
    """Calculate a stream-weighted V2 diagnostic score."""

    weights = DEFAULT_V2_WEIGHTS if weights is None else weights
    total = float(weights["resnet"] + weights["temporal"] + weights["spatial"])
    if total <= 0:
        raise BlurAwareV2Error("V2 stream weights must sum to a positive value.")
    return float(
        (weights["resnet"] * resnet_normalized_distance
        + weights["temporal"] * temporal_normalized_distance
        + weights["spatial"] * spatial_normalized_distance)
        / total
    )


def stream_level_attribution(
    resnet_normalized_distance: float,
    temporal_normalized_distance: float,
    spatial_normalized_distance: float,
    blur_loss_value: float,
    tie_tolerance: float = 1.0e-9,
) -> str:
    """Return the dominant V2 stream-level attribution label."""

    values = {
        "resnet_dominant": float(resnet_normalized_distance),
        "temporal_dominant": float(temporal_normalized_distance),
        "spatial_quality_dominant": float(spatial_normalized_distance),
        "blur_loss_dominant": float(blur_loss_value),
    }
    if max(abs(value) for value in values.values()) <= tie_tolerance:
        return "no_difference"
    ordered = sorted(values.items(), key=lambda item: (-item[1], item[0]))
    if len(ordered) > 1 and abs(ordered[0][1] - ordered[1][1]) <= tie_tolerance:
        return "approximately_equal"
    return ordered[0][0]


def compare_v2_digest_bundles(
    reference: V2DigestBundle,
    query: V2DigestBundle,
    reference_spatial_raw: np.ndarray,
    query_spatial_raw: np.ndarray,
    weights: dict[str, float] | None = None,
) -> list[V2SegmentComparison]:
    """Compare V2 digests on common segment IDs and validate hybrid distance invariant."""

    ref_index = {int(segment_id): index for index, segment_id in enumerate(reference.segment_ids.tolist())}
    query_index = {int(segment_id): index for index, segment_id in enumerate(query.segment_ids.tolist())}
    common_ids = sorted(set(ref_index) & set(query_index))
    comparisons: list[V2SegmentComparison] = []
    for segment_id in common_ids:
        ri = ref_index[segment_id]
        qi = query_index[segment_id]
        resnet = hamming_distance(reference.resnet_binary_digests[ri], query.resnet_binary_digests[qi])
        temporal = hamming_distance(reference.temporal_binary_digests[ri], query.temporal_binary_digests[qi])
        spatial = hamming_distance(reference.spatial_binary_digests[ri], query.spatial_binary_digests[qi])
        hybrid = hamming_distance(reference.hybrid_binary_digests[ri], query.hybrid_binary_digests[qi])
        if hybrid.raw_distance != resnet.raw_distance + temporal.raw_distance + spatial.raw_distance:
            raise BlurAwareV2Error("V2 hybrid raw distance does not equal ResNet plus temporal plus spatial.")
        blur_loss = float(spatial_blur_loss(reference_spatial_raw[ri : ri + 1], query_spatial_raw[qi : qi + 1])[0])
        score = weighted_v2_score(
            resnet.normalized_distance,
            temporal.normalized_distance,
            spatial.normalized_distance,
            weights,
        )
        comparisons.append(
            V2SegmentComparison(
                segment_id=segment_id,
                reference_index=ri,
                query_index=qi,
                resnet_raw_distance=resnet.raw_distance,
                temporal_raw_distance=temporal.raw_distance,
                spatial_raw_distance=spatial.raw_distance,
                hybrid_raw_distance=hybrid.raw_distance,
                resnet_normalized_distance=resnet.normalized_distance,
                temporal_normalized_distance=temporal.normalized_distance,
                spatial_normalized_distance=spatial.normalized_distance,
                hybrid_normalized_distance=hybrid.normalized_distance,
                weighted_score=score,
                blur_loss=blur_loss,
                stream_attribution=stream_level_attribution(
                    resnet.normalized_distance,
                    temporal.normalized_distance,
                    spatial.normalized_distance,
                    blur_loss,
                ),
            )
        )
    return comparisons


def v2_digest_payload(
    *,
    bundle: V2DigestBundle,
    normalization_id: str,
    quantization_id: str,
    source_video_sha256: str,
) -> dict[str, Any]:
    """Build a canonical V2 authentication payload from a digest bundle."""

    segments = []
    for index, segment_id in enumerate(bundle.segment_ids.tolist()):
        segments.append(
            {
                "segment_id": int(segment_id),
                "start_time_microseconds": int(round(float(bundle.segment_start_times[index]) * 1_000_000)),
                "end_time_microseconds": int(round(float(bundle.segment_end_times[index]) * 1_000_000)),
                "resnet_packed_digest_hex": np.asarray(bundle.resnet_packed_digests[index], dtype=np.uint8).tobytes().hex(),
                "temporal_packed_digest_hex": np.asarray(bundle.temporal_packed_digests[index], dtype=np.uint8).tobytes().hex(),
                "spatial_packed_digest_hex": np.asarray(bundle.spatial_packed_digests[index], dtype=np.uint8).tobytes().hex(),
                "hybrid_packed_digest_hex": np.asarray(bundle.hybrid_packed_digests[index], dtype=np.uint8).tobytes().hex(),
            }
        )
    return {
        "schema_version": V2_SCHEMA_VERSION,
        "pipeline_id": PIPELINE_V2_ID,
        "video_id": bundle.video_id,
        "normalization_id": normalization_id,
        "quantization_id": quantization_id,
        "source_video_sha256": source_video_sha256,
        "segment_count": int(bundle.segment_ids.shape[0]),
        "continuous_dimensions": V2_CONTINUOUS_DIMENSIONS,
        "digest_lengths": V2_DIGEST_LENGTHS,
        "stream_boundaries": V2_STREAM_BOUNDARIES,
        "bit_order": bundle.bit_order,
        "segments": segments,
        "development_only": False,
    }


def build_v2_authentication_record(payload: dict[str, Any], key_info: HMACKeyInfo) -> dict[str, Any]:
    """Build an HMAC-SHA-256 protected V2 authentication record."""

    tag = compute_hmac_sha256_hex(key_info.key, canonical_json_bytes(payload))
    return {
        "payload": payload,
        "authentication": {
            "record_schema_version": V2_SCHEMA_VERSION,
            "algorithm": DEFAULT_ALGORITHM,
            "key_id": key_info.key_id,
            "key_fingerprint": key_info.key_fingerprint,
            "key_source_type": key_info.source_type,
            "key_length_bytes": key_info.key_length_bytes,
            "tag_hex": tag,
            "canonical_payload_sha256": canonical_payload_sha256(payload),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "hmac_schema": "V2 canonical payload plus HMAC-SHA-256 over sorted compact JSON bytes",
        },
    }


def verify_v2_authentication_record(record: dict[str, Any], key_info: HMACKeyInfo) -> dict[str, Any]:
    """Verify a V2 authentication record and return JSON-friendly flags."""

    checked_at = datetime.now(timezone.utc).isoformat()
    payload = record.get("payload")
    authentication = record.get("authentication")
    if not isinstance(payload, dict) or not isinstance(authentication, dict):
        return {
            "record_valid": False,
            "hmac_valid": False,
            "payload_checksum_valid": False,
            "key_fingerprint_match": False,
            "schema_valid": False,
            "algorithm_supported": False,
            "video_id": payload.get("video_id") if isinstance(payload, dict) else None,
            "key_id": authentication.get("key_id") if isinstance(authentication, dict) else None,
            "failure_reason": "missing payload or authentication object",
            "verification_timestamp": checked_at,
        }
    try:
        schema_valid = (
            int(payload.get("schema_version", -1)) == V2_SCHEMA_VERSION
            and int(authentication.get("record_schema_version", -1)) == V2_SCHEMA_VERSION
        )
        algorithm_supported = authentication.get("algorithm") == DEFAULT_ALGORITHM
        payload_bytes = canonical_json_bytes(payload)
        payload_checksum_valid = (
            canonical_payload_sha256(payload) == authentication.get("canonical_payload_sha256")
        )
        key_fingerprint_match = authentication.get("key_fingerprint") == key_info.key_fingerprint
        hmac_valid = algorithm_supported and verify_hmac_sha256_hex(
            key_info.key,
            payload_bytes,
            str(authentication.get("tag_hex", "")),
        )
    except (TypeError, ValueError, OverflowError) as exc:
        return {
            "record_valid": False,
            "hmac_valid": False,
            "payload_checksum_valid": False,
            "key_fingerprint_match": False,
            "schema_valid": False,
            "algorithm_supported": False,
            "video_id": payload.get("video_id"),
            "key_id": authentication.get("key_id"),
            "failure_reason": f"malformed canonical payload: {exc}",
            "verification_timestamp": checked_at,
        }
    failures = []
    if not schema_valid:
        failures.append("schema mismatch")
    if not algorithm_supported:
        failures.append("unsupported algorithm")
    if not payload_checksum_valid:
        failures.append("payload checksum mismatch")
    if not key_fingerprint_match:
        failures.append("key fingerprint mismatch")
    if not hmac_valid:
        failures.append("HMAC tag mismatch")
    return {
        "record_valid": not failures,
        "hmac_valid": hmac_valid,
        "payload_checksum_valid": payload_checksum_valid,
        "key_fingerprint_match": key_fingerprint_match,
        "schema_valid": schema_valid,
        "algorithm_supported": algorithm_supported,
        "video_id": payload.get("video_id"),
        "key_id": authentication.get("key_id"),
        "failure_reason": "; ".join(failures) if failures else None,
        "verification_timestamp": checked_at,
    }


def save_v2_digest_npz(path: str | Path, bundle: V2DigestBundle) -> Path:
    """Save a V2 digest bundle to NPZ."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        segment_ids=bundle.segment_ids.astype(np.int64),
        segment_start_times=bundle.segment_start_times.astype(np.float64),
        segment_end_times=bundle.segment_end_times.astype(np.float64),
        resnet_binary_digests=bundle.resnet_binary_digests.astype(np.uint8),
        temporal_bin_indices=bundle.temporal_bin_indices.astype(np.uint8),
        temporal_binary_digests=bundle.temporal_binary_digests.astype(np.uint8),
        spatial_bin_indices=bundle.spatial_bin_indices.astype(np.uint8),
        spatial_binary_digests=bundle.spatial_binary_digests.astype(np.uint8),
        hybrid_binary_digests=bundle.hybrid_binary_digests.astype(np.uint8),
        resnet_packed_digests=bundle.resnet_packed_digests.astype(np.uint8),
        temporal_packed_digests=bundle.temporal_packed_digests.astype(np.uint8),
        spatial_packed_digests=bundle.spatial_packed_digests.astype(np.uint8),
        hybrid_packed_digests=bundle.hybrid_packed_digests.astype(np.uint8),
        resnet_bit_length=np.asarray(bundle.resnet_bit_length, dtype=np.int64),
        temporal_bit_length=np.asarray(bundle.temporal_bit_length, dtype=np.int64),
        spatial_bit_length=np.asarray(bundle.spatial_bit_length, dtype=np.int64),
        hybrid_bit_length=np.asarray(bundle.hybrid_bit_length, dtype=np.int64),
    )
    return output


def save_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Save JSON with deterministic formatting."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return output
