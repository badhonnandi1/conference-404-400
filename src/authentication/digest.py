"""Digest construction, bit packing, and diagnostic statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.authentication.quantization import (
    DIGEST_LENGTHS,
    HYBRID_DIGEST_LENGTH,
    RESNET_DIGEST_LENGTH,
    TEMPORAL_DIGEST_LENGTH,
    QuantizationError,
    QuantizationParameters,
    assign_temporal_bins,
    build_hybrid_digest,
    gray_encode_temporal_bins,
    quantize_resnet_binary,
)


class DigestError(RuntimeError):
    """Raised when binary digest construction or packing fails."""


@dataclass(frozen=True)
class DigestBundle:
    """Unpacked and packed binary digests for one video's segments."""

    video_id: str
    segment_ids: np.ndarray
    segment_start_times: np.ndarray
    segment_end_times: np.ndarray
    resnet_binary_digests: np.ndarray
    temporal_bin_indices: np.ndarray
    temporal_binary_digests: np.ndarray
    hybrid_binary_digests: np.ndarray
    resnet_packed_digests: np.ndarray
    temporal_packed_digests: np.ndarray
    hybrid_packed_digests: np.ndarray
    resnet_bit_length: int
    temporal_bit_length: int
    hybrid_bit_length: int
    resnet_padding_bits: int
    temporal_padding_bits: int
    hybrid_padding_bits: int
    bit_order: str
    clipping_statistics: dict[str, Any]

    def validate_round_trips(self) -> bool:
        """Return whether all packed digests unpack exactly to original bits."""

        return (
            np.array_equal(
                unpack_packed_bits(self.resnet_packed_digests, self.resnet_bit_length, self.bit_order),
                self.resnet_binary_digests,
            )
            and np.array_equal(
                unpack_packed_bits(self.temporal_packed_digests, self.temporal_bit_length, self.bit_order),
                self.temporal_binary_digests,
            )
            and np.array_equal(
                unpack_packed_bits(self.hybrid_packed_digests, self.hybrid_bit_length, self.bit_order),
                self.hybrid_binary_digests,
            )
        )


def _validate_bit_matrix(bits: np.ndarray, expected_length: int, name: str) -> np.ndarray:
    matrix = np.asarray(bits, dtype=np.uint8)
    if matrix.ndim != 2 or matrix.shape[1] != expected_length:
        raise DigestError(f"{name} bits must have shape (segments, {expected_length}), got {matrix.shape}.")
    if not np.all(np.isin(matrix, [0, 1])):
        raise DigestError(f"{name} bits contain values other than 0 or 1.")
    return matrix.copy()


def pack_bit_matrix(bits: np.ndarray, bit_order: str = "big") -> tuple[np.ndarray, int, int]:
    """Pack one bit matrix row-wise and return packed bytes, bit length, and padding bits."""

    if bit_order not in {"big", "little"}:
        raise DigestError("bit_order must be 'big' or 'little'.")
    matrix = np.asarray(bits, dtype=np.uint8)
    if matrix.ndim != 2:
        raise DigestError("Bit packing expects a 2D bit matrix.")
    if not np.all(np.isin(matrix, [0, 1])):
        raise DigestError("Cannot pack non-binary values.")
    bit_length = int(matrix.shape[1])
    packed = np.packbits(matrix, axis=1, bitorder=bit_order).astype(np.uint8)
    padding_bits = int(packed.shape[1] * 8 - bit_length)
    return packed, bit_length, padding_bits


def unpack_packed_bits(packed: np.ndarray, bit_length: int, bit_order: str = "big") -> np.ndarray:

    if bit_order not in {"big", "little"}:
        raise DigestError("bit_order must be 'big' or 'little'.")
    packed_matrix = np.asarray(packed, dtype=np.uint8)
    if packed_matrix.ndim != 2:
        raise DigestError("Packed digest array must be 2D.")
    if bit_length < 0:
        raise DigestError("bit_length must be non-negative.")
    unpacked = np.unpackbits(packed_matrix, axis=1, bitorder=bit_order).astype(np.uint8)
    if bit_length > unpacked.shape[1]:
        raise DigestError("bit_length is larger than the unpacked byte capacity.")
    return unpacked[:, :bit_length]


def bit_statistics(bits: np.ndarray) -> dict[str, Any]:
    """Return bit counts and one-bit ratios for a bit matrix."""

    matrix = np.asarray(bits, dtype=np.uint8)
    total = int(matrix.size)
    ones = int(np.count_nonzero(matrix == 1))
    zeros = total - ones
    per_segment_ones = np.sum(matrix, axis=1).astype(int).tolist() if matrix.ndim == 2 else []
    return {
        "total_bits": total,
        "zero_count": zeros,
        "one_count": ones,
        "one_ratio": float(ones / total) if total else 0.0,
        "per_segment_one_counts": per_segment_ones,
        "per_segment_one_ratios": [
            float(count / matrix.shape[1]) for count in per_segment_ones
        ]
        if matrix.ndim == 2 and matrix.shape[1]
        else [],
    }


def temporal_bin_distribution(bin_indices: np.ndarray) -> dict[str, int]:
    """Return counts for temporal bin indices 0 through 3."""

    bins = np.asarray(bin_indices, dtype=np.uint8)
    if not np.all(np.isin(bins, [0, 1, 2, 3])):
        raise DigestError("Temporal bin indices must be 0, 1, 2, or 3.")
    return {str(index): int(np.count_nonzero(bins == index)) for index in range(4)}


def clipping_statistics_for_stream(
    values: np.ndarray,
    clip_min: float = -5.0,
    clip_max: float = 5.0,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Calculate lower/upper clipping diagnostics for one normalized stream."""

    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise DigestError("Clipping statistics expect a 2D normalized feature matrix.")
    if not np.all(np.isfinite(matrix)):
        raise DigestError("Cannot calculate clipping statistics for non-finite values.")
    lower = matrix <= (clip_min + tolerance)
    upper = matrix >= (clip_max - tolerance)
    clipped = lower | upper
    total = int(matrix.size)
    lower_count = int(np.count_nonzero(lower))
    upper_count = int(np.count_nonzero(upper))
    clipped_count = int(np.count_nonzero(clipped))
    per_segment_total = matrix.shape[1]
    return {
        "total_values": total,
        "lower_clip_count": lower_count,
        "upper_clip_count": upper_count,
        "clipped_value_count": clipped_count,
        "clipping_percentage": float(100.0 * clipped_count / total) if total else 0.0,
        "per_segment_clipping_percentage": [
            float(100.0 * np.count_nonzero(clipped[index]) / per_segment_total)
            for index in range(matrix.shape[0])
        ],
        "clip_min": float(clip_min),
        "clip_max": float(clip_max),
        "tolerance": float(tolerance),
    }


def calculate_clipping_statistics(
    resnet_values: np.ndarray,
    temporal_values: np.ndarray,
    clip_min: float = -5.0,
    clip_max: float = 5.0,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Calculate clipping diagnostics for ResNet, temporal, and combined normalized values."""

    resnet = clipping_statistics_for_stream(resnet_values, clip_min, clip_max, tolerance)
    temporal = clipping_statistics_for_stream(temporal_values, clip_min, clip_max, tolerance)
    total = resnet["total_values"] + temporal["total_values"]
    clipped = resnet["clipped_value_count"] + temporal["clipped_value_count"]
    return {
        "resnet": resnet,
        "temporal": temporal,
        "combined": {
            "total_values": total,
            "clipped_value_count": clipped,
            "clipping_percentage": float(100.0 * clipped / total) if total else 0.0,
        },
    }


def build_digest_bundle(
    video_id: str,
    segment_ids: np.ndarray,
    segment_start_times: np.ndarray,
    segment_end_times: np.ndarray,
    resnet_normalized_features: np.ndarray,
    temporal_normalized_features: np.ndarray,
    parameters: QuantizationParameters,
    clip_min: float = -5.0,
    clip_max: float = 5.0,
) -> DigestBundle:
    """Build unpacked and packed digests for normalized feature streams."""

    try:
        resnet_bits = quantize_resnet_binary(resnet_normalized_features, parameters.resnet_thresholds)
        temporal_bins = assign_temporal_bins(
            temporal_normalized_features,
            parameters.temporal_q1_thresholds,
            parameters.temporal_median_thresholds,
            parameters.temporal_q3_thresholds,
        )
        temporal_bits = gray_encode_temporal_bins(temporal_bins, parameters.temporal_gray_code_table)
        hybrid_bits = build_hybrid_digest(resnet_bits, temporal_bits)
    except QuantizationError as exc:
        raise DigestError(str(exc)) from exc

    resnet_bits = _validate_bit_matrix(resnet_bits, RESNET_DIGEST_LENGTH, "ResNet")
    temporal_bits = _validate_bit_matrix(temporal_bits, TEMPORAL_DIGEST_LENGTH, "Temporal")
    hybrid_bits = _validate_bit_matrix(hybrid_bits, HYBRID_DIGEST_LENGTH, "Hybrid")
    resnet_packed, resnet_length, resnet_padding = pack_bit_matrix(resnet_bits, parameters.bit_order)
    temporal_packed, temporal_length, temporal_padding = pack_bit_matrix(temporal_bits, parameters.bit_order)
    hybrid_packed, hybrid_length, hybrid_padding = pack_bit_matrix(hybrid_bits, parameters.bit_order)
    bundle = DigestBundle(
        video_id=video_id,
        segment_ids=np.asarray(segment_ids, dtype=np.int64).copy(),
        segment_start_times=np.asarray(segment_start_times, dtype=np.float64).copy(),
        segment_end_times=np.asarray(segment_end_times, dtype=np.float64).copy(),
        resnet_binary_digests=resnet_bits,
        temporal_bin_indices=np.asarray(temporal_bins, dtype=np.uint8),
        temporal_binary_digests=temporal_bits,
        hybrid_binary_digests=hybrid_bits,
        resnet_packed_digests=resnet_packed,
        temporal_packed_digests=temporal_packed,
        hybrid_packed_digests=hybrid_packed,
        resnet_bit_length=resnet_length,
        temporal_bit_length=temporal_length,
        hybrid_bit_length=hybrid_length,
        resnet_padding_bits=resnet_padding,
        temporal_padding_bits=temporal_padding,
        hybrid_padding_bits=hybrid_padding,
        bit_order=parameters.bit_order,
        clipping_statistics=calculate_clipping_statistics(
            resnet_normalized_features,
            temporal_normalized_features,
            clip_min=clip_min,
            clip_max=clip_max,
        ),
    )
    if not bundle.validate_round_trips():
        raise DigestError("Packed/unpacked digest round-trip validation failed.")
    if bundle.resnet_bit_length != DIGEST_LENGTHS["resnet"]:
        raise DigestError("Unexpected ResNet digest length.")
    if bundle.temporal_bit_length != DIGEST_LENGTHS["temporal"]:
        raise DigestError("Unexpected temporal digest length.")
    if bundle.hybrid_bit_length != DIGEST_LENGTHS["hybrid"]:
        raise DigestError("Unexpected hybrid digest length.")
    return bundle


def digest_summary(bundle: DigestBundle) -> dict[str, Any]:
    """Return JSON-friendly digest counts and distributions."""

    return {
        "resnet": bit_statistics(bundle.resnet_binary_digests),
        "temporal": bit_statistics(bundle.temporal_binary_digests),
        "hybrid": bit_statistics(bundle.hybrid_binary_digests),
        "temporal_bin_distribution": temporal_bin_distribution(bundle.temporal_bin_indices),
    }
